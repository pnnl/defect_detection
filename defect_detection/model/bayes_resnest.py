"""Bayesian SegNeSt Model Definitions."""
# internals
import itertools
import collections.abc
from itertools import repeat
# externals
import torch
from torch import nn
import torch.nn.functional as F
# this repo
from .bayes_segnet import IndexedDropoutModel, AlwaysDropout, SingleNeuronDropout


# From PyTorch internals
def _ntuple(n):
    """Converts several other 'tuple' types to a proper python ``tuple``."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """Make a value ``v`` divisiable by the ``divisor``.
    
    :param int v: Value which we want to round so it's divisible.
    :param int divisor: The denominator by which ``v`` must be divisible.
    :param int min_value: Minimum value the rounded ``v`` can take.
    :param float round_limit: The ratio of the original value below which we
        cannot round ``v``.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class RadixSoftmax(nn.Module):
    """RadixSoftmax layer for implementing Splat layer.
    
    :param int radix: The radius of the SoftMax operation.
    :param int cardinality: The number of groups to split over which the SoftMax
        will not be performed.
    """
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


class BayesSegNeSt(IndexedDropoutModel):
    """Implements the SegNet Architecture with Bayesian Dropout and Radix
    Softmax to follow ResNeSt paradigm.

    :param int nclass: Number of segmentation classes
    :param int channels: Number of channels of input image, default ``1``
    :param float drop_rate: The rate of dropout in the ``AlwaysDropout`` layers.
        Default ``0.5``, set to ``0.0`` for non-bayesian SegNet.
    :param list filter_config: Number of filters per layer, default
        ``(64, 128, 256, 512, 512)``
    """
    __count__ = itertools.count()
    def __init__(self, nclass, channels=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(BayesSegNeSt, self).__init__()
        self.__parent__ = None
        self.__name__ = f'bsns.{next(self.__count__)}.'
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (channels,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate,
                                          parent=self))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate,
                                          parent=self))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], nclass, 3, 1, 1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        clf = self.classifier(feat)
        sm = self.softmax(clf)
        return sm

class _Encoder(IndexedDropoutModel):
    __count__ = itertools.count()
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5,
                 parent=None, radix=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()
        self.__name__ = f'encoder.{next(self.__count__)}.'
        self.__parent__ = [parent]
        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        for i_block in range(n_blocks - 1):
            if i_block == 0:
                layers += [SplitAttn(n_out_feat, n_out_feat, kernel_size=3,
                                     stride=1, radix=radix),
                           nn.Identity(),
                           nn.Identity()]
            else:
                layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]
            if i_block == 1:
                snd = SingleNeuronDropout()
                layers += [AlwaysDropout(drop_rate),
                            snd]
                self.register_dropout(self.__name__ + snd.__name__, snd)


        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()

class _Decoder(IndexedDropoutModel):
    __count__ = itertools.count()
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5,
                 parent=None, radix=2):
        super(_Decoder, self).__init__()
        self.__parent__ = [parent]
        self.__name__ = f'decoder.{next(self.__count__)}.'
        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        for i_block in range(n_blocks - 1):
            if i_block == 0:
                layers += [SplitAttn(n_out_feat, n_out_feat, kernel_size=3,
                                     stride=1, radix=radix),
                           nn.Identity(),
                           nn.Identity()]
            else:
                layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                           nn.BatchNorm2d(n_out_feat),
                           nn.ReLU(inplace=True)]
            if i_block == 1:
                snd = SingleNeuronDropout()
                layers += [AlwaysDropout(drop_rate),
                            snd]
                self.register_dropout(self.__name__ + snd.__name__, snd)

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        x = self.features(unpooled)
        #x = self.softmax(x)
        return x