"""Module for the BayesSegNet Model and Advanced Dropout Layers."""
import itertools
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class IndexedDropoutModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_layers = {}
    
    def register_dropout(self, name, layer):
        if name[-1] == '.':
            name = name[:-1]
        root = self
        while root.__parent__ is not None:
            root = root.__parent__[0]
        root.dropout_layers[name] = layer
## Something
class AlwaysDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = F.dropout(x, self.p, training=True)
        return x

class SingleNeuronDropout(nn.Module):
    __count__ = itertools.count()
    def __init__(self, neuron=None):
        super().__init__()
        self.__name__ = f'singleneurondropout.{next(self.__count__)}.'
        #self.neuron = neuron
        self.neurons = []

    def set_neuron(self, neuron):
        #self.neuron = neuron
        self.neurons.append(neuron)
        
    def reset(self):
        self.neurons = []
        

    def forward(self, x):
        shape = x.shape
        batch_size = shape[0]
        self.n_neurons = torch.numel(x) // batch_size
        n_dropped = float(len(self.neurons))
        # flatten this to two dimension, (n_batch, n_neurons)
        x = x.reshape(shape[0], -1)
        # make a copy
        coeff = torch.ones_like(x)
        for neuron in self.neurons:
            coeff[:, neuron] = 0.0
        # calulate the percent of neurons we've actually dropped out
        p = n_dropped/self.n_neurons
        factor = 1.0 / (1.0 - p)
        # actually drop out
        x = factor * coeff * x
        # reshape it to the original shape
        x = x.reshape(*shape)
        return  x

#https://github.com/trypag/pytorch-unet-segnet/blob/master/segnet.py

class BayesSegNet(IndexedDropoutModel):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        nclass (int): number of classes to segment
        channels (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    __count__ = itertools.count()
    def __init__(self, nclass, channels=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512),
                 encoder_n_layers=(2, 2, 3, 3, 3),
                 decoder_n_layers=(3, 3, 3, 2, 1),
                 encoder_do_layer=[None, None, 3, 3, 3],
                 decoder_do_layer=[3, 3, 3, None, None]):
        super(BayesSegNet, self).__init__()
        self.__parent__ = None
        self.__name__ = f'bsn.{next(self.__count__)}.'
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_filter_config = (channels,) + filter_config
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate,
                                          do_layer=encoder_do_layer[i],
                                          parent=self))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate,
                                          do_layer=decoder_do_layer[i],
                                          parent=self))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], nclass, 3, 1, 1)

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

        return self.classifier(feat)


class _Encoder(IndexedDropoutModel):
    __count__ = itertools.count()
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5,
                 do_layer=3, parent=None):
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
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if do_layer is not None and i_block == do_layer - 2:
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
                 do_layer=3, parent=None):
        super(_Decoder, self).__init__()
        self.__parent__ = [parent]
        self.__name__ = f'decoder.{next(self.__count__)}.'
        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        for i_block in range(n_blocks - 1):
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if do_layer is not None and i_block == do_layer - 2:
                snd = SingleNeuronDropout()
                layers += [AlwaysDropout(drop_rate),
                           snd]
                self.register_dropout(self.__name__ + snd.__name__, snd)

        self.features = nn.Sequential(*layers)
        self.softmax = nn.Softmax2d()

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        x = self.features(unpooled)
        x = self.softmax(x)
        return x

