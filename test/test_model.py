"""Testing for the ``defect_detection models``."""
import torch
import numpy as np
import defect_detection

def test_unet_baseline_dims():
    """Make sure UnetBaseline input and output dimensions match."""
    shape = (1, 3, 96, 96)
    input_data = torch.rand(shape)
    model = defect_detection.model.unet_baseline.UnetBaseline()
    output = model(input_data)
    assert output.shape == shape

def test_segnet_baseline_dims():
    """Make sure SegNet input and output dimensions match."""
    shape = (1, 3, 96, 96)
    input_data = torch.rand(shape)
    model = defect_detection.model.segnet_baseline.SegNet(3, 3,
                                                          pretrained=False)
    output = model(input_data)
    assert output.shape == shape

def test_bsn_baseline_dims():
    """Make sure SegNet input and output dimensions match."""
    shape = (1, 1, 96, 96)
    input_data = torch.rand(shape)
    model = defect_detection.model.bayes_segnet.BayesSegNet(1)
    output = model(input_data)
    assert output.shape == shape

def test_unetconv2_dims():
    """Testing that the unet convolution keeps correct dims."""
    layer = defect_detection.model.unet_utils.unetConv2(3, 3, False)
    shape = (1, 3, 96, 96)
    input_data = torch.rand(shape)
    output = layer(input_data)
    assert output.shape == shape
    layer = defect_detection.model.unet_utils.unetConv2(3, 3, True)
    output = layer(input_data)
    assert output.shape == shape

def test_unetup_dims():
    """Testing that the unet transpose convolution really upsamples."""
    layer = defect_detection.model.unet_utils.unetUp(3, 4, False)
    shape = (1, 1, 96, 96)
    up_shape = (1, 2, 96*2, 96*2)
    out_shape = (1, 4, 96*2, 96*2)
    input_data = torch.rand(shape)
    residual = torch.rand(up_shape)
    output = layer(residual, input_data)
    assert output.shape == out_shape

def test_rgb2gray():
    """Test that rgb2gray converts a 3 channel images into greyscale."""
    x = np.random.uniform(0, 1, (100, 100, 3))
    y = defect_detection.model.greyscale_baseline.rgb2gray(x)
    assert len(y.shape) == 2

def test_gscale_model():
    """Test that the greyscale model returns the right sizes."""
    X = np.random.uniform(0, 1, (100, 100, 3))
    Ycat = np.random.uniform(0, 5, (100, 100)).astype(int)
    Ypred = defect_detection.model.greyscale_baseline.greyscale_model(X, Ycat)
    assert Ycat.shape == Ypred.shape

def test_bayessegnet():
    """Test that the bayessegnest is instantiated and returns different
    results.
    """
    model = defect_detection.model.bayes_resnest.BayesSegNeSt(nclass=5,
                                                              drop_rate=0.5)
    x = torch.rand((1, 1, 128, 128))
    x2 = x.clone()
    y = model(x)
    y2 = model(x2)
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert not torch.all(y == y2)
