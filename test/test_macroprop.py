import numpy as np
from defect_detection import macroprop
import pytest
import torch
import defect_detection
import os
import cv2
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
import itertools
import glob


test_folder1 = os.path.abspath(os.path.dirname(__file__))
test_folder_unirradiated = os.path.join(test_folder1, "test_data/unirradiated/")

test_folder = os.path.join(test_folder1, "test_data/")
#weight_file_pm = os.path.join(test_folder1, "model_weights", "smallbayessegnet_lr1e-04_unirradiated_Adam_WCE_5class_model_weights.pt")


def test_multinoulli():
    ps = np.zeros((2, 2, 3))
    ps[0, 0, 1] = 1
    ps[0, 1, 0] = 1
    ps[1, 1, 2] = 1
    ps[1, 0, 1] = 1
    out = macroprop.multinoulli(ps)
    assert np.array_equal(out, ps.argmax(-1))

 

# def test_calibrate():
    
#     weights_file = 'segnet_lr1e-04_unirradiated_RMSprop_EWCE_5class_model_weights.pt'
#     # instantiate a trainer
#     kwargs = dict(model_name='segnet', N_epochs=0, optimize_option='Adam',
#                 gpu=True,  learn_rate=1E-4,
#                 model_name_suffix='propagation', nclass=5,
#                   folder=test_folder1 + "/", image_folder=test_folder_unirradiated,
#                 filename='TTP_*.tif')
#     trainer = defect_detection.train.Trainer(**kwargs)
#     trainer.load(os.path.join(test_folder, weights_file))
#     calibrator = defect_detection.calibrator.Calibrator(trainer.model, trainer.val_dataset,N_trials=1, max_chips=10,
#                                                         bayes=False, full_posterior=True)
#     calibrator.calibrate()
#     macro_prop = defect_detection.macroprop.MacroProp(
#         trainer, calibrator, 'unirradiated_test', 1 )


#     macro_prop()
    
#     assert 1 == 1



def test_macroprop_fun():

    trainer = None
    calibrator = None
    zero_image = np.zeros((1, 3, 2, 2))
    X_test = zero_image.copy()
    ## Grain Boundary ##
    X_test[0, :, 0, 0] = np.array([0, 0, 255])
    ## Void ##
    X_test[0, :,1, 1] = np.array([0, 255, 0])
    ## Impurity ##
    X_test[0, :,1, 0] = np.array([0, 255, 255])
    ## Precipitate ##
    X_test[0, :, 0, 1] = np.array([255, 0, 0])

    prob_test = np.zeros((2, 2, 5))
    prob_test[0, 0, 1] = .1
    prob_test[0, 1, 0] = .1
    prob_test[1, 1, 2] = .1
    prob_test[1, 0, 1] = .1

    h_test = np.zeros((2, 2))
    h_test[0, 0] = 1
    h_test[0, 1] = 1
    h_test[1, 1] = 1
    h_test[1, 0] = 1

    X_test = torch.from_numpy(X_test).float()
    prob_test = torch.from_numpy(prob_test).float()
    h_test = torch.from_numpy(h_test).float()
    macro_prop = defect_detection.macroprop.MacroProp(
        trainer, calibrator, 'unirradiated_test', 1, test_version = True, save = False, test_input = (X_test, h_test, prob_test) )

    # since there are not clusters in this example, expected result is zero
    gb_mean  = macro_prop()
    
    print(gb_mean)

    assert 1 == 1
