"""Testing for the ``defect_detection Trainer``."""
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
test_folder_irradiated = os.path.join(test_folder1, "test_data/unirradiated/")

test_folder = os.path.join(test_folder1, "test_data/")
#weight_file_pm = os.path.join(test_folder1, "model_weights", "smallbayessegnet_lr1e-04_unirradiated_Adam_WCE_5class_model_weights.pt")

    

def test_performance_metrics():

    image_folders = ['unirradiated']
    nclasses = [5]
    # ['segnet', 'bayessegnet', 'smallbayessegnet', 'bayessegnest']
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam']
    losses = ['EWCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    image_type, nclass, model, lr, opt, loss = combos[int(0)]

    # set up the trainer
    trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                             filename='TTP*.tif',
                                             learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                             optimize_option=opt,
                                             model_name=model, N_epochs=1, test_version=True)#load_pt_files=True)
    trainer.test_predictions()
    return_values = trainer.performance_metrics(performance_calibrate=False, saveimage = False)
    assert round(0.354839, 2) == round(return_values["iou_3"], 2)
    true_positive = return_values["iou_df"]['pcm']["big_image"].loc[0, 0]
    all_positives = return_values["iou_df"]['pcm']["big_image"].loc[0, :].sum()
    recall_grain = return_values["iou_df"]['pout']["big_image"].loc['Grain', 'Recall']
    assert true_positive/all_positives == recall_grain

def test_performance_metrics_metadata():

    image_folders = ['unirradiated']
    nclasses = [5]
    # ['segnet', 'bayessegnet', 'smallbayessegnet', 'bayessegnest']
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam']
    losses = ['EWCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    image_type, nclass, model, lr, opt, loss = combos[int(0)]

    # set up the trainer
    trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                             filename='TTP*.tif',
                                             learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                             optimize_option=opt,
                                             model_name=model, N_epochs=1, test_version=True, dummy_ycat = True, include_meta= True)#load_pt_files=True)
    return_values = trainer.performance_metrics(performance_calibrate=False, saveimage = False)
    assert round(0.354839, 2) == round(return_values["iou_3"], 2)
    true_positive = return_values["iou_df"]['pcm']["big_image"].loc[0, 0]
    all_positives = return_values["iou_df"]['pcm']["big_image"].loc[0, :].sum()
    recall_grain = return_values["iou_df"]['pout']["big_image"].loc['Grain', 'Recall']
    assert true_positive/all_positives == recall_grain

def test_performance_metrics():

    image_folders = ['unirradiated']
    nclasses = [5]
    # ['segnet', 'bayessegnet', 'smallbayessegnet', 'bayessegnest']
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam']
    losses = ['TOPO', 'WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))

    
    image_type, nclass, model, lr, opt, loss = combos[int(0)]

    # set up the trainer
    trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                             filename='TTP*.tif',
                                             learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                             optimize_option=opt,
                                             model_name=model, N_epochs=1, test_version=True)  # load_pt_files=True)

    return_values = trainer.performance_metrics(
        performance_calibrate=False, saveimage=False)
    assert round(0.354839, 2) == round(return_values["iou_3"], 2)
    true_positive = return_values["iou_df"]['pcm']["big_image"].loc[0, 0]
    all_positives = return_values["iou_df"]['pcm']["big_image"].loc[0, :].sum()
    recall_grain = return_values["iou_df"]['pout']["big_image"].loc['Grain', 'Recall']
    assert true_positive/all_positives == recall_grain



def test_loading_ops():
    image_folders = ['unirradiated']
    nclasses = [5]
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam', 'RMSprop']
    losses = ['WCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    for i in range(len(combos)):
        image_type, nclass, model, lr, opt, loss = combos[int(i)]


        # set up the trainer
        trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                                filename='TTP*.tif',
                                                learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                                optimize_option=opt,
                                                model_name=model, N_epochs=1, test_version=True)  # load_pt_files=True)

        assert trainer.optimize_option  == opt

def test_loading_ops():
    image_folders = ['unirradiated']
    nclasses = [5]
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam', 'RMSprop']
    losses = ['WCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    for i in range(len(combos)):
        image_type, nclass, model, lr, opt, loss = combos[int(i)]


        # set up the trainer
        trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                                filename='TTP*.tif',
                                                learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                                optimize_option=opt,
                                                model_name=model, N_epochs=1, test_version=True)  # load_pt_files=True)

        assert trainer.optimize_option  == opt
        

def test_loading_models():
    image_folders = ['unirradiated']
    nclasses = [5]
    models = ['segnet', 'resnet50', 'segnetmeta',
              'bayessegnet', 'smallbayessegnet']
    lrs = [1e-4]
    opts = ['Adam']
    losses = ['WCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    for i in range(len(combos)):
        image_type, nclass, model, lr, opt, loss = combos[int(i)]

        # set up the trainer
        trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                                 filename='TTP*.tif',
                                                 learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                                 optimize_option=opt,
                                                 model_name=model, N_epochs=1, test_version=True)  # load_pt_files=True)

        assert trainer.model_name == model

def test_loading_losses():
    image_folders = ['unirradiated']
    nclasses = [5]
    models = ['segnet', 'resnet50', 'segnetmeta', 'bayessegnet', 'smallbayessegnet']
    lrs = [1e-4]
    opts = ['Adam', 'RMSprop']
    losses = ['WCE', 'EWCE', 'FL', 'TOPO']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    for i in range(len(combos)):
        image_type, nclass, model, lr, opt, loss = combos[int(i)]


        # set up the trainer
        trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                                filename='TTP*.tif',
                                                learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                                optimize_option=opt,
                                                model_name=model, N_epochs=1, test_version=True)  # load_pt_files=True)

        assert trainer.loss_type == loss

# metadata is not used but making sure it doesn't get  in the way
def test_performance_metrics_meta_data():

    image_folders = ['unirradiated']
    nclasses = [5]
    # ['segnet', 'bayessegnet', 'smallbayessegnet', 'bayessegnest']
    models = ['segnet']
    lrs = [1e-4]
    opts = ['Adam']
    losses = ['EWCE']  # ['WCE',  'EWCE']
    combos = list(itertools.product(
        image_folders, nclasses, models, lrs, opts, losses))
    image_type, nclass, model, lr, opt, loss = combos[int(0)]

    # set up the trainer
    trainer = defect_detection.train.Trainer(nclass=nclass, folder=test_folder1 + "/", image_folder=test_folder_irradiated, loss_type=loss,
                                             filename='TTP*.tif',
                                             learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}',
                                             optimize_option=opt,
                                             model_name=model, N_epochs=1, test_version=True, include_metadata=False)  # load_pt_files=True)
    return_values = trainer.performance_metrics(saveimage=False)
    assert round(0.354839, 2) == round(return_values["iou_3"], 2)
    true_positive = return_values["iou_df"]['pcm']["big_image"].loc[0, 0]
    all_positives = return_values["iou_df"]['pcm']["big_image"].loc[0, :].sum()
    recall_grain = return_values["iou_df"]['pout']["big_image"].loc['Grain', 'Recall']
    assert true_positive/all_positives == recall_grain


def test_ripley():
    r = [0.0000, 0.3125, 0.6250, 0.9375, 1.2500]

    #set max to the image width (shape[1]) and height (shape[0])
    mins = np.array([0., 0.])
    maxs = np.array([10.0, 10.0])
    X = np.array([[.6, .1], [5.4, 5.4],  [5.6,	5.6], [1, 1],
                                          [2.3, 2.3], [2.4, 2.4], [3.6, 3.6]])
    center1 = np.array([ [5.4, 5.4],  [5.6,	5.6], [1, 1],
                        [2.3, 2.3]])
    center2 = np.array([[.6, .1], [2.4, 2.4], [3.6, 3.6]])

    Kest = defect_detection.ripley_univariate.RipleysKEstimator_univariate(area=np.prod(maxs - mins), x_max=maxs[0], y_max=maxs[1], x_min=mins[0],
                                                   y_min=mins[1])
    Hp_univariate = Kest.Hfunction(data1=X, radii=r, mode='ripley')

    Kest = defect_detection.ripley.RipleysKEstimator(area=np.prod(
        maxs - mins), x_max=maxs[0], y_max=maxs[1], x_min=mins[0], y_min=mins[1])
    Hp = Kest.Hfunction(data1=center1, data2=center2, radii=r, mode='ripley')
    #check if 
    assert round(Hp[4], 2) == round(1.053294, 2)
    assert round(Hp_univariate[4], 2) == round(1.1656315, 2)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def test_image_in_folder():
    fname = 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif'
    Xold = cv2.imread(os.path.join(test_folder, fname))
    grey = rgb2gray(Xold)
    assert type(Xold) != None

def test_trainer_init():
    fname = 'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif'
    trainer = defect_detection.train.Trainer(folder=test_folder,
                                             filename=fname)
    assert isinstance(trainer, defect_detection.train.Trainer)

def test_trainer_image_folder():
    trainer = defect_detection.train.Trainer(folder=test_folder, N_epochs = 1, test_version = True, loss_type = "CE")
    assert trainer.folder == trainer.image_folder
    
    tmp_folder = os.path.join(test_folder1, "tmp/")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    trainer2 = defect_detection.train.Trainer(folder=tmp_folder, image_folder=test_folder, test_version = True)
    assert trainer2.folder != trainer2.image_folder


def test_train_load():
    trainer = defect_detection.train.Trainer(folder=test_folder, filename='T*.tif', N_epochs = 1, model_name = "segnetmeta",  nclass=5, test_version = True, include_metadata=True)
    # trainer.train()
    trainer.test_predictions()
    trainer.performance_metrics()
    return_values = trainer.performance_metrics(saveimage=False)
    assert round(0.354839, 2) == round(return_values["iou_3"], 2)
    true_positive = return_values["iou_df"]['pcm']["big_image"].loc[0, 0]
    all_positives = return_values["iou_df"]['pcm']["big_image"].loc[0, :].sum()
    recall_grain = return_values["iou_df"]['pout']["big_image"].loc['Grain', 'Recall']
    assert true_positive/all_positives == recall_grain


def test_save_results_we():
    num_epoch = 1
    trainer = defect_detection.train.Trainer(folder=test_folder, N_epochs = num_epoch, test_version = True, loss_type = "CE", n_save =1)
    trainer.train()
    filename = f"{trainer.train_description}class_train_loss.csv"
    with open(filename,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    assert row_count == num_epoch



def test_save_results_wce():
    num_epoch = 1
    trainer = defect_detection.train.Trainer(folder=test_folder, N_epochs = num_epoch, test_version = True, loss_type = "WCE", n_save =1, dummy_ycat = False)
    trainer.train()
    filename = f"{trainer.train_description}class_train_loss.csv"
    with open(filename,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    assert row_count == num_epoch

def test_save_results_ewce():
    num_epoch = 1
    trainer = defect_detection.train.Trainer(folder=test_folder, N_epochs = num_epoch, test_version = True, loss_type = "EWCE", n_save =1)
    trainer.train()
    filename = f"{trainer.train_description}class_train_loss.csv"
    with open(filename,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
    assert row_count == num_epoch

def check_two_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def test_trainer_load():
    trainer = defect_detection.train.Trainer(folder=test_folder, N_epochs = 1, test_version = True, loss_type = "FL")
    trainer.save_model()
    filename = f"{trainer.train_description}class_model_weights.pt"
    trainer2 = defect_detection.train.Trainer(folder=test_folder)
    trainer2.load(filename)
    assert check_two_models(trainer.model, trainer2.model)
    trainer3 = defect_detection.train.Trainer(folder=test_folder,
                                             weights_file=filename)
    assert check_two_models(trainer.model, trainer3.model)

def test_losses():
    """Test the correct behavior for different losses."""
    with pytest.raises(Exception):
        _ = defect_detection.train.Trainer(folder=test_folder, loss_type='XYZ')
    trainer = defect_detection.train.Trainer(folder=test_folder, loss_type='CE')
    assert isinstance(trainer.loss_fn, torch.nn.CrossEntropyLoss)
    #trainer = defect_detection.train.Trainer(folder=test_folder,
    #                                         loss_type='MSE')
    #assert isinstance(trainer.loss_fn, torch.nn.MSELoss)
    #trainer = defect_detection.train.Trainer(folder=test_folder,
    #                                         loss_type='MAE')
    #assert isinstance(trainer.loss_fn, torch.nn.L1Loss)

def test_opt():
    """Test that the optimizers are initialized properly"""
    trainer = defect_detection.train.Trainer(folder=test_folder,
                                             optimize_option='SGD')
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    trainer = defect_detection.train.Trainer(folder=test_folder,
                                             optimize_option='RMSprop')
    assert isinstance(trainer.optimizer, torch.optim.RMSprop)
    trainer = defect_detection.train.Trainer(folder=test_folder,
                                             optimize_option='Adam')
    assert isinstance(trainer.optimizer, torch.optim.Adam)

def test_device():
    """Test device setup - THIS DOES NOT TEST GPUS, just loading of devices."""
    trainer = defect_detection.train.Trainer(folder=test_folder, device=None)
    assert trainer.device == torch.device('cpu')
    trainer = defect_detection.train.Trainer(folder=test_folder, device='cpu')
    assert trainer.device == torch.device('cpu')
    cpu = torch.device('cpu')
    trainer = defect_detection.train.Trainer(folder=test_folder, device=cpu)
    assert trainer.device == torch.device('cpu')

def test_weights():
    """Test that we can specify our own weights."""
    trainer = defect_detection.train.Trainer(folder=test_folder,
                                             weights=[1.0, 1.0, 1.0, 1.0, 0.1])
    assert isinstance(trainer.loss_fn, torch.nn.CrossEntropyLoss)


def test_one_epoch():
    """Test that one epoch is trained."""
    pass

def test_validate():
    """Test that the validation step operates properly."""
    pass

if __name__ == "__main__":
    test_trainer_load()
