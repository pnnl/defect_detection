import os
import sys
import logging
import itertools
import glob
import defect_detection

import torch
import numpy as np
import pickle
from torch import nn, optim

combo_num = int(sys.argv[1])
save_folder = sys.argv[2] 
model_path = sys.argv[3] 

no_running_mean = False
# read in jobarray number and set options
image_folders = ['irradiated'] # ['irradiated', 'unirradiated']
nclasses = [5]
models = ['smallbayessegnet']#['smallbayessegnet']#['smallbayessegnet', 'segnet', 'bayessegnet',  'bayessegnest']
lrs = [1e-4]
opts = ['Adam']#['Adam','RMSprop']
losses =['EWCE'] #['EWCE', 'WCE']

combos = list(itertools.product(image_folders, nclasses, models, lrs, opts, losses))
image_type, nclass, model, lr, opt, loss = combos[combo_num]
folder = os.path.join('/qfs/projects/tritium', image_type)

# set the logging level so we can see training updates
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logging.info(f'Setting image folder to {folder}')
logging.info(f'Using model {model}')
logging.info(f'Setting learning rate to {lr:.2e}')
logging.info(f'Using optimizer {opt}')
logging.info(f'Using loss {loss}')

# set up the trainer
trainer = defect_detection.train.Trainer(nclass=nclass, image_folder=folder, loss_type=loss,
                                   filename='TTP*.tif', folder=save_folder,
                                   learn_rate=lr, model_name_suffix=f'_lr{lr:.0e}_{image_type}_{opt}_new_augmentation',
                                   model_path=model_path,
                                   optimize_option=opt,
                                   model_name=model, gpu=True, device='cuda:0', N_epochs=500, include_metadata = False, load_pt_files = True)

if not os.path.exists(f"{trainer.train_description}"):
    os.makedirs(f"{trainer.train_description}")
if not os.path.exists(os.path.join(f"{trainer.train_description}", "image")):
    os.makedirs(os.path.join(f"{trainer.train_description}", "image"))

if no_running_mean:
    for m in trainer.model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None

calibrator = defect_detection.calibrator.Calibrator(trainer.model, trainer.val_dataset, device='cuda:0', 
                                                N_trials=10, max_chips=20, image_name = f'{model}_lr{lr:.0e}_{image_type}_{opt}_{loss}_{nclass}_') # 10, 20

calibrator.calibrate()

for i in range(len(trainer.X_test)):
    X_test = torch.from_numpy(trainer.X_test[i]).float().permute((2, 0, 1)).unsqueeze(0).to(trainer.device)

    # # use the images already calibrated
    try:
        p_correct_per_pixel_file = f"{trainer.train_description}_cv{trainer.cv_split}_image{trainer.files[i]}_p_correct_per_pixel.pkl"
        with open(p_correct_per_pixel_file, 'rb') as f:
            p_correct_per_pixel = pickle.load(f).squeeze()
        hypothesis_class_per_pixel_file = f"{trainer.train_description}_cv{trainer.cv_split}_image{trainer.files[i]}_hypothesis_class_per_pixel.pkl"
        with open(hypothesis_class_per_pixel_file, 'rb') as f:
            hypothesis_class_per_pixel = pickle.load(f).squeeze()
    except:
        hypothesis_class_per_pixel, p_correct_per_pixel  = calibrator(X_test)
        try:
            pickle.dump(hypothesis_class_per_pixel, open(f"{trainer.train_description}_cv{trainer.cv_split}_image{trainer.files[i]}_hypothesis_class_per_pixel.pkl", 'wb'))
            pickle.dump(p_correct_per_pixel, open(f"{trainer.train_description}_cv{trainer.cv_split}_image{trainer.files[i]}_p_correct_per_pixel.pkl", 'wb'))
        except:
            print("Pickle dump failed.")
    scale = trainer.dict_scaling[trainer.files[i]]['scale']
    defect_detection.vis.vis_with_batch_true_and_hypothesis(X_test.squeeze().permute((1, 2, 0))/255.0, trainer.Y_test[i],
                                    hypothesis_class_per_pixel, alpha=p_correct_per_pixel, nclass=nclass,
                                   filename=os.path.join(f"{trainer.train_description}",  "image", f'{model}_lr{lr:.0e}_{image_type}_{opt}_{loss}_{nclass}_{trainer.files[i]}_batch_image.png'), scale = scale, figsize=(18, 10))
    defect_detection.vis.vis_chip(hypothesis_class_per_pixel,  nclass=nclass,  alpha=p_correct_per_pixel,
                                   filename=os.path.join(f"{trainer.train_description}",  "image", f'{model}_lr{lr:.0e}_{image_type}_{opt}_{loss}_{nclass}_{trainer.files[i]}_calibrate_single_image.png'), figsize=(18, 10))
    defect_detection.vis.vis_chip(hypothesis_class_per_pixel,  nclass=nclass,
                                   filename=os.path.join(f"{trainer.train_description}",  "image", f'{model}_lr{lr:.0e}_{image_type}_{opt}_{loss}_{nclass}_{trainer.files[i]}_single_image.png'), figsize=(18, 10))

    print(os.path.join(f"{trainer.train_description}",  "image", f'{model}_lr{lr:.0e}_{image_type}_{opt}_{loss}_{nclass}_{trainer.files[i]}_batch_image.png'), ' complete!')

trainer.performance_metrics(performance_calibrate = True)

