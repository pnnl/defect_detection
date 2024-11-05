import os
import sys
import logging
import itertools
import glob
import defect_detection

import torch
import numpy as np
import pickle

combo_num = int(sys.argv[1])
save_folder = sys.argv[2] 
model_path = sys.argv[3] 

# read in jobarray numb
# read in jobarray number and set options
image_folders = ['irradiated', 'unirradiated']
nclasses = [5]
models = ['smallbayessegnet', 'segnet', 'segnetmeta', 'segnet',   'bayessegnet',  'bayessegnest']
# models = ['segnetmeta']
lrs = [1e-4]
opts = ['Adam','RMSprop']
losses =['EWCE', 'WCE']

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
                                   model_name=model, gpu=True, device='cuda:0', N_epochs=500, include_metadata = False, load_pt_files = False)

trainer.train()
