"""TTP Segmentation Model Training Module."""
import argparse
import errno
from defect_detection.model.bayes_resnest import BayesSegNeSt
import logging
from time import time
import glob
import os
import sys
import torch
import pickle
from torch import nn, optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json

from ..image_preprocessing import image_prep, image_split, image_augment, read_Y
try:
    from IPython import get_ipython
except ModuleNotFoundError:
    # if IPython(jupyter) isn't installed, such as on headless servers,
    # we should make a dummy function which returns an object which has
    # another dummy function "run_line_magic"
    class DummyIPython:
        def run_line_magic(self, *args, **kwargs):
            pass

    def get_ipython(*args, **kwargs):
        return DummyIPython()
from ..dl_functions import torch_load_data, torch_load_data_metadata, return_meta_data, return_scale
from ..model.unet_baseline import UnetBaseline
from ..model.segnet_baseline import SegNet
from ..model.segnet_metadata import SegNetMeta
from ..model.bayes_segnet import BayesSegNet
from ..model.bayes_resnest import BayesSegNeSt
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101, fcn_resnet50, deeplabv3_resnet50
from ..defect_locate import defect_summary, box_iou
from ..performance_metrics import *
from ..vis import *
import queue
#https://github.com/AdeelH/pytorch-multi-class-focal-loss
from typing import Optional, Sequence
from itertools import permutations

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from ..model.focal_loss import FocalLoss
from ..model.topoloss import TopoLoss


from ..model.topoloss import getTopoLoss

from ..ripley import RipleysKEstimator, MCconfidence
from ..ripley_univariate import RipleysKEstimator_univariate, MCconfidence_univariate



MODELS = dict(resnet50=dict(model=fcn_resnet50, kw=dict(pretrained=False)),
              resnet101=dict(model=fcn_resnet101, kw=dict(pretrained=False)),
              segnet=dict(model=SegNet, kw=dict()),
              segnetmeta=dict(model=SegNetMeta, kw=dict()),
              unet=dict(model=UnetBaseline, kw=dict(feature_scale=1, is_deconv=True, is_batchnorm=True)),
              bayessegnet=dict(model=BayesSegNet, kw=dict()),
              smallbayessegnet=dict(model=BayesSegNet,
                                    kw=dict(encoder_n_layers=(2, 2, 2, 2, 2),
                                            decoder_n_layers=(2, 2, 2, 2, 1),
                                            encoder_do_layer=(None, None, 2, 2, 2),
                                            decoder_do_layer=(2, 2, 2, None, None)
              )),
              bayessegnest=dict(model=BayesSegNeSt, kw=dict()))

class Trainer:
    """The trainer is an object which trains TTP Segmentation models."""
    def __init__(self, seed=1337, N_epochs=500, showplot=False, saveimage = True, saveres=True,
                 model_name='segnet', label='SegNet', loss_type='WCE',
                 cv_split=4, nclass=6, channels=3,
                 folder='test/test_data/', image_folder=None,
                 filename='C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif',
                 model_path=None,
                 dict_scaling = {

                    'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif': {'scale': 42.8, 'pixel_width': 640, 'pixel_height': 1920},\
                    'TTP_C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-B.tif': {'scale': 42.8, 'pixel_width': 640,'pixel_height': 1920}
                    },
                 alim_um = 0.001888,  # 20/(2048/19.9)^2 -> px^2/(px/um)^2 (equals 20px2 for image with largest scale, 3.5px2 for image with lowest scale)
                 on_cluster=True, gpu=True, device=None, batch_size=96, learn_rate=1E-4,
                 optimize_option='SGD', weights=None,
                 factor=.00001, patience=50, model_name_suffix='',
                 weights_file=None, subpix = 128, test_version = False, load_pt_files = False, dummy_ycat = False, include_metadata = False,
                 **kwargs):
        """Initialize the Trainer object.

        :param int seed: Random seed for torch and numpy, to ensure
            reproducibility.
        :param str folder: Folder from which to load images (if ``image_folder``
            is not set) and to which to save model weights and results.
        :param str image_folder: Folder from which to load images. Defaultfb
            ``None``, if unset, defaults to that specified by ``folder``.
        :param int N_epochs: Number of epochs over which to train the model.
        :param str weights_file: Optional filename to load pretrained weights.
            Default ``None`` (no pretrained weights loading, uses default
            weight initializations)
        """
        # save all params to class vars
        self.N_epochs = N_epochs
        self.n_save = 10
        self.showplot = showplot
        self.saveres = saveres
        self.model_name = model_name
        self.optimize_option = optimize_option
        self.label = label
        self.loss_type = loss_type
        self.cv_split = cv_split
        self.nclass = nclass
        self.folder = folder
        self.filename = filename
        self.model_path = model_path
        self.on_cluster = on_cluster
        self.gpu = gpu
        self.saveimage = saveimage
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.subpix = subpix
        self.test_version = test_version
        self.load_pt_files = load_pt_files
        self.dummy_ycat = dummy_ycat
        self.model_name_suffix = model_name_suffix
        self.dict_scaling = dict_scaling
        # set the image folder if unset
        if image_folder is None:
            self.image_folder = folder
        else:
            self.image_folder = image_folder
        
        # set the model path if unset
        if model_path is None:
            self.model_path = folder
        else:
            self.model_path = model_path
        self.include_metadata = include_metadata
        # alim_um calculated with 16px**2/scaling factor(px/um)**2 with scaling the width of image in pixels/HFW
        self.alim_um =  alim_um
        
               # set the random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # prepare visualization, commenting out to avoid plt show
        # plt.set_cmap('jet')
        if showplot: # pragma: no cover
            get_ipython().run_line_magic('matplotlib', 'inline')
        # set up the dataset object
        if '*' in filename:
            files = glob.glob(os.path.join(self.image_folder, filename))
            files = [os.path.basename(file) for file in files]
        else:
            files = [filename]
        self.files = files

        logging.info(f'Found files {files}')

        X_train = []
        X_val = []
        X_test = []
        Y_train = []
        Y_val = []
        Y_test = []
        meta_list_test = []
        Ycat = []
        xrange = []
        meta_list_train  = []
        meta_list_val  = []
        for filename in files:
            logging.info(filename)
            if filename not in self.dict_scaling.keys():
                self.dict_scaling[filename] = return_scale(self.image_folder, filename)
            if self.include_metadata:
                one_meta_list = return_meta_data(self.image_folder, filename)
            X, _Ycat, names = image_prep(self.image_folder, filename, nclass=nclass, showplot=showplot, dummy_ycat = self.dummy_ycat)
            self.names = names

            if X is None:
                if len(files) == 1:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT),
                        os.path.join(self.image_folder, filename))
                else:
                    pass
            if showplot: # pragma: no cover
                plt.figure(figsize=(10,10))
                plt.imshow(_Ycat)
                plt.show()

            self.names = names
            strat_n, _xrange = image_split(X, filename, cv_split=cv_split, saveplot=False, subpix = subpix)
            
            _X_train, _X_val, _Y_train, _Y_val = image_augment(X, _Ycat, strat_n, nclass=nclass, cv_split=cv_split, subpix = subpix)
            if self.dummy_ycat:
                _X_test = X
                _Y_test = _Ycat
            else:
                _X_test = X[:,_xrange[0]:,:]
                _Y_test = _Ycat[:,_xrange[0]:]
         
            if self.test_version:
                _X_train, _X_val, _Y_train, _Y_val = _X_train[:1, :, :, :], _X_val[:1, :, :, :], _Y_train[:1, :, :, :], _Y_val[:1, :, :, :]

            if _Y_train.shape[-1] < nclass:
                pw = [(0, 0) for _ in range(len(_Y_train.shape)-1)]
                pw.append((0, nclass - _Y_train.shape[-1]))
                _Y_train = np.pad(_Y_train, pw, mode='constant', constant_values=0)
            if _Y_val.shape[-1] < nclass:
                pw = [(0, 0) for _ in range(len(_Y_val.shape)-1)]
                pw.append((0, nclass - _Y_val.shape[-1]))
                _Y_val = np.pad(_Y_val, pw, mode='constant', constant_values=0)
            if _Y_test.shape[-1] < nclass:
                pw = [(0, 0) for _ in range(len(_Y_test.shape)-1)]
                pw.append((0, nclass - _Y_test.shape[-1]))
                _Y_test = np.pad(_Y_test, pw, mode='constant', constant_values=0)

            X_train.append(_X_train)
            X_val.append(_X_val)
            X_test.append(_X_test)

            Y_train.append(_Y_train)
            Y_val.append(_Y_val)
            Y_test.append(_Y_test)

            # add width and height
            self.dict_scaling[filename]['pixel_width'] = _X_test.shape[1]
            self.dict_scaling[filename]['pixel_height'] = _X_test.shape[0]

            if self.include_metadata:
                _meta_list_train = [one_meta_list[:] for _ in range(len(_X_train))]
                _meta_list_val = [one_meta_list[:] for _ in range(len(_X_val))]

                meta_list_train.append(_meta_list_train)
                meta_list_val.append(_meta_list_val)
                meta_list_test.append(one_meta_list)
        

        
        X_train = np.concatenate(X_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)

        Y_train = np.concatenate(Y_train, axis=0)
        Y_val = np.concatenate(Y_val, axis=0)

        if self.include_metadata:
            meta_list_train = np.concatenate(meta_list_train, axis=0)
            meta_list_val = np.concatenate(meta_list_val, axis=0)

            # transform data (fit only to training data)
            scaler = MinMaxScaler()
            meta_list_train = scaler.fit_transform(meta_list_train)
            meta_list_val = scaler.transform(meta_list_val)
            meta_list_test = scaler.transform(np.array(meta_list_test)).tolist()

            self.meta_list_test = meta_list_test

        #saving so can use this later in predictions
        self.X_test = X_test
        self.Y_test = Y_test

        # setup the dataloader object
        if self.include_metadata:
            (self.train_data_loader, self.val_data_loader,
            self.train_dataset, self.val_dataset) \
                = torch_load_data_metadata(X_train, meta_list_train, X_val, meta_list_val, Y_train, Y_val, batch_size, shuffle=True)
        else:
            (self.train_data_loader, self.val_data_loader,
            self.train_dataset, self.val_dataset) \
                = torch_load_data(X_train, X_val, Y_train, Y_val, batch_size, shuffle=True)

        # setup the model
        if Ycat is not None:
            if self.loss_type in ['CE', 'EWCE', 'FL', 'WCE', "TOPO"]:
                pass

            else:
                logging.error('Incorrect loss_type. Should be CE, FL, EWCE, WCE, TOPO.')
                raise Exception('Incorrect loss_type. Should be CE, FL, EWCE, WCE, TOPO.')
            if nclass is None:
                unique, counts = np.unique(Ycat, return_counts=True)
                nclass = len(unique)
                logging.info(f'Number of classes set to {nclass}')

        model_dict = MODELS[model_name]

        if self.model_name == "resnet50" \
                or self.model_name == "resnet101":
            self.model = model_dict['model'](num_classes=nclass, **model_dict['kw'])
        else:
            self.model = model_dict['model'](channels=channels, nclass=nclass, **model_dict['kw'])
        if gpu:
            #Detect and print if there's a GPU
            if device is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            self.model.to(self.device)
            logging.debug("Model transferred to GPUs.")
        # setup the loss
        if self.loss_type == "CE":
            self.loss_fn = nn.modules.loss.CrossEntropyLoss() #Cross entropy loss
        if self.loss_type == "TOPO":
            if weights is None: # use EWCE weight
                weights = [10.0]* nclass
                weights[0] = 1.0
                if nclass == 6:
                    weights[-1] = 1.0
                logging.info(f"TOPO EWCE as: " + str(weights))
                weights = torch.tensor(weights)
                if gpu:
                    weights = weights.to(self.device)
            self.loss_fn = TopoLoss(weights = weights) 
        elif self.loss_type == "MSE": # pragma: no cover
            self.loss_fn = nn.modules.loss.MSELoss() #MSE Loss
        elif self.loss_type == "MAE": # pragma: no cover
            self.loss_fn = nn.modules.loss.L1Loss() #MAE loss
        elif self.loss_type == "FL": # pragma: no cover
            if weights is None: # use EWCE weight
                weights = [10.0]* nclass
                weights[0] = 1.0
                if nclass == 6:
                    weights[-1] = 1.0
                logging.info(f"EWCE as: " + str(weights))
                weights = torch.tensor(weights)
                if gpu:
                    weights = weights.to(self.device)
            self.loss_fn = FocalLoss(alpha = weights, gamma = 2.) 
        elif self.loss_type == "WCE":
            if weights is None:
                logging.info("Calculating weights from the training set as 1/N")
                labels = []
                for train_data in self.train_data_loader:
                    if len(train_data) == 3:
                        _, _, true = train_data
                    else:
                        _, true = train_data
                    labels.append(true)

                true = torch.cat(labels).view(-1, nclass)
                weights = torch.zeros((nclass,))
                for i in range(nclass):
                    weights[i] = true[:, i].sum() + 1
                weights = weights / weights.min()
                weights = 1.0 / weights
                logging.info(f"Calculated weights as: {weights.numpy()}")
                if gpu:
                    weights = weights.to(self.device)
                self.loss_fn = nn.modules.loss.CrossEntropyLoss(weight = weights)
                #self.loss_fn = nn.modules.loss.CrossEntropyLoss() #Cross entropy loss
                #logging.info("No weights supplied. Equal weighting cross entropy used.")
            else:
                #For grain, boundary, void, precipitate, and impurities, respectively
                weights = torch.tensor(weights)
                #Weights need to be converted to a torch tensor and loaded to your GPU device(s)
                if gpu:
                    weights = weights.to(self.device)
                self.loss_fn = nn.modules.loss.CrossEntropyLoss(weight = weights)
        elif self.loss_type == "EWCE":
            if weights is None:
                weights = [10.0]* nclass
                weights[0] = 1.0
                if nclass == 6:
                    weights[-1] = 1.0
                logging.info(f"EWCE as: " + str(weights))
                weights = torch.tensor(weights)
                if gpu:
                    weights = weights.to(self.device)

                self.loss_fn = nn.modules.loss.CrossEntropyLoss(weight = weights)
                #self.loss_fn = nn.modules.loss.CrossEntropyLoss() #Cross entropy loss
                #logging.info("No weights supplied. Equal weighting cross entropy used.")
            else:
                #For grain, boundary, void, precipitate, and impurities, respectively
                weights = torch.tensor(weights)
                #Weights need to be converted to a torch tensor and loaded to your GPU device(s)
                if gpu:
                    weights = weights.to(self.device)
                self.loss_fn = nn.modules.loss.CrossEntropyLoss(weight = weights)
        # setup the optimizer
        if  self.optimize_option == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = learn_rate) #RMSprop optimizer
        elif  self.optimize_option == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr = learn_rate, momentum = 0.9, nesterov=True)
        elif  self.optimize_option == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = learn_rate)
        else:
            pass #Add more if/else statements with more optimizers for more flexibility
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = factor, patience = patience)
        # create a descriptive string for this training
        self.train_description = f'{self.folder}{self.model_name}{self.model_name_suffix}_' \
            + f'{self.loss_type}_{self.nclass}'
        
        with open(self.train_description + "_scale.json", 'w') as fp:
            json.dump(self.dict_scaling, fp)

        # pt_path is just train_description if original models located same place (folder) as everything else
        # if original model is not in the "folder" lcoation, it won't be overwritten with training here
        self.pt_path = f'{self.model_path}{self.model_name}{self.model_name_suffix}_'  \
            + f'{self.loss_type}_{self.nclass}'

        logging.info(self.pt_path)
        # Set up the training loss trackers
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = [0.0]
        self.best_val_acc = -np.inf
        if weights_file is not None:
            self.load(weights_file)
        elif load_pt_files == True and os.path.isfile(f"{self.pt_path}class_model_weights.pt"):
            self.load(f"{self.pt_path}class_model_weights.pt")
            logging.info("pt file loaded")
            logging.info(f"{self.pt_path}class_model_weights.pt")
        else:
            logging.info("pt file NOT loaded")
            logging.info(f"{self.pt_path}class_model_weights.pt")
            pass

    def load(self, file):
        """Load pretrained weights from file.

        :param str file: Filename of a saved pytorch state dict including the
            desired weights of the model.  Will be mapped to currently set
            model device.  Must match the model architecture.
        """
        if isinstance(file, str):
            file = torch.load(file, map_location=self.device)
        self.model.load_state_dict(file)


    def __call__(self):
        self.train()

    def train(self):
        """Perform gradient descent to train the model."""
        self.t0 = time()
        for self.i_epoch in range(self.N_epochs):
            self.one_epoch()
        self.t0 = time() - self.t0
        with open(f'{self.train_description}_seconds.txt', 'w') as f:
            f.write(f'{self.t0:0.3f}')
        self.save_results()
        self.save_model()

    def save_results(self, result_type='train'):
        """Save the loss and accuracy history to a csv file.

        :param str result_type: Either ``'train'`` or ``'val'``, defining
            whether to save training or validation results
        """
        if result_type == 'train':
            _loss = self.train_loss
            _acc = self.train_acc
        elif result_type == 'val':
            _loss = self.val_loss
            _acc = self.val_acc
        np.savetxt(f"{self.train_description}class_{result_type}_loss.csv",
                   _loss, delimiter = ",")
        np.savetxt(f"{self.train_description}class_{result_type}_accuracy.csv",
                   _acc, delimiter = ",")

    def save_model(self):
        """Save the current model weights to file."""
        filename = f"{self.train_description}class_model_weights.pt"
        if self.val_acc[-1] > self.best_val_acc:
            logging.info(f"Saving model to {filename}")
            torch.save(self.model.state_dict(), filename)
            self.best_val_acc = self.val_acc[-1]

    def one_epoch(self):
        """Train the model through one epoch."""
        temp_loss = []
        temp_acc = []
        for self.i_batch, input_files in enumerate(self.train_data_loader):
            if len(input_files) == 3:
                (batch, metadata , true) = input_files
                metadata = metadata.to(self.device)
            else:
                (batch, true) = input_files
            #train
            batch = batch.to(self.device)
            true = true.to(self.device)

            self.optimizer.zero_grad()
            if self.model_name == "segnet":
                output = self.model(batch.permute(0,3,1,2))
            if self.model_name == "segnetmeta":
                output = self.model(batch.permute(0,3,1,2), metadata )
            elif self.model_name == "resnet50" \
                    or self.model_name == "resnet101":
                output = self.model(batch.permute(0,3,1,2))["out"]
            else:
                output = self.model(batch.permute(0,3,1,2))
            
            if self.loss_type == "TOPO":
                loss = self.loss_fn(output, true.argmax(3), self.i_epoch)
            else:
                loss = self.loss_fn(output, true.argmax(3))
            
            loss.backward()
            self.optimizer.step()
            acc = (output.argmax(1) == true.argmax(3)).float().mean().item()
            logging.info(f'[{self.i_epoch:3d}, {self.i_batch:5d}]: {loss.detach().item():.2e} | {100.0 * acc:0.2f}')
            if len(temp_loss) > 9:
                del temp_loss[0]
            if len(temp_acc) > 10:
                del temp_acc[0]
            temp_loss.append(loss.item())
            temp_acc.append(acc)
        self.train_loss.append(np.mean(temp_loss))
        self.train_acc.append(np.mean(temp_acc))
        logging.info(f'{self.i_epoch + 1} loss: {np.mean(temp_loss):.5f}')
        if self.i_epoch % self.n_save == 0:
            self.save_results('train')
        self.scheduler.step(loss)
        self.validate()



    def validate(self):
        """Calculate and display prediction performance on the validation set.
        """
        self.model.eval()
        logging.info('')
        with torch.no_grad():
                temp_loss = []
                temp_acc = []
                for j, zipped_input in enumerate(self.val_data_loader):
                    if len(zipped_input) == 3:
                        (val_batch, val_metadata, val_true) = zipped_input
                        val_metadata =  val_metadata.to(self.device)
                    else:
                        (val_batch, val_true) = zipped_input
                    val_batch =  val_batch.to(self.device)
                    val_true =  val_true.to(self.device)
                    if self.model_name == "segnet":
                        output = self.model(val_batch.permute(0,3,1,2))
                    elif self.model_name == "segnetmeta":
                        output = self.model(val_batch.permute(0,3,1,2), val_metadata)
                    elif self.model_name == "resnet50" \
                            or self.model_name == "resnet101":
                        output = self.model(val_batch.permute(0,3,1,2))["out"]
                    else:
                        output = self.model(val_batch.permute(0,3,1,2))
                    if self.loss_type == "TOPO":
                        loss = self.loss_fn(output, val_true.argmax(3), self.i_epoch)
                    else:
                        loss = self.loss_fn(output, val_true.argmax(3))
                    acc = (output.argmax(1) == val_true.argmax(3)).float()\
                        .mean().item()
                    logging.info(f"Validation Accuracy: {acc:.2f}")
                    temp_loss.append(loss.item())
                    temp_acc.append(acc)
                    #if j > 10:
                    #    break
                # Mean by epoch #
                self.val_loss.append(np.mean(temp_loss))
                self.val_acc.append(np.mean(temp_acc))


                if self.i_epoch % self.n_save == 0:
                    np.savetxt(f"{self.train_description}class_val_loss.csv",
                               self.val_loss, delimiter = ",")
                    np.savetxt(f"{self.train_description}class_val_accuracy.csv",
                               self.val_acc, delimiter = ",")
                    self.save_results('val')
                    self.save_model()
                logging.info(f'[{self.i_epoch + 1}, {self.i_batch:5d}] validation loss: {np.mean(temp_loss):.5f}')
        self.model.train()


    def run_ripley(self, clusters_pred = True):
        """run ripley on predicted result
        """

        # collecting ripley dfs to save to csv
        ripley_df_center =[]
        for i_image, (_Y_test, filename) in enumerate(zip(self.Y_test, self.files)):
            total_area_um = self.dict_scaling[filename]['pixel_width'] * \
                self.dict_scaling[filename]['pixel_height']/self.dict_scaling[filename]['scale']**2
            
            
            pix_to_scale = self.dict_scaling[filename]['scale']

            image_identifier = self.image_folder.replace("/","_")

            self.file_prefix = f"{image_identifier}_{self.model_name}{self.model_name_suffix}_{self.loss_type}_{self.nclass}"
            if not os.path.exists(os.path.join(f"{self.train_description}", "ripley")):
                os.makedirs(os.path.join(f"{self.train_description}", "ripley"))
            
            if not os.path.exists(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(True))):
                os.makedirs(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(True)))

            if not os.path.exists(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(False))):
                os.makedirs(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(False)))
            
            #set max to the image width (shape[1]) and height (shape[0])
            mins = np.array([0., 0.])
            maxs = np.array([float(_Y_test.shape[1])/pix_to_scale,
                            float(_Y_test.shape[0])/pix_to_scale])

            #return locations of defects
            if clusters_pred == False:
                centers_with_zeros, area, rect, box, onbnd, nitems, full_df, dict_ob_wrapper  = defect_summary(_Y_test, names = self.names, nclass = self.nclass)
            else:
                #get _Y_pred if clusters_pred is true
                pkl_filename = f"{self.train_description}_cv{self.cv_split}_image{filename}_class_predmap.pkl"
                
                logging.info(pkl_filename)
                _Y_pred = pickle.load(open(pkl_filename, 'rb')).squeeze()
                centers_with_zeros, area, rect, box, onbnd, nitems, full_df, dict_ob_wrapper  = defect_summary(_Y_pred, names = self.names, nclass = self.nclass)

            #remove any center with 0 area and nan border
            centers = []
            for i,c in enumerate(centers_with_zeros):
                
                if len(c) > 0:
                    x, y = np.split(c, 2, axis=1)
                    cex = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(
                    ), "area": area[i], "on_boundary": onbnd[i], "names": self.names[i + 2], "filenames": filename, "clusters_pred": clusters_pred})
                    cex.iloc[:, 0] = cex.iloc[:, 0].apply(
                        lambda x: x/pix_to_scale)
                    cex.iloc[:, 1] = cex.iloc[:, 1].apply(
                        lambda x: x/pix_to_scale)
                    cex.iloc[:, 2] = cex.iloc[:, 2].apply(
                        lambda x: x/pix_to_scale**2)
                    cex = cex.loc[cex['area'] > self.alim_um] 
                    ripley_df_center.append(cex)
                    # eleminate centers with area below alim and convert to um
                    centers.append(cex[['x', 'y']].to_numpy())
                else:
                    centers.append([])
            
            # since everything has been converted to um, convert radius values to 20 um
            r = np.linspace(0, 20, 200)

            for num, X in enumerate(centers):
                
                plt.ioff()
                plt.close('all')
                if len(X) > 1:
                    #the confidence will have the same number of clusters as the val image, but random distribution
                    CR_univariate = MCconfidence_univariate(r, mins, maxs, n1 = len(X), alpha=0.01, N=1000, mode='ripley')
                    Kest_univariate = RipleysKEstimator_univariate(area=np.prod(maxs - mins), x_max=maxs[0], y_max=maxs[1], x_min=mins[0],
                                             y_min=mins[1])
                    Hp_univariate = Kest_univariate.Hfunction(data1=X, radii=r, mode='ripley')

                    #retun r, Hp_univariate (actual data), and CR_univariate lower and upper bound ripley dataframe
                    ripley_dataset = np.column_stack([r, Hp_univariate, CR_univariate[:, 0], CR_univariate[:, 1]])
                    np.savetxt(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                                                f'{self.file_prefix}_{self.names[num + 2]}_pred_{clusters_pred}_{filename}_univariate_ripley_clusters.csv'),
                               ripley_dataset, delimiter = ",", header = "r,ripley,LCI,UCI")

                    #return plot in train_description folder
                    plt.plot(r, np.zeros(len(r)), label=r'$H_{pois}$')
                    plt.plot(r, Hp_univariate, color='blue', label=r'$H_{ripley}$')
                    plt.plot(r, CR_univariate[:, 0], color='green', ls=':', label=r'$LCI_{ripley}$')
                    plt.plot(
                        r, CR_univariate[:, 1], color='green', ls=':', label=r'$UCI_{ripley}$')
                    plt.title(f'{self.file_prefix}_{self.names[num+ 2]}')
                    plt.legend()
                    plt.savefig(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                f'{self.file_prefix}_{self.names[num+ 2]}_pred_{clusters_pred}_{filename}_univariate_ripley_clusters.pdf'))
                    plt.savefig(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                f'{self.file_prefix}_{self.names[num+ 2]}_pred_{clusters_pred}_{filename}_univariate_ripley_clusters.png'))
                    plt.ioff()
                    plt.close('all')
                else:
                    print("not enough points " + self.names[num+ 2])
            #for each combinations of centers, run ripley
            for subset in permutations(range(len(centers)), 2):
                #index for centers
                num1 = subset[0]
                num2 = subset[1]
                center1 = centers[num1]
                center2 = centers[num2]

                #number of data points
                n1 = len(center1)
                n2 = len(center2)

                if n1 > 0 and n2 > 0:
                    #the confidence will have the same number of clusters as the val image, but random distribution
                    CR = MCconfidence(r, mins, maxs, n1 = n1, n2 = n2, alpha=0.01, N=1000, mode='ripley')
                    Kest = RipleysKEstimator(area=np.prod(maxs - mins), x_max=maxs[0], y_max=maxs[1], x_min=mins[0], y_min=mins[1])
                    Hp = Kest.Hfunction(data1 = center1, data2 = center2, radii=r, mode='ripley')

                    #retun r, Hp (actual data), and CR lower and upper bound ripley dataframe
                    ripley_dataset = np.column_stack([r, Hp, CR[:, 0], CR[:, 1]])
                    np.savetxt(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                                                f'{self.file_prefix}_{self.names[num1 + 2]}_{self.names[num2 + 2]}_pred_{clusters_pred}_{filename}_ripley_clusters.csv'),
                               ripley_dataset, delimiter = ",", header = "r,ripley,LCI,UCI")

                    #return plot in train_description folder
                    plt.plot(r, np.zeros(len(r)), label=r'$H_{pois}$')
                    plt.plot(r, Hp, color='blue', label=r'$H_{ripley}$')
                    plt.plot(r, CR[:, 0], color='green', ls=':', label=r'$LCI_{ripley}$')
                    plt.plot(r, CR[:, 1], color='green', ls=':', label=r'$UCI_{ripley}$')
                    plt.title(f'{self.names[num1 + 2]}_{self.names[num2 + 2]}')
                    plt.legend()
                    plt.savefig(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                f'{self.file_prefix}_{self.names[num1 + 2]}_{self.names[num2 + 2]}_pred_{clusters_pred}_{filename}_ripley_clusters.pdf'))
                    plt.savefig(os.path.join(f"{self.train_description}", "ripley", "cluster_pred_" + str(clusters_pred),
                                f'{self.file_prefix}_{self.names[num1 + 2]}_{self.names[num2 + 2]}_pred_{clusters_pred}_{filename}_ripley_clusters.png'))
                    plt.ioff()
                    plt.close('all')

                else:
                    print("not enough points " + self.names[num1 + 2] + self.names[num2 + 2])
        pd.concat(ripley_df_center).to_csv(os.path.join(
            f"{self.train_description}", "ripley",  "ripley_values.csv"))
        return

    def test_predictions(self, no_running_mean=False):
        """Obtain the predictions for the hold-out test set.
            """
        if self.include_metadata:
            zipped_files = zip(self.files, self.X_test, self.Y_test, self.meta_list_test)
        else:
            zipped_files = zip(self.files, self.X_test, self.Y_test)

        for zipped_input in zipped_files:
            if len(zipped_input) == 4:
                (file, _X_test, _Y_test, metadata) = zipped_input
                metadata = torch.from_numpy(np.array(metadata)).float()
            else:
                (file, _X_test, _Y_test) = zipped_input
            x = torch.from_numpy(_X_test).float()
            x = x[None, ...]  # Add batch dim of 1
            
            #deleted the dataloader because with the for-loop for file, _X_test, _Y_test, I don't think it's necessary

            #load model parameters--map location says where to load and hold model params. Adjust path for cluster vs local machine.
            #state_dict = torch.load('{}class_model_weights.pt'.format(self.train_description), map_location='cpu')
            #self.model.load_state_dict(state_dict)  # use new_state_dict if 'module' prefix needs to be removed

            #Put loaded model in evaluation mode so we don't accidentally train or modify parameters as we use the model.
            # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/66?page=4
            if no_running_mean:
                for m in self.model.modules():
                    for child in m.children():
                        if type(child) == nn.BatchNorm2d:
                            child.track_running_stats = False
                            child.running_mean = None
                            child.running_var = None
            self.model.eval()
            s = nn.Softmax2d()


            #Freeze model parameters
            with torch.no_grad():
                if self.gpu:
                    x = x.to(self.device)
                if self.include_metadata:
                    metadata = metadata.to(self.device)

                if self.model_name == "segnetmeta":
                    output = s(self.model(x.permute(0,3,1,2), metadata))
                elif self.model_name == "resnet50" \
                        or self.model_name == "resnet101":
                    output = s(self.model(x.permute(0,3,1,2))["out"])
                else:
                    output = s(self.model(x.permute(0,3,1,2)))
    

                #Get predictions from fitted model, collapse the 5 channel prediction maps into one image, then save
                pred = torch.argmax(output, dim=1)

                if self.gpu:
                    _seg_arr = pred[0, ...].data.cpu().numpy()
                else:
                    _seg_arr = pred[0, ...].data.numpy()

            if self.saveres:
                filename = f"{self.train_description}_cv{self.cv_split}_image{file}_class_predmap.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(_seg_arr, f)

    def performance_metrics(self, saveres= False, showres = False, saveimage =True, performance_calibrate= False, short_version = False):
        """Return performance metrics)
        """
        label = self.label
        loss = self.loss_type
        names = self.names
        #save extra csv files
        saveres = saveres
        showres = showres
        model = self.model_name
        nclass = self.nclass
        test_version = self.test_version
        image_identifier = self.image_folder.replace("/","_")

        self.file_prefix = f"{image_identifier}_{self.model_name}{self.model_name_suffix}_{self.loss_type}_{self.nclass}"
        # crete train_description_folder, then add results, which include self.train_description = f'{self.folder}{self.model_name}{self.model_name_suffix}_' \
        #    + f'{self.loss_type}_{self.nclass}'
        file_path_defect = os.path.join(
            f"{self.train_description}", f"{self.file_prefix}_table_defects.csv")
        file_path_defect_truth = os.path.join(
            f"{self.train_description}", f"{self.file_prefix}_table_defects_truth.csv")
        file_path_extra = os.path.join(
            f"{self.train_description}", "extra_outputs")
        file_path_csv = os.path.join( f"{self.train_description}", f"{self.file_prefix}_performance_metrics.csv")
        file_path_short = os.path.join(
            f"{self.train_description}", f"{self.file_prefix}_performance_metrics_short.csv")

        if not os.path.exists(f"{self.train_description}"):
            os.makedirs(f"{self.train_description}")
        if not os.path.exists(file_path_extra):
                    os.makedirs(file_path_extra)
        if not os.path.exists(os.path.join(f"{self.train_description}", "image")):
            os.makedirs(os.path.join(f"{self.train_description}", "image"))

        if performance_calibrate:
            names.append("extra")

        return_values = {}
        file_pred_truth = []

        short_df_title = pd.DataFrame(["Output:"])
        short_df_title.to_csv(file_path_short, mode='w',
                             header=True, index=False)


        num_files = 0
        for file, Y_test, X_test in zip(self.files, self.Y_test, self.X_test):
            

            # total area in um, for calculating density
            total_area_um = self.dict_scaling[file]['pixel_width'] * \
                self.dict_scaling[file]['pixel_height']/self.dict_scaling[file]['scale']**2
            
            
            pix_to_scale = self.dict_scaling[file]['scale']

            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2

            num_files += 1
            if test_version:
                zero_image = np.zeros((100, 100))
                Y_test = zero_image.copy()
                Y_test[0:20, 0:20] = 3
                Y_test[20:50, 10:40] = 2
                Y_test[60:80, 10:40] = 4
                Y_test[40, 42] = 1
                p_test = zero_image.copy()
                if num_files > 1:
                    p_test[0:20, 10:30] = 3
                    p_test[25:50, 10:30] = 2
                    p_test[60:90, 10:20] = 4
                    # p_test[80:100, 40:80] =5
                    p_test[40, 10] = 1
                else:

                    p_test[0:20, 10:30] = 3
                    p_test[20:50, 5:20] = 2
                    p_test[60:80, 10:20] = 4
                    # p_test[80:100, 40:80] =5
                    p_test[40, 30] = 1

            else:
                if performance_calibrate:
                    #get pred and truth for the y_test part of the image
                    p_correct_per_pixel = f"{self.train_description}_cv{self.cv_split}_image{file}_p_correct_per_pixel.pkl"
                    with open(p_correct_per_pixel, 'rb') as f:
                        p_correct_per_pixel = pickle.load(f).squeeze()
                    hypothesis_class_per_pixel = f"{self.train_description}_cv{self.cv_split}_image{file}_hypothesis_class_per_pixel.pkl"
                    with open(hypothesis_class_per_pixel, 'rb') as f:
                        hypothesis_class_per_pixel = pickle.load(f).squeeze()
                    p_test = np.array(hypothesis_class_per_pixel)
                    p_test = np.where(
                        p_correct_per_pixel < .95, 5, hypothesis_class_per_pixel)
                    for i in [.40, .60, .70, .80, .90, .95]:
                        p_test_temp = np.where(
                        p_correct_per_pixel < i, 5, hypothesis_class_per_pixel)
                        
                        vis_chip(p_test_temp,  nclass = 6, filename=os.path.join(f"{self.train_description}",  "image",
                                                                     f"{self.file_prefix}_{file}_p_test_temp{i}.pdf"), use_cbar=True)
                    if saveimage:
                        vis_chip(p_test, filename=os.path.join(f"{self.train_description}",  "image",
                                                                     f"{self.file_prefix}_{file}_p_test_{i}.pdf"), use_cbar=True)

                        vis_chip(hypothesis_class_per_pixel, alpha=p_correct_per_pixel, filename=os.path.join(f"{self.train_description}",  "image",
                                                                                                               f"{self.file_prefix}_{file}_hypothesis_class_per_pixel.pdf"),  use_cbar=True)
                        
                else:
                    #get pred and truth for the y_test part of the image
                    filename = f"{self.train_description}_cv{self.cv_split}_image{file}_class_predmap.pkl"
                    with open(filename, 'rb') as f:
                        p_test = pickle.load(f).squeeze()
                    print("p_test")
                    print(np.unique(p_test, return_counts=True))

            file_pred_truth.append((Y_test, p_test, file))
            #save truth and pred value
            if saveimage:
                vis_chip(p_test, filename= os.path.join(f"{self.train_description}", "image",
                                                         f"{self.file_prefix}_{file}_pred_image.pdf"), use_cbar=True)
                vis_chip(p_test, filename=os.path.join(f"{self.train_description}", "image",
                                                       f"{self.file_prefix}_{file}_pred_image.jpg"), use_cbar=True)
                vals = X_test.flatten().astype(float)
                plt.hist(vals, 10)
                plt.title('Image Intensity')
                plt.xlabel('Intensity')
                plt.ylabel('# of counts')
                plt.savefig(os.path.join(f"{self.train_description}", "image",
                                         f"{self.file_prefix}_{file}_histogram.png"))
                plt.ioff()
                plt.close('all')

                if not self.dummy_ycat:

                    logging.info(Y_test.shape)
                    vis_chip(Y_test, filename=os.path.join(f"{self.train_description}", "image",
                                                        f"{self.file_prefix}_{file}_truth_image.pdf"), use_cbar=True)
                                                        
                    vis_chip(Y_test, filename=os.path.join(f"{self.train_description}", "image",
                                                        f"{self.file_prefix}_{file}_truth_image.jpg"), use_cbar = True)


                    locs = np.where(Y_test == p_test)
                    imgray_pred = p_test.copy()
                    imgray_pred[locs] = 16777215
                    
                    vis_chip(imgray_pred, filename = os.path.join(f"{self.train_description}", "image",
                                                        f"{self.file_prefix}_{file}_diff_pred_image.pdf"), use_cbar = True)
                                                        
                    vis_chip(imgray_pred, filename=os.path.join(f"{self.train_description}", "image",
                                                                f"{self.file_prefix}_{file}_diff_pred_image.jpg"), use_cbar=True)

                    imgray_true = Y_test.copy()
                    imgray_true[locs] = 16777215
                    
                    vis_chip(imgray_true, filename = os.path.join(f"{self.train_description}", "image",
                                                                f"{self.file_prefix}_{file}_diff_true_image.pdf"), use_cbar = True)
                    
                    vis_chip(imgray_true, filename = os.path.join(f"{self.train_description}", "image",
                                                                f"{self.file_prefix}_{file}_diff_true_image.jpg"), use_cbar = True)



        total_preds = np.concatenate([p_test.ravel() for Y_test, p_test, file in file_pred_truth])
        total_Y_test = np.concatenate([Y_test.ravel() for Y_test, p_test, file in file_pred_truth])

        pixel_proportion_list = []

        for (Y_test, p_test, file) in file_pred_truth:
            pix_to_scale = self.dict_scaling[file]['scale']
            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2

            #save the name of the file
            Ppvaldf = pixel_proportion(Y_test.ravel(), p_test.ravel(), label=label, names=names, showres=showres,
                                       saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass=nclass)
            pixel_proportion_pred = pd.DataFrame()

            pixel_proportion_pred["Percentage"] = Ppvaldf['Pred']
            pixel_proportion_pred["Defect"] = Ppvaldf.index
            pixel_proportion_pred["Image"] = file
            pixel_proportion_pred["Type"] = 'prediction'

            pixel_proportion_pred.to_csv(file_path_short, mode='a',  header=False)

            pixel_proportion_truth = pixel_proportion_pred
            pixel_proportion_truth["Percentage"] = Ppvaldf['Proportion']
            pixel_proportion_truth["Type"] = 'truth'

            pixel_proportion_truth.to_csv(file_path_short, mode='a',  header=False)

            pixel_proportion_list.append(Ppvaldf)

            fig, ax = plt.subplots()

            plt.title('Pixel Proportion:' + file)
            plt.xlabel('Percentage')
            plt.ylabel('Defects Types')

            # Plot each bar separately and give it a label.
            for index, row in Ppvaldf.iterrows():
                ax.bar([index], [row["Pred"]], label=index)


            plt.savefig(os.path.join(f"{self.train_description}", "image",
                                        f"{self.file_prefix}_{file}_proportion_graph.png"))
            plt.close()

        #accuracy as if big image
        df1 = pd.DataFrame([[""], ["Overall as One Image"]])
        df1.to_csv(file_path_short, mode='a', header=False, index=False)
        pixel_proportion_total = pixel_proportion(total_Y_test, total_preds, label=label, names=names, showres=showres,
                                                  saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass=nclass)

        pixel_proportion_total.to_csv(file_path_short, mode='a')
        # write everything to this csv file
        df1 = pd.DataFrame(["Output:"])
        df1.to_csv(file_path_csv, mode='w', header=False, index=False)


  
        accuracy_df_list = []
        #store total preds and truth
        p_test_list  = []
        Y_test_list  = []

        return_values["accuracy_df"] = {}
        return_values["accuracy_df"]['file'] = {}
        for (Y_test, p_test, file) in file_pred_truth:
            pix_to_scale = self.dict_scaling[file]['scale']
            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2
            #save the name of the file
            df1 = pd.DataFrame([[""],[file]])
            # df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            accuracy_df = accuracy_class(Y_test, p_test, names)
            # accuracy_df.to_csv(file_path_csv, mode='a',  header=False)
            accuracy_df.to_csv(file_path_csv, mode='a',  header=False)
            accuracy_df_list.append(accuracy_df)
            return_values["accuracy_df"]['file'][file] = accuracy_df

        #accuracy as if big image
        df1 = pd.DataFrame([[""],["Overall as One Image"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        accuracy_df_total = accuracy_class(total_Y_test, total_preds, names)
        # accuracy_df_total.to_csv(file_path_csv, mode='a', header=False)
        accuracy_df_total.to_csv(file_path_csv, mode='a', header=False)
        return_values["accuracy_df"]["overall"] = accuracy_df_total



        #accuarcy as average
        df1 = pd.DataFrame([[""],["Average Cell by Cell"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(accuracy_df_list).groupby(level=0).mean()
        df1 = df1.reindex(accuracy_df_total.index)
        # df1.to_csv(file_path_csv, mode='a', header=False)
        df1.to_csv(file_path_csv, mode='a', header=False)
        return_values["accuracy_df"]["average"] = df1

        pout_list = []
        pss_equal_list = []
        pss_weighted_list = []
        pcm_list = []

        return_values["iou_df"] = {}
        return_values["iou_df"]['pout'] = {}
        return_values["iou_df"]['pss_weighted'] = {}
        return_values["iou_df"]['pss_equal'] = {}
        return_values["iou_df"]['pcm'] = {}

        for (Y_test, p_test, file) in file_pred_truth:
            pix_to_scale = self.dict_scaling[file]['scale']

            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2
            #save the name of the file
            df1 = pd.DataFrame([[""],["IOU Report"], [file]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)

            pix_to_scale = self.dict_scaling[file]['scale']

            pout, pss_weighted, pcm = iou_report(Y_test.ravel(), p_test.ravel(), weights ='weighted', names=names, showres=showres, saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass = nclass)
            _, pss_equal, _ = iou_report(Y_test.ravel(), p_test.ravel(), weights ='equal', names=names, showres=showres, saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass = nclass)

            df1 = pd.DataFrame([[""],["Individual Summaries:"]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            pout.to_csv(file_path_csv, mode='a')
            return_values["iou_df"]['pout'][file] = pout

            df1 = pd.DataFrame([[""],["Average Summaries - Weighted:"]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            pss_weighted.to_csv(file_path_csv, mode='a')
            return_values["iou_df"]['pss_weighted'][file] = pss_weighted

            df1 = pd.DataFrame([[""],["Average Summaries - Equal Weights:"]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            pss_equal.to_csv(file_path_csv, mode='a')
            return_values["iou_df"]['pss_equal'][file] = pss_equal

            df1 = pd.DataFrame([[""],["Confusion Matrix: "]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            pcm.to_csv(file_path_csv, mode='a')
            return_values["iou_df"]['pcm'][file] = pcm

            pout_list.append(pout)
            pss_equal_list.append(pss_equal)
            pss_weighted_list.append(pss_weighted)
            pcm_list.append(pcm)

        #accuracy as if big image
        df1 = pd.DataFrame([[""],["IOU Report: As Big Image"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

        pout, pss_weighted, pcm = iou_report(total_Y_test, total_preds, weights ='weighted', names=names, showres=showres, saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass = nclass)
        _, pss_equal, _ = iou_report(total_Y_test, total_preds, weights='equal',  names=names, showres=showres,
                                     saveres=saveres, path=file_path_extra, model_name=model, loss_type=loss, nclass=nclass)

        df1 = pd.DataFrame([[""],["Individual Summaries:"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        pout.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pout']["big_image"] = pout

        df1 = pd.DataFrame([[""],["Average Summaries - Weighted:"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        pss_weighted.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pss_weighted']["big_image"] = pss_weighted

        df1 = pd.DataFrame([[""],["Average Summaries - Equal Weights:"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        pss_equal.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pss_equal']["big_image"] = pss_equal

        df1 = pd.DataFrame([[""],["Confusion Matrix: "]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        pcm.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pcm']["big_image"] = pcm

        #accuarcy as average
        df1 = pd.DataFrame([[""], ["Average Cell by Cell"], ["Individual Summaries:"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(pout_list).groupby(level=0).mean()
        df1 = df1.reindex(pout.index)
        df1.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pout']["average"] = df1

        df1 = pd.DataFrame([[""],["Average Summaries - Weighted:"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(pss_weighted_list).groupby(level=0).mean()
        df1 = df1.reindex(pss_weighted.index)
        df1.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pss_weighted']["average"] = df1


        df1 = pd.DataFrame([[""],["Average Summaries - Equal"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(pss_equal_list).groupby(level=0).mean()
        df1 = df1.reindex(pss_equal.index)
        df1.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pss_equal']["average"] = df1

        df1 = pd.DataFrame([[""],["Confusion Matrix: "]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(pcm_list).groupby(level=0).mean()
        df1 = df1.reindex(pcm.index)
        df1.to_csv(file_path_csv, mode='a')
        return_values["iou_df"]['pcm']["average"] = df1

        #accuarcy as average
        df1 = pd.DataFrame([[""], ["Average Cell by Cell"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

        df1 = pd.concat(pixel_proportion_list).groupby(level=0).mean()
        df1 = df1.reindex(pixel_proportion_total.index)
        df1.to_csv(file_path_csv, mode='a')

        class_report_df_list = []

        for (Y_test, p_test, file) in file_pred_truth:
            pix_to_scale = self.dict_scaling[file]['scale']

            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2
            #save the name of the file
            df1 = pd.DataFrame([[""],["Modified Class Report"],[file]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            (outp, fscoreA, class_report_df) = modified_class_report(Y_test.ravel(), p_test.ravel(), names=names, B=[0.5,1,2], weights='both', path = file_path_extra)
            class_report_df.to_csv(file_path_csv, mode='a')
            class_report_df_list.append(class_report_df)

        #accuracy as if big image
        df1 = pd.DataFrame([[""], ["Overall as One Image"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        (outp, fscoreA, class_report_df) = modified_class_report(total_Y_test,
                                                                 total_preds, names=names, B=[0.5, 1, 2], weights='both', path=file_path_extra)
        class_report_df.to_csv(file_path_csv, mode='a')

        all_df_defect = pd.DataFrame(columns = ["x", "y",  "area", "on_boundary","names", "filenames"])
        all_df_defect.to_csv(file_path_defect , mode='w', header= True, index=False)
        all_df_defect.to_csv(file_path_defect_truth , mode='w', header= True, index=False)

        
        
        #accuarcy as average
        df1 = pd.DataFrame([[""], ["Average Cell by Cell"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

        df1 = pd.concat(class_report_df_list).groupby(level=0).mean()
        df1 = df1.reindex(class_report_df.index)
        df1.to_csv(file_path_csv, mode='a')

        defect_summary_dict = {'truth': {}, 'pred': {}}
        total_dict_iou = {'TP': {}, "PT": {}}
        Iou_report = []
        if short_version:
            return
        
        for (Y_test, p_test, file) in file_pred_truth:
            pix_to_scale = self.dict_scaling[file]['scale']
            # convert alim_um to pixels so can filter box_iou (which is still using pixels)
            alim_px = self.alim_um*pix_to_scale**2
            for img_type in ["truth", "pred"]:

                #save the name of the file
                df1 = pd.DataFrame([[""],[file], ["Defect Summary: " + img_type]])
                df1.to_csv(file_path_csv, mode='a', header=False, index = False)
                if img_type == "truth":
                    Tcenters, Tarea, Trect, Tbox, Tonbnd, nitems, defect_df, dict_ob_wrapper = defect_summary(Y_test, tbinom=None, showplot=False, saveplot=True, saveres=True, filename = file,  path=file_path_extra, model_name='truth', loss_type='', nclass = nclass, alim =alim_px)
                else:
                    Pcenters, Parea, Prect, Pbox, Ponbnd, nitems, defect_df, dict_ob_wrapper = defect_summary(p_test, tbinom=Tonbnd, showplot=False, saveplot=True, saveres=True, filename = file,  path=file_path_extra, model_name=model, loss_type=loss, nclass = nclass, alim =alim_px)

                #add defects
                df1 = pd.DataFrame([[""], [img_type], [file]])
                
                # before saving convert px/(px/um) and px**2/(px**2/um**2)
                defect_df.iloc[:, 0] = defect_df.iloc[:, 0].apply(
                    lambda x: x/pix_to_scale)
                defect_df.iloc[:, 1] = defect_df.iloc[:, 1].apply(
                    lambda x: x/pix_to_scale)
                defect_df.iloc[:, 2] = defect_df.iloc[:, 2].apply(
                    lambda x: x/pix_to_scale**2)

                if img_type == "pred":
                    df1.to_csv(file_path_defect, mode='a',
                               header=False, index=False)
                    
                    defect_df.to_csv(file_path_defect, mode='a',
                                     header=False, index=False)
                if img_type == "truth":
                    df1.to_csv(file_path_defect_truth, mode='a',
                               header=False, index=False)

                    defect_df.to_csv(file_path_defect_truth, mode='a',
                                     header=False, index=False)


                df1 = pd.DataFrame(['There are %i' %nitems + ' total clustered defects '])
                df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                for i in range(2,5):
                    #only run if defect exists in pred
                    if names[i] in dict_ob_wrapper:
                        ob = dict_ob_wrapper[names[i]]['ob']
                        tbinom = dict_ob_wrapper[names[i]]['tbinom']
                        nitems = dict_ob_wrapper[names[i]]['nitems']
                        if names[i] not in defect_summary_dict[img_type]:
                            defect_summary_dict[img_type][names[i]] = {'ob': [], 'tbinom': [], 'nitems': 0}
                        else:
                            pass
                        defect_summary_dict[img_type][names[i]]['ob'].extend(list(ob))
                        defect_summary_dict[img_type][names[i]]['nitems'] += nitems

                        df1 = pd.DataFrame([[""], ['Grain boundary for %s ' %names[i] + '= %.2f' %(np.nanmean(ob)) + '(%i)' %np.nansum(ob)]])
                        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                        df1 = pd.DataFrame([[""], ['There are %i' %nitems + " " + names[i] + ' defects']])
                        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                        stat, pval = proportions_ztest(np.nansum(ob), len(ob) - np.sum(np.isnan(ob)), 0.5)
                        df1 = pd.DataFrame([' P-value (equal to 0.5) = %.5f' % pval])
                        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                        if tbinom is not None:
                            defect_summary_dict[img_type][names[i]]['tbinom'].extend(tbinom[i-2])

                            x = np.array((np.nansum(tbinom[i-2]), np.nansum(ob)))
                            n = np.array((len(tbinom[i-2]) - np.sum(np.isnan(tbinom[i-2])), len(ob) - np.sum(np.isnan(ob))))
                            stat, pval = proportions_ztest(x, n, alternative='two-sided')

                            # write to csv file
                            df1 = pd.DataFrame([' P-value (equal to the truth) = %.5f' %pval])
                            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
                    else:

                        # write to csv file
                        df1 = pd.DataFrame([names[i] + " not found"])
                        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

            iou, dict_iou = box_iou(Pcenters, Prect, Parea, Tcenters, Trect, Tarea, alim = alim_px, saveres=saveres, model_name=label, loss_type=loss)
            return_values["iou_3"] = iou.iloc[1]['Recall']
            df1 = pd.DataFrame([[""],[file], ["Box Iou: " + img_type]])
            df1.to_csv(file_path_csv, mode='a', header=False, index = False)
            iou.to_csv(file_path_csv, mode='a')
            Iou_report.append(iou)

            #add iou items to total list so can get overall iou
            for truth_pred in ["TP", "PT"]:
                for key, value in dict_iou[truth_pred].items():
                    if key in total_dict_iou[truth_pred]:
                       total_dict_iou[truth_pred][key].extend(list(value))
                    else:
                       total_dict_iou[truth_pred][key] = list(value)


        df1 = pd.DataFrame([[""], ["Overall as One Image"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)

        for img_type in ['truth', 'pred']:
            for i in range(2,5):
                #only run if defect exists in pred
                if names[i] in defect_summary_dict[img_type]:
                    df1 = pd.DataFrame([[""], [img_type]])
                    df1.to_csv(file_path_csv, mode='a', header=False, index = False)
                    #take the ob for every image, which is stored in a list
                    ob = np.asarray(defect_summary_dict[img_type][names[i]]['ob'])
                    tbinom = np.asarray(defect_summary_dict[img_type][names[i]]['tbinom'])
                    nitems = defect_summary_dict[img_type][names[i]]['nitems']

                    df1 = pd.DataFrame([[""], ['Grain boundary for %s ' %names[i] + '= %.2f' %(np.nanmean(ob)) + '(%i)' %np.nansum(ob)]])
                    df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                    df1 = pd.DataFrame(['There are %i' %nitems + " " + names[i] + ' defects'])
                    df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                    stat, pval = proportions_ztest(np.nansum(ob), len(ob) - np.sum(np.isnan(ob)), 0.5)
                    df1 = pd.DataFrame([' P-value (equal to 0.5) = %.5f' % pval])
                    df1.to_csv(file_path_csv, mode='a', header=False, index = False)

                    if len(tbinom) > 0:
                        current_tbinom = np.asarray(tbinom)
                        x = np.array((np.nansum(current_tbinom), np.nansum(ob)))
                        n = np.array((len(current_tbinom) - np.sum(np.isnan(current_tbinom)), len(ob) - np.sum(np.isnan(ob))))

                        stat, pval = proportions_ztest(x, n, alternative='two-sided')
                        # write to csv file
                        df1 = pd.DataFrame([' P-value (equal to the truth) = %.5f' %pval])
                        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
                else:

                    # write to csv file
                    df1 = pd.DataFrame([names[i] + " not found"])
                    df1.to_csv(file_path_csv, mode='a', header=False, index = False)


        # write to csv file
        df1 = pd.DataFrame([[""],['Box Iou as Big Image']])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        IoU_as_image = np.zeros((len(total_dict_iou['TP']), 2))

        for j in range(2):
            x = ["TP", "PT"][j]
            for key, value in total_dict_iou[x].items():
                IoU_as_image[int(key),j] = np.nanmean(np.array(value))

        df3 = pd.DataFrame(IoU_as_image)
        df3.index = iou.index
        df3.to_csv(file_path_csv, mode='a')

        #accuarcy as average
        df1 = pd.DataFrame([[""], ["Average Cell by Cell"]])
        df1.to_csv(file_path_csv, mode='a', header=False, index = False)
        df1 = pd.concat(Iou_report).groupby(level=0).mean()
        df1 = df1.reindex(iou.index)
        df1.to_csv(file_path_csv, mode='a')

        return return_values
        


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=3, nargs='?',
                        help='Random seed for reproducibility')
    parser.add_argument('--N-epochs', default=10, nargs='?',
                        help='Number of epochs to train')
    parser.add_argument('--folder', default='./', nargs='?',
                        help='The folder enclosing the image(s)')
    parser.add_argument('--filename', default='C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif',
                        help='The image file on which to train')
    parser.add_argument('--logging-level', default='INFO', nargs='?',
                        help='How verbose should the information be')
    return parser

def _run_cli(): # pragma: no cover
    args = _make_parser().parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=args.logging_level)
    trainer = Trainer(**vars(args))
    trainer()

if __name__ == "__main__":

    _run_cli()