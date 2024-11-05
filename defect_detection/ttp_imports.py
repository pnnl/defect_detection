############################################################################################
# Imports necessary for TTP image segmentation project.
############################################################################################


## Basic modules ##
from __future__ import print_function
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import math
import re
import pickle
import itertools as it
from time import time
import scipy as scipy
from collections import OrderedDict


## Plotting and image modules ##
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from PIL import Image


## Statistics modules ##
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportions_ztest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances



## Deep Learning modules ##
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101, fcn_resnet50, deeplabv3_resnet50
import torch.utils.data as data
from .model.unet_baseline import *
from .model.segnet_baseline import SegNet
