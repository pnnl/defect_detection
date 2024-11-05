"""Functions to clean and prepare the SEM image."""

import cv2
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_Y(filepath, nclass=5, showplot=False, saveplot=False, edge=None):
    """Read in categories from a tif file.
    """
    
    logging.info(filepath)
    Yold = cv2.imread(filepath)
    logging.info(Yold)
    if edge is not None:
        Y = Yold[:edge,:,:]
    else:
        Y = Yold
    if showplot:# pragma: no cover
        plt.figure(figsize=(10,10))
        plt.imshow(Y[:,:,[2,1,0]])
        plt.ioff()
        plt.close('all')

    if saveplot:  # pragma: no cover
        plt.savefig(filepath + '_labelled.png', bbox_inches="tight", dpi=1000)
        plt.ioff()
        plt.close('all')
    else:  # pragma: no cover
        plt.show()
        plt.ioff()
        plt.close('all')
    color_codes = np.unique(Y[np.where(np.std(Y, axis=2) != 0)], axis=0)
    Ycat = np.zeros(Y.shape[:2], dtype=int)
    if nclass == 5:
        if color_codes.shape[0] != 4:
            ## Void on boundary ##
            tmp = np.absolute(Y - np.array([0,185,16]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 2
            ## Impurity on boundary ##
            tmp = np.absolute(Y - np.array([0,181,185]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 3
            ## Precipitate on boundary ##
            tmp = np.absolute(Y - np.array([139,0,23]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 4
        # Transform the response data into 5 categories ##
        # 0 = (:,:,:)       = grain (grey)
        # 1 = (0, 0, 255)   = grain boundary (red)
        # 2 = (0, 255, 0)   = void (green)
        # 3 = (0, 255, 255) = impurity (yellow)
        # 4 = (255, 0, 0)   = precipitate (blue)
        ## Grains ##
        names = ['Grain','Boundary','Void','Impurity','Precipitate']
        ## Grain Boundary ##
        tmp = np.absolute(Y - np.array([0,0,255]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 1
        ## Void ##
        tmp = np.absolute(Y - np.array([0,255,0]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 2
        ## Impurity ##
        tmp = np.absolute(Y - np.array([0,255,255]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 3
        ## Precipitate ##
        tmp = np.absolute(Y - np.array([255,0,0]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 4
    elif nclass == 6:
        if color_codes.shape[0] > 5:
            ## Void on boundary ##
            tmp = np.absolute(Y - np.array([0,185,16]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 2
            ## Impurity on boundary ##
            tmp = np.absolute(Y - np.array([0,181,185]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 3
            ## Precipitate on boundary ##
            tmp = np.absolute(Y - np.array([139,0,23]))
            tmpl = np.where(np.sum(tmp, axis=2) == 0)
            Ycat[tmpl[0],tmpl[1]] = 4
        # Transform the response data into 6 categories ##
        # 0 = (:,:,:)       = grain (grey)
        # 1 = (0, 0, 255)   = grain boundary (red)
        # 2 = (0, 255, 0)   = void (green)
        # 3 = (0, 255, 255) = impurity (yellow)
        # 4 = (255, 0, 0)   = precipitate (blue)
        # 5 = (255, 0, 255) = edge (purple)
        ## Grains ##
        names = ['Grain','Boundary','Void','Impurity','Precipitate', 'Edge']
        ## Grain Boundary ##
        tmp = np.absolute(Y - np.array([0,0,255]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 1
        ## Void ##
        tmp = np.absolute(Y - np.array([0,255,0]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 2
        ## Impurity ##
        tmp = np.absolute(Y - np.array([0,255,255]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 3
        ## Precipitate ##
        tmp = np.absolute(Y - np.array([255,0,0]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 4
        ## Edge ##
        tmp = np.absolute(Y - np.array([255,0,255]))
        tmpl = np.where(np.sum(tmp, axis=2) == 0)
        Ycat[tmpl[0],tmpl[1]] = 5
    return Ycat, names



###############################################
## Importing image and prepping for analysis ##

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def image_prep(folder, filename, nclass=5, showplot=False, saveplot=False, X_only=False, dpi=1000, dummy_ycat = False):
    X_filename = os.path.join(folder, filename)
    if not os.path.isfile(X_filename):
        logging.warning(f"No file found at {X_filename}")
        if X_only:
            return None
        else:
            return None, None, None
    # Read in data
    Xold = cv2.imread(X_filename)

    # Remove boundary with image acquisition information
    
    grey = rgb2gray(Xold)

    thresh = 1*(grey > 0)
    edge_lines = np.where(np.std(grey, axis=1) == 0)
    if len(edge_lines[0]) > 0:
        edge = edge_lines[0][0]
        if edge < thresh.shape[0]*.5:
            edge = thresh.shape[0]

        X = Xold[:edge,:,:]
    else:
        X = Xold
    if saveplot:# pragma: no cover
        plt.savefig(filename + '_data.png', bbox_inches="tight", dpi=dpi)
        plt.ioff()
        plt.close('all')

    # Show/save images
    if showplot: # pragma: no cover
        plt.figure(figsize=(10,10))
        plt.imshow(X)
       
    if X_only:
        return X
    else:
        if dummy_ycat == False:
            Ycat, names = read_Y(os.path.join(folder,  'Labeled ' + filename),
                                 nclass=nclass, edge=edge, showplot=showplot,
                                 saveplot=saveplot)
        else:

            if nclass == 5:
                names = ['Grain','Boundary','Void','Impurity','Precipitate']
                Ycat = np.random.randint(1, 2, ( X.shape[0], X.shape[1]), dtype=int)
                Ycat[0:20, 0:20] = 3
                Ycat[20:50, 10:40] = 2
                Ycat[60:80, 10:40] = 4
                Ycat[40, 42] = 1
            else:
                names = ['Grain','Boundary','Void','Impurity','Precipitate', 'Edge']
                Ycat = np.random.randint(1, 2, ( X.shape[0], X.shape[1]), dtype=int)
                Ycat[0:20, 0:20] = 3
                Ycat[20:50, 10:40] = 2
                Ycat[60:80, 10:40] = 4
                Ycat[40, 42] = 1
                Ycat[80:100, 40:80] = 5
        return X, Ycat, names







##############################################
## Splitting and prepping for deep learning ##


def to_categorical(y, nclass=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if nclass is None:
        nclass = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nclass), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (nclass,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



## Creating a test/train split ##

def image_split(X, filename, cv_split=4, subpix=128, showplot=False, saveplot=False, dpi=1000):

    strat_n = [int(i/subpix) for i in X.shape[:2]]
    cvi = int(strat_n[1]/4)
    xrange = [(cv_split-1)*cvi*subpix, cvi*subpix-1]

    if showplot: # pragma: no cover
        plt.figure(figsize=(10,10))
        plt.imshow(X)
        plt.axis('off')
        rect = Rectangle(((xrange[0]+1), 1), (xrange[1]-1), (X.shape[0]-2), fill=None, alpha=1, edgecolor='magenta', linewidth=2)
        currentAxis = plt.gca()
        currentAxis.add_patch(rect)


    if saveplot: # pragma: no cover 
        plt.savefig(filename + '_image_' + str(cv_split) + '.png', bbox_inches="tight", dpi=dpi)
        plt.ioff()
        plt.close('all')
    elif showplot: # pragma: no cover 
        plt.show()
        plt.ioff()
        plt.close('all')
    else:  # pragma: no cover
        plt.ioff()
        plt.close('all')
        pass

    xrange[1] = cv_split*cvi*subpix

    return strat_n, xrange




## Segregating the input and output image for training ##

def image_augment(X, Ycat, strat_n, nclass=None, cv_split=4, mult=1, subpix=128, seed=1984):

    if nclass is None:
        nclass = len(np.unique(Ycat))

    Xa = np.zeros((np.prod(strat_n),subpix*mult,subpix*mult,3))
    Ya = np.zeros((np.prod(strat_n),subpix*mult,subpix*mult))
    Xa_index = []
    # Subset the image
    for i in range(strat_n[0]):
        for j in range(strat_n[1]):
            Xa[j+i*strat_n[1],:,:,:] = X[(i*subpix*mult):((i+1)*subpix*mult), (j*subpix*mult):((j+1)*subpix*mult),:]
            Ya[j+i*strat_n[1],:,:] = Ycat[(i*subpix*mult):((i+1)*subpix*mult), (j*subpix*mult):((j+1)*subpix*mult)]
            Xa_index.append(j+i*strat_n[1])

    M = Xa.shape[0]
    Ya = np.reshape(to_categorical(Ya, nclass=nclass), Xa.shape[:3]+(nclass,))

    #------------------------------------------------------#
    #--- Determine sample sizes and shuffle the indices ---#

    rangex = np.arange((cv_split-1)*int(strat_n[1]/4), cv_split*int(strat_n[1]/4))
    rangey = np.arange(0, np.prod(strat_n), strat_n[1])
    test_n = np.sort(np.add.outer(rangex, rangey).ravel()).tolist()
    test_n_index = list(set(test_n))
    val_n = list(set(np.arange(Xa.shape[0])) - set(test_n))

    # percentages used = 0.6, 0.15, 0.25
    div = np.array([.8, .20])
    
    n = len(val_n)*div
    n = n.astype(int)
    n[0] = n[0] + len(val_n) - sum(n)


    # Could change to stratified sampling #
    np.random.seed(seed)
    np.random.shuffle(val_n)


    #-----------------------------------------------------------#
    #--- Divide the data into training, test, and validation ---#

    binary = False
    X_train, X_val = Xa[val_n[:n[0]],:,:,:], Xa[val_n[n[0]:sum(n)],:,:,:]
    Y_train, Y_val = Ya[val_n[:n[0]],:,:,:], Ya[val_n[n[0]:sum(n)],:,:,:]

    M = X_train.shape[0]

    ## Augment Training Data
    # Subset, flip, rotate, and adjust the contrast in order to augment the data sample size.

    ## Flip the image along 3 axes ##
    new_shape = tuple(
        map(sum, zip(X_train.shape, (X_train.shape[0]*(2**3 - 2), 0, 0, 0))))
    X_train = np.append(X_train, np.zeros(new_shape), axis=0)
    Y_train  = np.append(Y_train, np.zeros(new_shape[:3] + (nclass,)), axis=0)



    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m in range(M):

                    tmp = X_train[m,:,:,:]
                    tmpy = Y_train[m,:,:,:]

                    if i == 1:
                        tmp = np.transpose( tmp, (1,0,2) )
                        tmpy = np.transpose( tmpy, (1,0,2) )
                    if j == 1:
                        tmp = np.flip( tmp, 0 )
                        tmpy = np.flip( tmpy, 0 )
                    if k == 1:
                        tmp = np.flip( tmp, 1 )
                        tmpy = np.flip( tmpy, 1 )

                    if (i==1) | (j==1) | (k==1):
                        up_loc = m + M*(i*4 + j*2 + k)
                        X_train[up_loc, :, :, :] = tmp
                        Y_train[up_loc, :, :, :] = tmpy

    # Could change to stratified sampling #
    np.random.seed(seed)
    train_n = list(set(np.arange(X_train.shape[0])))
    np.random.shuffle(train_n)

    #-----------------------------------------------------------#
    #--- Divide the data into training, test, and validation ---#

    binary = False
    X_train_shuffled = X_train[train_n,:,:,:]
    Y_train_shuffled = Y_train[train_n, :, :, :]
   
    logging.info("only trained augmented, correctly shuffled")
    return X_train_shuffled, X_val, Y_train_shuffled, Y_val