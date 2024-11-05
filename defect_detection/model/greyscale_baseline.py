"""Greyscale threshold predicted model."""
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    """Convert a 3 channel image to greyscale."""
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def greyscale_model(X, Ycat,
                    names=['Grain','Boundary','Void','Precipitate','Impurity'],
                    showplot=False, logscale=False):
    """Return categorization based on the greyscale values in the image."""
    X_grey = rgb2gray(X)
    if showplot: # pragma: no cover
        bins = np.linspace(0, 260, 50)
        plt.figure(figsize=(10,8))
        for i in range(0,len(names)):
            plt.hist(X_grey[np.where(Ycat == i)], bins, alpha=0.5,
                     label=names[i], ec='black')
        plt.legend(loc='upper right')
        plt.xlabel('Greyscale Value')
        if logscale:
            plt.yscale('log')
            plt.ylabel('log( Count )')
        else:
            plt.ylabel('Count')
        plt.show()
    # Predicted Categories
    base_preds = np.zeros(X_grey.shape, dtype=int)
    base_preds[np.where(X_grey == 0)] = 2
    base_preds[np.where(X_grey == 255)] = 3
    return base_preds