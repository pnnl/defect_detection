import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import cv2
from ..defect_locate import createLineIterator, rectContains
from .. import vis
from ..modified_ripley import ModifiedRipleysK
import warnings


def multinoulli(ps):
    """Choose a class from a multinoulli distribution with probabilities in last dimension."""
    shape_ps = list(ps.shape)
    n_classes = shape_ps[-1]
    shape_ps[-1] = 1
    cum_ps = ps.cumsum(axis=-1) - ps
    choices = np.random.uniform(0.0, 1.0, size=shape_ps)
    choices = choices - cum_ps
    choices[choices < 0] = np.inf
    # reverse the last axis so we get the last time something is below, not the first
    choices = choices[:, :, ::-1]
    choices = np.argmin(choices, axis=-1) + 1
    choices = n_classes - choices
    return choices

def bernoulli(ps):
    """Choose positive or negative from bernoulli with probability in the last dimension."""
    shape_ps = ps.shape
    choices = np.random.uniform(0.0, 1.0, size=shape_ps)
    choices = choices < ps
    return choices

def ob_binary(Dcenters, Darea, markD):
    """Determine if Dcenters are on boundary and mark with a binary"""
    obtmp = np.zeros(Dcenters.shape[0])
    obtmp[:] = np.nan
    for j in range(Dcenters.shape[0]):
        if Darea[j] > 0:
            if markD[Dcenters[j,1].astype(int), Dcenters[j,0].astype(int)] > 0:
                obtmp[j] = 1
            else:
                obtmp[j] = 0
    return obtmp

class MacroProp:
    """A class for calculating macroscopic properties from a calibrated model."""

    def __init__(self, trainer, calibrator, dataset_name='image', N_trial=25, test_version=False, test_input=None, plot=False, save=False):
        self.trainer = trainer
        self.calibrator = calibrator
        self.N_trial = N_trial
        self.dataset_name = dataset_name
        self.test_version = test_version
        self.test_input = test_input
        self.plot = plot
        self.save =save

    def __call__(self):
        gb_mean = self.calculate_macroscopic_properties()
        return gb_mean

    def calculate_macroscopic_properties(self):
        """Calculate the set of macroscopic properties and visualize or save to file.
        
        :param bool plot: Whether to plot values. Default ``True``
        :param bool save: Whether to save values. Default ``True``
        """
        if self.test_version:
            len_total = 1
        else:
            len_total = len(self.trainer.X_test)

        for i_test_image in range(len_total):
            if self.test_version:
                (X_test, hypothesis_class_per_pixel, p_correct_per_pixel)  = self.test_input

            else:
                X_test = torch.from_numpy(self.trainer.X_test[i_test_image]).float().permute((2, 0, 1)).unsqueeze(0).to(self.trainer.device)
                hypothesis_class_per_pixel, p_correct_per_pixel  = self.calibrator(X_test)

            names = list(vis.labels5)
            image = (X_test.cpu().squeeze().permute((1, 2, 0))/255.0).numpy()

            N_radii = 100

            gb_mean = np.empty((self.N_trial, 3))
            gb_std = np.empty((self.N_trial, 3))
            gb_sum = np.empty((self.N_trial, 3))
            n_items = np.empty((self.N_trial, 3))
            gb_areas = np.empty((self.N_trial, 3))
            gb_areas_std = np.empty((self.N_trial, 3))
            all_multivariate_Ks = np.empty((self.N_trial, 3, 3, N_radii))
            all_multivariate_q1s = np.empty((self.N_trial, 3, 3, N_radii))
            all_multivariate_q99s = np.empty((self.N_trial, 3, 3, N_radii))
            csr_multivariate_Ks = np.empty((self.N_trial, 3, 3, N_radii))

            for i_trial in tqdm.trange(self.N_trial):
                imageC = multinoulli(p_correct_per_pixel.numpy())
                dict_ob_wrapper = {}
                all_centers = []
                all_area = []
                all_box = []
                all_rect = []
                _Dcenters = list()
                _Darea = list()
                _Drect = list()
                _Dbox = list()
                _Donbnd = list()
                _Dnames= list()
                for i_class in range(2,5):
                    locs = np.where(imageC == i_class)
                    imgray = np.zeros(imageC.shape)
                    imgray[locs] = 255
                    thresh = cv2.convertScaleAbs(imgray)
                    # Dilate to remove gaps #
                    imageD = cv2.dilate(thresh, (5,5), iterations=5)
                    # may need to refine kernel size and amount of dilation
                    # Connect defects #
                    ret, Dmarkers = cv2.connectedComponents(imageD)
                    ## find_defects
                    ## Find defects in dilated image ##
                    #Ccenters, Carea, Crect, Cbox, nitems = locate_defects(imageD, defect_label = 'clustered', showplot=showplot, saveplot=saveplot, filename = filename, path = path)
                    Dcenters = list()
                    Darea = list()
                    Drect = list()
                    Dbox = list()
                    ## Locate Center Points ##
                    fcout = cv2.findContours(imageD.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(fcout) == 2:
                        cnts = fcout[0]
                        hier = fcout[1]
                    else:
                        cnts = fcout[1]
                        hier = fcout[2]
                    coords = np.zeros((len(cnts),2))
                    msize = np.zeros(len(cnts))
                    brect = np.zeros((len(cnts),4))
                    bbox = np.zeros((len(cnts),4,2))
                    nitems = 0
                    for j,c in enumerate(cnts):
                        # compute the center of the contour
                        M = cv2.moments(c)
                        if (M["m00"] > 0) & (hier[0,j,3] == -1):
                            nitems += 1
                            coords[j,0] = M["m10"] / M["m00"]
                            coords[j,1] = M["m01"] / M["m00"]
                            msize[j] = M["m00"]
                            x,y,w,h = cv2.boundingRect(c)
                            brect[j,:] = np.array([x,y,x+w,y+h])
                            rect = cv2.minAreaRect(c)
                            box = cv2.boxPoints(rect)
                            bbox[j,:,:] = np.int0(box)
                        else:
                            coords[j, :] = (np.nan, np.nan)

                    Dcenters.append(coords)
                    Darea.append(msize)
                    Drect.append(brect)
                    Dbox.append(bbox)
                    ## on_boundary
                    markD = np.zeros(Dmarkers.shape)
                    predD = np.zeros(Dcenters[0].shape[0])
                    for j in range(Dcenters[0].shape[0]):
                        if Darea[0][j] > 0:
                            ## Plot defect with bounding box ##
                            box = Dbox[0][j,:,:].astype(int)
                            center = Dcenters[0][j,:].astype(int)
                            rect = Drect[0][j,:].astype(int)
                            offset = np.sqrt(Darea[0][j])*1.5
                            offset = max(20, offset.astype(int))
                            lowlims = np.max(np.vstack((rect[:2] - offset, [0,0])),axis=0)
                            highlims = np.min(np.vstack((rect[2:] + offset, [image.shape[1],image.shape[0]])),axis=0)
                            ## Plot predicted grain boundary lines ##
                            subimageC = imageC[lowlims[1]:highlims[1], lowlims[0]:highlims[0]]
                            subimage = image[lowlims[1]:highlims[1], lowlims[0]:highlims[0]]
                            locs = np.where(subimage == 1)
                            imgray = np.zeros(subimage.shape)
                            imgray[locs] = 255
                            thresh = cv2.convertScaleAbs(imgray)
                            ## Find cluster ID ##
                            markerC = Dmarkers[rect[1]:rect[3],rect[0]:rect[2]]
                            mlocs = np.where(markerC > 0)
                            unique, counts = np.unique(markerC[mlocs], return_counts=True)
                            markID = unique[np.argmax(counts)]
                            mlocs = tuple((x+[rect[1],rect[0]][k]) for k,x in enumerate(mlocs))
                            ## Find lines ##
                            lines = cv2.HoughLinesP(thresh.astype('uint8')[:, :, 0], 0.9, np.pi/180/8, int(np.log(offset*2)), int(np.log(offset*.5)), 1)
                            # Inputs: image, pixel distance accuracy, angle accuracy,
                            #         minimum vote threshold, minimum line length, maximum line gap
                            # Note: log worked better than sqrt, despite sqrt being more intuitive.
                            if lines is not None:
                                for l in range(lines.shape[0]):
                                    if lines.shape[2] == 2:
                                        a = np.cos(lines[l,0,1])
                                        b = np.sin(lines[l,0,1])
                                        x0 = a*lines[l,0,0]
                                        y0 = b*lines[l,0,0]
                                        x1 = int(x0 + 2*offset*(-b))
                                        y1 = int(y0 + 2*offset*(a))
                                        x2 = int(x0 - 2*offset*(-b))
                                        y2 = int(y0 - 2*offset*(a))
                                        itnums = createLineIterator(np.array([x1,y1],dtype=int), np.array([x2,y2],dtype=int), subimageC, offset=10)
                                    elif lines.shape[2] == 4:
                                        itnums = createLineIterator(lines[l,0,:2], lines[l,0,2:], subimageC, offset=10)
                                    ## Check that line crosses defect ##
                                    prop = np.mean(itnums[:,2] == 1)*100
                                    if rectContains(rect - np.tile(lowlims,2), itnums) & (prop > 0):

                                        markD[mlocs] = 1
                                        predD[j] = 1

                    #adding to make txt file easier to read with new lines per defect
                    ## Isolate defect of interest ##
                    locs = np.where(imageC == i_class)
                    if len(locs[0]) == 0:
                        _Dcenters.append(np.array([[]]))
                        _Darea.append(np.array([[]]))
                        _Drect.append(np.array([[]]))
                        _Dbox.append(np.array([[]]))
                        _Donbnd.append(np.array([[]]))
                        _Dnames.append(np.array([[]]))
                        gb_mean[i_trial, i_class - 2] = np.nan
                        gb_sum[i_trial, i_class - 2] = np.nan
                    #only return centers if names[i] is part of image
                    else:
                        dict_ob_wrapper[names[i_class]] = {}
                        imgray = np.zeros(image.shape)
                        imgray[locs] = 255
                        thresh = cv2.convertScaleAbs(imgray)
                        ## Locate Center Points ##
                        c, a, r, b, nitems = Dcenters, Darea, Drect, Dbox, nitems
                        #c, a, r, b, nitems = locate_defects(thresh, defect_label = names[i], path = path)
                        dict_ob_wrapper[names[i_class]]['nitems'] = nitems
                        ob = ob_binary(c[0], a[0], markD)
                        dict_ob_wrapper[names[i_class]]['ob'] = ob
                        _Dcenters.append(c[0])
                        _Darea.append(a[0])
                        _Drect.append(r[0])
                        _Dbox.append(b[0])
                        _Donbnd.append(ob)
                        _Dnames.append(np.array([names[i_class]]*len(b[0])))
                        ## Print statistical summaries ##
                        gb_mean[i_trial, i_class-2] = np.nanmean(ob)
                        gb_std[i_trial, i_class-2] = np.nanstd(ob)
                        gb_sum[i_trial, i_class-2] = np.nansum(ob)

                    #use 1/5 of image height as the max radius
                    locs = np.where(imageC == i_class)
                    imgray = np.zeros(imageC.shape)
                    imgray[locs] = 255
                    thresh = cv2.convertScaleAbs(imgray)
                    # Dilate to remove gaps #

                    ## Locate Center Points ##
                    fcout = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if len(fcout) == 2:
                        cnts = fcout[0]
                        hier = fcout[1]
                    else:
                        cnts = fcout[1]
                        hier = fcout[2]
                    coords = np.zeros((len(cnts),2))
                    msize = np.zeros(len(cnts))
                    brect = np.zeros((len(cnts),4))
                    bbox = np.zeros((len(cnts),4,2))
                    nitems = 0
                    for j,c in enumerate(cnts):
                        # compute the center of the contour
                        M = cv2.moments(c)
                        if (M["m00"] > 0) & (hier[0,j,3] == -1):
                            nitems += 1
                            coords[j,0] = M["m10"] / M["m00"]
                            coords[j,1] = M["m01"] / M["m00"]
                            msize[j] = M["m00"]
                            x,y,w,h = cv2.boundingRect(c)
                            brect[j,:] = np.array([x,y,x+w,y+h])
                            rect = cv2.minAreaRect(c)
                            box = cv2.boxPoints(rect)
                            bbox[j,:,:] = np.int0(box)
                        else:
                            coords[j, :] = (np.nan, np.nan)

                    all_centers.append(coords)
                    all_area.append(msize)
                    all_rect.append(brect)
                    all_box.append(bbox)
                if i_trial == 0 and self.plot:
                    plt.figure(figsize=(40, 16))
                    plt.imshow(imageC, alpha=0.5, cmap=vis.cmap5, interpolation='nearest')
                    plt.colorbar()
                    for i in range(3):
                        centers = all_centers[i]
                        areas = all_area[i]
                        idx = areas > 16
                        centers = centers[idx]
                        plt.scatter(centers[:, 0], centers[:, 1], s=1, marker='+', label=names[2 + i])
                    plt.legend(loc=1, facecolor='white', framealpha=1.0, frameon=True)
                    plt.show()
                r = torch.linspace(1, round(max(hypothesis_class_per_pixel.shape[-1], hypothesis_class_per_pixel.shape[-2])),
                                N_radii)
                #set max to the image width (shape[1]) and height (shape[0])
                x1, y1 = 0., 0.
                x2, y2 = float(hypothesis_class_per_pixel.shape[-1]), float(hypothesis_class_per_pixel.shape[-2])
                Kest = ModifiedRipleysK(x1, x2, y1, y2)
                for i in range(len(all_centers)):
                    centersi = all_centers[i].copy()
                    idx = np.all(np.isfinite(centersi), axis=1)
                    centersi = centersi[idx]
                    areai = all_area[i].copy()
                    areai = areai[idx]
                    n_items[i_trial, i] = len(centersi)
                    gb_areas[i_trial, i] = np.mean(areai)
                    gb_areas_std[i_trial, i] = np.nanmean(areai)
                for i in range(len(all_centers)):
                    for j in range(len(all_centers)):
                        centersi = all_centers[i].copy()
                        centersi = centersi[np.all(np.isfinite(centersi), axis=1)]
                        centersj = all_centers[j].copy()
                        centersj = centersj[np.all(np.isfinite(centersj), axis=1)]
                        n = min(len(centersi), len(centersj))
                        centersi = centersi[:n]
                        centersj = centersj[:n]
                        if (len(centersi) > 1) and (len(centersj) > 1):
                            K, q1, q99 = Kest(torch.from_numpy(centersi),
                                    torch.from_numpy(centersj), r=r)
                        else:
                            K, q1, q99 = np.nan, np.nan, np.nan
                        all_multivariate_Ks[i_trial, i, j] = K
                        all_multivariate_q1s[i_trial, i, j] = q1
                        all_multivariate_q99s[i_trial, i, j] = q99
                        #for i_csr_trial in range(N_trials):
                        centersi = torch.stack(((x2 - x1) * torch.rand((n,)),
                                                (y2 - y1) * torch.rand((n,))),
                                            dim=-1)
                        centersj = torch.stack(((x2 - x1) * torch.rand((n,)),
                                                (y2 - y1) * torch.rand((n,))),
                                            dim=-1)
                        if (len(centersi) > 1) and (len(centersj) > 1):
                            K, _, _ = Kest(centersi, centersj, r=r)
                        else: 
                            K, _, _ = np.nan, np.nan, np.nan
                        csr_multivariate_Ks[i_trial, i, j] = K
               
            if self.plot:
                N_plot = 4
                fig, axes = plt.subplots(nrows=N_plot, sharex=True, figsize=(3, 3*N_plot/1.618))
                fig.suptitle(f'Unirradiated {os.path.basename(self.trainer.files[i_test_image])}')
                plt.sca(axes[0])
                plt.violinplot(gb_mean)
                plt.ylabel('Percent on' + "\n" + 'Boundary')
                plt.sca(axes[1])
                plt.violinplot(gb_sum)
                plt.ylabel('Number on' + "\n" + 'Boundary')
                plt.sca(axes[2])
                plt.violinplot(gb_areas)
                plt.ylabel('Area of Particle')
                plt.sca(axes[3])
                plt.violinplot(n_items)
                plt.ylabel('Number of Particles')
                names_subset = names[2:]
                plt.xticks(np.arange(len(names_subset)) + 1, names_subset)
                plt.tight_layout()
                plt.show()
            
                alpha = 0.05
                fig, axes = plt.subplots(ncols=3, nrows=3,
                                        sharey=True, sharex=True,
                                        figsize=(6.5, 6.5))
                fig.suptitle(f'{self.dataset_name} {os.path.basename(self.trainer.files[i_test_image])}')
                for i in range(3):
                    for j in range(3):
                        K = np.nanmean(all_multivariate_Ks[:, i, j, :], axis=0)
                        K_csr = np.nanmean(csr_multivariate_Ks[:, i, j, :], axis=0)
                        K_ll = K - np.quantile(all_multivariate_Ks[:, i, j, :], (alpha/2), axis=0)
                        K_ul = np.quantile(all_multivariate_Ks[:, i, j, :], 1.0-(alpha/2), axis=0) - K
                        K_csr_ll = np.quantile(csr_multivariate_Ks[:, i, j, :], (alpha/2), axis=0) - K_csr
                        K_csr_ul = np.quantile(csr_multivariate_Ks[:, i, j, :], 1.0-(alpha/2), axis=0) - K_csr
                        q1 = np.nanmean(all_multivariate_q1s[:, i, j, :], axis=0)
                        q99 = np.nanmean(all_multivariate_q99s[:, i, j, :], axis=0)
                        plt.sca(axes[i, j])
                        with warnings.catch_warnings(record=True):
                            warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
                            warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')
                            warnings.filterwarnings(action='ignore', message='invalid value encountered in add')
                            delta_K = np.subtract(K, K_csr)
                        plt.errorbar(r, delta_K, yerr=np.stack((K_ll, K_ul), axis=0),
                                    errorevery=len(r)//10, label='Sample')
                        plt.fill_between(r, K_csr_ll, K_csr_ul, alpha=0.2, label=rf'CSR ($\alpha$={alpha:.2f})',
                                        facecolor='#616265')
                        plt.title(f'{names[2:][i]} vs {names[2:][j]}')
                        plt.yscale('symlog')
                        if i == 0 and j == 0:
                            plt.legend()
                        if i == 2:
                            plt.xlabel('Radius')
                plt.tight_layout()
                plt.show()

            if self.save:
                for object_to_save, filename_prefix in zip([gb_mean, gb_sum, gb_areas, n_items,
                                                            all_multivariate_Ks, csr_multivariate_Ks, r.numpy()],
                                                        ['onboundary_percent', 'onboundary_number',
                                                            'defect_area', 'number_defects',
                                                            'ripleys_K', 'ripleys_K_csr', 'radii']):
                    if self.test_version:
                        name_file  = "test_only"
                    else:
                        name_file = os.path.basename(self.trainer.files[i_test_image])
                    filename = f'{filename_prefix}_{self.dataset_name}_image{name_file}.npy'
                    np.save(filename, object_to_save)
        return gb_mean            