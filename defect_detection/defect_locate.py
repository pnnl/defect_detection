"""Functions to locate and characterize defects."""


from .ttp_imports import *
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


pnnl_colors = {'black': '#000000',
               'white': '#ffffff',
               'orange': '#D77600',
               'lightblue': '#748EB3',
               'blue': '#3b73af',
               'darkblue': '#00338E',
               'gray': '#606060',
               'graydark': '#707276',
               'lightgray': '#eeeeee',
               'lightgreen': '#68AE8C',
               'green': '#719500',
               'purple': '#9682B3',
               'teal': '#66B6CD',
               'red': '#BE0F34',
               'copper': '#D77600',
               'silver': '#616265',
               'bronze': '#A63F1E',
               'gold': '#F4AA00',
               'platinum': '#B3B3B3',
               'onyx': '#191C1F',
               'emerald': '#007836',
               'sapphire': '#00338E',
               'ruby': '#BE0F34',
               'mercury': '#7E9AA9',
               'topaz': '#0081AB',
               'amethyst': '#502D7F',
               'garnet': '#870150',
               'emslgreen': '#719500'}
_c = pnnl_colors
colors5 = [_c['silver'], _c['copper'],  _c['emerald'], _c['garnet'], _c['sapphire']]
cmap5, norm5 = from_levels_and_colors(np.arange(6) - 0.5, colors5)
labels5 = ['Grain', 'Boundary', 'Void', 'Impurity', 'Precipitate']
cmap6, norm6 = from_levels_and_colors(np.arange(
    7) - 0.5, [_c['silver'], _c['copper'],  _c['emerald'], _c['garnet'], _c['sapphire'], _c['ruby']])
labels6 = [_x for _x in labels5]
labels6.append('Edge')

cmaps = {5: (cmap5, norm5, labels5), 6: (cmap6, norm6, labels6)}

##############################################
## Extracting pixels along a grain boundary ##

def rectContains(rect,pts):
    logic = (rect[0] <= pts[:,0])&(rect[2] >= pts[:,0])&(rect[1] <= pts[:,1])&(rect[3] >= pts[:,1])
    return np.any(logic)


def createLineIterator(P1, P2, img, offset=0):
    """
    Produces an array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """

    # extend line past endpoints
    if (P2[0] - P1[0]) != 0:
        m = (P2[1] - P1[1]) / (P2[0] - P1[0])
        b = P1[1] - m*P1[0]
        P1[0] = P1[0] - offset
        P2[0] = P2[0] + offset
        P1[1] = b + m*P1[0]
        P2[1] = b + m*P2[0]
    else:
        P1[1] = P1[1] - offset
        P2[1] = P2[1] + offset


    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            #changing np.int to int to deal with DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself.
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer



###########################################
### Functions for on-boundary detection ###


### Merge clustered defects ###
def defect_clusters(imageC):

    # Group as defect or not #
    locs = np.where(imageC == 1)

    imgray = np.zeros(imageC.shape)
    imgray[locs] = 255
    thresh = cv2.convertScaleAbs(imgray)


    # Dilate to remove gaps #
    imageD = cv2.dilate(thresh, (5,5), iterations=5)
    # may need to refine kernel size and amount of dilation

    # Connect defects #
    ret, markers = cv2.connectedComponents(imageD)

    return thresh, imageD, markers






### Find defects in image ###
def locate_defects(imageD, defect_label = '', showplot=False, saveplot=True, filename = "", path=''):
    lst_locate_defects = []

    path_csv = os.path.join(path,  "extra_csv_info.csv")

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


    # write to txt file
    with open(path_csv, "a") as f:
        print('There are %i' %nitems, '%s defects' %defect_label, file=f)

    Dcenters.append(coords)
    Darea.append(msize)
    Drect.append(brect)
    Dbox.append(bbox)

    return Dcenters, Darea, Drect, Dbox, nitems





## Detect if cluster is on-boundary ##
def on_boundary(Dmarkers, Dcenters, Darea, Drect, Dbox, image, imageC, imageD, showplot=False, saveplot=False, filename = "",  path=''):

    path_csv = os.path.join(path,  "extra_csv_info.csv")
    markD = np.zeros(Dmarkers.shape)
    predD = np.zeros(Dcenters[0].shape[0])
    t0 = time()
    
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


            if showplot:
                plt.figure(figsize=(15,10))
                plt.subplot(1,3,1)
                plt.imshow(image)
                plt.xlim(lowlims[0],highlims[0])
                plt.ylim(highlims[1],lowlims[1])

                plt.plot(np.append(box[:,0],box[0,0]), np.append(box[:,1],box[0,1]), color="red", linewidth=2)

                plt.plot(rect[[0,2]], rect[[1,1]], color="green", linewidth=2)
                plt.plot(rect[[0,2]], rect[[3,3]], color="green", linewidth=2)
                plt.plot(rect[[0,0]], rect[[1,3]], color="green", linewidth=2)
                plt.plot(rect[[2,2]], rect[[1,3]], color="green", linewidth=2)


            ## Plot predicted grain boundary lines ##
            subimageC = imageC[lowlims[1]:highlims[1], lowlims[0]:highlims[0]]
            subimage = image[lowlims[1]:highlims[1], lowlims[0]:highlims[0]]
            locs = np.where(subimage == 1)

            imgray = np.zeros(subimage.shape)
            imgray[locs] = 255
            thresh = cv2.convertScaleAbs(imgray)

            if showplot:
                plt.subplot(1,3,2)
                plt.imshow(thresh, cmap=plt.get_cmap('gray'))
                plt.xlim(0, highlims[0]-lowlims[0])
                plt.ylim(highlims[1]-lowlims[1], 0)
                plt.title('Boundary')


            ## Find cluster ID ##
            markerC = Dmarkers[rect[1]:rect[3],rect[0]:rect[2]]
            mlocs = np.where(markerC > 0)
            unique, counts = np.unique(markerC[mlocs], return_counts=True)
            markID = unique[np.argmax(counts)]
            mlocs = tuple((x+[rect[1],rect[0]][k]) for k,x in enumerate(mlocs))


            ## Find lines ##
            lines = cv2.HoughLinesP(thresh, 0.9, np.pi/180/8, int(np.log(offset*2)), int(np.log(offset*.5)), 1)

            # Inputs: image, pixel distance accuracy, angle accuracy,
            #         minimum vote threshold, minimum line length, maximum line gap
            # Note: log worked better than sqrt, despite sqrt being more intuitive.
            
            if lines is not None:

                if showplot:
                    plt.subplot(1,3,3)
                    plt.imshow(imageD[lowlims[1]:highlims[1], lowlims[0]:highlims[0]])
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
                        if showplot:
                            plt.plot(itnums[:,0], itnums[:,1], color="violet", linewidth=2)

                        markD[mlocs] = 1
                        predD[j] = 1


                    else:
                        if showplot:
                            plt.plot(itnums[:,0], itnums[:,1], color="yellow", linewidth=2)

                if showplot:
                    plt.xlim(0, highlims[0]-lowlims[0])
                    plt.ylim(highlims[1]-lowlims[1], 0)
                    plt.title("Cluster")

            if showplot:
                plt.show()
    
    with open(path_csv, "a") as f:
        print('done in %0.3fs.' % (time() - t0), file=f)


    return markD, predD




###################################################
### Wrapper Functions for on-boundary detection ###

def ob_binary(Dcenters, Darea, markD):
    obtmp = np.zeros(Dcenters.shape[0])
    obtmp[:] = np.nan

    for j in range(Dcenters.shape[0]):
        if Darea[j] > 0:
            if markD[Dcenters[j,1].astype(int), Dcenters[j,0].astype(int)] > 0:
                obtmp[j] = 1
            else:
                obtmp[j] = 0

    return obtmp



def ob_wrapper(image, markD, tbinom=None, names=['Grain','Boundary','Void','Impurity','Precipitate'], showplot=False, saveplot=False, filename = "",  path='', model_name='', loss_type='', nclass=5, alim = 16):
    dict_ob_wrapper = {}
    path_csv  = os.path.join(path,  "extra_csv_info.csv")

    Dcenters = list()
    Darea = list()
    Drect = list()
    Dbox = list()
    Donbnd = list()
    Dnames= list()
    Dfilenames= list()

    cmap, norm, labels = cmaps[nclass]
    image_cmap = cmap(norm(image))

    plt.figure(figsize=(14,17))
    plt.subplot(1,4,1)
    plt.imshow(image_cmap)
    plt.title('Data')

    for i in range(2,5):
        #adding to make txt file easier to read with new lines per defect
        with open(path_csv, "a") as f:
            print('', file=f)


        ## Isolate defect of interest ##
        locs = np.where(image == i)

        if len(locs[0]) == 0:

            Dcenters.append([])
            Darea.append([])
            Drect.append([])
            Dbox.append([])
            Donbnd.append([])
            Dnames.append([])
            Dfilenames.append([])

            with open(path_csv, "a") as f:
                print(names[i] + ' is not present in image', file=f)

        #only return centers if names[i] is part of image
        else:
            dict_ob_wrapper[names[i]] = {}

            imgray = np.zeros(image.shape)
            imgray[locs] = 255
            thresh = cv2.convertScaleAbs(imgray)

            ## Locate Center Points ##

            c, a, r, b, nitems = locate_defects(thresh, defect_label = names[i], path = path)
            dict_ob_wrapper[names[i]]['nitems'] = nitems

            ob = ob_binary(c[0], a[0], markD)
            dict_ob_wrapper[names[i]]['ob'] = ob
            dict_ob_wrapper[names[i]]['tbinom'] = tbinom
            Dcenters.append(c[0])
            Darea.append(a[0])
            Drect.append(r[0])
            Dbox.append(b[0])
            Donbnd.append(ob)
            Dnames.append(np.array([names[i]]*len(b[0])))
            Dfilenames.append(np.array([filename]*len(b[0])))


            if showplot | saveplot:
                plt.subplot(1,4,i)
                plt.imshow(thresh, cmap = plt.get_cmap('gray'))
                plt.title(names[i])
                x, y = np.split(c[0], 2, axis=1)
                cex = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(
                ), "area": a[0]})
                cex = cex.loc[cex['area'] > alim] 
                plt.scatter(cex['x'], cex['y'], s =3,  color="green", alpha = .5)
            


            ## Print statistical summaries ##
            with open(path_csv, "a") as f:
                print('Grain boundary for %s ' %names[i], '= %.2f' %(np.nanmean(ob)), '(%i)' %np.nansum(ob), file=f)

            stat, pval = proportions_ztest(np.nansum(ob), len(ob) - np.sum(np.isnan(ob)), 0.5)


            with open(path_csv, "a") as f:
                print(' P-value (equal to 0.5) = %.5f' %pval, file=f)

            if tbinom is not None:

                x = np.array((np.nansum(tbinom[i-2]), np.nansum(ob)))
                n = np.array((len(tbinom[i-2]) - np.sum(np.isnan(tbinom[i-2])), len(ob) - np.sum(np.isnan(ob))))

                stat, pval = proportions_ztest(x, n, alternative='two-sided')
                with open(path_csv, "a") as f:
                    print(' P-value (equal to the truth) = %.5f' %pval, file=f)

    if showplot:
        plt.show()

    elif saveplot:
        plt.savefig(os.path.join(path, "{}_defect_locate_".format(filename.split('.')[0])  + model_name + ".png"), bbox_inches="tight", dpi=1000)

        plt.savefig(os.path.join(path, "{}_ob_wrapper.png".format(filename.split('.')[0])), bbox_inches="tight", dpi=1000)
        plt.close()
    return Dcenters, Darea, Drect, Dbox, Donbnd, Dnames, Dfilenames, dict_ob_wrapper



def defect_summary(image, tbinom=None, names=['Grain','Boundary','Void','Impurity','Precipitate'], showplot=False, saveplot=False, saveres=False, filename ="copy.png", path='', model_name='', loss_type='', nclass=5,  alim=16):

    ## Defects must be categories greater than 1 ##
    imageC = 1*(np.isin(image, [2, 3, 4]))
    thresh, imageD, markers = defect_clusters(imageC)

    ## Find defects in dilated image ##
    Ccenters, Carea, Crect, Cbox, nitems = locate_defects(imageD, defect_label = 'clustered', showplot=showplot, saveplot=saveplot, filename = filename, path = path)
    mark, pred = on_boundary(markers, Ccenters, Carea, Crect, Cbox, image, imageC, imageD, showplot=showplot, saveplot=saveplot, filename = filename,  path=path)
    centers, area, rect, box, onbnd,  names_list, filenames, dict_ob_wrapper = ob_wrapper(
        image, mark, tbinom=tbinom, names=names, showplot=showplot, saveplot=saveplot, filename=filename, path=path, model_name=model_name, loss_type='', nclass=nclass,  alim=alim)

    ## Save results for R testing ##
    all_df = []
    for i,c in enumerate(centers):
        if len(area[i]) >0:
            x, y = np.split(c, 2, axis=1)
            cex = pd.DataFrame({'x': x.flatten(), 'y': y.flatten(
            ), "area": area[i], "on_boundary": onbnd[i], "names": names_list[i], "filenames": filenames[i]})
            all_df.append(cex)
            
    full_df = pd.concat(all_df)
    full_df = full_df[full_df["on_boundary"].notna()]
    return centers, area, rect, box, onbnd, nitems, full_df, dict_ob_wrapper





#######################################
### Bounding Box Performance Metric ###

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



############################################################
### Wrapper Function for Bounding Box Performance Metric ###

def box_iou(Pcenters, Prect, Parea, Tcenters, Trect, Tarea, names=['Grain','Boundary','Void','Impurity','Precipitate'], alim=16, saveres=False, path='', model_name='', loss_type='', nclass=5):

    IoU = np.zeros((len(Trect), 2))

    dict_iou = {'TP': {}, 'PT': {}}
    for i in range(len(Trect)):
        ## True vs. Prediction ##
        if len(Pcenters[i]) != 0 :
            Tcenters[i] = Tcenters[i][Tarea[i] > alim]
            Pcenters[i] = Pcenters[i][Parea[i] > alim]

            Trect[i] = Trect[i][Tarea[i] > alim]
            Prect[i] = Prect[i][Parea[i] > alim]
        else:
            pass


        if len(Pcenters[i]) != 0 :
            d = euclidean_distances(Tcenters[i], Pcenters[i])
            dmatch = np.argmin(d, axis=1)

            TPiou = np.zeros(len(dmatch))
            TPiou[:] = np.nan
            for j,dj in enumerate(dmatch):

                rect = Trect[i][j,:]
                prect = Prect[i][dj,:]

                TPiou[j] = bb_iou(rect,prect)

            dict_iou["TP"][str(i)] = TPiou
            IoU[i,0] = np.nanmean(TPiou)

            ## Prediction vs. True ##
            dmatch = np.argmin(d, axis=0)

            PTiou = np.zeros(len(dmatch))
            PTiou[:] = np.nan
            for j,dj in enumerate(dmatch):

                rect = Prect[i][j,:]
                prect = Trect[i][dj,:]

                PTiou[j] = bb_iou(rect,prect)

            dict_iou["PT"][str(i)] = PTiou
            IoU[i,1] = np.nanmean(PTiou)
        else:
            # zero recall if no predicted centers are found
            TPiou = np.zeros(len(Tcenters[i]))
            dict_iou["TP"][str(i)] = TPiou
            IoU[i,0] = 0

            # percision can't be calculated.
            PTiou = np.zeros(1)
            PTiou[:] = np.nan
            dict_iou["PT"][str(i)] = PTiou
            IoU[i,1] = np.nan

    IoUdf = pd.DataFrame(IoU, index=names[2:], columns=['Recall','Precision'])


    if saveres:
        BoxIoU = IoUdf.mean().values.tolist()
        BoxIoU.append(IoUdf.mean().mean())
        BoxIoU = pd.DataFrame(BoxIoU, index=['BoxR IoU','BoxP IoU','BoxM IoU'], columns=[model_name])

        BoxIoU.round(3).to_csv(os.path.join(path, model_name + '_' + loss_type + str(nclass) + "_BoxIoU.csv"))
        IoUdf.round(3).to_csv(os.path.join(path, model_name + '_' + loss_type + str(nclass) + "_BoxIoU_ind.csv"))


    return IoUdf, dict_iou
