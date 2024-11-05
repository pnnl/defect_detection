"""Functions to assess performance of the segmentation model."""
from .ttp_imports import *

def fB_score(prec, rec, B=1.0):
    """Return the fbeta score."""
    # zero guard
    if float(B)**2 * prec + rec == 0: return 0.0
    return( (1 + float(B)**2) * ((prec*rec) / (float(B)**2*prec + rec)) )


def modified_class_report(true, pred,  names=None,  B=[0.5, 1, 2], weights='weighted', path=''):
    lst_class_report = []
    list_index = []
    path_csv  = path.replace("txt", "csv")

    tmp = confusion_matrix(true, pred)
    n = tmp.shape[0]

    # precision and recall
    prec = np.empty(n)
    rec = np.empty(n)
    fscore = np.empty((n,len(B)))
    for i in range(n):
        prec[i] = tmp[i,i]/np.sum(tmp[:,i])
        rec[i] = tmp[i,i]/np.sum(tmp[i,:])

        # fB-score
        for j in range(len(B)):
            fscore[i,j] = fB_score(prec[i], rec[i], B=B[j])

    # sample size
    unique, counts = np.unique(true, return_counts=True)
    if names != None:
        pred_counts_unique = [(i, np.sum(pred == i))
                              for i, _ in enumerate(names)]
        unique, counts = zip(*pred_counts_unique)
            
    # only weighting in not na value
    if weights == 'weighted':
        # remove the count in denominator if prec or rec is nan
        precA = np.nansum([x*counts[i] for i,x in enumerate(prec)]) / sum([count for count, p in zip(counts, prec) if not np.isnan(p)])
        recA = np.nansum([x*counts[i] for i,x in enumerate(rec)]) / sum([count for count, p in zip(counts, rec) if not np.isnan(p)])
        fscoreA = np.empty((1,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nansum([x*counts[i] for i,x in enumerate(fscore[:,j])]) / sum([count for count, p in zip(counts, fscore[:,j]) if not np.isnan(p)])
        #unique = np.append(unique, 'Weigthed Avg')
    # removing nan from sum and length
    elif weights == 'equal':
        precA = np.nansum(prec) / len([x for x in prec if not np.isnan(x)])
        recA = np.nansum(rec) / len([x for x in rec if not np.isnan(x)])
        fscoreA = np.empty((1,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nanmean(fscore[:,j])
        #unique = np.append(unique, 'Equal Avg')

    else:
        precA = [np.nansum([x*counts[i] for i,x in enumerate(prec)]) / sum([count for count, p in zip(counts, prec) if not np.isnan(p)]), \
                 np.nansum(prec) / len([x for x in prec if not np.isnan(x)])]
        recA = [np.nansum([x*counts[i] for i,x in enumerate(rec)]) / sum([count for count, p in zip(counts, rec) if not np.isnan(p)]), \
                 np.nansum(rec) / len([x for x in rec if not np.isnan(x)])]
        fscoreA = np.empty((2,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nansum([x*counts[i] for i,x in enumerate(fscore[:,j])]) / sum([count for count, p in zip(counts, fscore[:,j]) if not np.isnan(p)])
            fscoreA[1,j] = np.nanmean(fscore[:,j])
        #unique = np.append(unique, ['Weigthed Avg', 'Equal Avg'])


    # Combine output
    outp = ('             precision  recall | ' + '  '.join(['f' + str(b) +'-score' for b in B]) + ' | support\n\n')
    list_header = ["precision", "recall"]
    for b in B:
        list_header.append('f' + str(b) +'-score')
    list_header.append("support")

    for i in range(n):
        outp = outp + '%s' %str(unique[i]) + '                 %.2f' %prec[i] + '    %.2f |    ' %rec[i] + \
            '       '.join(['%.2f' %f for f in fscore[i,:]]) + '  | %i' %counts[i] + '\n'

        list_index.append(unique[i])
        list_fscore = [prec[i], rec[i]]
        for f in fscore[i,:]:
            list_fscore.append(f)
        list_fscore.append(counts[i])
        lst_class_report.append(list_fscore)

    if weights == 'weighted':
        outp = outp + 'Weighted Avg      %.2f' %precA + '    %.2f |    ' %recA + \
            '       '.join(['%.2f' %f for f in fscoreA[0,:]]) + '  | %i' %sum(counts) + '\n'

        # write to csv file
        list_index.append('Weighted Avg')
        list_weight = [recA, recA]
        for f in fscore[0,:]:
            list_weight.append(f)
        list_weight.append(sum(counts))
        lst_class_report.append(list_weight)

    elif weights == 'equal':
        outp = outp + 'Equal Avg         %.2f' %precA + '    %.2f |    ' %recA + \
            '       '.join(['%.2f' %f for f in fscoreA[0,:]]) + '  | %i' %sum(counts) + '\n'

        # write to csv file
        list_index.append('Equal Avg')
        list_equal = [precA, recA]
        for f in fscore[0,:]:
            list_equal.append(f)
        list_equal.append(sum(counts))
        lst_class_report.append(list_equal)

    else:
        outp = outp + 'Weighted Avg      %.2f' %precA[0] + '    %.2f |    ' %recA[0] + \
            '       '.join(['%.2f' %f for f in fscoreA[0,:]]) + '  | %i' %sum(counts) + '\n'

        list_index.append('Weighted Avg')
        # write to csv file
        list_weight= [ precA[0], recA[0]]
        for f in fscoreA[0,:]:
            list_weight.append(f)
        list_weight.append(sum(counts))
        lst_class_report.append(list_weight)

        outp = outp + 'Equal Avg         %.2f' %precA[1] + '    %.2f |    ' %recA[1] + \
            '       '.join(['%.2f' %f for f in fscoreA[1,:]]) + '  | %i' %sum(counts) + '\n'

        list_index.append('Equal Avg')
        # write to csv file
        list_equal =[precA[1], recA[1]]
        for f in fscoreA[1,:]:
            list_equal.append(f)
        list_equal.append(sum(counts))
        lst_class_report.append(list_equal)

    df1 = pd.DataFrame(lst_class_report)
    df1.columns = list_header
    df1.index = list_index
    return (outp, fscoreA, df1)

## Intersection over Union Functions ##

def iou_report(true, pred, B=[0.5,1,2], weights='weighted', names=None, showres=False, saveres=False, path='', model_name='', loss_type='', nclass=5):
    lst_iou_report = []
    path_csv  = path.replace("txt", "csv")

    tmp = confusion_matrix(true, pred)
    n = tmp.shape[0]

    # precision and recall
    prec = np.empty(n)
    rec = np.empty(n)
    fscore = np.empty((n,len(B)))
    for i in range(n):
        prec[i] = tmp[i,i]/np.sum(tmp[:,i])
        rec[i] = tmp[i,i]/np.sum(tmp[i,:])


        # fB-score
        for j in range(len(B)):
            fscore[i,j] = fB_score(prec[i], rec[i], B=B[j])


    # sample size
    unique, counts = np.unique(true, return_counts=True)

    if names != None:
        pred_counts_unique_true = [(names[i], np.sum(true == i))
                                   for i, _ in enumerate(names)]
        unique, counts = zip(*pred_counts_unique_true)

    counts = np.array(counts)

    # only weighting if not na value
    if weights == 'weighted':
        # remove the count in denominator if prec or rec is nan
        precA = np.nansum([x*counts[i] for i,x in enumerate(prec)]) / sum([count for count, p in zip(counts, prec) if not np.isnan(p)])
        recA = np.nansum([x*counts[i] for i,x in enumerate(rec)]) / sum([count for count, p in zip(counts, rec) if not np.isnan(p)])
        fscoreA = np.empty((1,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nansum([x*counts[i] for i,x in enumerate(fscore[:,j])]) / sum([count for count, p in zip(counts, fscore[:,j]) if not np.isnan(p)])
        #unique = np.append(unique, 'Weigthed Avg')
    # removing nan values in both the sum and the len to get average without na
    elif weights == 'equal':
        precA = np.nansum(prec) / len([x for x in prec if not np.isnan(x)])
        recA = np.nansum(rec) / len([x for x in rec if not np.isnan(x)])
        fscoreA = np.empty((1,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nanmean(fscore[:,j])
        #unique = np.append(unique, 'Equal Avg')

    else:
        # removing nan values in both the sum and the len to get average without na
        precA = [np.nansum([x*counts[i] for i,x in enumerate(prec)]) / sum([count for count, p in zip(counts, prec) if not np.isnan(p)]), \
                 np.nansum(prec) / len([x for x in prec if not np.isnan(x)])]
        recA = [np.nansum([x*counts[i] for i,x in enumerate(rec)]) / sum([count for count, p in zip(counts, rec) if not np.isnan(p)]), \
                 np.nansum(rec) / len([x for x in rec if not np.isnan(x)])]
        fscoreA = np.empty((2,len(B)))
        for j in range(len(B)):
            fscoreA[0,j] = np.nansum([x*counts[i] for i,x in enumerate(fscore[:,j])]) / sum([count for count, p in zip(counts, fscore[:,j]) if not np.isnan(p)])
            fscoreA[1,j] = np.nanmean(fscore[:,j])
        #unique = np.append(unique, ['Weigthed Avg', 'Equal Avg'])


    ## summary statistics ##
    nii = tmp.diagonal()
    nij = np.sum(tmp, axis=1)

    pacc = np.sum(nii)/np.sum(counts)
    macc = np.sum(nii/counts)/n
    miou = np.sum(nii/(counts + np.sum(tmp, axis=0) - nii))/n
    wiou = np.sum((nii*counts)/(counts + np.sum(tmp, axis=0) - nii))/np.sum(counts)

    if len(B) > 1:
        # get average with differently weighted grain scores for F-D score
        # removing na values
        dfscores_list = np.insert(fscore[1:,len(B)-1], 0, fscore[0,0])
        wdfs = np.nansum([x*counts[i] for i,x in enumerate(dfscores_list)])/np.sum([count for count, p in zip(counts, dfscores_list) if not np.isnan(p)])
        mdfs = np.nansum(dfscores_list)/len([f for f in dfscores_list if not np.isnan(f)])


    # Create DataFrames
    #prec = np.append(prec, precA)
    #rec = np.append(rec, recA)
    #fscore = np.vstack((fscore, fscoreA))
    #counts = np.append(counts, np.repeat(sum(counts),fscoreA.shape[0]))

    # By group #
    out = {'Category':unique, 'Precision':prec, 'Recall':rec}
    onames = ['Category', 'Precision', 'Recall']
    for j,b in enumerate(B):
        out['Fscore-'+str(b)] = fscore[:,j]
        onames.append('Fscore-'+str(b))
    out['Support'] = counts
    onames.append('Support')
    outdf = pd.DataFrame(data=out)
    outdf = outdf[onames]

    # Overall #
    ssnames = ['Precision', 'Recall']
    ssvalues = np.append(precA, recA)
    for j,b in enumerate(B):
        ssnames.append('Fscore-'+str(b))
        ssvalues = np.append(ssvalues, fscoreA[:,j])
    ssnames.extend(['Fscore-D','IoU'])

    if weights == 'weighted':
        ssvalues = np.append(ssvalues, [wdfs,wiou])
    elif weights == 'equal':
        ssvalues = np.append(ssvalues, [mdfs,miou])
    else:
        ssvalues = np.append(ssvalues, [wdfs,mdfs,wiou,miou])
        ssnames = np.repeat(ssnames, 2)

    ss = {'Measure': ssnames, 'Value': ssvalues}
    ssdf = pd.DataFrame(data=ss)

    tmp = pd.DataFrame(data=tmp)

    outdf = outdf.set_index('Category')
    ssdf = ssdf.set_index('Measure')

    if showres:
        with open(path, "a") as f:

            print('\n\n----------------------------------------------------------', file=f)
            print('Confusion Matrix:', file=f)
            print(tmp, file=f)

            print('\n\n----------------------------------------------------------', file=f)
            print('Individual Summaries:', file=f)
            print(outdf, file=f)

            print('\n\n----------------------------------------------------------', file=f)
            print('Average Summaries:', file=f)
            print(ssdf, file=f)


    if saveres:
        outdf.round(3).to_csv(path + model_name + '_' + loss_type + str(nclass) + "_performance_ind.csv")
        ssdf.round(3).to_csv(path + model_name + '_' + loss_type + str(nclass) + "_performance_ave.csv")

    return outdf, ssdf, tmp


#############################################
### Pixel Proportion Statistical Analysis ###

def pixel_proportion(true, pred, label='Predicted', names=None, showres=False, saveres=False, path='', model_name='', loss_type='', nclass=5):
    lst_pixel_proportion = []
    path_csv  = path.replace("txt", "csv")

    ## Percentage of each pixel type ##
    n_pix = np.prod(pred.shape)
    unique_truth, counts_truth = np.unique(true, return_counts=True)

    # need to insure classes in pred and truth match
    if names != None:
        pred_counts_unique_true = [(i, np.sum(true == i))
                              for i, _ in enumerate(names)]
        unique_truth, counts_truth = zip(*pred_counts_unique_true)
    
    counts_truth = np.array(counts_truth)
    unique_truth = np.array(unique_truth)

    ## Test of unequal proportions ##
    # Note: generally uninformative because n = 1M #

    Ppvals = []
    unique, counts = np.unique(pred, return_counts=True)
    # need to insure even if a class isn't predicted, it's still in the list of names of predicted defects, so matches with truth
    if names != None:
        pred_counts_unique = [(i, np.sum(pred == i)) for i, _ in enumerate(names)]
        unique, counts = zip(*pred_counts_unique)    
    
    counts = np.array(counts)
    unique = np.array(unique)
    
    for i in range(len(counts_truth)):
        if unique_truth[i] in unique:
            j = np.where(unique_truth[i] == unique)[0][0]
            tmp = np.array((counts_truth[i], counts[j], n_pix, counts_truth[j]/n_pix, counts[j]/n_pix)).tolist()
        else:
            tmp = np.array((counts_truth[i], 0, n_pix, counts_truth[i]/n_pix, 0.0)).tolist()
        tmp.append(proportions_ztest(tmp[:2], np.repeat(n_pix,2), alternative='two-sided')[1])
        Ppvals.append(tmp)


    Ppvaldf = pd.DataFrame(Ppvals, columns=['Truth',label,'Pixels','Proportion', 'Pred', 'P-value'], index=names)

    Ppvaldf.columns = ['Truth',label,'Pixels','Proportion','Pred','P-value']
    
    # write to txt file
    if showres:
        with open(path, "a") as f:
            print(Ppvaldf.round(3), file=f)

    if saveres:
        Ppvaldf.round(3).to_csv(path + model_name + '_' + loss_type + str(nclass) + "_pixel_counts.csv")

    return Ppvaldf


# get accuracy per class (grain, boundary, etc.)
def accuracy_class(Y_test, max_preds, names):
    #IOU report
    lst_pandas = []
    index_pandas = []
    acc_sgl = np.zeros(5)
    for i in range(5):
        acc_sgl[i] = np.mean(max_preds[np.where(Y_test==i)] == i)
        lst_pandas.append(acc_sgl[i])
        index_pandas.append(names[i])

    #save accuracy overall
    index_pandas.append('Overall')
    lst_pandas.append(np.mean(acc_sgl))

    #write values to csv file before a iou function also writes to the file
    df1 = pd.DataFrame(lst_pandas, index = index_pandas)
    return df1


