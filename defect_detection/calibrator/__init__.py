from ast import Import
import numpy as np
import defect_detection
import torch
import logging
#import tqdm
from tqdm.auto import tqdm
import numpy as np
from collections import Counter
from matplotlib import cm

from sklearn.metrics import confusion_matrix, accuracy_score
import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from ..vis import colors5, labels5

from scipy.stats import binned_statistic

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def DUMMY(x, *args, **kwargs):
    return x

if False:
    import faiss

    def knn_init(x):
        res = faiss.StandardGpuResources()
        deviceId = 0
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        quantizer = faiss.IndexFlatL2(5)
        faiss_metric = faiss.METRIC_L2
        #index = faiss.IndexHNSWFlat(5, 36)
        index = faiss.IndexPQ(quantizer, 5)
        #index = faiss.IndexFlat(5)
        #index.hnsw.efConstruction = 10
        #faiss.ParameterSpace().set_index_parameter(index, "efSearch", 10)
        #index.efConstruction = 200
        #index.efSearch = 200
        #index = faiss.IndexIVFFlat(quantizer, 5, 8192, faiss_metric)
        index = faiss.index_cpu_to_gpu(res, deviceId, index, co)

        #assert not index.is_trained
        #index.train(x)  # add vectors to the index
        #assert index.is_trained
        index.add(x)
        #index.nprobe = 40
        return index

    def knn_query(x, tree, k):
        return tree.search(x, k)
elif False:
    from pykeops.numpy import Vi, Vj

    def knn_init(x):
        X_i = Vi(0, 5)
        X_j = Vj(x)
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        knn_fun = D_ij.argKmin(25, dim=1)
        return X_i, D_ij, knn_fun

    def knn_query(x, tree, k):
        X_i, D_ij, knn_fun = tree
        indices = knn_fun(x)
        d = D_ij[indices]
        return d
elif False:
    class TorchKNN(torch.nn.Module):
        def __init__(self, x):
            self.x_train = torch.from_numpy(x).to(self.device)
            self.N_train, self.D = self.x_train.shape
            self.x_train_norm = (self.x_train ** 2).sum(-1)

        def __call__(self, x, k):
            x = torch.from_numpy(x).to(self.device)
            N_test = x.shape[0]
            av_mem = int(5e8)
            Ntest_loop = min(max(1, av_mem // (4 * self.D * self.N_train)),
                             N_test)
            N_loop = (N_test - 1) // Ntest_loop + 1
            out = torch.empty((N_test, k))

            for i in range(N_loop):
                chunk = slice(i*Ntest_loop, (i+1)*Ntest_loop)
                x_test_i = x[chunk]
                out[chunk] = self.nearest_neighbors(x_test_i, k)

            return out.cpu().numpy()
            
        def nearest_neighbors(self, x, k):
            largest = False
            x_test_norm = (x**2).sum(-1)
            diss = (x_test_norm.view(-1, 1) + self.x_train_norm.view(1, -1)
                    - 2.0 * x @ self.x_train.t())
            d, _ = diss.topk(k, dim=1, largest=False)
            return d

    def knn_init(x):
        return TorchKNN(x)

    def knn_query(x, tree, k):
        return tree(x, k)
else:
    from sklearn.neighbors import KDTree

    def knn_init(x):
        return KDTree(x)
    
    def knn_query(x, tree, k):
        return tree.query(x, k)
#plt.style.use('ah')

__eps__ = 1e-9

def wilsons(phat, n):
    pkup = np.inf
    pkupnext = phat
    
    for i in range(1000):
        pkup = np.clip(pkupnext, __eps__, 1.0 - __eps__)
        pkupnext = phat + 1.96 * np.sqrt(pkup * (1.0 - pkup) / n)
        if np.all(np.abs(pkupnext - pkup) < 1e-4):
            break
    pkdown = np.inf
    pkdownnext = phat
    for i in range(1000):
        pkdown = np.clip(pkdownnext, __eps__, 1.0 - __eps__)
        pkdownnext = phat - 1.96 * np.sqrt(pkdown * (1.0 - pkdown) / n)
        if np.all(np.abs(pkdownnext - pkdown) < 1e-4):
            break
    return pkdownnext, pkupnext

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :pacheram quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if len(values) == 0:
        return np.zeros_like(quantiles)
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


class Calibrator:
    """A wrapper class which takes a defect_detection model and statistically
    calibrates it.
    
    :param model: A ``torch.nn.Module`` object which returns softmax outputs.
    :param list-like weights: Weights for each class of the softmax outputs.
    """
    def __init__(self, model, val_dataset, N_trials=5, max_chips=40,
                 transform=None, device='cpu', bayes=False, h0_aggregation='vote',
                 full_posterior=False, nclass=5, image_name ="default"):
        self.model = model
        self.model.eval()
        self.nclass = nclass
        self.npix = 128
        self.N = 15
        self.N_trials = N_trials
        self.max_chips = max_chips
        self.image_name = image_name
        if transform is None:
            transform = lambda x: x
        self.transform = transform
        self.V_d = 8.0 * np.pi * np.pi / 15.0
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.val_dataset = val_dataset
        # set up ks for our density estimate
        self.k = 25
        self.ks = np.expand_dims(np.arange(1, self.k+1), 0)
        if bayes:
            self.get_probability = self.get_probability_bayes
        else:
            self.get_probability = self.get_probability_frequentist
        if h0_aggregation == 'sum':
            self.get_h0 = self.get_h0_sum
        elif h0_aggregation == 'complement':
            self.get_h0 = self.get_h0_complement
        elif h0_aggregation == 'truncate':
            self.get_h0 = self.get_h0_truncate
        elif h0_aggregation == 'complement_renorm':
            self.get_h0 = self.get_h0_complement_renorm
        elif h0_aggregation == 'vote':
            self.get_h0 = self.get_h0_vote
        elif h0_aggregation == 'truncate_1/N':
            self.get_h0 = self.get_h0_truncate_1oN
        else:
            raise Exception(f"No such H0 Aggregation Method {h0_aggregation}")
        self.full_posterior = full_posterior

    def get_variational_samples(self, idx, dataset):
        N_train = len(idx)
        min_idx = int(np.min(idx))
        preds = torch.empty((self.N_trials, N_train, self.nclass, 128, 128))
        imags = torch.empty((N_train, 128, 128))
        trues = torch.empty((N_train, 1, 128, 128))
        with torch.no_grad():
            for i_trial in DUMMY(np.arange(self.N_trials), ncols=80,
                                 desc='Computing variational samples'):
                for i_datapoint in idx:
                    if len(dataset[i_datapoint]) == 3:
                        image, _, label = dataset[i_datapoint]
                    else:
                        image, label = dataset[i_datapoint]
                    label = label.argmax(2).long()
                    image = image.to(self.device).permute((2, 0, 1)).unsqueeze(0)
                    _pred = self.model(image)
                    preds[i_trial, i_datapoint - min_idx, ...] = _pred.detach().cpu()
                    imags[i_datapoint - min_idx, ...] = image[0, 0, ...].detach().cpu()
                    if i_trial == 0:
                        trues[i_datapoint - min_idx, ...] = label.detach().cpu()
        maxes = preds.mean(dim=0).argmax(dim=1)
        trues = trues.squeeze()
        return preds, trues, maxes, imags

    def compute_batch(self, batch):
        batch_size = batch.shape[0]
        preds = torch.empty((self.N_trials, batch_size, self.nclass, 128, 128))
        with torch.no_grad():
            for i_trial in DUMMY(np.arange(self.N_trials), ncols=80,
                                desc='Computing variational samples',
                                position=1, keep=False):
                batch = batch.to(self.device).permute((0, 3, 1, 2))
                preds[i_trial] = self.model(batch)
        maxes = preds.mean(dim=0).argmax(dim=1)
        return preds, maxes


    def calibrate(self):
        with tqdm(desc='Calibrating', total=5, ncols=80) as pbar:
            # first enumerate the validation set and split into
            # val_train and val_val
            val_dataset = self.val_dataset
            N = np.min([len(val_dataset), self.max_chips])
            N_train = int(0.8 * N)
            N_val = N - N_train
            self.idx_train = np.arange(N_train)
            self.idx_val = np.arange(N_train, N)
            # then, create predictions from all of the val_val set
            preds, trues, maxes, imags \
                = self.get_variational_samples(self.idx_train, val_dataset)
            # update our progress bar
            pbar.update()
            # find the weights
            count = Counter(trues.view(-1).cpu().numpy())
            self.ws = np.array([1.0/count[i] for i in range(self.nclass)])
            self.ws = np.ones_like(self.ws)
            # then, determine the accuracy and split the samples into correct and
            # incorrect examples for val_train
            flat_trues = trues.clone().reshape(-1).cpu().numpy()
            flat_maxes = maxes.clone().reshape(-1).cpu().numpy()
            # preds are right now in N_trials, N_chips, N_classes, N_rows, N_cols
            # need to reshape to the same format as flat maxes, flat true
            flat_preds = preds.clone().permute((0, 2, 1, 3, 4))
            flat_preds = flat_preds.reshape(self.N_trials, self.nclass, -1).cpu().numpy()
            cm = confusion_matrix(flat_trues, flat_maxes)
            acc = accuracy_score(flat_trues, flat_maxes,
                                sample_weight=self.ws[flat_trues.astype(int)])

            # then, construct a KDTree for the log of the correct and incorrect
            # scores
            is_correct = np.expand_dims(flat_trues, 0) == flat_preds.argmax(axis=1)
            _acc = np.nanmean(is_correct)
            logging.debug(f'Accuracy = {100.0 * acc:.2f} {100.0 * _acc:.2f}')
            not_correct = np.logical_not(is_correct)
            # do the same for all
            all = self.reshape_preds(preds)
            all = all[np.logical_or(is_correct, not_correct)]
            # all is now in N_trials, -1, N_classes, need to just be -1, N_classes
            all = all.reshape(-1, self.nclass)
            self.N_all = all.shape[0]
            if self.N_all > 1E7:
                logging.warning("Your calibration settings lead to more than 10^7" +
                                " samples in the calibration set. This may lead" +
                                " to very slow computation times")
            self.all = all#[:int(1E7), :]
            self.all_tree = knn_init(self.transform(self.all))
            if not self.full_posterior:
                # find only the predictions which are correct
                correct = self.reshape_preds(preds)
                correct = correct[is_correct]
                correct = correct.reshape(-1, self.nclass)
                # downsample corrects
                self.N_correct = correct.shape[0]
                self.correct = correct#[:int(acc * 1E7), :]
                # create a KDTree for fast lookup of nearest neighbors
                self.correct_tree = knn_init(self.transform(self.correct))
            else:
                self.trees = []
                for i_class in range(self.nclass):
                    is_class_i = flat_trues == i_class
                    
                    class_i = self.reshape_preds(preds)
                    class_i = class_i[:, is_class_i, :]
                    class_i = class_i.reshape(-1, self.nclass)
                    self.trees.append(knn_init(self.transform(class_i)))
            # update the progress bar
            pbar.update()

            _, _, _, N_rows, N_cols = preds.shape
            batch = self.reshape_preds(preds)
            self.val_p, val_h0 = self.get_probability(batch)
            # update the progress bar
            pbar.update()

            if self.full_posterior:
                ps = []
                cs = []
                Ns = []
                bins = np.linspace(0.0, 1.0, 25)
                midps = ((bins[1:] + bins[:-1])/2.0)
                fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
                plt.sca(axes[0])
                for i_class in range(self.nclass):
                    i_class_ps = self.val_p[:, i_class].numpy()
                    ns, _ = np.histogram(i_class_ps, bins=bins)
                    i_class_correct = (flat_trues == i_class).astype(float)
                    accs, _, _ = binned_statistic(i_class_ps, i_class_correct, bins=bins)
                    down, up = wilsons(accs, ns)
                    plt.errorbar(midps, accs,
                                 yerr=np.stack((accs - down, up - accs), axis=0),
                                 linestyle='none', marker='.',
                                 label=labels5[i_class],
                                 color=colors5[i_class])
                    Ns.extend(np.ones_like(accs))
                    ps.extend(midps)
                    cs.extend(accs)
                plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='#cccccc', label='desired')
                plt.legend(loc=4, ncol=2, fontsize='xx-small')
                plt.xlabel("Calibrated\nScore")
                plt.ylabel("Sample\nAccuracy")
                plt.gca().set_aspect('equal')
                plt.sca(axes[1])

                ps = np.array(ps)
                idx = np.isfinite(ps)
                cs = np.array(cs)
                Ns = np.array(Ns)
                ps = ps[idx]
                cs = cs[idx]
                Ns = Ns[idx]
                z = np.abs(np.array(ps) - np.array(cs))
                A, _, _ = binned_statistic(ps, cs, statistic='mean', bins=bins)
                C, _, _ = binned_statistic(ps, ps, statistic='mean', bins=bins)
                Ns, _, _ = binned_statistic(ps, Ns, statistic='sum', bins=bins)
                bins = np.geomspace(1e-3, 1e0, 10)
                hz, bz, _ = plt.hist(z, bins=bins, histtype='step', color='#616265')
                plt.semilogx()
                #plt.xticks([1e-5, 1e-3], ['10$^{-5}$', '10$^{-3}$'])
                plt.xlabel('Residual')
                plt.ylabel('Samples')
                #ece = np.sum(z * np.array(Ns)) / np.sum(Ns)
                ece = np.nansum(np.array(Ns) * np.abs(A - C) / np.nansum(Ns))
                mean_bz = np.exp(np.mean(np.log(bz)))
                plt.text(mean_bz, np.max(hz), "ECE\n"+f'{100.0 * ece:.2f}%',
                        ha='center', va='top', bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))
                #plt.ylim(0, 20)
                plt.tight_layout(pad=0.5)
                #plt.savefig('calibration_residuals.pgf', dpi=300)
                #plt.savefig('calibration_residuals.png', dpi=300)
                # plt.show()
                return
            else:
                self.val_p = self.val_p.reshape(N_train, N_rows, N_cols)
                val_h0 = val_h0.reshape(N_train, N_rows, N_cols)
                self.compute_quantiles(self.val_p, trues, val_h0, imags)
                # update the progress bar
                pbar.update()
                self.evaluate_calibration()
                # update the progress bar
                pbar.update()
                return

    def reshape_preds(self, batch):
        # batch should be N_trials, N_chips, N_classes, N_rows, N_cols
        # we want to get it to N_trials, N_chips * N_rows * N_cols, N_classes, 
        batch = batch.cpu().permute((0, 1, 3, 4, 2))
        batch = batch.reshape(self.N_trials, -1, self.nclass)
        return batch.numpy()


    def compute_quantiles(self, p, trues, maxes, batch):
        row1 = torch.cat([batch[0, ...], batch[1, ...]], axis=1)
        row2 = torch.cat((batch[2, ...], batch[3, ...]), axis=1)
        combined_batch = torch.cat((row1, row2), axis=0)
        row_pred1 = torch.cat([maxes[0, ...], maxes[1, ...]], axis=1)
        row_pred2 = torch.cat([maxes[2, ...], maxes[3, ...]], axis=1)
        combined_pred = torch.cat((row_pred1, row_pred2), axis=0)
        row_p1 = torch.cat([p[0, ...], p[1, ...]], axis=1)
        row_p2 = torch.cat([p[2, ...], p[3, ...]], axis=1)
        combined_p = torch.cat((row_p1, row_p2), axis=0)
        defect_detection.vis.vis_with_batch(combined_batch, combined_pred, alpha=combined_p,
                                      figsize=(3, 1.75), filename=self.image_name + 'pred_batch.png', lines=[p[0, ...].shape[-1]])
        is_correct = (trues == maxes).cpu().numpy()
        not_correct = np.logical_not(is_correct)
        # get the weights of everything correct
        correct_true_labels = trues
        is_correct = torch.from_numpy(is_correct).bool()
        correct_true_labels = correct_true_labels[is_correct]
        correct_true_labels = correct_true_labels.reshape(-1)
        correct_true_labels = correct_true_labels.numpy().astype(int)
        correct_ws = self.ws[correct_true_labels]
        # get the weights of everything incorrect
        incorrect_true_labels = trues
        not_correct = torch.from_numpy(not_correct).bool()
        incorrect_true_labels = incorrect_true_labels[not_correct]
        incorrect_true_labels = incorrect_true_labels.reshape(-1)
        incorrect_true_labels = incorrect_true_labels.numpy().astype(int)
        incorrect_ws = self.ws[incorrect_true_labels]
        all_labels = trues
        all_labels = all_labels.reshape(-1)
        all_labels = all_labels.numpy().astype(int)
        all_ws = self.ws[all_labels]
        flat_correct = p[is_correct].reshape(-1).numpy()
        flat_incorrect = p[not_correct].reshape(-1).numpy()
        flat_all = p.reshape(-1).numpy()
        max_all = maxes.reshape(-1).numpy()
        plt.figure(figsize=(3, 3/1.618))
        plt.hist(flat_correct, bins=25, label='correct', histtype='step', density=True, weights=correct_ws)
        plt.hist(flat_incorrect, bins=25, label='incorrect', histtype='step', density=True, weights=incorrect_ws)
        plt.hist(flat_all, bins=25, label='all', histtype='step', density=True, weights=all_ws)
        plt.ylabel('Number of Pixels')
        plt.xlabel('Calculated Probability Correct')
        plt.semilogy()
        plt.legend(loc='upper center')
        plt.tight_layout(pad =0.5)
        plt.savefig(self.image_name + 'pixel_histograms.png', dpi=300)
        # plt.show()
        
        bins = np.linspace(0.0, 1.0, 25)
        midps = (bins[:-1] + bins[1:]) / 2.0
        pr_correct, _ = np.histogram(flat_all, bins=bins)
        p_correct = midps
        #pr_correct / flat_all.shape[0]
        p_down, p_up = wilsons(p_correct, pr_correct)
        p_down = p_correct - p_down
        p_up = p_up - p_correct
        x = []
        y = []
        ns = []
        uyd = []
        uyu = []
        uxd = []
        uxu = []
        delta = bins[1] - bins[0]
        ps = []
        cs = []
        Ns = []
        for low, p, pd, pu in zip(bins, p_correct, p_down, p_up):
            idx = np.logical_and(flat_all >= low, flat_all < low + delta).astype(bool)
            n = np.sum(idx)
            ps.extend(flat_all[idx])
            cs.extend(max_all[idx] == all_labels[idx])
            Ns.extend(np.ones_like(flat_all[idx]))
            if n > 100:
                acc = accuracy_score(max_all[idx], all_labels[idx],
                                     sample_weight=all_ws[idx])
                x.append(p)
                y.append(acc)
                down, up = wilsons(acc, n)
                uyd.append(acc - down)
                uyu.append(up - acc)
                uxd.append(pd)
                uxu.append(pu)
                ns.append(n)
        ps = np.array(ps)
        cs = np.array(cs)
        Ns = np.array(Ns)
        fig, axes = plt.subplots(ncols=2, figsize=(3, 1.5))
        plt.sca(axes[0])
        plt.errorbar(x, y, xerr=np.stack((uxd, uxu)), yerr=np.stack((uyd, uyu)),
                     linestyle='none', markersize=4, marker='.', alpha=0.5,
                     barsabove=True, ecolor='#A63F1E', capsize=2, capthick=0.5,
                     label='true')
        plt.plot([0.0, 1.0], [0.0, 1.0], color='#cccccc', linestyle='--', 
                 label='desired')
        plt.xlabel("Calibrated\nScore")
        plt.ylabel("Sample\nAccuracy")
        plt.gca().set_aspect('equal')
        #plt.tight_layout()
        plt.savefig(self.image_name + 'calibrated_quantiles.png', dpi=300)
        #plt.show()
        #plt.figure(figsize=(3, 3/1.618))
        plt.sca(axes[1])
        z = np.abs(np.array(x) - np.array(y))
        idx = np.isfinite(ps)
        ps = ps[idx]
        cs = cs[idx]
        Ns = Ns[idx]
        A, _, _ = binned_statistic(ps, cs, statistic='mean', bins=bins)
        C, _, _ = binned_statistic(ps, ps, statistic='mean', bins=bins)
        Ns, _, _ = binned_statistic(ps, Ns, statistic='sum', bins=bins)
        bins = np.geomspace(1e-3, 1e0, 10)
        hz, bz, _ = plt.hist(z, bins=bins, histtype='step', color='#616265')
        plt.semilogx()
        #plt.xticks([1e-5, 1e-3], ['10$^{-5}$', '10$^{-3}$'])
        plt.xlabel('Residual')
        plt.ylabel('Samples')
        #ece = np.sum(z * np.array(ns)) / np.sum(ns)
        ece = np.nansum(np.array(Ns) * np.abs(A - C) / np.nansum(Ns))
        mean_bz = np.exp(np.mean(np.log(bz)))
        plt.text(mean_bz, np.max(hz), "ECE\n"+f'{100.0 * ece:.2f}%',
                 ha='center', va='top', bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))
        #plt.ylim(0, 20)
        plt.tight_layout(pad=0.5)
        #plt.savefig('calibration_residuals.pgf', dpi=300)
        plt.savefig(self.image_name + 'calibration_residuals.png', dpi=300)
        # plt.show()

    def __call__(self, image, vis=False):
        h, w = image.shape[-2:]
        preds = torch.empty((self.N_trials, 1, self.nclass, h, w))
        imags = torch.empty((1, h, w))
        with torch.no_grad():
            for i_trial in DUMMY(np.arange(self.N_trials), ncols=80,
                                desc='Computing variational samples', leave=False):
                _pred = self.model(image)
                preds[i_trial, 0, ...] = _pred
        batch = self.reshape_preds(preds)
        p, h0 = self.get_probability(batch, from_val=False)
        if vis:
            defect_detection.vis.vis_with_batch(image.cpu().squeeze().permute((1, 2, 0)),
                                                h0.cpu(), alpha=p.cpu())
        if self.full_posterior:
            return h0.reshape(h, w), p.reshape(h, w, self.nclass)
        return h0.reshape(h, w), p.reshape(h, w)

    def evaluate_calibration(self):
        # evaluate the quantiles of the density estimates for the val_val set
        preds, trues, maxes, imags \
            = self.get_variational_samples(self.idx_val, self.val_dataset)
        _, N_val, _, N_rows, N_cols = preds.shape
        batch = self.reshape_preds(preds)
        self.valval_p, valval_h0 = self.get_probability(batch, from_val=False)
        self.valval_p = self.valval_p.reshape(N_val, N_rows, N_cols)
        valval_h0 = valval_h0.reshape(N_val, N_rows, N_cols)
        self.compute_quantiles(self.valval_p, trues, valval_h0, imags)
        
    def get_p_is_h0(self, batch, from_val=True):
        _p = []
        for i_trial in range(self.N_trials):
            # the probability of correctness is the weighted density of a
            # pixel being correct in that location in softmax space divided by the
            # weighted density of a pixel being incorrect in that location in
            # softmax space + the probability of it being correct
            p_correct = self.get_density(batch[i_trial], self.correct_tree,
                                         1.0, from_val=from_val)
            p_all = self.get_density(batch[i_trial], self.all_tree,
                                     1.0, from_val=from_val)

            p = p_correct / p_all
            _p.append(p)
        p = torch.stack(_p, dim=0)
        smallest_possible = 1.0 / self.N_all
        p = torch.clamp(p, min=smallest_possible, max=1.0 - smallest_possible)
        # for each prediction, we need one hypothesis. This hypothesis is the
        # maximum class of the sum of all the probabilities
        trial_h0 = torch.argmax(torch.from_numpy(batch), -1)
        h0 = self.get_h0(trial_h0, batch, p)
        # now, we need to evaluate that p is h_0
        p_is_h0 = torch.zeros_like(p)
        idx_c = h0 == trial_h0
        idx_nc = h0 != trial_h0
        p_is_h0[idx_c] = p[idx_c]
        p_is_h0[idx_nc] = 1.0 - p[idx_nc]
        return p_is_h0, h0

    def get_full_posterior(self, batch, from_val=True):
        N_preds = batch.shape[1]
        p = np.nan * torch.ones((self.N_trials, N_preds, self.nclass))
        for i_trial in range(self.N_trials):
            p_all = self.get_density(batch[i_trial], self.all_tree,
                                     1.0, from_val=from_val)
            for i_class in range(self.nclass):
                p_class_i = self.get_density(batch[i_trial], self.trees[i_class],
                                             1.0, from_val=from_val)
                p[i_trial, :, i_class] = p_class_i
            p[i_trial, :, :] = p[i_trial, :, :] / p[i_trial, :, :].sum(axis=-1, keepdims=True)
        p = p / p.sum(dim=-1, keepdim=True)
        smallest_possible = 1.0 / self.N_all
        p = torch.clamp(p, min=smallest_possible, max=1.0 - smallest_possible)
        trial_h0 = torch.argmax(torch.from_numpy(batch), -1)
        h0 = self.get_h0(trial_h0, batch, None)
        return p, h0

    def get_h0_truncate(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            p_h0[idx_c, i_class] = p.numpy()[idx_c]
        p_h0[p_h0 < 0.5] = 0.0
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0

    def get_h0_truncate_1oN(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            p_h0[idx_c, i_class] = p.numpy()[idx_c]
        p_h0[p_h0 <= (1.0 / self.nclass)] = 0.0
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0

    def get_h0_complement(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            idx_nc = trial_h0 != i_class
            p_h0[idx_c, i_class] = p.numpy()[idx_c]
            p_h0[idx_nc, i_class] = 1.0 - p.numpy()[idx_nc]
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0

    def get_h0_complement_renorm(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            idx_nc = trial_h0 != i_class
            p_h0[idx_c, i_class] = (p.numpy()[idx_c]) / (p.numpy()[idx_c] + (float(self.nclass - 1) * (1.0 - p.numpy()[idx_c])))
            p_h0[idx_nc, i_class] = (1.0 - p.numpy()[idx_nc]) / (p.numpy()[idx_nc] + (float(self.nclass - 1) * (1.0 - p.numpy()[idx_nc])))
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0

    def get_h0_sum(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            p_h0[idx_c, i_class] = p.numpy()[idx_c]
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0

    def get_h0_vote(self, trial_h0, batch, p):
        p_h0 = np.zeros_like(batch)
        for i_class in range(self.nclass):
            idx_c = trial_h0 == i_class
            p_h0[idx_c, i_class] = 1.0
        p_h0 = torch.sum(torch.from_numpy(p_h0), dim=0)
        h0 = torch.argmax(p_h0, dim=-1)
        return h0
        

    def get_probability_frequentist(self, batch, from_val=True):
        if self.full_posterior:
            p_is_h0, h0 = self.get_full_posterior(batch, from_val=from_val)
        else:
            p_is_h0, h0 = self.get_p_is_h0(batch, from_val=from_val)
        # now use the geometric mean to combine these probabilities
        p = torch.exp(torch.mean(torch.log(p_is_h0), dim=0))
        return p, h0

    def get_probability_bayes(self, batch, from_val=True):
        p_is_h0, h0 = self.get_p_is_h0(batch, from_val=from_val)
        # now, we need to use bayes rule to combine the probabilities. We do
        # so by converting to odds and then multiplying by the bayes factor
        K = p_is_h0 / (1.0 - p_is_h0)
        p_prior = 0.5
        o_prior = p_prior / (1.0 - p_prior)
        o = o_prior * torch.exp(torch.sum(torch.log(K), dim=0))
        p = o / (1.0 + o)
        return p, h0
        

    def get_probabilities(self, batch):
        preds, maxes = self.compute_batch(batch)
        _, N_val, _, N_rows, N_cols = preds.shape
        batch = self.reshape_preds(preds)
        p = self.get_probability(batch)
        p = p.reshape(N_val, N_rows, N_cols)
        return p, maxes


    def get_density(self, batch, tree, n, from_val=True):
        N_points = batch.shape[0]
        N_chunks = (N_points // 20_000) + 1
        subbatches = np.array_split(batch, N_chunks, axis=0)
        ps = []
        if from_val:
            k = self.k + 1
        else:
            k = self.k
        for subbatch in DUMMY(subbatches, ncols=80, leave=False,
                              desc='Calculating the density of batches'):
            # find the distances between nearest neighbors and the index of those
            # neighbors
            subbatch = np.ascontiguousarray(subbatch)
            dist, idx = knn_query(self.transform(subbatch), tree, k=k)
            # remove the innermost point
            if from_val:
                dist = dist[:, 1:]
            else:
                dist = dist
            # calculate the density
            p = np.sum(np.arange(k)) / (np.sum(np.power(dist, self.nclass), axis=1))
            ps.append(p)
        p = np.concatenate(ps, axis=0)
        return torch.from_numpy(p)
