
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter



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


def vis_chip(chip, nclass=5, filename=None, alpha=None, figsize=None, use_cbar= False):  # pragma: no cover
    """Visualize a chip from a batch.

    :param array-like chip: A numpy array or torch tensor of integers
        representing the true or predicted class.
    :param int nclass: The number of classes represented in the image. Default
        5.
    :param str filename: If given, a filename to which to save the plot.
    :param array-like alpha: If given, the uncertainty, which is visualized
        as transparency in the image. Must be the same size as the image, and
        all values between 0 and 1, where 1 corresponds to zero uncertainty.
    """
    if isinstance(chip, torch.Tensor):
        chip = chip.cpu().numpy()

    dpi=600

    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))

    if figsize is None:
        ypixels = image.shape[0]
        xpixels = image.shape[1]
        figsize = (xpixels/dpi, ypixels/dpi)

    plt.figure(figsize=figsize, dpi=dpi)

    if alpha is not None:
        image = np.stack((image[..., 0], image[..., 1], image[..., 2], alpha),
                         axis=-1)
    plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)

    # import pdb; pdb.set_trace()
    # if use_cbar == True:
    #     plt.axis('off')
    #     cbar = plt.colorbar(orientation='horizontal')
    #     cbar.ax.set_xticks(np.arange(nclass))
    #     cbar.ax.set_xticklabels(labels, rotation=90)
    if filename is not None:
        plt.savefig(filename)
    plt.ioff()
    plt.close('all')


def vis_with_batch(batch, chip, nclass=5, filename=None, alpha=None, figsize=None,
                   lines=None):  # pragma: no cover
    """Visualize a chip from a batch.
    :param array-like chip: A numpy array or torch tensor of integers
        representing the true or predicted class.
    :param int nclass: The number of classes represented in the image. Default
        5.
    :param str filename: If given, a filename to which to save the plot.
    :param array-like alpha: If given, the uncertainty, which is visualized
        as transparency in the image. Must be the same size as the image, and
        all values between 0 and 1, where 1 corresponds to zero uncertainty.
    :param tuple figsize: The size of the figure in inches.
    :param array-like lines: Lines to plot both vertically and horizontally on the image.
    """
    if isinstance(chip, torch.Tensor):
        chip = chip.cpu().numpy()
    if figsize is None:
        figsize = (6, 7)
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    plt.sca(axes[0])
    plt.imshow(batch.cpu(), cmap='gray')
    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    plt.axis('off')
    plt.sca(axes[1])
    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))
    if alpha is not None:
        image = np.stack((image[..., 0], image[..., 1], image[..., 2], alpha),
                         axis=-1)
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)
    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    plt.axis('off')
    ax1_divider = make_axes_locatable(axes[0])
    bax = ax1_divider.append_axes("left", size="10%", pad="3%")
    plt.sca(bax)
    plt.axis('off')
    ax2_divider = make_axes_locatable(axes[1])
    cax = ax2_divider.append_axes("right", size="10%", pad="3%")
    cbar = fig.colorbar(im, cax=cax)
    yt = np.arange(nclass)
    cax.set_yticks(yt)
    cax.set_yticklabels(labels[:len(yt)], fontsize=12)
    plt.tight_layout(pad=0.5)
    if filename is not None:
        # plt.savefig(filename.split('.')[0] + '.pgf') #removing due to dependency issues
        plt.savefig(filename, dpi=600)
    plt.show()

def vis_with_batch_and_hypothesis(batch, chip, nclass=5, filename=None, alpha=None, figsize=None,
                                  lines=None, scale = 1):  # pragma: no cover
    """Visualize a chip from a batch with the hypothesis plotted alongside.
    
    :param array-like batch: A numpy array or torch tensor of floats representing
        the original input image.
    :param array-like chip: A numpy array or torch tensor of integers
        representing the true or predicted class.
    :param int nclass: The number of classes represented in the image. Default
        5.
    :param str filename: If given, a filename to which to save the plot.
    :param array-like alpha: If given, the uncertainty, which is visualized
        as transparency in the image. Must be the same size as the image, and
        all values between 0 and 1, where 1 corresponds to zero uncertainty.
    :param tuple figsize: The size of the figure in inches.
    :param array-like lines: Lines to plot both vertically and horizontally on the image.
    """
    print("scale")
    print(scale)
    if isinstance(chip, torch.Tensor):
        chip = chip.cpu().numpy()
    if figsize is None:
        figsize = (6, 7)
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=figsize)

    formatter = lambda x, pos: x / scale  # 0.5 is the resolution

    
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_xlabel('μm')
    axes[0].set_ylabel('μm')

    plt.sca(axes[0])
    plt.imshow(batch.cpu(), cmap='gray')
    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    plt.axis('off')

    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_xlabel('μm')
    axes[1].set_ylabel('μm')
    plt.sca(axes[1])
    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)
    plt.axis('off')

    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].yaxis.set_major_formatter(formatter)
    axes[2].set_xlabel('μm')
    axes[2].set_ylabel('μm')
    plt.sca(axes[2])
    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))
    if alpha is not None:
        image = np.stack((image[..., 0], image[..., 1], image[..., 2], alpha),
                         axis=-1)
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)

    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    plt.axis('off')
    ax1_divider = make_axes_locatable(axes[0])
    bax = ax1_divider.append_axes("left", size="10%", pad="3%")
    plt.sca(bax)

    plt.axis('off')
    ax2_divider = make_axes_locatable(axes[1])
    ax3_divider = make_axes_locatable(axes[2])
    cax = ax3_divider.append_axes("right", size="10%", pad="3%")
    cbar = fig.colorbar(im, cax=cax)
    yt = np.arange(nclass)
    cax.set_yticks(yt)
    cax.set_yticklabels(labels[:len(yt)], fontsize=10)
    plt.tight_layout(pad=0.5)
    if filename is not None:
        # plt.savefig(filename.split('.')[0] + '.pgf') #removing due to dependency issues
        plt.savefig(filename, dpi=600)
    plt.show()

def vis_with_batch_true_and_hypothesis(batch, true, chip, nclass=5, filename=None, alpha=None, figsize=None,
                                       scale=0.1, lines=None):  # pragma: no cover
    """Visualize a chip from a batch with the hypothesis and truth.
    
    :param array-like batch: A numpy array or torch tensor of floats representing
        the original input image.
    :param array-like true: A numpy array or torch tensor of floats representing
        the true labels of the image.
    :param array-like chip: A numpy array or torch tensor of integers
        representing the true or predicted class.
    :param int nclass: The number of classes represented in the image. Default
        5.
    :param str filename: If given, a filename to which to save the plot.
    :param array-like alpha: If given, the uncertainty, which is visualized
        as transparency in the image. Must be the same size as the image, and
        all values between 0 and 1, where 1 corresponds to zero uncertainty.
    :param tuple figsize: The size of the figure in inches.
    :param array-like lines: Lines to plot both vertically and horizontally on the image.
    """
    if isinstance(chip, torch.Tensor):
        chip = chip.cpu().numpy()
    if figsize is None:
        figsize = (7, 7)
    fig, axes = plt.subplots(ncols=4, sharey=True, figsize=figsize)
    formatter = lambda x, pos: round(x / scale, 1)  # 0.5 is the resolution

    
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].set_xlabel('μm')
    axes[0].set_ylabel('μm')
    plt.sca(axes[0])
    plt.title("Original\nImage")
    plt.imshow(batch.cpu(), cmap='gray')
    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    #plt.axis('off')
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_xlabel('μm')
    axes[1].set_ylabel('μm')
    plt.sca(axes[1])
    plt.title('Prediction')
    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)
    #plt.axis('off')
    axes[2].xaxis.set_major_formatter(formatter)
    axes[2].yaxis.set_major_formatter(formatter)
    axes[2].set_xlabel('μm')
    axes[2].set_ylabel('μm')
    plt.sca(axes[2])
    plt.title("Prediction\nwith Uncertainty")
    cmap, norm, labels = cmaps[nclass]
    image = cmap(norm(chip))
    if alpha is not None:
        image = np.stack((image[..., 0], image[..., 1], image[..., 2], alpha),
                         axis=-1)
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)
    if lines:
        for line in lines:
            plt.axvline(line, color='white')
            plt.axhline(line, color='white')
    #plt.axis('off')
    axes[3].xaxis.set_major_formatter(formatter)
    axes[3].yaxis.set_major_formatter(formatter)
    axes[3].set_xlabel('μm')
    axes[3].set_ylabel('μm')
    plt.sca(axes[3])
    image = cmap(norm(true))
    im = plt.imshow(image, cmap=cmap, vmin=-0.5, vmax=nclass-0.5)
    #plt.axis('off')
    plt.title("Truth")
    ax1_divider = make_axes_locatable(axes[0])
    bax = ax1_divider.append_axes("left", size="10%", pad="3%")
    plt.sca(bax)
    plt.axis('off')
    ax2_divider = make_axes_locatable(axes[1])
    ax3_divider = make_axes_locatable(axes[2])
    ax4_divider = make_axes_locatable(axes[3])
    cax = ax4_divider.append_axes("right", size="10%", pad="3%")
    cbar = fig.colorbar(im, cax=cax)
    yt = np.arange(nclass)
    cax.set_yticks(yt)
    cax.set_yticklabels(labels[:len(yt)], fontsize=12)
    plt.tight_layout(pad=0.5)
    if filename is not None:
        # plt.savefig(filename.split('.')[0] + '.pgf') #removing due to dependency issues
        plt.savefig(filename, dpi=600)
    plt.show()