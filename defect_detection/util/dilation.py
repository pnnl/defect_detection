import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
import os
from defect_detection.image_preprocessing import image_prep

class Dilator:
    """A class to dilate certain labels in a hand labeled image.
    
    :param str image: Path to the .tif formatted labeled image.
    :param int class_no: Number of the class in the image which you desire to
        dilate. Default ``1``
    :param int nclass: Number of total classes. Default ``5``.
    :param int amount: The number of pixels to dilate. Must be an integer, but
        can be negative. If negative, all pixels that are "freed" by the inset
        operation are converted to class number ``0``. Default ``1``.
    :param str order: If ``"below"``, then the dilator first isolates the class,
        then dilates it, then adds back in all other classes with higher class
        numbers. In effect, this dilates the class without reducing the area of
        any class with a higher class number. If ``"above"``, then dilator
        dilates the class over all other classes, reducing the area of all
        other classes.  
    """
    def __init__(self, image, class_no=1, nclass=5, amount=1, order='below',
                 preview=False, save=False, **kwargs):
        self.nclass = nclass
        self.filename = 'image.tif'
        if isinstance(image, str):
            self.filename = image
            oi, image, _ = image_prep(os.path.dirname(image),
                                     os.path.basename(image), nclass)
            imh, imw = image.shape[:2]
            self.original_image = oi
            _image = np.zeros((imh, imw, nclass))
            for i in range(nclass):
                _image[image==i, i] = 1
            image = _image
        self.image = image
        
        self.class_no = class_no
        self.amount = amount + 1
        self.order = order
        self.preview = preview
        self.save = save

    def __call__(self):
        if self.amount == 0:
            return self.image
        elif self.amount >= 0:
            # extract the class layer
            layer = self.image[..., self.class_no]
            ksize = self.amount
            dilated = F.max_pool2d(torch.from_numpy(layer.copy()).unsqueeze(0),
                                   ksize, stride=1, padding=ksize//2)
            if ksize % 2 == 0:
                dilated = F.avg_pool2d(dilated, 2, stride=1, padding=0)
            dilated = torch.round(dilated)
            dilated = dilated.numpy()
            
            if self.order == 'below':
                for i_layer in range(self.class_no+1, self.image.shape[-1]):
                    dilated = dilated - self.image[..., i_layer]
                self.image[..., self.class_no] = dilated
                for i_layer in range(self.class_no):
                    self.image[..., i_layer] = self.image[..., i_layer] - dilated
            elif self.order == 'above':
                for i_layer in range(self.image.shape[-1]):
                    if i_layer == self.class_no:
                        self.image[..., i_layer] = dilated
                    else:
                        self.image[..., i_layer] = self.image[..., i_layer] - dilated
            self.image = np.clip(self.image, 0, 1)
            if self.preview: # pragma: no cover
                imh, imw = self.image.shape[:2]
                _image = np.zeros((imh, imw))
                for i in range(self.nclass):
                    _image[self.image[...,i] == 1] = i
                plt.figure(figsize=(6, 6))
                plt.imshow(_image, cmap='tab10', vmin=0, vmax=self.nclass,
                           interpolation='nearest')
                plt.colorbar()
                plt.show()
            if self.save:
                resave_name = self.filename.replace('.tif', '_dilated.tif')
                imh, imw = self.image.shape[:2]
                _image = (self.original_image).astype(np.uint8)
                replacements = [[255, 0, 0], [0, 255, 0],
                                [255, 255, 0], [0, 0, 255]]
                for i, replace in enumerate(replacements):
                    _image[self.image[...,i+1] == 1] = replace
                im = Image.fromarray(_image)
                im.save(resave_name)
            return self.image



def _make_parser(): # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-no', type=int, default=1,
                        help='Number of class to dilate')
    parser.add_argument('--amount', type=int, default=1,
                        help='Number of pixels to dilate, can be negative')
    parser.add_argument('--order', default='below',
                        help='If "below", dilate the class, then add back in' +
                        'all classes above, if "above", place all other ' +
                        'classes, then dilate the specified class.')
    parser.add_argument('--preview', default=False,
                        help='Whether to show a preview (only for a system' +
                        'with access to a display).')
    parser.add_argument('--image', default='C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif',
                        help='Path to the image labels to dilate')
    parser.add_argument('--save', default=False,
                        help='should we save the image')
    parser.add_argument('--logging-level', default='INFO', nargs='?',
                        help='How verbose should the information be')
    return parser

def _run_cli(): # pragma: no cover
    args = _make_parser().parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=args.logging_level)
    dilator = Dilator(**vars(args))
    dilator()

if __name__ == "__main__":
    _run_cli()