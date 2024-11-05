import argparse
import logging
import matplotlib.pyplot as plt
import os
import cv2

class Cutter:
    """A class to cut images, removing portions that aren't labeled properly or
        have low-quality data.

    :param str image: Path to the .tif formatted image.
    :param int x0: Minimum value along the x-axis to include in the new image.
        Default ``0``.
    :param int y0: Minimum value along the y-axis to include in the new image.
        Default ``0``.
    :param int x1: Maximum value along the x-axis to include in the new image.
        Default ``None`` which is converted to the length of the x-axis.
    :param int y1: Maximum value along the y-axis to include in the new image.
        Default ``None`` which is converted to the length of the y-axis.
    """
    def __init__(self, image, x0=0, y0=0, x1=None, y1=None,
                 preview=False, save=False, **kwargs):
        self.filename = 'image.tif'

        if isinstance(image, str):
            self.filename = image
            oi = cv2.imread(os.path.join(os.path.dirname(image), os.path.basename(image)))
            imh, imw = oi.shape[:2]
            self.original_image = oi
            self.imw = imw
            self.imh = imh
            self.labeled_image = cv2.imread(os.path.join(os.path.dirname(image), 'Labeled ' + os.path.basename(image)))

        self.x0 = x0
        self.y0 = y0
        if x1 is None:
            self.x1 = imw
        else:
            self.x1 = x1
        if y1 is None:
            self.y1 = imh
        else:
            self.y1 = y1


        self.preview = preview
        self.save = save

    def __call__(self):
        if self.x0==0 and self.y0==0 and self.x1==self.imw and self.y1==self.imh:
            return self
        else:
            self.original_image = self.original_image[self.y0:self.y1, self.x0:self.x1]
            self.labeled_image = self.labeled_image[self.y0:self.y1, self.x0:self.x1]


            if self.preview: # pragma: no cover
                plt.figure(figsize=(6, 6))
                plt.imshow(self.original_image)
                plt.show()


                plt.figure(figsize=(6,6))
                plt.imshow(self.labeled_image[:,:,[2,1,0]])
                plt.show()

            if self.save: # pragma: no cover
                resave_name = self.filename.replace('.tif', '_cropped.tif')
                cv2.imwrite(resave_name, self.original_image)
                cv2.imwrite('Labeled ' + resave_name, self.labeled_image)

            return self



def _make_parser(): # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--x0', default=0,
                        help='Minimum value along the x-axis to include in the new image.')
    parser.add_argument('--y0', default=0,
                        help='Minimum value along the y-axis to include in the new image.')
    parser.add_argument('--x1', default=None,
                        help='Maximum value along the x-axis to include in the new image.')
    parser.add_argument('--y1', default=None,
                        help='Maximum value along the y-axis to include in the new image.')
    parser.add_argument('--preview', default=False,
                        help='Whether to show a preview (only for a system' +
                        'with access to a display).')
    parser.add_argument('--image', default='C13-PA-PHYS-THK-121_LV_5kv_PC16_WD4.6_20Pa_2kx-A-2.tif',
                        help='Path to the image labels to dilate')
    parser.add_argument('--logging-level', default='INFO', nargs='?',
                        help='How verbose should the information be')
    return parser


def _run_cut(): # pragma: no cover
    args = _make_parser().parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=args.logging_level)
    cutter = Cutter(**vars(args))
    cutter()

if __name__ == "__main__":
    _run_cut()
