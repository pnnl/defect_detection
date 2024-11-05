"""Testing for the ``defect_detection util``."""
import torch
import os
import numpy as np
import defect_detection


test_folder1 = os.path.abspath(os.path.dirname(__file__))
test_folder = os.path.join(test_folder1, "test_data")

def test_Cutter():
    """Test image cutting along the x and y axes."""

    fname = os.path.join(test_folder, 'test.tiff')

    # None #
    cutter = defect_detection.util.image_cutting.Cutter(fname)
    cut = cutter()
    new_imh, new_imw = cut.original_image.shape[:2]
    assert cut.x1 == cut.imw
    assert cut.y1 == cut.imh
    assert new_imw == cut.imw
    assert new_imh == cut.imh

    # X and Y cuts #
    cutter = defect_detection.util.image_cutting.Cutter(fname, x0=10, y1=50)
    cut = cutter()
    new_imh, new_imw = cut.original_image.shape[:2]
    assert cut.x1 == cut.imw
    assert cut.y1 < cut.imh
    assert new_imw < cut.imw
    assert new_imh < cut.imh


if __name__ == "__main__":
    test_Cutter()
