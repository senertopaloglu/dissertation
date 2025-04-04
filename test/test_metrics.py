import unittest
import numpy as np
from unittest.mock import patch
from metrics import DICE, new_assd

class TestMetrics(unittest.TestCase):
    def test_DICE_identical(self):
        # Test DICE when both arrays are identical (Dice coefficient should be 1)
        V = np.array([[1, 0], [0, 1]])
        self.assertAlmostEqual(DICE(V, V), 1.0)

    def test_DICE_partial(self):
        # Test DICE with partial overlap.
        # Vref = [[1, 0], [0, 1]] and Vseg = [[1, 1], [0, 0]]
        # Intersection sum = 1; Vref.sum()=2, Vseg.sum()=2; Expected Dice = 2*1/4 = 0.5
        Vref = np.array([[1, 0], [0, 1]])
        Vseg = np.array([[1, 1], [0, 0]])
        self.assertAlmostEqual(DICE(Vref, Vseg), 0.5)

    def test_new_assd_identical(self):
        # Test new_assd when both segmentation volumes are identical.
        # With the transformation patched to a trivial conversion,
        # the distances between corresponding border voxels should be zero.
        V = np.zeros((5, 5, 5), dtype=bool)
        V[1:4, 1:4, 1:4] = True  # Create a simple cube segmentation

        dummy_dicom_dir = "dummy_dir"
        # Patch transformToRealCoordinates to simply return the transposed index array.
        # np.where returns a tuple and we convert it to an array of shape (n, 3).
        with patch('metrics.transformToRealCoordinates', side_effect=lambda indexPoints, dicom_dir: np.transpose(indexPoints)):
            self.assertAlmostEqual(new_assd(V, V, dummy_dicom_dir), 0.0)

if __name__ == "__main__":
    unittest.main()