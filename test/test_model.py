import unittest
import numpy as np
import itk
from model import ImageModel
from unittest.mock import patch

class TestImageModel(unittest.TestCase):

    def setUp(self):
        # Create a dummy 3D numpy array of shape (5, 5, 5)
        self.dummy_array = np.arange(125, dtype=np.int16).reshape((5, 5, 5))
        # Create an ITK image from the NumPy array
        self.itk_image = itk.GetImageFromArray(self.dummy_array)

    def create_model_with_dummy_image(self):
        # Create an ImageModel instance without triggering file I/O,
        # then assign the dummy ITK image.
        model = ImageModel(filename=None)
        model.image = self.itk_image
        return model

    def test_get_slice_axial(self):
        # Axial slicing fixes the z-index (ITK image index 2)
        model = self.create_model_with_dummy_image()
        slice_index = 2
        # Expected: the axial slice from the dummy array corresponds to self.dummy_array[slice_index, :, :]
        expected = self.dummy_array[slice_index, :, :]
        result = model.get_slice(axis=0, slice_index=slice_index)
        np.testing.assert_array_equal(result, expected)

    def test_get_slice_coronal(self):
        # Coronal slicing fixes the image index 1 (y-direction)
        model = self.create_model_with_dummy_image()
        slice_index = 3
        # Expected: the coronal slice corresponds to self.dummy_array[:, slice_index, :] 
        expected = self.dummy_array[:, slice_index, :]
        result = model.get_slice(axis=1, slice_index=slice_index)
        np.testing.assert_array_equal(result, expected)

    def test_get_slice_sagittal(self):
        # Sagittal slicing fixes the image index 0 (x-direction)
        model = self.create_model_with_dummy_image()
        slice_index = 1
        # Expected: the sagittal slice corresponds to self.dummy_array[:, :, slice_index]
        expected = self.dummy_array[:, :, slice_index]
        result = model.get_slice(axis=2, slice_index=slice_index)
        np.testing.assert_array_equal(result, expected)

    def test_get_slice_invalid_axis(self):
        # For an invalid axis, get_slice should raise a ValueError.
        model = self.create_model_with_dummy_image()
        with self.assertRaises(ValueError):
            model.get_slice(axis=3, slice_index=0)

    def test_change_image(self):
        # Verify that change_image updates the filename and calls _load_image.
        model = ImageModel(filename=None)
        with patch.object(model, '_load_image', return_value=None) as mock_load:
            new_file = 'new_file.nii'
            model.change_image(new_file)
            self.assertEqual(model.filename, new_file)
            mock_load.assert_called_once()

if __name__ == "__main__":
    unittest.main()