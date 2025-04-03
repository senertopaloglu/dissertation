import unittest
import tempfile
import os
import numpy as np
import nibabel as nib
from nifti_to_jpg import nifti_to_jpg

class TestNiftiToJpg(unittest.TestCase):
    def setUp(self):
        # Create dummy 3D data with shape (10, 10, 5)
        self.data = np.arange(10 * 10 * 5, dtype=np.float32).reshape((10, 10, 5))
        self.affine = np.eye(4)
        self.nifti_img = nib.Nifti1Image(self.data, self.affine)
        # Create a temporary file for the NIfTI image and immediately close it.
        temp_file = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
        self.temp_nifti_name = temp_file.name
        temp_file.close()
        nib.save(self.nifti_img, self.temp_nifti_name)

    def tearDown(self):
        os.remove(self.temp_nifti_name)

    def test_default_axial(self):
        # For axial slicing (default) slice_axis is 2, so we expect 5 output images.
        with tempfile.TemporaryDirectory() as output_dir:
            nifti_to_jpg(self.temp_nifti_name, output_dir, axis="axial", downsampled=False)
            files = os.listdir(output_dir)
            self.assertEqual(len(files), 5)
            for i in range(5):
                expected_filename = f"{i:05d}.jpg"
                self.assertIn(expected_filename, files)

    def test_coronal(self):
        # For coronal slicing, slice_axis is 1, so we expect 10 output images.
        with tempfile.TemporaryDirectory() as output_dir:
            nifti_to_jpg(self.temp_nifti_name, output_dir, axis="coronal", downsampled=False)
            files = os.listdir(output_dir)
            self.assertEqual(len(files), 10)

    def test_sagittal(self):
        # For sagittal slicing, slice_axis is 0, so we expect 10 output images.
        with tempfile.TemporaryDirectory() as output_dir:
            nifti_to_jpg(self.temp_nifti_name, output_dir, axis="sagittal", downsampled=False)
            files = os.listdir(output_dir)
            self.assertEqual(len(files), 10)

    def test_downsampled(self):
        # When downsampled==True, the function forces slice_axis to 0.
        # With our dummy data shape (10,10,5), we expect 10 output images.
        with tempfile.TemporaryDirectory() as output_dir:
            nifti_to_jpg(self.temp_nifti_name, output_dir, axis="anything", downsampled=True)
            files = os.listdir(output_dir)
            self.assertEqual(len(files), 10)

if __name__ == "__main__":
    unittest.main()
    