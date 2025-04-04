import os
import tempfile
import shutil
import unittest
import numpy as np
import nibabel as nib
from PIL import Image

import nifti_to_png

class TestNiftiToPng(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for input and output
        self.test_dir = tempfile.mkdtemp()
        self.input_filename = os.path.join(self.test_dir, "test.nii")
        self.output_dir = os.path.join(self.test_dir, "output")
    
    def tearDown(self):
        # Remove temporary directories
        shutil.rmtree(self.test_dir)

    def create_nifti_file(self, data):
        # Create a Nifti1Image and save it to self.input_filename
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, self.input_filename)

    def check_output_images(self, expected_count, extension=".png"):
        # Check that the output directory contains the expected number of converted image files with the given extension
        files = os.listdir(self.output_dir)
        image_files = [f for f in files if f.endswith(extension)]
        self.assertEqual(len(image_files), expected_count, "Number of output images mismatch")
        # Additionally, try to open one image, and close immediately
        if image_files:
            sample_path = os.path.join(self.output_dir, image_files[0])
            with Image.open(sample_path) as img:
                self.assertIsNotNone(img)

    def run_conversion(self, axis):
        # Run the conversion using nifti_to_png.nifti_to_png()
        # Note: nifti_to_png.py should implement a function with the signature:
        # nifti_to_png(nifti_path, output_folder, axis)
        nifti_to_png.nifti_to_png(self.input_filename, self.output_dir, axis)

    def test_grayscale_axial(self):
        # Create a 3D grayscale volume (H, W, Z)
        H, W, Z = 10, 10, 5
        data = np.random.rand(H, W, Z)
        self.create_nifti_file(data)
        self.run_conversion("axial")
        # For axial, the number of slices equals dimension 2 (Z)
        self.check_output_images(expected_count=Z)

    def test_grayscale_coronal(self):
        # Create a 3D grayscale volume (H, W, Z)
        H, W, Z = 12, 8, 6
        data = np.random.rand(H, W, Z)
        self.create_nifti_file(data)
        self.run_conversion("coronal")
        # For coronal, the number of slices equals dimension 1 (W)
        self.check_output_images(expected_count=W)

    def test_grayscale_sagittal(self):
        # Create a 3D grayscale volume (H, W, Z)
        H, W, Z = 15, 9, 7
        data = np.random.rand(H, W, Z)
        self.create_nifti_file(data)
        self.run_conversion("sagittal")
        # For sagittal, the number of slices equals dimension 0 (H)
        self.check_output_images(expected_count=H)

    def test_rgb_axial(self):
        # Create a 4D RGB volume with shape (Z, H, W, 3)
        Z, H, W, C = 4, 10, 10, 3
        data = np.random.rand(Z, H, W, C)
        self.create_nifti_file(data)
        self.run_conversion("axial")
        # For axial with RGB, number of slices equals dimension 0 (Z)
        self.check_output_images(expected_count=Z, extension=".png")
    
    def test_invalid_axis_defaults_to_axial(self):
        # Create a 3D grayscale volume
        H, W, Z = 10, 10, 5
        data = np.random.rand(H, W, Z)
        self.create_nifti_file(data)
        # Use an invalid axis; the function should default to axial
        self.run_conversion("invalid_axis")
        self.check_output_images(expected_count=Z)

    def test_binary_axial(self):
        # Create a binary 3D grayscale volume (values 0 or 1)
        H, W, Z = 10, 10, 5
        data = np.random.randint(0, 2, (H, W, Z)).astype(np.float64)
        self.create_nifti_file(data)
        self.run_conversion("axial")
        # For axial, the number of slices equals dimension 2 (Z)
        self.check_output_images(expected_count=Z, extension=".png")

    def test_binary_coronal(self):
        # Create a binary 3D grayscale volume (values 0 or 1)
        H, W, Z = 12, 8, 6
        data = np.random.randint(0, 2, (H, W, Z)).astype(np.float64)
        self.create_nifti_file(data)
        self.run_conversion("coronal")
        # For coronal, the number of slices equals dimension 1 (W)
        self.check_output_images(expected_count=W, extension=".png")

    def test_binary_sagittal(self):
        # Create a binary 3D grayscale volume (values 0 or 1)
        H, W, Z = 15, 9, 7
        data = np.random.randint(0, 2, (H, W, Z)).astype(np.float64)
        self.create_nifti_file(data)
        self.run_conversion("sagittal")
        # For sagittal, the number of slices equals dimension 0 (H)
        self.check_output_images(expected_count=H, extension=".png")

    def test_rgb_coronal(self):
        # Create a 4D RGB volume with shape (Z, H, W, 3)
        Z, H, W, C = 4, 10, 10, 3
        data = np.random.rand(Z, H, W, C)
        self.create_nifti_file(data)
        self.run_conversion("coronal")
        # For coronal with RGB, number of slices equals dimension 1 (H)
        self.check_output_images(expected_count=H, extension=".png")

    def test_rgb_sagittal(self):
        # Create a 4D RGB volume with shape (Z, H, W, 3)
        Z, H, W, C = 4, 10, 10, 3
        data = np.random.rand(Z, H, W, C)
        self.create_nifti_file(data)
        self.run_conversion("sagittal")
        # For sagittal with RGB, number of slices equals dimension 2 (W)
        self.check_output_images(expected_count=W, extension=".png")

if __name__ == "__main__":
    unittest.main()