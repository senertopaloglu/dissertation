import nibabel as nib
import numpy as np
import os
from PIL import Image

def nifti_to_jpg(nifti_path, output_folder):
    # Load NIfTI image
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    data = (data / np.max(data)) * 255
    data = data.astype(np.uint8)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each slice
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        img_pil = Image.fromarray(slice_img)
        img_pil.save(os.path.join(output_folder, f'{i:05d}.jpg'))

# Example usage:
nifti_path = 'CHAOS_TEST_CT_3_DICOM_ANON.nii'      # replace with your NIfTI file path
output_folder = 'CHAOS_TEST_CT_3_JPG'        # desired output folder name

nifti_to_jpg(nifti_path, output_folder)
