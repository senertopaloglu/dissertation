import argparse
import nibabel as nib
import numpy as np
import os
from PIL import Image

def nifti_to_jpg(nifti_path, output_folder, axis, downsampled=False):
    # Load NIfTI image
    img = nib.load(nifti_path)
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    data = (data / np.max(data)) * 255
    data = data.astype(np.uint8)

    print("Data shape:", data.shape)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    axis_lower = axis.lower()
    if axis_lower == 'axial':
        slice_axis = 2
    elif axis_lower == 'coronal':
        slice_axis = 1
    elif axis_lower == 'sagittal':
        slice_axis = 0
    else:
        print(f"Axis '{axis}' not implemented. Defaulting to axial slices.")
        slice_axis = 2
        axis_lower = 'axial'

    # Iterate through each slice
    for i in range(data.shape[slice_axis]):
        if slice_axis == 0:
            slice_img = data[i, :, :]
            if not downsampled:
                slice_img = np.rot90(slice_img, k=1)  # Rotate 90 degrees once, counter-clockwise
                slice_img = np.flipud(slice_img)          # Flip vertically
                slice_img = np.fliplr(slice_img)
        elif slice_axis == 1:
            slice_img = data[:, i, :]
            slice_img = np.rot90(slice_img, k=-1)  # Rotate 90 degrees once, clockwise
        else: # axia;
            slice_img = data[:, :, i]
            slice_img = np.rot90(slice_img, k=1)
            slice_img = np.fliplr(slice_img)
        
        img_pil = Image.fromarray(slice_img)
        img_pil.save(os.path.join(output_folder, f'{i:05d}.jpg'))

def main():
    # Example usage: `python nifti_to_jpg.py CHAOS_TEST_CT_3_DICOM_ANON.nii CHAOS_TEST_CT_3_JPG --axis axial`
    parser = argparse.ArgumentParser(description="Convert a NIfTI file to a directory of corresponding JPG files.")
    parser.add_argument("nifti_path", help="Path to the input NIfTI file")
    parser.add_argument("output_folder", help="Directory where JPG images will be saved")
    parser.add_argument("--axis", default="axial", help="Orientation axis to slice the image (default: axial)")
    parser.add_argument("--downsampled", action="store_true",
                        help="Indicate that the NIfTI file is downsampled (slices forced to be square)")
    args = parser.parse_args()

    nifti_to_jpg(args.nifti_path, args.output_folder, args.axis, downsampled=args.downsampled)

if __name__ == "__main__":
    main()

