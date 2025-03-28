import argparse
import nibabel as nib
import numpy as np
import os
from PIL import Image

def nifti_to_png(nifti_path, output_folder, axis):
    # Load NIfTI image
    img = nib.load(nifti_path)
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()

    # Print the data shape for debugging
    print("Data shape:", data.shape)

    # Normalize the data to 0-255
    data = data - np.min(data)
    data = (data / np.max(data)) * 255
    data = data.astype(np.uint8)

    is_rgb = (data.ndim == 4 and data.shape[-1] == 3)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    axis_lower = axis.lower()
    if is_rgb:
        # For a 4D RGB image with shape (Z, H, W, 3)
        if axis_lower == 'axial':
            slice_axis = 0
        elif axis_lower == 'coronal':
            slice_axis = 1
        elif axis_lower == 'sagittal':
            slice_axis = 2
        else:
            print(f"Axis '{axis}' not implemented. Defaulting to axial slices.")
            slice_axis = 0
            axis_lower = 'axial'
    else:
        # For a 3D grayscale image with shape (H, W, Z)
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
        if is_rgb:
            # For RGB, extract slice based on the chosen axis
            if axis_lower == 'axial':  # slice out along Z
                slice_img = data[i, :, :, :]
            elif axis_lower == 'coronal':  # slice from middle of H dimension
                slice_img = data[:, i, :, :]
            elif axis_lower == 'sagittal':  # slice from W dimension
                slice_img = data[:, :, i, :]
        else:
            if slice_axis == 0:
                slice_img = data[i, :, :]
                slice_img = np.rot90(slice_img, k=1)    # Rotate 90 degrees once, counter-clockwise
                slice_img = np.flipud(slice_img)          # Flip vertically
                slice_img = np.fliplr(slice_img)          # Flip horizontally
            elif slice_axis == 1:
                slice_img = data[:, i, :]
                slice_img = np.rot90(slice_img, k=-1)      # Rotate 90 degrees once, clockwise
            else:  # axial
                slice_img = data[:, :, i]
                slice_img = np.rot90(slice_img, k=1)       # Rotate 90 degrees once, clockwise
                slice_img = np.fliplr(slice_img)           # Flip horizontally

        if is_rgb and slice_img.ndim == 4 and slice_img.shape[0] == 1:
            # Remove the first singleton dimension so that (1, H, W, 3) becomes (H, W, 3)
            slice_img = np.squeeze(slice_img, axis=0)
        
        img_pil = Image.fromarray(slice_img)
        img_pil.save(os.path.join(output_folder, f'{i:05d}.png'))

def main():
    # Example usage: `python nifti_to_png.py input_volume.nii output_png_folder --axis axial`
    parser = argparse.ArgumentParser(description="Convert a NIfTI file to a directory of corresponding PNG files.")
    parser.add_argument("nifti_path", help="Path to the input NIfTI file")
    parser.add_argument("output_folder", help="Directory where PNG images will be saved")
    parser.add_argument("--axis", default="axial", help="Orientation axis to slice the image (default: axial)")
    args = parser.parse_args()

    nifti_to_png(args.nifti_path, args.output_folder, args.axis)

if __name__ == "__main__":
    main()