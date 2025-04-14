import os
import argparse
import nibabel as nib
import numpy as np
from model import ImageModel
from modal_handler import segment
from metrics import dice_coefficient

def main():
    parser = argparse.ArgumentParser(description="Test segmentation and compare with ground truth using DICE metric.")
    parser.add_argument("--image", required=True, help="Path to input NIfTI image.")
    parser.add_argument("--ground_truth", required=True, help="Path to NIfTI ground truth segmentation mask.")
    parser.add_argument("--view", required=True, choices=["axial", "coronal", "sagittal"], help="View for segmentation.")
    parser.add_argument("--slice", type=int, required=True, help="Slice index to test.")
    parser.add_argument("--click", required=True, help="Click coordinate in format 'x,y' (e.g., 100,150).")
    
    args = parser.parse_args()
    x, y = map(int, args.click.split(","))
    
    # Map view to an axis number:
    view_to_axis = {"axial": 0, "coronal": 1, "sagittal": 2}
    axis = view_to_axis[args.view]
    
    # Load the input image using our ImageModel
    model = ImageModel(args.image)
    image_slice = model.get_slice(axis, args.slice)
    
    # Prepare one-click input: a dictionary mapping an object ID to a list of (x, y, pos_flag)
    # In this example, we assume object id 1 and a positive click (flag=1)
    one_click_points = {1: [(x, y, 1)]}
    
    # Derive a folder name based on image name and view (this is used by our segmentation function)
    base_name = os.path.basename(args.image).split('.')[0]
    foldername = f"{base_name}_{args.view.upper()}_JPG"
    
    # Call the segmentation function (this should return a dictionary mapping frame index to segmentation masks)
    print("Running segmentation...")
    segmentation_result = segment(image_slice, one_click_points, args.slice, foldername)
    
    # Assume segmentation_result is a dict where key is the frame index—get the mask for object id 1.
    if args.slice not in segmentation_result:
        print(f"Segmentation result for slice {args.slice} not found.")
        return
    # Here we assume that segmentation_result[args.slice] is a dict { object_id: mask }
    if 1 not in segmentation_result[args.slice]:
        print("Segmentation result for object id 1 not found.")
        return
    seg_mask = segmentation_result[args.slice][1]
    
    # Load the ground truth
    gt_img = nib.load(args.ground_truth)
    gt_data = gt_img.get_fdata()
    #gt_slice = extract_slice(gt_data, axis, args.slice)
    
    # Binarise ground truth and segmentation mask if they are not already binary.
    #seg_mask_bin = (seg_mask > 0).astype(np.uint8)
    #gt_mask_bin = (gt_slice > 0).astype(np.uint8)
    
    #dice = dice_coefficient(gt_mask_bin, seg_mask_bin)
    #print(f"DICE Coefficient: {dice:.4f}")

if __name__ == '__main__':
    main()