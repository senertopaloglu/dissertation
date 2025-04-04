import os
import re
import argparse
import numpy as np
from PIL import Image
import glob
import cv2
from metrics import dice_coefficient, new_assd, DICE

def load_mask(filepath):
    """
    Load a PNG image and binarize it.
    """
    img = Image.open(filepath)
    data = np.array(img)
    # If the image is RGB, take one channel (or convert to grayscale) as needed.
    if data.ndim == 3:
        # Using the first channel
        data = data[:, :, 0]
    # Binarize: consider any nonzero pixel as foreground.
    return (data > 0).astype(np.uint8)

def png_series_reader(dir):
    V = []
    png_file_list=glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list: 
        image = cv2.imread(filename,0)
        V.append(image)
    V = np.array(V,order='A')
    V = V.astype(bool)
    return V

def main():
    parser = argparse.ArgumentParser(
        description="Test segmentation using folders of PNG images and compute the DICE metric."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing segmentation PNG images.")
    parser.add_argument("--ground_truth_dir", required=True, help="Folder containing ground truth PNG images.")
    parser.add_argument("--modality", required=True, choices=['CT', 'MR'], help="Folder containing ground truth PNG images.")
    parser.add_argument("--dicom_dir", required=True, help="Folder containing DICOM images.")
    args = parser.parse_args()

    #####################################
    ######### DICE COEFFICIENT ##########
    #####################################

    # List PNG files in each folder; assuming filenames match between folders.
    input_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith('.png')])
    if not input_files:
        print("No PNG files found in the input folder.")
        return
    
    total_dice = 0.0
    count = 0
    for filename in input_files:
        input_path = os.path.join(args.input_dir, filename)
        # Extract slice number from the segmentation filename.
        # For example, if the segmentation filename is "liver_000.png" then we get 000 as the index.
        m = re.search(r'(\d+)', filename)
        if not m:
            print(f"Could not extract slice index from filename {filename}, skipping.")
            continue

        slice_idx = int(m.group(1))
        gt_path = None
        actual_gt_name = None
        
        if args.modality == 'CT':
            expected_gt_name = f"liver_GT_{slice_idx:03d}.png"
            temp_gt_name = os.path.join(args.ground_truth_dir, expected_gt_name)
            if os.path.exists(temp_gt_name):
                gt_path = temp_gt_name
                actual_gt_name = expected_gt_name
        elif args.modality == 'MR':
            pattern = re.compile(r'IMG-\d{4}-' + f'{slice_idx:05d}' + r'\.png')
            for f in os.listdir(args.ground_truth_dir):
                if pattern.match(f):
                    gt_path = os.path.join(args.ground_truth_dir, f)
                    actual_gt_name = f
                    break
        
        if gt_path is None:
            print(f"Ground truth file not found for {filename} matching modality {args.modality}.")
            continue

        seg_mask = load_mask(input_path)
        gt_mask = load_mask(gt_path)
        dice = dice_coefficient(gt_mask, seg_mask)

        print(f"{filename} vs {actual_gt_name}: DICE Coefficient = {dice}")
        total_dice += dice
        count += 1

    if count > 0:
        print(f"Average DICE Coefficient over {count} slices: {total_dice/count}")
    else:
        print("No matching files to compare.")

    ######################################
    ################ SETUP ###############
    ######################################

    Vref = png_series_reader(args.ground_truth_dir)
    Vseg = png_series_reader(args.input_dir)

    ######################################
    ########### DICE COEFFICIENT #########
    ######################################

    dice_score = DICE(Vref, Vseg)
    print(f"DICE: {dice_score}")

    ######################################
    ############## ASSD ##################
    ######################################

    assd = new_assd(Vref, Vseg, args.dicom_dir)
    print(f"ASSD: {assd}")
    

if __name__ == '__main__':
    main()