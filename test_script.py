import os
import re
import argparse
import numpy as np
from PIL import Image
from metrics import dice_coefficient  # ensure this function is defined

def load_mask(filepath):
    """
    Load a PNG image and binarize it.
    """
    from PIL import Image
    img = Image.open(filepath)
    data = np.array(img)
    # If the image is RGB, take one channel (or convert to grayscale) as needed.
    if data.ndim == 3:
        # Using the first channel
        data = data[:, :, 0]
    # Binarize: consider any nonzero pixel as foreground.
    return (data > 0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(
        description="Test segmentation using folders of PNG images and compute the DICE metric."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing segmentation PNG images.")
    parser.add_argument("--ground_truth_dir", required=True, help="Folder containing ground truth PNG images.")
    args = parser.parse_args()

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
        if m:
            slice_idx = int(m.group(1))
            # Construct the expected ground truth filename.
            expected_gt_name = f"liver_GT_{slice_idx:03d}.png"
        else:
            print(f"Could not extract slice index from filename {filename}, skipping.")
            continue

        gt_path = os.path.join(args.ground_truth_dir, expected_gt_name)
        if not os.path.exists(gt_path):
            print(f"Ground truth file {expected_gt_name} not found for {filename}, skipping.")
            continue

        seg_mask = load_mask(input_path)
        gt_mask = load_mask(gt_path)

        dice = dice_coefficient(gt_mask, seg_mask)
        print(f"{filename} vs {expected_gt_name}: DICE Coefficient = {dice}")
        total_dice += dice
        count += 1

    if count > 0:
        print(f"Average DICE Coefficient over {count} slices: {total_dice/count}")
    else:
        print("No matching files to compare.")

if __name__ == '__main__':
    main()