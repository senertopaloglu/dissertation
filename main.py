import argparse
import gui

parser = argparse.ArgumentParser(description="Process A 2D Slice Of A 3D Image.")
parser.add_argument("input_3D_image_filename") # both .nii and .nii.gz work
args = parser.parse_args()
image_filename = args.input_3D_image_filename

gui.main(image_filename)