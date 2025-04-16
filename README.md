# Interactive 3D Medical Image Segmentation Software

## Installation
*NOTE: if you come across any problems that are undocumented here, then please visit https://github.com/Chuyun-Shen/SAM_2_Medical_3D/blob/main/INSTALL.md*

### To run the software both remotely and locally



When the container is run for the first time it will be slower as configuration is required.

*NOTE:* After clicking segmentation, please ignore any file not found errors related to usage: `modal volume rm` . This is a safeguard to delete and rewrite any files with the same name.

### To run the software only remotely
1. unzip `20364425_software.zip` and rename unzipped folder to `dissertation`, or alternatively, unzip `20364425_software.zip` and extract all files to a folder called `dissertation`.
*NOTE:* `dissertation` must be the top-level directory, such that `app.py` (and all files/folders on the same level as `app.py`) are directly inside the dissertation folder, if there are folders between `dissertation` and `app.py`, move their contents up the directory levels and delete the folders.
2. The segmentation model is very large so it cannot be compressed with the rest of the software and satisfy the upload file size limit on Moodle (250MB). The checkpoint is also too large for git therefore it is necessary to fetch the segmentation model checkpoint (850MB). Please use one of the three options below to fetch the checkpoint (`sam2_hiera_large.pt`):
    -  [RECOMMENDED] Download from https://uniofnottm-my.sharepoint.com/:u:/g/personal/psyst6_nottingham_ac_uk/Eb_ozfzEOpFJqfH4Fl9G760BNyBUZXS9mTlnW4pOKGzxZQ?e=htRdLt and move into `dissertation/SAM_2_Medical_3D folder`
    - Download from https://drive.google.com/file/d/1BRZJJveK0HF6bd96O4ZKXdXe9pZWqnzX/view?usp=sharing and move into dissertation/SAM_2_Medical_3D folder
    - Run the script at `dissertation/SAM_2_Medical_3D/checkpoints/download_ckpts.sh` to download the checkpoint and move the downloaded `sam2_hiera_large.pt` file into `dissertation/SAM_2_Medical_3D`. This is a bash script so you will need WSL or perhaps git bash (with some extensions)
3. [It is recommended to complete the subsequent steps via anaconda prompt] Move to `dissertation` folder
4. Create conda environment with python version >=3.12: `conda create -n <env_name> python=3.12`
5. Activate the conda environment: `conda activate <env_name>`
6. Install `torch` from URL: `pip install torch torchvision torchaudio --no-build-isolation --index-url https://download.pytorch.org/whl/cu121`
7. Install software requirements: `pip install -r requirements.txt`
8. *NOTE:* if any packages cant be found, try `pip install <name_of_package>` followed by `pip install -r requirements.txt` (this was not necessary at time of testing, but may be necessary if newly installated packages happen to bump the versions of their dependencies). If installation of `hydra-core` fails check that MSVC Build Tools are installed.
9. Create a modal account via modal.com/signup (sign up can be completed via a Google or GitHub account)
10. Generate a modal API authentication token by running `python -m modal setup` and follow on-screen instructions
11. Create a modal file storage volume `modal volume create my_adapted_sam_2_medical_3d`
12. Upload the local SAM_2_Medical_3D folder to the newly created volume: `modal volume put my_adapted_sam_2_medical_3d SAM_2_Medical_3D`
13. Run the program in remote mode: `python app.py --mode remote` (always run this command from `dissertation` directory)

When the container is run for the first time it will be slower as configuration is required.

*NOTE:* After clicking segmentation, please ignore any file not found errors related to usage: `modal volume rm` . This is a safeguard to delete and rewrite any files with the same name.

### To run the program only locally

to run the program (always run the program from the top level directory):
- locally: `python app.py` or `python app.py --mode local`
- remotely: `python app.py --mode remote`









## Testing
To run unit tests via (`unittest`):
1. navigate to top-level directory (`dissertation`)
2. run: `python -m unittest discover`

To do statistical tests on segmentation results (DICE, ASSD):
1. Export the axial view (with overlayed segmentation masks) via the GUI; binary (suitable single-object segmentation) and grayscale (suitable for single-object and multi-object segmentation) is supported for statistical testing. You can save the exported file (.nii) with a custom filename.
2. Open command prompt and navigate to `dissertation` (top-level directory) 
3. Convert the exported .nii file into a folder containing png slices from the axial perspective by running `python nifti_to_png.py <path_to_exported_nii_file>.nii <path_to_output_png_folder> --axis axial`
4. Navigate to the test directory `cd test`
5. Run `python segmentation_results_script.py --input_dir <path_to_output_png_folder> --ground_truth_dir <path_to_ground_truth_folder> --modality <{CT,MR}> --dicom_dir <path_to_dicom_folder_of_ground_truth>`

## Usage Information
When exporting multiobject segmentation masks, the order of "overlaying" masks is as follows:
1. red
2. blue
3. green
4. orange
5. purple
6. cyan
7. magenta
8. teal
9. black
10. grey

This order is also the order of colours in the pointer colour listbox.


## Documentation
To generate documentation via `pydoctor` please follow the instructions below:
1. Navigate to top-level directory
2. Enter the following command:`pydoctor --make-html --html-output docs --project-name "Interactive 3D Medical Imaging Segmentation" --docformat=google views/frame_builder.py views/sidebar.py views/progress_dialog.py views/view.py app.py controller.py dicom_to_nifti.py exporter.py local_handler.py modal_handler.py model.py nifti_to_jpg.py nifti_to_png.py stdout_capture.py`
3. Navigate to docs folder: `cd docs`
4. View documentation:`start index.html`

## Disclaimer
*The segmentation model and GUI application in this project is intended for advisory purposes only and should not be relied upon as a definitive decision-making tool in clinical or healthcare settings.*