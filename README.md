# Interactive 3D Medical Image Segmentation Software
This interactive 3D medical image segmentation software is designed to integrate seamlessly into the medical imaging workflow and can play a pivotal role in enhancing diagnosis, monitoring, and also treatment plans of patients. Its primary purpose is to facilitate detailed and accurate segmentation of medical images (including MR and CT images) using artificial intelligence techniques that do not require extensive expert training and guidance.

The software has been thoughtfully designed for use by clinicians, such as oncologists and radiologists, in addition to imaging technicians and reduces the overhead of medical image segmentation - giving you more time for crucial aspects such as treatment planning.

Major functionalities include:
- Compatibility with standard medical imaging formats; DICOM and NIfTI images and exporting to STL.
- Perform segmentation across axial, coronal, and sagittal views to capture comprehensive anatomical information.
- Utilise segmentation at various resolutions to adapt to different clinical requirements and levels of detail.
- Support segmentation of multiple objects simultaneously and enable cross-view integration for thorough spatial context.
- Leverage immersive 3D visualisation to enhance understanding of complex anatomical structures.
- Easily export segmentation results for further analysis, reporting, or integration into downstream clinical systems.

Furthermore, the software offers flexibility in processing, which can be conducted locally, or remotely, if limited hardware is available.


## Installation
*NOTE: if you come across any problems that are undocumented here, then please visit https://github.com/Chuyun-Shen/SAM_2_Medical_3D/blob/main/INSTALL.md*

*another option if you do come across any undocumented issues: is removing the user manual pdf, although I have not ever needed this step*

Local installation was tested and successfully performed on a Windows 11 machine with NVIDIA RTX 6000 Ada Generation GPU (48GB GPU Memory) (Cuda toolkit 14.4)

Remote installation was tested and successfully performed on: 1) 64 bit Windows 10 machine (16GB RAM memory), 2) Windows 11 machine in room A32 at the School of Computer Science, University of Nottingham.

### Installation to Run Segmentation Model Remotely and Locally

1. unzip `20364425_software.zip` and rename unzipped folder to `dissertation`, or alternatively, unzip `20364425_software.zip` and extract all files to a folder called `dissertation`.
*NOTE:* `dissertation` must be the top-level directory, such that `app.py` (and all files/folders on the same level as `app.py`) are directly inside the dissertation folder, if there are folders between `dissertation` and `app.py`, move their contents up the directory levels and delete the folders.
2. [It is recommended to complete the subsequent steps via anaconda prompt] Move to `dissertation` folder
3. Create conda environment with python version >=3.12: `conda create -n <env_name> python=3.12`
4. Activate the conda environment: `conda activate <env_name>`
5. Install `torch` from URL: `pip install torch torchvision torchaudio --no-build-isolation --index-url https://download.pytorch.org/whl/cu121`
6. Install software requirements: `pip install -r requirements.txt`
7. *NOTE:* if any packages cant be found, try `pip install <name_of_package>` followed by `pip install -r requirements.txt` (this was not necessary at time of testing, but may be necessary if newly installated packages happen to bump the versions of their dependencies). If installation of `hydra-core` fails check that MSVC Build Tools are installed.
8. Navigate to SAM_2_Medical_3D folder: `cd SAM_2_Medical_3D`
9. Install segmentation model (SAM_2_Medical_3D) dependencies: `pip install --prefer-binary --no-build-isolation -e .[demo]` or if that doesn't work try `pip install --prefer-binary --no-build-isolation -e ".[demo]"`
10. Navigate back to dissertation folder: `cd ..`
11. The segmentation model is very large so it cannot be compressed with the rest of the software and satisfy the upload file size limit on Moodle (250MB). The checkpoint is also too large for git therefore it is necessary to fetch the segmentation model checkpoint (850MB). Please use one of the three options below to fetch the checkpoint (`sam2_hiera_large.pt`):
    -  [RECOMMENDED] Download from https://uniofnottm-my.sharepoint.com/:u:/g/personal/psyst6_nottingham_ac_uk/EbRTpkiSK0tOibGLN4fe15AB3qLeBPclU_sJiEN6gShfzw?e=oOFcsB and move into `dissertation/SAM_2_Medical_3D` folder
    - Download from https://drive.google.com/file/d/1BRZJJveK0HF6bd96O4ZKXdXe9pZWqnzX/view?usp=sharing and move into dissertation/SAM_2_Medical_3D folder
    - Run the script at `dissertation/SAM_2_Medical_3D/checkpoints/download_ckpts.sh` to download the checkpoint and move the downloaded `sam2_hiera_large.pt` file into `dissertation/SAM_2_Medical_3D`. This is a bash script so you will need WSL or perhaps git bash (with some extensions)
12. Create a modal account via modal.com/signup (sign up can be completed via a Google or GitHub account)
13. Generate a modal API authentication token by running `python -m modal setup` and follow on-screen instructions
14. Create a modal file storage volume `modal volume create my_adapted_sam_2_medical_3d`
15. Upload the local SAM_2_Medical_3D folder to the newly created volume: `modal volume put my_adapted_sam_2_medical_3d SAM_2_Medical_3D`
16. To run the program (always run the program from the top level directory): LOCALLY: `python app.py` or `python app.py --mode local`. REMOTELY: `python app.py --mode remote`. *NOTE:* any errors you may encounter at this stage can be troubleshooted by visiting https://github.com/Chuyun-Shen/SAM_2_Medical_3D/blob/main/INSTALL.md

When the container is run for the first time it will be slower as configuration is required.

*NOTE:* After clicking segmentation, please ignore any file not found errors related to usage: `modal volume rm` . This is a safeguard to delete and rewrite any files with the same name.

### Installation to Run Segmentation Model Only Remotely
1. unzip `20364425_software.zip` and rename unzipped folder to `dissertation`, or alternatively, unzip `20364425_software.zip` and extract all files to a folder called `dissertation`.
*NOTE:* `dissertation` must be the top-level directory, such that `app.py` (and all files/folders on the same level as `app.py`) are directly inside the dissertation folder, if there are folders between `dissertation` and `app.py`, move their contents up the directory levels and delete the folders.
2. The segmentation model is very large so it cannot be compressed with the rest of the software and satisfy the upload file size limit on Moodle (250MB). The checkpoint is also too large for git therefore it is necessary to fetch the segmentation model checkpoint (850MB). Please use one of the three options below to fetch the checkpoint (`sam2_hiera_large.pt`):
    -  [RECOMMENDED] Download from https://uniofnottm-my.sharepoint.com/:u:/g/personal/psyst6_nottingham_ac_uk/EbRTpkiSK0tOibGLN4fe15AB3qLeBPclU_sJiEN6gShfzw?e=oOFcsB and move into `dissertation/SAM_2_Medical_3D` folder
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

### Installation to Run Segmentation Model Only Locally
Running the program locally will rely on CUDA toolkit installed on the PC. Local testing was completed successfully with CUDA toolkit 14.4. 

Although unexpected; there is the possibility of encountering warnings/errors about os environment variables - please follow instructions on screen to set these.

Installating this software may require MSVC build tools if you do not have them (any .h file not found errors are an indicator that MSVC build tools must be installed. MSVC 2019 will suffice).

1. unzip `20364425_software.zip` and rename unzipped folder to `dissertation`, or alternatively, unzip `20364425_software.zip` and extract all files to a folder called `dissertation`.
*NOTE:* `dissertation` must be the top-level directory, such that `app.py` (and all files/folders on the same level as `app.py`) are directly inside the dissertation folder, if there are folders between `dissertation` and `app.py`, move their contents up the directory levels and delete the folders.
2. [It is recommended to complete the subsequent steps via anaconda prompt] Move to `dissertation` folder
3. Create conda environment with python version >=3.12: `conda create -n <env_name> python=3.12`
4. Activate the conda environment: `conda activate <env_name>`
5. Install `torch` from URL: `pip install torch torchvision torchaudio --no-build-isolation --index-url https://download.pytorch.org/whl/cu121`
6. Install software requirements: `pip install -r requirements.txt`
7. *NOTE:* if any packages cant be found, try `pip install <name_of_package>` followed by `pip install -r requirements.txt` (this was not necessary at time of testing, but may be necessary if newly installated packages happen to bump the versions of their dependencies). If installation of `hydra-core` fails check that MSVC Build Tools are installed.
8. Navigate to SAM_2_Medical_3D folder: `cd SAM_2_Medical_3D`
9. Install segmentation model (SAM_2_Medical_3D) dependencies: `pip install --prefer-binary --no-build-isolation -e .[demo]` or if that doesn't work try `pip install --prefer-binary --no-build-isolation -e ".[demo]"`
10. Navigate back to dissertation folder: `cd ..`
11. The segmentation model is very large so it cannot be compressed with the rest of the software and satisfy the upload file size limit on Moodle (250MB). The checkpoint is also too large for git therefore it is necessary to fetch the segmentation model checkpoint (850MB). Please use one of the three options below to fetch the checkpoint (`sam2_hiera_large.pt`):
    -  [RECOMMENDED] Download from https://uniofnottm-my.sharepoint.com/:u:/g/personal/psyst6_nottingham_ac_uk/EbRTpkiSK0tOibGLN4fe15AB3qLeBPclU_sJiEN6gShfzw?e=oOFcsB and move into `dissertation/SAM_2_Medical_3D` folder
    - Download from https://drive.google.com/file/d/1BRZJJveK0HF6bd96O4ZKXdXe9pZWqnzX/view?usp=sharing and move into dissertation/SAM_2_Medical_3D folder
    - Run the script at `dissertation/SAM_2_Medical_3D/checkpoints/download_ckpts.sh` to download the checkpoint and move the downloaded `sam2_hiera_large.pt` file into `dissertation/SAM_2_Medical_3D`. This is a bash script so you will need WSL or perhaps git bash (with some extensions)
12. To run the program (always run the program from the top level directory):`python app.py` or `python app.py --mode local` *NOTE:* any errors you may encounter at this stage can be troubleshooted by visiting https://github.com/Chuyun-Shen/SAM_2_Medical_3D/blob/main/INSTALL.md








## Testing
To run unit tests via (`unittest`):
1. navigate to top-level directory (`dissertation`)
2. run: `python -m unittest discover`

To do statistical tests on segmentation results (DICE, ASSD):
1. Export the axial view (with overlayed segmentation masks) via the GUI; binary (suitable single-object segmentation) and grayscale (suitable for single-object and multi-object segmentation) is supported for statistical testing. You can save the exported file (.nii) with a custom filename.
2. Open command prompt and navigate to `dissertation` (top-level directory) 
3. Convert the exported .nii file into a folder containing png slices from the axial perspective by running `python nifti_to_png.py <path_to_exported_nii_file>.nii <path_to_output_png_folder> --axis axial`
4. Navigate to the test directory `cd test`
5. Run `python segmentation_results_script.py --input_dir <path_to_output_png_folder> --ground_truth_dir <path_to_ground_truth_folder> --modality <{CT,MR}> --dicom_dir <path_to_dicom_folder_of_ground_truth>` *NOTE:* make sure paths are absolute or relative to `test` directory

## Usage Information
Please see the pdf user manual for more information on usage details (including images).

Start by importing a valid DICOM/NIfTI image (via buttons in the top left corner of the screen). Under the import buttons are tabs (axial, coronal, sagittal) to control how the corresponding plot (on the right of the screen) is manipulated.

Importantly, a “Draft Mode” feature exists to experiment with segmentation results before commiting the results to a more-refined, final version. Draft mode is particularly useful when you want to use multiple slices in the same view as the start point for independent segmentations (for example, when performing multiorgan segmentation with occlusions that prevent segmenting all organs in the same slice). Draft mode can be entered by checking the checkbox with the same name in the sidebar. Whether in or out of draft mode, all of the features below are supported:

In each tab;
- There is an option to choose a pointer colour (each pointer colour corresponds to a different object/region of interest that is to be segmented). 
- As well as the option to make positive and negative clicks -  positive clicks encourage the the segmentation model to expand the mask and negative clicks constrain masks (for this reason, negative clicks are useful correction tools). 
- To make a selection, simply click on the desired area of the corresponding plot on the right side of the screen.  The type of click (positive/negative) and coordinates of the selected point will show in the listbox below the pointer colour option. 
- You may make any number of positive and negative clicks - if you make any mistakes, simply click “Undo”. 
- After making the desired selections, you may segment the image (by clicking the button at the bottom of the tab). 

After completing segmentation, you will see that segmentation masks have been overlayed on the respective view and the 3D mesh view (discussed below) has been generated. You can track the segmentation mask throughout the view by moving the slider at the top of the respective view in the grid on the right of the screen. You may also use the zoom and pan tools (located under the slider) to closely inspect the segmentation results. Also located under the slider are the options to hide/show both segmentation masks and points for cleaner evaluation of the quality of the segmentatoin mask.

If you were previously in draft mode, you can now accept and merge your results from the draft segmentation and keep them.  NOTE: if you have performed segmentation using the same colour (ie on the same object), accepting and merging results from draft segmentation will overwrite your existing segmentations with the draft segmentations.  

The 3D mesh view can be rotated for better visualisation. It tracks and display the 3D mesh model of all segmentation masks across all frames (for this reason, the 3D mesh view in draft mode is distinct from the 3D mesh view used when the user is out of draft mode). 

You may repeat segmentation on the current view; or switch views by changing tabs and then changing the pointer colour etc. 

Global view segmentation (accessible via a checkbox with the same name above the “Segment Image” button) allows users to select points in coronal and sagittal views and see the resulting segmentation mask in axial view. It is important that the axial slice index corresponds to at least one y-coordinate of a selection in coronal or sagittal view. Once finished with global view segmentation, please remember to uncheck the box to prevent unwanted masks appearing in axial view.

At any point, segmentation results corresponding to the view can be exported as binary, grayscale or RGB NIfTI files, by clicking the last button at the bottom of the tab.
Exporting the generated 3D mesh as an STL file is also supported at any point, by clicking Export 3D Mesh View button in the top left corner of the application.

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