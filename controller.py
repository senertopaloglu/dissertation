from collections import defaultdict
import sys
import shutil
import os
import subprocess
import contextlib
import threading
from typing import Any, Callable, Optional
from type_aliases import Points, SegmentationResult

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent

from dicom_to_nifti import DicomToNifti
from model import ImageModel
from stdout_capture import StdoutCapture
from progress_dialog import ProgressDialog

import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import * # colors and styles

import cv2
import numpy as np

from exporter import Exporter

try:
    import nibabel as nib
except ImportError:
    print("nibabel is required for saving temporary NIfTI images.")


class SegmentationController:
    """
    The Controller in our MVC. Knows about the Model (ImageModel) and the View,
    and coordinates the logic between them (point selection, undo, redo, etc.).

    Attributes:
        model (ImageModel): The image model responsible for loading and storing the image.
        view (MainView): The view that displays the GUI and manages user interactions.
        points (list): List of final segmentation points.
        line_objects (list): List of line objects drawn on the canvas representing selected points.
        undo_stack (list): Stack to keep track of operations for undo functionality.
        redo_stack (list): Stack to keep track of operations for redo functionality.
    """
    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.exporter = Exporter(self.view)

        self.points = []
        self.line_objects = []
        self.undo_stack = []
        self.redo_stack = []

        # Hook up the callbacks in the View
        self.view.set_on_click_callback(self.on_click)
        self.view.set_undo_callback(self.undo)
        self.view.set_redo_callback(self.redo)
        self.view.set_slice_request_callback(self.handle_slice_request)
        

    def load_image(self, file_path: str, is_nifti: bool = True) -> None:
        """
        Loads the image from the given file path into the model.

        Args:
            file_path (str): The path to the image file that needs to be loaded.
            is_nifti (bool, optional): Indicates whether the file is in NIfTI format.
                If False, the file will be converted from DICOM to NIfTI. Defaults to True.

        Returns:
            None
        """
        if not is_nifti:
            file_path = DicomToNifti().convert(file_path, f"{os.path.split(file_path)[-1]}.nii")
        self.model.change_image(file_path)
        self.view.show_image()
        self.refresh_selection_state()
        self.view.reset_views()
        # clear the mesh view so that it stays empty until segmentation is done.
        self.view.clear_mesh_view()

    def handle_slice_request(self, axis: int, slice_index: int):
        """
        Called by the View whenever a slider is moved to change slices.
        Retrieves the requested slice from the model based on the given axis and slice index.
        
        Args:
            axis (int): The axis number from which to retrieve the slice.
            slice_index (int): The index of the slice for the specified axis.

        Returns:
            np.ndarray: The image slice corresponding to the axis and slice index.
        """
        return self.model.get_slice(axis, slice_index)

    def on_click(self, event: MouseEvent, pointer_color: str, ax: Axes) -> None:
        """
        Callback for mouse clicks on the matplotlib canvas.
        Records the point location and updates the view.

        Args:
            event (MouseEvent): The mouse event containing click coordinates and context.
            pointer_color (str): The color for the pointer click.
            ax (Axes): The matplotlib Axes instance where the click occurred.

        Returns:
            None
        """
        if event.inaxes is None:
            return  # ignore clicks outside the axes

        x, y = int(event.xdata), int(event.ydata)
        color = pointer_color.lower()

        # Use the canvas attribute to determine which tab to update.
        if hasattr(event.canvas, "axis"):
            active_index = event.canvas.axis
            current_tab = self.view.sidebar.tabs[active_index]
        else:
            # Fallback: use the current active tab
            active_index = self.view.sidebar.tabControl.index("current")
            current_tab = self.view.sidebar.tabs[active_index]

        pos_flag = 1 if current_tab.positive_click_var.get() else 0

        if current_tab.positive_click_checkbox["state"] == "disabled":
            current_tab.positive_click_checkbox["state"] = "normal"
        
        if self.view.sidebar.global_draft_mode.get():
            # If in draft mode, do not add the point to the points list.
            current_tab.draft_points.append((x, y, color, pos_flag))
            current_tab.draft_redo_stack.clear()
            self.view.add_point_to_listbox(x, y, pos_flag, True, active_index=active_index, color=color)
        else:
            # Add to points list
            current_tab.points.append((x, y, color, pos_flag))
            # Clear the redo stack if a new point is added
            current_tab.redo_stack.clear()
            self.view.add_point_to_listbox(x, y, pos_flag, False, active_index=active_index)

        # Update the View
        line = self.view.plot_point(x, y, color, ax)
        current_tab.line_objects.append(line)

        # update the state of the positive_click_checkbox based on points for the current color.
        points_current_color = [pt for pt in current_tab.points if pt[2].lower() == color]
        if points_current_color:
            current_tab.positive_click_checkbox.config(state="normal")
        else:
            current_tab.positive_click_var.set(True)
            current_tab.positive_click_checkbox.config(state="disabled")

    def undo(self, is_draft: bool) -> None:
        """
        Undo the last point addition in the active tab.

        Args:
            is_draft (bool): True if undoing a draft point; False for a final point.

        Returns:
            None
        """
        active_index = self.view.sidebar.tabControl.index("current")
        current_tab = self.view.sidebar.tabs[active_index]

        if is_draft:
            if current_tab.draft_points and current_tab.draft_points_listbox.size() > 0:
                removed_point = current_tab.draft_points.pop()
                current_tab.draft_redo_stack.append(removed_point)
                self.view.remove_last_point_from_listbox(is_draft)
                if current_tab.draft_line_objects:
                    line = current_tab.draft_line_objects.pop()
                    line.remove()
        else:
            if current_tab.points and current_tab.final_points_listbox.size() > 0:
                removed_point = current_tab.points.pop()
                current_tab.redo_stack.append(removed_point)
                self.view.remove_last_point_from_listbox(is_draft)
                if current_tab.line_objects:
                    line = current_tab.line_objects.pop()
                    line.remove()

        if active_index == 0:
            canvas = self.view.axial_view.canvas
            label = "Axial View"
            axis = 0
        elif active_index == 1:
            canvas = self.view.coronal_view.canvas
            label = "Coronal View"
            axis = 1
        elif active_index == 2:
            canvas = self.view.sagittal_view.canvas
            label = "Sagittal View"
            axis = 2
        slice_idx = int(canvas.slider.get())
        self.view._update_slice(canvas.figure.axes[0], canvas, axis, slice_idx)

    def redo(self, is_draft: bool) -> None:
        """
        Redo the last undone point addition in the active tab.

        Args:
            is_draft (bool): True if redoing a draft point; False for a final point.

        Returns:
            None
        """
        active_index = self.view.sidebar.tabControl.index("current")
        current_tab = self.view.sidebar.tabs[active_index]

        canvas = self.view._get_canvas(active_index)
        ax = canvas.figure.axes[0]

        if is_draft:
            if current_tab.draft_redo_stack:
                restored_point = current_tab.draft_redo_stack.pop()
                current_tab.draft_points.append(restored_point)
                current_tab.draft_undo_stack.append(restored_point)
                x, y, color, pos_flag = restored_point
                self.view.add_point_to_listbox(x, y, pos_flag, True, active_index=active_index, color=color)
                line = self.view.plot_point(x, y, color, ax)
                current_tab.draft_line_objects.append(line)
        else:
            if current_tab.redo_stack:
                restored_point = current_tab.redo_stack.pop()
                current_tab.points.append(restored_point)
                current_tab.undo_stack.append(restored_point)
                x, y, color, pos_flag = restored_point
                self.view.add_point_to_listbox(x, y, pos_flag, False, active_index=active_index, color=color)
                line = self.view.plot_point(x, y, color, ax)
                current_tab.line_objects.append(line)

        if active_index == 0:
            canvas = self.view.axial_view.canvas
            label = "Axial View"
            axis = 0
        elif active_index == 1:
            canvas = self.view.coronal_view.canvas
            label = "Coronal View"
            axis = 1
        elif active_index == 2:
            canvas = self.view.sagittal_view.canvas
            label = "Sagittal View"
            axis = 2
        
        slice_idx = int(canvas.slider.get())
        self.view._update_slice(ax, canvas, axis, slice_idx)
    
    def refresh_selection_state(self) -> None:
        """
        Resets the selection state for all tabs.

        Clears both the draft and final points listboxes, removes any drawn line objects from the canvas,
        resets the corresponding points lists and undo/redo stacks for each tab, and disables the positive click
        checkboxes. Also disables the "show-points" checkboxes in each view if no points remain.

        Returns:
            None
        """
        for tab in self.view.sidebar.tabs:
            # Clear the draft listbox
            if tab.draft_points_listbox:
                tab.draft_points_listbox.delete(0, "end")
            # Clear the listbox
            if tab.final_points_listbox:
                tab.final_points_listbox.delete(0, "end")
            
            # Remove any drawn draft line objects from the canvas.
            while tab.draft_line_objects:
                line = tab.draft_line_objects.pop()
                try:
                    line.remove()
                except Exception as e:
                    print(f"Error removing line: {e}")
            # Remove any drawn line objects from the canvas.
            while tab.line_objects:
                line = tab.line_objects.pop()
                try:
                    line.remove()
                except Exception as e:
                    print(f"Error removing line: {e}")

            # Reset the lists and stacks.
            tab.draft_points = []
            tab.points = []
            tab.draft_undo_stack = []
            tab.undo_stack = []
            tab.draft_redo_stack = []
            tab.redo_stack = []
            if hasattr(tab, "positive_click_checkbox"):
                tab.positive_click_checkbox.config(state = "disabled")

        # disable the show-points checkbox in each view if no points remain
        if hasattr(self.view, "axial_view") and hasattr(self.view.axial_view.canvas, "show_points_checkbox"):
            self.view.axial_view.canvas.show_points_checkbox.config(state="disabled")
        if hasattr(self.view, "coronal_view") and hasattr(self.view.coronal_view.canvas, "show_points_checkbox"):
            self.view.coronal_view.canvas.show_points_checkbox.config(state="disabled")
        if hasattr(self.view, "sagittal_view") and hasattr(self.view.sagittal_view.canvas, "show_points_checkbox"):
            self.view.sagittal_view.canvas.show_points_checkbox.config(state="disabled")

    def segment_image(
        self, 
        slices: np.ndarray, 
        points: Points, 
        frame_idx: int, 
        axis_str_suffix: str, 
        custom_filename: Optional[str] = None, 
        completion_callback: Optional[Callable[[Any], None]]=None, 
        downsampled: bool = False, 
        multi_resolution: bool = False, 
        is_first: bool = False, 
        is_final: bool = False, 
        progress_title: Optional[str] = None,
        is_global: bool = False
    ) -> None:
        """
        Prepares the necessary filenames and local folders, converts the input NIfTI image
        to JPG (with optional downsampling), and then invokes the segmentation process with progress 
        reporting. In remote mode, it handles the file transfers using Modal commands. Finally, it calls 
        the segmentation process wrapped in a progress dialog.

        Args:
            slices (np.ndarray): A numpy array representing the image volume slices.
            points (Points): A collection of user-selected points for segmentation.
            frame_idx (int): Index of the current frame/slice to segment.
            axis_str_suffix (str): A string representing the axis (e.g., "AXIAL", "CORONAL", or "SAGITTAL")
                used for file naming and processing.
            custom_filename (Optional[str], optional): A custom filename prefix. If provided, it is used to
                derive the folder name. Defaults to None.
            completion_callback (Optional[Callable[[Any], None]], optional): A callback function that is invoked
                upon segmentation completion. Defaults to None.
            downsampled (bool, optional): Flag specifying if segmentation should be performed on a downsampled image.
                Defaults to False.
            multi_resolution (bool, optional): If True, enables multi-resolution segmentation. Defaults to False.
            is_first (bool, optional): Indicates if this is the first iteration in a multi-resolution segmentation.
                Defaults to False.
            is_final (bool, optional): Indicates if this is the final iteration in a multi-resolution segmentation.
                Defaults to False.
            progress_title (Optional[str], optional): Title for the progress dialog. Defaults to None.
            is_global (bool, optional): If True, segmentation is performed globally across multiple views.
                Defaults to False.

        Returns:
            None
        """
        # get filename (without prefix and file format)
        filename = custom_filename if custom_filename is not None else self.model.filename
        if custom_filename is not None:
            filename = f"{custom_filename}_{axis_str_suffix}"
            foldername = f"{filename}_JPG"
        else:
            filename = os.path.splitext(os.path.basename(self.model.filename))[0]
            foldername = f"{filename}_{axis_str_suffix}_JPG"

        local_folder = f"./temp/{foldername}"

        # Step 0: Delete local folder if it exists
        if os.path.exists(local_folder):
            shutil.rmtree(local_folder)

        # Step 1: Convert NIfTI to JPG and create local folder
        # sys.executable will auto select the running Python interpreter (supports cross-platform)
        if downsampled:
            subprocess.run([sys.executable, "nifti_to_jpg.py", f"./temp/{filename}.nii", local_folder, "--axis", axis_str_suffix.lower(), "--downsampled"], check=True)
        else:
            subprocess.run([sys.executable, "nifti_to_jpg.py", f"{filename}.nii", local_folder, "--axis", axis_str_suffix.lower()], check=True)
        
        if self.is_remote:
            # Step 2: Delete folder in modal
            # do not check=True because no graceful handling if folder not exists in modal volume
            subprocess.run(["modal", "volume", "rm", "-r", "my_adapted_sam_2_medical_3d", f"SAM_2_Medical_3D/frames/{foldername}"])

            # Step 3: Create folder and transfer files to modal
            subprocess.run(["modal", "volume", "put", "my_adapted_sam_2_medical_3d", local_folder, f"SAM_2_Medical_3D/frames/{foldername}"], check=True)

            # Step 4: Delete local JPG folder and contents
            shutil.rmtree(local_folder)
        
        self.run_segmentation_with_progress(slices, points, frame_idx, axis_str_suffix, foldername, completion_callback, multi_resolution, is_first, is_final, progress_title, is_global)
    
    def run_segmentation_with_progress(
        self,
        slices: np.ndarray, 
        points: Points, 
        frame_idx: int, 
        axis_str_suffix: str, 
        foldername: str, 
        completion_callback: Optional[Callable[[Any], None]] = None, 
        multi_resolution: bool = False, 
        is_first: bool = False, 
        is_final: bool = False, 
        progress_title: Optional[str] = None,
        is_global: bool = False
    ) -> None:
        """
        Runs the segmentation process while displaying a progress dialog.

        This method creates and displays a progress dialog, then starts the segmentation process in a 
        separate thread. It redirects stdout to capture progress updates and handles both local and remote 
        segmentation. Upon completion, it cleans up temporary files and invokes the provided completion callback,
        or updates the view with the segmentation result.

        Args:
            slices (np.ndarray): A numpy array representing the image volume slices.
            points (Points): A collection of user-selected points for segmentation.
            frame_idx (int): Index of the current frame/slice to segment.
            axis_str_suffix (str): A string representing the axis (e.g., "AXIAL", "CORONAL", or "SAGITTAL")
                used for file naming and processing.
            foldername (str): The name of the folder where the intermediate JPG images are stored.
            completion_callback (Optional[Callable[[Any], None]], optional): A callback function invoked upon 
                segmentation completion. Defaults to None.
            multi_resolution (bool, optional): Flag indicating if multi-resolution segmentation is enabled.
                Defaults to False.
            is_first (bool, optional): Indicates if this is the first iteration in a multi-resolution segmentation.
                Defaults to False.
            is_final (bool, optional): Indicates if this is the final iteration in a multi-resolution segmentation.
                Defaults to False.
            progress_title (Optional[str], optional): Title for the progress dialog. Defaults to None.
            is_global (bool, optional): If True, segmentation is performed globally across multiple views.
                Defaults to False.

        Returns:
            None
        """
        # create progress dialog as a child of the main view
        progress_dialog = ProgressDialog(self.view, title=progress_title if progress_title is not None else "Progress")
        progress_dialog.top.transient(self.view)
        progress_dialog.update_progress() # start polling the progress queue

        # runs in a separate thread
        def run_segmentation() -> None:
            capture = StdoutCapture(sys.stdout, progress_dialog.progress_queue)
            try:
                # redirect stdout to capture progress messages
                with contextlib.redirect_stdout(capture):
                    if self.is_remote:
                        import modal_handler
                        volume_segments = modal_handler.segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final, is_global)
                        # signal completion
                        progress_dialog.progress_queue.put(("done", 100))
                    else:
                        import local_handler
                        volume_segments = local_handler.segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final, is_global)
                        # signal completion
                        progress_dialog.progress_queue.put(("done", 100))
                
                # once segmentation finished, update the view on the main thread
                if self.is_remote:
                    local_folder = f"./temp/{foldername}"
                    if os.path.exists(local_folder):
                        shutil.rmtree(local_folder)

                self.view.after(0, lambda: finish_segmentation(volume_segments))
            except Exception as e:
                self.view.after(0, lambda: tk.messagebox.showerror("Segmentation Error", f"An exception occurred in the container:\n{e}"))
    
        def finish_segmentation(volume_segments: SegmentationResult) -> None:
            progress_dialog.close()
            if completion_callback:
                completion_callback(volume_segments)
            else:
                self.view.show_segmentation(volume_segments, axis_str_suffix)
                self.view.update_mesh_view()
        
        seg_thread = threading.Thread(target=run_segmentation)
        seg_thread.start()
    
    # multiresolution segmentation routine
    def multiresolution_segmentation(self, tab: int) -> None:
        """
        Performs multi-resolution segmentation on the image for a specific view.

        This method retrieves the original image and its corresponding slice based on the active tab.
        It then downsamples the image volume iteratively and performs segmentation at each resolution.
        Segmentation masks from the lower resolution are used to compute new seeds for further iterations.
        Debug figures may be displayed for verification of the downsampled masks and recomputed seed points.

        Args:
            tab (int): The index of the tab on which segmentation is performed (0 for axial, 1 for coronal,
                2 for sagittal).

        Returns:
            None

        Raises:
            tk.messagebox.showerror: If no image is loaded or if the image dimensions are too small for the 
                desired downsampling resolution.
        """
        if self.model.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        full_image = np.asarray(self.model.image)

        # get the original image as a numpy array
        axis = self.view.sidebar.tabs.index(tab)
        if axis == 0 or self.view.sidebar.global_segmentation_var.get():
            slice_idx = int(self.view.axial_view.canvas.slider.get())
            original_image = full_image[slice_idx, :, :]
            original_h, original_w = original_image.shape
            axis_str_suffix = "AXIAL"
        elif axis == 1:
            slice_idx = int(self.view.coronal_view.canvas.slider.get())
            original_image = full_image[:, slice_idx, :]
            original_h, original_w = original_image.shape
            axis_str_suffix = "CORONAL"
        elif axis == 2:
            slice_idx = int(self.view.sagittal_view.canvas.slider.get())
            original_image = full_image[:, :, slice_idx]
            original_h, original_w = original_image.shape
            axis_str_suffix = "SAGITTAL"
        else:
            messagebox.showerror("Error", "Invalid axis.")
            return

        start_res = 128
        if original_h < start_res or original_w < start_res:
            messagebox.showerror("Downsampling Error", "Image is too small to downsample to 128x128.")
            return

        current_res = start_res

        if not hasattr(tab, "points") or not tab.points:
            messagebox.showerror("Error", "Please select at least one point on the image for segmentation.")
            return
        
        color_mapping = {
            "red": 1,
            "blue": 2,
            "green": 3,
            "orange": 4,
            "purple": 5,
            "cyan": 6,
            "magenta": 7,
            "teal": 8,
            "black": 9,
            "gray": 10
        }
        seeds = {}
        if self.view.sidebar.global_segmentation_var.get():
            for i, tab in enumerate(self.view.sidebar.tabs):
                for point in tab.points:
                    x, y, color, pos_flag = point
                    obj_id = color_mapping.get(color.lower(), 1)
                    temp_scale_x = int(x * current_res / original_w)
                    temp_scale_y = int(y * current_res / original_h)
                    curr_slice = int(self.view.axial_view.canvas.slider.get())
                    if obj_id not in seeds:
                        seeds[obj_id] = defaultdict(list)
                    if i == 0:
                        seeds[obj_id][curr_slice].append((temp_scale_x, temp_scale_y, pos_flag))
                    elif i == 1:
                        seeds[obj_id][curr_slice].append((temp_scale_x, curr_slice, pos_flag))
                    elif i == 2:
                        seeds[obj_id][curr_slice].append((curr_slice, temp_scale_x, pos_flag))
        else:
            for (x, y, color, pos_flag) in tab.points:
                obj_id = color_mapping.get(color.lower(), 1)
                temp_scale_x = int(x * current_res / original_w)
                temp_scale_y = int(y * current_res / original_h)
                seeds.setdefault(obj_id, []).append((temp_scale_x, temp_scale_y, pos_flag))

        original_click_counts = {}
        if self.view.sidebar.global_segmentation_var.get():
            for obj_id, pts_dict in seeds.items():
                count = sum(1 for pt in pts_dict.get(slice_idx, []) if pt[2] == 1)
                original_click_counts[obj_id] = max(5,count)
        else:
            for obj_id, pts in seeds.items():
                count = sum(1 for pt in pts if pt[2] == 1)
                original_click_counts[obj_id] = max(5,count)

        # helper: downsample the entire volume for the selected view
        def downsample_volume(res: int) -> np.ndarray:
            if axis == 0 or self.view.sidebar.global_segmentation_var.get():
                N = full_image.shape[0]
                vol = np.zeros((N, res, res), dtype=full_image.dtype)
                for i in range(N):
                    vol[i] = cv2.resize(full_image[i,:,:], (res,res), interpolation=cv2.INTER_AREA)
                return vol
            elif axis == 1:
                H = full_image.shape[1]
                vol = np.zeros((H, res, res), dtype=full_image.dtype)
                for j in range(H):
                    vol[j] = cv2.resize(full_image[:,j,:], (res,res), interpolation=cv2.INTER_AREA)
                return vol
            elif axis == 2:
                W = full_image.shape[2]
                vol = np.zeros((W, res, res), dtype=full_image.dtype)
                for k in range(W):
                    vol[k] = cv2.resize(full_image[:,:,k], (res, res), interpolation=cv2.INTER_AREA)
                return vol

        # helper: compute upsample size based on current resolution and original size
        def upsample_target(current: int, original: int) -> int:
            standards = [128,256,512,1024]
            for s in standards:
                if s > current and s <= original:
                    return s
            return original

        most_recent_video_segments = None

        # This inner function performs one iteration.
        def iteration(resolution: int, seeds: Points) -> None:
            if resolution > max(original_h, original_w):
                def show_final():
                    nonlocal most_recent_video_segments
                    if most_recent_video_segments:
                        self.view.show_segmentation(most_recent_video_segments, axis_str_suffix)
                        self.view.update_mesh_view()
                self.view.after(100, show_final)
                return
            
            downsampled_volume = downsample_volume(resolution)

            points = seeds

            filename_without_extension = os.path.splitext(os.path.basename(self.model.filename))[0]

            temp_filename = f"downsampled_{filename_without_extension}_{resolution}"
            os.makedirs("./temp", exist_ok=True)
            nii_img = nib.Nifti1Image(downsampled_volume, affine=np.eye(4))
            nib.save(nii_img, f"./temp/{temp_filename}_{axis_str_suffix}.nii")

            # Completion callback to capture segmentation result.
            result_container = {}
            def completion_callback(volume_segments):
                result_container['volume_segments'] = volume_segments
            
            # Call segment_image on the downsampled volume.
            # Save the downsampled image temporarily as a NIfTI file.
            is_first = start_res == resolution
            is_final = resolution == min(original_h, original_w)
            self.segment_image(
                downsampled_volume,
                points,
                frame_idx=slice_idx if not self.view.sidebar.global_segmentation_var.get() else int(self.view.axial_view.canvas.slider.get()),
                axis_str_suffix=axis_str_suffix if not self.view.sidebar.global_segmentation_var.get() else "AXIAL",
                custom_filename=temp_filename,
                completion_callback=completion_callback,
                downsampled=True,
                multi_resolution=True,
                is_first=is_first,
                is_final=is_final,
                progress_title=f"Progress {resolution}x{resolution}",
                is_global=self.view.sidebar.global_segmentation_var.get()
            )

            def check_result() -> None:
                nonlocal most_recent_video_segments
                if 'volume_segments' in result_container:
                    volume_segments = result_container['volume_segments']
                    most_recent_video_segments = result_container['volume_segments']
                    
                    os.remove(f"./temp/{temp_filename}_{axis_str_suffix}.nii")
                    
                    # if slice_idx not in volume_segments:
                    #     messagebox.showerror("Error", f"No segmentation found for slice {slice_idx}.")
                    #     return
                    
                    nonlocal upsampled_masks
                    upsampled_masks = {}

                    for obj_id, seg_mask_down in volume_segments[slice_idx].items():
                        if seg_mask_down.ndim == 3 and seg_mask_down.shape[0] == 1:
                            seg_mask_down = seg_mask_down[0]
                        if seg_mask_down.dtype == np.bool_:
                            seg_mask_down = seg_mask_down.astype(np.uint8)
                        upsampled_masks[obj_id] = cv2.resize(seg_mask_down, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

                        # for debugging: display upsampled mask in temorary popup figure
                        plt.figure(figsize=(12, 8))
                        plt.title(f"Upsampled Segmentation Mask overlayed on original image with seeds")
                        if axis == 0 or self.view.sidebar.global_segmentation_var.get():
                            plt.imshow(downsampled_volume[slice_idx, :, :])
                        elif axis == 1:
                            plt.imshow(downsampled_volume[:, slice_idx, :])
                        else:
                            plt.imshow(downsampled_volume[:, :, slice_idx])
                        plt.imshow(seg_mask_down, alpha=0.5)
                        if isinstance(seeds[obj_id], dict):
                            pts_list = seeds[obj_id].get(slice_idx, [])
                        else:
                            pts_list = seeds[obj_id]
                        mpl_color = {1:"r",2:"b",3:"g",4:"orange",5:"purple",6:"c",7:"m",8:"teal",9:"k",10:"gray"}.get(obj_id, "r")
                        xs = [int(p[0]) for p in pts_list]
                        ys = [int(p[1]) for p in pts_list] 
                        plt.gca().scatter(xs, ys, color=mpl_color, marker='*', s=200, edgecolor='white', linewidth=1.25)
                        plt.show(block=True)

                    # auto compute new seeds for each object
                    new_seeds = {}
                    for obj_id in seeds.keys():
                        seg_mask_down = volume_segments[slice_idx].get(obj_id)
                        if seg_mask_down is not None:
                            if seg_mask_down.ndim == 3 and seg_mask_down.shape[0] == 1:
                                seg_mask_down = seg_mask_down[0]
                            mask_uint8 = (seg_mask_down > 0).astype(np.uint8) * 255
                            
                            # erode mask to get inner contour
                            kernel = np.ones((3, 3), np.uint8)
                            eroded_mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=2)

                            # find contours of the eroded mask
                            contours, _ = cv2.findContours(eroded_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                contour = max(contours, key=cv2.contourArea)
                                sampled_points = []

                                # sample N points evenly along the contour
                                base_N = original_click_counts.get(obj_id, 5)
                                if resolution == 128:
                                    N = base_N
                                elif resolution == 256:
                                    N = base_N + 2
                                else:
                                    N = base_N + 3

                                if len(contour) >= N:
                                    step = len(contour) // N
                                    selected = contour[::step][:N]
                                else:
                                    selected = contour
                                
                                for pt in selected:
                                    cx, cy = pt[0] # contour points have shape (N, 1, 2)
                                    # upscale to original image size
                                    cx = int(cx * 2)
                                    cy = int(cy * 2)
                                    if self.view.sidebar.global_segmentation_var.get():
                                        new_seeds.setdefault(obj_id, defaultdict(list))
                                        new_seeds[obj_id][int(self.view.axial_view.canvas.slider.get())].append((cx, cy, 1))
                                    else:
                                        new_seeds.setdefault(obj_id, []).append((cx, cy, 1))
                            else:
                                # if no contours found then use the original seed
                                new_seeds[obj_id] = seeds[obj_id]
                        else:
                            # if segmentation mask is not returned then use the original seed
                            new_seeds[obj_id] = seeds[obj_id]
                    
                    # for debugging: display new seed in temorary popup figure
                    plt.figure(figsize=(12, 8))
                    plt.title(f"New Seed Points")
                    plt.imshow(original_image)
                    for obj_id, pts in new_seeds.items():
                        mpl_color = {1:"r",2:"b",3:"g",4:"orange",5:"purple",6:"c",7:"m",8:"teal",9:"k",10:"gray"}.get(obj_id, "r")
                        if isinstance(pts, dict):
                            pts_list = pts.get(slice_idx, [])
                        else:
                            pts_list = pts
                        xs = [int(p[0]) for p in pts_list]
                        ys = [int(p[1]) for p in pts_list] 
                        plt.gca().scatter(xs, ys, color=mpl_color, marker='*', s=200, edgecolor='white', linewidth=1.25)
                    plt.show(block=True)

                    new_res = resolution * 2
                    iteration(new_res, new_seeds)
                else:
                    self.view.after(100, check_result)
            
            check_result()
        
        # initialise variable to hold upsampled masks from last iteration
        upsampled_masks = {}
        iteration(current_res, seeds)

    def merge_drafts(self, active_index: int = None) -> None:
        """
        Merges draft points into regular points for the current active tab.
        If a draft point has the same color as an existing final point,
        it overwrites that final point. Afterwards, resets draft points,
        listbox, undo/redo stacks and draft line objects.
        
        Args:
            active_index (int, optional): The index of the active tab to merge. If None, the currently 
                selected tab is used. Defaults to None.

        Returns:
            None
        """
        if active_index is None:
            active_index = self.view.sidebar.tabControl.index("current")
        current_tab = self.view.sidebar.tabs[active_index]

        from collections import defaultdict

        final_points_dict = defaultdict(list)
        for pt in current_tab.points:
            final_points_dict[pt[2].lower()].append(pt)
        
        draft_points_dict = defaultdict(list)
        for pt in current_tab.draft_points:
            draft_points_dict[pt[2].lower()].append(pt)

        merged_points = []
        colors = set(final_points_dict.keys()).union(set(draft_points_dict.keys()))
        for color in colors:
            # If any draft points were recorded for this color, use them to overwrite
            if draft_points_dict[color]:
                merged_points.extend(draft_points_dict[color])
            else:
                merged_points.extend(final_points_dict[color])
        current_tab.points = merged_points
        
        # Clear and repopulate the final points listbox.
        if current_tab.final_points_listbox:
            current_tab.final_points_listbox.delete(0, "end")
            for pt in merged_points:
                x, y, color, pos_flag = pt
                prefix = "Positive click" if pos_flag else "Negative click"
                current_tab.final_points_listbox.insert("end", f"{prefix} at ({x},{y})")
                idx = current_tab.final_points_listbox.size() - 1
                try:
                    current_tab.final_points_listbox.itemconfig(idx, {'fg': color.lower()})
                except Exception as e:
                    print(f"Error setting color in listbox: {e}")
        
        # Reset all draft state: points, listbox, undo/redo stacks and drawn lines.
        current_tab.draft_points = []
        if current_tab.draft_points_listbox:
            current_tab.draft_points_listbox.delete(0, "end")
        current_tab.draft_undo_stack = []
        current_tab.draft_redo_stack = []
        while current_tab.draft_line_objects:
            line = current_tab.draft_line_objects.pop()
            current_tab.line_objects.append(line)
            try:
                line.remove()
            except Exception as e:
                print(f"Error removing draft line: {e}")
        
        # merge segmentation masks
        if active_index == 0:
            base_mask = self.view.axial_view_mask or {}
            draft_mask = self.view.draft_axial_view_mask or {}
        elif active_index == 1:
            base_mask = self.view.coronal_view_mask or {}
            draft_mask = self.view.draft_coronal_view_mask or {}
        elif active_index == 2:
            base_mask = self.view.sagittal_view_mask or {}
            draft_mask = self.view.draft_sagittal_view_mask or {}
        else:
            base_mask = {}
            draft_mask = {}
        
        # for every slice in the draft mask, update (or add) each object mask.
        for slice_idx, obj_masks in draft_mask.items():
            if slice_idx not in base_mask:
                base_mask[slice_idx] = obj_masks.copy()
            else:
                base_mask[slice_idx].update(obj_masks)
        
        # store the merged masks back and reinitialise the draft masks.
        if active_index == 0:
            self.view.axial_view_mask = base_mask
            self.view.draft_axial_view_mask = {}
        elif active_index == 1:
            self.view.coronal_view_mask = base_mask
            self.view.draft_coronal_view_mask = {}
        elif active_index == 2:
            self.view.sagittal_view_mask = base_mask
            self.view.draft_sagittal_view_mask = {}
        
        # merge 3d mesh results
        base_mesh = self.view.last_result or {}
        draft_mesh = self.view.last_draft_result or {}
        for obj_id, mesh_data in draft_mesh.items():
            base_mesh[obj_id] = mesh_data # overwrite or add the draft result
        self.view.last_result = base_mesh
        self.view.last_draft_result = None

        self.view.sidebar.global_draft_mode.set(False)

        # Update the current view slice to reflect merged points.
        canvas = self.view._get_canvas(active_index)
        label = "Axial View" if active_index == 0 else "Coronal View" if active_index == 1 else "Sagittal View"
        slice_idx = int(canvas.slider.get())
        self.view._update_slice(canvas.figure.axes[0], canvas, active_index, slice_idx)
        self.view.update_mesh_view()
        self.view.update_all_views()
    
    def merge_all_drafts(self) -> None:
        """
        Merges draft points, segmentation masks, and mesh results across all views.

        This method iterates over all tabs (Axial, Coronal, and Sagittal), calling merge_drafts for each tab 
        to consolidate draft data into the final segmentation results.

        Returns:
            None
        """
        # Merge draft points from all tabs into regular points
        for idx in [0,1,2]:
            self.merge_drafts(idx)
    
    def export_3d_mesh(self):
        self.exporter.export_3d_mesh()

    def export_view_with_mask(self):
        self.exporter.export_view_with_mask()