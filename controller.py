from collections import defaultdict
import sys
import shutil
import os
import subprocess
import contextlib
import threading

from dicom_to_nifti import DicomToNifti
from stdout_capture import StdoutCapture
from progress_dialog import ProgressDialog

import tkinter as tk
from tkinter import messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import * # colors and styles

import cv2
import numpy as np
try:
    import nibabel as nib
except ImportError:
    print("nibabel is required for saving temporary NIfTI images.")


class SegmentationController:
    """
    The Controller in our MVC. Knows about the Model (ImageModel) and the View,
    and coordinates the logic between them (point selection, undo, redo, etc.).
    """
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # Store the points, line objects, and stacks for undo/redo
        self.points = []
        self.line_objects = []
        self.undo_stack = []
        self.redo_stack = []

        # Hook up the callbacks in the View
        self.view.set_on_click_callback(self.on_click)
        self.view.set_undo_callback(self.undo)
        self.view.set_redo_callback(self.redo)
        self.view.set_slice_request_callback(self.handle_slice_request)
        

    def load_image(self, file_path, is_nifti=True):
        """
        Loads the image from the given file path into the model.
        """
        if not is_nifti:
            file_path = DicomToNifti().convert(file_path, None)
        self.model.change_image(file_path)
        self.view.show_image()
        self.refresh_selection_state()
        self.view.reset_views()
        # clear the mesh view so that it stays empty until segmentation is done.
        self.view.clear_mesh_view()

    def handle_slice_request(self, axis, slice_index):
        """
        Called by the View whenever a slider is moved to change slices.
        We simply go to the Model, get the requested slice, and hand it back.
        """
        return self.model.get_slice(axis, slice_index)

    def on_click(self, event, pointer_color, ax):
        """
        Callback for mouse clicks on the matplotlib canvas.
        This is where we record the point location and update the view.
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

        pos_flag = 1 if current_tab.pos_click_var.get() else 0

        if current_tab.pos_click_checkbox["state"] == "disabled":
            current_tab.pos_click_checkbox["state"] = "normal"
        
        # Add to points list
        current_tab.points.append((x, y, color, pos_flag))
        # Clear the redo stack if a new point is added
        current_tab.redo_stack.clear()

        # Update the View
        self.view.add_point_to_listbox(x, y, pos_flag, active_index=active_index)
        line = self.view.plot_point(x, y, color, ax)
        current_tab.line_objects.append(line)

        # update the state of the pos_click_checkbox based on points for the current color.
        points_for_color = [pt for pt in current_tab.points if pt[2].lower() == color]
        if points_for_color:
            current_tab.pos_click_checkbox.config(state="normal")
        else:
            current_tab.pos_click_var.set(True)
            current_tab.pos_click_checkbox.config(state="disabled")

    def undo(self):
        """
        Undo the last point addition.
        """
        active_index = self.view.sidebar.tabControl.index("current")
        current_tab = self.view.sidebar.tabs[active_index]

        if not current_tab.points:
            return

        # Pop the last point from points
        last_point = current_tab.points.pop()
        current_tab.undo_stack.append(last_point)
        current_tab.redo_stack.append(last_point)

        # Remove from the View’s listbox and figure
        self.view.remove_last_point_from_listbox()
        if current_tab.line_objects:
            last_line = current_tab.line_objects.pop()
            last_line.remove()

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
        self.view._update_slice(canvas.figure.axes[0], canvas, axis, slice_idx, label)

    def redo(self):
        """
        Redo the last undone point addition.
        """
        active_index = self.view.sidebar.tabControl.index("current")
        current_tab = self.view.sidebar.tabs[active_index]

        if not current_tab.redo_stack:
            return

        restored_point = current_tab.redo_stack.pop()
        current_tab.points.append(restored_point)
        current_tab.undo_stack.append(restored_point)

        x, y, color, pos_flag = restored_point
        self.view.add_point_to_listbox(x, y, pos_flag, color=color)

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
        
        self.view.plot_point(x, y, color, getattr(self.view, f"{label.lower().split()[0]}_view").canvas_ax)
        slice_idx = int(canvas.slider.get())
        self.view._update_slice(canvas.figure.axes[0], canvas, axis, slice_idx, label)
    
    def refresh_selection_state(self):
        # Iterate over all tabs and reset their state.
        for tab in self.view.sidebar.tabs:
            # Clear the Listbox (if it exists)
            if tab.points_listbox:
                tab.points_listbox.delete(0, "end")
            # Remove any drawn line objects from the canvas.
            while tab.line_objects:
                line = tab.line_objects.pop()
                try:
                    line.remove()
                except Exception as e:
                    print(f"Error removing line: {e}")
            # Reset the lists and stacks.
            tab.points = []
            tab.undo_stack = []
            tab.redo_stack = []
            if hasattr(tab, "pos_click_checkbox"):
                tab.pos_click_checkbox.config(state = "disabled")

        # disable the show-points checkbox in each view if no points remain
        if hasattr(self.view, "axial_view") and hasattr(self.view.axial_view.canvas, "show_points_checkbox"):
            self.view.axial_view.canvas.show_points_checkbox.config(state="disabled")
        if hasattr(self.view, "coronal_view") and hasattr(self.view.coronal_view.canvas, "show_points_checkbox"):
            self.view.coronal_view.canvas.show_points_checkbox.config(state="disabled")
        if hasattr(self.view, "sagittal_view") and hasattr(self.view.sagittal_view.canvas, "show_points_checkbox"):
            self.view.sagittal_view.canvas.show_points_checkbox.config(state="disabled")

    def segment_image(self, slices, points, frame_idx, axis_str_suffix, custom_filename=None, completion_callback=None, downsampled=False, multi_resolution=False, is_first=False, is_final=False, progress_title=None, is_global=False):
        """
        Calls the model to segment the image based on the user's clicks.
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
            subprocess.run(["modal", "volume", "rm", "-r", "sam_2_medical_3d", f"SAM_2_Medical_3D/frames/{foldername}"])

            # Step 3: Create folder and transfer files to modal
            subprocess.run(["modal", "volume", "put", "sam_2_medical_3d", local_folder, f"SAM_2_Medical_3D/frames/{foldername}"], check=True)

            # Step 4: Delete local JPG folder and contents
            shutil.rmtree(local_folder)
        
        self.run_segmentation_with_progress(slices, points, frame_idx, axis_str_suffix, foldername, completion_callback, multi_resolution, is_first, is_final, progress_title, is_global)
    
    def run_segmentation_with_progress(self, slices, points, frame_idx, axis_str_suffix, foldername, completion_callback=None, multi_resolution=False, is_first=False, is_final=False, progress_title=None, is_global=False):
        import modal_handler
        # create progress dialog as a child of the main view
        progress_dialog = ProgressDialog(self.view, title=progress_title if progress_title is not None else "Progress")
        progress_dialog.top.transient(self.view)
        progress_dialog.update_progress() # start polling the progress queue

        # runs in a separate thread
        def run_segmentation():
            capture = StdoutCapture(sys.stdout, progress_dialog.progress_queue)
            try:
                # redirect stdout to capture progress messages
                with contextlib.redirect_stdout(capture):
                    if self.is_remote:
                        import modal_handler
                        video_segments = modal_handler.segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final, is_global)
                        # signal completion
                        progress_dialog.progress_queue.put(("done", 100))
                    else:
                        import local_handler
                        video_segments = local_handler.segment(slices, points, frame_idx, foldername, multi_resolution, is_first, is_final, is_global)
                        # signal completion
                        progress_dialog.progress_queue.put(("done", 100))
                
                # once segmentation finished, update the view on the main thread
                if self.is_remote:
                    local_folder = f"./temp/{foldername}"
                    if os.path.exists(local_folder):
                        shutil.rmtree(local_folder)

                self.view.after(0, lambda: finish_segmentation(video_segments))
            except Exception as e:
                self.view.after(0, lambda: tk.messagebox.showerror("Segmentation Error", f"An exception occurred in the container:\n{e}"))
    
        def finish_segmentation(video_segments):
            progress_dialog.close()
            if completion_callback:
                completion_callback(video_segments)
            else:
                self.view.show_segmentation(video_segments, axis_str_suffix)
                self.view.update_mesh_view(video_segments, axis_str_suffix)
        
        seg_thread = threading.Thread(target=run_segmentation)
        seg_thread.start()
    
    # multiresolution segmentation routine
    def multiresolution_segmentation(self, tab):
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

        start_res = 64
        if original_h < start_res or original_w < start_res:
            messagebox.showerror("Downsampling Error", "Image is too small to downsample to 64x64.")
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

        # helper: downsample the entire volume for the selected view
        def downsample_volume(res):
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
        def upsample_target(current, original):
            standards = [64,128,256,512,1024]
            for s in standards:
                if s > current and s <= original:
                    return s
            return original

        most_recent_video_segments = None

        # This inner function performs one iteration.
        def iteration(resolution, seeds):
            if resolution > max(original_h, original_w):
                def show_final():
                    nonlocal most_recent_video_segments
                    if most_recent_video_segments:
                        self.view.show_segmentation(most_recent_video_segments, axis_str_suffix)
                        self.view.update_mesh_view(most_recent_video_segments, axis_str_suffix)
                self.view.after(100, show_final)
                return
            
            downsampled_volume = downsample_volume(resolution)
            print("*", downsampled_volume.shape)
            points = seeds

            filename_without_extension = os.path.splitext(os.path.basename(self.model.filename))[0]

            temp_filename = f"downsampled_{filename_without_extension}_{resolution}"
            os.makedirs("./temp", exist_ok=True)
            nii_img = nib.Nifti1Image(downsampled_volume, affine=np.eye(4))
            print("**", nii_img.header.get_data_shape())
            nib.save(nii_img, f"./temp/{temp_filename}_{axis_str_suffix}.nii")

            # Completion callback to capture segmentation result.
            result_container = {}
            def completion_callback(video_segments):
                result_container['video_segments'] = video_segments
            
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

            def check_result():
                nonlocal most_recent_video_segments
                if 'video_segments' in result_container:
                    video_segments = result_container['video_segments']
                    print(f"resolution: {resolution}")
                    most_recent_video_segments = result_container['video_segments']
                    
                    os.remove(f"./temp/{temp_filename}_{axis_str_suffix}.nii")
                    
                    # if slice_idx not in video_segments:
                    #     messagebox.showerror("Error", f"No segmentation found for slice {slice_idx}.")
                    #     return
                    
                    nonlocal upsampled_masks
                    upsampled_masks = {}

                    for obj_id, seg_mask_down in video_segments[slice_idx].items():
                        if seg_mask_down.ndim == 3 and seg_mask_down.shape[0] == 1:
                            seg_mask_down = seg_mask_down[0]
                        if seg_mask_down.dtype == np.bool_:
                            seg_mask_down = seg_mask_down.astype(np.uint8)
                        upsampled_masks[obj_id] = cv2.resize(seg_mask_down, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

                    # auto compute new seeds for each object
                    new_seeds = {}
                    for obj_id in seeds.keys():
                        seg_mask_down = video_segments[slice_idx].get(obj_id)
                        if seg_mask_down is not None:
                            if seg_mask_down.ndim == 3 and seg_mask_down.shape[0] == 1:
                                seg_mask_down = seg_mask_down[0]
                            mask_uint8 = (seg_mask_down > 0).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                contour = max(contours, key=cv2.contourArea)
                                # use centroid of the largest segmentation mask
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    if self.view.sidebar.global_segmentation_var.get():
                                        # if global segmentation, use the original image size
                                        cx = int(cx * (original_w / resolution))
                                        cy = int(cy * (original_h / resolution))
                                        new_seeds[obj_id] = defaultdict(list)
                                        new_seeds[obj_id][int(self.view.axial_view.canvas.slider.get())] = [(cx, cy, 1)]
                                    else:
                                        new_seeds[obj_id] = [(cx, cy, 1)]
                                    print("new seeds in m00")
                                else:
                                    new_seeds[obj_id] = seeds[obj_id]
                                    print("new seeds in m00 else")
                            else:
                                # if no contours found then use the original seed
                                new_seeds[obj_id] = seeds[obj_id]
                                print("new seeds in contours else")
                        else:
                            # if segmentation mask is not returned then use the original seed
                            new_seeds[obj_id] = seeds[obj_id]
                            print("new seeds in seg_mask_down else")
                    
                    # for debugging: display new seed in temorary popup figure
                    # plt.figure(figsize=(12, 8))
                    # plt.title(f"New Seed Points")
                    # plt.imshow(original_image)
                    # for obj_id, pts in new_seeds.items():
                    #     mpl_color = {1:"r",2:"b",3:"g",4:"orange",5:"purple",6:"c",7:"m",8:"y",9:"k",10:"gray"}.get(obj_id, "r")
                    #     xs = [int(p[0]*(original_w/resolution)) for p in pts]
                    #     ys = [int(p[1]*(original_h/resolution)) for p in pts]
                    #     plt.gca().scatter(xs, ys, color=mpl_color, marker='*', s=200, edgecolor='white', linewidth=1.25)
                    # plt.close() 

                    new_res = resolution * 2
                    iteration(new_res, new_seeds)
                else:
                    self.view.after(100, check_result)
            
            check_result()
        
        # initialise variable to hold upsampled masks from last iteration
        upsampled_masks = {}
        iteration(current_res, seeds)
