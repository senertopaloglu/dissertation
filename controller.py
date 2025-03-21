import sys
import shutil
import os
import subprocess
import re
import contextlib
import threading
import queue
import time

import tkinter as tk
from tkinter import ttk

from dicom_to_nifti import DicomToNifti

class StdoutCapture:
    def __init__(self, original_stdout, progress_queue):
        self.original_stdout = original_stdout
        self.progress_queue = progress_queue
        self.buffer = ""
    
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        self.buffer += text
        if "\n" in text:
            lines = self.buffer.split("\n")
            for line in lines[:-1]:
                self.process_line(line)
            self.buffer = lines[-1]
    
    def process_line(self, line):
        # look for "frame loading" messages
        m = re.search(r"frame loading \(JPEG\):\s*(\d+)%", line)
        if m:
            percent = int(m.group(1))
            self.progress_queue.put(("Loading frames", percent))
        # look for "propagate in video" messages
        m2 = re.search(r"propagate in video:\s*(\d+)%", line)
        if m2:
            percent = int(m2.group(1))
            self.progress_queue.put(("Propagating segmentation", percent))
        # use successfully installed dependencies as a signal that container is ready
        if "Successfully installed" in line:
            self.progress_queue.put(("Preparing container", 100))
    
    def flush(self):
        self.original_stdout.flush()

class ProgressDialog:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Progress")
        self.top.geometry("300x250")
        self.top.grab_set() # make progress dialog modal

        self.prep_label = tk.Label(self.top, text="Preparing container: 0%")
        self.prep_label.pack(padx=10, pady=5)
        self.prep_progress = ttk.Progressbar(self.top, length=300, mode="determinate", maximum=100)
        self.prep_progress.pack(padx=10, pady=5)

        self.load_label = tk.Label(self.top, text="Loading frames: 0%")
        self.load_label.pack(padx=10, pady=5)
        self.load_progress = ttk.Progressbar(self.top, orient="horizontal", length=300,
                                                mode="determinate", maximum=100)
        self.load_progress.pack(padx=10, pady=5)

        self.prop_label = tk.Label(self.top, text="Propagating segmentation: 0%")
        self.prop_label.pack(padx=10, pady=5)
        self.prop_progress = ttk.Progressbar(self.top, orient="horizontal", length=300,
                                                mode="determinate", maximum=100)
        self.prop_progress.pack(padx=10, pady=5)

        # create a queue to receive progress updates
        self.progress_queue = queue.Queue()

        # for container prep ETA: 360 seconds => 99%
        self.prep_start_time = time.time()
        
        self.current_progress = {
            "Preparing container": 0,
            "Loading frames": 0,
            "Propagating segmentation": 0
        }
    
    def update_progress(self):
        elapsed = time.time() - self.prep_start_time
        if self.current_progress["Preparing container"] < 100:
            computed = min(99, (elapsed / 360) * 99)
            self.current_progress["Preparing container"] = max(self.current_progress["Preparing container"], computed)
            self.prep_progress["value"] = self.current_progress["Preparing container"]
            self.prep_label.config(text=f"Preparing container: {int(self.current_progress['Preparing container'])}%")

        try:
            while True:
                stage, value = self.progress_queue.get_nowait()
                if stage == "Preparing container":
                    self.current_progress["Preparing container"] = 100
                    self.prep_progress["value"] = 100
                    self.prep_label.config(text=f"Preparing container: 100%")
                elif stage == "Loading frames":
                    self.current_progress["Loading frames"] = value
                    self.load_progress["value"] = value
                    self.load_label.config(text=f"Loading frames: {value}%")
                    # if loading frames has started and prepping is not at 100%, force it
                    if value > 0 and self.current_progress["Preparing container"] < 100:
                        self.current_progress["Preparing container"] = 100
                        self.prep_progress["value"] = 100
                        self.prep_label.config(text="Preparing container: 100%")
                elif stage == "Propagating segmentation":
                    self.current_progress["Propagating segmentation"] = value
                    self.prop_progress["value"] = value
                    self.prop_label.config(text=f"Propagating segmentation: {value}%")
                    # if propagation has started and loading frames is not at 100%, force it
                    if value > 0 and self.current_progress["Loading frames"] < 100:
                        self.current_progress["Loading frames"] = 100
                        self.load_progress["value"] = 100
                        self.load_label.config(text="Loading frames: 100%")
                elif stage == "done":
                    pass
        except queue.Empty:
            pass

        # Continue polling every 100ms until the window is destroyed.
        if self.top.winfo_exists():
            self.top.after(100, self.update_progress)
        
    def close(self):
        self.top.destroy()


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
            file_path = DicomToNifti.convert(file_path)
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

    def on_click(self, event, pointer_color):
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
            current_tab = self.view.tabs[active_index]
        else:
            # Fallback: use the current active tab
            active_index = self.view.tabControl.index("current")
            current_tab = self.view.tabs[active_index]

        pos_flag = 1 if current_tab.pos_click_var.get() else 0

        if current_tab.pos_click_checkbox["state"] == "disabled":
            current_tab.pos_click_checkbox["state"] = "normal"
        
        # record the point in the current view
        # note: interpretation of (x,y) depends on which view (tab) is active
        current_tab.points.append((x, y, color, pos_flag))
        # Clear the redo stack if a new point is added
        current_tab.redo_stack.clear()

        # Update the View
        self.view.add_point_to_listbox(x, y, pos_flag, active_index=active_index)
        line = self.view.plot_point(x, y, color)
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
        active_index = self.view.tabControl.index("current")
        current_tab = self.view.tabs[active_index]

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
        active_index = self.view.tabControl.index("current")
        current_tab = self.view.tabs[active_index]

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
        for tab in self.view.tabs:
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

    def convert_point_to_axial(self, point, slider_value, active_index):
        """
        HELPER METHOD
        Convert a point (x, y, color, pos_flag) from the current view to axial coordinates.
        
        For:
          - Axial view (active_index==0): point is (x, y) and slider_value is the axial slice (z).
          - Coronal view (active_index==1): the displayed image is (x, z) with fixed y=slider_value.
              → The corresponding axial 3D coordinate is (x, y_fixed, z); on the axial slice,
                the point appears as (x, y_fixed) and the target axial slice index is taken from z.
          - Sagittal view (active_index==2): the displayed image is (y, z) with fixed x=slider_value.
              → The corresponding axial 3D coordinate is (x_fixed, y, z); on the axial slice,
                the point appears as (y) in the second coordinate and the target axial slice index is z.
        """
        if active_index == 0:
            # axial view: no conversion needed
            return (point[0], point[1]), slider_value
        elif active_index == 1:
            # Coronal view: point = (x, z); slider_value gives y.
            return (point[0], slider_value), int(point[1])
        elif active_index == 2:
            # Sagittal view: point = (y, z); slider_value gives x.
            return (slider_value, point[0]), int(point[1])
        else:
            raise ValueError("Invalid active index.")
    
    def reproject_segmentation(self, axial_video_segments, target_axis):
        """
        HELPER METHOD
        Convert a full axial segmentation (a dict mapping axial slice index to per-object masks)
        into a segmentation dictionary for the target view.
        
        target_axis:
            0 → axial (return as is),
            1 → coronal (slicing along y),
            2 → sagittal (slicing along x).
        """
        import numpy as np
        # Reconstruct the 3D volume from axial segmentation.
        z_indices = sorted(axial_video_segments.keys())
        first_mask = np.squeeze(next(iter(axial_video_segments[z_indices[0]].values())))
        x_dim, y_dim = first_mask.shape  # axial mask shape: (x, y)
        z_dim = len(z_indices)
        combined = {}
        for obj_id in axial_video_segments[z_indices[0]].keys():
            volume = np.zeros((z_dim, x_dim, y_dim), dtype=int)
            for i, z in enumerate(z_indices):
                volume[i, :, :] = np.squeeze(axial_video_segments[z][obj_id])
            combined[obj_id] = volume

        reprojected = {}
        if target_axis == 1:
            # Coronal view: fixed y coordinate; each slice is obtained by taking volume[:, :, y].
            for y in range(y_dim):
                slice_dict = {}
                for obj_id, vol in combined.items():
                    mask_slice = vol[:, :, y]  # shape: (z, x)
                    if y == 0:
                        print("[DEBUG] mask_slice shape before any transpose/rotation:", mask_slice.shape)
                    # Transpose to match the (x, z) orientation used in the coronal view.
                    #mask_slice = np.transpose(mask_slice)
                    if y == 0:
                        print("[DEBUG] mask_slice shape after transpose:", mask_slice.shape)
                    #mask_slice = np.rot90(mask_slice, k=-1) # rotate 90 deg clockwise, once
                    if y == 0:
                        print("[DEBUG] mask_slice shape after transpose and rot90:", mask_slice.shape)
                    slice_dict[obj_id] = mask_slice
                reprojected[y] = slice_dict
        elif target_axis == 2:
            # Sagittal view: fixed x coordinate; each slice is volume[:, x, :].
            for x in range(x_dim):
                slice_dict = {}
                for obj_id, vol in combined.items():
                    mask_slice = vol[:, x, :]  # shape: (z, y)
                    mask_slice = np.transpose(mask_slice)  # to (y, z) orientation
                    mask_slice = np.rot90(mask_slice, k=1)
                    mask_slice = np.fliplr(mask_slice)
                    slice_dict[obj_id] = mask_slice
                reprojected[x] = slice_dict
        else:
            # For axial view, no reprojecting is needed.
            return axial_video_segments

        return reprojected
    
    def segment_image(self, slices, points, frame_idx, axis_str_suffix):
        """
        Calls the model to segment the image based on the user's clicks.

        Instead of directly segmenting using the current view’s slice,
        we convert all user-selected points into axial coordinates and always use the axial slice.
        Later, if the active view is not axial, we reproject the full segmentation volume.
        """
        active_index = self.view.tabControl.index("current")
        current_tab = self.view.tabs[active_index]
        if not current_tab.points:
            print("No slice selected for segmentation.")
            return

        # Get the slider value (which is interpreted differently per view).
        if active_index == 0:
            slider_value = int(self.view.axial_view.canvas.slider.get())
        elif active_index == 1:
            slider_value = int(self.view.coronal_view.canvas.slider.get())
        elif active_index == 2:
            slider_value = int(self.view.sagittal_view.canvas.slider.get())
        else:
            raise ValueError("Invalid active view index.")
        
        # Convert all points from the current view into axial coordinates.
        axial_points = []
        for pt in current_tab.points:
            axial_pt, axial_z = self.convert_point_to_axial(pt, slider_value, active_index)
            axial_points.append((axial_pt, axial_z, pt[2], pt[3]))
        # Use the axial slice index from the most recent click.
        target_axial_slice = axial_points[-1][1]

        # Build points_grouped (using your color_map convention) with the converted axial points.
        points_grouped = {}
        color_map = {
            "red": 1,
            "blue": 2,
            "green": 3,
            "orange": 4,
            "purple": 5,
            "cyan": 6,
            "magenta": 7,
            "yellow": 8,
            "black": 9,
            "gray": 10
        }
        for pt in axial_points:
            click, _, color, pos_flag = pt
            k = color_map.get(color.lower(), 1)
            if k not in points_grouped:
                points_grouped[k] = []
            points_grouped[k].append((int(click[0]), int(click[1]), int(pos_flag)))
        
        # Always use the axial slice for segmentation.
        axial_slices = self.view._slice_request_callback(0, target_axial_slice)
        
        # get filename (without prefix and file format)
        foldername = f"{self.model.filename.split('.')[0].split('/')[-1]}_{axis_str_suffix}_JPG"
        local_folder = f"./temp/{foldername}"

        # Step 0: Delete local folder if it exists
        if os.path.exists(local_folder):
            shutil.rmtree(local_folder)

        # Step 1: Convert NIfTI to JPG and create local folder
        # sys.executable will auto select the running Python interpreter (supports cross-platform)
        subprocess.run([sys.executable, "nifti_to_jpg.py", self.model.filename, local_folder, "--axis", "axial"], check=True)

        # Step 2: Delete folder in modal
        # do not check=True because no graceful handling if folder not exists in modal volume
        subprocess.run(["modal", "volume", "rm", "-r", "sam_2_medical_3d", f"SAM_2_Medical_3D/frames/{foldername}"])

        # Step 3: Create folder and transfer files to modal
        subprocess.run(["modal", "volume", "put", "sam_2_medical_3d", local_folder, f"SAM_2_Medical_3D/frames/{foldername}"], check=True)

        # Step 4: Delete local JPG folder and contents
        shutil.rmtree(local_folder)
        
        self.run_segmentation_with_progress(axial_slices, points_grouped, target_axial_slice, "AXIAL", foldername, active_index)
    
    def run_segmentation_with_progress(self, slices, points, frame_idx, axis_str_suffix, foldername, active_index):
        import modal_handler
        # create progress dialog as a child of the main view
        progress_dialog = ProgressDialog(self.view)
        progress_dialog.top.transient(self.view)
        progress_dialog.update_progress() # start polling the progress queue

        # runs in a separate thread
        def run_segmentation():
            capture = StdoutCapture(sys.stdout, progress_dialog.progress_queue)
            # redirect stdout to capture progress messages
            with contextlib.redirect_stdout(capture):
                video_segments = modal_handler.segment(slices, points, frame_idx, foldername)
                # signal completion
                progress_dialog.progress_queue.put(("done", 100))
            # once segmentation finished, update the view on the main thread
            self.view.after(0, lambda: finish_segmentation(video_segments))
    
        def finish_segmentation(video_segments):
            progress_dialog.close()
            
            # If the active view is non-axial, reproject the axial segmentation.
            if active_index == 0:
                self.view.show_segmentation(video_segments, "AXIAL")
                self.view.update_mesh_view(video_segments, "AXIAL")
            elif active_index == 1:
                reprojected = self.reproject_segmentation(video_segments, target_axis=1)
                self.view.show_segmentation(reprojected, "CORONAL")
                self.view.update_mesh_view(reprojected, "CORONAL")
            elif active_index == 2:
                reprojected = self.reproject_segmentation(video_segments, target_axis=2)
                self.view.show_segmentation(reprojected, "SAGITTAL")
                self.view.update_mesh_view(reprojected, "SAGITTAL")
        
        seg_thread = threading.Thread(target=run_segmentation)
        seg_thread.start()