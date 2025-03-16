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
from tkinter import ttk, messagebox

from dicom_to_nifti import DicomToNifti

import cv2
import numpy as np
try:
    import nibabel as nib
except ImportError:
    print("nibabel is required for saving temporary NIfTI images.")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        
        # Add to points list
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

    def segment_image(self, slices, points, frame_idx, axis_str_suffix, custom_filename=None, completion_callback=None, downsampled=False):
        """
        Calls the model to segment the image based on the user's clicks.
        """
        # get filename (without prefix and file format)
        filename = custom_filename if custom_filename is not None else self.model.filename
        foldername = f"{filename.split('.')[0].split(os.sep)[-1]}_{axis_str_suffix}_JPG"
        local_folder = f"./temp/{foldername}"




        # Step 0: Delete local folder if it exists
        if os.path.exists(local_folder):
            shutil.rmtree(local_folder)

        # Step 1: Convert NIfTI to JPG and create local folder
        # sys.executable will auto select the running Python interpreter (supports cross-platform)
        if downsampled:
            subprocess.run([sys.executable, "nifti_to_jpg.py", filename, local_folder, "--axis", axis_str_suffix.lower(), "--downsampled"], check=True)
        else:
            subprocess.run([sys.executable, "nifti_to_jpg.py", filename, local_folder, "--axis", axis_str_suffix.lower()], check=True)
        # Step 2: Delete folder in modal
        # do not check=True because no graceful handling if folder not exists in modal volume
        subprocess.run(["modal", "volume", "rm", "-r", "sam_2_medical_3d", f"SAM_2_Medical_3D/frames/{foldername}"])

        # Step 3: Create folder and transfer files to modal
        subprocess.run(["modal", "volume", "put", "sam_2_medical_3d", local_folder, f"SAM_2_Medical_3D/frames/{foldername}"], check=True)

        # Step 4: Delete local JPG folder and contents
        shutil.rmtree(local_folder)

        self.run_segmentation_with_progress(slices, points, frame_idx, axis_str_suffix, foldername, completion_callback)

    def run_segmentation_with_progress(self, slices, points, frame_idx, axis_str_suffix, foldername, completion_callback=None):
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
            if completion_callback:
                completion_callback(video_segments)
            else:
                self.view.show_segmentation(video_segments, axis_str_suffix)
                self.view.update_mesh_view(video_segments, axis_str_suffix)
        
        seg_thread = threading.Thread(target=run_segmentation)
        seg_thread.start()
    
    # automated multiresolution segmentation routine
    def automated_multires_segmentation(self, tab):
        if self.model.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        full_image = np.asarray(self.model.image)

        # get the original image as a numpy array
        axis = self.view.tabs.index(tab)
        if axis == 0:
            slice_idx = int(self.view.axial_view.canvas.slider.get())
            original_image = full_image[slice_idx, :, :]
            original_h, original_w = original_image.shape
            axis_str_suffix = "AXIAL"
        elif axis == 1:
            slice_idx = int(self.view.coronal_view.canvas.slider.get())
            coronal_slice = full_image[:, slice_idx, :]
            original_h, original_w = coronal_slice.shape
            # axial dim != other dims, so resize it to square image so the segmentation algorithm always gets a square image.
            original_image = cv2.resize(coronal_slice, (512, 512), interpolation=cv2.INTER_AREA)
            axis_str_suffix = "CORONAL"
        elif axis == 2:
            slice_idx = int(self.view.sagittal_view.canvas.slider.get())
            sagittal_slice = full_image[:, :, slice_idx]
            original_h, original_w = sagittal_slice.shape
            # resize, like coronal slice
            original_image = cv2.resize(sagittal_slice, (512, 512), interpolation=cv2.INTER_AREA)
            axis_str_suffix = "SAGITTAL"
        else:
            messagebox.showerror("Error", "Invalid axis.")
            return

        start_res = 64
        if original_h < start_res or original_w < start_res:
            messagebox.showerror("Downsampling Error", "Image is too small to downsample to 64x64.")
            return

        popup = tk.Toplevel(self.view)
        popup.title("Automated Multiresolution Segmentation")

        current_res = start_res

        # compute initial seeds on the downsampled image
        def compute_seeds(image):
            # Use cv2.goodFeaturesToTrack as an example.
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=5)
            seeds = []
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    seeds.append((int(x), int(y), 1))
            return seeds

        seeds = compute_seeds(cv2.resize(original_image, (current_res, current_res), interpolation=cv2.INTER_AREA))

        scale_x = original_w / current_res
        scale_y = original_h / current_res

        plt.figure(figsize=(12, 8))
        plt.title(f"seed points")
        plt.imshow(original_image)
        x = [int(seed[0]*scale_x) for seed in seeds] # upscale them to meet scale of original image
        y = [int(seed[1]*scale_y) for seed in seeds]
        plt.gca().scatter(x, y, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.show()

        # helper: downsample the entire volume for the selected view
        def downsample_volume(res):
            if axis == 0:
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

        # This inner function performs one iteration.
        def iteration(resolution, seeds):
            if resolution > max(original_h, original_w):
                messagebox.showerror("Downsampling Error", f"Cannot downsample to {resolution}x{resolution}.")
                return
            
            downsampled_volume = downsample_volume(resolution)
            print("*", downsampled_volume.shape)
            points = {1: seeds}
        
            temp_filename = f"./temp/downsampled_{os.path.basename(self.model.filename)}_{axis_str_suffix}_{resolution}.nii"
            os.makedirs("./temp", exist_ok=True)
            nii_img = nib.Nifti1Image(downsampled_volume, affine=np.eye(4))
            print("**", nii_img.header.get_data_shape())
            nib.save(nii_img, temp_filename)

            # Completion callback to capture segmentation result.
            result_container = {}
            def completion_callback(video_segments):
                result_container['video_segments'] = video_segments
            
            # Call segment_image on the downsampled volume.
            # Save the downsampled image temporarily as a NIfTI file.
            self.segment_image(
                downsampled_volume,
                points,
                frame_idx=slice_idx,
                axis_str_suffix=axis_str_suffix,
                custom_filename=temp_filename,
                completion_callback=completion_callback,
                downsampled=True
            )

            def check_result():
                if 'video_segments' in result_container:
                    video_segments = result_container['video_segments']
                    
                    os.remove(temp_filename)
                    
                    if slice_idx not in video_segments:
                        messagebox.showerror("Error", f"No segmentation found for slice {slice_idx}.")
                        return
                    
                    seg_mask_down = list(video_segments[slice_idx].values())[0]

                    print("***", seg_mask_down.shape)

                    if seg_mask_down.shape[0] == 1:
                        seg_mask_down = seg_mask_down[0]

                    if seg_mask_down.dtype == np.bool_:
                        seg_mask_down = seg_mask_down.astype(np.uint8)

                    # Upsample mask to target (closest higher standard resolution).
                    target_res = upsample_target(resolution, max(original_h, original_w))
                    upsampled_mask = cv2.resize(seg_mask_down, (target_res, target_res), interpolation=cv2.INTER_NEAREST)

                    # clear popup and show result
                    for widget in popup.winfo_children():
                        widget.destroy()
                    
                    fig, ax = plt.subplots(figsize=(4,4))
                    ax.imshow(upsampled_mask, cmap='gray')
                    ax.set_title(f"Segmentation at {resolution}x{resolution} (upsampled to {target_res}x{target_res})")
                    canvas = FigureCanvasTkAgg(fig, master=popup)
                    canvas.get_tk_widget().pack()
                    canvas.draw()

                    def next_iteration():
                        # Extract new seeds from the downsampled mask.
                        mask_uint8 = (seg_mask_down > 0).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        new_seeds = []
                        if contours:
                            contour = max(contours, key=cv2.contourArea)
                            for pt in contour:
                                x, y = pt[0]
                                new_seeds.append((int(x), int(y), 1))
                        
                        scale_x = original_w / resolution
                        scale_y = original_h / resolution

                        plt.figure(figsize=(12, 8))
                        plt.title(f"seed points")
                        plt.imshow(original_image)
                        x = [int(seed[0]*scale_x) for seed in new_seeds] # upscale them to meet scale of original image
                        y = [int(seed[1]*scale_y) for seed in new_seeds]
                        plt.gca().scatter(x, y, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
                        plt.show()


                        new_res = resolution * 2
                        if new_res > max(original_h, original_w):
                            for widget in popup.winfo_children():
                                widget.destroy()
                            fig2, ax2 = plt.subplots(figsize=(4,4))
                            ax2.imshow(upsampled_mask, cmap='gray')
                            ax2.set_title("Final Segmentation")
                            canvas2 = FigureCanvasTkAgg(fig2, master=popup)
                            canvas2.get_tk_widget().pack()
                            canvas2.draw()
                            apply_btn = ttk.Button(popup, text="Apply Segmentation",
                                                command=lambda: self.apply_segmentation_to_frame(upsampled_mask, tab))
                            apply_btn.pack(pady=5)
                        else:
                            iteration(new_res, new_seeds)
                    next_btn = ttk.Button(popup, text="Next", command=next_iteration)
                    next_btn.pack(pady=5)
                else:
                    popup.after(100, check_result)
            
            check_result()
            
        # start the first iteration
        iteration(current_res, seeds)
    
    # NEW: Apply final segmentation mask to the respective frame.
    def apply_segmentation_to_frame(self, mask, tab):
        active_index = self.view.tabs.index(tab)
        if active_index == 0:
            axis_str_suffix = "AXIAL"
            canvas = self.view.axial_view.canvas
            self.view.axial_view_mask = {int(canvas.slider.get()): {1: mask}}
            label = "Axial View"
        elif active_index == 1:
            axis_str_suffix = "CORONAL"
            canvas = self.view.coronal_view.canvas
            self.view.coronal_view_mask = {int(canvas.slider.get()): {1: mask}}
            label = "Coronal View"
        elif active_index == 2:
            axis_str_suffix = "SAGITTAL"
            canvas = self.view.sagittal_view.canvas
            self.view.sagittal_view_mask = {int(canvas.slider.get()): {1: mask}}
            label = "Sagittal View"
        self.view._update_slice(canvas.figure.axes[0], canvas, active_index, int(canvas.slider.get()), label)
