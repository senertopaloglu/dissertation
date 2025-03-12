import sys
import shutil
import os
import subprocess

from dicom_to_nifti import DicomToNifti

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
        self.refresh_selection_state()
        self.view.show_image()

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
            print("why am i in the fallback?")
            active_index = self.view.tabControl.index("current")
            current_tab = self.view.tabs[active_index]

        # Add to points list
        current_tab.points.append((x, y, color))
        # Clear the redo stack if a new point is added
        current_tab.redo_stack.clear()

        # Update the View
        self.view.add_point_to_listbox(x, y, active_index=active_index)
        line = self.view.plot_point(x, y, color)
        current_tab.line_objects.append(line)

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
        self.view.draw_canvas(active_index)

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

        x, y, color = restored_point
        self.view.add_point_to_listbox(x, y, color=color)
        if active_index == 0:
            line = self.view.plot_point(x, y, color, self.view.axial_view.canvas_ax)
            current_tab.line_objects.append(line)
            self.view.draw_canvas(0)
        elif active_index == 1:
            line = self.view.plot_point(x, y, color, self.view.coronal_view.canvas_ax)
            current_tab.line_objects.append(line)
            self.view.draw_canvas(1)
        elif active_index == 2:
            line = self.view.plot_point(x, y, color, self.view.sagittal_view.canvas_ax)
            current_tab.line_objects.append(line)
            self.view.draw_canvas(2)
        
    
    def refresh_selection_state(self):
        active_index = self.view.tabControl.index("current")
        current_tab = self.view.tabs[active_index]
        self.view.clear_listbox()
        while current_tab.line_objects:
            last_line = current_tab.line_objects.pop()
            last_line.remove()
        current_tab.points = []
        current_tab.line_objects = []
        current_tab.undo_stack = []
        current_tab.redo_stack = []

    def segment_image(self, slices, points, frame_idx, axis_str_suffix):
        import modal_handler
        """
        Calls the model to segment the image based on the user's clicks.
        """
        # get filename (without prefix and file format)
        foldername = f"{self.model.filename.split('.')[0].split('/')[-1]}_{axis_str_suffix}_JPG"
        local_folder = f"./temp/{foldername}"




        # Step 0: Delete local folder if it exists
        if os.path.exists(local_folder):
            shutil.rmtree(local_folder)

        # Step 1: Convert NIfTI to JPG and create local folder
        print(f"Running nifti_to_jpg.py using python executable at {sys.executable}, with arguments: {self.model.filename}, {local_folder}")
        # sys.executable will auto select the running Python interpreter (supports cross-platform)
        subprocess.run([sys.executable, "nifti_to_jpg.py", self.model.filename, local_folder, "--axis", axis_str_suffix.lower()], check=True)

        # Step 2: Delete folder in modal
        # do not check=True because no graceful handling if folder not exists in modal volume
        print(f"Deleting folder in modal: SAM_2_Medical_3D/frames/{foldername}")
        subprocess.run(["modal", "volume", "rm", "-r", "sam_2_medical_3d", f"SAM_2_Medical_3D/frames/{foldername}"])

        # Step 3: Create folder and transfer files to modal
        print(f"Transferring files to modal: {local_folder} to SAM_2_Medical_3D/frames/{foldername}")
        subprocess.run(["modal", "volume", "put", "sam_2_medical_3d", local_folder, f"SAM_2_Medical_3D/frames/{foldername}"], check=True)

        # Step 4: Delete local JPG folder and contents
        print(f"Deleting local folder: {local_folder}")
        shutil.rmtree(local_folder)





        print(f"Calling modal_handler.segment with foldername: {foldername}")
        video_segments=modal_handler.segment(slices, points, frame_idx, foldername)
        print("0. inside controller")
        self.view.show_segmentation(video_segments)
        self.view.update_mesh_view(video_segments)
        print("segmenting image completed.")
        
