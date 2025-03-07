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

        # Add to points list
        self.points.append((x, y, color))
        # Clear the redo stack if a new point is added
        self.redo_stack.clear()

        # Update the View
        self.view.add_point_to_listbox(x, y)
        line = self.view.plot_point(x, y, color)
        self.line_objects.append(line)

    def undo(self):
        """
        Undo the last point addition.
        """
        if not self.points:
            return

        # Pop the last point from points
        last_point = self.points.pop()
        self.undo_stack.append(last_point)
        self.redo_stack.append(last_point)

        # Remove from the View’s listbox and figure
        self.view.remove_last_point_from_listbox()
        if self.line_objects:
            last_line = self.line_objects.pop()
            last_line.remove()
        self.view.draw_canvas()

    def redo(self):
        """
        Redo the last undone point addition.
        """
        if not self.redo_stack:
            return

        restored_point = self.redo_stack.pop()
        self.points.append(restored_point)
        self.undo_stack.append(restored_point)

        x, y, color = restored_point
        self.view.add_point_to_listbox(x, y)
        line = self.view.plot_point(x, y, color)
        self.line_objects.append(line)
        self.view.draw_canvas()
    
    def refresh_selection_state(self):
        self.view.clear_listbox()
        while self.line_objects:
            last_line = self.line_objects.pop()
            last_line.remove()
        self.points = []
        self.line_objects = []
        self.undo_stack = []
        self.redo_stack = []

    def segment_image(self, slices, points, frame_idx):
        import modal_handler
        """
        Calls the model to segment the image based on the user's clicks.
        """
        print("segmenting image begins...")
        video_segments=modal_handler.segment(slices, points, frame_idx)
        self.view.show_segmentation(video_segments)
        # TODO: self.view.show_mesh_view(...)
        print("segmenting image completed.")
        
