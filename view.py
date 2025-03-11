import os

from collections import defaultdict

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import scipy.ndimage as ndimage

import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D 

class MainView(tk.Tk):
    """
    The View in our MVC. Responsible for building and displaying the GUI.
    For all user interactions, we will invoke Controller callbacks.
    """
    def __init__(self, model, controller):
        super().__init__()

        self.model = model
        self.controller = controller

        # Keep references to callback functions
        self._on_click_callback = None
        self._undo_callback = None
        self._redo_callback = None
        self._slice_request_callback = None

        self.last_used_axis = None
        self.last_used_slice_index = None

        self.title("Interactive 3D Medical Image Segmentation")
        self.state("zoomed")

        # Prepare main window layout
        self.columnconfigure(0, weight=1, minsize=250)  # Left sidebar
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=3)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=3)

        # Sidebar + main frames
        self.sidebar = tk.Frame(self, bg="lightgray", padx=10, pady=10)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Prepare internal structures for user selections
        self.points_listbox = None
        
        self.pointer_color_var = None
        self.pointer_color_optionmenu = None

        self.pointer_color_combobox = None

        # Build the UI
        self._build_sidebar()

        # Behavior for closing
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_sidebar(self):
        """
        Builds the left sidebar with buttons, color combobox, listbox, etc.
        """
        # Buttons
        btn_import = ttk.Button(self.sidebar, text="Import Image", command=self._import_image)
        btn_import.pack(fill="x", pady=5)

        btn_segment = ttk.Button(self.sidebar, text="Segment Image", command=self._segment_image)
        btn_segment.pack(fill="x", pady=5)

        btn_export = ttk.Button(self.sidebar, text="Export Image")
        btn_export.pack(fill="x", pady=5)

        # Color dropdown
        pointer_label = tk.Label(self.sidebar, text="Pointer Colour")
        pointer_label.pack(pady=(10, 2))

        # Replace the ttk.Combobox with a tk.OptionMenu for colored options
        pointer_color_var = tk.StringVar(value="Red")

        # Define a callback to update the OptionMenu button color.
        def update_option_menu_color(*args):
            selected = pointer_color_var.get().lower()  # Convert to lowercase for consistency.
            self.pointer_color_optionmenu.config(fg=selected, activeforeground=selected)
        
        pointer_color_var.trace_add("write", update_option_menu_color)

        self.pointer_color_optionmenu = tk.OptionMenu(self.sidebar, pointer_color_var, "Red", "Blue", "Green")
        self.pointer_color_optionmenu.pack(fill="x")

        # Access the underlying menu and configure each item's text color
        menu = self.pointer_color_optionmenu["menu"]
        menu.entryconfig(0, foreground="red")
        menu.entryconfig(1, foreground="blue")
        menu.entryconfig(2, foreground="green")

        # Set the default text color to red at startup
        update_option_menu_color()

        # Optionally, store the variable for later use:
        self.pointer_color_var = pointer_color_var



        self.pointer_color_combobox = ttk.Combobox(
            self.sidebar,
            values=["Red", "Blue", "Green"]
        )
        self.pointer_color_combobox.pack(fill="x")
        self.pointer_color_combobox.set("Red")

        # Selected points listbox
        points_label = tk.Label(self.sidebar, text="Selected Points")
        points_label.pack(pady=(10, 2))

        points_frame = tk.Frame(self.sidebar)
        points_frame.pack(fill="x")

        scrollbar = tk.Scrollbar(points_frame, orient="vertical")
        self.points_listbox = tk.Listbox(
            points_frame, 
            height=5, 
            yscrollcommand=scrollbar.set
        )
        self.points_listbox.pack(side="left", fill="x", expand=True)
        scrollbar.config(command=self.points_listbox.yview)
        scrollbar.pack(side="right", fill="y")

        # Undo/Redo
        btn_undo = ttk.Button(self.sidebar, text="Undo", command=self._on_undo_click)
        btn_undo.pack(fill="x", pady=2)

        btn_redo = ttk.Button(self.sidebar, text="Redo", command=self._on_redo_click)
        btn_redo.pack(fill="x", pady=2)

    def _build_image_frames(self):
        """
        Builds the frames that display the axial, coronal, sagittal, and
        (placeholder) mesh views.
        """
        self.axial_view_mask = None
        self.coronal_view_mask = None
        self.sagittal_view_mask = None
        
        self.axial_view = self._create_image_frame("Axial View", axis=0)
        self.coronal_view = self._create_image_frame("Coronal View", axis=1)
        self.sagittal_view = self._create_image_frame("Sagittal View", axis=2)
        self.mesh_view = self._create_image_frame("Segmentation Result (Axial)", axis=0)

        # Attach these views to the grid
        self.axial_view.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.coronal_view.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.sagittal_view.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.mesh_view.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        self.update_idletasks()
        self.update()
    
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        print("SHOW MASK: START")
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        print("SHOW MASK: END")

    def _create_image_frame(self, text, axis):
        """
        Helper to create a labeled frame with a matplotlib FigureCanvas and a slider.
        """
        print("4. start view._create_image_frame. axis=", axis)
        frame = tk.Frame(self, bd=1, relief="solid")
        label = tk.Label(frame, text=text)
        label.pack()

        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        frame.canvas = canvas

        # Connect click events
        fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: self._on_click(event, ax, canvas)
        )

        if self.model.image:
            print("4. a) inside view._create_image_frame, when self.model.image is not None")
            sizes = self.model.image.GetLargestPossibleRegion().GetSize()
            dim = 2 if axis == 0 else 1 if axis == 1 else 0
            max_slice = sizes[dim] - 1

            # Slider to navigate slices
            slider = ttk.Scale(
                frame, 
                from_=0, 
                to=max_slice, 
                orient="horizontal",
                command=lambda val: self._update_slice(ax, canvas, axis, val, text)
            )
            slider.pack(side="top", fill="x", expand=True)
            
            initial_slice = max_slice // 2
            slider.set(initial_slice)

            # checkbox to show/hide segmentation mask
            show_mask_var=tk.BooleanVar()
            canvas.show_mask_var = show_mask_var
            # when checkbox is clicked, show_mask_var is auto updated
            checkbox = tk.Checkbutton(
                frame,
                text="Show Segmentation Mask",
                variable=show_mask_var,
                command=lambda: self._update_slice(ax, canvas, axis, int(canvas.slider.get()), text) # need to re-render image (either with or without mask depending on canvas.show_mask_var)
            )
            checkbox.pack(side="top", fill="x", pady=5)

            if axis==0 and self.axial_view_mask is None:
                checkbox.config(state="disabled")
            if axis==1 and self.coronal_view_mask is None:
                checkbox.config(state="disabled")
            if axis==2 and self.sagittal_view_mask is None:
                checkbox.config(state="disabled")

            canvas.checkbox = checkbox
            
            # Initialize the slice display
            self._update_slice(ax, canvas, axis, initial_slice, text)

            canvas.slider = slider

        print("4. b) inside view._create_image_frame, now not in the big if statement")

        return frame

    def _update_slice(self, ax, canvas, axis, val, text):
        """
        Update the displayed slice in the given axis whenever the slider changes.
        """
        print("2. a) start view._update_slice")
        if self._slice_request_callback is None:
            return
        
        slice_index = int(float(val))
        slice_array = self._slice_request_callback(axis, slice_index)
        print("2. b) inside view._update_slice, self._slice_request_callback = ", self._slice_request_callback)
        print("2. c) inside view._update_slice, slice_array = ", slice_array)
        ax.clear()
        ax.imshow(slice_array, cmap='gray')
        ax.set_title(f"{text} - Slice {slice_index}")

        if canvas.show_mask_var.get():
            if axis == 0:
                print("2. d) inside view._update_slice, axis == 0 (axial)")
                for out_obj_id, out_mask in self.axial_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            if axis==1:
                print("2. d) inside view._update_slice, axis == 1 (coronal)")
                for out_obj_id, out_mask in self.coronal_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            if axis==2:
                print("2. d) inside view._update_slice, axis == 2 (sagittal)")
                for out_obj_id, out_mask in self.sagittal_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
        
        canvas.draw()
        self.update_idletasks()
        self.update()

    def _import_image(self):
        """
        Opens a file dialog to select a .nii file and loads it.
        """
        file_path = tk.filedialog.askopenfilename(
            title="Select a NIfTI file",
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )

        if not file_path:
            return
        
        if os.path.isfile(file_path) and file_path.endswith(".nii"):
            self.controller.load_image(file_path)
        elif os.path.isdir(file_path):
            dicom_files = [f for f in os.listdir(file_path) if f.lower().endswith(".dcm")]
            if dicom_files:
                self.controller.load_image(file_path, False)
            else:
                print("The selected folder does not contain any DICOM files.")
        else:
            print("Invalid selection. Please select a .nii file or a folder containing DICOM files.")

    def _segment_image(self):
        """
        image segmentation functionality.
        """
        if self.last_used_axis is None or self.last_used_slice_index is None:
            print("No slice selected for segmentation.")
            return
        axis_str = self.last_used_axis
        if "Axial" in axis_str:
            axis = 0
            axis_str_suffix = "AXIAL"
        elif "Coronal" in axis_str:
            axis = 1
            axis_str_suffix = "CORONAL"
        elif "Sagittal" in axis_str:
            axis = 2
            axis_str_suffix = "SAGITTAL"
        else:
            raise ValueError("Invalid axis string.")
        
        slice_array = self._slice_request_callback(axis, self.last_used_slice_index) # get slice array from most recent canvas/axes
        
        points = defaultdict(list) # obj_id -> (x, y, pos (1) or neg (0) flag)
        for idx, point in enumerate(self.points_listbox.get(0, 'end')):
            x, y = point.strip('()').split(',')
            color = self.points_listbox.itemcget(idx, "fg")
            k = 1 if color == "red" else 2 if color == "green" else 3
            points[k].append((int(x), int(y), 1))

        self.controller.segment_image(slice_array, points, self.last_used_slice_index, axis_str_suffix)
    
    def show_image(self):
        self._build_image_frames()
    
    def show_segmentation(self, segmentation_mask):
        """
        Display the segmentation mask in view of self.last_used_axis.
        """
        print("1. a) inside view.show_segmentation")
        axis_str = self.last_used_axis

        if "Axial" in axis_str:
            print("1. b. a) inside view.show_segmentation axial")
            axis = self.axial_view
            self.axial_view_mask=segmentation_mask
            label = "Axial View"
            axis_num=0
            print("1. b. b) inside view.show_segmentation axial")
        elif "Coronal" in axis_str:
            print("1. c. a) inside view.show_segmentation coronal")
            axis = self.coronal_view
            self.coronal_view_mask=segmentation_mask
            label = "Coronal View"
            axis_num=1
            print("1. c. b) inside view.show_segmentation coronal")
        elif "Sagittal" in axis_str:
            print("1. d. a) inside view.show_segmentation sagittal")
            axis = self.sagittal_view
            self.sagittal_view_mask=segmentation_mask
            label = "Sagittal View"
            axis_num=2
            print("1. d. b) inside view.show_segmentation sagittal")
        else:
            raise ValueError("Invalid axis string.")
        canvas = axis.canvas
        canvas.checkbox.config(state="normal")
        canvas.show_mask_var.set(True)

        self._update_slice(canvas.figure.axes[0], canvas, axis_num, self.last_used_slice_index, label)  
        print("1. e) inside view.show_segmentation")      

    def set_on_click_callback(self, callback):
        """Sets the callback invoked on mouse click in the figure."""
        self._on_click_callback = callback

    def set_undo_callback(self, callback):
        """Sets the callback invoked when the user clicks Undo."""
        self._undo_callback = callback

    def set_redo_callback(self, callback):
        """Sets the callback invoked when the user clicks Redo."""
        self._redo_callback = callback

    def set_slice_request_callback(self, callback):
        """Sets the callback to request a slice from the Model via the Controller."""
        self._slice_request_callback = callback

    def _on_click(self, event, ax, canvas):
        """
        Internal method to pass click events to the controller's on_click.
        """
        if event.inaxes is None:
            return
        
        self.last_used_axis = ax.get_title()
        print(f"last used canvas {canvas.slider.get()}")
        self.last_used_slice_index = int(canvas.slider.get())

        if self._on_click_callback is not None:
            color = self.pointer_color_var.get() if self.pointer_color_var else "Red"
            self._on_click_callback(event, color)

        # Redraw after any changes
        canvas.draw()

    def _on_undo_click(self):
        if self._undo_callback:
            self._undo_callback()

    def _on_redo_click(self):
        if self._redo_callback:
            self._redo_callback()

    def add_point_to_listbox(self, x, y):
        color = self.pointer_color_var.get() if self.pointer_color_var and self.pointer_color_var.get() else "Red"
        self.points_listbox.insert("end", f"({x},{y})")
        idx=self.points_listbox.size()-1
        try:
            """
            Tkinters standard Listbox widget doesnt offer robust per-item styling in all versions.
            If Tk version supports it (typically Tk 8.6 or later), you can use the Listbox's item configuration to set the foreground color for each item.
            """
            self.points_listbox.itemconfig(idx, {'fg': color.lower()})
        except Exception as e:
            print(f"Could not set item color: {e}")
        self.points_listbox.yview_moveto(1.0)

    def remove_last_point_from_listbox(self):
        if self.points_listbox.size() > 0:
            self.points_listbox.delete("end")

    def clear_listbox(self):
        self.points_listbox.delete(0, "end")

    def plot_point(self, x, y, color):
        """
        Plot the point on the 'most recently used' Axes (which is the last user-clicked Axes).
        Since we have multiple Axes, we can track the event.inaxes or store references from the event.
        """
        # We can glean the current figure from plt.gcf(), but typically you'd keep references.
        ax = plt.gca()
        mpl_color = {'red': 'r', 'green': 'g', 'blue': 'b'}.get(color.lower(), 'r')
        return ax.plot(x, y, mpl_color + 'o')[0]

    def draw_canvas(self):
        """
        Redraw the active matplotlib figure.
        """
        plt.gcf().canvas.draw()
    
    def update_mesh_view(self, video_segments):
        """
        Update mesh view
        """
        print("******** START: UPDATE MESH VIEW ********")
        # Step 1: Convert segmented frames into a 3D volume
        z_dim = len(video_segments) # Number of frames
        first_frame_object_ids = list(video_segments[0].keys())

        shape = video_segments[0][first_frame_object_ids[0]].shape # Shape of the mask
        x_dim = shape[1]
        y_dim = shape[2]
        combined_meshes = {obj_id : np.zeros((z_dim, x_dim, y_dim), dtype=int) for obj_id in first_frame_object_ids}
        combined_mesh = np.zeros((z_dim, x_dim, y_dim), dtype=int)

        # Populate the 3D array with the segmentation masks and image data
        for z, frame_data in video_segments.items():
            # inner loop necessary to support multiple objects (segmentation masks) in a single frame
            for obj_id, mask in frame_data.items():
                combined_meshes[obj_id][z, :, :] = np.squeeze(mask)  # Assign '1' to masked regions

        # non_segmented_mesh = np.ones_like(combined_mesh) - combined_mesh

        # Create a 3D plot
        fig = self.mesh_view.canvas.figure
        fig.clf()  # Clear current figure
        ax = fig.add_subplot(111, projection='3d')

        cmap = plt.get_cmap('tab10')
        for obj_id, combined_mesh in combined_meshes.items():
            verts, faces, _, _ = measure.marching_cubes(combined_mesh, level=0.5)
            color = cmap(obj_id % 10)
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=color, alpha=0.7)

        print("3D MESH COMPUTATION FINISHED.")

        #ax.plot_trisurf(verts_non_segmented[:, 0], verts_non_segmented[:, 1], faces_non_segmented, verts_non_segmented[:, 2], color='grey', alpha=0.3)
        
        # Customize the plot (optional)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Mesh of Video Segmentation')  

        self.mesh_view.canvas.draw()
        

        """
        OLD APPROACH

        frame_indices = sorted(list(video_segments.keys()))
        h,w=list(video_segments.values())[0][list(video_segments.values())[0].keys()[0]].shape
        num_slices = len(frame_indices)
        volume = np.zeros((num_slices, h, w), dtype=np.uint8)

        for i, frame_idx in enumerate(frame_indices):
            for obj_id, mask in video_segments[frame_idx].items():
                volume[i] = np.maximum(volume[i], mask.astype(np.uint8))

        # Step 2: Apply a 3D surface extraction algorithm (Marching Cubes alternative)
        verts, faces = ndimage.measurements.center_of_mass(volume, labels=volume, index=np.unique(volume)[1:]), []
        
        for z in range(volume.shape[0] - 1):
            for y in range(volume.shape[1] - 1):
                for x in range(volume.shape[2] - 1):
                    cube = volume[z:z+2, y:y+2, x:x+2]
                    if np.any(cube):
                        faces.append([(x, y, z), (x+1, y, z), (x, y+1, z)])
                        faces.append([(x+1, y+1, z), (x, y+1, z), (x+1, y, z)])

        faces = np.array(faces)

        # Step 3: Create a mesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Step 4: Visualize the 3D mesh using Matplotlib
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh_plot = Poly3DCollection(verts[faces], alpha=0.7)
        mesh_plot.set_facecolor((0.3, 0.6, 1, 0.6))  # Light blue with transparency
        ax.add_collection3d(mesh_plot)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, volume.shape[2])
        ax.set_ylim(0, volume.shape[1])
        ax.set_zlim(0, volume.shape[0])
        plt.show()
        """
        print("******** END: UPDATE MESH VIEW ********")

    def _on_close(self):
        """
        Called when the user closes the window via the title bar or otherwise.
        """
        self.quit()
        self.destroy()
