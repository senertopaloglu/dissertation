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
import matplotlib.colors as mcolors

try:
    import nibabel as nib
except ImportError:
    nib = None

class MainView(tk.Tk):
    """
    The View in our MVC. Responsible for building and displaying the GUI.
    For all user interactions, we will invoke Controller callbacks.
    """
    def __init__(self, model, controller):
        super().__init__()

        self.model = model
        self.controller = controller

        # map pointer ID (1-10) to color names
        self.pointer_color_mapping = {
            1: "red",
            2: "blue",
            3: "green",
            4: "orange",
            5: "purple",
            6: "cyan",
            7: "magenta",
            8: "teal",
            9: "black",
            10: "gray"
        }


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

        btn_export = ttk.Button(self.sidebar, text="Export 3D Mesh Model", command=self._export_3d_mesh)
        btn_export.pack(fill="x", pady=5)

        tabControl = ttk.Notebook(self.sidebar)
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)

        tabControl.add(tab1, text="Axial")
        tabControl.add(tab2, text="Coronal")
        tabControl.add(tab3, text="Sagittal")
        tabControl.pack(fill="x", pady=5) # TODO: if doesnt work, try pack(expand=1, fill="both")

        tabs = [tab1, tab2, tab3]

        self.tabControl = tabControl
        self.tabs = tabs

        for tab in tabs:
            tab.pointer_color_var = None
            tab.pointer_color_optionmenu = None
            tab.points_listbox = None
            tab.points = []
            tab.line_objects = []
            tab.undo_stack = []
            tab.redo_stack = []

        for tab in tabs:
            pointer_label = tk.Label(tab, text="Pointer Colour")
            pointer_label.pack(pady=(10, 2))

            pointer_color_var = tk.StringVar(value="Red")
            tab.pointer_color_var = pointer_color_var

            pos_click_var = tk.BooleanVar(value=True)
            tab.pos_click_var = pos_click_var
            pos_click_checkbox = tk.Checkbutton(
                tab,
                text="Positive Click",
                variable=pos_click_var
            )
            if not self.model.image:
                pos_click_checkbox.config(state="disabled")

            pos_click_checkbox.pack(pady=(5,2))
            tab.pos_click_checkbox = pos_click_checkbox

            colors = ["Red", "Blue", "Green", "Orange", "Purple", "Cyan", "Magenta", "Teal", "Black", "Gray"]

            tab.pointer_color_optionmenu = tk.OptionMenu(tab, tab.pointer_color_var, *colors)
            tab.pointer_color_optionmenu.pack(fill="x")

            # Access the underlying menu and configure each item's text color
            menu = tab.pointer_color_optionmenu["menu"]
            for i, color in enumerate(colors):
                menu.entryconfig(i, foreground=color.lower())

            # Define a callback to update the OptionMenu button color.
            def update_option_menu_color(*args, current_tab=tab):
                selected = current_tab.pointer_color_var.get().lower()  # Convert to lowercase for consistency.
                current_tab.pointer_color_optionmenu.config(fg=selected, activeforeground=selected)
                # Check if there is any point in current_tab.points with the same color.
                points_for_color = [pt for pt in current_tab.points if pt[2].lower() == selected]
                if points_for_color:
                    # There is at least one point with the current color; allow toggling.
                    current_tab.pos_click_checkbox.config(state="normal")
                else:
                    # No point with the current color yet, so force positive and disable toggling.
                    current_tab.pos_click_var.set(True)
                    current_tab.pos_click_checkbox.config(state="disabled")
            
            tab.pointer_color_var.trace_add("write", update_option_menu_color)

            

            # Set the default text color to red at startup
            update_option_menu_color()
            
            

            points_label = tk.Label(tab, text="Selected Points")
            points_label.pack(pady=(10, 2))

            points_frame = tk.Frame(tab)
            points_frame.pack(fill="x")

            scrollbar = tk.Scrollbar(points_frame, orient="vertical")
            tab.points_listbox = tk.Listbox(
                points_frame,
                height=5,
                yscrollcommand=scrollbar.set
            )
            tab.points_listbox.pack(side="left", fill="x", expand=True)
            scrollbar.config(command=tab.points_listbox.yview)
            scrollbar.pack(side="right", fill="y")

            btn_undo = ttk.Button(tab, text="Undo", command=self._on_undo_click)
            btn_undo.pack(fill="x", pady=2)
            btn_redo = ttk.Button(tab, text="Redo", command=self._on_redo_click)
            btn_redo.pack(fill="x", pady=2)

            tab.btn_segment = ttk.Button(tab, text="Segment Image", command=self._segment_image)
            tab.btn_segment.pack(fill="x", pady=5)

            tab.btn_export_view = ttk.Button(tab, text="Export View with Segmentation Mask", command=lambda: self._export_view_with_mask(binary=True))
            tab.btn_export_view.pack(fill="x", pady=2)

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
        self.mesh_view = self._create_mesh_view_frame("Segmentation Result (3D Mesh View)")

        # Attach these views to the grid
        self.axial_view.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.coronal_view.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.sagittal_view.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.mesh_view.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        self.update_idletasks()
        self.update()
    
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            if obj_id is not None and obj_id in self.pointer_color_mapping:
                base_color = self.pointer_color_mapping[obj_id]
                color = np.array([*mcolors.to_rgb(base_color), 0.6])
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _create_image_frame(self, text, axis):
        """
        Helper to create a labeled frame with a matplotlib FigureCanvas and a slider.
        """
        frame = tk.Frame(self, bd=1, relief="solid")
        
        label = tk.Label(frame, text=text)
        label.pack()

        control_frame = tk.Frame(frame)
        control_frame.pack(side="top", fill="x", padx=5, pady=5)

        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.tab_index = axis
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        frame.canvas = canvas
        canvas.axis = axis
        frame.canvas_ax=ax

        # Connect click events
        fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: self._on_click(event, ax, canvas)
        )

        if self.model.image:
            sizes = self.model.image.GetLargestPossibleRegion().GetSize()
            dim = 2 if axis == 0 else 1 if axis == 1 else 0
            max_slice = sizes[dim] - 1

            # initialise show/hide mask and show/hide points vars before slider.set
            show_mask_var=tk.BooleanVar()
            canvas.show_mask_var = show_mask_var

            show_points_var=tk.BooleanVar(value=True) 
            canvas.show_points_var = show_points_var

            # Slider to navigate slices
            slider = ttk.Scale(
                control_frame, 
                from_=0, 
                to=max_slice, 
                orient="horizontal",
                command=lambda val: self._update_slice(ax, canvas, axis, val, text)
            )
            slider.pack(side="top", fill="x", expand=True)
            
            initial_slice = max_slice // 2
            slider.set(initial_slice)

            # checkbox to show/hide segmentation mask, show_mask_var is auto updated on click
            show_mask_checkbox = tk.Checkbutton(
                control_frame,
                text="Show Segmentation Mask",
                variable=show_mask_var,
                command=lambda: self._update_slice(ax, canvas, axis, int(canvas.slider.get()), text) # need to re-render image (either with or without mask depending on canvas.show_mask_var)
            )
            show_mask_checkbox.pack(side="top", fill="x", pady=5)

            if axis==0 and self.axial_view_mask is None:
                show_mask_checkbox.config(state="disabled")
            if axis==1 and self.coronal_view_mask is None:
                show_mask_checkbox.config(state="disabled")
            if axis==2 and self.sagittal_view_mask is None:
                show_mask_checkbox.config(state="disabled")

            canvas.show_mask_checkbox = show_mask_checkbox

            # checkbox to show/hide points
            show_points_checkbox = tk.Checkbutton(
                control_frame,
                text="Show Points",
                variable=show_points_var,
                command=lambda: self._update_slice(ax, canvas, axis, int(canvas.slider.get()), text)
            )
            show_points_checkbox.pack(side="top", fill="x", pady=5)
            # initially disable the checkbox because there are no points to show yet
            show_points_checkbox.config(state="disabled")
            canvas.show_points_checkbox = show_points_checkbox
            
            # initialize the slice display
            self._update_slice(ax, canvas, axis, initial_slice, text)

            canvas.slider = slider

        return frame
    
    def _create_mesh_view_frame(self, text):
        """
        Creates a dedicated frame for the 3D mesh view that is initially empty and
        ignores click events.
        """
        frame = tk.Frame(self, bd=1, relief="solid")
        
        label = tk.Label(frame, text=text)
        label.pack()
        frame.label = label

        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        frame.canvas = canvas
        frame.canvas_ax=ax

        return frame


    def _update_slice(self, ax, canvas, axis, val, text):
        """
        Update the displayed slice in the given axis whenever the slider changes.
        """
        if self._slice_request_callback is None:
            return
        
        slice_index = int(float(val))
        slice_array = self._slice_request_callback(axis, slice_index)
        ax.clear()
        ax.imshow(slice_array, cmap='gray')
        ax.set_title(f"{text} - Slice {slice_index}")

        if canvas.show_mask_var.get():
            if axis == 0:
                for out_obj_id, out_mask in self.axial_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            if axis==1:
                for out_obj_id, out_mask in self.coronal_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            if axis==2:
                for out_obj_id, out_mask in self.sagittal_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
        
        if canvas.show_points_var.get():
            current_tab = self.tabs[axis]
            for point in current_tab.points:
                x, y, color, _ = point
                self.plot_point(x, y, color, ax)

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
        
        active_index = self.tabControl.index("current")
        current_tab = self.tabs[active_index]

        if active_index == 0:
            axis = 0
            axis_str_suffix = "AXIAL"
            frame_idx = int(self.axial_view.canvas.slider.get())
            slice_array = self._slice_request_callback(axis, frame_idx)
        elif active_index == 1:
            axis = 1
            axis_str_suffix = "CORONAL"
            frame_idx = int(self.coronal_view.canvas.slider.get())
            slice_array = self._slice_request_callback(axis, frame_idx)
        elif active_index == 2:
            axis = 2
            axis_str_suffix = "SAGITTAL"
            frame_idx = int(self.sagittal_view.canvas.slider.get())
            slice_array = self._slice_request_callback(axis, frame_idx)
        else:
            raise ValueError("Invalid axis string.")
            return
        
        points = defaultdict(list) # obj_id -> (x, y, pos (1) or neg (0) flag)
        
        for idx, entry in enumerate(current_tab.points_listbox.get(0, 'end')):
            pos_flag = 1 if "Positive click" in entry else 0
            x, y = entry.split(' at ')[-1].strip('()').split(',')
            color = current_tab.points_listbox.itemcget(idx, "fg")
            obj_id = self.pointer_color_mapping.get(color, 1)
            points[obj_id].append((int(x), int(y), int(pos_flag)))

        self.controller.segment_image(slice_array, points, frame_idx, axis_str_suffix)
    
    def show_image(self):
        self._build_image_frames()
    
    def show_segmentation(self, segmentation_mask, axis_str_suffix):
        """
        Display the segmentation mask in view of self.last_used_axis.
        """
        axis_str = self.last_used_axis

        if axis_str_suffix == "AXIAL":
            axis = self.axial_view
            self.axial_view_mask=segmentation_mask
            label = "Axial View"
            axis_num=0
        elif axis_str_suffix == "CORONAL":
            axis = self.coronal_view
            self.coronal_view_mask=segmentation_mask
            label = "Coronal View"
            axis_num=1
        elif axis_str_suffix == "SAGITTAL":
            axis = self.sagittal_view
            self.sagittal_view_mask=segmentation_mask
            label = "Sagittal View"
            axis_num=2
        else:
            raise ValueError("Invalid axis string.")
            return
        canvas = axis.canvas
        canvas.show_mask_checkbox.config(state="normal")
        canvas.show_mask_var.set(True)
        slice_idx = int(canvas.slider.get())

        self._update_slice(canvas.figure.axes[0], canvas, axis_num, slice_idx, label)      

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
        self.last_used_slice_index = int(canvas.slider.get())

        if self._on_click_callback is not None:
            # Use the canvas’ own tab index if available; fallback to current tab.
            active_index = getattr(canvas, "tab_index", self.tabControl.index("current"))
            current_tab = self.tabs[active_index]
            color = current_tab.pointer_color_var.get() if current_tab.pointer_color_var else "Red"
            
            self._on_click_callback(event, color)

        # Redraw after any changes
        canvas.draw()

    def _on_undo_click(self):
        if self._undo_callback:
            self._undo_callback()

    def _on_redo_click(self):
        if self._redo_callback:
            self._redo_callback()

    def add_point_to_listbox(self, x, y, pos_flag, color=None, active_index=None):
        if active_index is None:
            active_index = self.tabControl.index("current")
        current_tab = self.tabs[active_index]
        
        if color is None:
            color = current_tab.pointer_color_var.get() or "Red"
        
        prefix = "Positive click" if pos_flag else "Negative click"
        current_tab.points_listbox.insert("end", f"{prefix} at ({x},{y})")
        idx=current_tab.points_listbox.size()-1
        try:
            """
            Tkinters standard Listbox widget doesnt offer robust per-item styling in all versions.
            If Tk version supports it (typically Tk 8.6 or later), you can use the Listbox's item configuration to set the foreground color for each item.
            """
            current_tab.points_listbox.itemconfig(idx, {'fg': color.lower()})
        except Exception as e:
            print(f"Could not set item color: {e}")
        current_tab.points_listbox.yview_moveto(1.0)

        if active_index==0:
            self.axial_view.canvas.show_points_checkbox.config(state="normal")
        if active_index==1:
            self.coronal_view.canvas.show_points_checkbox.config(state="normal")
        if active_index==2:
            self.sagittal_view.canvas.show_points_checkbox.config(state="normal")

    def remove_last_point_from_listbox(self):
        active_index = self.tabControl.index("current")
        current_tab = self.tabs[active_index]
        if current_tab.points_listbox.size() > 0:
            current_tab.points_listbox.delete("end")
        else:
            if active_index == 0:
                self.axial_view.canvas.show_points_checkbox.config(state="disabled")
            elif active_index == 1:
                self.coronal_view.canvas.show_points_checkbox.config(state="disabled")
            elif active_index == 2:
                self.sagittal_view.canvas.show_points_checkbox.config(state="disabled")

    def clear_listbox(self):
        active_index = self.tabControl.index("current")
        current_tab = self.tabs[active_index]
        current_tab.points_listbox.delete(0, "end")
        
        canvas = None
        if active_index == 0:
            canvas = self.axial_view.canvas
        elif active_index == 1:
            canvas = self.coronal_view.canvas
        elif active_index == 2:
            canvas = self.sagittal_view.canvas
        
        if canvas and hasattr(canvas, 'show_points_checkbox'):
            canvas.show_points_checkbox.config(state="disabled")

    def plot_point(self, x, y, color, ax=None):
        """
        Plot the point on the 'most recently used' Axes (which is the last user-clicked Axes).
        Since we have multiple Axes, we can track the event.inaxes or store references from the event.
        """
        # We can glean the current figure from plt.gcf(), but typically you'd keep references.
        if ax is None:
            ax = plt.gca()
        return ax.plot(x, y, marker='o', color=color.lower())[0]

    def draw_canvas(self, ax_idx=None):
        """
        Redraw the active matplotlib figure.
        """
        if ax_idx is None:
            plt.gcf().canvas.draw()
            return
        if ax_idx == 0:
            canvas = self.axial_view.canvas
        elif ax_idx == 1:
            canvas = self.coronal_view.canvas
        elif ax_idx == 2:
            canvas = self.sagittal_view.canvas
        else:
            canvas = plt.gcf().canvas  # fallback, though it should not happen
        canvas.draw()

    def reset_views(self):
        """
        Force a refresh of all image frames to remove lingering drawn points.
        """
        if self.model.image:
            try:
                slider_val = int(self.axial_view.canvas.slider.get())
                self._update_slice(self.axial_view.canvas.figure.axes[0],
                                   self.axial_view.canvas, 0, slider_val, "Axial View")
            except Exception as e:
                print("Error resetting axial view:", e)
            try:
                slider_val = int(self.coronal_view.canvas.slider.get())
                self._update_slice(self.coronal_view.canvas.figure.axes[0],
                                   self.coronal_view.canvas, 1, slider_val, "Coronal View")
            except Exception as e:
                print("Error resetting coronal view:", e)
            try:
                slider_val = int(self.sagittal_view.canvas.slider.get())
                self._update_slice(self.sagittal_view.canvas.figure.axes[0],
                                   self.sagittal_view.canvas, 2, slider_val, "Sagittal View")
            except Exception as e:
                print("Error resetting sagittal view:", e)
    
    def update_mesh_view(self, video_segments, axis_str_suffix):
        """
        Update mesh view with the segmentation result and update the label to
        reflect the view used for segmentation.
        """
        # Step 1: Convert segmented frames into a 3D volume
        z_dim = len(video_segments) # Number of frames
        first_frame_object_ids = list(video_segments[0].keys())

        shape = video_segments[0][first_frame_object_ids[0]].shape # Shape of the mask
        x_dim = shape[1]
        y_dim = shape[2]
        combined_meshes = {obj_id : np.zeros((z_dim, x_dim, y_dim), dtype=int) for obj_id in first_frame_object_ids}

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

        downsample_factor = 0.85 # Downsample factor for faster rendering

        for obj_id, combined_mesh in combined_meshes.items():
            downsampled = ndimage.zoom(combined_mesh, zoom=downsample_factor, order=0)
            verts, faces, _, _ = measure.marching_cubes(downsampled, level=0.5) # faster, optimised version of measure.marching_cubes
            verts = verts / downsample_factor  # Rescale to original dimensions
            base_color = self.pointer_color_mapping.get(obj_id, "red")
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=base_color, alpha=0.7)
        
        # Customize the plot (optional)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Mesh View')  

        self.mesh_view.canvas.draw()

        # update the mesh view label
        if hasattr(self.mesh_view, "label"):
            new_text = f"{axis_str_suffix.title()} Segmentation Result (3D Mesh View)"
            self.mesh_view.label.config(text=new_text)
        
        # store the segmentation result for later exporting
        self.last_video_segments = video_segments
    
    def _export_3d_mesh(self):
        """
        Exports the 3D mesh currently displayed in the mesh view as an STL file.
        The method uses the latest segmentation (stored in self.last_video_segments)
        to recompute the mesh using marching cubes and then exports it.
        """
        try:
            from stl import mesh
        except ImportError:
            tk.messagebox.showerror("Export Error", "The 'numpy-stl' package is required for exporting 3D meshes.")
            return
    
        # ensure segmentation has been run, else, 3d segmentation mesh will not exist
        if not hasattr(self, "last_video_segments"):
            tk.messagebox.showerror("Error", "No segmentation available to export.")
            return

        video_segments = self.last_video_segments

        # Assume all frames have the same dimensions and combine them per object.
        z_dim = len(video_segments)
        first_frame_object_ids = list(video_segments[0].keys())
        shape = video_segments[0][first_frame_object_ids[0]].shape  # (1, H, W) or (H, W)
        x_dim = shape[1]
        y_dim = shape[2]
        combined_meshes = {obj_id: np.zeros((z_dim, x_dim, y_dim), dtype=int) for obj_id in first_frame_object_ids}

        for z, frame_data in video_segments.items():
            for obj_id, mask in frame_data.items():
                # remove any extra dimensions if needed
                combined_meshes[obj_id][z, :, :] = np.squeeze(mask)
        
        stl_meshes = []
        for obj_id, combined_mesh in combined_meshes.items():
            try:
                verts, faces, _, _ = measure.marching_cubes(combined_mesh, level=0.5)
            except Exception as e:
                continue
        
            # convert faces into triangles using the vertices
            triangles = verts[faces] # shape: (n_faces, 3, 3)
            m = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
            for i, triangle in enumerate(triangles):
                m.vectors[i] = triangle
            stl_meshes.append(m)
        
        if not stl_meshes:
            tk.messagebox.showerror("Export Error", "No valid mesh could be generated.")
            return
    
        # combine all mesh data into one STL
        combined_data = np.concatenate([m.data for m in stl_meshes])
        exported_mesh = mesh.Mesh(combined_data.copy())

        export_filename = tk.filedialog.asksaveasfilename(
            title="Export 3D Mesh as STL",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            defaultextension=".stl"
        )
        if not export_filename:
            return
        
        exported_mesh.save(export_filename)
        tk.messagebox.showinfo("Export Successful", f"3D mesh exported as '{export_filename}'.")

    def _export_view_with_mask(self, binary=False):
        """
        Exports the original image with overlayed segmentation mask in the active view
        as a 3D NIfTI file.
        """
        alpha = 0.4

        active_index = self.tabControl.index("current")

        if active_index == 0:
            if self.axial_view_mask is None:
                tk.messagebox.showerror("Export Error", "No segmentation mask available for axial view.")
                return
            original_np = np.asarray(self.model.image)
            mask_dict = self.axial_view_mask
        elif active_index == 1:
            if self.coronal_view_mask is None:
                tk.messagebox.showerror("Export Error", "No segmentation mask available for coronal view.")
                return
            original_np = np.asarray(self.model.image)  # Expect shape (Z, H, W)
            mask_dict = self.coronal_view_mask
        elif active_index == 2:
            # Sagittal view: slices along axis 2
            if self.sagittal_view_mask is None:
                tk.messagebox.showerror("Export Error", "No segmentation mask available for Sagittal view.")
                return
            original_np = np.asarray(self.model.image)  # Expect shape (Z, H, W)
            mask_dict = self.sagittal_view_mask
        else:
            tk.messagebox.showerror("Export Error", "Invalid view selected.")
            return
        
        sizes = self.model.image.GetLargestPossibleRegion().GetSize()
        dim = 2 if active_index == 0 else 1 if active_index == 1 else 0
        max_slice = sizes[dim] - 1

        composite_slices = []
        last_slice = None

        original_np = np.asarray(self.model.image)
        num_slices = original_np.shape[active_index]

        for i in range(num_slices):
            # get the correct original slice for the orientation.
            if active_index == 0:
                # axial view: slices along axis 0
                orig_slice = original_np[i, :, :]
            elif active_index == 1:
                # coronal view: slices along axis 1
                orig_slice = original_np[:, i, :]
            elif active_index == 2:
                # sagittal view: slices along axis 2
                orig_slice = original_np[:, :, i]
                orig_slice = np.rot90(orig_slice, k=-1)
                orig_slice = np.fliplr(orig_slice)

            if last_slice is not None:
                print(f"is same: {last_slice == orig_slice}")
            last_slice = orig_slice.copy()

            # normalise original slice to 0-255
            slice_min, slice_max = orig_slice.min(), orig_slice.max()
            if slice_max > slice_min:  # to avoid divide-by-zero
                norm_slice = (orig_slice - slice_min) / (slice_max - slice_min)
            else:
                norm_slice = orig_slice * 0  # all zeros if it's a uniform slice
            norm_slice_255 = (norm_slice * 255).astype(np.uint8)

            # convert grayscale to RGB
            rgb = np.stack([norm_slice_255] * 3, axis=-1).astype(np.float32)
            
            # in binary mode, start with a black canvas, else start with original image
            if binary:
                composite = np.zeros_like(rgb)
            else:
                composite = rgb.copy()

            # if a segmentation mask exists for this slice, overlay each object. 
            if i in mask_dict:
                for obj_id, mask in mask_dict[i].items():
                    
                    if active_index == 2:
                        mask = np.squeeze(mask.astype(np.float32))
                        mask = np.rot90(mask, k=-1)
                        mask = np.fliplr(mask)
                        mask_expanded = mask[..., np.newaxis] # expand *exactly one* axis for alpha blending
                    else:
                        mask_expanded = np.expand_dims(mask.astype(np.float32), axis=-1)

                    if binary:
                        print(f"mask expanded: {mask_expanded.shape}")
                        binary_mask = (mask_expanded > 0).squeeze()
                        print(f"binary mask: {binary_mask.shape}")
                        composite[binary_mask] = 255  # Set the mask area to white
                    else:
                        # Get pointer color for this object, default to red if missing.
                        color_name = self.pointer_color_mapping.get(obj_id, "red")
                        rgb_color = np.array(mcolors.to_rgb(color_name)) * 255
                        # Prepare an overlay of the pointer color.
                        overlay = np.zeros_like(composite)
                        overlay[:, :, 0] = rgb_color[0]
                        overlay[:, :, 1] = rgb_color[1]
                        overlay[:, :, 2] = rgb_color[2]
                        # Alpha blend the overlay where mask is True.
                        composite = (1 - alpha * mask_expanded) * composite + (alpha * mask_expanded) * overlay
                if not binary:
                    composite = np.clip(composite, 0, 255)
            
            if composite.ndim == 4 and composite.shape[0] == 1:
                composite = np.squeeze(composite, axis=0)
            
            composite_slices.append(composite.astype(np.uint8))
        
        composite_volume = np.stack(composite_slices, axis=0)  # shape: (num_slices, H, W, 3)

        # Sometimes an extra singleton dimension appears as axis 1; if so, squeeze it:
        if composite_volume.ndim == 5 and composite_volume.shape[1] == 1:
            composite_volume = np.squeeze(composite_volume, axis=1)
        
        # reorient volume
        if active_index == 1: # coronal
            composite_volume = composite_volume.transpose(1, 0, 2, 3)
        if active_index == 2: # sagittal
            composite_volume = composite_volume.transpose(2, 1, 0, 3)

        if nib is None:
            tk.messagebox.showerror("Export Error", "The 'nibabel' package is required for exporting 3D NIfTI files.")
            return

        # Open file dialog for user to choose save location.
        export_filename = tk.filedialog.asksaveasfilename(
            title="Export 3D Image with Segmentation Overlay as NIfTI",
            defaultextension=".nii",
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )
        if not export_filename:
            return

        # Create a NIfTI image and save
        nii_img = nib.Nifti1Image(composite_volume, affine=np.eye(4))
        nib.save(nii_img, export_filename)
        tk.messagebox.showinfo("Export Successful", f"Image with segmentation overlay exported as:\n{export_filename}")

    def clear_mesh_view(self):
        """
        Clears the 3D mesh view (i.e. removes any previously segmented mesh).
        """
        fig = self.mesh_view.canvas.figure
        fig.clf()  # Clear all content from the figure
        # Re-create an empty 3D axes with a title.
        ax = fig.add_subplot(111, projection='3d')
        self.mesh_view.label.configure(text="Segmentation Result (3D Mesh View)")
        ax.set_title("3D Mesh View")
        self.mesh_view.canvas.draw()

    def _on_close(self):
        """
        Called when the user closes the window via the title bar or otherwise.
        """
        self.quit()
        self.destroy()
