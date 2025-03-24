import os

from collections import defaultdict

import tkinter.filedialog as filedialog

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Window, Frame, Label, Button, Notebook, OptionMenu, Scale, Checkbutton

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import scipy.ndimage as ndimage

import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.colors as mcolors

class MainView(Window):
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
            8: "yellow",
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
        self.sidebar = Frame(self, padding=10)
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
        btn_import = Button(self.sidebar, text="Import Image", command=self._import_image, bootstyle="primary")
        btn_import.pack(fill="x", pady=5)

        btn_export = Button(self.sidebar, text="Export Image", bootstyle="secondary")
        btn_export.pack(fill="x", pady=5)

        style = ttk.Style()

        style.configure(
            "DarkerTabs.TNotebook",
            background="white",
            tabmargins=[2, 5, 2, 0] # extra spacing around tabs
        )

        # Define how each tab looks
        style.configure(
            "DarkerTabs.TNotebook.Tab",
            background="#DDDDDD",         # darker gray for inactive tabs
            padding=[10, 4],          # extra space around the tab text
            font=("TkDefaultFont", 10)
        )

        # Map the 'selected' state to a different background/foreground
        style.map(
            "DarkerTabs.TNotebook.Tab",
            background=[("selected", "white")],   # light blue background on active tab
            foreground=[("selected", "black")],      # make the active tab text blue
        )

        tabControl = Notebook(self.sidebar, style="DarkerTabs.TNotebook")
        tab1 = Frame(tabControl)
        tab2 = Frame(tabControl)
        tab3 = Frame(tabControl)

        tabControl.add(tab1, text="Axial")
        tabControl.add(tab2, text="Coronal")
        tabControl.add(tab3, text="Sagittal")
        tabControl.pack(fill="x", pady=5)

        tabs = [tab1, tab2, tab3]

        self.tabControl = tabControl
        self.tabs = tabs

        

        for i, tab in enumerate(tabs):
            tab.style_name = f"PointerColor.TMenubutton.Tab{i}"
            style.layout(tab.style_name, style.layout("TMenubutton"))
            tab.pointer_color_var = None
            tab.pointer_color_optionmenu = None
            tab.points_listbox = None
            tab.points = []
            tab.line_objects = []
            tab.undo_stack = []
            tab.redo_stack = []

        for tab in tabs:
            content_frame = Frame(tab, padding=(10, 5))
            content_frame.pack(fill="both", expand=True)

            pointer_label = Label(content_frame, text="Pointer Colour")
            pointer_label.pack(pady=(10, 2))

            pointer_color_var = ttk.StringVar(value="Red")
            tab.pointer_color_var = pointer_color_var

            pos_click_var = ttk.BooleanVar(value=True)
            tab.pos_click_var = pos_click_var
            pos_click_checkbox = Checkbutton(
                content_frame,
                text="Positive Click",
                variable=pos_click_var
            )
            if not self.model.image:
                pos_click_checkbox.config(state="disabled")

            pos_click_checkbox.pack(pady=(5, 2))
            tab.pos_click_checkbox = pos_click_checkbox

            colors = ["Red", "Blue", "Green", "Orange", "Purple",
                      "Cyan", "Magenta", "Yellow", "Black", "Gray"]

            tab.pointer_color_optionmenu = OptionMenu(content_frame, tab.pointer_color_var, "")
            tab.pointer_color_optionmenu.pack(fill="x")

            tab.pointer_color_optionmenu.configure(textvariable=tab.pointer_color_var)
            # Access the underlying menu and configure each item's text color
            menu = tab.pointer_color_optionmenu["menu"]
            # clear any auto-added items
            menu.delete(0, "end")

            for color in colors:
                menu.add_command(
                    label=color,
                    foreground=color.lower(),
                    background="white",
                    activeforeground="white",
                    activebackground=color.lower(),
                    command=lambda c=color, var=pointer_color_var: var.set(c)
                )
            
            tab.pointer_color_var.set("Red")

            # Define a callback to update the OptionMenu button color.
            def update_option_menu_color(*args, current_tab=tab):
                selected = current_tab.pointer_color_var.get().lower()  # Convert to lowercase for consistency.
                
                # update option menu to have a white bg and 'selected' text color
                style.configure(
                    current_tab.style_name,
                    foreground=selected,
                    background="white",
                    relief="solid",
                    borderwidth=1
                )
                style.map(
                    current_tab.style_name,
                    background=[
                        ("active", "white"),
                        ("pressed", "white")
                    ]
                )
                current_tab.pointer_color_optionmenu.configure(style=current_tab.style_name)

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
            
            

            points_label = Label(content_frame, text="Selected Points")
            points_label.pack(pady=(10, 2))

            points_frame = Frame(content_frame)
            points_frame.pack(fill="x")

            scrollbar = ttk.Scrollbar(points_frame, orient="vertical")
            tab.points_listbox = tk.Listbox(
                points_frame,
                height=5,
                yscrollcommand=scrollbar.set
            )
            tab.points_listbox.pack(side="left", fill="x", expand=True)
            scrollbar.config(command=tab.points_listbox.yview)
            scrollbar.pack(side="right", fill="y")

            btn_undo = Button(content_frame, text="Undo", command=self._on_undo_click, bootstyle="info")
            btn_undo.pack(fill="x", pady=2)
            btn_redo = Button(content_frame, text="Redo", command=self._on_redo_click, bootstyle="info")
            btn_redo.pack(fill="x", pady=2)

            tab.btn_segment = Button(content_frame, text="Segment Image", command=self._segment_image, bootstyle="success")
            tab.btn_segment.pack(fill="x", pady=5)

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
        outer_frame = Frame(self, relief="solid", borderwidth=1)
        
        content_frame = Frame(outer_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        label = Label(content_frame, text=text)
        label.pack(side="top", fill="x", expand=True)
        label.configure(anchor="center", justify="center")

        control_frame = Frame(content_frame)
        control_frame.pack(side="top", fill="x", pady=5)

        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.tab_index = axis
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        outer_frame.canvas = canvas
        canvas.axis = axis
        outer_frame.canvas_ax=ax

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
            show_mask_var = ttk.BooleanVar()
            canvas.show_mask_var = show_mask_var

            show_points_var = ttk.BooleanVar(value=True)
            canvas.show_points_var = show_points_var

            # Slider to navigate slices
            slider = Scale(
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
            show_mask_checkbox = Checkbutton(
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
            show_points_checkbox = Checkbutton(
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

        return outer_frame
    
    def _create_mesh_view_frame(self, text):
        """
        Creates a dedicated frame for the 3D mesh view that is initially empty and
        ignores click events.
        """
        outer_frame = Frame(self, relief="solid", borderwidth=1)

        content_frame = Frame(outer_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        label = Label(content_frame, text=text)
        label.pack()
        outer_frame.label = label

        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        
        outer_frame.canvas = canvas
        outer_frame.canvas_ax = ax

        return outer_frame


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
            if axis == 1:
                for out_obj_id, out_mask in self.coronal_view_mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)
            if axis == 2:
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
        file_path = filedialog.askopenfilename(
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

        for idx, entry in enumerate(current_tab.points_listbox.get(0, 'end')):
            pos_flag = 1 if "Positive click" in entry else 0
            x, y = entry.split(' at ')[-1].strip('()').split(',')
            color = current_tab.points_listbox.itemcget(idx, "fg")
            k = color_map.get(color, 1)
            points[k].append((int(x), int(y), int(pos_flag)))

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
            self.axial_view_mask = segmentation_mask
            label = "Axial View"
            axis_num = 0
        elif axis_str_suffix == "CORONAL":
            axis = self.coronal_view
            self.coronal_view_mask = segmentation_mask
            label = "Coronal View"
            axis_num = 1
        elif axis_str_suffix == "SAGITTAL":
            axis = self.sagittal_view
            self.sagittal_view_mask = segmentation_mask
            label = "Sagittal View"
            axis_num = 2
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
        idx = current_tab.points_listbox.size()-1
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
