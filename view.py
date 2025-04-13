import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D

from type_aliases import SegmentationResult
from export_format import ExportFormat
from model import ImageModel
from sidebar import Sidebar

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Window, Frame, Label, Button, Notebook, OptionMenu, Scale, Checkbutton

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import scipy.ndimage as ndimage

import numpy as np

from skimage import measure

try:
    import nibabel as nib
except ImportError:
    nib = None

class CustomNavigationToolbar2Tk(NavigationToolbar2Tk):
    # Remove the 'Save' tool from the toolbar
    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != 'Save']

class MainView(Window):
    """
    The View in our MVC. Responsible for building and displaying the GUI.
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

        self.color_obj_id_mapping = {v: k for k, v in self.pointer_color_mapping.items()}

        # Keep references to callback functions
        self._on_click_callback = None
        self._undo_callback = None
        self._redo_callback = None
        self._slice_request_callback = None

        self.last_used_axis = None
        self.last_used_slice_index = None

        self.last_result = None
        self.last_draft_result = None

        self.title("Interactive 3D Medical Image Segmentation")
        self.state("zoomed")

        # Prepare main window layout
        self.columnconfigure(0, weight=2, minsize=300)  # Left sidebar
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=3)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=3)

        # Sidebar + main frames
        self.sidebar = Sidebar(self, self.model, self, self.controller) # Frame(self, padding=10)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        plt.ion()  # Enable interactive mode for matplotlib
        
        # Behavior for closing
        self.protocol("WM_DELETE_WINDOW", self._on_close)        

    def _build_image_frames(self) -> None:
        """
        Builds the frames that display the axial, coronal, sagittal, and
        (placeholder) mesh view.
        """
        self.sidebar.controller = self.controller

        self.axial_view_mask = None
        self.coronal_view_mask = None
        self.sagittal_view_mask = None

        self.draft_axial_view_mask = None
        self.draft_coronal_view_mask = None
        self.draft_sagittal_view_mask = None
        
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
    
    def show_mask(self, mask: np.ndarray, ax: Axes, obj_id: Optional[int] = None, random_color: bool = False) -> None:
        """
        Render and display a segmentation mask overlay on the given matplotlib Axes.

        Parameters:
            mask (ndarray): The segmentation mask array.
            ax (matplotlib.axes.Axes): The Axes object where the mask will be overlaid.
            obj_id (int, optional): Object ID to determine the base color mapping. Defaults to None.
            random_color (bool, optional): If True, generates a random color for the mask. Defaults to False.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            if obj_id is not None and obj_id in self.pointer_color_mapping:
                obj_color = self.pointer_color_mapping[obj_id]
                color = np.array([*mcolors.to_rgb(obj_color), 0.6])
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _create_title_frame(self, parent: ttk.Frame, text: str) -> ttk.Frame:
        """
        Create a composite title widget with a left label for the draft prefix (initially empty)
        and a right label for the view title (e.g. "Axial View").
        """
        title_frame = ttk.Frame(parent)

        # Label for "[DRAFT] " (bold)
        draft_text = "[DRAFT] " if self.sidebar.global_draft_mode.get() else ""
        draft_label = ttk.Label(title_frame, text=draft_text, font=("TkDefaultFont", 13, "bold"))
        draft_label.grid(row=0, column=0, sticky="e")
        
        # Label for the view name (normal)
        view_label = ttk.Label(title_frame, text=text, font=("TkDefaultFont", 13))
        view_label.grid(row=0, column=1, sticky="w")

        title_frame.columnconfigure(0, weight=1)
        title_frame.columnconfigure(1, weight=1)

        # Store references for later updates
        title_frame.draft_label = draft_label
        title_frame.view_label = view_label
        return title_frame

    def _create_image_frame(self, text: str, axis: int) -> Frame:
        """
        Creates a labeled frame containing a Matplotlib FigureCanvas along with a slider 
        and checkboxes to control the display of segmentation masks and points.

        Parameters:
            text (str): The title text to be displayed on the frame.
            axis (int): The image axis associated with the frame.

        Returns:
            Frame: A configured Tkinter frame embedding the Matplotlib FigureCanvas and control widgets.
        """
        outer_frame = Frame(self, relief="solid", borderwidth=1)
        
        content_frame = Frame(outer_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        title_frame = self._create_title_frame(content_frame, text)
        title_frame.pack(side="top", fill="x", pady=(5,0))
        outer_frame.title_frame = title_frame

        control_frame = Frame(content_frame)
        control_frame.pack(side="top", fill="x", pady=5)

        fig = Figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.tab_index = axis
        widget = canvas.get_tk_widget()
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

        toolbar = CustomNavigationToolbar2Tk(canvas, control_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.toolbar = toolbar

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
                command=lambda val: self._update_slice(ax, canvas, axis, val)
            )
            slider.pack(side="top", fill="x", expand=True)
            
            initial_slice = max_slice // 2
            slider.set(initial_slice)

            # checkbox to show/hide segmentation mask, show_mask_var is auto updated on click
            show_mask_checkbox = Checkbutton(
                control_frame,
                text="Show Segmentation Masks",
                variable=show_mask_var,
                command=lambda: self._update_slice(ax, canvas, axis, int(canvas.slider.get())) # need to re-render image (either with or without mask depending on canvas.show_mask_var)
            )
            show_mask_checkbox.pack(side="top", anchor="center", pady=5)

            mask = self._get_mask(axis)
            if mask is None:
                show_mask_checkbox.config(state="disabled")

            canvas.show_mask_checkbox = show_mask_checkbox

            # checkbox to show/hide points
            show_points_checkbox = Checkbutton(
                control_frame,
                text="Show Points",
                variable=show_points_var,
                command=lambda: self._update_slice(ax, canvas, axis, int(canvas.slider.get()))
            )
            show_points_checkbox.pack(side="top", anchor="center", pady=(3,2))
            # initially disable the checkbox because there are no points to show yet
            show_points_checkbox.config(state="disabled")
            canvas.show_points_checkbox = show_points_checkbox
            
            # initialize the slice display
            self._update_slice(ax, canvas, axis, initial_slice)

            canvas.slider = slider

        return outer_frame
    
    def _create_mesh_view_frame(self, text: str) -> Frame:
        """
        Creates and returns a dedicated frame for the 3D mesh view.

        This frame is used to display the 3D segmentation mesh. It contains a label
        for descriptive text and an embedded Matplotlib FigureCanvasTkAgg to render
        the 3D plot. Click events are ignored in this view since it is not meant 
        for interactive segmentation.

        Parameters:
            text (str): The title text to be displayed on the mesh view frame.

        Returns:
            Frame: A configured Tkinter frame with an embedded Matplotlib canvas.
        """
        outer_frame = Frame(self, relief="solid", borderwidth=1)

        content_frame = Frame(outer_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        title_frame = self._create_title_frame(content_frame, text)
        title_frame.pack(side="top", fill="x", pady=(5,0))
        outer_frame.title_frame = title_frame  # store for later updates

        fig = Figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        
        outer_frame.canvas = canvas
        outer_frame.canvas_ax = ax

        return outer_frame


    def _update_slice(self, ax: Axes, canvas: FigureCanvasTkAgg, axis: int, val: Union[int, str]) -> None:
        """
        Update the displayed slice in the given axis whenever the slider changes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes where the image slice is rendered.
            canvas (FigureCanvasTkAgg): The canvas containing the Matplotlib figure.
            axis (int): The image axis (0 for axial, 1 for coronal, 2 for sagittal).
            val (int or str): The slider value representing the current slice index.

        Returns:
            None
        """
        if self._slice_request_callback is None:
            return
        
        slice_index = int(float(val))

        # capture current zoom limits to keep them as the underlying slice changes
        if getattr(self, "_preserve_zoom", True):
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

        slice_array = self._slice_request_callback(axis, slice_index)
        ax.clear()
        ax.imshow(slice_array, cmap='gray')
        ax.set_title(f"Slice {slice_index}")

        # restore zoom limits
        if getattr(self, "_preserve_zoom", True):
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                ax.set_xlim(current_xlim)
                ax.set_ylim(current_ylim)
        else:
            ax.autoscale()
            self._preserve_zoom = True

        current_tab = self.sidebar.tabs[axis]

        # show masks
        if canvas.show_mask_var.get():
            if self.sidebar.global_draft_mode.get():
                mask = self._get_mask(axis, is_draft=True)
            else:
                mask = self._get_mask(axis)
            if mask and slice_index in mask:
                for out_obj_id, out_mask in mask[slice_index].items():
                    self.show_mask(out_mask, ax, obj_id=out_obj_id)

        # show points    
        if canvas.show_points_var.get():
            if self.sidebar.global_draft_mode.get():
                for point in current_tab.draft_points:
                    x, y, color, _ = point
                    self.plot_point(x, y, color, ax)
            else:
                for point in current_tab.points:
                    x, y, color, _ = point
                    self.plot_point(x, y, color, ax)

        canvas.draw()
        self.update_idletasks()
        self.update()

    def import_nifti(self) -> None:
        """
        Opens a file dialog to select a .nii file and loads it.
        """
        file_path = tk.filedialog.askopenfilename(
            title="Select a NIfTI file",
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )

        if file_path and os.path.isfile(file_path) and file_path.endswith(".nii"):
            self.controller.load_image(file_path)
        else:
            tk.messagebox.showerror("Import Error", "Please select a valid NIfTI file.")
    
    def import_dicom(self) -> None:
        """
        Opens a file dialog to select a folder containing DICOM files and loads it.
        """
        folder_path = tk.filedialog.askdirectory(
            title="Select a folder containing DICOM files"
        )

        if not folder_path:
            return
        
        dicom_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
        if dicom_files:
            self.controller.load_image(folder_path, False)
        else:
            tk.messagebox.showerror(
                "Import Error",
                "The selected folder does not contain any DICOM files."
            )

    def _import_image(self) -> None:
        """
        Opens a file dialog to select a .nii file and loads it.
        """
        file_path = tk.filedialog.askopenfilename(
            title="Select a NIfTI file (cancel to choose DICOM folder)",
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )

        if file_path and os.path.isfile(file_path) and file_path.endswith(".nii"):
            self.controller.load_image(file_path)
            return

        # If the file dialog is cancelled or the file is not valid,
        # ask for a folder.
        folder_path = tk.filedialog.askdirectory(
            title="Select a folder containing DICOM files"
        )

        if not folder_path:
            return
        
        dicom_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
        if dicom_files:
            self.controller.load_image(folder_path, False)
        else:
            tk.messagebox.showerror(
                "Import Error",
                "The selected folder does not contain any DICOM files."
            )

    def _segment_image(self) -> None:
        """
        Perform image segmentation on the currently selected slice and annotated points.
        """
        if self.last_used_axis is None or self.last_used_slice_index is None:
            tk.messagebox.showerror(
                "Segmentation Error",
                "No slice selected for segmentation.")
            return
        
        if self.sidebar.global_segmentation_var.get():
            points = {}
            for i, tab in enumerate(self.sidebar.tabs):
                # use draft points if draft selection is enabled in tab
                used_points = tab.draft_points if self.sidebar.global_draft_mode.get() else tab.points
                for point in used_points:
                    x, y, color, pos_flag = point
                    obj_id = self.color_obj_id_mapping.get(color, 1)
                    if i == 0:
                        curr_slice = int(self.axial_view.canvas.slider.get())
                        if obj_id not in points:
                            points[obj_id] = defaultdict(list)
                        points[obj_id][curr_slice].append((int(x), int(y), int(pos_flag)))
                    elif i == 1:
                        curr_slice = int(self.coronal_view.canvas.slider.get())
                        if obj_id not in points:
                            points[obj_id] = defaultdict(list)
                        points[obj_id][int(y)].append((int(x), curr_slice, int(pos_flag)))
                    elif i == 2:
                        curr_slice = int(self.sagittal_view.canvas.slider.get())
                        if obj_id not in points:
                            points[obj_id] = defaultdict(list)
                        points[obj_id][int(y)].append((curr_slice, int(x), int(pos_flag)))  
             
            if not points:
                tk.messagebox.showerror(
                    "Input Error",
                    f"No points selected for segmentation."
                )
                return
            
            for k, v in points.items():
                for a, b in v.items():
                    print(f"{k}|{a}|{b}")

            self.controller.segment_image(
                self._slice_request_callback(0, 1),
                points,
                int(self.axial_view.canvas.slider.get()),
                "AXIAL",
                is_final=True,
                is_global=True
            )
        else:
            active_index = self.sidebar.tabControl.index("current")
            current_tab = self.sidebar.tabs[active_index]

            if active_index == 0:
                axis_str_suffix = "AXIAL"
            elif active_index == 1:
                axis_str_suffix = "CORONAL"
            elif active_index == 2:
                axis_str_suffix = "SAGITTAL"
            else:
                raise ValueError("Invalid axis string.")
                return
            
            # use draft points if draft selection is enabled in tab
            if self.sidebar.global_draft_mode.get():
                used_points = current_tab.draft_points
            else:
                used_points = current_tab.points
            
            if not used_points:
                tk.messagebox.showerror(
                    "Input Error",
                    f"No points selected for segmentation."
                )
                return

            axis = active_index
            canvas = self._get_canvas(axis)
            frame_idx = int(canvas.slider.get())
            slice_array = self._slice_request_callback(axis, frame_idx)
        
            # object id -> (x, y, <flag for positive (1) or negative (0) click>)
            points = defaultdict(list)

            for point in used_points:
                x, y, color, pos_flag = point
                obj_id = self.color_obj_id_mapping.get(color.lower(), 1)
                points[obj_id].append((x, y, pos_flag))

            self.controller.segment_image(
                slice_array,
                points,
                frame_idx,
                axis_str_suffix,
                is_final=True,
                is_global=False
            )
    
    def show_image(self) -> None:
        """Re-initializes and displays all image frames for the current image.

        This method builds the image frames (axial, coronal, sagittal, and 3D mesh)
        and renders them on the GUI.
        """
        self._preserve_zoom = False
        self._build_image_frames()
        
    
    def show_segmentation(self, segmentation_mask: SegmentationResult, axis_str_suffix: str) -> None:
        """
        Display the segmentation mask in view of self.last_used_axis.

        Parameters:
            segmentation_mask (dict): A dictionary containing segmentation masks keyed by slice index.
            axis_str_suffix (str): A string representing the image orientation (e.g., "AXIAL", "CORONAL", "SAGITTAL").
        """
        active_index = self.sidebar.tabControl.index("current")
        current_tab = self.sidebar.tabs[active_index]

        # if global segmentation is enabled, update the axial view
        if axis_str_suffix == "AXIAL" or self.sidebar.global_segmentation_var.get():
            axis = self.axial_view
            label = "Axial View"
            axis_num = 0
            if self.sidebar.global_draft_mode.get():
                self.draft_axial_view_mask = segmentation_mask
            else:
                self.axial_view_mask = segmentation_mask
        elif axis_str_suffix == "CORONAL":
            axis = self.coronal_view
            label = "Coronal View"
            axis_num = 1
            if self.sidebar.global_draft_mode.get():
                self.draft_coronal_view_mask = segmentation_mask
            else:
                self.coronal_view_mask = segmentation_mask
        elif axis_str_suffix == "SAGITTAL":
            axis = self.sagittal_view
            label = "Sagittal View"
            axis_num = 2
            if self.sidebar.global_draft_mode.get():
                self.draft_sagittal_view_mask = segmentation_mask
            else:
                self.sagittal_view_mask = segmentation_mask
        else:
            raise ValueError("Invalid axis string.")
            return

        canvas = axis.canvas
        canvas.show_mask_checkbox.config(state="normal")
        canvas.show_mask_var.set(True)
        slice_idx = int(canvas.slider.get())
        self._update_slice(canvas.figure.axes[0], canvas, axis_num, slice_idx)   

    def set_on_click_callback(self, callback: Callable[[Any, str, Any], None]) -> None:
        """Sets the callback invoked on mouse click in the figure."""
        self._on_click_callback = callback

    def set_undo_callback(self, callback: Callable[[], None]) -> None:
        """Sets the callback invoked when the user clicks Undo."""
        self._undo_callback = callback

    def set_redo_callback(self, callback: Callable[[], None]) -> None:
        """Sets the callback invoked when the user clicks Redo."""
        self._redo_callback = callback

    def set_slice_request_callback(self, callback: Callable[[int, int], Any]) -> None:
        """Sets the callback to request a slice from the Model via the Controller."""
        self._slice_request_callback = callback

    def _on_click(self, event: MouseEvent, ax: Axes, canvas: FigureCanvasTkAgg) -> None:
        """
        Internal method to pass click events to the controller's on_click.

        Parameters:
            event: The matplotlib event triggered by a click.
            ax (matplotlib.axes.Axes): The Axes object where the click event occurred.
            canvas (FigureCanvasTkAgg): The canvas containing the matplotlib figure.
        """
        if event.inaxes is None:
            return

        # Do not register pointer clicks if a zoom/pan tool is active.
        if hasattr(canvas, "toolbar") and canvas.toolbar.mode != "":
            return
        
        # return if user has clicked on an empty plot
        if not hasattr(canvas, "slider"):
            return
        
        self.last_used_axis = ax.get_title()
        self.last_used_slice_index = int(canvas.slider.get())

        if self._on_click_callback is not None:
            # Use the canvas’ own tab index if available; fallback to current tab.
            active_index = getattr(canvas, "tab_index", self.sidebar.tabControl.index("current"))
            current_tab = self.sidebar.tabs[active_index]
            color = current_tab.pointer_color_var.get() if current_tab.pointer_color_var else "Red"
            
            self._on_click_callback(event, color, ax)

        # Redraw after any changes
        canvas.draw()

    def _on_undo_click(self, is_draft: bool) -> None:
        """Sets the callback invoked on undo button click."""
        if self._undo_callback:
            self._undo_callback(is_draft)

    def _on_redo_click(self, is_draft: bool) -> None:
        """Sets the callback invoked on redo button click."""
        if self._redo_callback:
            self._redo_callback(is_draft)

    def add_point_to_listbox(self, x: float, y: float, pos_flag: int, is_draft: bool, color: Optional[str] = None, active_index: Optional[int] = None) -> None:
        if active_index is None:
            active_index = self.sidebar.tabControl.index("current")
        current_tab = self.sidebar.tabs[active_index]
        
        if color is None:
            color = current_tab.pointer_color_var.get() or "Red"
        
        prefix = "Positive click" if pos_flag else "Negative click"
        
        if is_draft:
            target_listbox = current_tab.draft_points_listbox
        else:
            target_listbox = current_tab.final_points_listbox

        target_listbox.insert("end", f"{prefix} at ({x},{y})")
        idx = target_listbox.size()-1
        
        try:
            target_listbox.itemconfig(idx, {'fg': color.lower()})
        except Exception as e:
            tk.messagebox.showerror("Item Color Error", f"Could not set item color: {e}")
        target_listbox.yview_moveto(1.0)

        canvas = self._get_canvas(active_index)
        canvas.show_points_checkbox.config(state="normal")

        self.update_global_segmentation_state()

    def remove_last_point_from_listbox(self, is_draft: bool) -> None:
        active_index = self.sidebar.tabControl.index("current")
        current_tab = self.sidebar.tabs[active_index]
        
        if is_draft:
            points_listbox = current_tab.draft_points_listbox
        else:
            points_listbox = current_tab.final_points_listbox
        
        if points_listbox.size() > 0:
            points_listbox.delete("end")
        else:
            canvas = self._get_canvas(active_index)
            canvas.show_points_checkbox.config(state="disabled")

        self.update_global_segmentation_state()

    def clear_listbox(self) -> None:
        active_index = self.sidebar.tabControl.index("current")
        current_tab = self.sidebar.tabs[active_index]
        current_tab.final_points_listbox.delete(0, "end")
        
        canvas = None
        canvas = self._get_canvas(active_index)
        
        if canvas and hasattr(canvas, 'show_points_checkbox'):
            canvas.show_points_checkbox.config(state="disabled")

    def plot_point(self, x: float, y: float, color: str, ax: Optional[Axes] = None) -> Line2D:
        """
        Plot the point on the specified Matplotlib Axes
        If no Axes object is provided, it defaults to the current active Axes.

        Parameters:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            color (str): The color for the point (e.g., "red", "blue").
            ax (matplotlib.axes.Axes, optional): The Axes on which to plot the point. Defaults to None.

        Returns:
            matplotlib.lines.Line2D: The line object representing the plotted point.
        """
        # We can glean the current figure from plt.gcf(), but typically you'd keep references.
        if ax is None:
            ax = plt.gca()
        return ax.plot(x, y, marker='o', color=color.lower())[0]

    def draw_canvas(self, ax_idx: Optional[int] = None) -> None:
        """
        Redraw the Matplotlib figure for the specified view.

        Parameters:
            ax_idx (int, optional): An index representing the view to redraw.
                - 0: Axial view
                - 1: Coronal view
                - 2: Sagittal view
                If None, redraws the current active figure's canvas.
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

    def reset_views(self) -> None:
        """
        Force a refresh of all image frames to remove any residual points.
        """
        if self.model.image:
            try:
                slider_val = int(self.axial_view.canvas.slider.get())
                self._update_slice(self.axial_view.canvas.figure.axes[0],
                                   self.axial_view.canvas, 0, slider_val)
            except Exception as e:
                print("Error resetting axial view:", e)
            try:
                slider_val = int(self.coronal_view.canvas.slider.get())
                self._update_slice(self.coronal_view.canvas.figure.axes[0],
                                   self.coronal_view.canvas, 1, slider_val)
            except Exception as e:
                print("Error resetting coronal view:", e)
            try:
                slider_val = int(self.sagittal_view.canvas.slider.get())
                self._update_slice(self.sagittal_view.canvas.figure.axes[0],
                                   self.sagittal_view.canvas, 2, slider_val)
            except Exception as e:
                print("Error resetting sagittal view:", e)
    
    def update_mesh_view(self) -> None:
        """
        Update the 3D mesh view by aggregating segmentation masks from all views
        that share the same global draft state. If draft mode is checked, uses draft masks;
        otherwise uses the non-draft masks. If no masks exist for the current mode, an empty
        3D mesh view is shown.

        Parameters:
            axis_str_suffix (str): A string indicating the segmentation view orientation
                (e.g., "AXIAL", "CORONAL", or "SAGITTAL").
        """
        is_draft = self.sidebar.global_draft_mode.get()

        original_np = np.asarray(self.model.image)
        A, C, S = original_np.shape

        # collect all object ids from all mask sources across the views
        all_object_ids = set()
        for axis in [0,1,2]:
            if axis == 0:
                mask_source = self.draft_axial_view_mask if is_draft else self.axial_view_mask
            elif axis == 1:
                mask_source = self.draft_coronal_view_mask if is_draft else self.coronal_view_mask
            elif axis == 2:
                mask_source = self.draft_sagittal_view_mask if is_draft else self.sagittal_view_mask
            else:
                mask_source = None

            if mask_source:
                for slice_idx, obj_masks in mask_source.items():
                    all_object_ids.update(obj_masks.keys())
        
        # if no meshes available for the current mode, show empty
        if not all_object_ids:
            fig = self.mesh_view.canvas.figure
            fig.clf()  # Clear current figure
            ax = fig.add_subplot(111, projection='3d')
            if hasattr(self.mesh_view, "label"):
                draft_text = "[DRAFT]" if is_draft else ""
                self.mesh_view.title_frame.draft_label.config(text=draft_text)
                self.mesh_view.title_frame.view_label.config(text="Segmentation Results (3D Mesh View)")
            self.mesh_view.canvas.draw()
            return
        
        # initialise combined meshes as an axial volume (A, C, S) for each object
        combined_meshes = {obj_id: np.zeros((A, C, S), dtype=int) for obj_id in all_object_ids}

        # process each view and aggregate the masks
        for axis in [0,1,2]:
            if axis == 0:
                # Axial view – masks should be in axial space: each slice is (C, S), keyed by axial index.
                mask_source = self.draft_axial_view_mask if is_draft else self.axial_view_mask
                if not mask_source:
                    continue
                for z, frame_data in mask_source.items():
                    # z is axial slice index in range [0, A-1]
                    for obj_id, mask in frame_data.items():
                        m = np.squeeze(mask) # get expected shape (C, S)
                        if m.shape != (C, S):
                            continue
                        combined_meshes[obj_id][z, :, :] = np.maximum(combined_meshes[obj_id][z, :, :], m)  # combine masks
            elif axis == 1:
                # Coronal view – masks are of shape (A, S) keyed by a coronal index j (0 <= j < C).
                mask_source = self.draft_coronal_view_mask if is_draft else self.coronal_view_mask
                if not mask_source:
                    continue
                for j, frame_data in mask_source.items():
                    # j is row coordinate in axial view
                    for obj_id, mask in frame_data.items():
                        m = np.squeeze(mask) # get expected shape (A, S)
                        if m.shape != (A, S):
                            continue
                        for a in range(A):
                            combined_meshes[obj_id][a, j, :] = np.maximum(combined_meshes[obj_id][a, j, :], m[a, :])
            elif axis == 2:
                # Sagittal view – masks are of shape (A, C) keyed by a sagittal index k (0 <= k < S).
                mask_source = self.draft_sagittal_view_mask if is_draft else self.sagittal_view_mask
                if not mask_source:
                    continue
                for k, frame_data in mask_source.items():
                    # k is column coordinate in axial view
                    for obj_id, mask in frame_data.items():
                        m = np.squeeze(mask) # get expected shape (A, C)
                        if m.shape != (A, C):
                            continue
                        for a in range(A):
                            # remap pixel (a, c) in sagittal view to (a, c, k)
                            combined_meshes[obj_id][a, :, k] = np.maximum(combined_meshes[obj_id][a, :, k], m[a, :])

        # store current mesh data for export
        self.current_mesh_data = []

        # Create a 3D plot
        fig = self.mesh_view.canvas.figure
        fig.clf()  # Clear current figure
        ax = fig.add_subplot(111, projection='3d')

        downsample_factor = 0.5 # Downsample factor for faster rendering

        for obj_id, combined_mesh in combined_meshes.items():
            downsampled = ndimage.zoom(combined_mesh, zoom=downsample_factor, order=0)
            verts, faces, _, _ = measure.marching_cubes(downsampled, level=0.5) # faster, optimised version of measure.marching_cubes
            verts = verts / downsample_factor  # Rescale to original dimensions
            
            try:
                import trimesh
                mesh = trimesh.base.Trimesh(vertices=verts, faces=faces)
                # set the target face count
                target_faces = max(100, len(faces) // 2)
                simplified_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                verts = simplified_mesh.vertices
                faces = simplified_mesh.faces
            except Exception as e:
                print(f"Mesh optimisation failed, continuing without optimising: {e}")

            base_color = self.pointer_color_mapping.get(obj_id, "red")
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=base_color, alpha=0.7)

            self.current_mesh_data.append((verts, faces, obj_id))
        
        # Customize the plot (optional)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        self.mesh_view.canvas.draw()

        if is_draft:
            # store the mesh for each object in a dictionary
            self.last_draft_result = {obj_id: (verts, faces) 
                                    for verts, faces, obj_id in self.current_mesh_data}
        else:
            self.last_result = {obj_id: (verts, faces) 
                                for verts, faces, obj_id in self.current_mesh_data}

        # update the mesh view label
        if hasattr(self.mesh_view, "label"):
            draft_text = "[DRAFT]" if is_draft else ""
            self.mesh_view.title_frame.draft_label.config(text=draft_text)
            self.mesh_view.title_frame.view_label.config(text="Segmentation Results (3D Mesh View)")
    
    def _export_3d_mesh(self) -> None:
        """
        Exports the 3D mesh currently displayed in the mesh view as an STL file.
        Uses the latest segmentation
        to recompute the mesh using an optimised marching cubes approach.
        """
        try:
            from stl import mesh
        except ImportError:
            tk.messagebox.showerror("Export Error", "The 'numpy-stl' package is required for exporting 3D meshes.")
            return
    
        # ensure segmentation has been run, else, 3d segmentation mesh will not exist
        if self.sidebar.global_draft_mode.get():
            if not self.last_draft_result:
                tk.messagebox.showerror("Export Error", "No segmentation result available for export.")
                return
            else:
                video_segments = self.last_draft_result
        else:
            if not self.last_result:
                tk.messagebox.showerror("Export Error", "No segmentation result available for export.")
                return
            else:
                video_segments = self.last_result
        
        stl_meshes = []
        for obj_id, (verts, faces) in video_segments.items():
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

    def _export_view_with_mask(self) -> None:
        """
        Exports the original image with overlayed segmentation mask in the active view
        as a 3D NIfTI file.
        Presents a popup using radio buttons (with an Enum) for user export format selection.
        """
        popup = tk.Toplevel(self)
        popup.title("Choose Export Color Format")
        popup.transient(self)

        instruction_label = tk.Label(popup, text="Select export format for the exported view:")
        instruction_label.pack(pady=10)

        export_var = tk.StringVar(value=ExportFormat.BINARY.value)

        # Create a radio button for each enum option.
        for fmt in ExportFormat:
            rb = ttk.Radiobutton(popup, text=str(fmt), variable=export_var, value=fmt.value)
            rb.pack(anchor="w", padx=20)

        def on_confirm():
            # Convert the selected value to an enum instance.
            chosen_format = ExportFormat(export_var.get())
            popup.destroy()
            self._export_view_with_mask_process(chosen_format)
            
        confirm_button = ttk.Button(popup, text="OK", command=on_confirm, bootstyle="info")
        confirm_button.pack(pady=10)
        
    def _export_view_with_mask_process(self, chosen_format: ExportFormat) -> None:
        """
        Carries out the export process using the chosen format:
          - BINARY: Convert composite volume to binary image.
          - GRAYSCALE: Convert composite volume to a weighted grayscale image.
          - RGB: Overlay each segmentation mask on the original image in RGB format.
        """
        alpha = 0.4

        original_np = np.asarray(self.model.image)

        active_index = self.sidebar.tabControl.index("current")
        axis = active_index
        current_tab = self.sidebar.tabs[active_index]

        num_slices = original_np.shape[axis]

        if self.sidebar.global_draft_mode.get():
            mask_dict = self._get_mask(axis, is_draft=True)
        else:
            mask_dict = self._get_mask(axis)

        if mask_dict is None:
            tk.messagebox.showerror("Export Error", "No segmentation mask available for selected view.")
            return

        composite_slices = []

        grayscale_mapping = {1: 63, 2: 252, 3: 189, 4: 126, 5: 111, 6: 96, 7: 71, 8: 56, 9: 41, 10: 26}

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

            # normalise original slice to 0-255
            slice_min, slice_max = orig_slice.min(), orig_slice.max()
            if slice_max > slice_min:  # to avoid divide-by-zero
                norm_slice = (orig_slice - slice_min) / (slice_max - slice_min)
            else:
                norm_slice = orig_slice * 0  # all zeros if it's a uniform slice
            norm_slice_255 = (norm_slice * 255).astype(np.uint8)

            # convert grayscale to RGB
            rgb = np.stack([norm_slice_255] * 3, axis=-1).astype(np.float32)
            
            # if binary or grayscale; start with a black canvas
            if chosen_format == ExportFormat.BINARY or chosen_format == ExportFormat.GRAYSCALE:
                composite = np.zeros_like(rgb)
            else: # start with original rgb image
                composite = rgb.copy()

            # if a segmentation mask exists for this slice, overlay each object. 
            if i in mask_dict:
                for obj_id, mask in sorted(mask_dict[i].items(), key=lambda item: item[0]):
                    if active_index == 2:
                        mask = np.squeeze(mask.astype(np.float32))
                        mask = np.rot90(mask, k=-1)
                        mask = np.fliplr(mask)
                        mask_expanded = mask[..., np.newaxis] # expand *exactly one* axis for alpha blending
                    else:
                        mask_expanded = np.expand_dims(mask.astype(np.float32), axis=-1)

                    if chosen_format == ExportFormat.BINARY:
                        binary_mask = (mask_expanded > 0).squeeze()
                        composite[binary_mask] = 255  # Set the mask area to white
                    elif chosen_format == ExportFormat.GRAYSCALE:
                        gray_val = grayscale_mapping.get(obj_id, 0)
                        overlay = np.full_like(composite, gray_val)
                        # only update pixels where mask is active without affecting other overlays
                        composite = np.where(mask_expanded > 0, (1-alpha) * composite + alpha * overlay, composite)
                    else:
                        colour_name = self.pointer_color_mapping.get(obj_id, "red")
                        colour_rgb = np.array(mcolors.to_rgb(colour_name)) * 255
                        overlay = np.full_like(composite, colour_rgb)
                        # blend overlay with original image using alpha
                        composite = np.where(mask_expanded > 0, (1 - alpha) * composite + alpha * overlay, composite)
                    
                if chosen_format != ExportFormat.BINARY:
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
    
    def _get_canvas(self, axis: int) -> FigureCanvasTkAgg:
        """
        Returns the canvas corresponding to the specified axis.

        Parameters:
            axis (int): The axis number (0 for axial, 1 for coronal, 2 for sagittal).

        Returns:
            FigureCanvasTkAgg: The canvas associated with the specified axis.
        """
        if axis == 0:
            return self.axial_view.canvas
        elif axis == 1:
            return self.coronal_view.canvas
        elif axis == 2:
            return self.sagittal_view.canvas
        else:
            raise ValueError("Invalid axis number. Must be 0, 1, or 2.")
        
    def _get_mask(self, axis: int, is_draft: bool = False) -> SegmentationResult:
        """
        Returns the canvas corresponding to the specified axis.

        Parameters:
            axis (int): The axis number (0 for axial, 1 for coronal, 2 for sagittal).
        
        Returns:
            dict: The segmentation mask associated with the specified axis.
        """
        if axis == 0:
            return self.draft_axial_view_mask if is_draft else self.axial_view_mask
        elif axis == 1:
            return self.draft_coronal_view_mask if is_draft else self.coronal_view_mask
        elif axis == 2:
            return self.draft_sagittal_view_mask if is_draft else self.sagittal_view_mask
        else:
            raise ValueError("Invalid axis number. Must be 0, 1, or 2.")

    def clear_mesh_view(self) -> None:
        """
        Clears the 3D mesh view (i.e. removes any previously segmented mesh).
        """
        self.last_result = None
        self.last_draft_result = None
        
        fig = self.mesh_view.canvas.figure
        fig.clf()  # Clear all content from the figure
        # Re-create an empty 3D axes with a title.
        ax = fig.add_subplot(111, projection='3d')
        draft_text = "[DRAFT]" if self.sidebar.global_draft_mode.get() else ""
        self.mesh_view.title_frame.draft_label.configure(text=draft_text)
        self.mesh_view.title_frame.view_label.configure(text="Segmentation Result (3D Mesh View)")
        self.mesh_view.canvas.draw()
    
    def update_global_segmentation_state(self) -> None:
        """
        Update the state of the global segmentation checkbox.

        This method enables or disables the global segmentation checkbox
        based on the current mode and the presence of points. When in draft mode,
        the checkbox is enabled if at least one tab has draft points. Otherwise, it
        is enabled if at least one tab has non-draft points.

        Returns:
            None.
        """
        if self.sidebar.global_draft_mode.get():
            enable = any(tab.draft_points for tab in self.sidebar.tabs)
        else:
            enable = any(tab.points for tab in self.sidebar.tabs)
        state = "normal" if enable else "disabled"
        for tab in self.sidebar.tabs:
            tab.global_segmentation_checkbox.config(state=state)

    def update_all_views(self) -> None:
        """
        Loop through all image views (axial, coronal, sagittal)
        and update their displayed slices.
        """
        self.update_view_labels()
        
        for axis in [0, 1, 2]:
            canvas = self._get_canvas(axis)
            slice_idx = int(canvas.slider.get())  # use the current slider value in each view
            self._update_slice(canvas.figure.axes[0], canvas, axis, slice_idx)
        

    def update_view_labels(self) -> None:
        """
        Update the labels for the Axial, Coronal, and Sagittal frames.
        If global draft mode is enabled, prepend "[DRAFT]" to each label.
        """
        draft_text = "[DRAFT]" if self.sidebar.global_draft_mode.get() else ""

        self.axial_view.title_frame.draft_label.config(text=draft_text)
        self.axial_view.title_frame.view_label.config(text="Axial View")
        self.coronal_view.title_frame.draft_label.config(text=draft_text)
        self.coronal_view.title_frame.view_label.config(text="Coronal View")
        self.sagittal_view.title_frame.draft_label.config(text=draft_text)
        self.sagittal_view.title_frame.view_label.config(text="Sagittal View")
        self.mesh_view.title_frame.draft_label.config(text=draft_text)
        self.mesh_view.title_frame.view_label.config(text="Segmentation Result (3D Mesh View)")

    def update_tabs(self) -> None:
        """
        Selects the correct points listbox (Draft or Final Points) for each view.
        """
        for tab in self.sidebar.tabs:
            if self.sidebar.global_draft_mode.get():
                tab.final_points_frame.pack_forget()
                tab.draft_points_frame.pack(fill="both", expand=True, pady=(0,10))
            else:
                tab.draft_points_frame.pack_forget()
                tab.final_points_frame.pack(fill="both", expand=True, pady=(0,10))

    def _on_close(self) -> None:
        """
        Called when the user closes the window via the title bar or otherwise.
        """
        self.quit()
        self.destroy()
