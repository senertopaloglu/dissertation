"""
Module for building and configuring GUI frames.

This module provides helper functions and custom toolbar classes to create various
frames used in the application, including 2D image views and a 3D mesh view.
"""
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Frame, Scale, Checkbutton
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt

class CustomNavigationToolbar2Tk(NavigationToolbar2Tk):
    # Remove the 'Save' tool from the toolbar
    toolitems = [item for item in NavigationToolbar2Tk.toolitems if item[0] != 'Save']

def create_title_frame(parent, main_view, text: str) -> ttk.Frame:
    """
    Create a composite title widget, including a [DRAFT] label if the global draft mode is active.
    
    Args:
        parent (tk.Widget): The parent widget for the title frame.
        main_view (MainView): The main view instance to check the global draft mode.
        text (str): The title text to display.

    Returns:
        ttk.Frame: The configured title frame.
    """
    title_frame = ttk.Frame(parent)
    draft_text = "[DRAFT] " if main_view.sidebar.global_draft_mode.get() else ""
    draft_label = ttk.Label(title_frame, text=draft_text, font=("TkDefaultFont", 13, "bold"))
    draft_label.grid(row=0, column=0, sticky="e")
    view_label = ttk.Label(title_frame, text=text, font=("TkDefaultFont", 13))
    view_label.grid(row=0, column=1, sticky="w")
    title_frame.columnconfigure(0, weight=1)
    title_frame.columnconfigure(1, weight=1)
    # Store references for later update
    title_frame.draft_label = draft_label
    title_frame.view_label = view_label
    return title_frame

def create_image_frame(main_view, text: str, axis: int) -> Frame:
    """
    Creates a Tkinter Frame for one of the 2D image views (axial, coronal, sagittal),
    complete with a Matplotlib FigureCanvas, toolbar, slider and checkboxes.
    
    Args:
        main_view (MainView): The main view instance that holds the model and callbacks.
        text (str): The title text for the image frame.
        axis (int): The image axis (0 for axial, 1 for coronal, 2 for sagittal).

    Returns:
        Frame: The fully configured image frame.
    """
    outer_frame = Frame(main_view, relief="solid", borderwidth=1)
    content_frame = Frame(outer_frame)
    content_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    title_frame = create_title_frame(content_frame, main_view, text)
    title_frame.pack(side="top", fill="x", pady=(5, 0))
    outer_frame.title_frame = title_frame

    control_frame = Frame(content_frame)
    control_frame.pack(side="top", fill="x", pady=5)

    fig = Figure(figsize=(4, 4))
    ax_obj = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=content_frame)
    # Save axis index as a property of the canvas (used later in events)
    canvas.tab_index = axis
    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

    toolbar = CustomNavigationToolbar2Tk(canvas, control_frame)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.toolbar = toolbar

    outer_frame.canvas = canvas
    canvas.axis = axis
    outer_frame.canvas_ax = ax_obj

    # Connect the click event to the MainView handler
    fig.canvas.mpl_connect(
        'button_press_event',
        lambda event: main_view._on_click(event, ax_obj, canvas)
    )

    # If an image is loaded, set up slider, checkboxes, and initial slice rendering.
    if main_view.model.image:
        sizes = main_view.model.image.GetLargestPossibleRegion().GetSize()
        dim = 2 if axis == 0 else 1 if axis == 1 else 0
        max_slice = sizes[dim] - 1

        show_mask_var = ttk.BooleanVar()
        canvas.show_mask_var = show_mask_var

        show_points_var = ttk.BooleanVar(value=True)
        canvas.show_points_var = show_points_var

        slider = Scale(
            control_frame,
            from_=0,
            to=max_slice,
            orient="horizontal",
            command=lambda val: main_view._update_slice(ax_obj, canvas, axis, val)
        )
        slider.pack(side="top", fill="x", expand=True)

        initial_slice = max_slice // 2
        slider.set(initial_slice)

        show_mask_checkbox = Checkbutton(
            control_frame,
            text="Show Segmentation Masks",
            variable=show_mask_var,
            command=lambda: main_view._update_slice(ax_obj, canvas, axis, int(slider.get()))
        )
        show_mask_checkbox.pack(side="top", anchor="center", pady=5)
        if main_view._get_mask(axis) is None:
            show_mask_checkbox.config(state="disabled")
        canvas.show_mask_checkbox = show_mask_checkbox

        show_points_checkbox = Checkbutton(
            control_frame,
            text="Show Points",
            variable=show_points_var,
            command=lambda: main_view._update_slice(ax_obj, canvas, axis, int(slider.get()))
        )
        show_points_checkbox.pack(side="top", anchor="center", pady=(3, 2))
        show_points_checkbox.config(state="disabled")
        canvas.show_points_checkbox = show_points_checkbox

        # Render the initial slice.
        main_view._update_slice(ax_obj, canvas, axis, initial_slice)
        canvas.slider = slider

    return outer_frame

def create_mesh_view_frame(main_view, text: str) -> Frame:
    """
    Creates and returns a Tkinter Frame for the 3D mesh view.

    Args:
        main_view (MainView): The main view instance.
        text (str): The title text for the 3D mesh view frame.

    Returns:
        Frame: The configured 3D mesh view frame.
    """
    outer_frame = Frame(main_view, relief="solid", borderwidth=1)
    content_frame = Frame(outer_frame)
    content_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    title_frame = create_title_frame(content_frame, main_view, text)
    title_frame.pack(side="top", fill="x", pady=(5, 0))
    outer_frame.title_frame = title_frame

    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=content_frame)
    canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
    
    outer_frame.canvas = canvas
    outer_frame.canvas_ax = ax
    return outer_frame
