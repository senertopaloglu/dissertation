import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itk
import tkinter as tk
from tkinter import ttk


# gracefully close the window
def close_window(root):
    root.quit()  # stop the tkinter main loop
    root.destroy()  # kill the py interpreter instance running in the background

def load_image(filename):
    PixelType = itk.ctype("short")
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def get_slice(image, axis, slice_index):
    size = list(image.GetLargestPossibleRegion().GetSize())
    start = list(image.GetLargestPossibleRegion().GetIndex())

    if axis == 0:
        size[2]=1
        start[2]=slice_index
    elif axis == 1:
        size[1]=1
        start[1]=slice_index
    elif axis == 2:
        size[0] = 1
        start[0] = slice_index
    else:
        raise ValueError("Invalid view")

    RegionType = itk.ImageRegion[3]
    desiredRegion = RegionType()
    desiredRegion.SetIndex(start)
    desiredRegion.SetSize(size)

    extractor = itk.ExtractImageFilter.New(image)
    extractor.SetExtractionRegion(desiredRegion)
    extractor.SetDirectionCollapseToIdentity()
    extractor.Update()
    slice_image = extractor.GetOutput()

    # convert the slice to a numpy array and remove the third dimension i.e. (2d_imgwidth, 2d_imgheight, 1) is converted to (2d_imgwidth, 2d_imgheight)
    slice_array = itk.GetArrayViewFromImage(slice_image)

    return slice_array.squeeze() # remove singleton dimension

def undo(undo_stack, redo_stack, points, listbox, line_objects, canvas):
    print("undo clicked")
    if points:
        last_point = points.pop()
        undo_stack.append(last_point)
        redo_stack.append(last_point)
        listbox.delete("end")
        if line_objects:
            last_line = line_objects.pop()
            last_line.remove()
        if canvas[0]: # should exist. extra precaution.
            canvas[0].draw()


def redo(redo_stack, undo_stack, points, listbox, ax, line_objects, canvas):
    print("redo clicked")
    if redo_stack:
        restored_point = redo_stack.pop()
        points.append(restored_point)
        undo_stack.append(restored_point)

        restored_point, color = restored_point[:-1], restored_point[-1]
        listbox.insert("end", f"{restored_point}")
        
        if ax[0] and canvas[0]:
            line=ax[0].plot(restored_point[0], restored_point[1], f'{color}o')[0]
            line_objects.append(line)
            canvas[0].draw()

def on_click(event, points, points_listbox, ax, line_objects, canvas, most_recent_canvas, most_recent_ax, pointer_color):
    if event.inaxes: # ensure the click is inside the axes
        x, y = int(event.xdata), int(event.ydata) # get the clicked coordinates
        
        print(f"Point selected: ({x}, {y})") # TODO: remove. print for debugging
        points_listbox.insert("end", f"({x},{y})")
        points_listbox.yview_moveto(1.0)
        
        selected_color = pointer_color.get().lower() # convert to lowercase for consistency
        color_map = {"red": "r", "green": "g", "blue": "b"}  # map color names to Matplotlib codes
        plot_color = color_map.get(selected_color, "r")

        line = ax.plot(x, y, f'{plot_color}o')[0]
        points.append((x, y, plot_color)) # add the point to the list
        line_objects.append(line)
        most_recent_canvas[0] = canvas
        most_recent_ax[0] = ax
        canvas.draw() # update the canvas

def main(image_filename):
    image = load_image(image_filename)
    sizes = image.GetLargestPossibleRegion().GetSize()


    root = tk.Tk() # root window
    root.state('zoomed')
    root.title("Interactive 3D Medical Image Segmentation")
    root.protocol("WM_DELETE_WINDOW", lambda: close_window(root))  # 

    # configure layout
    root.columnconfigure(0, weight=1, minsize=250) # left sidebar
    root.columnconfigure(1, weight=3) # image grid column 1
    root.columnconfigure(2, weight=3) # image grid column 2

    root.rowconfigure(0, weight=3) # image grid row 1
    root.rowconfigure(1, weight=3) # image grid row 2


    most_recent_canvas = [None]
    most_recent_ax = [None]

    # left sidebar
    sidebar = tk.Frame(root, bg="lightgray", padx=10, pady=10)
    sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

    # left sidebar: buttons
    btn_import = ttk.Button(sidebar, text="Import Image")
    btn_import.pack(fill="x", pady=5)

    btn_segment = ttk.Button(sidebar, text="Segment Image")
    btn_segment.pack(fill="x", pady=5)

    btn_export = ttk.Button(sidebar, text="Export Image")
    btn_export.pack(fill="x", pady=5)

    # left sidebar: colour dropdown
    pointer_label = tk.Label(sidebar, text="Pointer Colour")
    pointer_label.pack(pady=(10, 2))

    pointer_color = ttk.Combobox(sidebar, values=["Red", "Blue", "Green"])
    pointer_color.pack(fill="x")
    pointer_color.set("Red")

    # left sidebar: selected points listbox
    points_label = tk.Label(sidebar, text="Selected Points")
    points_label.pack(pady=(10, 2))

    points = []
    line_objects = []

    points_frame = tk.Frame(sidebar)
    points_frame.pack(fill="x")
    points_listbox = tk.Listbox(points_frame, height=5, yscrollcommand=lambda f, l: points_listbox_scrollbar.set(f,l))
    points_listbox.pack(side="left", fill="x", expand=True)
    points_listbox_scrollbar = tk.Scrollbar(points_frame, orient="vertical", command=points_listbox.yview) # in case we have lots of points in the future
    points_listbox_scrollbar.pack(side="right", fill="y")
    

    # left sidebar: undo/redo buttons
    undo_stack = []
    redo_stack = []

    btn_undo = ttk.Button(sidebar, text="Undo", command=lambda:undo(undo_stack, redo_stack, points, points_listbox, line_objects, most_recent_canvas))
    btn_undo.pack(fill="x", pady=2)
    
    btn_redo = ttk.Button(sidebar, text="Redo", command=lambda:redo(redo_stack, undo_stack, points, points_listbox, most_recent_ax, line_objects, most_recent_canvas))
    btn_redo.pack(fill="x", pady=2)
    


    # create image views
    def create_image_frame(parent, text, axis):
        frame = tk.Frame(parent, bd=1, relief="solid")

        label = tk.Label(frame, text=text)
        label.pack()

        def update(val):
            slice_index = int(float(slider.get()))
            slice_array = get_slice(image, axis, slice_index)
            ax.clear()
            ax.imshow(slice_array, cmap='gray')
            ax.set_title(f"{text} - Slice {slice_index}")
            canvas.draw()
        
        fig, ax = plt.subplots(figsize=(4,4))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side="bottom", fill="both")
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, points_listbox, ax, line_objects, canvas, most_recent_canvas, most_recent_ax, pointer_color))
        canvas.draw()

        dim = 2 if axis == 0 else 1 if axis == 1 else 0
        slider = ttk.Scale(frame, from_=0, to=sizes[dim]-1, orient="horizontal", command=lambda val: update(val))
        slider.pack(side="top", fill="x",expand=True)
        slider.set(sizes[dim]//2)  # set initial slider pos to middle slice

        update(sizes[dim]//2) # TODO: uncomment. inits view at middle slice

        return frame

    axial_view = create_image_frame(root, "Axial View", 0)
    coronal_view = create_image_frame(root, "Coronal View", 1)
    sagittal_view = create_image_frame(root, "Sagittal View", 2)
    mesh_view = create_image_frame(root, "Segmentation Result (Axial)", 0) # TODO: make 3d mesh view 


    # image grid: attach image views
    axial_view.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    coronal_view.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
    sagittal_view.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
    mesh_view.grid(row=1, column=2, sticky="nsew", padx=5, pady=5) # TODO: is a copy of axial view at the moment




    root.mainloop()
