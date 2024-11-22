import argparse
import itk
import numpy as np
import tkinter as tk
from tkinter import ttk, IntVar, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# load the input 3D image
def load_image(filename):
    PixelType = itk.ctype("short")
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# extract a 2D slice from the 3D image
def get_slice(image, slice_index):
    size = list(image.GetLargestPossibleRegion().GetSize())
    size[2] = 1  # set the size along the z-axis to 1

    start = list(image.GetLargestPossibleRegion().GetIndex())
    start[2] = slice_index  # set the desired slice index

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
    slice_array = itk.GetArrayViewFromImage(slice_image)[0]
    return slice_array

# init the GUI
root = tk.Tk()
root.title("3D Image Viewer")

# close function to end the program gracefully
def close_window():
    root.quit()  # stops the tkinter main loop
    root.destroy()  # kills the py interpreter instance running in the bg

root.protocol("WM_DELETE_WINDOW", close_window)  # bind the close event

axial_view_slice_string=StringVar()

# update the displayed slice based on slider value
def update_slice(val):
    slice_index = int(float(val)) # convert val (string containing floating point number) to float then int - cant do immediate conversion
    slice_array = get_slice(image, slice_index)
    ax.imshow(slice_array, cmap="gray")
    axial_view_slice_string.set(f"Slice: {slice_index}")
    canvas.draw()

# load the 3D (NIfTI) image
parser = argparse.ArgumentParser(description="Process A 2D Slice Of A 3D Image.")
parser.add_argument("input_3D_image_filename") # both .nii and .nii.gz work
args = parser.parse_args()

image_filename = args.input_3D_image_filename
image = load_image(image_filename)
max_slices = image.GetLargestPossibleRegion().GetSize()[2] - 1  # Z dimension size

# list to store users selected points
points = []
line_objects = []

# stacks to undo and redo points
undo_stack = []
redo_stack = []

# listbox for points
Lb1 = tk.Listbox(root)
Lb1.pack()

def on_click(event):
    if event.inaxes:  # Ensure the click is inside the axes
        x, y = int(event.xdata), int(event.ydata)  # Get the clicked coordinates
        points.append((x, y))  # Add the point to the list
        print(f"Point selected: ({x}, {y})")  # Print for debugging
        Lb1.insert("end", f"({x},{y})")
        #ax.plot(x, y, 'ro')  # Plot the point on the figure
        line = ax.plot(x, y, 'ro')[0]
        line_objects.append(line)
        canvas.draw()  # Update the canvas

# create 2x2 grid for views
grid_frame = ttk.Frame(root)
grid_frame.pack(fill="both", expand=True)
# configure grid to make all rows and cols equally resizable
for i in range(2):
    grid_frame.rowconfigure(i, weight=1)
    grid_frame.columnconfigure(i, weight=1)

# create the axial view ( a matplotlib figure)
frame = ttk.Frame(grid_frame)
frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
fig, ax = plt.subplots(figsize=(6, 6))
initial_slice = get_slice(image, 0)
img_plot = ax.imshow(initial_slice, cmap="gray")
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

fig.canvas.mpl_connect('button_press_event', on_click)

# create the axial view ( a matplotlib figure)
frame = ttk.Frame(grid_frame)
frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
fig, ax = plt.subplots(figsize=(6, 6))
initial_slice = get_slice(image, 0)
img_plot = ax.imshow(initial_slice, cmap="gray")
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

fig.canvas.mpl_connect('button_press_event', on_click)

# create the axial view ( a matplotlib figure)
frame = ttk.Frame(grid_frame)
frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
fig, ax = plt.subplots(figsize=(6, 6))
initial_slice = get_slice(image, 0)
img_plot = ax.imshow(initial_slice, cmap="gray")
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

fig.canvas.mpl_connect('button_press_event', on_click)

# create the axial view ( a matplotlib figure)
frame = ttk.Frame(grid_frame)
frame.grid(row=1, column=1, padx=2, pady=2, sticky="nsew")
fig, ax = plt.subplots(figsize=(4, 4))
initial_slice = get_slice(image, 0)
img_plot = ax.imshow(initial_slice, cmap="gray")
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

fig.canvas.mpl_connect('button_press_event', on_click)

def undo_action():
    if points:
        last_point = points.pop()
        undo_stack.append(last_point)
        redo_stack.clear()
        print(f"Undo: Removed point {last_point}")
        Lb1.delete("end")  # remove last point from listbox
        #ax.lines = ax.lines[:-1]  # remove the last plotted point

        if line_objects:
            last_line = line_objects.pop()
            last_line.remove()

        canvas.draw()  # update the canvas

def redo_action():
    if undo_stack:
        restored_point = undo_stack.pop()
        points.append(restored_point)
        redo_stack.append(restored_point)
        print(f"Redo: Restored point {restored_point}")
        Lb1.insert("end", f"({restored_point[0]},{restored_point[1]})")
        #ax.plot(restored_point[0], restored_point[1], 'ro')
        line = ax.plot(restored_point[0], restored_point[1], 'ro')[0]
        line_objects.append(line)
        canvas.draw()

# Undo button
undo_button = ttk.Button(root, text="Undo", command=undo_action)
undo_button.pack(fill="x", expand=True)

# Redo button
redo_button = ttk.Button(root, text="Redo", command=redo_action)
redo_button.pack(fill="x", expand=True)

# add axial view label
axial_view_label=ttk.Label(text="Axial View")
axial_view_label.pack(fill="x", expand=True)

axial_view_slice_label=ttk.Label(textvariable=axial_view_slice_string)
axial_view_slice_label.pack(fill="x", expand=True)

# add axial view to the window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# create and add slider to window
slider = ttk.Scale(root, from_=0, to=max_slices, orient="horizontal", command=update_slice)
slider.pack(fill="x", expand=True) # manage layout of slide in window
slider.set(max_slices//2)  # set initial slider pos to middle slice

# run the tkinter main loop
root.mainloop()
