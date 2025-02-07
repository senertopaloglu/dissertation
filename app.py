import argparse
import tkinter as tk

from model import ImageModel
from controller import SegmentationController
from view import MainView

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Medical Image Segmentation")
    parser.add_argument("filename", help="Path to the 3D medical image file")
    args = parser.parse_args()

    # Create the Model
    model = ImageModel(args.filename)

    # Initialize the view (GUI). 
    # We pass 'None' temporarily for the controller and will set it after instantiating the controller.
    view = MainView(model, None)

    # Create the Controller
    controller = SegmentationController(model, view)

    # Now that the controller is created, we can assign it to the view properly
    # (though we already pass it in the constructor, you could pass references either way).
    view.controller = controller

    # build frames now view._slice_request_callback is set
    view._build_image_frames()

    # Start the Tkinter main loop
    view.mainloop()

if __name__ == "__main__":
    main()
