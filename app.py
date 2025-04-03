import argparse
import tkinter as tk

from model import ImageModel
from controller import SegmentationController
from view import MainView


def main():
    """
    Main entry point of the application.

    Creates the Model, View, and Controller components,
    and starts the Tkinter main loop.
    """
    # Create the Model
    model = ImageModel(None)

    # Initialize the view (GUI). 
    # We pass 'None' temporarily for the controller and will set it after instantiating the controller.
    view = MainView(model, None)

    # Create the Controller
    controller = SegmentationController(model, view)

    # we now assign controller to the view properly
    view.controller = controller

    # trigger view and mesh frames (it relies on view._slice_request_callback)
    view._build_image_frames()

    # Start the Tkinter main loop
    view.mainloop()

if __name__ == "__main__":
    main()
