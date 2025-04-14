"""
Module for running the Image Segmentation Application.

This module serves as the main entry point for the application. It creates the Model,
View, and Controller components, configures the mode of operation (local or remote),
and starts the tkinter main loop.

Example:
    To run locally:
        python app.py --mode local
    To run remotely:
        python app.py --mode remote
"""
import argparse
import tkinter as tk

from model import ImageModel
from controller import SegmentationController
from views.view import MainView

# Example Usages:
# Run locally via `python app.py --mode local` or `python app.py`
# Run remotely via `python app.py --mode remote`

def main():
    """
    Main entry point of the application.

    Creates the Model, View, and Controller components,
    and starts the Tkinter main loop.
    """
    parser = argparse.ArgumentParser(description="Image Segmentation Application")
    parser.add_argument('--mode', choices=['local', 'remote'], default='local', help='Mode of operation: local or remote')
    args = parser.parse_args()
    
    model = ImageModel(None)

    # We pass 'None' temporarily for the controller and will set it after instantiating the controller.
    view = MainView(model, None)

    controller = SegmentationController(model, view)
    controller.is_remote = (args.mode == 'remote')

    # we now assign controller to the view properly
    view.controller = controller

    # trigger view and mesh frames (it relies on view._slice_request_callback)
    view._build_image_frames()

    # Start the Tkinter main loop
    view.mainloop()

if __name__ == "__main__":
    main()
