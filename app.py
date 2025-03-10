import argparse
import tkinter as tk

from model import ImageModel
from controller import SegmentationController
from view import MainView

#import modal_handler

#import modal

# app = modal.App("example-hello-world")

# @app.local_entrypoint()
def main():
    #modal_handler.set_app(app)

    # Create the Model
    model = ImageModel(None)

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
