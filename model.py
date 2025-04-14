"""
Module for managing image data. Defines ImageModel class for loading and processing 3D images using ITK.
"""
import itk
import numpy as np

class ImageModel:
    """
    The Model in our MVC. Responsible for loading the ITK image and providing
    slices for visualisation.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.image = None
        if filename:
            self._load_image()

    def _load_image(self) -> None:
        """
        Load a 3D image from a file using ITK.

        Initialises an ITK image reader for a 3D image.
        It reads the image data from the provided filename and stores the output image for further processing.

        Returns:
            None
        """
        PixelType = itk.ctype("short")
        Dimension = 3
        ImageType = itk.Image[PixelType, Dimension]

        reader = itk.ImageFileReader[ImageType].New()
        reader.SetFileName(self.filename)
        reader.Update()
        self.image = reader.GetOutput()
    
    def change_image(self, filename: str) -> None:
        """
        Change the currently loaded image.

        This method updates the filename state and reloads the image using the internal ITK reader.

        Args:
            filename (str): The new image file path.

        Returns:
            None
        """
        self.filename = filename
        self._load_image()

    def get_slice(self, axis: int, slice_index: int) -> np.ndarray:
        """
        Extract a 2D slice from the 3D image.

        Args:
            axis (int): The axis along which to slice (0=axial,1=coronal,2=sagittal).
            slice_index (int): Which slice to extract.

        Returns:
            np.ndarray: A 2D NumPy array of the slice.
        """
        size = list(self.image.GetLargestPossibleRegion().GetSize())
        start = list(self.image.GetLargestPossibleRegion().GetIndex())

        if axis == 0:  # Axial
            size[2] = 1
            start[2] = slice_index
        elif axis == 1:  # Coronal
            size[1] = 1
            start[1] = slice_index
        elif axis == 2:  # Sagittal
            size[0] = 1
            start[0] = slice_index
        else:
            raise ValueError("Invalid view axis. Must be 0,1, or 2.")

        RegionType = itk.ImageRegion[3]
        desiredRegion = RegionType()
        desiredRegion.SetIndex(start)
        desiredRegion.SetSize(size)

        extractor = itk.ExtractImageFilter.New(self.image)
        extractor.SetExtractionRegion(desiredRegion)
        extractor.SetDirectionCollapseToIdentity()
        extractor.Update()
        slice_image = extractor.GetOutput()

        # Convert to NumPy and remove singleton dimension
        slice_array = itk.GetArrayViewFromImage(slice_image)
        return slice_array.squeeze()
