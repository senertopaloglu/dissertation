import itk

class ImageModel:
    """
    The Model in our MVC. Responsible for loading the ITK image and providing
    slices for visualization.
    """
    def __init__(self, filename):
        self.filename = filename
        self.image = None
        self._load_image()

    def _load_image(self):
        """
        Load a 3D image from a file using ITK.
        """
        PixelType = itk.ctype("short")
        Dimension = 3
        ImageType = itk.Image[PixelType, Dimension]

        reader = itk.ImageFileReader[ImageType].New()
        reader.SetFileName(self.filename)
        reader.Update()
        self.image = reader.GetOutput()
    
    def change_image(self, filename):
        self.filename = filename
        self._load_image()

    def get_slice(self, axis, slice_index):
        """
        Extract a 2D slice from the 3D image.

        Args:
            axis (int): The axis along which to slice (0=axial,1=coronal,2=sagittal).
            slice_index (int): Which slice to extract.
        Returns:
            A 2D NumPy array of the slice.
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
