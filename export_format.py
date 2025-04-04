from enum import Enum

class ExportFormat(Enum):
    BINARY = "binary"
    GRAYSCALE = "grayscale"
    RGB = "rgb"

    def __str__(self):
        # Allows nicer labels in radio buttons if needed
        if self == ExportFormat.BINARY:
            return "Binary (suitable for single object segmentation only)"
        elif self == ExportFormat.GRAYSCALE:
            return "Grayscale"
        elif self == ExportFormat.RGB:
            return "RGB"
        return self.value