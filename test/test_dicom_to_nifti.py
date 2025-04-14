import os
import unittest
import tempfile
from dicom_to_nifti import DicomToNifti

class TestDicomToNifti(unittest.TestCase):
    # Testing was performed on CHAOS_TEST_CT_37 however the DICOM folder was too big for submission
    # def test_convert_dicom_to_nifti(self):
    #     # Construct the path to the DICOM folder within the tests/resources directory
    #     current_dir = os.path.dirname(__file__)
    #     dicom_folder = os.path.join(current_dir, "resources", "CHAOS_TEST_CT_37_DICOM_ANON")
        
    #     # Create a temporary file for the output NIfTI file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
    #         output_file = tmp_file.name
    #     # Remove the temporary file so that SimpleITK can write to the path without conflicts
    #     if os.path.exists(output_file):
    #         os.remove(output_file)
        
    #     converter = DicomToNifti()
    #     nifti_file = converter.convert(dicom_folder, output_file)
        
    #     # Verify the file was created
    #     self.assertTrue(os.path.exists(nifti_file))
        
    #     # Clean up by removing the generated NIfTI file
    #     os.remove(nifti_file)

    # Test for the case when a non-existent DICOM folder is provided
    def test_convert_non_existent_dicom_folder(self):
        converter = DicomToNifti()
        non_existent_folder = "non_existent_folder"
        
        # Check if the function raises an exception for a non-existent folder
        with self.assertRaises(Exception) as context:
            converter.convert(non_existent_folder, None)

if __name__ == "__main__":
    unittest.main()