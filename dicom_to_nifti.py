import SimpleITK as sitk

class DicomToNifti:
    def __init__(self, dicom_path, nifti_path):
        self.dicom_path = dicom_path
        self.nifti_path = nifti_path

    def convert(dicom_folder_path, nifti_file_path=None):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        if nifti_file_path is None:
            nifti_file_path = dicom_folder_path + ".nii"

        sitk.WriteImage(image, nifti_file_path) # TODO: decide whether to compress this file to .nii.gz

        return nifti_file_path