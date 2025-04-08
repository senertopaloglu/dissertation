from typing import Optional
import SimpleITK as sitk
import argparse

class DicomToNifti:
    def convert(self, dicom_folder_path: str, nifti_file_path: Optional[str]) -> str:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        if nifti_file_path is None:
            nifti_file_path = dicom_folder_path + ".nii"

        sitk.WriteImage(image, nifti_file_path) # TODO: decide whether to compress this file to .nii.gz

        return nifti_file_path

def main():
    parser = argparse.ArgumentParser(description="Convert a DICOM folder to a NIfTI file.")
    parser.add_argument("dicom_folder_path", type=str, help="Path to the DICOM folder.")
    parser.add_argument("--nifti_file_path", type=str, help="Path to save the NIfTI file.", default=None)
    args = parser.parse_args()

    converter = DicomToNifti()
    try:
        nifti_file_path = converter.convert(args.dicom_folder_path, args.nifti_file_path)
        print(f"NIfTI file saved to: {nifti_file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to convert DICOM to NIfTI: {e}")

if __name__ == "__main__":
    main()

#converter = DicomToNifti()
#converter.convert("c:\Users\sbtop\Downloads\CHAOS_Test_Sets\Test_Sets\CT\35\DICOM_anon")