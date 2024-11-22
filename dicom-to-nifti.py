import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames('CHAOS_TEST_CT_40_DICOM_ANON')
reader.SetFileNames(dicom_names)
image = reader.Execute()

sitk.WriteImage(image, 'CHAOS_TEST_CT_40_DICOM_ANON.nii') # should i compress this