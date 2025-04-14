"""
Module for computing performance metrics (DICE, ASSD) from segmentation results.
"""
import numpy as np
from scipy import ndimage
from sklearn.neighbors import KDTree
import pydicom
import glob
import cv2
import SimpleITK as sitk
from typing import List

def DICE(Vref: np.ndarray, Vseg: np.ndarray) -> float:
    """
    Compute the Dice Similarity Coefficient between two binary volumes.

    The Dice coefficient is calculated as:
      DICE = 2 * (size(Vref n Vseg)) / (size(Vref) + size(Vseg))

    Args:
        Vref (np.ndarray): The ground truth binary volume.
        Vseg (np.ndarray): The segmented binary volume.

    Returns:
        float: Dice coefficient.
    """
    dice=2*(Vref & Vseg).sum()/(Vref.sum() + Vseg.sum())
    return dice

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Dice coefficient between two segmentation masks.
    
    Args:
        y_true (np.ndarray): The ground truth mask.
        y_pred (np.ndarray): The predicted mask.
    Returns:
        The Dice coefficient between the two masks.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    denominator = y_true.sum() + y_pred.sum()
    if denominator == 0:
        return 1.0  # both masks are empty
    return 2.0 * intersection / float(denominator)

def transformToRealCoordinates(indexPoints: np.ndarray, dicom_dir: str) -> List[np.ndarray]:
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.
    
    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """
    
    dicom_file_list=glob.glob(dicom_dir + '/*.dcm')
    dicom_file_list.sort()
    #Read position and orientation info from first image
    ds_first = pydicom.dcmread(dicom_file_list[0])
    img_pos_first=list( map(float, list(ds_first.ImagePositionPatient)))
    img_or=list( map(float, list(ds_first.ImageOrientationPatient)))
    pix_space=list( map(float, list(ds_first.PixelSpacing)))
    #Read position info from first image from last image
    ds_last = pydicom.dcmread(dicom_file_list[-1])
    img_pos_last=list( map(float, list(ds_last.ImagePositionPatient)))

    T1=img_pos_first
    TN=img_pos_last
    X=img_or[:3]
    Y=img_or[3:]
    deltaI=pix_space[0]
    deltaJ=pix_space[1]
    N=len(dicom_file_list)
    M=np.array([[X[0]*deltaI,Y[0]*deltaJ,(T1[0]-TN[0])/(1-N),T1[0]], [X[1]*deltaI,Y[1]*deltaJ,(T1[1]-TN[1])/(1-N),T1[1]], [X[2]*deltaI,Y[2]*deltaJ,(T1[2]-TN[2])/(1-N),T1[2]], [0,0,0,1]])

    realPoints=[]
    for i in range(len(indexPoints[0])):
        P=np.array([indexPoints[1,i],indexPoints[2,i],indexPoints[0,i],1])
        R=np.matmul(M,P)
        realPoints.append(R[0:3])

    return realPoints

def new_assd(Vref: np.ndarray, Vseg: np.ndarray, dicom_dir: str) -> float:
    """
    Compute the Average Symmetric Surface Distance (ASSD) between two volumes.

    This function computes the border (surface) of each volume and converts the indices to real-world 
    coordinates. It then calculates the average distance from the segmented border to the reference border 
    and vice versa.

    Args:
        Vref (np.ndarray): The ground truth binary volume.
        Vseg (np.ndarray): The segmented binary volume.
        dicom_dir (str): The directory containing DICOM images used for coordinate transformation.

    Returns:
        float: The calculated ASSD.
    """
    struct = ndimage.generate_binary_structure(3, 1)  
    
    ref_border=Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels=np.array(np.where(ref_border))
        
    seg_border=Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels=np.array(np.where(seg_border))  
    
    ref_border_voxels_real=transformToRealCoordinates(ref_border_voxels,dicom_dir)
    seg_border_voxels_real=transformToRealCoordinates(seg_border_voxels,dicom_dir)    
  
    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)   
    
    assd=(dist_seg_to_ref.sum() + dist_ref_to_seg.sum())/(len(dist_seg_to_ref)+len(dist_ref_to_seg))
    return assd