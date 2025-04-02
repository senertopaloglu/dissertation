import numpy as np
from scipy import ndimage
from sklearn.neighbors import KDTree
import pydicom
import glob
import cv2
import SimpleITK as sitk
from scipy import ndimage
from sklearn.neighbors import KDTree

def DICE(Vref,Vseg):
    dice=2*(Vref & Vseg).sum()/(Vref.sum() + Vseg.sum())
    return dice

def dice_coefficient(y_true, y_pred):
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

def _get_spacing():
    """
    Compute the spacing (pixel/voxel size) along each dimension. Default is 1.0.
    
    Returns:
        A tuple representing the spacing along each dimension.
    """

def transformToRealCoordinates(indexPoints,dicom_dir):
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

def new_assd(Vref, Vseg, dicom_dir):
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

def assd(y_true, y_pred):
    """
    Compute the Average Symmetric Surface Distance (ASSD) between two segmentation masks.
    
    The function extracts the border pixels/voxels of each mask and computes
    the average of the minimum distances from each border point in one mask to 
    the other mask. A spacing value (or tuple) can be provided to account for physical
    spacing.
    
    Args:
        y_true (np.ndarray): The ground truth mask.
        y_pred (np.ndarray): The predicted mask.
        spacing (float or tuple): The spacing (pixel/voxel size) along each dimension.
                                  Default is 1.0.
    
    Returns:
        The ASSD value between the two masks or np.nan if one of the borders is empty.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    dims = y_true.ndim
    struct = ndimage.generate_binary_structure(dims, 1)

    # Extract border elements using XOR between mask and its erosion
    true_border = y_true ^ ndimage.binary_erosion(y_true, structure=struct)
    pred_border = y_pred ^ ndimage.binary_erosion(y_pred, structure=struct)
    
    # get coordinates of the border points
    true_coords = np.array(np.where(true_border)).T
    pred_coords = np.array(np.where(pred_border)).T

    if true_coords.shape[0] == 0 or pred_coords.shape[0] == 0:
        # one of the masks is empty or has no border points
        return np.nan
    
    spacing = _get_spacing()
    
    true_coords = true_coords * spacing
    pred_coords = pred_coords * spacing

    true_coords = true_coords.astype(float)
    pred_coords = pred_coords.astype(float)

    # build kd-tree for each set of border points
    tree_true = KDTree(true_coords)
    distances_pred_to_true = tree_true.query(pred_coords)

    tree_pred = KDTree(pred_coords)
    distances_true_to_pred = tree_pred.query(true_coords)

    # compute the average symmetric distance
    assd_value = (distances_pred_to_true.sum() + distances_true_to_pred.sum()) / (len(distances_pred_to_true) + len(distances_true_to_pred))
    return assd_value