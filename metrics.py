import numpy as np

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