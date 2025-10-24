import torch
import numpy as np

def iou_score(y_pred, y_true):
    smooth = 1e-5
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum()
    union = (y_pred | y_true).sum()

    return (intersection + smooth) / (union + smooth)
def dice_score(y_pred, y_true, smooth=0.):

    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()

    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
def precision_score(y_pred, y_true):
    smooth = 1e-5
    y_pred = torch.sigmoid(y_pred)

    # Move tensors to CPU and convert to numpy arrays
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    # Binarize the output at threshold 0.5
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum(y_true * y_pred)
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum(y_true * (1 - y_pred))

    # Calculate Precision
    precision = TP / (TP + FP + smooth)

    return precision
def recall_score(y_pred, y_true):
    smooth = 1e-5
    y_pred = torch.sigmoid(y_pred)

    # Move tensors to CPU and convert to numpy arrays
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    # Binarize the output at threshold 0.5
    y_pred = y_pred > 0.5
    y_true = y_true > 0.5

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum(y_true * y_pred)
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum(y_true * (1 - y_pred))

    # Calculate Recall
    recall = TP / (TP + FN + smooth)

    return recall
def F_score(y_pred, y_true):
    smooth = 1e-5
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    F_score = 2 * (precision * recall) / (precision + recall + smooth)
    return F_score