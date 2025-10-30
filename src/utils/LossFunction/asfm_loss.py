import torch
import torch.nn as nn

def ASFM_loss(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)

    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    FP = torch.sum((1 - y_true) * y_pred)
    TN = torch.sum((1 - y_true) * (1 - y_pred))
    smooth = 1e-5
    # sensitivity = TP / (TP + FN + smooth)
    # specificity = TN / (TN + FP + smooth)
    numer = TP * TP + smooth
    deno = TP*TP + TP*FN + TP*FP + FN*FP + smooth
    map = numer/deno
    # score = 2 * sensitivity * specificity / (sensitivity + specificity + smooth)
    loss = 2*(1-map)/(1+torch.exp(-0.25*map))
    return loss