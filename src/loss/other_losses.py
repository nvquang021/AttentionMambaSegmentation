import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.trainer.metrics import dice_score

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dice = dice_score(y_pred, y_true, smooth=1e-3)

        return 1 - dice
epsilon = 1e-3
smooth = 1e-3
def tversky(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred)

    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)
def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1 - pt_1), gamma)
def bce_tversky_loss(pred,target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(torch.sigmoid(pred), target)
    tv = tversky_loss(target, pred)

    loss = bce * bce_weight + tv * (1 - bce_weight)

    return loss
def dice_tversky_loss(pred,target, bce_weight = 0.5):
    dice = DiceLoss()(pred, target)
    tv = tversky_loss(target, pred)

    loss = dice * bce_weight + tv * (1 - bce_weight)

    return loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss