# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Calculates the Dice Loss, a common metric for image segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply softmax to logits to get probabilities
        probs = F.softmax(logits, dim=1)
        # We are interested in the runway class (class 1)
        probs = probs[:, 1, :, :]
        targets = targets.float()

        intersection = (probs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    """
    A combined loss function that includes Dice Loss for segmentation,
    Cross-Entropy for segmentation, and MSE for line coordinate regression.
    """
    def __init__(self, seg_weight=0.85, line_weight=0.15):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.line_weight = line_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, seg_preds, line_preds, seg_targets, line_targets):
        # Segmentation Loss (Dice + Cross Entropy)
        dice = self.dice_loss(seg_preds, seg_targets)
        ce = self.ce_loss(seg_preds, seg_targets)
        segmentation_loss = dice + ce

        # Line Prediction Loss (MSE)
        line_loss = self.mse_loss(line_preds, line_targets)

        # Total Weighted Loss
        total_loss = (self.seg_weight * segmentation_loss) + (self.line_weight * line_loss)

        return total_loss
