"""Loss functions for crack segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined loss: 0.5 * BCE-with-logits + 0.5 * soft Dice."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice_loss(pred, target)
        return 0.5 * loss_bce + 0.5 * loss_dice


class JointEdgeSegLoss(nn.Module):
    """Joint loss for the boundary-guided model, Eq. (7).

    Supervises the segmentation output with BCE+Dice and the predicted boundary
    map with a dynamically balanced weighted cross-entropy.
    """

    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.seg_loss = BCEDiceLoss()
        self.edge_weight = edge_weight

    def extract_edge_from_mask(self, mask):
        """Derive a high-frequency edge target from the ground-truth mask.

        Morphological dilation minus erosion, implemented with 3x3 convolutions.
        """
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)

        # Dilation.
        dilate = F.conv2d(mask, kernel, padding=1)
        dilate = torch.clamp(dilate, 0, 1)

        # Erosion.
        erode = 1 - F.conv2d(1 - mask, kernel, padding=1)
        erode = torch.clamp(erode, 0, 1)

        # Edge = dilation - erosion.
        edge = dilate - erode
        return edge

    def boundary_loss(self, edge_pred, edge_target):
        """Dynamically balanced weighted cross-entropy, Eq. (5) and Eq. (6).

        Eq. (6): beta is the proportion of non-boundary pixels in the slice,
        recomputed for every image. Eq. (5) weights the boundary term by beta
        and the background term by (1 - beta).
        """
        beta = 1.0 - edge_target.mean()

        logit_bce = F.binary_cross_entropy_with_logits(
            edge_pred, edge_target, reduction='none')
        weight = beta * edge_target + (1.0 - beta) * (1.0 - edge_target)
        return (weight * logit_bce).mean()

    def forward(self, preds, target):
        """
        preds:  tuple (segmentation logits, boundary logits)
        target: ground-truth binary fracture mask
        """
        seg_pred, edge_pred = preds

        # 1. Main segmentation loss, L_s = 0.5 * L_BCE + 0.5 * L_Dice, Eq. (8).
        loss_seg = self.seg_loss(seg_pred, target)

        # 2. Boundary label from the morphological gradient, Eq. (4), then the
        #    balanced boundary loss, Eq. (5).
        edge_target = self.extract_edge_from_mask(target)
        loss_edge = self.boundary_loss(edge_pred, edge_target)

        # 3. L_t = L_s + lambda * L_b, Eq. (7), with lambda = 0.5.
        total_loss = loss_seg + self.edge_weight * loss_edge
        return total_loss
