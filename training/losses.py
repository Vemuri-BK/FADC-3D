import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Works on probabilities (after softmax), not raw logits.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred:   (B, 2, H, W, D) — raw logits, 2 classes
        # target: (B, 1, H, W, D) — binary mask, values 0 or 1

        # Convert logits to probabilities, take tumor channel (class 1)
        pred = F.softmax(pred, dim=1)[:, 1:, ...]   # (B, 1, H, W, D)

        pred_flat   = pred.reshape(-1)
        target_flat = target.reshape(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1.0 - dice


class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross Entropy loss.
    Dice: handles class imbalance (focuses on overlap quality)
    CE:   provides strong gradient signal everywhere
    Both together train faster and more stably than either alone.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice        = DiceLoss(smooth=smooth)
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # pred:   (B, 2, H, W, D) — raw logits
        # target: (B, 1, H, W, D) — binary mask

        # CE expects (B, H, W, D) long tensor as target
        target_ce = target.squeeze(1).long()   # (B, H, W, D)

        loss_dice = self.dice(pred, target)
        loss_ce   = self.ce(pred, target_ce)

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce, loss_dice, loss_ce


if __name__ == "__main__":
    criterion = DiceCELoss()

    # Simulate model output and ground truth
    pred   = torch.randn(2, 2, 128, 128, 64)   # batch=2, 2 classes, patch size
    target = torch.zeros(2, 1, 128, 128, 64)
    target[:, :, 60:70, 60:70, 30:35] = 1      # small tumor region

    total_loss, dice_loss, ce_loss = criterion(pred, target)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Dice loss:  {dice_loss.item():.4f}  (0=perfect, 1=worst)")
    print(f"CE loss:    {ce_loss.item():.4f}")
    print("Loss test passed.")
