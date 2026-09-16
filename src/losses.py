import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Flatten each image separately
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2 * intersection + self.smooth
        ) / (
            probs.sum(dim=1) +
            targets.sum(dim=1) +
            self.smooth
        )

        return 1 - dice.mean()
