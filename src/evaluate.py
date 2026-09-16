import torch
import torch.nn as nn


def dice_score(outputs, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denominator + eps)

    return dice.mean().item()


def evaluate_model(model, loader, criterion, device):
    """Runs one full pass over loader, returns (avg_loss, avg_dice)."""
    model.eval()
    loss_total, dice_total, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, masks).item() * images.size(0)
            dice_total += dice_score(outputs, masks) * images.size(0)
            n += images.size(0)
    return loss_total / n, dice_total / n
