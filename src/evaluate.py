import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def dice_score(outputs, masks, threshold=0.5, eps=1e-6):
    """Plain pixel Dice. Kept for reference: it punishes thick lines and
    1-pixel shifts, so it underrates boundary predictions."""
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denominator + eps)

    return dice.mean().item()


def _boundary_prf_single(pred, gt, tolerance):
    """
    Boundary precision/recall/F1 for one image.

    pred, gt: 2D bool arrays (True = boundary).

    1. Both are skeletonized to 1-pixel-wide lines, so line thickness
       does not affect the score.
    2. Precision = share of predicted skeleton pixels that lie within
       `tolerance` pixels of a ground-truth line (penalises false boundaries).
    3. Recall = share of ground-truth skeleton pixels that lie within
       `tolerance` pixels of a predicted line (penalises missed boundaries/gaps).
    """
    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)

    n_pred, n_gt = pred_skel.sum(), gt_skel.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0  # nothing to find, nothing predicted
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0

    # distance from every pixel to the nearest line pixel
    dist_to_gt = distance_transform_edt(~gt_skel)
    dist_to_pred = distance_transform_edt(~pred_skel)

    precision = (dist_to_gt[pred_skel] <= tolerance).mean()
    recall = (dist_to_pred[gt_skel] <= tolerance).mean()

    if precision + recall == 0:
        return float(precision), float(recall), 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def boundary_scores(outputs, masks, threshold=0.5, tolerance=3):
    """
    Batch-averaged boundary (precision, recall, F1), ignoring line thickness
    and allowing a `tolerance`-pixel offset.

    outputs: raw logits (B, 1, H, W); masks: (B, 1, H, W) with 1 = boundary.
    """
    preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().astype(bool)
    gts = (masks > 0.5).cpu().numpy().astype(bool)

    scores = np.array([
        _boundary_prf_single(p[0], g[0], tolerance)
        for p, g in zip(preds, gts)
    ])
    precision, recall, f1 = scores.mean(axis=0)
    return float(precision), float(recall), float(f1)


def boundary_f1_score(outputs, masks, threshold=0.5, tolerance=3):
    """Drop-in replacement for dice_score: same inputs, returns a float."""
    return boundary_scores(outputs, masks, threshold, tolerance)[2]


def evaluate_model(model, loader, criterion, device, metric="boundary_f1",
                   tolerance=3, threshold=0.5):
    """
    Runs one full pass over loader, returns (avg_loss, avg_score).

    metric: "boundary_f1" (default, thickness-independent, with tolerance)
            or "dice" (old pixel Dice).
    """
    if metric not in ("boundary_f1", "dice"):
        raise ValueError(f"metric must be 'boundary_f1' or 'dice', got {metric!r}")

    model.eval()
    loss_total, score_total, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, masks).item() * images.size(0)
            if metric == "dice":
                score = dice_score(outputs, masks, threshold)
            else:
                score = boundary_f1_score(outputs, masks, threshold, tolerance)
            score_total += score * images.size(0)
            n += images.size(0)
    return loss_total / n, score_total / n