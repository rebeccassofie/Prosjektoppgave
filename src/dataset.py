import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# SDA crops are named <sample>_<location>_<mag>_<modality>_<idx>.ext (modality
# is "iq" or "adp"), while their matching mask in PGs has no modality infix:
# <sample>_<location>_<mag>_<idx>.ext. Strip it to find the mask filename.
MODALITY_INFIX_RE = re.compile(r"_(iq|adp)(?=_\d+\.[^.]+$)")


def build_train_transform(rotate_limit=30, scale=(0.9, 1.1),
                           translate_percent=(-0.05, 0.05),
                           brightness_contrast_p=0.3, gamma_p=0.3,
                           hflip_p=0.5, vflip_p=0.5):
    """
    Builds the training-time augmentation pipeline.

    Rotate, Affine, and the flips apply to BOTH image and mask together
    (keeps them aligned). RandomBrightnessContrast and RandomGamma are
    pixel-level transforms, so Albumentations only applies them to the
    image, never the mask.
    """
    return A.Compose([
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.Rotate(limit=rotate_limit, p=0.5),
        A.Affine(
            scale=scale,
            translate_percent=translate_percent,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.RandomGamma(p=gamma_p),
    ])


class SegmentationDataset(Dataset):
    """
    Reads crops from `image_dir` (cropped/SDA, containing both iq and adp
    crops) and their matching boundary masks from `mask_dir` (cropped/PGs),
    as produced by scripts/generate_crops.py.

    image_dir holds iq and adp crops together, named
    <sample>_<location>_<mag>_<modality>_<idx>.png, while mask_dir holds one
    mask per crop location, named <sample>_<location>_<mag>_<idx>.png (no
    modality infix, since the same mask applies to both modalities). The
    modality infix is stripped to look up each crop's mask.

    "_full" images (the uncropped full-scan renders) are excluded; this
    dataset only yields fixed-size crops.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform  # an Albumentations Compose, or None
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        skipped = []

        for filename in sorted(os.listdir(self.image_dir)):
            if not filename.lower().endswith(VALID_EXT):
                continue
            if "_full." in filename.lower():
                continue  # full-scan renders aren't crops, skip them

            mask_filename = MODALITY_INFIX_RE.sub("", filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if not os.path.exists(mask_path):
                skipped.append(filename)
                continue

            samples.append({
                "image_path": os.path.join(self.image_dir, filename),
                "mask_path": mask_path,
            })

        if skipped:
            preview = skipped[:5]
            print(f"Warning: {len(skipped)} file(s) in {self.image_dir} had no "
                  f"matching mask and were skipped: {preview}"
                  + (" ..." if len(skipped) > 5 else ""))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # crops are single-channel grayscale, not RGB
        image = np.array(Image.open(sample["image_path"]).convert("L"))
        mask = np.array(Image.open(sample["mask_path"]).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # (1, H, W)
        mask = torch.from_numpy(mask).float()
        mask = (mask < 127).float().unsqueeze(0)  # 1 = boundary line, 0 = background

        return image, mask