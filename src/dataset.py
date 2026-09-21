import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


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
    Reads cropped IQ/ADP images from `image_dir` (datasets/cropped/SDA) and their
    matching boundary masks from `mask_dir` (datasets/cropped/PGs), as produced by
    scripts/generate_crops.py.

    Filenames follow:
        mask:  <sample>_<location>_<mag>_<idx>.png
        image: <sample>_<location>_<mag>_iq_<idx>.png
               <sample>_<location>_<mag>_adp_<idx>.png
    """

    # captures everything up to "_iq_" or "_adp_" as the shared key, then
    # the crop index -- e.g. "1_midt_x100_iq_3.png" -> key="1_midt_x100", idx="3"
    IMAGE_PATTERN = re.compile(
        r"^(?P<key>.+)_(?P<modality>iq|adp)_(?P<idx>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp)$",
        re.IGNORECASE,
    )

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

            match = self.IMAGE_PATTERN.match(filename)
            if not match:
                skipped.append(filename)
                continue

            key, idx = match.group("key"), match.group("idx")
            mask_filename = f"{key}_{idx}.png"
            mask_path = os.path.join(self.mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                skipped.append(filename)
                continue

            samples.append({
                "image_path": os.path.join(self.image_dir, filename),
                "mask_path": mask_path,
                "modality": match.group("modality").lower(),
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

        # iq/adp crops are single-channel grayscale, not RGB
        image = np.array(Image.open(sample["image_path"]).convert("L"))
        mask = np.array(Image.open(sample["mask_path"]).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # (1, H, W)
        mask = torch.from_numpy(mask).float()
        mask = (mask < 127).float().unsqueeze(0)  # 1 = boundary line, 0 = background

        return image, mask