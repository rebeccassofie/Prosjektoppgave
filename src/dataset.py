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


def build_train_transform(brightness_contrast_p=0.3, gamma_p=0.3,
                           hflip_p=0.5, vflip_p=0.5):  #undersøk hva alle gjør, få ut på bilder
    """
    Builds the training-time augmentation pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.RandomRotate90(p=0.5), #90, 180, 270, hvis den roterer en annen måte vil den fylle inn med piksler
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

    modality controls which crops are indexed:
      "iq" / "adp" -- one channel, only that modality
      "both"       -- one channel; iq and adp crops are separate samples
      "stacked"    -- two channels (channel 0 = iq, channel 1 = adp) from the
                      SAME location in one sample, so the model sees both at once
    """

    def __init__(self, image_dir, mask_dir, transform=None, modality="both"):
        """
        modality selects which crops to use: "both" (default, iq and adp),
        "iq", or "adp".
        """
        if modality not in ("both", "iq", "adp", "stacked"):
            raise ValueError(
                f"modality must be 'both', 'iq', 'adp', or 'stacked', got {modality!r}"
            )
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform  # an Albumentations Compose, or None
        self.modality = modality
        self.in_channels = 2 if modality == "stacked" else 1
        self.samples = self._build_index()

    def _build_index(self):
        if self.modality == "stacked":
            return self._build_stacked_index()

        samples = []
        skipped = []

        for filename in sorted(os.listdir(self.image_dir)):
            if not filename.lower().endswith(VALID_EXT):
                continue
            if "_full." in filename.lower():
                continue  # full-scan renders aren't crops, skip them

            match = MODALITY_INFIX_RE.search(filename)
            if self.modality != "both" and (
                match is None or match.group(1) != self.modality
            ):
                continue

            mask_filename = MODALITY_INFIX_RE.sub("", filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if not os.path.exists(mask_path):
                skipped.append(filename)
                continue

            samples.append({
                "image_paths": [os.path.join(self.image_dir, filename)],
                "mask_path": mask_path,
                "name": filename,
            })

        if skipped:
            preview = skipped[:5]
            print(f"Warning: {len(skipped)} file(s) in {self.image_dir} had no "
                  f"matching mask and were skipped: {preview}"
                  + (" ..." if len(skipped) > 5 else ""))

        return samples

    def _build_stacked_index(self):
        """One sample per crop location that has BOTH an iq and an adp crop
        (plus a mask). Channel order is always [iq, adp]."""
        samples = []
        skipped = []

        for filename in sorted(os.listdir(self.image_dir)):
            if not filename.lower().endswith(VALID_EXT):
                continue
            if "_full." in filename.lower():
                continue
            match = MODALITY_INFIX_RE.search(filename)
            if match is None or match.group(1) != "adp":
                continue  # index each location once, via its adp crop

            adp_path = os.path.join(self.image_dir, filename)
            iq_filename = MODALITY_INFIX_RE.sub("_iq", filename)
            iq_path = os.path.join(self.image_dir, iq_filename)
            mask_filename = MODALITY_INFIX_RE.sub("", filename)
            mask_path = os.path.join(self.mask_dir, mask_filename)

            if not (os.path.exists(iq_path) and os.path.exists(mask_path)):
                skipped.append(filename)
                continue

            samples.append({
                "image_paths": [iq_path, adp_path],
                "mask_path": mask_path,
                "name": MODALITY_INFIX_RE.sub("_stacked", filename),
            })

        if skipped:
            print(f"Warning: {len(skipped)} adp crop(s) had no matching iq crop "
                  f"or mask and were skipped: {skipped[:5]}"
                  + (" ..." if len(skipped) > 5 else ""))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # each crop is single-channel grayscale; "stacked" puts iq and adp
        # side by side as channels -> (H, W, C)
        channels = [np.array(Image.open(p).convert("L")) for p in sample["image_paths"]]
        image = np.stack(channels, axis=-1)
        mask = np.array(Image.open(sample["mask_path"]).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # (H, W, C) -> (C, H, W); C = 1 for iq/adp/both, 2 for stacked
        image = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float()
        mask = (mask < 127).float().unsqueeze(0)  # 1 = boundary line, 0 = background

        return image, mask