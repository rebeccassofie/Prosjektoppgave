import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(VALID_EXT))
        self.transform = transform  # an Albumentations Compose, or None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, filename)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, filename)).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float()
        mask = (mask < 127).float().unsqueeze(0)  # 1 = boundary line, 0 = background

        return image, mask
