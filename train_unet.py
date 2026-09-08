import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import albumentations as A

from unet_128 import UNet

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def dice_score(outputs, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (denominator + eps)

    return dice.mean().item()


# Rotate applies to BOTH image and mask together (keeps them aligned).
# RandomBrightnessContrast and RandomGamma are pixel-level transforms, so
# Albumentations only applies them to the image, never the mask.
# Added affine that does zoom and translations of both image and mask
TRAIN_TRANSFORM = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
])


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


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name, params, learning_rate):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from adam, adamw, sgd, rmsprop.")


def train_model(learning_rate=5e-4, pos_weight_value=3.0, batch_size=8,
                 num_epochs=100, checkpoint_path="models/best_unet.pth",
                 save_history=True, verbose=True, use_augmentation=True,
                 optimizer_name="adamw", train_frac=0.8, val_frac=0.1):
    """Trains a UNet with the given hyperparameters. Returns the best
    (val_loss, val_dice) achieved, and saves the best checkpoint.

    train_frac / val_frac are fractions of the total dataset (default 80%/10%,
    with the remaining 10% used as the test set)."""
    image_dir = "data/input_image"
    mask_dir = "data/expert_label"

    device = get_device()
    if verbose:
        print("Using device:", device)

    # Two dataset instances sharing the same file list/order: one with
    # augmentation (used for the training split), one without (val/test).
    aug = TRAIN_TRANSFORM if use_augmentation else None
    train_dataset = SegmentationDataset(image_dir, mask_dir, transform=aug)
    plain_dataset = SegmentationDataset(image_dir, mask_dir, transform=None)

    n = len(plain_dataset)
    train_split = int(n * train_frac)
    val_split = int(n * val_frac)
    # test_split = whatever's left, so rounding doesn't drop any samples

    if verbose:
        print(f"Dataset size: {n} -> train={train_split}, val={val_split}, "
              f"test={n - train_split - val_split}")

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:train_split]
    val_idx = indices[train_split:train_split + val_split]
    test_idx = indices[train_split + val_split:]

    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(plain_dataset, val_idx)
    test_set = Subset(plain_dataset, test_idx)  # never touched until final evaluation

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = UNet(n_class=1).to(device)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss = DiceLoss()

    def criterion(outputs, masks):
        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)
        return 0.5 * bce + 0.5 * dice

    optimizer = build_optimizer(
        optimizer_name, 
        model.parameters(), 
        learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_val_dice = -1.0
    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        model.eval()
        dice_total = 0.0
        num_images = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                batch_dice = dice_score(outputs, masks)

                dice_total += batch_dice * images.size(0)
                num_images += images.size(0)

        val_dice = dice_total / num_images

        scheduler.step(val_loss)

        history.append((train_loss, val_loss, val_dice))

        current_lr = optimizer.param_groups[0]["lr"]

        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val dice: {val_dice:.4f} | "
                f"lr: {current_lr:.2e}"
            )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss

            torch.save(model.state_dict(), checkpoint_path)

            if verbose:
                print(f"  -> saved new best model (val dice {val_dice:.4f})")

    if save_history:
        with open("output/history.csv", "w") as f:
            f.write("train_loss,val_loss,val_dice\n")
            for train_loss, val_loss, val_dice in history:
                f.write(f"{train_loss},{val_loss},{val_dice}\n")
        if verbose:
            print("Saved training history to output/history.csv")

    # Final, one-time evaluation on the held-out test set using the best
    # checkpoint (selected via validation, never touched by test data).
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_loss = run_epoch(model, test_loader, criterion, device)
    dice_total = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_total += dice_score(outputs, masks) * images.size(0)
    test_dice = dice_total / len(test_loader.dataset)

    if verbose:
        print(f"Test loss: {test_loss:.4f} | Test dice: {test_dice:.4f}")

    return best_val_loss, best_val_dice, test_loss, test_dice


def main():
    best_val_loss, best_val_dice, test_loss, test_dice = train_model()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}, best val dice: {best_val_dice:.4f}")
    print(f"Final test loss: {test_loss:.4f}, final test dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()