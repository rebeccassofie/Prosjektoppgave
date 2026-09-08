import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from unet_128 import UNet


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

IMAGE_DIR = "TBM/input_image"
MASK_DIR = "TBM/expert_label"

CHECKPOINT_PATH = "best_unet.pth"

BATCH_SIZE = 8

# Same split fractions as train_unet.py
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

# Same random seed as train_unet.py
SPLIT_SEED = 42

# Thresholds we want to test
THRESHOLDS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Same filename ordering as train_unet.py
        self.filenames = sorted(
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(VALID_EXT)
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = np.array(
            Image.open(
                os.path.join(self.image_dir, filename)
            ).convert("RGB")
        )

        mask = np.array(
            Image.open(
                os.path.join(self.mask_dir, filename)
            ).convert("L")
        )

        # Same image preprocessing as train_unet.py
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Same mask definition as train_unet.py:
        # dark pixels = boundary line = 1
        mask = torch.from_numpy(mask).float()
        mask = (mask < 127).float().unsqueeze(0)

        return image, mask


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------
# Create the same validation split as train_unet.py
# ---------------------------------------------------------

def create_validation_loader():
    dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR)

    n = len(dataset)

    train_split = int(n * TRAIN_FRAC)
    val_split = int(n * VAL_FRAC)

    # IMPORTANT:
    # This is exactly the same splitting procedure
    # used in train_unet.py.
    indices = torch.randperm(
        n,
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    ).tolist()

    train_idx = indices[:train_split]

    val_idx = indices[
        train_split:train_split + val_split
    ]

    # We only need validation images for this experiment.
    val_set = Subset(dataset, val_idx)

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Dataset size: {n}")
    print(f"Validation size: {len(val_set)}")

    return val_loader


# ---------------------------------------------------------
# Evaluate one threshold
# ---------------------------------------------------------

def evaluate_threshold(model, loader, device, threshold):
    model.eval()

    dice_total = 0.0
    num_images = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            # Model outputs logits
            outputs = model(images)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions
            preds = (probs > threshold).float()

            # Calculate Dice separately for every image
            intersection = (preds * masks).sum(dim=(1, 2, 3))

            denominator = (
                preds.sum(dim=(1, 2, 3))
                + masks.sum(dim=(1, 2, 3))
            )

            dice = (
                2 * intersection + 1e-6
            ) / (
                denominator + 1e-6
            )

            # Add the Dice scores from this batch
            dice_total += dice.sum().item()

            num_images += images.size(0)

    return dice_total / num_images


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # Select device
    device = get_device()

    print("Using device:", device)

    # -----------------------------------------------------
    # Check that checkpoint exists
    # -----------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Could not find checkpoint: {CHECKPOINT_PATH}\n"
            "Make sure best_unet.pth is in the same folder "
            "as this script."
        )

    # -----------------------------------------------------
    # Create validation loader
    # -----------------------------------------------------

    val_loader = create_validation_loader()

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = UNet(n_class=1).to(device)

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )
    )

    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)

    # -----------------------------------------------------
    # Test thresholds
    # -----------------------------------------------------

    results = {}

    print("\nValidation Dice by threshold:")
    print("--------------------------------")

    for threshold in THRESHOLDS:

        dice = evaluate_threshold(
            model,
            val_loader,
            device,
            threshold
        )

        results[threshold] = dice

        print(
            f"threshold={threshold:.1f} -> "
            f"Dice={dice:.4f}"
        )

    # -----------------------------------------------------
    # Find best threshold
    # -----------------------------------------------------

    best_threshold = max(
        results,
        key=results.get
    )

    best_dice = results[best_threshold]

    print("\n--------------------------------")
    print(
        f"Best threshold: {best_threshold:.1f}"
    )
    print(
        f"Best validation Dice: {best_dice:.4f}"
    )

    # -----------------------------------------------------
    # Compare against the standard 0.5 threshold
    # -----------------------------------------------------

    dice_at_05 = results[0.5]

    print(
        f"\nDice at threshold 0.5: {dice_at_05:.4f}"
    )

    improvement = best_dice - dice_at_05

    print(
        f"Improvement over 0.5: {improvement:+.4f}"
    )


if __name__ == "__main__":
    main()
