"""
Inspects build_train_transform() by running it several times on one
image/mask pair and saving everything to experiments/inspect_albumentations/:
  - images/aug_XXX_image.png, aug_XXX_mask.png  (every augmented pair)
  - preview_grid.png                             (all of them side by side)
  - augmentations.csv                            (exactly what happened to each one)

Usage:
    python inspect_augmentation.py                     # picks the first mask/image pair found
    python inspect_augmentation.py --filename 1_1.png  # picks a specific mask crop by filename
    python inspect_augmentation.py --modality adp      # restrict to the adp crops

Each sample is one independent call to build_train_transform() applied to the
SAME original image/mask -- exactly what happens to one crop across different
epochs/DataLoader draws during training.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image


def build_train_transform(brightness_contrast_p=0.3, gamma_p=0.3,
                           hflip_p=0.5, vflip_p=0.5):  # undersøk hva alle gjør, få ut på bilder
    """
    Builds the training-time augmentation pipeline.
    """
    return A.ReplayCompose([
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.RandomRotate90(p=0.5),  # 90, 180, 270, hvis den roterer en annen måte vil den fylle inn med piksler
        A.RandomBrightnessContrast(p=brightness_contrast_p),
        A.RandomGamma(p=gamma_p),
    ])


def load_pair(image_path, mask_path):
    image = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L"))
    return image, mask


def strip_modality(stem):
    """Removes an 'iq'/'adp' marker from an image crop's filename stem so it
    can be compared against the mask's filename (masks carry no modality
    marker). Strips the token plus any leftover separator either side."""
    stem = re.sub(r"(^|_)(iq|adp)(_|$)", "_", stem, flags=re.IGNORECASE)
    return stem.strip("_")


def find_pair(image_dir, mask_dir, filename=None, modality=None):
    """Picks one mask crop and its matching iq/adp image crop out of a
    shared image_dir (filenames differ only by an iq/adp marker, e.g.
    '1_1_iq.png' / '1_1_adp.png' vs the mask's plain '1_1.png')."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)

    mask_paths = sorted(p for p in mask_dir.glob("*.png") if "_full" not in p.name)
    if filename is not None:
        mask_paths = [p for p in mask_paths if p.name == filename]
        if not mask_paths:
            raise FileNotFoundError(f"{filename} not found in {mask_dir}")

    image_paths = sorted(image_dir.glob("*.png"))
    if modality is not None:
        image_paths = [p for p in image_paths if modality.lower() in p.stem.lower()]

    by_stem = {}
    for image_path in image_paths:
        by_stem.setdefault(strip_modality(image_path.stem), []).append(image_path)

    for mask_path in mask_paths:
        candidates = by_stem.get(mask_path.stem)
        if candidates:
            return sorted(candidates)[0], mask_path

    raise FileNotFoundError(f"No matching image/mask pair found in {image_dir} and {mask_dir}")


def describe_replay(replay):
    """Turns a ReplayCompose 'replay' dict into a flat, human-readable
    record of exactly what happened to one sample."""
    by_name = {t["__class_fullname__"]: t for t in replay["transforms"]}
    row = {}

    hflip = by_name.get("HorizontalFlip", {})
    row["hflip_applied"] = hflip.get("applied", False)

    rot = by_name.get("RandomRotate90", {})
    row["rotate90_applied"] = rot.get("applied", False)
    factor = (rot.get("params") or {}).get("factor")
    row["rotate90_degrees"] = 90 * factor if factor is not None else 0

    bc = by_name.get("RandomBrightnessContrast", {})
    row["brightness_contrast_applied"] = bc.get("applied", False)
    bc_params = bc.get("params") or {}
    row["brightness_alpha"] = bc_params.get("alpha", "")
    row["brightness_beta"] = bc_params.get("beta", "")

    gamma = by_name.get("RandomGamma", {})
    row["gamma_applied"] = gamma.get("applied", False)
    row["gamma_value"] = (gamma.get("params") or {}).get("gamma", "")

    parts = []
    parts.append("hflip" if row["hflip_applied"] else "no hflip")
    parts.append(f"rotate {row['rotate90_degrees']}°" if row["rotate90_applied"] else "no rotate")
    parts.append("brightness/contrast" if row["brightness_contrast_applied"] else "no brightness/contrast")
    parts.append(f"gamma {row['gamma_value']:.2f}" if row["gamma_applied"] else "no gamma")
    row["summary"] = ", ".join(parts)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="datasets/cropped/SDA",
                         help="Folder with image crops (filenames contain 'iq' or 'adp')")
    parser.add_argument("--mask-dir", default="datasets/cropped/PGs",
                         help="Folder with the matching mask crops (no iq/adp marker)")
    parser.add_argument("--modality", default=None, choices=["iq", "adp"],
                         help="Restrict to one modality (default: either, whichever is found first)")
    parser.add_argument("--filename", default=None,
                         help="Specific mask crop filename to use, e.g. 1_1.png (default: first pair found)")
    parser.add_argument("--n", type=int, default=6, help="Number of augmented samples to show")
    parser.add_argument("--out-dir", default="experiments/inspect_augmentation",
                         help="Folder to save the grid, individual crops, and the CSV into")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path, mask_path = find_pair(args.image_dir, args.mask_dir, args.filename, args.modality)
    print(f"Using image: {image_path}")
    print(f"Using mask:  {mask_path}")
    image, mask = load_pair(image_path, mask_path)

    transform = build_train_transform()

    fig, axes = plt.subplots(args.n + 1, 2, figsize=(5, 2.5 * (args.n + 1)))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original image")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Original mask")

    csv_rows = []
    for row in range(1, args.n + 1):
        augmented = transform(image=image, mask=mask)
        aug_image, aug_mask = augmented["image"], augmented["mask"]

        axes[row, 0].imshow(aug_image, cmap="gray")
        axes[row, 0].set_title(f"Augmented image #{row}")
        axes[row, 1].imshow(aug_mask, cmap="gray")
        axes[row, 1].set_title(f"Augmented mask #{row}")

        image_out = images_dir / f"aug_{row:03d}_image.png"
        mask_out = images_dir / f"aug_{row:03d}_mask.png"
        Image.fromarray(aug_image).save(image_out)
        Image.fromarray(aug_mask).save(mask_out)

        record = describe_replay(augmented["replay"])
        record = {
            "sample": row,
            "source_image": image_path.name,
            "source_mask": mask_path.name,
            "image_file": image_out.name,
            "mask_file": mask_out.name,
            **record,
        }
        csv_rows.append(record)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    grid_path = out_dir / "preview_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)

    csv_path = out_dir / "augmentations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved {args.n} augmented pair(s) and {images_dir}, "
          f"{grid_path.name}, and {csv_path.name} to {out_dir}")


if __name__ == "__main__":
    main()