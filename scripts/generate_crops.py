"""
Walks every .mat mask under MASK_ROOT, pairs each with its iq/adp .npy
files under IMAGE_ROOT, and for each pair:

  1. Loads, aligns (transpose-fix), NaN-cleans, and denoises the mask
  2. Converts it to a binary boundary image (white background, black lines)
  3. Saves the FULL boundary mask as <sample>_<location>_<mag>_full.png
  4. Crops the boundary mask, iq, and adp into non-overlapping
     CROP_SIZE x CROP_SIZE tiles (leftover edge strips are dropped, not
     padded), using the SAME tile grid for all three so crop N always
     lines up across mask/iq/adp
  5. Saves each crop:
       <sample>_<location>_<mag>_<n>.png       (mask boundary crop)
       <sample>_<location>_<mag>_iq_<n>.png    (iq crop)
       <sample>_<location>_<mag>_adp_<n>.png   (adp crop)

Usage (run from the project root):
    python -m scripts.generate_crops
"""
from pathlib import Path
import numpy as np
from skimage.segmentation import find_boundaries

from src.mask_utils import (
    build_pairs, load_mat_array, resize_mask_to,
    handle_nan_labels, denoise_labels, mask_to_binary_boundary,
    crop_non_overlapping, save_float01_as_png, normalize_for_display,
    remove_small_regions
)

MASK_ROOT = "datasets/raw/PGs"
IMAGE_ROOT = "datasets/raw/SDA"
OUTPUT_DIR = "datasets/cropped"
CROP_SIZE = 128
MODE_FILTER_RADIUS = 3
MIN_REGION_SIZE = 50



def process_pair(pair, output_dir, crop_size, mode_filter_radius):
    key = f"{pair['sample']}_{pair['location']}_{pair['magnification']}"
    print(f"\n--- {key} ---")

    mask = load_mat_array(str(pair["mask_path"]))
    iq = np.load(pair["iq_path"])
    adp = np.load(pair["adp_path"])

    print(f"mask raw: shape={mask.shape} | iq: shape={iq.shape} | adp: shape={adp.shape}")

    mask = resize_mask_to(mask, iq.shape)
    mask = handle_nan_labels(mask)
    denoised = denoise_labels(mask, radius=mode_filter_radius)
    denoised = remove_small_regions(denoised, min_size=MIN_REGION_SIZE)
    denoised = denoised.astype(np.int64)

    boundary_full = mask_to_binary_boundary(denoised)

    output_dir_full = output_dir / "PGs_full"
    output_dir_PGs= output_dir / "PGs"
    output_dir_SDA= output_dir / "SDA"

    # Save the full mask
    full_path = output_dir_full / f"{key}_full.png"
    save_float01_as_png(boundary_full, full_path)
    print(f"saved full mask -> {full_path.name}  (shape={boundary_full.shape})")

    # Crop mask, iq, adp using the SAME grid so indices line up
    mask_tiles = crop_non_overlapping(boundary_full, crop_size)
    iq_tiles = crop_non_overlapping(iq, crop_size)
    adp_tiles = crop_non_overlapping(adp, crop_size)

    n_crops = len(mask_tiles)
    if n_crops == 0:
        print(f"  WARNING: {key} is smaller than {crop_size}x{crop_size} "
              f"({boundary_full.shape}) -- no crops produced.")
        return 0

    for (idx, mask_tile), (_, iq_tile), (_, adp_tile) in zip(mask_tiles, iq_tiles, adp_tiles):
        save_float01_as_png(mask_tile, output_dir_PGs / f"{key}_{idx}.png")
        save_float01_as_png(normalize_for_display(iq_tile), output_dir_SDA / f"{key}_iq_{idx}.png")
        save_float01_as_png(normalize_for_display(adp_tile), output_dir_SDA / f"{key}_adp_{idx}.png")

    print(f"saved {n_crops} crop(s) of size {crop_size}x{crop_size}")
    return n_crops


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(MASK_ROOT, IMAGE_ROOT)

    total_crops = 0
    for pair in pairs:
        total_crops += process_pair(pair, output_dir, CROP_SIZE, MODE_FILTER_RADIUS)

    print(f"\nDone. {len(pairs)} mask(s) processed, {total_crops} total crop(s) "
          f"saved to {output_dir}/")


if __name__ == "__main__":
    main()