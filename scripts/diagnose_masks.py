"""
Checks every ground-truth .mat mask against its IQ/ADP maps and against the
mask pipeline in generate_crops.py, to find out why generated masks differ
from the MTEX reconstruction.

For each scan it reports:
  1. What is inside the .mat file (variables, shape, number of grain IDs, NaN %)
  2. What resize_mask_to() does with it (keep / transpose / STRETCH)
  3. Orientation check: tries all 8 flips/rotations of the label map and
     measures how dark the IQ and ADP maps are on its grain boundaries.
     The correct orientation puts the boundaries where IQ/ADP drop, so it
     scores highest. If generate_crops.py uses a different orientation than
     the best one, the mask is misaligned with the image.
  4. How much the pipeline (NaN sentinel + mode filter + small-region removal)
     changes the boundaries compared with the raw parent IDs.

Output (per scan) in OUT_DIR:
  <key>_diagnosis.png   IQ with boundaries overlaid, raw vs pipeline difference
  summary.csv           one row per scan, problems flagged

Usage (from the project root):
    python -m scripts.diagnose_masks                 # all scans
    python -m scripts.diagnose_masks --key 010       # only scans whose name contains "010"
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from src.mask_utils import (
    build_pairs, load_mat_array, resize_mask_to, handle_nan_labels,
    denoise_labels, remove_small_regions, mask_to_binary_boundary,
)
from scripts.generate_crops import MASK_ROOT, IMAGE_ROOT, MODE_FILTER_RADIUS, MIN_REGION_SIZE

# all 8 ways a 2D map can be flipped/rotated (the "dihedral group")
ORIENTATIONS = {
    "as-is":            lambda a: a,
    "transpose":        lambda a: a.T,
    "flip up-down":     lambda a: np.flipud(a),
    "flip left-right":  lambda a: np.fliplr(a),
    "rotate 180":       lambda a: np.rot90(a, 2),
    "rotate 90":        lambda a: np.rot90(a, 1),
    "rotate 270":       lambda a: np.rot90(a, 3),
    "anti-transpose":   lambda a: np.rot90(a, 2).T,
}


def fill_nan_nearest(labels):
    """Gives every NaN pixel the grain ID of its nearest labelled pixel."""
    nan = np.isnan(labels)
    if not nan.any():
        return labels
    idx = ndi.distance_transform_edt(nan, return_distances=False, return_indices=True)
    return labels[tuple(idx)]


def zscore(img):
    img = np.asarray(img, dtype=float)
    good = np.isfinite(img)
    mu, sd = img[good].mean(), img[good].std() + 1e-9
    out = (img - mu) / sd
    out[~good] = 0.0
    return out


def boundary_contrast(labels, iq_z, adp_z):
    """How much lower IQ and ADP are ON the boundaries than elsewhere
    (in standard deviations). Higher = boundaries sit where the image drops."""
    b = find_boundaries(fill_nan_nearest(labels).astype(np.int64), mode="inner")
    if b.sum() == 0 or (~b).sum() == 0:
        return float("nan")
    drop_iq = iq_z[~b].mean() - iq_z[b].mean()
    drop_adp = adp_z[~b].mean() - adp_z[b].mean()
    return 0.5 * (drop_iq + drop_adp)


def what_resize_does(mask_shape, target_shape):
    if mask_shape == target_shape:
        return "keep (same shape)"
    if mask_shape == target_shape[::-1]:
        return "transpose"
    return "STRETCH (zoom) - shapes do not match"


def overlay(img, boundary, color):
    g = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-9)
    rgb = np.dstack([g, g, g])
    rgb[boundary] = color
    return rgb


def diagnose(pair, out_dir):
    key = f"{pair['sample']}_{pair['location']}_{pair['magnification']}"
    print(f"\n=== {key} ===")
    raw = load_mat_array(str(pair["mask_path"])).astype(float)
    iq = np.load(pair["iq_path"]).astype(float)
    adp = np.load(pair["adp_path"]).astype(float)
    iq_z, adp_z = zscore(iq), zscore(adp)

    nan = np.isnan(raw)
    ids = np.unique(raw[~nan])
    nan_patches, n_patches = ndi.label(nan)
    largest_nan = int(ndi.sum(nan, nan_patches, range(1, n_patches + 1)).max()) if n_patches else 0
    print(f"mask shape {raw.shape} | iq shape {iq.shape} | {len(ids)} grain IDs | "
          f"NaN {100 * nan.mean():.1f}% ({n_patches} patches, largest {largest_nan} px)")
    if not np.all(ids == np.round(ids)):
        print("  !! values are NOT whole numbers -> this is probably not a grain-ID map")

    # --- 1. which orientation puts the boundaries where IQ/ADP drop? -------------
    scores = {}
    for name, fn in ORIENTATIONS.items():
        cand = fn(raw)
        if cand.shape == iq.shape:
            scores[name] = boundary_contrast(cand, iq_z, adp_z)
    best = max(scores, key=scores.get)
    resize_action = what_resize_does(raw.shape, iq.shape)
    pipeline_orientation = {"keep (same shape)": "as-is", "transpose": "transpose"}.get(resize_action, "stretched")
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        tag = "  <- best" if name == best else ""
        tag += "  <- used by generate_crops" if name == pipeline_orientation else ""
        print(f"  {name:16s} boundary contrast {s:+.3f}{tag}")

    second = sorted(scores.values())[-2] if len(scores) > 1 else float("nan")
    margin = scores[best] - second
    problems = []
    if pipeline_orientation == "stretched":
        problems.append("mask is STRETCHED, not transposed")
    elif pipeline_orientation != best:
        problems.append(f"MISALIGNED: pipeline uses '{pipeline_orientation}', best is '{best}'")
    if np.isfinite(margin) and margin < 0.05:
        problems.append("orientation check is uncertain (small margin) - check the figure by eye")

    # --- 2. how much does the cleaning pipeline change the boundaries? ----------
    aligned = ORIENTATIONS[best](raw)
    raw_b = find_boundaries(fill_nan_nearest(aligned).astype(np.int64), mode="outer")
    piped = resize_mask_to(raw, iq.shape)                          # exactly as generate_crops
    piped = handle_nan_labels(piped)
    piped = remove_small_regions(denoise_labels(piped, radius=MODE_FILTER_RADIUS),
                                 min_size=MIN_REGION_SIZE).astype(np.int64)
    pipe_b = mask_to_binary_boundary(piped) == 0
    near_raw = ndi.distance_transform_edt(~raw_b) <= 2
    near_pipe = ndi.distance_transform_edt(~pipe_b) <= 2
    extra = (pipe_b & ~near_raw).sum() / max(pipe_b.sum(), 1)
    missing = (raw_b & ~near_pipe).sum() / max(raw_b.sum(), 1)
    print(f"  pipeline vs raw IDs: {100 * extra:.1f}% extra boundary, {100 * missing:.1f}% missing boundary")
    if extra > 0.15 or missing > 0.15:
        problems.append("pipeline changes boundaries a lot")
    for p in problems:
        print(f"  !! {p}")

    # --- figure ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.5))
    ax[0].imshow(overlay(iq, pipe_b, (1, 0, 0)))
    ax[0].set_title(f"IQ + boundaries from generate_crops ({pipeline_orientation})")
    ax[1].imshow(overlay(iq, raw_b, (0, 0.8, 0)))
    ax[1].set_title(f"IQ + raw IDs in best orientation ({best})")
    diff = np.ones(iq.shape + (3,))
    diff[pipe_b & near_raw] = (0, 0, 0)
    diff[pipe_b & ~near_raw] = (0.9, 0, 0)
    diff[raw_b & ~near_pipe] = (0, 0.3, 1)
    ax[2].imshow(diff)
    ax[2].set_title("black = same, RED = only generate_crops, BLUE = only raw IDs")
    for a in ax:
        a.axis("off")
    fig.suptitle(key + ("   |   " + "; ".join(problems) if problems else "   |   OK"), fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}_diagnosis.png", dpi=80)
    plt.close(fig)

    return {
        "scan": key, "mat_file": pair["mask_path"].name,
        "mask_shape": raw.shape, "iq_shape": iq.shape, "n_grain_ids": len(ids),
        "nan_percent": round(100 * nan.mean(), 2), "largest_nan_patch_px": largest_nan,
        "resize_mask_to_does": resize_action, "best_orientation": best,
        "orientation_margin": round(margin, 3),
        "pipeline_extra_boundary_%": round(100 * extra, 1),
        "pipeline_missing_boundary_%": round(100 * missing, 1),
        "problems": "; ".join(problems) or "OK",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=None, help="only scans whose name contains this text")
    parser.add_argument("--out-dir", default="experiments/mask_diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(MASK_ROOT, IMAGE_ROOT)
    if args.key:
        pairs = [p for p in pairs
                 if args.key in f"{p['sample']}_{p['location']}_{p['magnification']}"
                 or args.key in p["mask_path"].name]

    rows = [diagnose(p, out_dir) for p in pairs]
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    bad = [r for r in rows if r["problems"] != "OK"]
    print(f"\nChecked {len(rows)} scan(s); {len(bad)} flagged. Figures + summary.csv in {out_dir}/")
    for r in bad:
        print(f"  {r['scan']}: {r['problems']}")


if __name__ == "__main__":
    main()