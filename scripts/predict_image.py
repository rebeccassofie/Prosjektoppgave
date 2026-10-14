"""
Run a trained model on a new (larger) image, e.g. a 512 x 512 region.

The U-Net is fully convolutional, so it accepts any image size (padded to a
multiple of 32 internally). Two ways to run it:

  --mode whole   the whole region in one pass. The model sees more context
                 than in training, but the image is normalised over the whole
                 region instead of per 128 px crop.
  --mode tiles   cut into 128 x 128 tiles, each normalised exactly like the
                 training crops, predicted one by one and stitched back.
                 Closest to how the model was trained.
Try both and compare.

Inputs can be the raw .npy maps (normalised here the same way as
generate_crops.py) or already-normalised PNGs. For a model trained with
--modality stacked, give the IQ map first, then the ADP map.

Examples (from the project root):
  # 512x512 region starting at x=0, y=0 of a full ADP scan, compared with its mask
  python -m scripts.predict_image --run-dir experiments/training/10.13.26/run3 \\
      --images "datasets/raw/SDA/4/ytre/x200_processed/021 adp_1Dsig.npy" \\
      --gt datasets/cropped/PGs_full/4_ytre_x200_full.png --region 0 0 512

  # stacked model: IQ then ADP
  python -m scripts.predict_image --run-dir <run> --images iq.npy adp.npy --region 0 0 512

IMPORTANT: if the region overlaps crops the model was trained on, the score is
too optimistic. Use a scan (or area) that was not in the training set.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from src.evaluate import boundary_scores

TILE = 128


def build_model(in_channels):
    """MUST match the model created in src/train.py (weights come from the checkpoint)."""
    return smp.UnetPlusPlus(encoder_name="resnet18", encoder_weights=None,
                            in_channels=in_channels, classes=1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_params(run_dir):
    with open(Path(run_dir) / "params.csv") as f:
        return {row["parameter"]: row["value"] for row in csv.DictReader(f)}


def to_training_scale(a):
    """Same as generate_crops.py: min-max to [0, 1], saved as 8-bit PNG, read back / 255."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    a = np.nan_to_num((a - lo) / (hi - lo + 1e-6), nan=0.0)
    return np.round(np.clip(a, 0, 1) * 255) / 255


def load_channel(path, region):
    """Returns (array, is_raw). Raw .npy maps still need normalising; PNGs are already 0-1."""
    path = Path(path)
    if path.suffix == ".npy":
        a, raw = np.load(path).astype(float), True
    else:
        a, raw = np.array(Image.open(path).convert("L")).astype(float) / 255.0, False
    if region:
        x, y, s = region
        a = a[y:y + s, x:x + s]
    return a, raw


@torch.no_grad()
def predict_array(model, x, device):
    """x: (C, H, W) in [0, 1]. Pads to a multiple of 32, returns (H, W) probabilities."""
    _, h, w = x.shape
    ph, pw = (-h) % 32, (-w) % 32
    xp = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(xp[None]).float().to(device)
    return torch.sigmoid(model(t))[0, 0, :h, :w].cpu().numpy()


def predict_whole(model, channels, raw, device):
    x = np.stack([to_training_scale(c) if r else c for c, r in zip(channels, raw)])
    return predict_array(model, x, device)


def predict_tiles(model, channels, raw, device):
    h, w = channels[0].shape
    prob = np.zeros((h, w))
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            tile = [c[y0:y0 + TILE, x0:x0 + TILE] for c in channels]
            tile = [to_training_scale(t) if r else t for t, r in zip(tile, raw)]  # per-tile, like training
            prob[y0:y0 + TILE, x0:x0 + TILE] = predict_array(model, np.stack(tile), device)
    return prob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="training run folder with best_unet.pth and params.csv")
    ap.add_argument("--images", nargs="+", required=True, help="input map(s): .npy or .png (stacked: IQ then ADP)")
    ap.add_argument("--gt", default=None, help="optional ground-truth mask PNG (black lines on white)")
    ap.add_argument("--region", nargs=3, type=int, metavar=("X", "Y", "SIZE"), default=None,
                    help="cut a SIZE x SIZE region starting at column X, row Y (applied to images and GT)")
    ap.add_argument("--mode", choices=["whole", "tiles"], default="whole")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tolerance", type=int, default=3, help="pixel tolerance for boundary F1")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/custom_predictions")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    params = read_params(run_dir)
    in_channels = int(params.get("in_channels", 1))
    if len(args.images) != in_channels:
        raise SystemExit(f"This model expects {in_channels} input map(s) "
                         f"(modality={params.get('modality')}), got {len(args.images)}.")

    device = get_device()
    model = build_model(in_channels).to(device)
    model.load_state_dict(torch.load(run_dir / "best_unet.pth", map_location=device))
    model.eval()   # use the stored BatchNorm statistics, not the batch's

    loaded = [load_channel(p, args.region) for p in args.images]
    channels, raw = [c for c, _ in loaded], [r for _, r in loaded]
    if len({c.shape for c in channels}) != 1:
        raise SystemExit(f"Input maps have different shapes: {[c.shape for c in channels]}")
    if args.mode == "tiles" and any(s % TILE for s in channels[0].shape):
        raise SystemExit(f"--mode tiles needs a size that is a multiple of {TILE}, got {channels[0].shape}")

    predict = predict_tiles if args.mode == "tiles" else predict_whole
    prob = predict(model, channels, raw, device)
    pred = prob > args.threshold

    gt = None
    if args.gt:
        gt_img, _ = load_channel(args.gt, args.region)
        gt = gt_img < 0.5                       # black lines = boundary, like the training masks
        if gt.shape != pred.shape:
            raise SystemExit(f"GT shape {gt.shape} does not match prediction {pred.shape}")
        logits = torch.logit(torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6))[None, None]
        P, R, F1 = boundary_scores(logits, torch.from_numpy(gt.astype(np.float32))[None, None],
                                   threshold=args.threshold, tolerance=args.tolerance)
        print(f"Boundary F1 = {F1:.4f}  (precision {P:.4f}, recall {R:.4f}, tolerance {args.tolerance} px)")

    # ---- save outputs
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "custom_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.images[-1]).stem.replace(" ", "_")
    if args.region:
        name += "_x{}_y{}_s{}".format(*args.region)
    name += f"_{args.mode}"
    Image.fromarray(((~pred) * 255).astype(np.uint8)).save(out_dir / f"{name}_mask.png")

    labels = ["IQ", "ADP"] if in_channels == 2 else [params.get("modality", "input").upper()]
    panels = [(to_training_scale(c) if r else c, f"Input {l}", "gray")
              for c, r, l in zip(channels, raw, labels)]
    panels += [(prob, "Boundary probability", "magma"), (~pred, "Prediction", "gray")]
    if gt is not None:
        panels.append((~gt, "Ground truth", "gray"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    for ax, (img, title, cmap) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    title = f"{name}   ({args.mode})"
    if gt is not None:
        title += f"   boundary F1 = {F1:.3f}"
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=120)
    print(f"Saved {out_dir / (name + '.png')} and {name}_mask.png")


if __name__ == "__main__":
    main()