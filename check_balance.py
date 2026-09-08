import os
import numpy as np
from PIL import Image

mask_dir = "TBM/expert_label"
valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

foreground_fracs = []

filenames = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(valid_ext))

for fname in filenames:
    mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L"))
    # Masks are white background with black boundary lines. The lines are the
    # class of interest, so "foreground" = dark pixels, not bright ones.
    binary = mask < 127
    foreground_frac = binary.mean()
    foreground_fracs.append(foreground_frac)

foreground_fracs = np.array(foreground_fracs)

print(f"Checked {len(foreground_fracs)} mask files in '{mask_dir}'")
print(f"Average foreground fraction: {foreground_fracs.mean():.3f}")
print(f"Min: {foreground_fracs.min():.3f}, Max: {foreground_fracs.max():.3f}")

# Suggested pos_weight for nn.BCEWithLogitsLoss(pos_weight=...) if imbalanced
avg_fg = foreground_fracs.mean()
if avg_fg > 0:
    suggested_pos_weight = (1 - avg_fg) / avg_fg
    print(f"\nSuggested pos_weight for BCEWithLogitsLoss: {suggested_pos_weight:.2f}")