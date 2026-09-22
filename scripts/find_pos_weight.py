import os
import numpy as np
from PIL import Image

pgs_dir = "datasets/cropped/PGs"
fractions = []

for f in os.listdir(pgs_dir):
    if not f.endswith(".png") or "_full" in f:
        continue
    arr = np.array(Image.open(os.path.join(pgs_dir, f)).convert("L"))
    frac_boundary = (arr < 127).mean()
    fractions.append(frac_boundary)

fractions = np.array(fractions)
mean_frac = fractions.mean()
recommended_pos_weight = (1 - mean_frac) / mean_frac

print(f"mean boundary pixel fraction: {mean_frac:.4f}  ({mean_frac*100:.2f}% of pixels are boundary)")
print(f"recommended pos_weight ≈ {recommended_pos_weight:.1f}   (you're using 3.0)")