import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from scripts.plot_denoised_mask import load_mask, resize_mask_to, handle_nan_labels, denoise_labels


MASK_PATH = "datasets/raw/PGs/1/midt/010 parentProp_parentId.mat"   
IQ_PATH = "datasets/raw/SDA/1/midt/x100_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/1/midt/x100_processed/021 adp_1Dsig.npy"
MODE_FILTER_RADIUS = 3


def mask_to_binary_boundary(mask):
    """
    Returns a strictly binary image: 1.0 (white) = background, 0.0 (black)
    = boundary. Only two classes -- no grain-ID color information kept.
    """
    boundaries = find_boundaries(mask, mode="outer")
    binary_img = np.ones(mask.shape, dtype=float)
    binary_img[boundaries] = 0.0
    return binary_img

def main():
    mask = load_mask(MASK_PATH)
    iq = np.load(IQ_PATH)
    adp = np.load(ADP_PATH)
 
    print(f"\nmask: shape={mask.shape}, {np.unique(mask).size} labels")
    print(f"iq:   shape={iq.shape}")
    print(f"adp:  shape={adp.shape}")
 
    mask = resize_mask_to(mask, iq.shape)
    mask = handle_nan_labels(mask)
    denoised = denoise_labels(mask, radius=MODE_FILTER_RADIUS)
    denoised = denoised.astype(np.int64)  
 
    binary_boundary = mask_to_binary_boundary(denoised)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
    axes[0].imshow(binary_boundary, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Boundary mask (binary)")
 
    axes[1].imshow(iq, cmap="gray")
    axes[1].set_title("Raw IQ")
 
    axes[2].imshow(adp, cmap="gray")
    axes[2].set_title("Raw ADP")
 
    for ax in axes:
        ax.axis("off")
 
    plt.tight_layout()
    plt.savefig("experiments/plot_processed_data/binary_boundary_vs_raw.png", dpi=150)
    print("\nSaved plot to experiments/plot_processed_data/binary_boundary_vs_raw.png")
    plt.show()
 
 
if __name__ == "__main__":
    main()