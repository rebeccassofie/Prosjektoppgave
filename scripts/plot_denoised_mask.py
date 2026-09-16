import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import disk


MASK_PATH = "datasets/raw/PGs/1/midt/010 parentProp_parentId.mat"   
IQ_PATH = "datasets/raw/SDA/1/midt/x100_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/1/midt/x100_processed/021 adp_1Dsig.npy"
MODE_FILTER_RADIUS = 3                   


# loader matlab arrayet og returnerer dimensjoene til bildet
def load_mask(path):
    data = scipy.io.loadmat(path)
    real_keys = [k for k in data if not k.startswith("__")]
    print(f"Keys found in {path}: {real_keys}")
    for k in real_keys:
        print(f"  '{k}': shape={data[k].shape}, dtype={data[k].dtype}")
    chosen_key = real_keys[0]
    if len(real_keys) > 1:
        print(f"Multiple keys found, using '{chosen_key}' -- edit this function to change.")
    return data[chosen_key]


def resize_mask_to(mask, target_shape):
    """
    Aligns mask onto target_shape.
 
    If mask.shape is the EXACT reverse of target_shape, this is almost
    certainly a MATLAB (column-major) vs Python (row-major) axis-order
    difference, not a genuine resolution mismatch -- transposing preserves
    every pixel value exactly, whereas resizing would stretch/distort the
    grain shapes and can introduce interpolation artifacts.
 
    Otherwise falls back to a genuine nearest-neighbor resize (scipy, not
    PIL -- PIL's float image mode does not reliably round-trip float64
    label arrays).
    """
    if mask.shape == target_shape:
        return mask
 
    if mask.shape == target_shape[::-1]:
        print(f"Mask shape {mask.shape} is the exact transpose of target "
              f"{target_shape} -- transposing instead of resizing.")
        return mask.T
 
    print(f"Mask shape {mask.shape} does not match target {target_shape} -- "
          f"resizing with nearest-neighbor interpolation.")
    from scipy.ndimage import zoom
    zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
    return zoom(mask, zoom_factors, order=0)


def handle_nan_labels(mask):
    """
    Replaces NaN pixels with an explicit sentinel label
    """
    nan_mask = np.isnan(mask)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        sentinel = np.nanmax(mask) + 1 if np.isfinite(np.nanmax(mask)) else -1
        print(f"Found {n_nan} NaN pixels ({100 * n_nan / mask.size:.1f}%) -- "
              f"replacing with sentinel label {sentinel}.")
        mask = np.where(nan_mask, sentinel, mask)
    return mask

def denoise_labels(mask, radius=3):
    """Majority/mode filter over a disk neighborhood to clean speckle noise."""
    unique_vals, remapped = np.unique(mask, return_inverse=True)
    remapped = remapped.reshape(mask.shape).astype(np.uint16)
    denoised_remapped = modal(remapped, disk(radius))
    return unique_vals[denoised_remapped]


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
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
    axes[0].imshow(denoised, cmap="nipy_spectral")
    axes[0].set_title(f"Denoised mask\n{np.unique(denoised).size} labels")
 
    axes[1].imshow(iq, cmap="gray")
    axes[1].set_title("Raw IQ")
 
    axes[2].imshow(adp, cmap="gray")
    axes[2].set_title("Raw ADP")
 
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("experiments/plot_processed_data/mask_vs_images_denoised.png", dpi=150)
    print("\nSaved plot to experiments/plot_processed_data/mask_vs_images_denoised.png")
    plt.show()


if __name__ == "__main__":
    main()