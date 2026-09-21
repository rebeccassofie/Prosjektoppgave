import numpy as np
import matplotlib.pyplot as plt
from src.mask_utils import load_mat_array, resize_mask_to, handle_nan_labels, denoise_labels


MASK_PATH = "datasets/raw/PGs/1/midt/010 parentProp_parentId.mat"   
IQ_PATH = "datasets/raw/SDA/1/midt/x100_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/1/midt/x100_processed/021 adp_1Dsig.npy"
MODE_FILTER_RADIUS = 3                   


def main():
    mask = load_mat_array(MASK_PATH)
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
    plt.savefig("experiments/data_inspection/results/mask_vs_images_denoised.png", dpi=150)
    print("\nSaved plot to experiments/data_inspection/results/mask_vs_images_denoised.png")
    plt.show()


if __name__ == "__main__":
    main()