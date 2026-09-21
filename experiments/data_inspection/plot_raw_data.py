import numpy as np
import matplotlib.pyplot as plt
from src.mask_utils import load_mat_array


MAT_PATH = "datasets/raw/PGs/1/midt/010 parentProp_parentId.mat"
IQ_PATH = "datasets/raw/SDA/1/midt/x100_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/1/midt/x100_processed/021 adp_1Dsig.npy"


def main():
    mask = load_mat_array(MAT_PATH)
    iq = np.load(IQ_PATH)
    adp = np.load(ADP_PATH)

    #raw figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(mask, cmap="nipy_spectral")
    axes[0].set_title(f"Mask (.mat)\nshape={mask.shape}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(iq, cmap="gray")
    axes[1].set_title(f"IQ\nshape={iq.shape}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(adp, cmap="gray")
    axes[2].set_title(f"ADP\nshape={adp.shape}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("experiments/data_inspection/results/mask_vs_images_raw.png", dpi=150)
    print("\nSaved plot to experiments/data_inspection/results/mask_vs_images_raw.png")
    plt.show()

    #added transpose and grayscale

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(mask.T, cmap="nipy_spectral")
    axes[0].set_title(f"Mask (.mat)\nshape={mask.T.shape}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(iq, cmap="gray")
    axes[1].set_title(f"IQ\nshape={iq.shape}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(adp, cmap="gray")
    axes[2].set_title(f"ADP\nshape={adp.shape}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("experiments/data_inspection/results/mask_vs_images_raw_transpose.png", dpi=150)
    print("\nSaved plot to experiments/data_inspection/results/mask_vs_images_raw_transpose.png")
    plt.show()


if __name__ == "__main__":
    main()