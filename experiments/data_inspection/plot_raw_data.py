import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.mask_utils import load_mat_array, ORIENTATIONS


MAT_PATH = "datasets/raw/PGs/2/910nr5/010 parentProp_parentId.mat"
IQ_PATH = "datasets/raw/SDA/2/910nr5/x200_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/2/910nr5/x200_processed/021 adp_1Dsig.npy"
OUT_DIR = "experiments/data_inspection/results"


def random_label_cmap(labels, seed=0):
    """One random colour per grain ID, so neighbouring grains stand out
    (with nipy_spectral, similar IDs get similar colours). NaN = white."""
    rng = np.random.default_rng(seed)  # fixed seed -> same colours every run
    n_ids = int(np.nanmax(labels)) + 1
    cmap = ListedColormap(rng.uniform(0.15, 1.0, size=(n_ids, 3)))  # no near-black
    cmap.set_bad("white")
    return cmap


def plot_mask_vs_images(mask, iq, adp, mask_title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # vmin/vmax pin ID n to colour n; "nearest" keeps grain edges sharp.
    # No colorbar for the mask: the ID numbers carry no meaning.
    axes[0].imshow(np.ma.masked_invalid(mask), cmap=random_label_cmap(mask),
                   vmin=0, vmax=np.nanmax(mask), interpolation="nearest")
    axes[0].set_title(f"{mask_title}\nshape={mask.shape}")

    im1 = axes[1].imshow(iq, cmap="gray")
    axes[1].set_title(f"IQ\nshape={iq.shape}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(adp, cmap="gray")
    axes[2].set_title(f"ADP\nshape={adp.shape}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()


def main():
    mask = load_mat_array(MAT_PATH).astype(float)
    iq = np.load(IQ_PATH)
    adp = np.load(ADP_PATH)

    plot_mask_vs_images(mask, iq, adp, "Mask (.mat), raw",
                        f"{OUT_DIR}/mask_vs_images_raw_910nr5.png")
    plot_mask_vs_images(ORIENTATIONS["transpose"](mask), iq, adp, "Mask (.mat), transposed",
                        f"{OUT_DIR}/mask_vs_images_raw_transpose_910nr5.png")
    # 910nr5 is the scan that needs anti-transpose (see diagnose_masks)
    plot_mask_vs_images(ORIENTATIONS["anti-transpose"](mask), iq, adp, "Mask (.mat), anti-transposed",
                        f"{OUT_DIR}/mask_vs_images_raw_antitranspose_910nr5.png")


if __name__ == "__main__":
    main()