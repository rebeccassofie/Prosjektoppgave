"""
Quick plot of a parent-grain .mat mask alongside its corresponding
IQ / ADP .npy scans.
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


MAT_PATH = "datasets/raw/PGs/1/midt/010 parentProp_parentId.mat"
IQ_PATH = "datasets/raw/SDA/1/midt/x100_processed/020 iq_sdan.npy"
ADP_PATH = "datasets/raw/SDA/1/midt/x100_processed/021 adp_1Dsig.npy"



def load_mat_array(path):
    """
    Loads a .mat file and returns its array data.

    scipy.io.loadmat gives back a dict with some housekeeping keys
    (__header__, __version__, __globals__) plus your actual variable(s).
    This prints what's inside so you can see the real variable name and
    pick the right one if there's more than one.
    """
    data = scipy.io.loadmat(path)
    real_keys = [k for k in data if not k.startswith("__")]

    print(f"Keys found in {path}: {real_keys}")
    for k in real_keys:
        print(f"  '{k}': shape={data[k].shape}, dtype={data[k].dtype}")

    if len(real_keys) == 1:
        return data[real_keys[0]]

    # More than one variable -- change this to the key you actually want.
    chosen_key = real_keys[0]
    print(f"Multiple keys found, using '{chosen_key}' -- edit load_mat_array() to change this.")
    return data[chosen_key]


def main():
    mask = load_mat_array(MAT_PATH)
    iq = np.load(IQ_PATH)
    adp = np.load(ADP_PATH)

    print(f"\nmask: shape={mask.shape}, dtype={mask.dtype}, "
          f"min={mask.min()}, max={mask.max()}, unique values={np.unique(mask).size}")
    print(f"iq:   shape={iq.shape}, dtype={iq.dtype}, min={iq.min():.4f}, max={iq.max():.4f}")
    print(f"adp:  shape={adp.shape}, dtype={adp.dtype}, min={adp.min():.4f}, max={adp.max():.4f}")

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
    plt.savefig("experiments/plot_raw_data/mask_vs_images_raw.png", dpi=150)
    print("\nSaved plot to experiments/plot_raw_data/mask_vs_images_raw.png")
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
    plt.savefig("experiments/plot_raw_data/mask_vs_images_transpose.png", dpi=150)
    print("\nSaved plot to experiments/plot_raw_data/mask_vs_images_transpose.png")
    plt.show()


if __name__ == "__main__":
    main()