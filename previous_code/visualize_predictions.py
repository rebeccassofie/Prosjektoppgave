import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

from unet_128 import UNet
from train_unet import SegmentationDataset


def main():
    image_dir = "data/input_image"
    mask_dir = "data/expert_label"
    checkpoint_path = "models/best_unet.pth"
    num_samples = 6

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")

    # No augmentation at inference time
    dataset = SegmentationDataset(image_dir, mask_dir, transform=None)

    model = UNet(n_class=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference on every image and save the predicted mask, using the
    # same filename as the original image, into predicted_label/
    output_dir = "predicted_label"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, filename in enumerate(dataset.filenames):
            image, _ = dataset[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

            # pred_mask: 1 = boundary line, 0 = background.
            # Convert back to white-background/black-line image, like the originals.
            out_img = ((1 - pred_mask) * 255).astype("uint8")
            Image.fromarray(out_img).save(os.path.join(output_dir, filename))

    print(f"Saved {len(dataset.filenames)} predicted masks to '{output_dir}/'")

    # Plot training history (loss + dice) from training
    with open("output/history.csv") as f:
        lines = f.readlines()[1:]  # skip header
    train_losses, val_losses, val_dices = zip(
        *[tuple(map(float, line.strip().split(","))) for line in lines]
    )
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_losses, marker="o", label="Train loss")
    ax1.plot(epochs, val_losses, marker="o", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per epoch")
    ax1.legend()

    ax2.plot(epochs, val_dices, marker="o", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Dice score")
    ax2.set_title("Validation Dice score per epoch")

    plt.tight_layout()
    plt.savefig("output/training_history.png", dpi=150)
    print("Saved training history plot to output/training_history.png")
    plt.show()

    # Plot sample predictions
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    axes[0, 0].set_title("Input image")
    axes[0, 1].set_title("Ground truth mask")
    axes[0, 2].set_title("Predicted mask")

    with torch.no_grad():
        for i in range(num_samples):
            image, gt_mask = dataset[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_mask = (pred > 0.5).float()

            img_np = image.permute(1, 2, 0).numpy()
            gt_np = gt_mask.squeeze(0).numpy()
            pred_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

            axes[i, 0].imshow(img_np)
            # Masks are stored as 1=line, 0=background for training;
            # invert here so lines display as black-on-white like the originals.
            axes[i, 1].imshow(1 - gt_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(1 - pred_np, cmap="gray", vmin=0, vmax=1)
            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig("output/predictions.png", dpi=150)
    print("Saved prediction grid to output/predictions.png")
    plt.show()


if __name__ == "__main__":
    main()