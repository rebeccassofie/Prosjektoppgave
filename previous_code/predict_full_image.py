import os
import numpy as np
import torch
from PIL import Image

from unet_128 import UNet

PATCH_SIZE = 128


def predict_full_image(image_path, checkpoint_path="models/best_unet.pth",
                        output_path="full_image_prediction.png", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")

    model = UNet(n_class=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load the full image
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"Loaded image: {w}x{h}")

    # Pad up to the next multiple of PATCH_SIZE in each dimension, using
    # mirror/reflect padding (same idea as the original U-Net paper's
    # overlap-tile strategy for handling borders).
    pad_h = (-h) % PATCH_SIZE
    pad_w = (-w) % PATCH_SIZE
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    padded_h, padded_w = padded.shape[:2]
    print(f"Padded to: {padded_w}x{padded_h} ({padded_h // PATCH_SIZE}x"
          f"{padded_w // PATCH_SIZE} tiles of {PATCH_SIZE}x{PATCH_SIZE})")

    # Predicted mask, same padded size, single channel
    pred_full = np.zeros((padded_h, padded_w), dtype="float32")

    with torch.no_grad():
        for y in range(0, padded_h, PATCH_SIZE):
            for x in range(0, padded_w, PATCH_SIZE):
                tile = padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                tile_tensor = tile_tensor.unsqueeze(0).to(device)

                pred = torch.sigmoid(model(tile_tensor))
                pred_mask = (pred > threshold).float().squeeze().cpu().numpy()

                pred_full[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = pred_mask

    # Crop back down to the original (unpadded) size
    pred_full = pred_full[:h, :w]

    # pred_full: 1 = boundary line, 0 = background.
    # Convert to white-background/black-line image, matching your mask style.
    out_img = ((1 - pred_full) * 255).astype("uint8")
    Image.fromarray(out_img).save(output_path)
    print(f"Saved full-image prediction to {output_path}")


if __name__ == "__main__":
    # Adjust this path to point at your full-size image
    predict_full_image("data/full_metallographic_images_for_reference/10.png")