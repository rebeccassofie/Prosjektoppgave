import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from PIL import Image
from src.experiment_utils import create_run_dir

from src.model import UNet
from src.dataset import SegmentationDataset, build_train_transform
from src.losses import DiceLoss
from src.evaluate import dice_score


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name, params, learning_rate):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from adam, adamw, sgd, rmsprop.")


def save_run_params(run_dir, params):
    """Writes every hyperparameter actually used for this run to params.csv."""
    with open(run_dir / "params.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for key, value in params.items():
            writer.writerow([key, value])
 
 
def save_predicted_masks(model, dataset, device, output_dir):
    """Runs inference over every sample in dataset and saves the predicted
    mask, using the same filename as the original image crop."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            image, _ = dataset[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()
 
            # 1 = boundary line, 0 = background -- invert back to
            # white-background/black-line, matching the original masks.
            out_img = ((1 - pred_mask) * 255).astype("uint8")
            filename = Path(sample["image_path"]).name
            Image.fromarray(out_img).save(output_dir / filename)
 
 
def plot_training_history(history, save_path):
    train_losses, val_losses, val_dices = zip(*history)
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
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def plot_prediction_grid(model, dataset, device, save_path, num_samples=6):
    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]  # keep indexing consistent for the single-sample case
 
    axes[0, 0].set_title("Input image")
    axes[0, 1].set_title("Ground truth mask")
    axes[0, 2].set_title("Predicted mask")
 
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            image, gt_mask = dataset[i]
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_mask = (pred > 0.5).float()
 
            # image is single-channel (1, H, W), not RGB
            img_np = image.squeeze(0).numpy()
            gt_np = gt_mask.squeeze(0).numpy()
            pred_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
 
            axes[i, 0].imshow(img_np, cmap="gray")
            # invert so lines display as black-on-white, like the originals
            axes[i, 1].imshow(1 - gt_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(1 - pred_np, cmap="gray", vmin=0, vmax=1)
            for ax in axes[i]:
                ax.axis("off")
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def train_model(image_dir="datasests/cropped/SDA", mask_dir="datasets/cropped/PGs",
                 learning_rate=5e-4, pos_weight_value=3.0, batch_size=8,
                 num_epochs=100, experiments_dir="experiments/training",
                 num_prediction_samples=6,
                 save_history=True, verbose=True, use_augmentation=True,
                 optimizer_name="adamw", train_frac=0.8, val_frac=0.1):
    """Trains a UNet with the given hyperparameters.
 
    Every call creates a new run directory at
    experiments_dir/<dd.mm.yy>/run<N>/, and saves everything from that run
    into it: params.csv (the actual hyperparameters used), the best
    checkpoint, history.csv, training_history.png, predictions.png, and a
    predicted_label/ folder with every predicted mask.
 
    train_frac / val_frac are fractions of the total dataset (default 80%/10%,
    with the remaining 10% used as the test set).
 
    Returns (run_dir, best_val_loss, best_val_dice, test_loss, test_dice).
    """
    run_dir = create_run_dir(experiments_dir)
    checkpoint_path = run_dir / "best_unet.pth"
    history_path = run_dir / "history.csv"
 
    if verbose:
        print(f"Run directory: {run_dir}")
 
    save_run_params(run_dir, {
        "image_dir": image_dir, "mask_dir": mask_dir,
        "learning_rate": learning_rate, "pos_weight_value": pos_weight_value,
        "batch_size": batch_size, "num_epochs": num_epochs,
        "use_augmentation": use_augmentation, "optimizer_name": optimizer_name,
        "train_frac": train_frac, "val_frac": val_frac,
    })
 
    device = get_device()
    if verbose:
        print("Using device:", device)
 
    aug = build_train_transform() if use_augmentation else None
    train_dataset = SegmentationDataset(image_dir, mask_dir, transform=aug)
    plain_dataset = SegmentationDataset(image_dir, mask_dir, transform=None)
 
    n = len(plain_dataset)
    train_split = int(n * train_frac)
    val_split = int(n * val_frac)
 
    if verbose:
        print(f"Dataset size: {n} -> train={train_split}, val={val_split}, "
              f"test={n - train_split - val_split}")
 
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:train_split]
    val_idx = indices[train_split:train_split + val_split]
    test_idx = indices[train_split + val_split:]
 
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(plain_dataset, val_idx)
    test_set = Subset(plain_dataset, test_idx)  # never touched until final evaluation
 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
 
    model = UNet(n_classes=1, n_channels=1).to(device)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice_loss = DiceLoss()
 
    def criterion(outputs, masks):
        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)
        return 0.5 * bce + 0.5 * dice
 
    optimizer = build_optimizer(optimizer_name, model.parameters(), learning_rate)
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
 
    best_val_loss = float("inf")
    best_val_dice = -1.0
    history = []
 
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
 
        model.eval()
        dice_total = 0.0
        num_images = 0
 
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                batch_dice = dice_score(outputs, masks)
                dice_total += batch_dice * images.size(0)
                num_images += images.size(0)
 
        val_dice = dice_total / num_images
        scheduler.step(val_loss)
        history.append((train_loss, val_loss, val_dice))
        current_lr = optimizer.param_groups[0]["lr"]
 
        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val dice: {val_dice:.4f} | "
                f"lr: {current_lr:.2e}"
            )
 
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  -> saved new best model (val dice {val_dice:.4f})")
 
    if save_history:
        with open(history_path, "w") as f:
            f.write("train_loss,val_loss,val_dice\n")
            for train_loss, val_loss, val_dice in history:
                f.write(f"{train_loss},{val_loss},{val_dice}\n")
        if verbose:
            print(f"Saved training history to {history_path}")
 
    # Final, one-time evaluation on the held-out test set using the best
    # checkpoint (selected via validation, never touched by test data).
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_loss = run_epoch(model, test_loader, criterion, device)
    dice_total = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_total += dice_score(outputs, masks) * images.size(0)
    test_dice = dice_total / len(test_loader.dataset)
 
    if verbose:
        print(f"Test loss: {test_loss:.4f} | Test dice: {test_dice:.4f}")
 
    # Visualization -- reuses the same best-checkpoint model already in
    # memory, no need to reload it a second time.
    plot_training_history(history, run_dir / "training_history.png")
    plot_prediction_grid(model, plain_dataset, device, run_dir / "predictions.png",
                          num_samples=num_prediction_samples)
    save_predicted_masks(model, plain_dataset, device, run_dir / "predicted_label")
 
    if verbose:
        print(f"Saved training_history.png, predictions.png, and predicted_label/ to {run_dir}")
 
    return run_dir, best_val_loss, best_val_dice, test_loss, test_dice
 