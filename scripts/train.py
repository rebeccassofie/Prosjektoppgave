#!/usr/bin/env python
"""
Entry point for training the segmentation UNet.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 16 --lr 1e-3
"""
import argparse

from src.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/input_image")
    parser.add_argument("--mask-dir", default="data/expert_label")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--optimizer", default="adamw",
                         choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--checkpoint-path", default="models/best_unet.pth")
    parser.add_argument("--history-path", default="output/history.csv")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    best_val_loss, best_val_dice, test_loss, test_dice = train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        learning_rate=args.lr,
        pos_weight_value=args.pos_weight,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        use_augmentation=not args.no_augmentation,
        optimizer_name=args.optimizer,
        verbose=not args.quiet,
    )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}, best val dice: {best_val_dice:.4f}")
    print(f"Final test loss: {test_loss:.4f}, final test dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()
