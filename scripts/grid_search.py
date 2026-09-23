import argparse
import csv
import itertools
import time

from src.experiment_utils import create_run_dir
from src.train import train_model

PARAM_GRID = {
    "lr": [1e-3], #[1e-4, 5e-4, 1e-3],
    "batch_size": [8],
    "pos_weight": [8.0, 11.2, 14],
    "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
    "use_augmentation": [True, False],
    "modality": ["both", "iq", "adp"],
}


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="datasets/cropped/SDA")
    parser.add_argument("--mask-dir", default="datasets/cropped/PGs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--experiments-dir", default="experiments/grid_search")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Running grid search over {len(combos)} combinations, {args.epochs} epochs each.")

    search_dir = create_run_dir(args.experiments_dir)
    trials_dir = search_dir / "trials"
    results_file = search_dir / "grid_search_results.csv"
    top_n_file = search_dir / "top_10_results.csv"
    print(f"Search run directory: {search_dir}")

    fieldnames = keys + ["best_val_dice", "best_train_loss", "best_val_loss"]
    all_results = []

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combos)}] {params}")

        start = time.time()
        run_dir, best_val_loss, best_val_dice, best_train_loss, test_loss, test_dice = train_model(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            learning_rate=params["lr"],
            pos_weight_value=params["pos_weight"],
            batch_size=params["batch_size"],
            num_epochs=args.epochs,
            experiments_dir=trials_dir,
            use_augmentation=params["use_augmentation"],
            optimizer_name=params["optimizer"],
            verbose=False,
            modality=params["modality"],
            save_history=False,
            save_visualizations=False,
        )
        elapsed = time.time() - start

        row = {
            **params,
            "best_val_dice": best_val_dice,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
        }
        all_results.append(row)
        write_csv(results_file, fieldnames, all_results)

        print(f"  -> val dice: {best_val_dice:.4f} | train loss: {best_train_loss:.4f} | val loss: {best_val_loss:.4f} | {elapsed:.1f}s")

        top_results = sorted(all_results, key=lambda r: r["best_val_dice"], reverse=True)[:args.top_n]
        write_csv(top_n_file, fieldnames, top_results)

    print("\nGrid search complete.")
    print(f"All results saved to: {results_file}")
    print(f"Top {args.top_n} results saved to: {top_n_file}")
    print(f"Best combination: {top_results[0]}")


if __name__ == "__main__":
    main()
