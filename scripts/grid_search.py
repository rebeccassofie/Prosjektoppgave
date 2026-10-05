import argparse
import csv
import itertools
import time

from src.experiment_utils import create_run_dir
from src.train import train_model

PARAM_GRID = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
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
    parser.add_argument("--pos-weight", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", default="rmsprop")
    parser.add_argument("--modality", default="adp")
    parser.add_argument("--metric", default="boundary_f1", choices=["boundary_f1", "dice"])
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    args = parser.parse_args()

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Running grid search over {len(combos)} combinations, {args.epochs} epochs each.")

    search_dir = create_run_dir(args.experiments_dir)
    trials_dir = search_dir / "trials"
    results_file = search_dir / "grid_search_results.csv"
    top_n_file = search_dir / f"top_{args.top_n}_results.csv"
    print(f"Search run directory: {search_dir}")

    score_key = f"best_val_{args.metric}"
    fieldnames = keys + [score_key, "best_train_loss", "best_val_loss"]
    all_results = []

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combos)}] {params}")

        start = time.time()
        run_dir, best_val_loss, best_val_score, best_train_loss, test_loss, test_dice = train_model(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            learning_rate=params["lr"],
            pos_weight_value=args.pos_weight,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            experiments_dir=trials_dir,
            use_augmentation=True,
            optimizer_name=args.optimizer,
            verbose=False,
            modality=args.modality,
            metric=args.metric,
            tolerance=args.tolerance,
            save_history=False,
            save_visualizations=False,
            momentum=args.momentum,
            beta2=args.beta2,
        )
        elapsed = time.time() - start

        row = {
            **params,
            score_key: best_val_score,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
        }
        all_results.append(row)
        write_csv(results_file, fieldnames, all_results)

        print(f"  -> val {args.metric}: {best_val_score:.4f} | train loss: {best_train_loss:.4f} | val loss: {best_val_loss:.4f} | {elapsed:.1f}s")

        top_results = sorted(all_results, key=lambda r: r[score_key], reverse=True)[:args.top_n]
        write_csv(top_n_file, fieldnames, top_results)

    print("\nGrid search complete.")
    print(f"All results saved to: {results_file}")
    print(f"Top {args.top_n} results saved to: {top_n_file}")
    print(f"Best combination: {top_results[0]}")


if __name__ == "__main__":
    main()
