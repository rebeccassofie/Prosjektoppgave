import itertools
import time
from train_unet import train_model

# Hyperparameter grid to try. Keep this small -- each combination trains
# a full model, so total run time = (number of combinations) * search_epochs.
learning_rates = [1e-4, 5e-4, 1e-3]
pos_weights = [3.0, 4.65, 7.0]
optimizer_names = ["adamw", "sgd"]
batch_sizes = [8, 16]

search_epochs = 30  # fewer epochs per combo, just to compare and find the best setting

combinations = list(itertools.product(learning_rates, pos_weights, optimizer_names, batch_sizes))
print(f"Grid search: {len(combinations)} combinations, {search_epochs} epochs each "
      f"= {len(combinations) * search_epochs} total epochs.")

results = []
start_time = time.time()

for i, (lr, pw, opt_name, bs) in enumerate(combinations, 1):
    print(f"\n=== [{i}/{len(combinations)}] lr={lr}, pos_weight={pw}, "
          f"optimizer={opt_name}, batch_size={bs} ===")

    val_loss, val_dice, test_loss, test_dice = train_model(
        learning_rate=lr,
        pos_weight_value=pw,
        optimizer_name=opt_name,
        batch_size=bs,
        num_epochs=search_epochs,
        checkpoint_path="models/grid_search_temp.pth",
        save_history=False,
        verbose=False,
    )

    print(f"  -> val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")
    # Selection is based on val_dice only -- test set stays untouched
    # until the very end, so it isn't tuned to during hyperparameter search.
    results.append((lr, pw, opt_name, bs, val_loss, val_dice))

elapsed = time.time() - start_time
print(f"\nGrid search finished in {elapsed / 60:.1f} minutes.")

# Sort by best (highest) val dice
results.sort(key=lambda r: r[5], reverse=True)

# Save full results to CSV for later reference
with open("output/grid_search_results.csv", "w") as f:
    f.write("learning_rate,pos_weight,optimizer,batch_size,val_loss,val_dice\n")
    for lr, pw, opt_name, bs, val_loss, val_dice in results:
        f.write(f"{lr},{pw},{opt_name},{bs},{val_loss},{val_dice}\n")
print("Saved full results to output/grid_search_results.csv")

print("\n=== Top 10 results (sorted by val dice) ===")
print(f"{'lr':>10} | {'pos_weight':>10} | {'optimizer':>10} | {'batch':>6} | "
      f"{'val_loss':>10} | {'val_dice':>10}")
for lr, pw, opt_name, bs, val_loss, val_dice in results[:10]:
    print(f"{lr:>10} | {pw:>10} | {opt_name:>10} | {bs:>6} | "
          f"{val_loss:>10.4f} | {val_dice:>10.4f}")

best_lr, best_pw, best_opt, best_bs, best_loss, best_dice = results[0]
print(f"\nBest combo: learning_rate={best_lr}, pos_weight={best_pw}, "
      f"optimizer={best_opt}, batch_size={best_bs} (val_dice={best_dice:.4f})")
print("Update train_unet.py's train_model() defaults with these values, "
      "then run a full training with more epochs.")