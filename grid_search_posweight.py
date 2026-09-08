import numpy as np
from train_unet import train_model

# Search pos_weight in [2.0, 4.0], everything else fixed
pos_weights = np.linspace(0.0, 2.1, 20)

learning_rate = 0.0005
batch_size = 8
optimizer_name = "adamw"
search_epochs = 60

results = []

for pw in pos_weights:
    print(f"\n=== pos_weight={pw:.3f} ===")
    val_loss, val_dice, test_loss, test_dice = train_model(
        learning_rate=learning_rate,
        pos_weight_value=pw,
        optimizer_name=optimizer_name,
        batch_size=batch_size,
        num_epochs=search_epochs,
        checkpoint_path="models/grid_search_temp.pth",
        save_history=False,
        verbose=False,
    )
    print(f"  -> val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")
    results.append((pw, val_loss, val_dice))

# Sort by best (highest) val dice
results.sort(key=lambda r: r[2], reverse=True)

print("\n=== Results (sorted by val dice) ===")
print(f"{'pos_weight':>12} | {'val_loss':>10} | {'val_dice':>10}")
for pw, val_loss, val_dice in results:
    print(f"{pw:>12.3f} | {val_loss:>10.4f} | {val_dice:>10.4f}")

best_pw, best_loss, best_dice = results[0]
print(f"\nBest pos_weight: {best_pw:.3f} (val_dice={best_dice:.4f})")