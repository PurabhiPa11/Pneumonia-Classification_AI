import csv
from pathlib import Path

import matplotlib.pyplot as plt


history_path = Path(
    "runs/cnn_lr_0001/custom_cnn/head_history.csv"
)

if not history_path.is_file():
    raise SystemExit(
        f"Training history not found: {history_path}\n"
        "Run this script from your main project folder."
    )

with history_path.open(
    "r", newline="", encoding="utf-8-sig"
) as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise SystemExit("The training history is empty.")

# The log starts at epoch 0; display epochs starting at 1.
epochs = [int(row["epoch"]) + 1 for row in rows]


def values(column):
    return [float(row[column]) for row in rows]


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Accuracy
axes[0, 0].plot(
    epochs, values("accuracy"), "o-", label="Training"
)
axes[0, 0].plot(
    epochs, values("val_accuracy"), "o-", label="Validation"
)
axes[0, 0].set_title("CNN: Accuracy")
axes[0, 0].set_ylabel("Accuracy (0–1)")
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].legend()

# Loss
axes[0, 1].plot(
    epochs, values("loss"), "o-", label="Training"
)
axes[0, 1].plot(
    epochs, values("val_loss"), "o-", label="Validation"
)
axes[0, 1].set_title("CNN: Binary Cross-Entropy Loss")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()

# ROC AUC
axes[1, 0].plot(
    epochs, values("auc"), "o-", label="Training"
)
axes[1, 0].plot(
    epochs, values("val_auc"), "o-", label="Validation"
)
axes[1, 0].set_title("CNN: ROC AUC")
axes[1, 0].set_ylabel("ROC AUC")
axes[1, 0].set_ylim(0, 1.05)
axes[1, 0].legend()

# Learning rate recorded in the history
axes[1, 1].step(
    epochs,
    values("learning_rate"),
    where="post",
    marker="o",
    color="purple",
)
axes[1, 1].set_title("CNN: Logged Learning Rate")
axes[1, 1].set_ylabel("Learning rate")
axes[1, 1].ticklabel_format(
    axis="y", style="sci", scilimits=(0, 0)
)

for ax in axes.flat:
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

fig.suptitle("Custom CNN — Training History", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95))

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

output_path = output_folder / "cnn_lr_0001_graphs.png"
fig.savefig(output_path, dpi=200)

best_index = max(
    range(len(rows)),
    key=lambda i: float(rows[i]["val_auc"]),
)

print(f"Completed epochs: {len(rows)}")
print(f"Best validation AUC epoch: {epochs[best_index]}")
print(
    "Best logged validation AUC: "
    f"{float(rows[best_index]['val_auc']):.4f}"
)
print(f"Graphs saved to: {output_path.resolve()}")

plt.show()