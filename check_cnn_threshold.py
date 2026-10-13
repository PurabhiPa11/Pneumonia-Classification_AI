import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from train import make_dataset


DATA_FOLDER = Path("data")
MODEL_PATH = Path(
    "runs/cnn_lr_0001/custom_cnn/best.keras"
)
OUTPUT_FOLDER = Path("results/cnn_threshold_check")


def calculate_metrics(actual, scores, threshold):
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"Threshold: {metrics['threshold']:.8f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(
        "Pneumonia recall: "
        f"{metrics['recall'] * 100:.2f}%"
    )
    print(
        "Normal specificity: "
        f"{metrics['specificity'] * 100:.2f}%"
    )
    print(
        "Normal images called pneumonia: "
        f"{metrics['false_positives']}"
    )
    print(
        "Pneumonia images missed: "
        f"{metrics['false_negatives']}"
    )


def main():
    manifest_path = DATA_FOLDER / "manifest.json"

    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    if not manifest_path.is_file():
        raise SystemExit(
            f"Dataset manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise SystemExit("Unexpected class order in manifest.")

    # Use validation images only, in their manifest order.
    records = [
        record
        for record in manifest["records"]
        if record["split"] == "val"
    ]

    actual = np.array(
        [record["label"] for record in records],
        dtype=int,
    )

    if set(actual.tolist()) != {0, 1}:
        raise SystemExit(
            "Validation data must contain both classes."
        )

    dataset = make_dataset(
        records,
        DATA_FOLDER,
        batch_size=16,
        training=False,
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Checking {len(records)} validation images...")

    scores = model.predict(
        dataset, verbose=1
    ).reshape(-1).astype(np.float64)

    if len(scores) != len(actual):
        raise SystemExit("Prediction count does not match labels.")

    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        raise SystemExit("Invalid model scores detected.")

    # Examine all score-based decision boundaries.
    # ROC thresholds use the same >= rule as predict.py.
    fpr, tpr, thresholds = roc_curve(
        actual,
        scores,
        drop_intermediate=False,
    )

    balanced_scores = (tpr + (1 - fpr)) / 2

    # Include the existing 0.5 threshold explicitly.
    baseline = calculate_metrics(actual, scores, 0.5)

    thresholds = np.append(thresholds, 0.5)
    balanced_scores = np.append(
        balanced_scores,
        baseline["balanced_accuracy"],
    )

    # Restrict candidate cutoffs to valid score thresholds.
    valid = (
        np.isfinite(thresholds)
        & (thresholds >= 0)
        & (thresholds <= 1)
    )

    thresholds = thresholds[valid]
    balanced_scores = balanced_scores[valid]

    best_value = np.max(balanced_scores)

    tied_indices = np.flatnonzero(
        np.isclose(
            balanced_scores,
            best_value,
            rtol=0,
            atol=1e-12,
        )
    )

    # If tied, prefer the threshold closest to 0.5.
    best_index = tied_indices[
        np.argmin(np.abs(thresholds[tied_indices] - 0.5))
    ]

    chosen_threshold = float(thresholds[best_index])

    candidate = calculate_metrics(
        actual,
        scores,
        chosen_threshold,
    )

    auc = float(roc_auc_score(actual, scores))

    print(f"\nValidation ROC AUC: {auc:.4f}")
    print_metrics("CURRENT THRESHOLD — VALIDATION", baseline)
    print_metrics("CANDIDATE THRESHOLD — VALIDATION", candidate)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    report = {
        "model_path": str(MODEL_PATH),
        "data_split": "validation",
        "validation_images": len(records),
        "selection_metric": "maximum validation balanced accuracy",
        "validation_roc_auc": auc,
        "baseline": baseline,
        "candidate": candidate,
        "automatically_applied": False,
        "note": (
            "The threshold was selected and measured on the same "
            "validation set. These are tuning results, not an "
            "independent estimate of future performance. "
            "No test images were used."
        ),
    }

    (OUTPUT_FOLDER / "threshold_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with (
        OUTPUT_FOLDER / "validation_scores.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "actual_label", "pneumonia_score"])

        for record, score in zip(records, scores):
            writer.writerow([
                record["path"],
                manifest["classes"][record["label"]],
                float(score),
            ])

    # Use a regular grid to display threshold trade-offs.
    plot_thresholds = np.unique(
        np.append(np.linspace(0, 1, 201), chosen_threshold)
    )

    plot_metrics = [
        calculate_metrics(actual, scores, threshold)
        for threshold in plot_thresholds
    ]

    with (
        OUTPUT_FOLDER / "threshold_comparison.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(plot_metrics[0].keys()),
        )
        writer.writeheader()
        writer.writerows(plot_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 31)

    # Each class histogram sums to 100%, avoiding class-count bias.
    for label, name, color in [
        (0, "Actual NORMAL", "steelblue"),
        (1, "Actual PNEUMONIA", "orange"),
    ]:
        class_scores = scores[actual == label]

        axes[0].hist(
            class_scores,
            bins=bins,
            weights=np.full(
                len(class_scores),
                100.0 / len(class_scores),
            ),
            alpha=0.55,
            color=color,
            label=name,
        )

    axes[0].set_title("Validation Score Distributions")
    axes[0].set_xlabel("Pneumonia model score")
    axes[0].set_ylabel("Percentage of each class per bin")

    for key, label in [
        ("recall", "Pneumonia recall"),
        ("specificity", "Normal specificity"),
        ("balanced_accuracy", "Balanced accuracy"),
    ]:
        axes[1].plot(
            plot_thresholds,
            [row[key] for row in plot_metrics],
            label=label,
        )

    axes[1].set_title("Validation Threshold Trade-offs")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(-0.02, 1.02)

    for ax in axes:
        ax.axvline(
            0.5,
            color="black",
            linestyle="--",
            label="Current threshold: 0.5",
        )
        ax.axvline(
            chosen_threshold,
            color="red",
            linestyle=":",
            label=f"Candidate: {chosen_threshold:.4f}",
        )
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle("Custom CNN — Validation-Only Threshold Diagnostic")
    fig.tight_layout()

    graph_path = OUTPUT_FOLDER / "validation_threshold_graphs.png"
    fig.savefig(graph_path, dpi=200)

    print(f"\nResults saved to: {OUTPUT_FOLDER.resolve()}")
    print("No model or prediction settings were changed.")
    print(
        "Candidate results are validation tuning results, "
        "not guaranteed performance on new images."
    )

    plt.show()


if __name__ == "__main__":
    main()

    