import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from train import make_dataset


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


def show_results(title, metrics):
    print(f"\n{title}")
    print(f"Threshold: {metrics['threshold']:.10f}")
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
        "Normal images incorrectly called pneumonia: "
        f"{metrics['false_positives']}"
    )
    print(
        "Pneumonia images missed: "
        f"{metrics['false_negatives']}"
    )


def main():
    report_path = Path(
        "results/cnn_threshold_check/threshold_report.json"
    )
    data_folder = Path("data")
    manifest_path = data_folder / "manifest.json"

    for path in (report_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if report.get("data_split") != "validation":
        raise SystemExit(
            "The threshold must have been selected on validation data."
        )

    # Load the exact threshold already selected on validation.
    # This script does not search for another threshold.
    threshold = float(report["candidate"]["threshold"])
    model_path = Path(report["model_path"])

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise SystemExit("Invalid saved threshold.")

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise SystemExit("Unexpected class order.")

    records = [
        record
        for record in manifest["records"]
        if record["split"] == "test"
    ]

    actual = np.array(
        [record["label"] for record in records],
        dtype=int,
    )

    if set(actual.tolist()) != {0, 1}:
        raise SystemExit(
            "Test data must contain both classes."
        )

    dataset = make_dataset(
        records,
        data_folder,
        batch_size=16,
        training=False,
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print(f"Loaded model: {model_path}")
    print(f"Evaluating {len(records)} test images...")

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
        raise SystemExit("Invalid model scores.")

    baseline = calculate_metrics(actual, scores, 0.5)
    selected = calculate_metrics(actual, scores, threshold)

    show_results("TEST — ORIGINAL THRESHOLD", baseline)
    show_results(
        "TEST — FIXED VALIDATION-SELECTED THRESHOLD",
        selected,
    )

    auc = float(roc_auc_score(actual, scores))
    print(f"\nTest ROC AUC: {auc:.4f}")
    print("ROC AUC is unchanged by changing the decision threshold.")

    output_path = report_path.parent / "test_threshold_results.json"

    results = {
        "model_path": str(model_path),
        "data_split": "test",
        "test_images": len(records),
        "threshold_source": str(report_path),
        "test_roc_auc": auc,
        "baseline": baseline,
        "validation_selected_threshold": selected,
        "note": (
            "Threshold fixed using validation data before this "
            "evaluation. No test-based threshold search performed. "
            "This test set has previously been inspected during "
            "project development."
        ),
    }

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path.resolve()}")
    print("Your model and predict.py were not changed.")


if __name__ == "__main__":
    main()