import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


MODELS = [
    {
        "name": "Custom CNN",
        "key": "custom_cnn",
        "run": Path("runs/cnn_lr_0001"),
    },
    {
        "name": "MobileNetV2",
        "key": "mobilenetv2",
        "run": Path("runs/final"),
    },
    {
        "name": "ResNet50",
        "key": "resnet50",
        "run": Path("runs/final"),
    },
]


def read_history(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"History file not found: {path}"
        )

    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"History file is empty: {path}"
        )

    return rows


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_stages(model):
    folder = model["run"] / model["key"]

    head = read_history(
        folder / "head_history.csv"
    )

    stages = [
        {
            "name": "Initial training",
            "rows": head,
            "start": 0,
        }
    ]

    fine_path = folder / "fine_tuned_history.csv"

    if fine_path.is_file():
        stages.append(
            {
                "name": "Fine-tuning",
                "rows": read_history(fine_path),
                "start": len(head),
            }
        )

    return stages


def set_window_title(fig, title):
    fig.canvas.manager.set_window_title(title)


def draw_history(metric, ylabel, window_title):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5)
    )

    for ax, model in zip(axes, MODELS):
        stages = load_stages(model)

        for index, stage in enumerate(stages):
            rows = stage["rows"]

            # Display fine-tuning epochs after initial training.
            epochs = np.arange(
                stage["start"] + 1,
                stage["start"] + len(rows) + 1,
            )

            training = [
                float(row[metric])
                for row in rows
            ]

            validation = [
                float(row["val_" + metric])
                for row in rows
            ]

            style = "-" if index == 0 else "--"

            # Stages are plotted separately because fine-tuning
            # starts from the selected initial checkpoint.
            ax.plot(
                epochs,
                training,
                marker=".",
                linestyle=style,
                color="steelblue",
                label=f"Training — {stage['name']}",
            )

            ax.plot(
                epochs,
                validation,
                marker=".",
                linestyle=style,
                color="darkorange",
                label=f"Validation — {stage['name']}",
            )

        if len(stages) > 1:
            boundary = (
                len(stages[0]["rows"]) + 0.5
            )

            ax.axvline(
                boundary,
                color="gray",
                linestyle=":",
                label="Fine-tuning begins",
            )

        ax.set_title(model["name"])
        ax.set_xlabel(
            "Completed epoch (stages shown in order)"
        )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        if metric in ("accuracy", "auc"):
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Project Training History — {ylabel}",
        fontsize=15,
    )

    note = (
        "CNN: lower-learning-rate run. "
        "MobileNetV2 / ResNet50: original final run.\n"
        "Fine-tuning begins from the selected initial checkpoint."
    )

    if metric == "accuracy":
        note += (
            "\nLogged accuracy uses the original "
            "0.5 decision threshold."
        )

    fig.text(
        0.5,
        0.015,
        note,
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=(0, 0.13, 1, 0.93)
    )

    set_window_title(fig, window_title)


def draw_learning_rates():
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5)
    )

    for ax, model in zip(axes, MODELS):
        stages = load_stages(model)

        for stage in stages:
            rows = stage["rows"]

            epochs = np.arange(
                stage["start"] + 1,
                stage["start"] + len(rows) + 1,
            )

            rates = [
                float(row["learning_rate"])
                for row in rows
            ]

            ax.step(
                epochs,
                rates,
                where="post",
                marker=".",
                label=stage["name"],
            )

        if len(stages) > 1:
            ax.axvline(
                len(stages[0]["rows"]) + 0.5,
                color="gray",
                linestyle=":",
                label="Fine-tuning begins",
            )

        ax.set_title(model["name"])
        ax.set_xlabel(
            "Completed epoch (stages shown in order)"
        )
        ax.set_ylabel(
            "Logged learning rate — logarithmic scale"
        )
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Project Learning-Rate History",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.93)
    )

    set_window_title(
        fig,
        "04 — Learning Rates",
    )


def draw_test_comparison():
    labels = []
    metrics = []

    for model in MODELS:
        comparison = read_json(
            model["run"] / "comparison.json"
        )

        test = comparison[model["key"]]["test"]

        labels.append(
            f"{model['name']}\nthreshold 0.5"
        )

        metrics.append(
            {
                "accuracy": float(test["accuracy"]),
                "recall": float(test["recall"]),
                "precision": float(test["precision"]),
            }
        )

    # Show threshold-adjusted CNN separately.
    threshold_path = Path(
        "results/cnn_threshold_check/"
        "test_threshold_results.json"
    )

    if threshold_path.is_file():
        report = read_json(threshold_path)

        expected_model = (
            MODELS[0]["run"]
            / MODELS[0]["key"]
            / "best.keras"
        )

        if (
            Path(report["model_path"]).resolve()
            != expected_model.resolve()
        ):
            raise ValueError(
                "Threshold report refers to a different CNN model."
            )

        adjusted = report[
            "validation_selected_threshold"
        ]

        threshold = float(
            adjusted["threshold"]
        )

        labels.append(
            f"Custom CNN\nthreshold {threshold:.4f}"
        )

        metrics.append(
            {
                "accuracy": float(adjusted["accuracy"]),
                "recall": float(adjusted["recall"]),
                "precision": float(adjusted["precision"]),
            }
        )

    else:
        print(
            "Threshold report not found; plotting only "
            "the three original-threshold configurations."
        )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    positions = np.arange(len(labels))
    width = 0.24

    for offset, key, label, color in [
        (
            -width,
            "accuracy",
            "Accuracy",
            "steelblue",
        ),
        (
            0,
            "recall",
            "Pneumonia recall",
            "darkorange",
        ),
        (
            width,
            "precision",
            "Pneumonia precision",
            "seagreen",
        ),
    ]:
        values = [
            row[key] * 100
            for row in metrics
        ]

        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=label,
            color=color,
        )

        ax.bar_label(
            bars,
            labels=[
                f"{value:.1f}"
                for value in values
            ],
            padding=3,
            fontsize=8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 115)

    ax.set_title(
        "Test Performance — Models and Decision Rules"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        ncol=3,
    )

    fig.text(
        0.5,
        0.015,
        "Adjusted CNN uses a validation-selected threshold. "
        "It is the same model, not an additional trained architecture.\n"
        "These test results have been inspected during development; "
        "they are not clinical validation.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=(0, 0.11, 1, 1)
    )

    set_window_title(
        fig,
        "05 — Test Performance Comparison",
    )


def draw_sigmoid():
    logits = np.linspace(-10, 10, 500)
    scores = 1 / (1 + np.exp(-logits))

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        logits,
        scores,
        color="steelblue",
        linewidth=2,
        label="Sigmoid: 1 / (1 + exp(-z))",
    )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        label="Original decision threshold: 0.5",
    )

    threshold_path = Path(
        "results/cnn_threshold_check/"
        "threshold_report.json"
    )

    if threshold_path.is_file():
        report = read_json(threshold_path)

        threshold = float(
            report["candidate"]["threshold"]
        )

        ax.axhline(
            threshold,
            color="red",
            linestyle=":",
            label=(
                "Updated CNN threshold: "
                f"{threshold:.4f}"
            ),
        )

    ax.axvline(
        0,
        color="gray",
        linestyle=":",
        alpha=0.5,
    )

    ax.set(
        title=(
            "Shared Sigmoid Output Function — "
            "All Three Models"
        ),
        xlabel="Raw output before sigmoid (logit)",
        ylabel="Pneumonia model score",
        ylim=(-0.05, 1.05),
    )

    ax.grid(alpha=0.25)
    ax.legend(
        loc="lower right",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.02,
        "All three models use sigmoid, but produce different "
        "input values to this function.\n"
        "Illustration only: not a measured accuracy graph, "
        "a calibrated clinical probability, or an ensemble.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.12, 1, 1)
    )

    set_window_title(
        fig,
        "06 — Shared Sigmoid Function",
    )


def main():
    draw_history(
        "accuracy",
        "Accuracy",
        "01 — Training and Validation Accuracy",
    )

    draw_history(
        "loss",
        "Binary Cross-Entropy Loss",
        "02 — Training and Validation Loss",
    )

    draw_history(
        "auc",
        "ROC AUC",
        "03 — Training and Validation AUC",
    )

    draw_learning_rates()
    draw_test_comparison()
    draw_sigmoid()

    print("\nOpening all six graph windows.")
    print("Some windows may appear behind others.")
    print("Use a window's Save icon if you want to save its graph.")
    print("Close all graph windows to finish.")
    print("No models were retrained or modified.")

    plt.show()


if __name__ == "__main__":
    main()