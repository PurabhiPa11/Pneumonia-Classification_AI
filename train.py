import argparse
import csv
import json

from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


IMAGE_SIZE = 224

MODEL_NAMES = [
    "custom_cnn",
    "mobilenetv2",
    "resnet50",
]


def make_dataset(records, data_folder, batch_size, training=False):
    paths = [
        str(data_folder / record["path"])
        for record in records
    ]

    labels = np.array(
        [record["label"] for record in records],
        dtype=np.float32,
    ).reshape(-1, 1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(records),
            seed=42,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            tf.cast(image, tf.float32),
            [IMAGE_SIZE, IMAGE_SIZE],
        )

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def build_model(name):
    layers = tf.keras.layers

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # These augmentations run only during training.
    x = layers.RandomRotation(0.025)(inputs)
    x = layers.RandomZoom(0.08)(x)

    backbone = None

    if name == "custom_cnn":
        x = layers.Rescaling(1.0 / 255)(x)

        for filters in [32, 64, 128, 256]:
            x = layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
            )(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)

    elif name == "mobilenetv2":
        x = layers.Rescaling(
            1.0 / 127.5,
            offset=-1,
        )(x)

        backbone = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )

    elif name == "resnet50":
        # RGB -> BGR, then subtract ImageNet means.
        x = tf.keras.ops.take(
            x,
            [2, 1, 0],
            axis=-1,
        )

        x = layers.Normalization(
            axis=-1,
            mean=[103.939, 116.779, 123.68],
            variance=[1.0, 1.0, 1.0],
        )(x)

        backbone = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    if backbone is not None:
        backbone.trainable = False

        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_score",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name=name,
    )

    return model, backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )


def make_callbacks(folder, stage):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(folder / f"{stage}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
        ),
        tf.keras.callbacks.CSVLogger(
            str(folder / f"{stage}_history.csv")
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def evaluate_model(model, dataset, records):
    probabilities = model.predict(
        dataset,
        verbose=0,
    ).reshape(-1)

    actual = np.array(
        [record["label"] for record in records]
    )

    predicted = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "roc_auc": float(
            roc_auc_score(actual, probabilities)
        ),
        "precision": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
    }


def main(args):
    if (
        args.epochs < 1
        or args.fine_tune_epochs < 0
        or args.batch_size < 1
    ):
        raise ValueError(
            "Check epochs, fine-tune epochs, and batch size."
        )

    data_folder = Path("data")
    output_folder = Path(args.output)

    manifest_path = data_folder / "manifest.json"

    if not manifest_path.is_file():
        raise ValueError(
            "Run prepare_data.py first."
        )

    if (
        output_folder.exists()
        and any(output_folder.iterdir())
    ):
        raise ValueError(
            "Output folder already contains results. "
            "Use another --output name."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    records = {
        split: [
            record
            for record in manifest["records"]
            if record["split"] == split
        ]
        for split in ["train", "val", "test"]
    }

    # Confirm no patient/hash overlap between splits.
    for field in ["patient", "sha256"]:
        groups = {
            split: {
                record[field]
                for record in records[split]
            }
            for split in records
        }

        for first, second in [
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ]:
            if groups[first] & groups[second]:
                raise ValueError(
                    f"{field} overlap: {first}/{second}"
                )

    datasets = {
        split: make_dataset(
            records[split],
            data_folder,
            args.batch_size,
            training=(split == "train"),
        )
        for split in records
    }

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = (
        MODEL_NAMES
        if args.model == "all"
        else [args.model]
    )

    results = {}

    for name in names:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        print(f"\nStarting training: {name}")

        folder = output_folder / name
        folder.mkdir()

        model, backbone = build_model(name)

        learning_rate = 0.0001 if name == "custom_cnn" else 0.001
        compile_model(model, learning_rate)

        model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=args.epochs,
            callbacks=make_callbacks(folder, "head"),
        )

        head_model = tf.keras.models.load_model(
            folder / "head.keras"
        )

        head_metrics = evaluate_model(
            head_model,
            datasets["val"],
            records["val"],
        )

        chosen_path = folder / "head.keras"

        if (
            backbone is not None
            and args.fine_tune_epochs > 0
        ):
            model.set_weights(
                head_model.get_weights()
            )

            backbone.trainable = True

            for layer in backbone.layers[:-30]:
                layer.trainable = False

            for layer in backbone.layers:
                if isinstance(
                    layer,
                    tf.keras.layers.BatchNormalization,
                ):
                    layer.trainable = False

            compile_model(model, 0.00001)

            print(f"\nFine-tuning: {name}")

            model.fit(
                datasets["train"],
                validation_data=datasets["val"],
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(
                    folder,
                    "fine_tuned",
                ),
            )

            tuned_model = tf.keras.models.load_model(
                folder / "fine_tuned.keras"
            )

            tuned_metrics = evaluate_model(
                tuned_model,
                datasets["val"],
                records["val"],
            )

            if (
                tuned_metrics["roc_auc"]
                > head_metrics["roc_auc"]
            ):
                chosen_path = (
                    folder / "fine_tuned.keras"
                )

            del tuned_model

        chosen_model = tf.keras.models.load_model(
            chosen_path
        )

        chosen_model.save(
            folder / "best.keras"
        )

        results[name] = {
            "validation": evaluate_model(
                chosen_model,
                datasets["val"],
                records["val"],
            )
        }

        del model, backbone, head_model, chosen_model

    # Select using validation results, not test results.
    best_name = max(
        results,
        key=lambda name: (
            results[name]["validation"]["roc_auc"]
        ),
    )

    selection = {
        "name": best_name,
        "model_path": f"{best_name}/best.keras",
        "selected_by": "validation ROC AUC",
    }

    (output_folder / "best_model.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    # Test evaluation happens after model selection.
    for name in names:
        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(
            output_folder / name / "best.keras"
        )

        results[name]["test"] = evaluate_model(
            model,
            datasets["test"],
            records["test"],
        )

        del model

    (output_folder / "comparison.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    with (
        output_folder / "comparison.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "validation_auc",
            "test_accuracy",
            "test_auc",
            "test_precision",
            "test_recall",
            "test_f1",
        ])

        for name, result in results.items():
            test = result["test"]

            writer.writerow([
                name,
                result["validation"]["roc_auc"],
                test["accuracy"],
                test["roc_auc"],
                test["precision"],
                test["recall"],
                test["f1"],
            ])

    print("\nTEST RESULTS")

    for name, result in results.items():
        test = result["test"]

        print(f"\n{name}")
        print(
            f"Accuracy: {test['accuracy'] * 100:.2f}%"
        )
        print(f"ROC AUC: {test['roc_auc']:.4f}")
        print(f"Precision: {test['precision']:.4f}")
        print(f"Recall: {test['recall']:.4f}")
        print(f"F1: {test['f1']:.4f}")
        print(
            "Confusion matrix:",
            test["confusion_matrix"],
        )

    print(
        f"\nBest model by validation AUC: {best_name}"
    )
    print(f"Results saved in: {output_folder.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["all"] + MODEL_NAMES,
        default="all",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output",
        default="runs/final",
    )

    main(parser.parse_args())