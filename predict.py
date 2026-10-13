import argparse
import json
from pathlib import Path

import tensorflow as tf


def predict_image(model, image_path, threshold):
    # Accept paths pasted with or without quotation marks.
    image_path = Path(
        image_path.strip().strip('"').strip("'")
    )

    if not image_path.is_file():
        raise ValueError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in (
        ".jpg", ".jpeg", ".png", ".bmp"
    ):
        raise ValueError("Use a JPG, JPEG, PNG, or BMP image.")

    image = tf.io.decode_image(
        tf.io.read_file(str(image_path)),
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        tf.cast(image, tf.float32),
        model.input_shape[1:3],
    )

    # Preprocessing is already included in the saved model.
    image = tf.expand_dims(image, axis=0)

    pneumonia_score = float(
        model(image, training=False).numpy()[0, 0]
    )

    if not 0 <= pneumonia_score <= 1:
        raise ValueError("The model returned an invalid score.")

    normal_score = 1 - pneumonia_score

    prediction = (
        "PNEUMONIA"
        if pneumonia_score >= threshold
        else "NORMAL"
    )

    print("\n--------------------------------")
    print(f"Prediction: {prediction}")
    print(f"Pneumonia model score: {pneumonia_score * 100:.2f}%")
    print(f"Normal model score:    {normal_score * 100:.2f}%")
    print(f"Decision threshold:   {threshold * 100:.4f}%")
    print("Prediction uses the decision threshold shown above.")
    print("These scores are not a medical diagnosis.")
    print("--------------------------------\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        default="runs/final",
        help="Folder containing training results",
    )

    parser.add_argument(
        "--model",
        help="Optional path to a specific .keras model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold between 0 and 1; default: 0.5",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    if args.model:
        model_path = Path(args.model)
    else:
        run_folder = Path(args.run)
        selection_path = run_folder / "best_model.json"

        if not selection_path.is_file():
            parser.error(
                "Trained model selection not found. "
                "Finish training first."
            )

        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )

        model_path = run_folder / selection["model_path"]

    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print(f"\nLoaded model: {model_path}")
    print(f"Decision threshold: {args.threshold:.10f}")

    while True:
        try:
            image_path = input(
                "Enter X-ray image path (or type quit): "
            ).strip()

            if image_path.lower() in ("quit", "exit", "q"):
                break

            predict_image(model, image_path, args.threshold)

        except (ValueError, tf.errors.OpError) as error:
            print(f"\nError: {error}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nPrediction closed.")
            break


if __name__ == "__main__":
    main()