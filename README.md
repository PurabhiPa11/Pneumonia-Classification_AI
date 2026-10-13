# Pneumonia Classification AI

Python project for classifying chest X-ray images as NORMAL or PNEUMONIA using a custom CNN, MobileNetV2, and ResNet50.

## Setup

Create a Python virtual environment, activate it, and install dependencies:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies are unpinned; compatibility depends on the Python and TensorFlow versions used.

## Dataset and training

The dataset and trained models are not included. Obtain an authorized chest X-ray dataset ZIP containing NORMAL and PNEUMONIA folders. Preparation expects filenames matching the patient-ID patterns in prepare_data.py. The exact original dataset source should be documented by the project contributors.

```sh
python prepare_data.py --zip "path/to/dataset.zip" --output data
python train.py --model all --output runs/final
python predict.py --run runs/final
```

Preparation removes duplicate images and infers patient groups from filenames. Inferred IDs are not verified patient metadata. Training saves model checkpoints and evaluation outputs locally.

## Scripts

- `prepare_data.py`: prepares dataset images and a split manifest.
- `train.py`: trains and compares the three models.
- `predict.py`: interactively classifies an image using a trained model.
- `check_cnn_threshold.py`: analyzes classification thresholds on validation data.
- `evaluate_cnn_threshold.py`: evaluates the selected CNN threshold on test data.
- `plot_cnn_history.py`: plots CNN training history.
- `plot_project_history.py`: plots project training and comparison results.

Threshold and plotting scripts reference specific local run folders, including `runs/cnn_lr_0001` and `runs/final`. Check the paths near the start of those scripts and create the corresponding training outputs before running them.

This repository contains source code only; training and prediction require the dataset and locally generated model files. Model scores are not a medical diagnosis.
