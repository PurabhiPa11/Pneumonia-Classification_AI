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

The dataset and trained models are not included. Obtain an authorized chest X-ray dataset ZIP containing NORMAL and PNEUMONIA folders. Preparation expects filenames matching the patient-ID patterns in prepare_data.py. Dataset source: [Chest X-Ray Images (Pneumonia), Paul Mooney on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Download the dataset from Kaggle and provide the ZIP path to the preparation command.

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

The supplementary archives include saved results, training histories, and selected model checkpoints. Retraining still requires downloading the dataset. Model scores are not a medical diagnosis.


## Saved results and runnable checkpoints

Download the two ZIP files from this repository and extract both into the repository root (alongside `train.py`). They preserve the `results/` and `runs/` paths used by the scripts:

- **results-and-cnn.zip**: final graphs, aggregate evaluation reports, threshold comparison table, training histories, and the lower-learning-rate CNN checkpoint used by the threshold evaluation scripts. Per-image validation scores and dataset images are omitted.
- **prediction-model.zip**: the final MobileNetV2 checkpoint and a demonstration model selector.

After extracting the prediction archive and installing dependencies:

```sh
python predict.py --run runs/demo
```

Enter a local image path when prompted. The default decision threshold is 0.5. To reproduce the CNN threshold demonstration:

```sh
python predict.py --model runs/cnn_lr_0001/custom_cnn/best.keras --threshold 0.8610543608665466
```

The original final run selected **ResNet50 by validation ROC AUC**. Its 203.7 MiB checkpoint is not included. The MobileNetV2 demonstration selector is separate and does not change that original selection. Training from scratch regenerates `runs/final/best_model.json`.

### Recorded final-run test results

| Model | Accuracy | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 68.45% | 0.9129 | 0.9528 | 0.5220 | 0.6745 |
| MobileNetV2 | 88.51% | 0.9725 | 0.8511 | 0.9897 | 0.9152 |
| ResNet50 | 84.79% | 0.9654 | 0.8084 | 0.9922 | 0.8910 |

These are saved project results from `runs/final/comparison.json`, not newly rerun measurements. The lower-learning-rate CNN threshold reports describe a separate experiment. Its validation-selected threshold achieved 77.35% test accuracy; the report notes that the test set had previously been inspected during project development.

The dataset preparation retained 5,808 images locally after filtering. Dataset images, the virtual environment, duplicate checkpoints, and individual image score records are not published here. The checkpoints were copied from the existing project; inference has not been rerun as part of this upload.
