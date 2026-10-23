# Plant Seedling Classifier

A reproducible deep learning project for classifying the 12 plant species in the Plant Seedlings Classification dataset. It trains an ImageNet-pretrained ResNet-34 and a small CNN baseline, selects checkpoints using validation macro F1, and evaluates each selected model once on a held-out test set.

The interactive run guide is in [`notebooks/plant_seedling_classification.ipynb`](notebooks/plant_seedling_classification.ipynb). The project write-up is in [`REPORT.md`](REPORT.md); its results section is intentionally pending until you run the models.

## Dataset

Download the Plant Seedlings Classification dataset from Kaggle. Arrange the labeled training images as:

```text
data/plant-seedlings-classification/train/<species-name>/*.png
```

The script makes a deterministic stratified 70/15/15 train/validation/test split. Test images are not used for model selection. Dataset files are excluded from Git.

## Setup

Use Python 3.10 or newer. Install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data-dir data/plant-seedlings-classification/train --model resnet34 --epochs 12
python src/train.py --data-dir data/plant-seedlings-classification/train --model small-cnn --epochs 20
```

For a quick CPU run, pass `--max-train-batches 35`. This option is intended for experimentation and can reduce model quality. To use a GPU, install the PyTorch build appropriate for your CUDA setup, then run the same command.

Outputs are written under `runs/<model>-<timestamp>/`: configuration, best validation checkpoint, per-class test metrics, confusion matrix, and learning curves. Accuracy and macro F1 are reported; macro F1 is useful when class sizes differ.

## Method

Images are resized/cropped to 224x224, normalized using ImageNet statistics, and augmented during training with random crops, flips, rotation, and mild color jitter. The ResNet-34 uses pretrained weights and fine-tunes its final residual block and classifier. The small CNN trains from scratch as a baseline. Both use cross-entropy with label smoothing, AdamW, and cosine learning-rate scheduling. The best validation macro-F1 checkpoint is evaluated on the test split.

## Limitations

Results depend on the downloaded dataset, hardware, software versions, and training duration. A good dataset score does not establish performance in a real field. Images from other locations, cameras, lighting, soils, and growth stages should be collected and evaluated before considering deployment. A mistaken prediction could leave a weed untreated or harm a crop.
