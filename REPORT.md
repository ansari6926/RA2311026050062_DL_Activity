# Plant Seedling Species Classification

## Project aim

This project studies image classification for 12 seedling species in the Plant Seedlings Classification dataset. The practical motivation is to help distinguish weeds from crops early enough to support targeted weed management. A model prediction is not, by itself, safe authorization to spray a plant.

## Data and split

The input is the labeled training directory from the Kaggle dataset, arranged in one folder per species. The training program creates a reproducible stratified 70/15/15 train, validation, and test split with seed 42. Images are resized to 224 by 224 and normalized with ImageNet channel statistics. Training images receive random crop, flip, rotation, and color transformations; validation and test images do not.

## Models and training plan

The primary model is an ImageNet-pretrained ResNet-34. The final residual block and a newly initialized classifier are trained while earlier layers remain frozen. A four-block convolutional network trained from scratch provides a baseline. Both models use cross-entropy with label smoothing, AdamW, and cosine learning-rate decay. The checkpoint with the best validation macro F1 is selected before a single test-set evaluation.

## Evaluation

Accuracy and macro F1 are reported, along with per-class precision, recall, F1, a confusion matrix, and learning curves. Macro F1 averages class-level F1 scores equally, which helps reveal weak performance on less frequent species. The test split is reserved for the final evaluation and should not be used to tune the models.

## Results

Results have not yet been generated for this repository. Run the commands in the README and record the values from each model's `results.json` and `classification_report.json` here. Do not copy results from another person's run: outcomes depend on the data, training duration, hardware, and software environment.

| Model | Test accuracy | Test macro F1 | Notes |
|---|---:|---:|---|
| ResNet-34 fine-tuned | Pending run | Pending run | Select by validation macro F1 |
| Small CNN from scratch | Pending run | Pending run | Baseline |

## Limitations and responsible use

This dataset comes from a limited image collection and does not establish performance under different cameras, field conditions, lighting, soil, or growth stages. Confusing a crop and a weed can cause financial and environmental harm. Any field system would need additional data, confidence calibration, field evaluation, and human oversight before use.

## References

- Aarhus University, [Plant Seedlings Classification dataset](https://www.kaggle.com/c/plant-seedlings-classification).
- Giselsson et al., [A Public Image Database for Benchmark of Plant Seedling Classification Algorithms](https://arxiv.org/abs/1711.05458), 2017.
