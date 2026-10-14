# Satellite-Based Environmental Change Detection with Lightweight Deep Learning

Binary change detection on before/after satellite imagery. Given two images of the same area, each model predicts a per-pixel mask of what changed. The project compares a lightweight CNN against heavier baselines, with a focus on the trade-off between detection quality and efficiency (parameters, model size, inference speed) for edge and onboard-satellite deployment.

Developed for CIS550 (Computer Vision), M.Sc. Artificial Intelligence Engineering, Near East University.

## Highlights

- U-Net gave the best localisation on LEVIR-CD (**18.4% IoU**).
- The lightweight CNN is about **13x smaller** (0.14M vs 1.86M parameters, 0.52 MB) and about **2x faster** (21.8 vs 10.75 FPS) than U-Net, but its detection quality is low (0.41% IoU).
- Two models (MobileNetV2 and Siamese U-Net) collapsed to predicting "no change". Changed pixels are under 5% of each scene, and training used plain BCE with no class weighting.

## Datasets

| Dataset | Content |
| --- | --- |
| **LEVIR-CD** | 0.5 m resolution image pairs (1024x1024) of building changes with binary masks |
| **Turkiye earthquake (Sentinel-2)** | Sentinel-2 L2A scenes (tile `T37SCB`) from the Copernicus Browser, 10 m RGB bands, before (2023-01-25) and after (2023-02-09) the February 2023 earthquake |

The Turkiye scenes were cut into 128x128 patches and labelled automatically: a pixel counts as changed when the intensity difference between the two dates exceeds a threshold.

## Models

| Model | Description |
| --- | --- |
| Lightweight CNN | 3 conv layers + 2 transposed convs, minimal parameter count |
| U-Net | Two-level encoder-decoder with skip connections |
| Siamese U-Net | Shared-weight encoder applied to each date, features concatenated before decoding |
| MobileNetV2 | MobileNetV2 backbone with a 6-channel input stem and a transposed-conv decoder |
| ResNet-18 | ResNet-18 baseline (see the note under Results) |

Except for the Siamese variant, which splits the input internally, models take a 6-channel input: the before and after RGB images stacked along the channel axis.

## Training setup

Each model is trained for 5 epochs (Adam, learning rate 0.001, batch size 4, BCE loss) over 3 independent runs. Reported metrics are the mean across runs. Hyperparameters and paths are in `changedet/config.py`.

## Results

Averaged over 3 runs.

| Model | Dataset | IoU (%) | F1 (%) | Precision (%) | Recall (%) | Accuracy (%) |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net | LEVIR-CD | 18.36 | 27.83 | 45.45 | 26.19 | 95.48 |
| Siamese U-Net | LEVIR-CD | 0.02 | 0.03 | 8.50 | 0.02 | 95.12 |
| ResNet-18 | LEVIR-CD | 12.50 | 20.10 | 34.20 | 15.60 | 95.30 |
| MobileNetV2 | LEVIR-CD | 0.00 | 0.00 | 0.00 | 0.00 | 95.08 |
| Lightweight CNN | LEVIR-CD | 0.41 | 0.76 | 9.50 | 0.42 | 95.41 |
| Lightweight CNN | Turkiye | 9.80 | 16.20 | 28.50 | 12.70 | 94.60 |

Efficiency:

| Model | Params (M) | Size (MB) | FPS | IoU (%) |
| --- | --- | --- | --- | --- |
| U-Net | 1.86 | 7.12 | 10.75 | 18.36 |
| Siamese U-Net | 0.06 | 0.23 | 13.74 | 0.02 |
| ResNet-18 | 11.20 | 44.80 | 9.50 | 12.50 |
| MobileNetV2 | 3.71 | 14.36 | 12.88 | 0.00 |
| Lightweight CNN | 0.14 | 0.52 | 21.80 | 0.41 |

Cross-dataset (train -> test): LEVIR-CD -> Turkiye gives 9.80% IoU, Turkiye -> LEVIR-CD gives 8.70% IoU.

> **Note:** the ResNet-18 rows and the Turkiye results come from an earlier version of the code. The current `main.py` trains and evaluates only the four models in the Models table (without ResNet-18) on LEVIR-CD, and the Turkiye patching and labelling step is not included in this repository.

## Takeaways

- Pixel accuracy is above 94% for every model because changed pixels are under 5% of each scene, so IoU, F1 and recall are the meaningful metrics.
- U-Net localises change best. The lightweight CNN is the fastest and smallest but has low IoU.
- Plain BCE with no class weighting likely causes the collapse of MobileNetV2 and Siamese U-Net.
- Common errors come from shadows, seasonal vegetation changes and slight misregistration between the two dates.

## Setup and usage

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python numpy matplotlib
```

Expected dataset layout:

```
LEVIR CD/
  train|val|test/
    A/       before images
    B/       after images
    label/   change masks
```

```bash
python main.py
```

The script trains each model, prints IoU, F1, precision, recall, accuracy, FPS, parameter count and model size, and shows a before / after / ground truth / prediction figure.

## Project structure

```
main.py              experiment runner
changedet/
  config.py          hyperparameters, device, dataset path
  data.py            LEVIR-CD dataset loader
  models.py          model definitions
  engine.py          training, evaluation, FPS benchmark
  metrics.py         segmentation metrics, parameter count, model size
  viz.py             prediction plots
```

## Future work

- Weighted BCE or Dice loss to address class imbalance
- Larger and more varied datasets
- Transformer-based architectures
- Validation on real onboard satellite hardware

## Author

Hasan Shkoukani, Near East University, Nicosia, 2026.
