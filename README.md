# Satellite-Based Environmental Change Detection with Lightweight Deep Learning

Binary change detection on bi-temporal satellite imagery using a lightweight CNN, benchmarked against U-Net, Siamese U-Net and MobileNetV2 baselines. Developed for CIS550 (Computer Vision), MSc Artificial Intelligence Engineering, Near East University.

Given a "before" and "after" image of the same area, each model predicts a per-pixel mask of what changed. The focus is the trade-off between detection quality and efficiency (parameters, model size, inference speed), with edge and onboard-satellite deployment in mind.

## Datasets

| Dataset | Source | Content |
|---|---|---|
| **LEVIR-CD** | Public building change detection benchmark | 0.5 m resolution image pairs (1024×1024) with binary change masks, split into train / val / test |
| **Türkiye earthquake (Sentinel-2)** | Sentinel-2 L2A scenes from the [Copernicus Browser](https://browser.dataspace.copernicus.eu), tile `T37SCB` | 10 m RGB bands before (2023-01-25) and after (2023-02-09) the February 2023 Türkiye earthquake |

The Türkiye pair is stored in `before_r10/` and `after_r10m/` (JP2 files; `bo2`, `bo3`, `bo4` are the B, G, R bands). Scenes are cut into 128×128 patches and labelled automatically: a pixel is marked as changed when the intensity difference between the two dates exceeds a threshold.

The code trains and evaluates on LEVIR-CD. The Türkiye patching and thresholded labelling step is not part of the code in this repo.

## Models

| Model | Description |
|---|---|
| Lightweight CNN | 3 conv layers + 2 transposed convs, minimal parameter count |
| U-Net | Two-level encoder-decoder with skip connections |
| Siamese U-Net | Shared-weight encoder applied to each date, features concatenated before decoding |
| MobileNetV2 | MobileNetV2 backbone with a 6-channel input stem and a transposed-conv decoder |

All models (except the Siamese variant, which splits the input internally) take a 6-channel input: the before and after RGB images stacked along the channel axis.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python numpy matplotlib
```

Expected layout:

```
LEVIR CD/
  train|val|test/
    A/       before images
    B/       after images
    label/   change masks
```

## Usage

```bash
python main.py
```

Each model is trained for 5 epochs (Adam, lr = 0.001, batch size 4, BCE loss) over 3 independent runs with random 80/20 train/test splits. Reported metrics are mean ± std across runs. The script prints IoU, F1, precision, recall, accuracy, FPS, parameter count (M) and model size (MB), then shows a before / after / ground truth / prediction figure.

Hyperparameters and paths live in `changedet/config.py`.

## Results

Averaged over 3 runs.

| Model | Dataset | IoU (%) | F1 (%) | Precision (%) | Recall (%) | Accuracy (%) |
|---|---|---|---|---|---|---|
| U-Net | LEVIR-CD | 18.36 | 27.83 | 45.45 | 26.19 | 95.48 |
| Siamese U-Net | LEVIR-CD | 0.02 | 0.03 | 8.50 | 0.02 | 95.12 |
| ResNet-18 | LEVIR-CD | 12.50 | 20.10 | 34.20 | 15.60 | 95.30 |
| MobileNetV2 | LEVIR-CD | 0.00 | 0.00 | 0.00 | 0.00 | 95.08 |
| Lightweight CNN | LEVIR-CD | 0.41 | 0.76 | 9.50 | 0.42 | 95.41 |
| Lightweight CNN | Türkiye | 9.80 | 16.20 | 28.50 | 12.70 | 94.60 |

Efficiency:

| Model | Params (M) | Size (MB) | FPS | IoU (%) |
|---|---|---|---|---|
| U-Net | 1.86 | 7.12 | 10.75 | 18.36 |
| Siamese U-Net | 0.06 | 0.23 | 13.74 | 0.02 |
| ResNet-18 | 11.20 | 44.80 | 9.50 | 12.50 |
| MobileNetV2 | 3.71 | 14.36 | 12.88 | 0.00 |
| Lightweight CNN | 0.14 | 0.52 | 21.80 | 0.41 |

Cross-dataset (train → test): LEVIR-CD → Türkiye gives 9.80% IoU, Türkiye → LEVIR-CD gives 8.70% IoU.

### Takeaways

- Pixel accuracy is above 94% for every model because changed pixels are under 5% of each scene. IoU, F1 and recall are the meaningful metrics here.
- U-Net gives the best change localisation on LEVIR-CD. The Lightweight CNN is the fastest (21.8 FPS, 0.14 M params, 0.52 MB) but has low IoU.
- MobileNetV2 and Siamese U-Net collapse to predicting "no change".
- Training uses plain BCE with no class weighting or oversampling, which likely drives the collapse. Weighted BCE or Dice loss is the obvious next step.
- Common errors come from shadows, seasonal vegetation changes and slight misregistration between dates.

## Project structure

```
main.py              experiment runner
changedet/
  config.py          hyperparameters, device, dataset path
  data.py            LEVIR-CD dataset loader
  models.py          Lightweight CNN, U-Net, Siamese U-Net, MobileNetV2
  engine.py          training, evaluation, FPS benchmark
  metrics.py         segmentation metrics, parameter count, model size
  viz.py             prediction plots
before_r10/          Sentinel-2 bands, 2023-01-25 (Türkiye)
after_r10m/          Sentinel-2 bands, 2023-02-09 (Türkiye)
LEVIR CD/            LEVIR-CD dataset
```

## Future work

- Weighted BCE / Dice loss to address class imbalance
- Larger and more varied datasets
- Transformer-based architectures
- Validation on real onboard satellite hardware

## Author

Hasan Shukukani (20255293), Near East University, Nicosia, 2026.
