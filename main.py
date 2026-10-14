import numpy as np
from torch.utils.data import DataLoader, random_split

from changedet.config import BATCH_SIZE, DEVICE, RUNS
from changedet.data import LEVIRDataset
from changedet.engine import benchmark_fps, evaluate, train
from changedet.metrics import count_params, model_size
from changedet.models import LightweightCNN, MobileNetSeg, SiameseUNet, UNet
from changedet.viz import show_predictions


def run_model(name, model_fn, dataset):
    print(f"\n{'='*50}")
    print(name)
    print(f"{'='*50}")

    all_runs = []

    for run in range(RUNS):
        print(f"\nRun {run+1}/{RUNS}")

        n_train = int(0.8 * len(dataset))
        train_ds, test_ds = random_split(dataset, [n_train, len(dataset)-n_train])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = model_fn().to(DEVICE)

        train(model, train_loader)

        metrics = evaluate(model, test_loader)

        fps = benchmark_fps(model, test_loader)
        params = count_params(model)
        size = model_size(model)

        metrics["FPS"] = fps
        metrics["Params"] = params
        metrics["SizeMB"] = size

        all_runs.append(metrics)

    print("\nFINAL RESULTS")

    for k in all_runs[0].keys():
        vals = [r[k] for r in all_runs]
        print(f"{k:12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    show_predictions(model, dataset)


if __name__ == "__main__":

    dataset = LEVIRDataset()

    models_to_run = [
        ("Lightweight CNN", LightweightCNN),
        ("U-Net", UNet),
        ("Siamese U-Net", SiameseUNet),
        ("MobileNetV2", MobileNetSeg),
    ]

    for name, fn in models_to_run:
        run_model(name, fn, dataset)
