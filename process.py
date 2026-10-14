import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 5
RUNS = 3
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEVIR_ROOT = "LEVIR CD"


class LEVIRDataset(Dataset):
    def __init__(self, root=LEVIR_ROOT, splits=("train", "val", "test")):
        self.samples = []

        for split in splits:
            a_dir = os.path.join(root, split, "A")
            b_dir = os.path.join(root, split, "B")
            l_dir = os.path.join(root, split, "label")

            for fname in sorted(os.listdir(a_dir)):
                self.samples.append((
                    os.path.join(a_dir, fname),
                    os.path.join(b_dir, fname),
                    os.path.join(l_dir, fname)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_path, b_path, l_path = self.samples[idx]

        before = cv2.imread(a_path)
        after = cv2.imread(b_path)
        label = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)

        before = cv2.resize(before, (PATCH_SIZE, PATCH_SIZE))
        after = cv2.resize(after, (PATCH_SIZE, PATCH_SIZE))
        label = cv2.resize(label, (PATCH_SIZE, PATCH_SIZE))

        x = np.concatenate([before, after], axis=2)

        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1) / 255.0

        label = (label > 127).astype(np.float32)
        y = torch.tensor(label).unsqueeze(0)

        return x, y


def segmentation_metrics(preds, targets):
    preds = (preds > 0.5).float()

    TP = (preds * targets).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "IoU": iou,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": acc,
    }


class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.block(6, 64)
        self.enc2 = self.block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.block(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = self.block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return torch.sigmoid(self.out(d2))


class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        before = x[:, :3]
        after = x[:, 3:]

        feat1 = self.encoder(before)
        feat2 = self.encoder(after)

        fusion = torch.cat([feat1, feat2], dim=1)

        return self.decoder(fusion)


class MobileNetSeg(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(weights=None)
        backbone.features[0][0] = nn.Conv2d(6, 32, 3, stride=2, padding=1, bias=False)

        self.encoder = backbone.features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(128,128))
        return x


def train(model, loader):
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        total_loss = 0

        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            preds = model(X)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(loader):.4f}")


def evaluate(model, loader):
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            preds = model(X)
            metrics = segmentation_metrics(preds.cpu(), y.cpu())
            all_metrics.append(metrics)

    final = {}

    for k in all_metrics[0].keys():
        final[k] = np.mean([m[k] for m in all_metrics])

    return final


def benchmark_fps(model, loader):
    model.eval()

    start = time.time()

    with torch.no_grad():
        for X, _ in loader:
            model(X.to(DEVICE))

    elapsed = time.time() - start

    return len(loader.dataset) / elapsed


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size


def show_predictions(model, dataset, idx=0):
    model.eval()

    X, y = dataset[idx]

    with torch.no_grad():
        pred = model(X.unsqueeze(0).to(DEVICE))[0][0].cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    img = X.permute(1,2,0).numpy()

    before = img[:,:,:3]
    after = img[:,:,3:]

    gt = y.squeeze().numpy()

    fig, axes = plt.subplots(1,4, figsize=(16,4))

    axes[0].imshow(before)
    axes[0].set_title("Before")

    axes[1].imshow(after)
    axes[1].set_title("After")

    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("Ground Truth")

    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


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
