import time

import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE, EPOCHS, LR
from .metrics import segmentation_metrics


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
