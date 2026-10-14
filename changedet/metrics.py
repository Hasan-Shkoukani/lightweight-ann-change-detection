import os

import torch


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


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size
