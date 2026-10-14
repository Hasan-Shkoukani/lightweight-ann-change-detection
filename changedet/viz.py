import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DEVICE


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
