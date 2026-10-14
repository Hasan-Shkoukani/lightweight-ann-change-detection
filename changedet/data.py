import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import LEVIR_ROOT, PATCH_SIZE


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
