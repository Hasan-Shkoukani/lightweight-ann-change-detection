import torch

PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 5
RUNS = 3
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEVIR_ROOT = "LEVIR CD"
