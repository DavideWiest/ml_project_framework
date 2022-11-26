"""
Main execution file
"""
from logger import logging
from pathlib import Path
import os, json
import torch
from torch import nn

from ml_modules import test_step, train_step, train_full_fn, accuracy_fn_regression

from model import BasicModel

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR_NAME = os.path.dirname(Path(__file__).resolve())
CONFIG_JSON_PATH = BASE_DIR / "config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = os.cpu_count()

with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
    mconfig = config["model"]

EPOCHS = mconfig["epochs"]
SAVE_EACH = mconfig["save_each"]
EARLY_STOP_EPOCH = mconfig["early_stop_epoch"]

# 1. Instantiate Model
# 2. Define Loss function and optimizer
# 3. Get data (more to device)
# 4. use train_full_fn
# 5. analyze?

model = BasicModel(mconfig["input_features"], mconfig["output_features"], mconfig["hidden_units"]).to(device)

loss_fn = nn.L1Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=mconfig["lr"])


if __name__ == '__main__':
    ...

