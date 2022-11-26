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

model = BasicModel(mconfig["input_features"], mconfig["output_features"], mconfig["hidden_units"]).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=mconfig["lr"]).to(device)

if __name__ == '__main__':
    ...

