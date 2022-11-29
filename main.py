"""
Main execution file
"""
from logger import logging
from pathlib import Path
import os, json
import torch
from torch import nn

from ml_modules import test_step, train_step, train_full_fn, accuracy_fn_regression
from ml_modules import ModelManager

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
COMPARE_SAVED_METRIC = mconfig["compare_saved_metric"]
BATCH_SIZE = config["computation"]["batch_size"]

OUT_FTS = mconfig["output_features"]
IN_FTS = mconfig["input_features"]
HDN_UNITS = mconfig["hidden_units"]

# 1. Instantiate Model
# 2. Define Loss function and optimizer
# 3. Get data (more to device)
# 4. use train_full_fn
# 5. analyze?

model = BasicModel(mconfig["input_features"], mconfig["output_features"], mconfig["hidden_units"]).to(device)

modelname = model.__class__.__name__
# modelname = ""

mm = ModelManager(logging)
model_statedict, path = mm.load(modelname, load_best_metric="loss")

if model_statedict != None:
    model.load_state_dict(model_statedict)
    logging.info(f"Loaded previous model (path: {path})")
else:
    logging.info(f"Could not load any previous model for {modelname}")


loss_fn = nn.L1Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=mconfig["lr"])


if __name__ == '__main__':
    ...

