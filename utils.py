# utils.py
import os
import random
from typing import Literal

import numpy as np
import torch

from config import cfg


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(arg_device: Literal["auto", "cpu", "cuda"] = "auto") -> torch.device:
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
