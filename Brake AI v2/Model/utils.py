"""
utils.py
========
Shared utilities used across the EdgeGuard AI Brake Health framework:
reproducibility, colored logging, checkpointing, early stopping and
evaluation metrics.
"""

import os
import random
import json
from typing import Dict, Optional

import numpy as np
import torch
from colorama import Fore, Style, init as colorama_init
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

colorama_init(autoreset=True)


# =====================================================================
# LOGGING
# =====================================================================
def log_info(message: str) -> None:
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")


def log_success(message: str) -> None:
    print(f"{Fore.GREEN}[ OK ]{Style.RESET_ALL} {message}")


def log_warning(message: str) -> None:
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {message}")


def log_error(message: str) -> None:
    print(f"{Fore.RED}[FAIL]{Style.RESET_ALL} {message}")


def log_header(message: str) -> None:
    bar = "=" * 70
    print(f"\n{Fore.MAGENTA}{bar}\n{message}\n{bar}{Style.RESET_ALL}")


# =====================================================================
# REPRODUCIBILITY
# =====================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# EARLY STOPPING
# =====================================================================
class EarlyStopping:
    """Stops training when the monitored (validation) loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, current_loss: float) -> bool:
        """Returns True if this is the best loss seen so far."""
        if self.best_loss is None or current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# =====================================================================
# AVERAGE METER
# =====================================================================
class AverageMeter:
    """Tracks a running average of a scalar value (e.g. loss per batch)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


# =====================================================================
# CHECKPOINTING
# =====================================================================
def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, best_loss: float) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location=None) -> Dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# =====================================================================
# OUTPUT CONSTRAINT CLAMPING
# =====================================================================
def clamp_value(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


# =====================================================================
# METRICS
# =====================================================================
def compute_regression_metrics(y_true, y_pred, name: str) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        f"{name}_mae": mae,
        f"{name}_rmse": rmse,
        f"{name}_r2": r2,
    }


def compute_classification_metrics(y_true, y_pred, name: str, target_names=None) -> Dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    return {
        f"{name}_accuracy": accuracy,
        f"{name}_precision": precision,
        f"{name}_recall": recall,
        f"{name}_f1": f1,
        f"{name}_confusion_matrix": cm.tolist(),
        f"{name}_report": report,
    }


# =====================================================================
# SIMPLE JSON HELPERS
# =====================================================================
def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
