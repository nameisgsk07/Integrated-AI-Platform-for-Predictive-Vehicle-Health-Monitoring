"""
utils.py
========

Cross-cutting utilities shared by train.py and predict.py:

    - Logging setup
    - Deterministic seeding
    - Device selection (CUDA with automatic CPU fallback)
    - Checkpoint save / resume (robust to PyTorch 2.6+ `weights_only` changes)
    - Training curve plotting
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch

import config


# ======================================================================
# LOGGING
# ======================================================================

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a logger that writes to stdout and, optionally, a file.

    Safe to call multiple times with the same `name`: handlers are not
    duplicated on repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE SELECTION
# ======================================================================

def get_device(logger: Optional[logging.Logger] = None) -> torch.device:
    """Select CUDA if available, otherwise automatically fall back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if logger:
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if logger:
            logger.info("CUDA not available. Falling back to CPU.")
    return device


# ======================================================================
# CHECKPOINTING
# ======================================================================

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    """Persist full training state so training can be resumed exactly.

    Includes model weights, optimizer state, LR scheduler state, AMP
    scaler state, and bookkeeping (epoch, best validation loss, early
    stopping counter).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    # Atomic-ish write: write to temp file then replace, to avoid corrupt
    # checkpoints if the process is interrupted mid-write.
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint`.

    Compatible with the PyTorch 2.6+ change of `torch.load`'s default
    `weights_only=True`: our checkpoints contain plain python scalars,
    dicts and tensors only, but we explicitly pass `weights_only=False`
    (the checkpoint file is one we produced ourselves, so this is safe)
    to avoid `_pickle.UnpicklingError` / `weights_only` load failures
    seen in earlier versions of this project.
    """
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ======================================================================
# TRAINING HISTORY
# ======================================================================

def append_history(history_path: str, epoch_record: Dict[str, Any]) -> None:
    """Append a single epoch's metrics to the JSON training history file."""
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(epoch_record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(history_path: str) -> list:
    if not os.path.exists(history_path):
        return []
    with open(history_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# ======================================================================
# PLOTTING
# ======================================================================

def plot_training_curves(history: list, output_dir: str) -> None:
    """Generate and save loss/metric curves from the training history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    os.makedirs(output_dir, exist_ok=True)
    epochs = [record["epoch"] for record in history]

    # --- Total loss curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r["train_loss"] for r in history], label="Train Loss")
    plt.plot(epochs, [r["val_loss"] for r in history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EdgeGuard AI - Total Multi-Task Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Regression MAE curve ---
    if "val_regression_mae" in history[0]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [r.get("train_regression_mae", np.nan) for r in history], label="Train MAE")
        plt.plot(epochs, [r.get("val_regression_mae", np.nan) for r in history], label="Val MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Brake Health MAE (%)")
        plt.title("EdgeGuard AI - Brake Health Regression MAE")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "regression_mae_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Classification accuracy curves ---
    if "val_fade_risk_acc" in history[0]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [r.get("train_fade_risk_acc", np.nan) for r in history], label="Train Fade-Risk Acc")
        plt.plot(epochs, [r.get("val_fade_risk_acc", np.nan) for r in history], label="Val Fade-Risk Acc")
        plt.plot(epochs, [r.get("train_maintenance_acc", np.nan) for r in history], label="Train Maintenance Acc")
        plt.plot(epochs, [r.get("val_maintenance_acc", np.nan) for r in history], label="Val Maintenance Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("EdgeGuard AI - Classification Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "classification_accuracy_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Learning rate curve ---
    if "lr" in history[0]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [r["lr"] for r in history])
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("EdgeGuard AI - Learning Rate Schedule (Cosine Annealing)")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "lr_schedule.png"), dpi=150, bbox_inches="tight")
        plt.close()


def dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert a dataclass config instance into a plain dict (for logging)."""
    return asdict(obj)
