"""
train.py
========
Training entry point for the EdgeGuard AI Brake Health Prediction model.

Run directly:
    python train.py

The script will prompt for the dataset CSV path in the terminal. It can also
be supplied non-interactively:
    python train.py --data /path/to/data.csv
    python train.py --data /path/to/data.csv --resume outputs/last_checkpoint.pt
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import load_and_prepare_data
from model import build_model, clamp_outputs
from utils import (
    log_header, log_info, log_success, log_warning, log_error,
    set_seed, get_device, EarlyStopping, AverageMeter,
    save_checkpoint, load_checkpoint,
    compute_regression_metrics, compute_classification_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EdgeGuard AI Brake Health model.")
    parser.add_argument("--data", type=str, default=None, help="Path to the training CSV file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs.")
    return parser.parse_args()


def prompt_for_dataset_path() -> str:
    while True:
        path = input("Enter dataset path: ").strip().strip('"').strip("'")
        if os.path.isfile(path):
            return path
        log_error(f"File not found: '{path}'. Please try again.")


def compute_total_loss(outputs, batch, criteria, loss_weights, device):
    health_loss = criteria["huber"](outputs["brake_health"], batch["brake_health"].to(device))
    pad_life_loss = criteria["huber"](outputs["remaining_pad_life"], batch["remaining_pad_life"].to(device))
    fade_loss = criteria["ce"](outputs["fade_risk_logits"], batch["fade_risk"].to(device))
    maintenance_loss = criteria["ce"](outputs["maintenance_logits"], batch["maintenance_action"].to(device))

    total = (
        loss_weights.brake_health_weight * health_loss
        + loss_weights.remaining_pad_life_weight * pad_life_loss
        + loss_weights.fade_risk_weight * fade_loss
        + loss_weights.maintenance_action_weight * maintenance_loss
    )
    breakdown = {
        "health_loss": health_loss.item(),
        "pad_life_loss": pad_life_loss.item(),
        "fade_loss": fade_loss.item(),
        "maintenance_loss": maintenance_loss.item(),
        "total_loss": total.item(),
    }
    return total, breakdown


def run_epoch(model, loader, criteria, loss_weights, device, optimizer=None, scaler=None, grad_clip=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    progress = tqdm(loader, desc="Train" if is_train else "Val  ", leave=False)

    with torch.set_grad_enabled(is_train):
        for batch in progress:
            inputs = batch["inputs"].to(device)

            if is_train:
                optimizer.zero_grad()

            use_amp = scaler is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                total_loss, breakdown = compute_total_loss(outputs, batch, criteria, loss_weights, device)

            if is_train:
                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            loss_meter.update(breakdown["total_loss"], n=inputs.size(0))
            progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    return loss_meter.average


@torch.no_grad()
def evaluate_on_test_set(model, loader, device, fade_encoder, maintenance_encoder, target_scaler, constraints):
    model.eval()

    all_health_true, all_health_pred = [], []
    all_pad_life_true, all_pad_life_pred = [], []
    all_fade_true, all_fade_pred = [], []
    all_maintenance_true, all_maintenance_pred = [], []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        outputs = model(inputs)

        # Model outputs and stored batch targets are both in normalized
        # [0, 1] target space. Inverse-transform back to physical units
        # (percent / km) before clamping and computing human-readable metrics.
        pred_normalized = np.column_stack([
            outputs["brake_health"].cpu().numpy(),
            outputs["remaining_pad_life"].cpu().numpy(),
        ])
        true_normalized = np.column_stack([
            batch["brake_health"].numpy(),
            batch["remaining_pad_life"].numpy(),
        ])
        pred_physical = target_scaler.inverse_transform(pred_normalized)
        true_physical = target_scaler.inverse_transform(true_normalized)

        health_pred_t = torch.tensor(pred_physical[:, 0])
        pad_life_pred_t = torch.tensor(pred_physical[:, 1])
        health_pred, pad_life_pred = clamp_outputs(health_pred_t, pad_life_pred_t, constraints)

        all_health_true.extend(true_physical[:, 0].tolist())
        all_health_pred.extend(health_pred.tolist())

        all_pad_life_true.extend(true_physical[:, 1].tolist())
        all_pad_life_pred.extend(pad_life_pred.tolist())

        all_fade_true.extend(batch["fade_risk"].tolist())
        all_fade_pred.extend(torch.argmax(outputs["fade_risk_logits"], dim=1).cpu().tolist())

        all_maintenance_true.extend(batch["maintenance_action"].tolist())
        all_maintenance_pred.extend(torch.argmax(outputs["maintenance_logits"], dim=1).cpu().tolist())

    metrics = {}
    metrics.update(compute_regression_metrics(all_health_true, all_health_pred, "brake_health"))
    metrics.update(compute_regression_metrics(all_pad_life_true, all_pad_life_pred, "remaining_pad_life"))
    metrics.update(compute_classification_metrics(
        all_fade_true, all_fade_pred, "fade_risk", target_names=[str(c) for c in fade_encoder.classes_]
    ))
    metrics.update(compute_classification_metrics(
        all_maintenance_true, all_maintenance_pred, "maintenance_action",
        target_names=[str(c) for c in maintenance_encoder.classes_]
    ))
    return metrics


def main():
    args = parse_args()
    config = CONFIG
    if args.epochs is not None:
        config.training.epochs = args.epochs

    output_dir = config.ensure_output_dir()
    set_seed(config.data.random_seed)
    device = get_device()

    log_header("EdgeGuard AI - Brake Health Prediction - Training")
    log_info(f"Using device: {device}")

    dataset_path = args.data if args.data else prompt_for_dataset_path()

    (train_dataset, val_dataset, test_dataset,
     scaler, target_scaler, fade_encoder, maintenance_encoder, detected) = load_and_prepare_data(dataset_path, config)

    # Update model config with the actual number of detected classes.
    config.model.input_dim = len(detected.ordered_input_names)
    config.model.num_fade_risk_classes = len(fade_encoder.classes_)
    config.model.num_maintenance_classes = len(maintenance_encoder.classes_)

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = build_model(config).to(device)
    log_success(f"Model built. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience,
        min_lr=config.training.lr_scheduler_min_lr,
    )

    criteria = {
        "huber": nn.HuberLoss(delta=config.loss.huber_delta),
        "ce": nn.CrossEntropyLoss(),
    }

    use_amp = config.training.use_amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        log_info("CUDA detected: automatic mixed precision (AMP) is ENABLED.")
    else:
        log_info("Training in full precision (AMP disabled: no CUDA device found).")

    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=config.training.early_stopping_min_delta,
    )

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        if os.path.isfile(args.resume):
            log_info(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            early_stopping.best_loss = best_loss
            log_success(f"Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")
        else:
            log_warning(f"Resume checkpoint not found at '{args.resume}'. Starting fresh.")

    history_path = os.path.join(output_dir, config.training.history_filename)
    write_header = not os.path.isfile(history_path) or start_epoch == 1
    history_file = open(history_path, "a" if not write_header else "w", newline="")
    history_writer = csv.writer(history_file)
    if write_header:
        history_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "epoch_time_sec"])
        history_file.flush()

    log_header("Training")
    for epoch in range(start_epoch, config.training.epochs + 1):
        epoch_start = time.time()

        train_loss = run_epoch(
            model, train_loader, criteria, config.loss, device,
            optimizer=optimizer, scaler=grad_scaler, grad_clip=config.training.grad_clip_norm,
        )
        val_loss = run_epoch(model, val_loader, criteria, config.loss, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        log_info(
            f"Epoch {epoch:4d}/{config.training.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
        )

        history_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.8f}", f"{epoch_time:.2f}"])
        history_file.flush()

        is_best = early_stopping.step(val_loss)
        if is_best:
            best_loss = val_loss
            save_checkpoint(
                os.path.join(output_dir, config.training.checkpoint_filename),
                model, optimizer, scheduler, epoch, best_loss,
            )
            log_success(f"New best model saved (val_loss={best_loss:.4f}).")

        save_checkpoint(
            os.path.join(output_dir, config.training.last_checkpoint_filename),
            model, optimizer, scheduler, epoch, best_loss,
        )

        if early_stopping.should_stop:
            log_warning(f"Early stopping triggered after {epoch} epochs (no improvement in {config.training.early_stopping_patience} epochs).")
            break

    history_file.close()

    # --- Final evaluation on the held-out test set using the BEST checkpoint ---
    log_header("Final Evaluation on Test Set (best checkpoint)")
    best_ckpt_path = os.path.join(output_dir, config.training.checkpoint_filename)
    if os.path.isfile(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model, map_location=device)
        log_success(f"Loaded best checkpoint from '{best_ckpt_path}'.")

    metrics = evaluate_on_test_set(
        model, test_loader, device, fade_encoder, maintenance_encoder, target_scaler, config.output_constraints
    )

    log_info(f"Brake Health       -> MAE: {metrics['brake_health_mae']:.3f} | RMSE: {metrics['brake_health_rmse']:.3f} | R2: {metrics['brake_health_r2']:.3f}")
    log_info(f"Remaining Pad Life -> MAE: {metrics['remaining_pad_life_mae']:.3f} | RMSE: {metrics['remaining_pad_life_rmse']:.3f} | R2: {metrics['remaining_pad_life_r2']:.3f}")
    log_info(f"Brake Fade Risk    -> Acc: {metrics['fade_risk_accuracy']:.3f} | Prec: {metrics['fade_risk_precision']:.3f} | Rec: {metrics['fade_risk_recall']:.3f} | F1: {metrics['fade_risk_f1']:.3f}")
    log_info(f"Maintenance Action -> Acc: {metrics['maintenance_action_accuracy']:.3f} | Prec: {metrics['maintenance_action_precision']:.3f} | Rec: {metrics['maintenance_action_recall']:.3f} | F1: {metrics['maintenance_action_f1']:.3f}")

    print("\n--- Brake Fade Risk Classification Report ---")
    print(metrics["fade_risk_report"])
    print("--- Maintenance Action Classification Report ---")
    print(metrics["maintenance_action_report"])

    metrics_path = os.path.join(output_dir, "test_set_metrics.json")
    serializable_metrics = {k: v for k, v in metrics.items() if not k.endswith("_report")}
    from utils import save_json
    save_json(serializable_metrics, metrics_path)
    log_success(f"Saved test set metrics to '{metrics_path}'.")

    log_header("Training complete")
    log_success(f"Best model checkpoint: {best_ckpt_path}")
    log_success(f"All artifacts saved under: {output_dir}/")


if __name__ == "__main__":
    main()
