"""
train.py
========

Command-line training entry point for EdgeGuard AI Brake Health Prediction.

Usage
-----
    python train.py --dataset /path/to/data.csv
    python train.py --dataset /path/to/more_data.csv --resume latest
    python train.py --dataset /path/to/more_data.csv --resume best
    python train.py --dataset /path/to/data.csv --fresh
    python train.py --dataset /path/to/data.csv --epochs 300 --batch-size 256

Resume semantics
----------------
    --resume latest   Resume optimizer/scheduler/model state from the most
                       recent checkpoint (default behavior if a checkpoint
                       exists and --fresh is not passed).
    --resume best      Resume from the best-validation-loss checkpoint
                       (useful for continued fine-tuning on new data
                       without carrying forward a recent overfit state).
    --fresh            Ignore any existing checkpoints and train from
                       scratch. Existing checkpoints are NOT deleted --
                       new checkpoints are written alongside them and
                       will overwrite latest/best going forward.

Early stopping never blocks a --resume: the `epochs_without_improvement`
counter is persisted in the checkpoint, so resuming reflects the true
training history rather than resetting patience to zero.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from dataset import (
    BrakeHealthDataset,
    fit_preprocessors,
    load_and_clean_csv,
    stratified_split,
    transform_split,
)
from losses import MultiTaskLoss
from metrics import compute_classification_metrics, compute_regression_metrics
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EdgeGuard AI Brake Health model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a CSV dataset (required, never hardcoded).")
    parser.add_argument("--resume", type=str, choices=["latest", "best"], default=None,
                        help="Resume training from a checkpoint ('latest' or 'best').")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints; train from scratch.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


def _dataloaders(
    train_df, val_df, test_df, preprocessors, batch_size: int, num_workers: int
):
    X_train, yr_train, yf_train, ym_train = transform_split(train_df, preprocessors)
    X_val, yr_val, yf_val, ym_val = transform_split(val_df, preprocessors)
    X_test, yr_test, yf_test, ym_test = transform_split(test_df, preprocessors)

    train_ds = BrakeHealthDataset(X_train, yr_train, yf_train, ym_train)
    val_ds = BrakeHealthDataset(X_val, yr_val, yf_val, ym_val)
    test_ds = BrakeHealthDataset(X_test, yr_test, yf_test, ym_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader


def _run_epoch(
    model,
    loader,
    loss_fn: MultiTaskLoss,
    device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip_norm: float,
    train: bool,
):
    """Run a single epoch of training or evaluation.

    When `train` is True, performs backprop with AMP + gradient clipping.
    When False, runs in no_grad eval mode and additionally collects
    predictions for metric computation.
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_fade_loss = 0.0
    total_maint_loss = 0.0
    n_batches = 0

    all_reg_pred, all_reg_true = [], []
    all_fade_pred, all_fade_true = [], []
    all_maint_pred, all_maint_true = [], []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for X, y_reg, y_fade, y_maint in loader:
            X = X.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_fade = y_fade.to(device, non_blocking=True)
            y_maint = y_maint.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and scaler.is_enabled()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                reg_pred, fade_logits, maint_logits = model(X)
                losses = loss_fn(reg_pred, fade_logits, maint_logits, y_reg, y_fade, y_maint)

            if train:
                if use_amp:
                    scaler.scale(losses.total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            total_loss += losses.total_loss.item()
            total_reg_loss += losses.regression_loss.item()
            total_fade_loss += losses.fade_risk_loss.item()
            total_maint_loss += losses.maintenance_loss.item()
            n_batches += 1

            all_reg_pred.append(reg_pred.detach().cpu())
            all_reg_true.append(y_reg.detach().cpu())
            all_fade_pred.append(torch.argmax(fade_logits.detach(), dim=1).cpu())
            all_fade_true.append(y_fade.detach().cpu())
            all_maint_pred.append(torch.argmax(maint_logits.detach(), dim=1).cpu())
            all_maint_true.append(y_maint.detach().cpu())

    import numpy as np
    reg_pred_np = torch.cat(all_reg_pred).numpy().flatten()
    reg_true_np = torch.cat(all_reg_true).numpy().flatten()
    fade_pred_np = torch.cat(all_fade_pred).numpy()
    fade_true_np = torch.cat(all_fade_true).numpy()
    maint_pred_np = torch.cat(all_maint_pred).numpy()
    maint_true_np = torch.cat(all_maint_true).numpy()

    fade_acc = float((fade_pred_np == fade_true_np).mean())
    maint_acc = float((maint_pred_np == maint_true_np).mean())
    # Regression MAE reported here is in SCALED [0,1] units (fast proxy for
    # monitoring during training); de-scaled MAE is computed separately at
    # final test-set evaluation time.
    reg_mae_scaled = float(np.mean(np.abs(reg_pred_np - reg_true_np)))

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "regression_loss": total_reg_loss / max(n_batches, 1),
        "fade_risk_loss": total_fade_loss / max(n_batches, 1),
        "maintenance_loss": total_maint_loss / max(n_batches, 1),
        "regression_mae_scaled": reg_mae_scaled,
        "fade_risk_acc": fade_acc,
        "maintenance_acc": maint_acc,
    }
    return metrics


def main() -> None:
    args = parse_args()
    config.ensure_directories()

    logger = utils.setup_logger("edgeguard.train", log_file=os.path.join(config.LOGS_DIR, "train.log"))
    utils.set_seed(config.RANDOM_SEED)

    train_cfg = config.TrainConfig()
    model_cfg = config.ModelConfig()

    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr

    logger.info(f"Loading dataset from: {args.dataset}")
    df, report = load_and_clean_csv(args.dataset, logger=logger)

    train_df, val_df, test_df = stratified_split(df, train_cfg.val_split, train_cfg.test_split)
    logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    preprocessors = fit_preprocessors(train_df)
    preprocessors.save(config.SCALERS_DIR)
    logger.info(f"Fitted preprocessors saved to {config.SCALERS_DIR}")

    train_loader, val_loader, test_loader = _dataloaders(
        train_df, val_df, test_df, preprocessors, train_cfg.batch_size, train_cfg.num_workers
    )

    device = utils.get_device(logger)
    model = build_model(model_cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.lr_scheduler_t_max, eta_min=train_cfg.lr_scheduler_eta_min)
    amp_enabled = train_cfg.use_amp and device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_fn = MultiTaskLoss(train_cfg)

    writer = SummaryWriter(log_dir=config.TENSORBOARD_DIR)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    latest_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, config.LATEST_CHECKPOINT_NAME)
    best_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, config.BEST_CHECKPOINT_NAME)

    resume_path = None
    if not args.fresh:
        if args.resume == "best" and os.path.exists(best_ckpt_path):
            resume_path = best_ckpt_path
        elif os.path.exists(latest_ckpt_path):
            resume_path = latest_ckpt_path

    if resume_path is not None:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = utils.load_checkpoint(
            resume_path, model, optimizer, scheduler, grad_scaler, device=device
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        logger.info(
            f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}, "
            f"epochs_without_improvement={epochs_without_improvement}"
        )
    else:
        logger.info("Starting fresh training run (no checkpoint loaded).")

    if start_epoch > train_cfg.num_epochs:
        logger.info(
            f"start_epoch ({start_epoch}) already exceeds num_epochs ({train_cfg.num_epochs}); "
            "nothing to do. Increase --epochs to continue training further."
        )
    else:
        for epoch in range(start_epoch, train_cfg.num_epochs + 1):
            epoch_start = time.time()

            train_metrics = _run_epoch(
                model, train_loader, loss_fn, device, optimizer, grad_scaler,
                train_cfg.grad_clip_norm, train=True,
            )
            val_metrics = _run_epoch(
                model, val_loader, loss_fn, device, optimizer=None, scaler=None,
                grad_clip_norm=train_cfg.grad_clip_norm, train=False,
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{train_cfg.num_epochs} | "
                f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} | "
                f"val_fade_acc={val_metrics['fade_risk_acc']:.3f} "
                f"val_maint_acc={val_metrics['maintenance_acc']:.3f} | "
                f"lr={current_lr:.2e} | {epoch_time:.1f}s"
            )

            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val_fade_risk", val_metrics["fade_risk_acc"], epoch)
            writer.add_scalar("Accuracy/val_maintenance", val_metrics["maintenance_acc"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

            utils.append_history(config.HISTORY_PATH, {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_regression_mae": train_metrics["regression_mae_scaled"],
                "val_regression_mae": val_metrics["regression_mae_scaled"],
                "train_fade_risk_acc": train_metrics["fade_risk_acc"],
                "val_fade_risk_acc": val_metrics["fade_risk_acc"],
                "train_maintenance_acc": train_metrics["maintenance_acc"],
                "val_maintenance_acc": val_metrics["maintenance_acc"],
                "lr": current_lr,
            })

            improved = val_metrics["loss"] < (best_val_loss - train_cfg.early_stopping_min_delta)
            if improved:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                utils.save_checkpoint(
                    best_ckpt_path, model, optimizer, scheduler, grad_scaler,
                    epoch, best_val_loss, epochs_without_improvement,
                )
                logger.info(f"  -> New best validation loss: {best_val_loss:.6f}. Best checkpoint saved.")
            else:
                epochs_without_improvement += 1

            utils.save_checkpoint(
                latest_ckpt_path, model, optimizer, scheduler, grad_scaler,
                epoch, best_val_loss, epochs_without_improvement,
            )

            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epochs_without_improvement} epochs "
                    f"without improvement (patience={train_cfg.early_stopping_patience})."
                )
                break

    writer.close()

    # ------------------------------------------------------------------
    # Final evaluation on the held-out test set, using the BEST checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(best_ckpt_path):
        logger.info("Loading best checkpoint for final test-set evaluation.")
        utils.load_checkpoint(best_ckpt_path, model, device=device)

    model.eval()
    all_reg_pred, all_reg_true = [], []
    all_fade_pred, all_fade_true = [], []
    all_maint_pred, all_maint_true = [], []

    with torch.no_grad():
        for X, y_reg, y_fade, y_maint in test_loader:
            X = X.to(device)
            reg_pred, fade_logits, maint_logits = model(X)
            all_reg_pred.append(reg_pred.cpu())
            all_reg_true.append(y_reg)
            all_fade_pred.append(torch.argmax(fade_logits, dim=1).cpu())
            all_fade_true.append(y_fade)
            all_maint_pred.append(torch.argmax(maint_logits, dim=1).cpu())
            all_maint_true.append(y_maint)

    import numpy as np
    reg_pred_scaled = torch.cat(all_reg_pred).numpy()
    reg_true_scaled = torch.cat(all_reg_true).numpy()
    reg_pred_real = preprocessors.regression_scaler.inverse_transform(reg_pred_scaled).flatten()
    reg_true_real = preprocessors.regression_scaler.inverse_transform(reg_true_scaled).flatten()

    reg_metrics = compute_regression_metrics(reg_true_real, reg_pred_real)
    logger.info(
        f"[TEST] Brake Health Regression -> MAE={reg_metrics.mae:.3f}%, "
        f"RMSE={reg_metrics.rmse:.3f}%, R2={reg_metrics.r2:.4f}"
    )

    fade_pred_np = torch.cat(all_fade_pred).numpy()
    fade_true_np = torch.cat(all_fade_true).numpy()
    fade_metrics = compute_classification_metrics(fade_true_np, fade_pred_np, config.FADE_RISK_CLASSES)
    logger.info(
        f"[TEST] Fade Risk Classification -> Accuracy={fade_metrics.accuracy:.4f}, "
        f"Macro-F1={fade_metrics.macro_f1:.4f}"
    )
    logger.info("\n" + fade_metrics.report_text)

    maint_pred_np = torch.cat(all_maint_pred).numpy()
    maint_true_np = torch.cat(all_maint_true).numpy()
    maint_metrics = compute_classification_metrics(maint_true_np, maint_pred_np, config.MAINTENANCE_CLASSES)
    logger.info(
        f"[TEST] Maintenance Classification -> Accuracy={maint_metrics.accuracy:.4f}, "
        f"Macro-F1={maint_metrics.macro_f1:.4f}"
    )
    logger.info("\n" + maint_metrics.report_text)

    history = utils.load_history(config.HISTORY_PATH)
    utils.plot_training_curves(history, config.PLOTS_DIR)
    logger.info(f"Training curves saved to {config.PLOTS_DIR}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
