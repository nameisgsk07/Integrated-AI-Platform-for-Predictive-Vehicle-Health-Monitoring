"""
train.py
========
Trains the Tyre Health Prediction model (Good vs Defective) using transfer
learning on top of a pretrained MobileNetV3-Large backbone.

WHAT THIS SCRIPT DOES, STEP BY STEP:
 1. Splits your dataset into Train (80%) / Val (10%) / Test (10%).
 2. Loads the images using torchvision's ImageFolder + ImageNet transforms.
 3. Downloads pretrained MobileNetV3-Large weights (only once, then cached).
 4. Freezes the backbone and trains only the new classifier head for a few
    epochs, then unfreezes the backbone and fine-tunes the whole network.
 5. Tracks training/validation loss & accuracy every epoch.
 6. Saves the BEST model (highest validation accuracy) to
    outputs/tyre_health_model.pt
 7. Saves training_history.csv, accuracy_graph.png, loss_graph.png.
 8. Evaluates the best model on the held-out Test set and saves
    confusion_matrix.png + classification_report.txt.

Run it with:
    python train.py

If training is interrupted (Ctrl+C, power cut, crash), just run
`python train.py` again - it will automatically resume from the last
saved checkpoint.

CONTINUE TRAINING FROM AN EXISTING MODEL:
By default, this script checks whether outputs/tyre_health_model.pt already
exists. If it does, training continues from those weights instead of
re-downloading/re-initializing from ImageNet. You can also point to a
specific model file to continue from using --resume_model, e.g.:

    python train.py --resume_model outputs/tyre_health_model.pt

The dataset location can also be overridden from the command line with
--dataset_dir, e.g.:

    python train.py --dataset_dir "D:\\path\\to\\online dataset"
"""

import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # so it works even without a display (headless servers)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import config
from common import (
    set_seed, get_device, print_info, print_success, print_warning,
    print_error, print_header, get_transforms, prepare_dataset_split,
    build_model, unfreeze_backbone, compute_class_weights,
    count_trainable_parameters, display_class_name,
)


# ============================================================
# ONE TRAINING EPOCH
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num}/{total_epochs} [Train]",
        leave=False,
        colour="green",
    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================
# ONE VALIDATION EPOCH
# ============================================================
@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch_num, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch_num}/{total_epochs} [Val]  ",
        leave=False,
        colour="blue",
    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================
# FULL EVALUATION (used on the Test set at the very end)
# ============================================================
@torch.no_grad()
def evaluate_full(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Evaluating on Test set", colour="magenta"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def build_optimizer(model, backbone_frozen):
    """
    Builds an optimizer with different learning rates for the backbone and
    the classifier head. While the backbone is frozen its parameters have
    requires_grad=False so they are simply not included.
    """
    head_params = list(model.classifier.parameters())

    if backbone_frozen:
        return optim.Adam(head_params, lr=config.LEARNING_RATE_HEAD,
                           weight_decay=config.WEIGHT_DECAY)

    backbone_params = list(model.features.parameters())
    return optim.Adam(
        [
            {"params": backbone_params, "lr": config.LEARNING_RATE_BACKBONE},
            {"params": head_params, "lr": config.LEARNING_RATE_FINETUNE_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train (or continue training) the EdgeGuard AI Tyre Health model."
    )
    parser.add_argument(
        "--resume_model",
        type=str,
        default=config.MODEL_PATH,
        help=(
            "Path to an existing trained model (.pt) to continue training from. "
            "Defaults to outputs/tyre_health_model.pt. If this file exists, its "
            "weights are loaded and training continues from them instead of "
            "starting over from ImageNet weights. If it does not exist, training "
            "starts from pretrained ImageNet weights as usual."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=config.SOURCE_DATASET_DIR,
        help="Path to the source dataset folder (containing the class sub-folders). "
             "Defaults to SOURCE_DATASET_DIR in config.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print_header("EdgeGuard AI - Tyre Health Prediction - TRAINING")

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print_info(f"Using device: {device}")
    if device.type == "cuda":
        print_info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print_warning("No GPU detected - training will run on CPU (slower).")

    # --------------------------------------------------------
    # 1. Prepare Train / Val / Test split
    # --------------------------------------------------------
    print_header("Step 1: Preparing dataset split (80% / 10% / 10%)")
    print_info(f"Dataset source directory: {args.dataset_dir}")
    prepare_dataset_split(
        source_dir=args.dataset_dir,
        output_dir=config.SPLIT_DATASET_DIR,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        valid_extensions=config.VALID_EXTENSIONS,
        seed=config.RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 2. Datasets & DataLoaders
    # --------------------------------------------------------
    print_header("Step 2: Loading datasets")
    train_transform, eval_transform = get_transforms(
        config.IMAGE_SIZE, config.IMAGENET_MEAN, config.IMAGENET_STD
    )

    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=eval_transform)

    class_names = train_dataset.classes  # automatically detected, e.g. ['defective', 'good']
    num_classes = len(class_names)
    display_names = [display_class_name(c) for c in class_names]

    print_info(f"Detected class names: {class_names}  -> displayed as {display_names}")
    print_info(f"Train images: {len(train_dataset)}")
    print_info(f"Val images:   {len(val_dataset)}")
    print_info(f"Test images:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # 3. Build model - CONTINUE TRAINING if a saved model already exists,
    #    otherwise start from pretrained ImageNet weights.
    # --------------------------------------------------------
    print_header("Step 3: Building MobileNetV3-Large (transfer learning)")

    if os.path.exists(args.resume_model):
        print_info(f"Found existing trained model at '{args.resume_model}'.")
        print_info("Continuing training from these weights (NOT reinitializing from ImageNet).")

        resume_checkpoint = torch.load(args.resume_model, map_location=device)
        saved_class_names = resume_checkpoint.get("class_names", class_names)

        if saved_class_names != class_names:
            print_warning(
                f"Class names in '{args.resume_model}' ({saved_class_names}) do not match "
                f"the current dataset's classes ({class_names}). Loading weights anyway, "
                f"but double-check your dataset folders if this is unexpected."
            )

        # pretrained=False: build the bare architecture only - we load the
        # already-trained weights ourselves right after, so no ImageNet
        # download/initialization happens.
        model = build_model(num_classes=num_classes, freeze_backbone=True, pretrained=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        print_success(f"Loaded existing model weights from '{args.resume_model}'. "
                       f"Previous validation accuracy: "
                       f"{resume_checkpoint.get('val_accuracy', 0.0) * 100:.2f}%")
    else:
        print_info(f"No existing trained model found at '{args.resume_model}'.")
        print_info("Starting training from pretrained ImageNet weights (downloaded automatically).")
        model = build_model(num_classes=num_classes, freeze_backbone=True, pretrained=True)

    model.to(device)
    print_success(f"Model ready. Trainable parameters: {count_trainable_parameters(model):,}")

    # --------------------------------------------------------
    # 4. Loss function (optionally class-weighted), optimizer, scheduler
    # --------------------------------------------------------
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        print_info(f"Using class weights (to balance Good/Defective): {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    backbone_frozen = True
    optimizer = build_optimizer(model, backbone_frozen)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # --------------------------------------------------------
    # 5. Resume from checkpoint if one exists
    # --------------------------------------------------------
    start_epoch = 1
    best_val_acc = 0.0
    early_stop_counter = 0
    history = []

    if os.path.exists(config.CHECKPOINT_PATH):
        print_header("Resuming training from last checkpoint")
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)

        backbone_frozen = checkpoint["backbone_frozen"]
        if not backbone_frozen:
            model = unfreeze_backbone(model)
            optimizer = build_optimizer(model, backbone_frozen=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print_warning("Could not restore optimizer state exactly (phase change). "
                           "Continuing with a fresh optimizer for this phase.")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        early_stop_counter = checkpoint["early_stop_counter"]
        history = checkpoint["history"]

        print_success(f"Resumed from epoch {checkpoint['epoch']}. "
                       f"Continuing at epoch {start_epoch}. Best val acc so far: {best_val_acc:.4f}")
    else:
        print_info("No existing checkpoint found - starting fresh training.")

    # --------------------------------------------------------
    # 6. Training loop
    # --------------------------------------------------------
    print_header("Step 4: Training")
    print_info(f"Total epochs: {config.NUM_EPOCHS} | "
                f"Backbone frozen for first {config.FREEZE_EPOCHS} epochs | "
                f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")

    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):

            # Unfreeze the backbone once we reach FREEZE_EPOCHS
            if backbone_frozen and epoch > config.FREEZE_EPOCHS:
                print_info(f"Epoch {epoch}: Unfreezing backbone for fine-tuning "
                            f"(lr backbone={config.LEARNING_RATE_BACKBONE}, "
                            f"lr head={config.LEARNING_RATE_FINETUNE_HEAD})")
                model = unfreeze_backbone(model)
                backbone_frozen = False
                optimizer = build_optimizer(model, backbone_frozen=False)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
                    patience=config.LR_SCHEDULER_PATIENCE,
                )

            epoch_start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS
            )
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion, device, epoch, config.NUM_EPOCHS
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            phase = "FROZEN " if backbone_frozen else "FINETUNE"
            print(
                f"Epoch [{epoch:02d}/{config.NUM_EPOCHS}] ({phase}) "
                f"| Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}% "
                f"| Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}% "
                f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "backbone_frozen": backbone_frozen,
            })

            # ---- Save BEST model ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.IMAGE_SIZE,
                    "mean": config.IMAGENET_MEAN,
                    "std": config.IMAGENET_STD,
                    "val_accuracy": best_val_acc,
                    "epoch": epoch,
                }, config.MODEL_PATH)
                print_success(f"New best model saved! Val Accuracy: {best_val_acc*100:.2f}% "
                               f"-> {config.MODEL_PATH}")
            else:
                early_stop_counter += 1
                print_warning(f"No improvement for {early_stop_counter} epoch(s) "
                               f"(best so far: {best_val_acc*100:.2f}%)")

            # ---- Save checkpoint every epoch (for resume) ----
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "early_stop_counter": early_stop_counter,
                "history": history,
                "backbone_frozen": backbone_frozen,
                "class_names": class_names,
            }, config.CHECKPOINT_PATH)

            # ---- Early stopping ----
            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print_warning(f"Early stopping triggered after {epoch} epochs "
                               f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user (Ctrl+C). "
                       "Progress has been saved - just run 'python train.py' again to resume.")
        return

    # --------------------------------------------------------
    # 7. Save training history CSV
    # --------------------------------------------------------
    print_header("Step 5: Saving training history & graphs")
    with open(config.HISTORY_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "backbone_frozen"
        ])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print_success(f"Training history saved to {config.HISTORY_CSV_PATH}")

    # --------------------------------------------------------
    # 8. Accuracy & Loss graphs
    # --------------------------------------------------------
    epochs_range = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, [h["train_acc"] * 100 for h in history], label="Train Accuracy")
    plt.plot(epochs_range, [h["val_acc"] * 100 for h in history], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.ACCURACY_PLOT_PATH, dpi=150)
    plt.close()
    print_success(f"Accuracy graph saved to {config.ACCURACY_PLOT_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(epochs_range, [h["val_loss"] for h in history], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print_success(f"Loss graph saved to {config.LOSS_PLOT_PATH}")

    # --------------------------------------------------------
    # 9. Final evaluation on the TEST set using the BEST model
    # --------------------------------------------------------
    print_header("Step 6: Final evaluation on Test set (using BEST saved model)")
    best_checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)

    y_true, y_pred = evaluate_full(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=display_names, digits=4)

    print(report)

    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("EdgeGuard AI - Tyre Health Prediction - Test Set Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print_success(f"Classification report saved to {config.CLASSIFICATION_REPORT_PATH}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print_success(f"Confusion matrix saved to {config.CONFUSION_MATRIX_PATH}")

    print_header("TRAINING COMPLETE")
    print_success(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print_success(f"Final model saved at: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
