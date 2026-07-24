"""
common.py
=========
Shared helper functions used by train.py, predict.py, export_onnx.py and
test_model.py. Keeping this logic in one place means every script builds
the model, transforms and dataset split in EXACTLY the same way.
"""

import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    """Fix every random seed so results are reproducible run after run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Automatically use GPU if available, otherwise fall back to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# ============================================================
# COLORED TERMINAL OUTPUT HELPERS
# ============================================================
def print_info(msg):
    print(Fore.CYAN + "[INFO] " + Style.RESET_ALL + str(msg))


def print_success(msg):
    print(Fore.GREEN + "[SUCCESS] " + Style.RESET_ALL + str(msg))


def print_warning(msg):
    print(Fore.YELLOW + "[WARNING] " + Style.RESET_ALL + str(msg))


def print_error(msg):
    print(Fore.RED + "[ERROR] " + Style.RESET_ALL + str(msg))


def print_header(msg):
    line = "=" * 65
    print(Fore.MAGENTA + Style.BRIGHT + "\n" + line)
    print(msg)
    print(line + Style.RESET_ALL)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
def get_transforms(image_size, mean, std):
    """
    Returns (train_transform, eval_transform).

    train_transform includes light data augmentation (helps the model
    generalize better on a modestly sized dataset).
    eval_transform (used for validation/test/prediction) only resizes
    and normalizes - no randomness, so results are consistent.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


# ============================================================
# DATASET SPLITTING (Train 80% / Val 10% / Test 10%)
# ============================================================
def prepare_dataset_split(source_dir, output_dir, train_ratio, val_ratio,
                           test_ratio, valid_extensions, seed=42):
    """
    Reads every class sub-folder inside `source_dir` (e.g. "good",
    "defective"), shuffles the images with a fixed seed, and COPIES them
    into output_dir/train/<class>, output_dir/val/<class>,
    output_dir/test/<class>.

    The original dataset is never modified - only copied.
    If the split already exists (marker file found), this function does
    nothing, so re-running train.py is fast and safe.
    """
    marker_file = os.path.join(output_dir, ".split_complete")
    if os.path.exists(marker_file):
        print_info(f"Dataset split already exists at '{output_dir}'. Skipping re-split.")
        return

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Dataset source directory not found:\n  {source_dir}\n"
            f"Please open config.py and fix SOURCE_DATASET_DIR."
        )

    class_names = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    if len(class_names) == 0:
        raise ValueError(f"No class sub-folders were found inside: {source_dir}")

    print_info(f"Detected classes from dataset folders: {class_names}")

    rng = random.Random(seed)

    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        images = [
            f for f in os.listdir(class_dir)
            if f.endswith(valid_extensions)
        ]
        rng.shuffle(images)

        n_total = len(images)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # whatever remains goes to test (avoids rounding drift)
        n_test = n_total - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for fname in files:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(split_class_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        print_success(
            f"Class '{class_name}': {n_total} images -> "
            f"{len(train_files)} train / {len(val_files)} val / {len(test_files)} test"
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(marker_file, "w") as f:
        f.write("done")

    print_success(f"Dataset split complete. Saved to: {output_dir}")


# ============================================================
# MODEL BUILDING (MobileNetV3-Large, transfer learning)
# ============================================================
def build_model(num_classes=2, freeze_backbone=True, pretrained=True):
    """
    Builds a MobileNetV3-Large model.

    - pretrained=True downloads ImageNet weights automatically the first
      time it is run (torchvision handles the download + local caching).
    - The final classifier layer is replaced with a new Linear layer that
      outputs `num_classes` scores (2 for Good / Defective).
    - freeze_backbone=True freezes every convolutional feature-extraction
      layer so only the new classifier head is trained initially.
    """
    if pretrained:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = models.mobilenet_v3_large(weights=weights)

    # model.classifier is:
    #   Sequential(Linear(960,1280), Hardswish(), Dropout(0.2), Linear(1280,1000))
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


def unfreeze_backbone(model):
    """Unfreezes every layer of the backbone so fine-tuning can begin."""
    for param in model.features.parameters():
        param.requires_grad = True
    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# CLASS WEIGHTS (to counter mild dataset imbalance)
# ============================================================
def compute_class_weights(image_folder_dataset, num_classes, device):
    """
    Computes inverse-frequency class weights from an ImageFolder dataset so
    that CrossEntropyLoss pays proportionally more attention to the
    minority class.
    """
    counts = [0] * num_classes
    for _, label in image_folder_dataset.samples:
        counts[label] += 1

    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# LOADING A TRAINED MODEL (used by predict.py / export_onnx.py / test_model.py)
# ============================================================
def load_trained_model(model_path, device):
    """
    Loads the checkpoint saved by train.py (tyre_health_model.pt) and
    rebuilds the exact same MobileNetV3-Large architecture around it.
    Returns (model, checkpoint_dict).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Please run train.py first to create it."
        )

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), freeze_backbone=False, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def display_class_name(raw_name):
    """Turns a folder name like 'defective' into a display string 'Defective'."""
    return raw_name.strip().capitalize()
