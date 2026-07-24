"""
config.py
=========
Central configuration file for the EdgeGuard AI - Tyre Health Prediction project.

Every other script (train.py, predict.py, export_onnx.py, test_model.py) imports
its settings from this file. If you need to change the dataset location, image
size, number of epochs, or any output path, change it HERE ONLY and every script
will automatically use the new value.
"""

import os

# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================
# Path to your ORIGINAL dataset (the one you already have, with
# "good" and "defective" sub-folders). Change this if your dataset
# is located somewhere else.
SOURCE_DATASET_DIR = r"D:\Tata Hackathon\Tyre AI Stable v1\dataset\online dataset"

# train.py will AUTOMATICALLY create a new folder here containing the
# 80% / 10% / 10% train / val / test split (it copies images, it does
# NOT touch or modify your original dataset).
SPLIT_DATASET_DIR = r"D:\Tata Hackathon\Tyre AI Stable v1\dataset\split_dataset"

TRAIN_DIR = os.path.join(SPLIT_DATASET_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DATASET_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DATASET_DIR, "test")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Valid image extensions to look for inside the dataset folders
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

# ============================================================
# 2. MODEL / TRAINING CONFIGURATION
# ============================================================
IMAGE_SIZE = 224                 # MobileNetV3 expects 224x224 images
BATCH_SIZE = 32
NUM_EPOCHS = 25
FREEZE_EPOCHS = 5                # Number of epochs to train ONLY the classifier
                                  # head (backbone frozen) before unfreezing the
                                  # backbone for fine-tuning.
LEARNING_RATE_HEAD = 1e-3        # LR used while the backbone is frozen
LEARNING_RATE_BACKBONE = 1e-5    # LR used for the backbone once unfrozen
LEARNING_RATE_FINETUNE_HEAD = 1e-4  # LR used for the head once fine-tuning starts
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7      # Stop training if val accuracy doesn't improve
                                  # for this many consecutive epochs
LR_SCHEDULER_PATIENCE = 3        # Reduce LR if val loss plateaus for this many epochs
LR_SCHEDULER_FACTOR = 0.5
NUM_WORKERS = 2                  # DataLoader worker processes (safe on Windows
                                  # because scripts are wrapped in
                                  # `if __name__ == "__main__":`)
RANDOM_SEED = 42
USE_CLASS_WEIGHTS = True         # Automatically balances the loss for the
                                  # slightly imbalanced Good/Defective classes

# ============================================================
# 3. OUTPUT PATHS
# ============================================================
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(OUTPUT_DIR, "tyre_health_model.pt")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_last.pt")
ONNX_PATH = os.path.join(OUTPUT_DIR, "tyre_health_model.onnx")

HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")
ACCURACY_PLOT_PATH = os.path.join(OUTPUT_DIR, "accuracy_graph.png")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_graph.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")

TEST_CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "test_confusion_matrix.png")
TEST_REPORT_PATH = os.path.join(OUTPUT_DIR, "test_classification_report.txt")

# ============================================================
# 4. IMAGE NET NORMALIZATION CONSTANTS
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
