"""
================================================================================
 EdgeGuard AI - Brake Health Prediction AI
================================================================================
This single script does EVERYTHING:
  1. Asks you for a CSV dataset path.
  2. Cleans and prepares the data automatically.
  3. Builds a Multi-Task Neural Network (PyTorch) that predicts 4 things
     at once from 11 sensor readings:
        - Brake Health (%)            -> regression
        - Remaining Pad Life (km)     -> regression
        - Brake Fade Risk             -> classification (Low/Medium/High)
        - Maintenance Action          -> classification (6 classes)
  4. Trains the network with early stopping, LR scheduling, gradient
     clipping, checkpointing and automatic resume.
  5. Saves graphs, reports and the trained model into an "outputs" folder.
  6. Lets you type in sensor readings and get an instant prediction.

Everything below is heavily commented so a complete beginner can follow
along. Just run:

    python train.py

and answer the questions it asks you in the terminal.
================================================================================
"""

# ------------------------------------------------------------------------
# STEP 0: IMPORTS
# ------------------------------------------------------------------------
# Standard library
import os
import sys
import time
import random
import warnings

# Data handling
import numpy as np
import pandas as pd

# Machine learning helpers (only used for splitting/scaling/encoding/metrics,
# NOT for the neural network itself - that part is pure PyTorch)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score

# Plotting
import matplotlib
matplotlib.use("Agg")  # Draw graphs without needing a screen/display
import matplotlib.pyplot as plt

# PyTorch - the actual deep learning library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Nice progress bars in the terminal
from tqdm import tqdm

# Colored terminal text so the output is easy to read
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

warnings.filterwarnings("ignore")  # Hide noisy library warnings for a clean beginner-friendly console


# ------------------------------------------------------------------------
# STEP 1: REPRODUCIBILITY (RANDOM SEED)
# ------------------------------------------------------------------------
# Setting a fixed "seed" means every time you run this script, the random
# numbers used for shuffling/splitting/initializing weights come out the
# same way, so results are repeatable.
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ------------------------------------------------------------------------
# STEP 2: DEVICE DETECTION (GPU IF AVAILABLE, ELSE CPU)
# ------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.CYAN + Style.BRIGHT + f"\n[INFO] Using device: {DEVICE}\n")


# ------------------------------------------------------------------------
# STEP 3: CONSTANTS - COLUMN NAMES
# ------------------------------------------------------------------------
# These are the 11 sensor readings the model looks at to make a prediction.
INPUT_FEATURES = [
    "Brake_Pad_Thickness_mm",
    "Brake_Disc_Temperature_C",
    "Brake_Fluid_Level_percent",
    "Brake_Fluid_Temperature_C",
    "Hydraulic_Pressure_bar",
    "Vehicle_Speed_kmph",
    "Wheel_Speed_kmph",
    "Brake_Pedal_Force_percent",
    "Ambient_Temperature_C",
    "Total_Mileage_km",
    "Brake_Usage_Count",
]

# These are the 4 things we want the AI to predict.
TARGET_HEALTH = "Brake_Health_percent"          # regression
TARGET_PADLIFE = "Remaining_Pad_Life_km"        # regression
TARGET_FADE = "Brake_Fade_Risk"                 # classification (3 classes)
TARGET_MAINTENANCE = "Maintenance_Action"       # classification (6 classes)

FADE_CLASSES = ["Low", "Medium", "High"]
MAINTENANCE_CLASSES = [
    "No Action",
    "Inspect Soon",
    "Replace Pads",
    "Replace Disc",
    "Replace Fluid",
    "Immediate Service",
]

# Human friendly prompts used later when we ask the user for sensor values.
FEATURE_PROMPTS = [
    "Brake Pad Thickness (mm)",
    "Brake Disc Temperature (C)",
    "Brake Fluid Level (%)",
    "Brake Fluid Temperature (C)",
    "Hydraulic Pressure (bar)",
    "Vehicle Speed (km/h)",
    "Wheel Speed (km/h)",
    "Brake Pedal Force (%)",
    "Ambient Temperature (C)",
    "Total Mileage (km)",
    "Brake Usage Count",
]

# ------------------------------------------------------------------------
# STEP 4: TRAINING HYPER-PARAMETERS
# ------------------------------------------------------------------------
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15      # stop if val loss doesn't improve for this many epochs
GRADIENT_CLIP_NORM = 5.0
DROPOUT_RATE = 0.3

OUTPUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_brake_model.pt")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "latest_checkpoint.pt")
HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")
LOSS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "loss_graph.png")
ACCURACY_GRAPH_PATH = os.path.join(OUTPUT_DIR, "accuracy_graph.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")
CONFUSION_FADE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_brake_fade.png")
CONFUSION_MAINTENANCE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_maintenance.png")


# ------------------------------------------------------------------------
# STEP 5: ASK THE USER FOR THE CSV DATASET PATH
# ------------------------------------------------------------------------
def ask_for_csv_path() -> str:
    """Keep asking the user until they give a path to a file that exists."""
    while True:
        path = input(Fore.YELLOW + "Enter CSV dataset path: " + Style.RESET_ALL).strip().strip('"')
        if os.path.isfile(path):
            return path
        print(Fore.RED + f"[ERROR] File not found: '{path}'. Please try again.")


# ------------------------------------------------------------------------
# STEP 6: LOAD + CLEAN THE DATASET
# ------------------------------------------------------------------------
def load_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    print(Fore.CYAN + f"\n[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_columns = INPUT_FEATURES + [TARGET_HEALTH, TARGET_PADLIFE, TARGET_FADE, TARGET_MAINTENANCE]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        print(Fore.RED + f"[ERROR] The CSV is missing required columns: {missing_columns}")
        sys.exit(1)

    # --- Handle missing values ---
    # Numeric columns: fill any blank cells with the column median (a safe, robust average).
    numeric_columns = INPUT_FEATURES + [TARGET_HEALTH, TARGET_PADLIFE]
    for col in numeric_columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Categorical target columns: fill any blank cells with the most common value (the "mode").
    for col in [TARGET_FADE, TARGET_MAINTENANCE]:
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_value)

    # Drop any row that is still broken beyond repair (all-blank row, etc.)
    before_rows = len(df)
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    after_rows = len(df)
    if after_rows < before_rows:
        print(Fore.YELLOW + f"[INFO] Dropped {before_rows - after_rows} unusable rows during cleaning.")

    print(Fore.GREEN + f"[OK] Dataset loaded and cleaned: {after_rows} rows ready for training.\n")
    return df


# ------------------------------------------------------------------------
# STEP 7: PYTORCH DATASET WRAPPER
# ------------------------------------------------------------------------
class BrakeDataset(Dataset):
    """
    Wraps our numpy arrays into something PyTorch's DataLoader can iterate
    over in batches. Each item returned is one full training example:
    (input_features, brake_health, pad_life, fade_label, maintenance_label)
    """

    def __init__(self, X, y_health, y_padlife, y_fade, y_maintenance):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_health = torch.tensor(y_health, dtype=torch.float32).unsqueeze(1)
        self.y_padlife = torch.tensor(y_padlife, dtype=torch.float32).unsqueeze(1)
        self.y_fade = torch.tensor(y_fade, dtype=torch.long)
        self.y_maintenance = torch.tensor(y_maintenance, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_health[idx],
            self.y_padlife[idx],
            self.y_fade[idx],
            self.y_maintenance[idx],
        )


# ------------------------------------------------------------------------
# STEP 8: THE MULTI-TASK NEURAL NETWORK
# ------------------------------------------------------------------------
class MultiTaskBrakeNet(nn.Module):
    """
    A feed-forward (fully connected) neural network with ONE shared "trunk"
    that learns general patterns from the sensor data, and FOUR separate
    small "heads" that each specialise in one prediction task.

    Shared trunk:  11 -> 256 -> 128 -> 64   (Linear + BatchNorm + ReLU + Dropout)
    Heads:
        - brake_health_head : 64 -> 32 -> 1   (regression)
        - pad_life_head     : 64 -> 32 -> 1   (regression)
        - fade_head         : 64 -> 32 -> 3   (classification logits)
        - maintenance_head  : 64 -> 32 -> 6   (classification logits)
    """

    def __init__(self, input_size: int, dropout: float = DROPOUT_RATE):
        super().__init__()

        # --- Shared trunk: learns general representations of the sensor data ---
        self.trunk = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Task-specific heads ---
        self.brake_health_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.pad_life_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.fade_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(FADE_CLASSES))
        )
        self.maintenance_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(MAINTENANCE_CLASSES))
        )

    def forward(self, x):
        shared = self.trunk(x)
        health_out = self.brake_health_head(shared)
        padlife_out = self.pad_life_head(shared)
        fade_logits = self.fade_head(shared)
        maintenance_logits = self.maintenance_head(shared)
        return health_out, padlife_out, fade_logits, maintenance_logits


# ------------------------------------------------------------------------
# STEP 9: EVALUATION HELPER (used for validation AND final testing)
# ------------------------------------------------------------------------
def evaluate(model, loader, mse_loss_fn, ce_loss_fn, health_scaler, padlife_scaler):
    """
    Runs the model over a whole dataloader WITHOUT training it (no gradient
    updates), and returns the average loss plus useful metrics.
    """
    model.eval()
    total_loss = 0.0

    all_health_pred, all_health_true = [], []
    all_padlife_pred, all_padlife_true = [], []
    all_fade_pred, all_fade_true = [], []
    all_maint_pred, all_maint_true = [], []

    with torch.no_grad():
        for X, y_health, y_padlife, y_fade, y_maint in loader:
            X = X.to(DEVICE)
            y_health = y_health.to(DEVICE)
            y_padlife = y_padlife.to(DEVICE)
            y_fade = y_fade.to(DEVICE)
            y_maint = y_maint.to(DEVICE)

            health_out, padlife_out, fade_logits, maint_logits = model(X)

            loss_health = mse_loss_fn(health_out, y_health)
            loss_padlife = mse_loss_fn(padlife_out, y_padlife)
            loss_fade = ce_loss_fn(fade_logits, y_fade)
            loss_maint = ce_loss_fn(maint_logits, y_maint)
            loss = loss_health + loss_padlife + loss_fade + loss_maint
            total_loss += loss.item() * X.size(0)

            all_health_pred.append(health_out.cpu().numpy())
            all_health_true.append(y_health.cpu().numpy())
            all_padlife_pred.append(padlife_out.cpu().numpy())
            all_padlife_true.append(y_padlife.cpu().numpy())
            all_fade_pred.append(torch.argmax(fade_logits, dim=1).cpu().numpy())
            all_fade_true.append(y_fade.cpu().numpy())
            all_maint_pred.append(torch.argmax(maint_logits, dim=1).cpu().numpy())
            all_maint_true.append(y_maint.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    # Undo the scaling so MAE is reported in REAL units (percent / km), not
    # in the abstract "scaled" numbers the network actually trains on.
    health_pred = health_scaler.inverse_transform(np.concatenate(all_health_pred))
    health_true = health_scaler.inverse_transform(np.concatenate(all_health_true))
    padlife_pred = padlife_scaler.inverse_transform(np.concatenate(all_padlife_pred))
    padlife_true = padlife_scaler.inverse_transform(np.concatenate(all_padlife_true))

    fade_pred = np.concatenate(all_fade_pred)
    fade_true = np.concatenate(all_fade_true)
    maint_pred = np.concatenate(all_maint_pred)
    maint_true = np.concatenate(all_maint_true)

    metrics = {
        "loss": avg_loss,
        "health_mae": mean_absolute_error(health_true, health_pred),
        "padlife_mae": mean_absolute_error(padlife_true, padlife_pred),
        "fade_acc": accuracy_score(fade_true, fade_pred),
        "maint_acc": accuracy_score(maint_true, maint_pred),
        # Raw arrays are also returned so we can build reports/confusion matrices later.
        "fade_true": fade_true, "fade_pred": fade_pred,
        "maint_true": maint_true, "maint_pred": maint_pred,
    }
    return metrics


# ------------------------------------------------------------------------
# STEP 9b: ASK WHETHER TO CONTINUE ON A NEW DATASET USING PREVIOUS WEIGHTS
# ------------------------------------------------------------------------
def ask_load_previous_weights() -> bool:
    """
    Ask the user whether they want to start this session using the model
    weights saved in best_brake_model.pt (e.g. because they are training on
    a brand-new dataset and want to fine-tune from what the model already
    learned), or whether they want to start completely from scratch.

    Y -> Load ONLY the model weights from best_brake_model.pt. The best
         validation metric, early-stopping counter, current epoch, optimizer
         state and scheduler state are ALL reset, so this new dataset is
         trained as a brand-new session and its validation results are
         never compared against the OLD dataset's best validation score.
    N -> Start from a freshly initialised (untrained) model.
    """
    while True:
        answer = input(
            Fore.CYAN + Style.BRIGHT + "\nLoad previous model weights? (Y/N): " + Style.RESET_ALL
        ).strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print(Fore.RED + "[ERROR] Please type Y or N.")


# ------------------------------------------------------------------------
# STEP 10: MAIN TRAINING PIPELINE
# ------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Ask up front whether this is a fresh/new-dataset session that ----
    # ---- should reuse previous weights, or a completely clean start.   ----
    load_previous_weights = ask_load_previous_weights()

    # ---- Load data ----
    csv_path = ask_for_csv_path()
    df = load_and_clean_dataset(csv_path)

    # ---- Encode categorical targets into numbers (0, 1, 2, ...) ----
    fade_encoder = LabelEncoder()
    fade_encoder.fit(FADE_CLASSES)
    maintenance_encoder = LabelEncoder()
    maintenance_encoder.fit(MAINTENANCE_CLASSES)

    # Some rows might use slightly different casing/spacing - normalise text first.
    df[TARGET_FADE] = df[TARGET_FADE].astype(str).str.strip()
    df[TARGET_MAINTENANCE] = df[TARGET_MAINTENANCE].astype(str).str.strip()

    # Any label not in our known class list gets mapped to the most common known class,
    # so unexpected text in the CSV can never crash the script.
    df[TARGET_FADE] = df[TARGET_FADE].apply(lambda v: v if v in FADE_CLASSES else FADE_CLASSES[0])
    df[TARGET_MAINTENANCE] = df[TARGET_MAINTENANCE].apply(
        lambda v: v if v in MAINTENANCE_CLASSES else MAINTENANCE_CLASSES[0]
    )

    y_fade_all = fade_encoder.transform(df[TARGET_FADE])
    y_maint_all = maintenance_encoder.transform(df[TARGET_MAINTENANCE])

    X_all = df[INPUT_FEATURES].values.astype(np.float32)
    y_health_all = df[TARGET_HEALTH].values.astype(np.float32).reshape(-1, 1)
    y_padlife_all = df[TARGET_PADLIFE].values.astype(np.float32).reshape(-1, 1)

    # ---- Split: 80% train / 10% validation / 10% test ----
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.10, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1111111, random_state=SEED, shuffle=True)

    # ---- Normalise (scale) INPUT features using ONLY training data statistics ----
    # This avoids "data leakage" - the model should never see information from
    # validation/test data during the fitting of the scaler.
    input_scaler = StandardScaler()
    input_scaler.fit(X_all[train_idx])
    X_scaled_all = input_scaler.transform(X_all).astype(np.float32)

    # ---- Also scale the two REGRESSION targets (helps the network train faster/better) ----
    health_scaler = StandardScaler()
    health_scaler.fit(y_health_all[train_idx])
    y_health_scaled_all = health_scaler.transform(y_health_all).astype(np.float32).flatten()

    padlife_scaler = StandardScaler()
    padlife_scaler.fit(y_padlife_all[train_idx])
    y_padlife_scaled_all = padlife_scaler.transform(y_padlife_all).astype(np.float32).flatten()

    def subset(idx):
        return (
            X_scaled_all[idx],
            y_health_scaled_all[idx],
            y_padlife_scaled_all[idx],
            y_fade_all[idx],
            y_maint_all[idx],
        )

    train_data = BrakeDataset(*subset(train_idx))
    val_data = BrakeDataset(*subset(val_idx))
    test_data = BrakeDataset(*subset(test_idx))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(Fore.GREEN + f"[OK] Split sizes -> Train: {len(train_data)} | "
          f"Validation: {len(val_data)} | Test: {len(test_data)}\n")

    # ---- Build the model ----
    model = MultiTaskBrakeNet(input_size=len(INPUT_FEATURES)).to(DEVICE)

    # ---- Losses ----
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # ---- Optimizer + LR scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ---- Fresh training state ----
    # These are ALWAYS initialised to brand-new values first. They only get
    # overwritten below in the ONE specific case where we are resuming the
    # SAME run on the SAME dataset (existing latest_checkpoint.pt AND the
    # user did NOT ask to load previous weights for a new dataset).
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    if load_previous_weights:
        # ---- NEW DATASET workflow ----
        # Load ONLY the model weights from best_brake_model.pt. Do NOT touch
        # the optimizer/scheduler state and do NOT reuse the old dataset's
        # best_val_loss/epoch/early-stopping counter - those belong to a
        # different dataset and must never be compared against this one.
        if os.path.isfile(BEST_MODEL_PATH):
            print(Fore.YELLOW + f"[INFO] Loading model weights ONLY from '{BEST_MODEL_PATH}'...")
            weights_checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(weights_checkpoint["model_state_dict"])
            print(Fore.GREEN + "[OK] Pretrained weights loaded. Starting a brand-new training "
                                "session on the new dataset (best_val_loss, early-stopping "
                                "counter, epoch counter, optimizer state and scheduler state "
                                "have all been reset).\n")
        else:
            print(Fore.RED + f"[WARNING] '{BEST_MODEL_PATH}' not found. "
                              f"Starting instead from a freshly initialised model.\n")
    else:
        # ---- Normal behaviour: resume an interrupted run on the SAME dataset ----
        if os.path.isfile(CHECKPOINT_PATH):
            print(Fore.YELLOW + f"[INFO] Found existing checkpoint at '{CHECKPOINT_PATH}'. Resuming training...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            history = checkpoint.get("history", [])
            print(Fore.GREEN + f"[OK] Resumed at epoch {start_epoch}. Best val loss so far: {best_val_loss:.4f}\n")
        else:
            print(Fore.CYAN + "[INFO] No checkpoint found. Training a brand-new model from scratch.\n")

    if start_epoch > EPOCHS:
        print(Fore.YELLOW + "[INFO] Checkpoint already reached the target number of epochs. Skipping training.\n")

    # ---- Training loop ----
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=Fore.MAGENTA + f"Epoch {epoch}/{EPOCHS}" + Style.RESET_ALL,
            leave=False,
        )

        for X, y_health, y_padlife, y_fade, y_maint in progress_bar:
            X = X.to(DEVICE)
            y_health = y_health.to(DEVICE)
            y_padlife = y_padlife.to(DEVICE)
            y_fade = y_fade.to(DEVICE)
            y_maint = y_maint.to(DEVICE)

            optimizer.zero_grad()

            health_out, padlife_out, fade_logits, maint_logits = model(X)

            loss_health = mse_loss_fn(health_out, y_health)
            loss_padlife = mse_loss_fn(padlife_out, y_padlife)
            loss_fade = ce_loss_fn(fade_logits, y_fade)
            loss_maint = ce_loss_fn(maint_logits, y_maint)

            # Combine all four task losses into a single number to backpropagate.
            loss = loss_health + loss_padlife + loss_fade + loss_maint
            loss.backward()

            # Gradient clipping: stops the network from taking dangerously large
            # update steps, which keeps training stable.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

            optimizer.step()

            running_loss += loss.item() * X.size(0)
            progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, mse_loss_fn, ce_loss_fn, health_scaler, padlife_scaler)
        val_loss = val_metrics["loss"]

        # Learning rate scheduler looks at validation loss to decide if the LR should drop.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "brake_health_mae": val_metrics["health_mae"],
            "remaining_pad_life_mae": val_metrics["padlife_mae"],
            "brake_fade_accuracy": val_metrics["fade_acc"],
            "maintenance_accuracy": val_metrics["maint_acc"],
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        })

        # ---- Print epoch summary ----
        print(
            Fore.CYAN + f"Epoch {epoch:3d}/{EPOCHS} | "
            + Fore.WHITE + f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            + Fore.GREEN + f"Health MAE: {val_metrics['health_mae']:.3f}% | "
            + f"PadLife MAE: {val_metrics['padlife_mae']:.2f}km | "
            + Fore.YELLOW + f"Fade Acc: {val_metrics['fade_acc']*100:.2f}% | "
            + f"Maint Acc: {val_metrics['maint_acc']*100:.2f}% | "
            + Fore.MAGENTA + f"LR: {current_lr:.6f} | "
            + Fore.WHITE + f"Time: {epoch_time:.2f}s"
        )

        # ---- Checkpointing ----
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_scaler": input_scaler,
                "health_scaler": health_scaler,
                "padlife_scaler": padlife_scaler,
                "fade_encoder": fade_encoder,
                "maintenance_encoder": maintenance_encoder,
                "input_features": INPUT_FEATURES,
            }, BEST_MODEL_PATH)
            print(Fore.GREEN + f"  -> New best model saved (val loss improved to {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Save the "latest" checkpoint every epoch so training can resume later.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }, CHECKPOINT_PATH)

        # ---- Early stopping ----
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(Fore.RED + f"\n[INFO] No improvement for {EARLY_STOPPING_PATIENCE} epochs. Stopping early.\n")
            break

    print(Fore.GREEN + Style.BRIGHT + "\n[OK] Training finished.\n")

    # ---- Save training history to CSV ----
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(Fore.GREEN + f"[OK] Saved training history -> {HISTORY_CSV_PATH}")

    # ---- Plot loss graph ----
    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH_PATH)
    plt.close()
    print(Fore.GREEN + f"[OK] Saved loss graph -> {LOSS_GRAPH_PATH}")

    # ---- Plot accuracy graph ----
    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["brake_fade_accuracy"] * 100, label="Brake Fade Accuracy (%)")
    plt.plot(history_df["epoch"], history_df["maintenance_accuracy"] * 100, label="Maintenance Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_GRAPH_PATH)
    plt.close()
    print(Fore.GREEN + f"[OK] Saved accuracy graph -> {ACCURACY_GRAPH_PATH}")

    # ---- Load the BEST model for final testing ----
    best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # ---- Final evaluation on the untouched TEST set ----
    test_metrics = evaluate(model, test_loader, mse_loss_fn, ce_loss_fn, health_scaler, padlife_scaler)

    print(Fore.CYAN + Style.BRIGHT + "\n[TEST SET RESULTS]")
    print(Fore.WHITE + f"  Brake Health MAE      : {test_metrics['health_mae']:.3f} %")
    print(Fore.WHITE + f"  Remaining Pad Life MAE: {test_metrics['padlife_mae']:.2f} km")
    print(Fore.WHITE + f"  Brake Fade Accuracy   : {test_metrics['fade_acc']*100:.2f} %")
    print(Fore.WHITE + f"  Maintenance Accuracy  : {test_metrics['maint_acc']*100:.2f} %\n")

    # ---- Classification report (both classifiers) saved to one text file ----
    # NOTE: We auto-detect which class labels are actually present in the true
    # values instead of assuming every class exists. This prevents a crash
    # when the test split happens to be missing one or more classes.
    fade_labels_present = sorted(np.unique(test_metrics["fade_true"]))
    fade_target_names_present = [fade_encoder.inverse_transform([lbl])[0] for lbl in fade_labels_present]

    maint_labels_present = sorted(np.unique(test_metrics["maint_true"]))
    maint_target_names_present = [maintenance_encoder.inverse_transform([lbl])[0] for lbl in maint_labels_present]

    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("=== Brake Fade Risk - Classification Report ===\n")
        f.write(classification_report(
            test_metrics["fade_true"], test_metrics["fade_pred"],
            labels=fade_labels_present, target_names=fade_target_names_present, zero_division=0,
        ))
        f.write("\n\n=== Maintenance Action - Classification Report ===\n")
        f.write(classification_report(
            test_metrics["maint_true"], test_metrics["maint_pred"],
            labels=maint_labels_present, target_names=maint_target_names_present, zero_division=0,
        ))
    print(Fore.GREEN + f"[OK] Saved classification report -> {CLASSIFICATION_REPORT_PATH}")

    # ---- Confusion matrices (both classifiers), each as its own PNG ----
    def save_confusion_matrix(y_true, y_pred, class_names, title, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    save_confusion_matrix(
        test_metrics["fade_true"], test_metrics["fade_pred"], FADE_CLASSES,
        "Brake Fade Risk - Confusion Matrix", CONFUSION_FADE_PATH,
    )
    print(Fore.GREEN + f"[OK] Saved confusion matrix -> {CONFUSION_FADE_PATH}")

    save_confusion_matrix(
        test_metrics["maint_true"], test_metrics["maint_pred"], MAINTENANCE_CLASSES,
        "Maintenance Action - Confusion Matrix", CONFUSION_MAINTENANCE_PATH,
    )
    print(Fore.GREEN + f"[OK] Saved confusion matrix -> {CONFUSION_MAINTENANCE_PATH}\n")

    # ---- Interactive inference ----
    run_interactive_inference(model, input_scaler, health_scaler, padlife_scaler,
                               fade_encoder, maintenance_encoder)


# ------------------------------------------------------------------------
# STEP 11: INTERACTIVE INFERENCE (ASK THE USER FOR SENSOR VALUES)
# ------------------------------------------------------------------------
def ask_float(prompt: str) -> float:
    """Keep asking until the user types a valid number."""
    while True:
        raw = input(Fore.YELLOW + f"{prompt}: " + Style.RESET_ALL).strip()
        try:
            return float(raw)
        except ValueError:
            print(Fore.RED + "[ERROR] Please enter a valid number.")


def run_interactive_inference(model, input_scaler, health_scaler, padlife_scaler,
                               fade_encoder, maintenance_encoder):
    answer = input(Fore.CYAN + "\nDo you want to test the model? (Y/N): " + Style.RESET_ALL).strip().lower()
    if answer != "y":
        print(Fore.CYAN + "\n[INFO] Skipping manual testing. Done!\n")
        return

    model.eval()
    while True:
        print(Fore.CYAN + Style.BRIGHT + "\nEnter the brake sensor readings one at a time:")
        values = [ask_float(prompt) for prompt in FEATURE_PROMPTS]

        # Prepare the input the exact same way the training data was prepared:
        # same column order, then scaled with the SAME scaler fitted on training data.
        raw_vector = np.array(values, dtype=np.float32).reshape(1, -1)
        scaled_vector = input_scaler.transform(raw_vector).astype(np.float32)
        input_tensor = torch.tensor(scaled_vector, dtype=torch.float32).to(DEVICE)

        start_time = time.perf_counter()
        with torch.no_grad():
            health_out, padlife_out, fade_logits, maint_logits = model(input_tensor)

            fade_probs = torch.softmax(fade_logits, dim=1)
            maint_probs = torch.softmax(maint_logits, dim=1)

            fade_confidence, fade_idx = torch.max(fade_probs, dim=1)
            maint_confidence, maint_idx = torch.max(maint_probs, dim=1)
        inference_time = (time.perf_counter() - start_time) * 1000.0  # milliseconds

        # Undo scaling on the regression outputs to get real-world numbers back.
        brake_health = health_scaler.inverse_transform(health_out.cpu().numpy())[0][0]
        pad_life = padlife_scaler.inverse_transform(padlife_out.cpu().numpy())[0][0]

        fade_label = fade_encoder.inverse_transform([fade_idx.item()])[0]
        maint_label = maintenance_encoder.inverse_transform([maint_idx.item()])[0]

        overall_confidence = (fade_confidence.item() + maint_confidence.item()) / 2.0

        print(Fore.GREEN + Style.BRIGHT + "\n===== PREDICTION RESULT =====")
        print(Fore.WHITE + f"  Brake Health           : {brake_health:.2f} %")
        print(Fore.WHITE + f"  Remaining Pad Life     : {pad_life:.2f} km")
        print(Fore.WHITE + f"  Brake Fade Risk        : {fade_label}  (confidence: {fade_confidence.item()*100:.2f}%)")
        print(Fore.WHITE + f"  Maintenance Action     : {maint_label}  (confidence: {maint_confidence.item()*100:.2f}%)")
        print(Fore.WHITE + f"  Overall Confidence     : {overall_confidence*100:.2f}%")
        print(Fore.WHITE + f"  Inference Time         : {inference_time:.3f} ms")
        print(Fore.GREEN + Style.BRIGHT + "==============================\n")

        again = input(Fore.CYAN + "Test another reading? (Y/N): " + Style.RESET_ALL).strip().lower()
        if again != "y":
            print(Fore.CYAN + "\n[INFO] Done. Goodbye!\n")
            break


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
