"""
EV Motor Health Prediction — PyTorch Pipeline
================================================
Predicts a continuous "health score" (0-100, where 100 = perfectly healthy)
for an EV motor from six sensor readings:

    RPM, current (A), voltage (V), torque (N*m), motor temperature (C), battery SOC (%)

Includes:
    1. Data loading / synthetic data generation
    2. Preprocessing (scaling, train/val/test split)
    3. Model definition (MLP regressor)
    4. Training loop with validation + early stopping
    5. Model + scaler saving
    6. Inference on new samples

Usage
-----
    python ev_motor_health.py --mode train --csv path/to/data.csv
    python ev_motor_health.py --mode train                       # uses synthetic demo data
    python ev_motor_health.py --mode infer --csv path/to/new.csv

If --csv is omitted in train mode, synthetic data is generated so the script
runs end-to-end out of the box. Replace `generate_synthetic_data()` (or just
pass --csv) with your real logged data as soon as you have it.

Expected CSV columns (train mode):
    rpm, current, voltage, torque, motor_temp, battery_soc, health_score

Expected CSV columns (infer mode):
    rpm, current, voltage, torque, motor_temp, battery_soc
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

FEATURE_COLS = ["rpm", "current", "voltage", "torque", "motor_temp", "battery_soc"]
TARGET_COL = "health_score"

MODEL_PATH = "ev_motor_health_model.pt"
SCALER_PATH = "ev_motor_health_scaler.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Data loading / synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generates plausible EV motor telemetry with a hand-crafted health score,
    purely so the pipeline can be demonstrated end-to-end without real data.

    The "true" health score degrades with high temperature, high current draw
    relative to torque produced (inefficiency), and low battery SOC under load,
    with added noise. Replace this with real labeled data for production use.
    """
    rng = np.random.default_rng(seed)

    rpm = rng.uniform(500, 8000, n_samples)
    torque = rng.uniform(5, 150, n_samples)
    voltage = rng.uniform(280, 420, n_samples)
    battery_soc = rng.uniform(5, 100, n_samples)

    # Current roughly scales with torque and rpm-derived power demand, plus noise
    base_current = (torque * 1.8 + rpm * 0.01) / (voltage / 350)
    current = base_current + rng.normal(0, 4, n_samples)
    current = np.clip(current, 1, 400)

    # Motor temp rises with current and rpm, falls somewhat with better cooling (proxy: noise)
    motor_temp = 25 + current * 0.35 + rpm * 0.004 + rng.normal(0, 5, n_samples)
    motor_temp = np.clip(motor_temp, 20, 180)

    # Efficiency proxy: torque produced per unit current (higher is healthier)
    efficiency = torque / (current + 1e-3)

    # Construct a synthetic ground-truth health score
    health = (
        100
        - np.clip((motor_temp - 60), 0, None) * 0.6      # penalty above 60C
        - np.clip((current - 150), 0, None) * 0.15       # penalty for excess current
        - np.clip((20 - battery_soc), 0, None) * 0.5      # penalty for very low SOC
        + np.clip(efficiency - 0.5, -2, 2) * 5            # small bonus/penalty for efficiency
    )
    health += rng.normal(0, 3, n_samples)  # measurement/label noise
    health = np.clip(health, 0, 100)

    df = pd.DataFrame(
        {
            "rpm": rpm,
            "current": current,
            "voltage": voltage,
            "torque": torque,
            "motor_temp": motor_temp,
            "battery_soc": battery_soc,
            "health_score": health,
        }
    )
    return df


def load_training_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return df
    print("No --csv provided: generating synthetic demo data instead.")
    return generate_synthetic_data()


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
class StandardScalerTorch:
    """Minimal standard scaler (mean/std) with JSON save/load, no sklearn dependency."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        scaler = cls()
        scaler.mean_ = np.array(d["mean"])
        scaler.std_ = np.array(d["std"])
        return scaler


class MotorHealthDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloaders(df: pd.DataFrame, batch_size: int = 64, val_frac: float = 0.15,
                         test_frac: float = 0.15, seed: int = 42):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    scaler = StandardScalerTorch()
    X_scaled = scaler.fit_transform(X)

    full_dataset = MotorHealthDataset(X_scaled, y)

    n = len(full_dataset)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
class MotorHealthNet(nn.Module):
    """Simple MLP regressor: 6 sensor features -> 1 health score (0-100)."""

    def __init__(self, input_dim: int = len(FEATURE_COLS), hidden_dims=(64, 32, 16), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Sigmoid * 100 keeps output bounded to a valid health-score range
        raw = self.net(x)
        return torch.sigmoid(raw) * 100.0


# ---------------------------------------------------------------------------
# 4. Training loop with validation + early stopping
# ---------------------------------------------------------------------------
def train_model(train_loader, val_loader, epochs: int = 100, lr: float = 1e-3,
                 patience: int = 10):
    model = MotorHealthNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                val_mae += torch.abs(preds - yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | train MSE {train_loss:7.3f} | val MSE {val_loss:7.3f} | val MAE {val_mae:6.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate(model, loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            total_loss += criterion(preds, yb).item() * xb.size(0)
            total_mae += torch.abs(preds - yb).sum().item()
            n += xb.size(0)
    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# 5. Saving
# ---------------------------------------------------------------------------
def save_artifacts(model: nn.Module, scaler: StandardScalerTorch,
                    model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    torch.save(model.state_dict(), model_path)
    scaler.save(scaler_path)
    print(f"Saved model to '{model_path}' and scaler to '{scaler_path}'.")


# ---------------------------------------------------------------------------
# 6. Inference
# ---------------------------------------------------------------------------
def load_artifacts(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    model = MotorHealthNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    scaler = StandardScalerTorch.load(scaler_path)
    return model, scaler


def predict(model, scaler, samples: np.ndarray) -> np.ndarray:
    """
    samples: array of shape (n, 6) in the order:
             [rpm, current, voltage, torque, motor_temp, battery_soc]
    returns: array of shape (n,) with predicted health scores (0-100)
    """
    X_scaled = scaler.transform(samples.astype(np.float32))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    return preds


def health_label(score: float) -> str:
    if score >= 80:
        return "Healthy"
    elif score >= 50:
        return "Warning"
    else:
        return "Critical"


def run_inference_on_csv(csv_path: str):
    model, scaler = load_artifacts()
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    preds = predict(model, scaler, df[FEATURE_COLS].values)
    df["predicted_health_score"] = preds
    df["status"] = [health_label(p) for p in preds]
    print(df[FEATURE_COLS + ["predicted_health_score", "status"]].to_string(index=False))
    return df


def run_inference_demo():
    """Fallback demo when no CSV is given for inference: predicts on a few hand-picked samples."""
    model, scaler = load_artifacts()
    demo_samples = np.array(
        [
            [3000, 60, 350, 40, 45, 80],      # Normal
            [9000, 350, 200, 180, 170, 10],   # Very unhealthy
            [8000, 180, 1500, 110, 98, 500],   # Extremely unhealthy
        ]
    )
    preds = predict(model, scaler, demo_samples)
    for sample, pred in zip(demo_samples, preds):
        print(f"Input {sample.tolist()} -> predicted health score: {pred:.1f} ({health_label(pred)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EV Motor Health Prediction Pipeline")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to CSV. Train mode needs feature+target cols; infer mode needs feature cols only.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "train":
        df = load_training_data(args.csv)
        train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
            df, batch_size=args.batch_size
        )

        print(f"Training on {len(train_loader.dataset)} samples, "
              f"validating on {len(val_loader.dataset)}, "
              f"testing on {len(test_loader.dataset)}. Device: {DEVICE}")

        model, best_val_loss = train_model(
            train_loader, val_loader, epochs=args.epochs, lr=args.lr
        )

        test_loss, test_mae = evaluate(model, test_loader)
        print(f"\nFinal test MSE: {test_loss:.3f} | test MAE: {test_mae:.3f}")

        save_artifacts(model, scaler)

    elif args.mode == "infer":
        if args.csv:
            run_inference_on_csv(args.csv)
        else:
            if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
                raise FileNotFoundError(
                    "No trained model found. Run with --mode train first."
                )
            run_inference_demo()


if __name__ == "__main__":
    main()
