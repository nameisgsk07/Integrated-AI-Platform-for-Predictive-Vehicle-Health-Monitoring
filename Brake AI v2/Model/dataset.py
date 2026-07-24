"""
dataset.py
==========
Data loading and preprocessing for the EdgeGuard AI Brake Health Prediction
framework.

Responsibilities:
    - Load an arbitrary CSV file (path supplied by the user, never hardcoded).
    - Automatically detect which columns correspond to the eleven input
      sensor features, the two regression targets and the two
      classification targets, using flexible keyword matching so the
      framework tolerates naming variations between datasets.
    - Normalize inputs with StandardScaler and encode class labels with
      LabelEncoder, persisting both to disk for later reuse at inference.
    - Split the dataset into train / validation / test partitions.
"""

import os
import re
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset

from config import EdgeGuardConfig
from utils import log_info, log_success, log_warning, save_json


def _normalize(name: str) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(columns: List[str], keywords: List[str]) -> str:
    """Finds the first dataframe column whose normalized name contains, or is
    contained by, one of the normalized keyword hints."""
    normalized_columns = {col: _normalize(col) for col in columns}
    normalized_keywords = [_normalize(k) for k in keywords]

    for col, norm_col in normalized_columns.items():
        for norm_kw in normalized_keywords:
            if norm_kw in norm_col or norm_col in norm_kw:
                return col

    raise ValueError(
        f"Could not automatically detect a column matching any of: {keywords}. "
        f"Available columns are: {columns}"
    )


class DetectedColumns:
    """Holds the mapping from canonical feature/target names to the actual
    column names found in the user's CSV file."""

    def __init__(self, input_columns: Dict[str, str],
                 regression_columns: Dict[str, str],
                 classification_columns: Dict[str, str]):
        self.input_columns = input_columns
        self.regression_columns = regression_columns
        self.classification_columns = classification_columns

    @property
    def ordered_input_names(self) -> List[str]:
        return list(self.input_columns.keys())

    @property
    def ordered_input_source_columns(self) -> List[str]:
        return list(self.input_columns.values())


def detect_columns(df: pd.DataFrame, config: EdgeGuardConfig) -> DetectedColumns:
    """Automatically maps canonical names to the actual CSV column names."""
    columns = list(df.columns)

    input_columns = {}
    for canonical_name, keywords in config.data.input_feature_keywords.items():
        input_columns[canonical_name] = _find_column(columns, keywords)

    regression_columns = {}
    for canonical_name, keywords in config.data.regression_target_keywords.items():
        regression_columns[canonical_name] = _find_column(columns, keywords)

    classification_columns = {}
    for canonical_name, keywords in config.data.classification_target_keywords.items():
        classification_columns[canonical_name] = _find_column(columns, keywords)

    log_success("Automatic column detection complete:")
    for canonical, source in {**input_columns, **regression_columns, **classification_columns}.items():
        log_info(f"    {canonical:28s} <- '{source}'")

    return DetectedColumns(input_columns, regression_columns, classification_columns)


class BrakeHealthDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed brake sensor tensors."""

    def __init__(self, inputs: np.ndarray, brake_health: np.ndarray,
                 remaining_pad_life: np.ndarray, fade_risk: np.ndarray,
                 maintenance_action: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.brake_health = torch.tensor(brake_health, dtype=torch.float32)
        self.remaining_pad_life = torch.tensor(remaining_pad_life, dtype=torch.float32)
        self.fade_risk = torch.tensor(fade_risk, dtype=torch.long)
        self.maintenance_action = torch.tensor(maintenance_action, dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return {
            "inputs": self.inputs[idx],
            "brake_health": self.brake_health[idx],
            "remaining_pad_life": self.remaining_pad_life[idx],
            "fade_risk": self.fade_risk[idx],
            "maintenance_action": self.maintenance_action[idx],
        }


def load_and_prepare_data(csv_path: str, config: EdgeGuardConfig
                           ) -> Tuple[BrakeHealthDataset, BrakeHealthDataset, BrakeHealthDataset,
                                      StandardScaler, MinMaxScaler, LabelEncoder, LabelEncoder, DetectedColumns]:
    """Loads the CSV, detects columns, fits preprocessing objects, splits the
    data and returns ready-to-use train/val/test Datasets plus the fitted
    scaler and encoders (which are also persisted to disk)."""

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    log_info(f"Loading dataset from '{csv_path}' ...")
    df = pd.read_csv(csv_path)
    log_success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    detected = detect_columns(df, config)

    # Drop rows with missing values in any of the required columns.
    required_columns = (
        detected.ordered_input_source_columns
        + list(detected.regression_columns.values())
        + list(detected.classification_columns.values())
    )
    before = len(df)
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        log_warning(f"Dropped {dropped} rows containing missing values in required columns.")

    # --- Inputs ---
    X = df[detected.ordered_input_source_columns].to_numpy(dtype=np.float64)

    # --- Regression targets ---
    y_health = df[detected.regression_columns["brake_health_pct"]].to_numpy(dtype=np.float64)
    y_pad_life = df[detected.regression_columns["remaining_pad_life_km"]].to_numpy(dtype=np.float64)

    # --- Classification targets ---
    fade_encoder = LabelEncoder()
    y_fade = fade_encoder.fit_transform(
        df[detected.classification_columns["brake_fade_risk"]].astype(str)
    )

    maintenance_encoder = LabelEncoder()
    y_maintenance = maintenance_encoder.fit_transform(
        df[detected.classification_columns["maintenance_action"]].astype(str)
    )

    log_success(f"Brake Fade Risk classes: {list(fade_encoder.classes_)}")
    log_success(f"Maintenance Action classes: {list(maintenance_encoder.classes_)}")

    # --- Split: 80 / 10 / 10 ---
    seed = config.data.random_seed
    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=config.data.train_split,
        random_state=seed,
        shuffle=True,
    )
    relative_val_size = config.data.val_split / (config.data.val_split + config.data.test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    # --- Fit input scaler on TRAIN split only, then transform all splits ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    # --- Fit a target scaler for the two regression outputs -----------------
    # Brake Health (0-100) and Remaining Pad Life (0-50000) live on wildly
    # different numeric scales. Training directly on raw values makes the
    # shared multi-task loss dominated by whichever target has the larger
    # magnitude, starving the other of gradient signal. We normalize both
    # regression targets to a comparable [0, 1] range for training, and
    # inverse-transform back to physical units for evaluation, reporting
    # and inference. The scaler is fit on the TRAIN split only.
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_targets = np.column_stack([y_health[train_idx], y_pad_life[train_idx]])
    target_scaler.fit(train_targets)

    def _scale_targets(idx):
        raw = np.column_stack([y_health[idx], y_pad_life[idx]])
        scaled = target_scaler.transform(raw)
        return scaled[:, 0], scaled[:, 1]

    y_health_train_n, y_pad_life_train_n = _scale_targets(train_idx)
    y_health_val_n, y_pad_life_val_n = _scale_targets(val_idx)
    y_health_test_n, y_pad_life_test_n = _scale_targets(test_idx)

    train_dataset = BrakeHealthDataset(
        X_train, y_health_train_n, y_pad_life_train_n,
        y_fade[train_idx], y_maintenance[train_idx],
    )
    val_dataset = BrakeHealthDataset(
        X_val, y_health_val_n, y_pad_life_val_n,
        y_fade[val_idx], y_maintenance[val_idx],
    )
    test_dataset = BrakeHealthDataset(
        X_test, y_health_test_n, y_pad_life_test_n,
        y_fade[test_idx], y_maintenance[test_idx],
    )

    log_success(
        f"Dataset split -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    # --- Persist preprocessing artifacts ---
    output_dir = config.ensure_output_dir()
    joblib.dump(scaler, os.path.join(output_dir, config.data.scaler_filename))
    joblib.dump(target_scaler, os.path.join(output_dir, config.data.target_scaler_filename))
    joblib.dump(fade_encoder, os.path.join(output_dir, config.data.fade_encoder_filename))
    joblib.dump(maintenance_encoder, os.path.join(output_dir, config.data.maintenance_encoder_filename))
    save_json(detected.ordered_input_names, os.path.join(output_dir, config.data.feature_order_filename))
    log_success(f"Saved input scaler, target scaler, label encoders and feature order to '{output_dir}/'.")

    return (train_dataset, val_dataset, test_dataset, scaler, target_scaler,
            fade_encoder, maintenance_encoder, detected)
