"""
dataset.py
==========

Everything related to turning a raw, user-supplied CSV file into clean,
validated, scaled tensors ready for training or evaluation.

Responsibilities:
    - Load CSV (path supplied at runtime, never hardcoded)
    - Verify exact column contract (no extra / missing columns)
    - Detect and remove duplicates
    - Detect and handle missing values
    - Enforce per-feature physical range validation
    - Enforce cross-field physical plausibility rules
    - Detect statistical outliers (IQR-based) and clip/flag them
    - Stratified train / validation / test split (stratified on the
      maintenance-recommendation class, the rarer of the two label sets,
      to keep all splits representative)
    - Fit/apply feature scaling (StandardScaler) and regression target
      scaling (MinMaxScaler -- this is what fixed the multi-task loss
      scaling bug in the previous session: without it, the 0-100 range
      regression loss dominates the classification losses)
    - Label-encode the two classification targets
    - Expose a torch.utils.data.Dataset wrapper
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

import config


class DatasetValidationError(Exception):
    """Raised when the input CSV violates the fixed schema or is unusable."""


@dataclass
class CleaningReport:
    original_rows: int
    duplicates_removed: int
    missing_value_rows_removed: int
    range_violation_rows_removed: int
    plausibility_violation_rows_removed: int
    outlier_rows_clipped: int
    final_rows: int


def _validate_schema(df: pd.DataFrame) -> None:
    """Ensure the CSV has exactly the required columns (no more, no less)."""
    csv_columns = set(df.columns)
    expected_columns = set(config.ALL_CSV_COLUMNS)

    missing = expected_columns - csv_columns
    extra = csv_columns - expected_columns

    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Expected exactly: {config.ALL_CSV_COLUMNS}"
        )
    if extra:
        raise DatasetValidationError(
            f"Dataset contains unexpected extra columns: {sorted(extra)}. "
            f"Expected exactly: {config.ALL_CSV_COLUMNS}"
        )


def _apply_range_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any feature falls outside its physically valid range."""
    mask = pd.Series(True, index=df.index)
    for feature, (low, high) in config.FEATURE_RANGES.items():
        mask &= df[feature].between(low, high)
    return df[mask]


def _apply_plausibility_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that violate cross-field physical plausibility rules."""
    keep_mask = pd.Series(True, index=df.index)
    records = df[config.FEATURE_COLUMNS].to_dict(orient="records")
    for idx, row in zip(df.index, records):
        for _description, rule_fn in config.PHYSICAL_PLAUSIBILITY_RULES:
            if rule_fn(row):
                keep_mask.loc[idx] = False
                break
    return df[keep_mask]


def _clip_outliers_iqr(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> Tuple[pd.DataFrame, int]:
    """Clip extreme statistical outliers in feature columns using the IQR rule.

    We CLIP rather than drop, since a wide IQR multiplier (3.0, i.e. far
    beyond the conventional 1.5) is used to only catch truly extreme
    values while preserving as much real data as possible. Rows are not
    removed; only feature values are pulled back to the fence.
    """
    df = df.copy()
    num_clipped = 0
    for feature in config.FEATURE_COLUMNS:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_fence = q1 - iqr_multiplier * iqr
        upper_fence = q3 + iqr_multiplier * iqr
        # Also respect the hard physical range so we never clip *into* an
        # invalid region.
        hard_low, hard_high = config.FEATURE_RANGES[feature]
        lower_fence = max(lower_fence, hard_low)
        upper_fence = min(upper_fence, hard_high)

        out_of_bounds = (df[feature] < lower_fence) | (df[feature] > upper_fence)
        num_clipped += int(out_of_bounds.sum())
        df[feature] = df[feature].clip(lower=lower_fence, upper=upper_fence)
    return df, num_clipped


def load_and_clean_csv(csv_path: str, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, CleaningReport]:
    """Load a user-supplied CSV, validate its schema, and clean it.

    Returns the cleaned DataFrame plus a CleaningReport summarizing every
    action taken, so the caller can log a transparent audit trail.
    """
    if not os.path.isfile(csv_path):
        raise DatasetValidationError(f"Dataset path does not exist or is not a file: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_schema(df)

    original_rows = len(df)

    # 1. Duplicate detection
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    # 2. Missing value / NaN detection
    before = len(df)
    df = df.dropna(subset=config.ALL_CSV_COLUMNS)
    missing_value_rows_removed = before - len(df)

    # 3. Label validity: reject rows whose class labels are not recognized
    df = df[df[config.FADE_RISK_TARGET].isin(config.FADE_RISK_CLASSES)]
    df = df[df[config.MAINTENANCE_TARGET].isin(config.MAINTENANCE_CLASSES)]

    # 4. Range validation
    before = len(df)
    df = _apply_range_validation(df)
    range_violation_rows_removed = before - len(df)

    # 5. Cross-field plausibility validation
    before = len(df)
    df = _apply_plausibility_rules(df)
    plausibility_violation_rows_removed = before - len(df)

    # 6. Outlier clipping (does not remove rows, only clips feature values)
    df, outlier_rows_clipped = _clip_outliers_iqr(df)

    # 7. Regression target sanity clamp (defense in depth)
    df[config.REGRESSION_TARGET] = df[config.REGRESSION_TARGET].clip(
        config.REGRESSION_MIN, config.REGRESSION_MAX
    )

    df = df.reset_index(drop=True)

    report = CleaningReport(
        original_rows=original_rows,
        duplicates_removed=duplicates_removed,
        missing_value_rows_removed=missing_value_rows_removed,
        range_violation_rows_removed=range_violation_rows_removed,
        plausibility_violation_rows_removed=plausibility_violation_rows_removed,
        outlier_rows_clipped=outlier_rows_clipped,
        final_rows=len(df),
    )

    if logger:
        logger.info(
            "Dataset cleaning report: "
            f"original={report.original_rows}, "
            f"duplicates_removed={report.duplicates_removed}, "
            f"missing_removed={report.missing_value_rows_removed}, "
            f"range_violations_removed={report.range_violation_rows_removed}, "
            f"plausibility_violations_removed={report.plausibility_violation_rows_removed}, "
            f"outliers_clipped={report.outlier_rows_clipped}, "
            f"final={report.final_rows}"
        )

    if len(df) < 50:
        raise DatasetValidationError(
            f"Only {len(df)} usable rows remain after cleaning. "
            "Need at least 50 rows to train a meaningful model."
        )

    return df, report


def stratified_split(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split, stratified on maintenance recommendation
    (the finer-grained of the two classification targets) so every split
    contains a representative distribution of the rarer classes.

    Falls back to a non-stratified split with a warning if any class has
    too few members to stratify (scikit-learn requires >= 2 members per
    class per split).
    """
    stratify_col = df[config.MAINTENANCE_TARGET]
    class_counts = stratify_col.value_counts()
    min_class_count = class_counts.min()

    can_stratify = min_class_count >= 3  # need enough members for 3-way split

    stratify_arg = stratify_col if can_stratify else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_split + test_split),
        random_state=seed,
        stratify=stratify_arg,
    )

    relative_test_size = test_split / (val_split + test_split)
    stratify_arg_2 = temp_df[config.MAINTENANCE_TARGET] if can_stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=stratify_arg_2,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class FittedPreprocessors:
    """Container for every fitted transformer needed to reproduce
    preprocessing exactly at inference time."""

    def __init__(
        self,
        feature_scaler: StandardScaler,
        regression_scaler: MinMaxScaler,
        fade_risk_encoder: LabelEncoder,
        maintenance_encoder: LabelEncoder,
    ):
        self.feature_scaler = feature_scaler
        self.regression_scaler = regression_scaler
        self.fade_risk_encoder = fade_risk_encoder
        self.maintenance_encoder = maintenance_encoder

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.feature_scaler, os.path.join(directory, config.FEATURE_SCALER_NAME))
        joblib.dump(self.regression_scaler, os.path.join(directory, config.REGRESSION_SCALER_NAME))
        joblib.dump(self.fade_risk_encoder, os.path.join(directory, config.FADE_RISK_ENCODER_NAME))
        joblib.dump(self.maintenance_encoder, os.path.join(directory, config.MAINTENANCE_ENCODER_NAME))

    @classmethod
    def load(cls, directory: str) -> "FittedPreprocessors":
        feature_scaler = joblib.load(os.path.join(directory, config.FEATURE_SCALER_NAME))
        regression_scaler = joblib.load(os.path.join(directory, config.REGRESSION_SCALER_NAME))
        fade_risk_encoder = joblib.load(os.path.join(directory, config.FADE_RISK_ENCODER_NAME))
        maintenance_encoder = joblib.load(os.path.join(directory, config.MAINTENANCE_ENCODER_NAME))
        return cls(feature_scaler, regression_scaler, fade_risk_encoder, maintenance_encoder)


def fit_preprocessors(train_df: pd.DataFrame) -> FittedPreprocessors:
    """Fit all scalers/encoders on the TRAINING split only (never on val/test)."""
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_df[config.FEATURE_COLUMNS].values)

    # MinMaxScaler on the regression target: this is what resolves the
    # multi-task loss scaling bug -- without normalizing the 0-100 health
    # target to [0, 1], its MSE loss dwarfs the classification cross-entropy
    # losses and the classification heads fail to learn.
    regression_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    regression_scaler.fit(train_df[[config.REGRESSION_TARGET]].values)

    fade_risk_encoder = LabelEncoder()
    fade_risk_encoder.fit(config.FADE_RISK_CLASSES)  # fixed class set, not data-derived

    maintenance_encoder = LabelEncoder()
    maintenance_encoder.fit(config.MAINTENANCE_CLASSES)  # fixed class set

    return FittedPreprocessors(feature_scaler, regression_scaler, fade_risk_encoder, maintenance_encoder)


def transform_split(df: pd.DataFrame, preprocessors: FittedPreprocessors) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply fitted preprocessors to a dataframe split.

    Returns:
        X: (N, num_features) scaled features
        y_regression: (N, 1) scaled brake health target
        y_fade_risk: (N,) integer-encoded fade risk labels
        y_maintenance: (N,) integer-encoded maintenance labels
    """
    X = preprocessors.feature_scaler.transform(df[config.FEATURE_COLUMNS].values).astype(np.float32)
    y_regression = preprocessors.regression_scaler.transform(
        df[[config.REGRESSION_TARGET]].values
    ).astype(np.float32)
    y_fade_risk = preprocessors.fade_risk_encoder.transform(df[config.FADE_RISK_TARGET].values).astype(np.int64)
    y_maintenance = preprocessors.maintenance_encoder.transform(df[config.MAINTENANCE_TARGET].values).astype(np.int64)
    return X, y_regression, y_fade_risk, y_maintenance


class BrakeHealthDataset(Dataset):
    """torch Dataset wrapping pre-scaled numpy arrays for a single split."""

    def __init__(
        self,
        X: np.ndarray,
        y_regression: np.ndarray,
        y_fade_risk: np.ndarray,
        y_maintenance: np.ndarray,
    ):
        assert len(X) == len(y_regression) == len(y_fade_risk) == len(y_maintenance)
        self.X = torch.from_numpy(X).float()
        self.y_regression = torch.from_numpy(y_regression).float()
        self.y_fade_risk = torch.from_numpy(y_fade_risk).long()
        self.y_maintenance = torch.from_numpy(y_maintenance).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.y_regression[idx],
            self.y_fade_risk[idx],
            self.y_maintenance[idx],
        )
