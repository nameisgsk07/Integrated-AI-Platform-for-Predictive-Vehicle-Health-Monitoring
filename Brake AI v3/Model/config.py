"""
config.py
=========

Central configuration for the EdgeGuard AI Brake Health Prediction System (v3).

This module is the SINGLE SOURCE OF TRUTH for:
    - Feature ordering and physical validation ranges
    - Output label definitions
    - Engineering constants used for physics-informed post-processing
      (in particular, the Remaining Pad Life calculation, which is NEVER
      predicted directly by the network -- it is derived analytically
      from Brake Health, Pad Thickness, Mileage and a calibrated wear model).
    - Model architecture hyperparameters
    - Training hyperparameters
    - File-system paths (all relative / configurable, never hardcoded
      dataset paths)

No other module should define magic numbers. If a new constant is needed,
it belongs here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

RANDOM_SEED: int = 42


# ======================================================================
# FEATURE DEFINITIONS (order is contractually fixed and used everywhere)
# ======================================================================

FEATURE_COLUMNS: List[str] = [
    "brake_pad_thickness_mm",
    "brake_disc_temp_c",
    "brake_fluid_level_pct",
    "brake_fluid_temp_c",
    "hydraulic_pressure_bar",
    "vehicle_speed_kmh",
    "brake_pedal_force_n",
    "ambient_temp_c",
    "vehicle_mileage_km",
]

NUM_FEATURES: int = len(FEATURE_COLUMNS)  # 9, contractually fixed

# Physical validation ranges: (min, max) inclusive.
# Any row outside these bounds is rejected during dataset cleaning and
# during live inference input validation.
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "brake_pad_thickness_mm": (0.0, 20.0),
    "brake_disc_temp_c": (-40.0, 900.0),
    "brake_fluid_level_pct": (0.0, 100.0),
    "brake_fluid_temp_c": (-40.0, 250.0),
    "hydraulic_pressure_bar": (0.0, 200.0),
    "vehicle_speed_kmh": (0.0, 320.0),
    "brake_pedal_force_n": (0.0, 2000.0),
    "ambient_temp_c": (-40.0, 60.0),
    "vehicle_mileage_km": (0.0, 1_000_000.0),
}


# ======================================================================
# CROSS-FIELD PHYSICAL PLAUSIBILITY RULES
# ======================================================================
# Each rule is a (description, predicate) pair. `predicate` receives a dict
# of feature_name -> value and returns True if the row is IMPLAUSIBLE
# (i.e. should be rejected). These encode "impossible sensor combination"
# checks such as: very high disc temperature at very low speed, or full
# fluid level with near-zero hydraulic pressure.

def _rule_high_temp_low_speed(row: Dict[str, float]) -> bool:
    return row["brake_disc_temp_c"] > 500.0 and row["vehicle_speed_kmh"] < 20.0


def _rule_full_fluid_no_pressure(row: Dict[str, float]) -> bool:
    return row["brake_fluid_level_pct"] > 90.0 and row["hydraulic_pressure_bar"] < 2.0


def _rule_negative_or_zero_thickness_with_pressure(row: Dict[str, float]) -> bool:
    return row["brake_pad_thickness_mm"] <= 0.05 and row["hydraulic_pressure_bar"] > 50.0


def _rule_extreme_pedal_force_zero_pressure(row: Dict[str, float]) -> bool:
    return row["brake_pedal_force_n"] > 200.0 and row["hydraulic_pressure_bar"] < 1.0


PHYSICAL_PLAUSIBILITY_RULES: List[Tuple[str, "callable"]] = [
    (
        "Disc temperature > 500C is implausible below 20 km/h "
        "(insufficient kinetic/frictional energy to reach that temperature).",
        _rule_high_temp_low_speed,
    ),
    (
        "Brake fluid level > 90% together with hydraulic pressure < 2 bar "
        "indicates a sensor fault (full reservoir should be able to build pressure).",
        _rule_full_fluid_no_pressure,
    ),
    (
        "Pad thickness <= 0.05mm (metal-to-metal) cannot sustain hydraulic "
        "pressure > 50 bar without catastrophic failure; sensor fault suspected.",
        _rule_negative_or_zero_thickness_with_pressure,
    ),
    (
        "Pedal force > 200N should always generate meaningful hydraulic "
        "pressure; < 1 bar indicates a hydraulic system fault or sensor fault.",
        _rule_extreme_pedal_force_zero_pressure,
    ),
]


# ======================================================================
# OUTPUT DEFINITIONS
# ======================================================================

# --- Regression target ---
REGRESSION_TARGET: str = "brake_health_pct"
REGRESSION_MIN: float = 0.0
REGRESSION_MAX: float = 100.0

# --- Classification target 1: Brake Fade Risk ---
FADE_RISK_TARGET: str = "brake_fade_risk"
FADE_RISK_CLASSES: List[str] = ["Low", "Medium", "High", "Very High", "Critical"]

# --- Classification target 2: Maintenance Recommendation ---
MAINTENANCE_TARGET: str = "maintenance_recommendation"
MAINTENANCE_CLASSES: List[str] = [
    "No Action",
    "Inspect Soon",
    "Replace Brake Pads",
    "Replace Brake Fluid",
    "Replace Brake Disc",
    "Immediate Service",
    "Emergency Stop",
]

TARGET_COLUMNS: List[str] = [REGRESSION_TARGET, FADE_RISK_TARGET, MAINTENANCE_TARGET]

ALL_CSV_COLUMNS: List[str] = FEATURE_COLUMNS + TARGET_COLUMNS


# ======================================================================
# ENGINEERING CONSTANTS: REMAINING PAD LIFE CALCULATION
# ======================================================================
# Remaining Pad Life is NEVER predicted by the neural network. It is
# calculated analytically at inference time using:
#
#   1. Current pad thickness (sensor)
#   2. The legal/manufacturer replacement thickness (minimum safe thickness)
#   3. A calibrated wear rate (mm consumed per 1000 km) that is MODULATED
#      by the network's predicted Brake Health (a healthier braking
#      system -- e.g. good fluid, good hydraulic pressure, moderate
#      temperatures -- wears pads more slowly and predictably; a
#      degraded system accelerates wear).
#   4. An engineering safety margin that further discounts the naive
#      estimate so the reported life is conservative (never optimistic).
#
# The result is ALWAYS clamped to be >= 0.

# Manufacturer new-pad thickness and minimum safe (replacement) thickness.
PAD_THICKNESS_NEW_MM: float = 12.0
PAD_THICKNESS_REPLACEMENT_MM: float = 2.0

# Baseline wear rate at 100% brake health, expressed in mm consumed per
# 1000 km of typical mixed driving. This is the calibrated "typical wear
# rate" referenced in the engineering spec.
BASE_WEAR_RATE_MM_PER_1000KM: float = 0.18

# Wear-rate modulation curve: as brake health degrades from 100 -> 0, the
# effective wear rate increases up to this multiplier at 0% health. This
# models accelerated pad consumption under degraded braking conditions
# (e.g. glazing, uneven pressure, excess heat).
MAX_WEAR_RATE_MULTIPLIER_AT_ZERO_HEALTH: float = 4.0

# Engineering safety margin applied to the final estimate (e.g. 0.85 means
# the reported remaining life is 85% of the raw analytical estimate, to
# keep the estimate conservative rather than optimistic).
PAD_LIFE_SAFETY_MARGIN: float = 0.85

# Absolute floor/ceiling guardrails for sanity (defense in depth).
PAD_LIFE_MIN_KM: float = 0.0
PAD_LIFE_MAX_KM: float = 150_000.0


# ======================================================================
# ENGINEERING CONSTANTS: FADE RISK / MAINTENANCE SANITY THRESHOLDS
# ======================================================================
# These are used only for post-hoc sanity clamping / warnings; they do not
# replace the classifier, but they prevent physically absurd combinations
# (e.g. Brake Health 95% simultaneously with "Emergency Stop").

BRAKE_HEALTH_CRITICAL_THRESHOLD: float = 15.0
BRAKE_HEALTH_LOW_THRESHOLD: float = 40.0
BRAKE_HEALTH_MODERATE_THRESHOLD: float = 70.0


# ======================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ======================================================================

@dataclass
class ModelConfig:
    input_dim: int = NUM_FEATURES
    backbone_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 256, 128])
    head_hidden_dim: int = 64
    dropout_p: float = 0.25
    num_fade_risk_classes: int = len(FADE_RISK_CLASSES)
    num_maintenance_classes: int = len(MAINTENANCE_CLASSES)


# ======================================================================
# TRAINING HYPERPARAMETERS
# ======================================================================

@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    lr_scheduler_t_max: int = 200
    lr_scheduler_eta_min: float = 1e-6
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 2
    use_amp: bool = True  # Automatic Mixed Precision (only active on CUDA)
    mc_dropout_passes: int = 30  # Monte Carlo Dropout passes for confidence

    # Multi-task loss weights (regression, fade_risk, maintenance)
    loss_weight_regression: float = 1.0
    loss_weight_fade_risk: float = 1.0
    loss_weight_maintenance: float = 1.0


# ======================================================================
# PATHS (relative, created on demand -- never hardcoded dataset paths)
# ======================================================================

PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR: str = os.path.join(PROJECT_ROOT, "artifacts")
CHECKPOINTS_DIR: str = os.path.join(ARTIFACTS_DIR, "checkpoints")
SCALERS_DIR: str = os.path.join(ARTIFACTS_DIR, "scalers")
LOGS_DIR: str = os.path.join(ARTIFACTS_DIR, "logs")
TENSORBOARD_DIR: str = os.path.join(ARTIFACTS_DIR, "tensorboard")
PLOTS_DIR: str = os.path.join(ARTIFACTS_DIR, "plots")
HISTORY_PATH: str = os.path.join(ARTIFACTS_DIR, "training_history.json")

LATEST_CHECKPOINT_NAME: str = "latest_checkpoint.pt"
BEST_CHECKPOINT_NAME: str = "best_checkpoint.pt"

FEATURE_SCALER_NAME: str = "feature_scaler.joblib"
REGRESSION_SCALER_NAME: str = "regression_scaler.joblib"
FADE_RISK_ENCODER_NAME: str = "fade_risk_encoder.joblib"
MAINTENANCE_ENCODER_NAME: str = "maintenance_encoder.joblib"


def ensure_directories() -> None:
    """Create every artifact directory required by the framework, if missing."""
    for directory in (
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        SCALERS_DIR,
        LOGS_DIR,
        TENSORBOARD_DIR,
        PLOTS_DIR,
    ):
        os.makedirs(directory, exist_ok=True)
