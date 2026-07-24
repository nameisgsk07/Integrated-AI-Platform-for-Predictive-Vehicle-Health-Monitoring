"""
config.py
=========
Central configuration for the EdgeGuard AI - Brake Health Prediction Framework.

Every tunable parameter used across data handling, model architecture,
training behaviour and inference constraints lives here so that experiments
are reproducible and the framework can be reused for other EdgeGuard AI
modules (Motor AI, Battery AI, Tyre AI, ...) by writing a new config file
with the same shape.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


# =====================================================================
# DATA CONFIGURATION
# =====================================================================
@dataclass
class DataConfig:
    """Controls automatic column detection and dataset splitting."""

    # Keyword hints (case-insensitive, punctuation-insensitive) used to
    # automatically locate the eleven raw sensor columns inside whatever
    # CSV the user provides. Order here defines the canonical feature
    # order fed into the network.
    input_feature_keywords: "Dict[str, List[str]]" = field(default_factory=lambda: {
        "brake_pad_thickness_mm": ["pad_thickness", "pad thickness", "padthickness"],
        "brake_disc_temperature_c": ["disc_temp", "disc temperature", "disctemperature", "disc_temperature"],
        "brake_fluid_level_pct": ["fluid_level", "fluid level", "fluidlevel"],
        "brake_fluid_temperature_c": ["fluid_temp", "fluid temperature", "fluidtemperature", "fluid_temperature"],
        "hydraulic_pressure_bar": ["hydraulic_pressure", "hydraulic pressure", "hydraulicpressure"],
        "vehicle_speed_kmh": ["vehicle_speed", "vehicle speed", "vehiclespeed"],
        "wheel_speed_kmh": ["wheel_speed", "wheel speed", "wheelspeed"],
        "brake_pedal_force_pct": ["pedal_force", "pedal force", "pedalforce"],
        "ambient_temperature_c": ["ambient_temp", "ambient temperature", "ambienttemperature", "ambient_temperature"],
        "total_mileage_km": ["mileage", "total_mileage", "odometer"],
        "brake_usage_count": ["usage_count", "usage count", "brake_usage", "usagecount"],
    })

    # Regression targets: canonical_name -> list of keyword hints
    regression_target_keywords: "Dict[str, List[str]]" = field(default_factory=lambda: {
        "brake_health_pct": ["brake_health", "brake health", "brakehealth", "health_percent", "health_pct"],
        "remaining_pad_life_km": ["remaining_pad_life", "remaining pad life", "pad_life_km", "pad_life",
                                   "remaining_life", "padlife"],
    })

    # Classification targets: canonical_name -> list of keyword hints
    classification_target_keywords: "Dict[str, List[str]]" = field(default_factory=lambda: {
        "brake_fade_risk": ["fade_risk", "fade risk", "brake_fade", "brakefaderisk"],
        "maintenance_action": ["maintenance_action", "maintenance action", "maintenanceaction",
                                "action_required", "recommended_action"],
    })

    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    random_seed: int = 42

    # Artifact locations (relative to output_dir)
    scaler_filename: str = "input_scaler.joblib"
    target_scaler_filename: str = "target_scaler.joblib"
    fade_encoder_filename: str = "brake_fade_risk_encoder.joblib"
    maintenance_encoder_filename: str = "maintenance_action_encoder.joblib"
    feature_order_filename: str = "feature_order.json"


# =====================================================================
# MODEL CONFIGURATION
# =====================================================================
@dataclass
class ModelConfig:
    """Controls the neural network architecture."""

    input_dim: int = 11  # Number of input sensor features (auto-verified at runtime)

    # Shared backbone: list of hidden layer sizes. Each entry becomes one
    # Linear -> BatchNorm -> ReLU -> Dropout block.
    backbone_hidden_dims: "List[int]" = field(default_factory=lambda: [128, 128, 64])

    # Each output head gets its own private hidden layer of this size
    # before its final projection (heads never share final layers).
    head_hidden_dim: int = 32

    dropout: float = 0.2

    # Number of classes, filled in automatically from the label encoders
    # after dataset preparation, but given sensible defaults here so the
    # model can be constructed standalone if needed.
    num_fade_risk_classes: int = 5          # Low, Medium, High, Very High, Critical
    num_maintenance_classes: int = 7        # No Action ... Emergency Stop


# =====================================================================
# OUTPUT CONSTRAINTS
# =====================================================================
@dataclass
class OutputConstraints:
    """Physical bounds the model outputs must never violate.

    These can be changed freely later (e.g. for a different vehicle
    platform) without touching model or training code.
    """

    brake_health_min: float = 0.0
    brake_health_max: float = 100.0

    remaining_pad_life_min: float = 0.0
    remaining_pad_life_max: float = 50000.0


# =====================================================================
# LOSS CONFIGURATION
# =====================================================================
@dataclass
class LossConfig:
    """Weights used to combine the four task losses into one scalar."""

    brake_health_weight: float = 1.0
    remaining_pad_life_weight: float = 1.0
    fade_risk_weight: float = 1.0
    maintenance_action_weight: float = 1.0

    huber_delta: float = 1.0


# =====================================================================
# TRAINING CONFIGURATION
# =====================================================================
@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # ReduceLROnPlateau scheduler
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6

    grad_clip_norm: float = 1.0

    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    num_workers: int = 0  # DataLoader workers (0 is safest across platforms)

    use_amp: bool = True  # Automatic mixed precision (only engages if CUDA is available)

    checkpoint_filename: str = "best_model.pt"
    last_checkpoint_filename: str = "last_checkpoint.pt"
    history_filename: str = "training_history.csv"


# =====================================================================
# CONFIDENCE ESTIMATION
# =====================================================================
@dataclass
class ConfidenceConfig:
    # Monte Carlo Dropout passes used to estimate regression uncertainty
    # at inference time (dropout layers stay active for these passes).
    mc_dropout_passes: int = 30


# =====================================================================
# MASTER CONFIG
# =====================================================================
@dataclass
class EdgeGuardConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_constraints: OutputConstraints = field(default_factory=OutputConstraints)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    output_dir: str = "outputs"

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir


# Single shared instance imported by every module. Editing this file is
# the ONLY place needed to retune the whole framework.
CONFIG = EdgeGuardConfig()
