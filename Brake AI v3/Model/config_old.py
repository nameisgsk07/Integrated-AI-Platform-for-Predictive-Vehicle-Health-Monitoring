"""
config.py
=========

Single Source of Truth (SSOT) configuration module for the
EdgeGuard AI Brake Health Prediction System.

Every other module in this project (dataset.py, model.py, train.py,
predict.py, metrics.py, losses.py, utils.py) MUST import all constants,
paths, hyperparameters, physical limits, and label mappings from this
file. No magic numbers should exist anywhere else in the codebase.

Python version: 3.12
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard project dependency
    torch = None  # type: ignore[assignment]


# ======================================================================
# 1. PROJECT INFORMATION
# ======================================================================

@dataclass(frozen=True)
class ProjectInfo:
    """Static metadata describing the project."""

    name: str = "EdgeGuard AI"
    subsystem: str = "Brake Health Prediction System"
    version: str = "3.0.0"
    author: str = "<AUTHOR NAME PLACEHOLDER>"
    description: str = (
        "Physics-informed, multi-task deep learning system for predicting "
        "automotive brake health, remaining pad life, brake fade risk, and "
        "recommended maintenance actions from tabular sensor telemetry."
    )


PROJECT: Final[ProjectInfo] = ProjectInfo()


# ======================================================================
# PROJECT ROOT / PATH RESOLUTION (no hardcoded absolute paths)
# ======================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


# ======================================================================
# 2. DATASET CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class DatasetConfig:
    """Defines dataset location, schema, and feature/target ordering."""

    csv_filename: str = "brake_health_synthetic_dataset.csv"
    data_dir: Path = PROJECT_ROOT / "data"

    # Exact, ordered list of input feature column names.
    input_features: tuple[str, ...] = (
        "brake_pad_thickness_mm",
        "brake_disc_temperature_c",
        "brake_fluid_level_pct",
        "brake_fluid_temperature_c",
        "hydraulic_pressure_bar",
        "vehicle_speed_kmh",
        "brake_pedal_force_n",
        "ambient_temperature_c",
        "mileage_km",
    )

    # Regression target column names, in output order.
    regression_targets: tuple[str, ...] = (
        "brake_health_pct",
        "remaining_pad_life_km",
    )

    # Classification target column names, in output-head order.
    classification_targets: tuple[str, ...] = (
        "brake_fade_risk",
        "maintenance_action",
    )

    @property
    def csv_path(self) -> Path:
        """Full relative path to the dataset CSV file."""
        return self.data_dir / self.csv_filename

    @property
    def num_input_features(self) -> int:
        """Number of input features fed into the network."""
        return len(self.input_features)

    @property
    def num_regression_outputs(self) -> int:
        """Number of continuous regression outputs."""
        return len(self.regression_targets)

    @property
    def num_classification_outputs(self) -> int:
        """Number of independent classification heads."""
        return len(self.classification_targets)


DATASET: Final[DatasetConfig] = DatasetConfig()


# ======================================================================
# 3. PHYSICAL LIMITS
# ======================================================================

@dataclass(frozen=True)
class Range:
    """A simple immutable (minimum, maximum) physical bound."""

    minimum: float
    maximum: float

    def clamp(self, value: float) -> float:
        """Clamp a value to lie within [minimum, maximum]."""
        return max(self.minimum, min(self.maximum, value))


@dataclass(frozen=True)
class PhysicalLimits:
    """Real-world physical operating limits for every sensor / target.

    These bounds are used for synthetic data generation, input
    validation, output clamping, and engineering-based post-processing.
    """

    brake_pad_thickness_mm: Range = field(default_factory=lambda: Range(0.0, 12.0))
    brake_disc_temperature_c: Range = field(default_factory=lambda: Range(-20.0, 800.0))
    brake_fluid_level_pct: Range = field(default_factory=lambda: Range(0.0, 100.0))
    brake_fluid_temperature_c: Range = field(default_factory=lambda: Range(-20.0, 200.0))
    hydraulic_pressure_bar: Range = field(default_factory=lambda: Range(0.0, 180.0))
    vehicle_speed_kmh: Range = field(default_factory=lambda: Range(0.0, 260.0))
    brake_pedal_force_n: Range = field(default_factory=lambda: Range(0.0, 900.0))
    ambient_temperature_c: Range = field(default_factory=lambda: Range(-30.0, 55.0))
    mileage_km: Range = field(default_factory=lambda: Range(0.0, 300_000.0))
    brake_health_pct: Range = field(default_factory=lambda: Range(0.0, 100.0))
    remaining_pad_life_km: Range = field(default_factory=lambda: Range(0.0, 80_000.0))


PHYSICAL_LIMITS: Final[PhysicalLimits] = PhysicalLimits()


# ======================================================================
# 4. CLASSIFICATION LABELS
# ======================================================================

@dataclass(frozen=True)
class BrakeFadeRiskLabels:
    """Ordinal severity labels for brake fade risk classification."""

    LOW: Final[str] = "Low"
    MEDIUM: Final[str] = "Medium"
    HIGH: Final[str] = "High"
    VERY_HIGH: Final[str] = "Very High"
    CRITICAL: Final[str] = "Critical"

    @property
    def ordered(self) -> tuple[str, ...]:
        return (self.LOW, self.MEDIUM, self.HIGH, self.VERY_HIGH, self.CRITICAL)

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.ordered)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.ordered)}


@dataclass(frozen=True)
class MaintenanceActionLabels:
    """Recommended maintenance action labels, in ascending urgency."""

    NO_ACTION: Final[str] = "No Action"
    INSPECT_SOON: Final[str] = "Inspect Soon"
    REPLACE_BRAKE_PADS: Final[str] = "Replace Brake Pads"
    REPLACE_BRAKE_FLUID: Final[str] = "Replace Brake Fluid"
    REPLACE_BRAKE_DISC: Final[str] = "Replace Brake Disc"
    IMMEDIATE_SERVICE: Final[str] = "Immediate Service"
    EMERGENCY_STOP: Final[str] = "Emergency Stop"

    @property
    def ordered(self) -> tuple[str, ...]:
        return (
            self.NO_ACTION,
            self.INSPECT_SOON,
            self.REPLACE_BRAKE_PADS,
            self.REPLACE_BRAKE_FLUID,
            self.REPLACE_BRAKE_DISC,
            self.IMMEDIATE_SERVICE,
            self.EMERGENCY_STOP,
        )

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.ordered)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.ordered)}


BRAKE_FADE_RISK: Final[BrakeFadeRiskLabels] = BrakeFadeRiskLabels()
MAINTENANCE_ACTION: Final[MaintenanceActionLabels] = MaintenanceActionLabels()


# ======================================================================
# 5. NEURAL NETWORK CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class NetworkConfig:
    """Architecture hyperparameters for the multi-task network."""

    input_dim: int = DATASET.num_input_features
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"
    regression_head_size: int = DATASET.num_regression_outputs
    classification_head_sizes: tuple[int, ...] = (
        len(BRAKE_FADE_RISK.ordered),
        len(MAINTENANCE_ACTION.ordered),
    )
    weight_init: str = "kaiming_uniform"


NETWORK: Final[NetworkConfig] = NetworkConfig()


# ======================================================================
# 6. TRAINING CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and settings controlling the training loop."""

    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine_annealing"
    early_stopping_patience: int = 20
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    num_dataloader_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True
    checkpoint_frequency_epochs: int = 10


TRAINING: Final[TrainingConfig] = TrainingConfig()


# ======================================================================
# 7. LOSS WEIGHTS
# ======================================================================

@dataclass(frozen=True)
class LossWeights:
    """Relative weighting of each task's loss in the combined objective.

    Brake health regression is weighted highest since it is the primary
    continuous safety signal. Fade risk classification is weighted above
    maintenance action classification because fade risk is more directly
    tied to acute safety outcomes, while maintenance action is a slower,
    advisory-oriented signal.
    """

    brake_health_regression: float = 1.0
    brake_fade_classification: float = 0.7
    maintenance_classification: float = 0.5


LOSS_WEIGHTS: Final[LossWeights] = LossWeights()


# ======================================================================
# 8. DATA NORMALIZATION
# ======================================================================

@dataclass(frozen=True)
class NormalizationConfig:
    """Feature scaling configuration.

    ImageNet-style normalization does not apply since this project uses
    tabular sensor data, not images. A standard scaler is used instead.
    """

    scaler_type: str = "standard"  # one of: "standard", "minmax", "robust"
    feature_scaler_filename: str = "feature_scaler.joblib"
    regression_target_scaler_filename: str = "regression_target_scaler.joblib"


NORMALIZATION: Final[NormalizationConfig] = NormalizationConfig()


# ======================================================================
# 9. LOGGING CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class LoggingConfig:
    """Filesystem locations and filenames for logs, checkpoints, models."""

    output_dir: Path = PROJECT_ROOT / "outputs"
    tensorboard_dir: Path = PROJECT_ROOT / "outputs" / "tensorboard"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    log_filename: str = "training.log"
    training_history_filename: str = "training_history.json"
    model_filename: str = "brake_health_model.pt"
    best_model_filename: str = "brake_health_model_best.pt"

    @property
    def log_path(self) -> Path:
        return self.output_dir / self.log_filename

    @property
    def training_history_path(self) -> Path:
        return self.output_dir / self.training_history_filename

    @property
    def model_path(self) -> Path:
        return self.checkpoint_dir / self.model_filename

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / self.best_model_filename

    def ensure_directories_exist(self) -> None:
        """Create all logging/output directories if they do not exist."""
        for directory in (self.output_dir, self.tensorboard_dir, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)


LOGGING: Final[LoggingConfig] = LoggingConfig()


# ======================================================================
# 10. ENGINEERING CONSTANTS
# ======================================================================

@dataclass(frozen=True)
class EngineeringConstants:
    """Domain engineering constants used for physics-informed synthetic
    data generation and rule-based post-processing / sanity checks.
    """

    # Pad thickness reference points (mm).
    brake_pad_new_thickness_mm: float = 12.0
    brake_pad_replacement_thickness_mm: float = 3.0

    # Expected service life.
    max_expected_pad_life_km: float = 60_000.0
    typical_wear_rate_mm_per_1000km: float = 0.15

    # Temperature thresholds (Celsius) for disc/fluid behavior.
    disc_temp_warning_threshold_c: float = 400.0
    disc_temp_critical_threshold_c: float = 600.0
    fluid_temp_boiling_risk_threshold_c: float = 150.0

    # Brake fade onset thresholds.
    fade_risk_temp_threshold_c: float = 350.0
    fade_risk_pressure_drop_threshold_bar: float = 20.0

    # Safety margins applied during engineering post-processing.
    safety_margin_pad_thickness_mm: float = 1.0
    safety_margin_health_pct: float = 5.0


ENGINEERING: Final[EngineeringConstants] = EngineeringConstants()


# ======================================================================
# 11. PREDICTION CONFIGURATION
# ======================================================================

@dataclass(frozen=True)
class PredictionConfig:
    """Settings controlling inference-time behavior."""

    confidence_threshold: float = 0.6
    max_inference_batch_size: int = 512
    clamp_outputs: bool = True
    enable_engineering_postprocessing: bool = True


PREDICTION: Final[PredictionConfig] = PredictionConfig()


# ======================================================================
# 12. DEVICE CONFIGURATION
# ======================================================================

def get_device() -> "torch.device":
    """Automatically select the best available compute device.

    Preference order: CUDA -> Apple MPS -> CPU.

    Returns:
        A torch.device instance representing the selected device.

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to determine the compute device but is "
            "not installed in this environment."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: Final["torch.device"] = get_device() if torch is not None else None  # type: ignore[assignment]


# ======================================================================
# 13. UTILITY FUNCTIONS
# ======================================================================

def seed_everything(seed: int = TRAINING.random_seed) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: The random seed to apply across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def validate_config() -> None:
    """Run internal consistency checks across the configuration.

    Raises:
        ValueError: If any configuration invariant is violated.
    """
    if DATASET.num_input_features != NETWORK.input_dim:
        raise ValueError(
            "NETWORK.input_dim does not match DATASET.num_input_features: "
            f"{NETWORK.input_dim} != {DATASET.num_input_features}"
        )

    if DATASET.num_regression_outputs != NETWORK.regression_head_size:
        raise ValueError(
            "NETWORK.regression_head_size does not match "
            "DATASET.num_regression_outputs: "
            f"{NETWORK.regression_head_size} != {DATASET.num_regression_outputs}"
        )

    if len(NETWORK.classification_head_sizes) != DATASET.num_classification_outputs:
        raise ValueError(
            "NETWORK.classification_head_sizes length does not match "
            "DATASET.num_classification_outputs: "
            f"{len(NETWORK.classification_head_sizes)} != "
            f"{DATASET.num_classification_outputs}"
        )

    if NETWORK.classification_head_sizes[0] != len(BRAKE_FADE_RISK.ordered):
        raise ValueError(
            "First classification head size does not match the number of "
            "brake fade risk labels."
        )

    if NETWORK.classification_head_sizes[1] != len(MAINTENANCE_ACTION.ordered):
        raise ValueError(
            "Second classification head size does not match the number of "
            "maintenance action labels."
        )

    total_split = TRAINING.validation_split + TRAINING.test_split
    if not 0.0 < total_split < 1.0:
        raise ValueError(
            "TRAINING.validation_split + TRAINING.test_split must be in "
            f"(0, 1); got {total_split}"
        )

    if ENGINEERING.brake_pad_replacement_thickness_mm >= ENGINEERING.brake_pad_new_thickness_mm:
        raise ValueError(
            "brake_pad_replacement_thickness_mm must be less than "
            "brake_pad_new_thickness_mm."
        )


if __name__ == "__main__":
    # Basic self-check when this module is executed directly.
    validate_config()
    LOGGING.ensure_directories_exist()
    print(f"{PROJECT.name} {PROJECT.subsystem} - config v{PROJECT.version} OK")
    print(f"Device: {DEVICE}")
