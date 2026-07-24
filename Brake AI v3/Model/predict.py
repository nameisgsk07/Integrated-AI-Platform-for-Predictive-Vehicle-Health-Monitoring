"""
predict.py
==========

Command-line inference entry point for EdgeGuard AI Brake Health Prediction.

Usage
-----
    # Interactive single-reading prediction (prompts for all 9 sensor values)
    python predict.py

    # Batch prediction from a CSV of sensor readings (feature columns only)
    python predict.py --input /path/to/readings.csv --output /path/to/report.csv

Pipeline (per the engineering spec)
------------------------------------
    1. Validate inputs (physical range + cross-field plausibility)
    2. Scale inputs (fitted StandardScaler)
    3. Run model (shared backbone + 3 heads)
    4. Clamp regression output to [0, 100]
    5. Apply engineering post-processing / sanity checks
    6. Calculate Remaining Pad Life analytically (never predicted directly)
    7. Estimate confidence:
         - Regression: Monte Carlo Dropout (mean/std across stochastic passes)
         - Classification: Softmax probability of the predicted class
    8. Generate a formatted report
    9. Report total inference time
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import config
import utils
from dataset import FittedPreprocessors, DatasetValidationError
from model import build_model


class InputValidationError(Exception):
    """Raised when a single sensor reading fails validation before inference."""


# ======================================================================
# INPUT VALIDATION (single reading)
# ======================================================================

def validate_reading(reading: Dict[str, float]) -> List[str]:
    """Validate a single sensor reading. Returns a list of human-readable
    error messages; an empty list means the reading is valid.
    """
    errors: List[str] = []

    missing = [f for f in config.FEATURE_COLUMNS if f not in reading]
    if missing:
        errors.append(f"Missing required features: {missing}")
        return errors  # cannot continue further checks without all features

    for feature, (low, high) in config.FEATURE_RANGES.items():
        value = reading[feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            errors.append(f"'{feature}' is missing/NaN.")
            continue
        if not (low <= value <= high):
            errors.append(
                f"'{feature}' = {value} is outside the physically valid range [{low}, {high}]."
            )

    if not errors:
        for description, rule_fn in config.PHYSICAL_PLAUSIBILITY_RULES:
            if rule_fn(reading):
                errors.append(f"Implausible sensor combination detected: {description}")

    return errors


# ======================================================================
# ENGINEERING POST-PROCESSING: REMAINING PAD LIFE
# ======================================================================

def calculate_remaining_pad_life_km(
    brake_health_pct: float,
    current_pad_thickness_mm: float,
) -> float:
    """Analytically derive Remaining Pad Life (km).

    This value is NEVER predicted by the neural network. It is calculated
    from:
        - current pad thickness (sensor)
        - replacement thickness threshold (config)
        - a calibrated baseline wear rate (config), modulated by the
          network's predicted Brake Health: healthier brakes wear pads
          more slowly/predictably, degraded brakes accelerate wear
        - an engineering safety margin (config) to keep the estimate
          conservative

    The result is always clamped to [PAD_LIFE_MIN_KM, PAD_LIFE_MAX_KM].
    """
    usable_thickness_mm = current_pad_thickness_mm - config.PAD_THICKNESS_REPLACEMENT_MM

    if usable_thickness_mm <= 0:
        return config.PAD_LIFE_MIN_KM

    # Health fraction in [0, 1]
    health_fraction = max(0.0, min(1.0, brake_health_pct / 100.0))

    # Wear-rate multiplier: 1.0x at 100% health, up to
    # MAX_WEAR_RATE_MULTIPLIER_AT_ZERO_HEALTH at 0% health. Linear
    # interpolation between the two extremes.
    wear_multiplier = 1.0 + (1.0 - health_fraction) * (
        config.MAX_WEAR_RATE_MULTIPLIER_AT_ZERO_HEALTH - 1.0
    )

    effective_wear_rate_mm_per_1000km = config.BASE_WEAR_RATE_MM_PER_1000KM * wear_multiplier

    if effective_wear_rate_mm_per_1000km <= 0:
        return config.PAD_LIFE_MIN_KM

    raw_remaining_km = (usable_thickness_mm / effective_wear_rate_mm_per_1000km) * 1000.0
    conservative_remaining_km = raw_remaining_km * config.PAD_LIFE_SAFETY_MARGIN

    clamped = max(config.PAD_LIFE_MIN_KM, min(config.PAD_LIFE_MAX_KM, conservative_remaining_km))
    return clamped


# ======================================================================
# MONTE CARLO DROPOUT CONFIDENCE (regression)
# ======================================================================

def mc_dropout_regression_confidence(
    model: torch.nn.Module,
    X_scaled: torch.Tensor,
    num_passes: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Run `num_passes` stochastic forward passes with dropout active
    (BatchNorm still in eval mode) and return (mean, std) of the
    regression output in SCALED [0,1] units.

    A smaller std indicates the model is more confident/consistent about
    its Brake Health estimate for this input.
    """
    model.enable_mc_dropout()
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            reg_pred, _, _ = model(X_scaled.to(device))
            predictions.append(reg_pred.item())
    model.eval()
    predictions_arr = np.array(predictions)
    return float(predictions_arr.mean()), float(predictions_arr.std())


def mc_dropout_classification_confidence(
    model: torch.nn.Module,
    X_scaled: torch.Tensor,
    num_passes: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout passes and return the averaged softmax probability
    vectors for (fade_risk, maintenance), used alongside the single-pass
    softmax confidence for a more robust estimate.
    """
    model.enable_mc_dropout()
    fade_probs_accum = None
    maint_probs_accum = None
    with torch.no_grad():
        for _ in range(num_passes):
            _, fade_logits, maint_logits = model(X_scaled.to(device))
            fade_probs = torch.softmax(fade_logits, dim=1).cpu().numpy()
            maint_probs = torch.softmax(maint_logits, dim=1).cpu().numpy()
            fade_probs_accum = fade_probs if fade_probs_accum is None else fade_probs_accum + fade_probs
            maint_probs_accum = maint_probs if maint_probs_accum is None else maint_probs_accum + maint_probs
    model.eval()
    return fade_probs_accum / num_passes, maint_probs_accum / num_passes


# ======================================================================
# ENGINEERING SANITY CROSS-CHECKS (warnings only, do not override the model)
# ======================================================================

def sanity_check_warnings(brake_health_pct: float, fade_risk: str, maintenance: str) -> List[str]:
    """Flag (but do not silently fix) physically inconsistent combinations
    of the network's own outputs, so the report is transparent about them.
    """
    warnings = []
    if brake_health_pct < config.BRAKE_HEALTH_CRITICAL_THRESHOLD and fade_risk in ("Low", "Medium"):
        warnings.append(
            f"Brake Health is critically low ({brake_health_pct:.1f}%) but predicted "
            f"Fade Risk is '{fade_risk}'. Recommend manual inspection to confirm."
        )
    if brake_health_pct > config.BRAKE_HEALTH_MODERATE_THRESHOLD and maintenance in (
        "Immediate Service", "Emergency Stop"
    ):
        warnings.append(
            f"Brake Health is high ({brake_health_pct:.1f}%) but predicted Maintenance "
            f"Recommendation is '{maintenance}'. Recommend manual inspection to confirm."
        )
    return warnings


# ======================================================================
# REPORT GENERATION
# ======================================================================

@dataclass
class PredictionReport:
    brake_health_pct: float
    brake_health_confidence_std: float
    remaining_pad_life_km: float
    fade_risk: str
    fade_risk_confidence: float
    maintenance: str
    maintenance_confidence: float
    warnings: List[str]
    inference_time_ms: float

    def render(self) -> str:
        lines = []
        lines.append("=" * 62)
        lines.append("EdgeGuard AI - Brake Health Prediction Report")
        lines.append("=" * 62)
        lines.append(f"Brake Health:                {self.brake_health_pct:6.2f} %  "
                      f"(confidence std: {self.brake_health_confidence_std:.4f})")
        lines.append(f"Remaining Pad Life:          {self.remaining_pad_life_km:8.1f} km "
                      f"(analytically calculated, not predicted)")
        lines.append(f"Brake Fade Risk:             {self.fade_risk:<12} "
                      f"(confidence: {self.fade_risk_confidence * 100:5.1f}%)")
        lines.append(f"Maintenance Recommendation:  {self.maintenance:<20} "
                      f"(confidence: {self.maintenance_confidence * 100:5.1f}%)")
        lines.append("-" * 62)
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("No sanity-check warnings.")
        lines.append("-" * 62)
        lines.append(f"Inference time: {self.inference_time_ms:.2f} ms")
        lines.append("=" * 62)
        return "\n".join(lines)


# ======================================================================
# CORE PREDICTION FUNCTION
# ======================================================================

def predict_single(
    reading: Dict[str, float],
    model: torch.nn.Module,
    preprocessors: FittedPreprocessors,
    device: torch.device,
    mc_passes: int,
) -> PredictionReport:
    errors = validate_reading(reading)
    if errors:
        raise InputValidationError("; ".join(errors))

    start_time = time.perf_counter()

    feature_vector = np.array([[reading[f] for f in config.FEATURE_COLUMNS]], dtype=np.float32)
    X_scaled_np = preprocessors.feature_scaler.transform(feature_vector).astype(np.float32)
    X_scaled = torch.from_numpy(X_scaled_np).float()

    model.eval()
    with torch.no_grad():
        reg_pred, fade_logits, maint_logits = model(X_scaled.to(device))

    # Single-pass softmax confidence
    fade_probs_single = torch.softmax(fade_logits, dim=1).cpu().numpy()[0]
    maint_probs_single = torch.softmax(maint_logits, dim=1).cpu().numpy()[0]

    # Monte Carlo Dropout: regression mean/std + averaged classification probs
    reg_mean_scaled, reg_std_scaled = mc_dropout_regression_confidence(model, X_scaled, mc_passes, device)
    fade_probs_mc, maint_probs_mc = mc_dropout_classification_confidence(model, X_scaled, mc_passes, device)

    # Blend single-pass and MC-averaged classification probabilities (equal weight)
    fade_probs = (fade_probs_single + fade_probs_mc[0]) / 2.0
    maint_probs = (maint_probs_single + maint_probs_mc[0]) / 2.0

    fade_idx = int(np.argmax(fade_probs))
    maint_idx = int(np.argmax(maint_probs))
    fade_risk = config.FADE_RISK_CLASSES[fade_idx]
    maintenance = config.MAINTENANCE_CLASSES[maint_idx]
    fade_confidence = float(fade_probs[fade_idx])
    maint_confidence = float(maint_probs[maint_idx])

    # De-scale regression output back to 0-100% and clamp (defense in depth)
    reg_mean_real = preprocessors.regression_scaler.inverse_transform([[reg_mean_scaled]])[0][0]
    brake_health_pct = float(np.clip(reg_mean_real, config.REGRESSION_MIN, config.REGRESSION_MAX))

    # Convert the scaled std into approximate real-world percentage-point units
    scale_range = (
        preprocessors.regression_scaler.data_max_[0] - preprocessors.regression_scaler.data_min_[0]
    )
    brake_health_confidence_std = float(reg_std_scaled * scale_range)

    remaining_pad_life_km = calculate_remaining_pad_life_km(
        brake_health_pct=brake_health_pct,
        current_pad_thickness_mm=reading["brake_pad_thickness_mm"],
    )

    warnings = sanity_check_warnings(brake_health_pct, fade_risk, maintenance)

    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionReport(
        brake_health_pct=brake_health_pct,
        brake_health_confidence_std=brake_health_confidence_std,
        remaining_pad_life_km=remaining_pad_life_km,
        fade_risk=fade_risk,
        fade_risk_confidence=fade_confidence,
        maintenance=maintenance,
        maintenance_confidence=maint_confidence,
        warnings=warnings,
        inference_time_ms=inference_time_ms,
    )


# ======================================================================
# INTERACTIVE CLI
# ======================================================================

def _prompt_for_reading() -> Dict[str, float]:
    print("\nEnter the 9 sensor readings for this braking system.\n")
    reading = {}
    prompts = {
        "brake_pad_thickness_mm": "Brake Pad Thickness (mm)",
        "brake_disc_temp_c": "Brake Disc Temperature (C)",
        "brake_fluid_level_pct": "Brake Fluid Level (%)",
        "brake_fluid_temp_c": "Brake Fluid Temperature (C)",
        "hydraulic_pressure_bar": "Hydraulic Pressure (bar)",
        "vehicle_speed_kmh": "Vehicle Speed (km/h)",
        "brake_pedal_force_n": "Brake Pedal Force (N)",
        "ambient_temp_c": "Ambient Temperature (C)",
        "vehicle_mileage_km": "Vehicle Mileage (km)",
    }
    for feature, label in prompts.items():
        low, high = config.FEATURE_RANGES[feature]
        while True:
            raw = input(f"{label} [{low} - {high}]: ").strip()
            try:
                value = float(raw)
                reading[feature] = value
                break
            except ValueError:
                print("  Please enter a valid number.")
    return reading


def _load_model_and_preprocessors(device: torch.device):
    model_cfg = config.ModelConfig()
    model = build_model(model_cfg).to(device)

    best_ckpt_path = f"{config.CHECKPOINTS_DIR}/{config.BEST_CHECKPOINT_NAME}"
    latest_ckpt_path = f"{config.CHECKPOINTS_DIR}/{config.LATEST_CHECKPOINT_NAME}"

    import os
    ckpt_path = best_ckpt_path if os.path.exists(best_ckpt_path) else latest_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No trained checkpoint found. Train a model first with train.py "
            f"(expected checkpoint at {best_ckpt_path} or {latest_ckpt_path})."
        )

    utils.load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    preprocessors = FittedPreprocessors.load(config.SCALERS_DIR)
    return model, preprocessors


def run_batch(input_csv: str, output_csv: Optional[str], model, preprocessors, device, mc_passes: int, logger):
    df = pd.read_csv(input_csv)
    missing_cols = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Input CSV is missing required feature columns: {missing_cols}")

    results = []
    for i, row in df.iterrows():
        reading = {f: float(row[f]) for f in config.FEATURE_COLUMNS}
        try:
            report = predict_single(reading, model, preprocessors, device, mc_passes)
            results.append({
                **reading,
                "brake_health_pct": report.brake_health_pct,
                "remaining_pad_life_km": report.remaining_pad_life_km,
                "fade_risk": report.fade_risk,
                "fade_risk_confidence": report.fade_risk_confidence,
                "maintenance": report.maintenance,
                "maintenance_confidence": report.maintenance_confidence,
                "warnings": " | ".join(report.warnings),
                "inference_time_ms": report.inference_time_ms,
            })
        except InputValidationError as e:
            logger.warning(f"Row {i} failed validation and was skipped: {e}")
            results.append({**reading, "brake_health_pct": None, "warnings": f"VALIDATION FAILED: {e}"})

    result_df = pd.DataFrame(results)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        logger.info(f"Batch prediction report written to {output_csv}")
    else:
        print(result_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeGuard AI - Brake Health Prediction (inference).")
    parser.add_argument("--input", type=str, default=None, help="CSV of sensor readings for batch prediction.")
    parser.add_argument("--output", type=str, default=None, help="Path to write batch prediction results CSV.")
    parser.add_argument("--mc-passes", type=int, default=None, help="Override number of Monte Carlo Dropout passes.")
    args = parser.parse_args()

    config.ensure_directories()
    logger = utils.setup_logger("edgeguard.predict", log_file=f"{config.LOGS_DIR}/predict.log")

    train_cfg = config.TrainConfig()
    mc_passes = args.mc_passes if args.mc_passes is not None else train_cfg.mc_dropout_passes

    device = utils.get_device(logger)

    try:
        model, preprocessors = _load_model_and_preprocessors(device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.input:
        run_batch(args.input, args.output, model, preprocessors, device, mc_passes, logger)
        return

    print("\nEdgeGuard AI - Brake Health Prediction System (Terminal Mode)")
    print("Type Ctrl+C at any time to exit.\n")

    while True:
        try:
            reading = _prompt_for_reading()
        except KeyboardInterrupt:
            print("\nExiting EdgeGuard AI. Stay safe on the road.")
            break

        try:
            report = predict_single(reading, model, preprocessors, device, mc_passes)
            print("\n" + report.render())
        except InputValidationError as e:
            print(f"\nInput validation failed: {e}")
            print("Please re-enter the reading.")

        again = input("\nPredict another reading? [y/N]: ").strip().lower()
        if again != "y":
            print("Exiting EdgeGuard AI. Stay safe on the road.")
            break


if __name__ == "__main__":
    main()
