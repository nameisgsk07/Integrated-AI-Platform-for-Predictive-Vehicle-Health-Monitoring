"""
predict.py
==========
Interactive inference script for the EdgeGuard AI Brake Health Prediction
model. Prompts the user for every raw brake sensor value, runs the trained
model, and reports Brake Health, Remaining Pad Life, Brake Fade Risk,
Maintenance Action, confidence for each output, and inference time.

Run directly:
    python predict.py
    python predict.py --checkpoint outputs/best_model.pt
"""

import argparse
import os
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from config import CONFIG
from model import build_model, clamp_outputs
from utils import log_header, log_info, log_success, log_error, get_device, load_json, load_checkpoint

# Human-readable prompts and units for each canonical input feature, in the
# same order they were saved to feature_order.json during training.
FEATURE_PROMPTS = {
    "brake_pad_thickness_mm": ("Brake Pad Thickness", "mm"),
    "brake_disc_temperature_c": ("Brake Disc Temperature", "°C"),
    "brake_fluid_level_pct": ("Brake Fluid Level", "%"),
    "brake_fluid_temperature_c": ("Brake Fluid Temperature", "°C"),
    "hydraulic_pressure_bar": ("Hydraulic Pressure", "bar"),
    "vehicle_speed_kmh": ("Vehicle Speed", "km/h"),
    "wheel_speed_kmh": ("Wheel Speed", "km/h"),
    "brake_pedal_force_pct": ("Brake Pedal Force", "%"),
    "ambient_temperature_c": ("Ambient Temperature", "°C"),
    "total_mileage_km": ("Total Mileage", "km"),
    "brake_usage_count": ("Brake Usage Count", "count"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the EdgeGuard AI Brake Health model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a model checkpoint (defaults to the best model saved during training).")
    return parser.parse_args()


def prompt_for_float(label: str, unit: str) -> float:
    while True:
        raw = input(f"Enter {label} ({unit}): ").strip()
        try:
            return float(raw)
        except ValueError:
            log_error("Please enter a valid number.")


def load_inference_artifacts(config, device, checkpoint_path=None):
    output_dir = config.output_dir

    scaler_path = os.path.join(output_dir, config.data.scaler_filename)
    target_scaler_path = os.path.join(output_dir, config.data.target_scaler_filename)
    fade_encoder_path = os.path.join(output_dir, config.data.fade_encoder_filename)
    maintenance_encoder_path = os.path.join(output_dir, config.data.maintenance_encoder_filename)
    feature_order_path = os.path.join(output_dir, config.data.feature_order_filename)

    for path in (scaler_path, target_scaler_path, fade_encoder_path, maintenance_encoder_path, feature_order_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required artifact not found: '{path}'. Have you run train.py yet?"
            )

    scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    fade_encoder = joblib.load(fade_encoder_path)
    maintenance_encoder = joblib.load(maintenance_encoder_path)
    feature_order = load_json(feature_order_path)

    config.model.input_dim = len(feature_order)
    config.model.num_fade_risk_classes = len(fade_encoder.classes_)
    config.model.num_maintenance_classes = len(maintenance_encoder.classes_)

    model = build_model(config).to(device)

    ckpt_path = checkpoint_path or os.path.join(output_dir, config.training.checkpoint_filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: '{ckpt_path}'. Have you run train.py yet?")

    load_checkpoint(ckpt_path, model, map_location=device)
    log_success(f"Loaded model checkpoint from '{ckpt_path}'.")

    return model, scaler, target_scaler, fade_encoder, maintenance_encoder, feature_order


def _to_physical_units(brake_health_norm, pad_life_norm, target_scaler):
    """Inverse-transforms normalized [0, 1] regression outputs back to
    physical units (percent, km) using the fitted target scaler."""
    normalized = np.column_stack([
        brake_health_norm.detach().cpu().numpy(),
        pad_life_norm.detach().cpu().numpy(),
    ])
    physical = target_scaler.inverse_transform(normalized)
    return torch.tensor(physical[:, 0]), torch.tensor(physical[:, 1])


@torch.no_grad()
def run_inference(model, input_tensor, config, device, target_scaler, mc_passes: int):
    """Runs a deterministic forward pass for classification outputs and
    Monte Carlo Dropout sampling for regression confidence estimation."""

    # Deterministic pass (eval mode, dropout off) -> point estimates & class logits.
    model.eval()
    outputs = model(input_tensor)
    health_physical, pad_life_physical = _to_physical_units(
        outputs["brake_health"], outputs["remaining_pad_life"], target_scaler
    )
    health_point, pad_life_point = clamp_outputs(health_physical, pad_life_physical, config.output_constraints)
    fade_probs = F.softmax(outputs["fade_risk_logits"], dim=1)[0]
    maintenance_probs = F.softmax(outputs["maintenance_logits"], dim=1)[0]

    # Monte Carlo Dropout passes -> regression uncertainty estimate (in physical units).
    model.enable_mc_dropout()
    health_samples, pad_life_samples = [], []
    for _ in range(mc_passes):
        mc_outputs = model(input_tensor)
        h_phys, p_phys = _to_physical_units(
            mc_outputs["brake_health"], mc_outputs["remaining_pad_life"], target_scaler
        )
        h, p = clamp_outputs(h_phys, p_phys, config.output_constraints)
        health_samples.append(h.item())
        pad_life_samples.append(p.item())
    model.eval()

    health_std = float(np.std(health_samples))
    pad_life_std = float(np.std(pad_life_samples))

    # Convert uncertainty (std dev) into an intuitive 0-100% confidence score
    # relative to each output's valid range: lower relative std => higher confidence.
    health_range = config.output_constraints.brake_health_max - config.output_constraints.brake_health_min
    pad_life_range = config.output_constraints.remaining_pad_life_max - config.output_constraints.remaining_pad_life_min

    health_confidence = float(np.clip(100.0 * (1.0 - (health_std / health_range) * 4.0), 0.0, 100.0))
    pad_life_confidence = float(np.clip(100.0 * (1.0 - (pad_life_std / pad_life_range) * 4.0), 0.0, 100.0))

    return {
        "brake_health_value": float(health_point.item()),
        "brake_health_confidence": health_confidence,
        "brake_health_std": health_std,
        "remaining_pad_life_value": float(pad_life_point.item()),
        "remaining_pad_life_confidence": pad_life_confidence,
        "remaining_pad_life_std": pad_life_std,
        "fade_probs": fade_probs.cpu().numpy(),
        "maintenance_probs": maintenance_probs.cpu().numpy(),
    }


def main():
    args = parse_args()
    config = CONFIG
    device = get_device()

    log_header("EdgeGuard AI - Brake Health Prediction - Inference")
    log_info(f"Using device: {device}")

    model, scaler, target_scaler, fade_encoder, maintenance_encoder, feature_order = load_inference_artifacts(
        config, device, checkpoint_path=args.checkpoint
    )

    log_header("Enter Brake Sensor Readings")
    raw_values = []
    for canonical_name in feature_order:
        label, unit = FEATURE_PROMPTS.get(canonical_name, (canonical_name, ""))
        raw_values.append(prompt_for_float(label, unit))

    input_array = scaler.transform(np.array([raw_values], dtype=np.float64))
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)

    start_time = time.time()
    result = run_inference(model, input_tensor, config, device, target_scaler, config.confidence.mc_dropout_passes)
    inference_time_ms = (time.time() - start_time) * 1000.0

    fade_class_idx = int(np.argmax(result["fade_probs"]))
    fade_class_name = fade_encoder.inverse_transform([fade_class_idx])[0]
    fade_confidence = float(result["fade_probs"][fade_class_idx]) * 100.0

    maintenance_class_idx = int(np.argmax(result["maintenance_probs"]))
    maintenance_class_name = maintenance_encoder.inverse_transform([maintenance_class_idx])[0]
    maintenance_confidence = float(result["maintenance_probs"][maintenance_class_idx]) * 100.0

    log_header("Prediction Results")
    print(f"Brake Health              : {result['brake_health_value']:.2f} %  "
          f"(confidence: {result['brake_health_confidence']:.1f}%)")
    print(f"Remaining Brake Pad Life  : {result['remaining_pad_life_value']:.0f} km  "
          f"(confidence: {result['remaining_pad_life_confidence']:.1f}%)")
    print(f"Brake Fade Risk           : {fade_class_name}  (confidence: {fade_confidence:.1f}%)")
    print(f"Maintenance Action        : {maintenance_class_name}  (confidence: {maintenance_confidence:.1f}%)")
    print(f"Inference Time            : {inference_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
