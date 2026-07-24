# EdgeGuard AI — Brake Health Prediction Framework

A production-quality, modular PyTorch framework for automotive brake health
prediction. One shared neural network backbone feeds four independent
output heads (two regression, two classification) trained jointly on brake
sensor telemetry. Built to be lightweight enough for later ONNX / TensorRT
export and deployment on an automotive infotainment ECU, and structured so
the same pattern can be reused for other EdgeGuard AI modules (Motor AI,
Battery AI, Tyre AI, ...).

## What it predicts

| Output | Type | Range / Classes |
|---|---|---|
| Brake Health | Regression | 0–100 % |
| Remaining Brake Pad Life | Regression | 0–50,000 km |
| Brake Fade Risk | Classification | Low, Medium, High, Very High, Critical |
| Maintenance Action | Classification | No Action, Inspect Soon, Replace Brake Fluid, Replace Brake Pads, Replace Brake Disc, Immediate Service, Emergency Stop |

## Input features (11 sensors)

Brake Pad Thickness (mm) · Brake Disc Temperature (°C) · Brake Fluid Level
(%) · Brake Fluid Temperature (°C) · Hydraulic Pressure (bar) · Vehicle
Speed (km/h) · Wheel Speed (km/h) · Brake Pedal Force (%) · Ambient
Temperature (°C) · Total Mileage (km) · Brake Usage Count.

## Project structure

```
config.py         Every tunable parameter (data, model, loss, training, confidence)
model.py          Shared backbone + 4 independent heads (EdgeGuardBrakeNet)
dataset.py        CSV loading, automatic column detection, preprocessing, splitting
train.py          Training loop: AMP, early stopping, checkpointing, resume, metrics
predict.py        Interactive CLI inference with confidence and MC-Dropout uncertainty
utils.py          Logging, seeding, checkpoint I/O, metrics
requirements.txt  Python dependencies
```

## How the model works

1. **Shared backbone** — a stack of `Linear -> BatchNorm -> ReLU -> Dropout`
   blocks (sizes configurable in `config.py`, default `[128, 128, 64]`)
   learns a general representation of brake behaviour from the 11 inputs.
2. **Four independent heads** branch off the shared features. Each head has
   its own private hidden layer and its own final projection layer — no
   head shares its output layer with another head or with the backbone.
   Regression heads use a Huber loss; classification heads use
   cross-entropy. The four losses are combined into one weighted scalar
   (weights configurable in `config.py`).
3. **Target normalization** — Brake Health (0–100) and Remaining Pad Life
   (0–50,000 km) live on very different numeric scales. Training directly
   on raw values lets whichever target has the larger magnitude dominate
   the shared loss. The framework fits a `MinMaxScaler` on the two
   regression targets (fit on the training split only) and trains in
   normalized `[0, 1]` space; predictions are inverse-transformed back to
   physical units before clamping, evaluation and display.
4. **Output clamping** — after inverse-transforming, Brake Health is
   clamped to `[0, 100]` and Remaining Pad Life to `[0, 50000]` (both
   limits configurable in `config.py`) so the model can never report a
   physically impossible value.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
```

You'll be prompted:

```
Enter dataset path:
```

Enter the path to your CSV file. The framework does not require exact
column names — it automatically detects the 11 input features and 4 target
columns using flexible keyword matching (e.g. a column named
`Brake_Health_pct`, `brake health (%)`, or `BrakeHealth` will all be
recognized as the Brake Health target). If a column can't be matched, the
error message lists the available columns so you can adjust your CSV
header or the keyword lists in `config.py`.

Non-interactive usage and useful flags:

```bash
python train.py --data /path/to/data.csv          # skip the prompt
python train.py --data data.csv --epochs 50        # override epoch count
python train.py --data data.csv --resume outputs/last_checkpoint.pt
```

Training produces, under `outputs/` (configurable via `CONFIG.output_dir`):

- `best_model.pt` — best checkpoint by validation loss (used for inference)
- `last_checkpoint.pt` — most recent checkpoint (used to `--resume`)
- `training_history.csv` — per-epoch train/val loss, learning rate, timing
- `test_set_metrics.json` — MAE/RMSE/R² and accuracy/precision/recall/F1
  on the held-out test set
- `input_scaler.joblib`, `target_scaler.joblib` — fitted preprocessing scalers
- `brake_fade_risk_encoder.joblib`, `maintenance_action_encoder.joblib` — label encoders
- `feature_order.json` — canonical input feature order used at inference

### 3. Predict

```bash
python predict.py
```

You'll be prompted for each of the 11 sensor readings, then the script
reports:

```
Brake Health              : 61.42 %  (confidence: 91.3%)
Remaining Brake Pad Life  : 21840 km  (confidence: 88.7%)
Brake Fade Risk           : Medium  (confidence: 94.2%)
Maintenance Action        : Inspect Soon  (confidence: 87.5%)
Inference Time            : 3.14 ms
```

Regression confidence is estimated with Monte Carlo Dropout (multiple
stochastic forward passes with dropout left active; configurable via
`CONFIG.confidence.mc_dropout_passes`, default 30) — lower prediction
variance across passes yields a higher confidence score. Classification
confidence is the Softmax probability of the predicted class.

## Configuration

Every tunable parameter lives in `config.py`:

- **`DataConfig`** — keyword hints used for automatic column detection,
  train/val/test split ratios, random seed, artifact filenames.
- **`ModelConfig`** — backbone hidden layer sizes, per-head hidden size,
  dropout rate.
- **`OutputConstraints`** — min/max clamping bounds for each regression
  output (change these if a different vehicle platform needs different
  physical limits).
- **`LossConfig`** — per-task loss weights and Huber delta.
- **`TrainingConfig`** — batch size, epochs, learning rate, weight decay,
  LR scheduler settings, gradient clip norm, early stopping patience,
  AMP toggle.
- **`ConfidenceConfig`** — number of Monte Carlo Dropout passes.

## Reusing this framework for other EdgeGuard AI modules

The pattern generalizes directly: swap `config.py`'s keyword dictionaries
and output constraints for a new module's inputs/targets, adjust
`ModelConfig.num_*_classes` for the new classification heads, and the rest
of `model.py`, `train.py`, `predict.py` and `utils.py` work unchanged.

## Deployment notes

The network uses only `Linear`, `BatchNorm1d`, `ReLU` and `Dropout` layers —
all natively supported by ONNX and TensorRT export paths. To export:

```python
model.eval()
dummy_input = torch.randn(1, config.model.input_dim)
torch.onnx.export(model, dummy_input, "edgeguard_brake_net.onnx",
                   input_names=["sensor_inputs"],
                   output_names=["brake_health", "remaining_pad_life",
                                 "fade_risk_logits", "maintenance_logits"])
```

## Notes on the training data

This framework does not ship with or generate a dataset. Provide a CSV
containing the 11 input columns and 4 target columns (any reasonable
naming — see automatic detection above). Rows with missing values in any
required column are dropped with a warning before training.
