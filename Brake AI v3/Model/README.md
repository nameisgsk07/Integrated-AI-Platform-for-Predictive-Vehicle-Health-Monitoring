# EdgeGuard AI — Brake Health Prediction System (v3)

A physics-informed, multi-task PyTorch framework that predicts automotive
hydraulic brake health from sensor readings, entirely from the terminal.

No images, no CAN bus libraries, no ONNX/TensorRT export, no web UI — pure
numerical sensor inputs in, engineering-grade report out.

---

## What it predicts

The neural network predicts **only**:

| Output | Type | Range / Classes |
|---|---|---|
| Brake Health (%) | Regression | 0–100 |
| Brake Fade Risk | Classification | Low, Medium, High, Very High, Critical |
| Maintenance Recommendation | Classification | No Action, Inspect Soon, Replace Brake Pads, Replace Brake Fluid, Replace Brake Disc, Immediate Service, Emergency Stop |

**Remaining Brake Pad Life is deliberately *not* predicted.** A previous
version predicted it directly and produced physically impossible results
(negative life, life estimates disconnected from pad thickness, etc.).
In this version, Remaining Pad Life is **calculated analytically at
inference time** from:

- current pad thickness (sensor),
- the manufacturer replacement thickness threshold,
- a calibrated baseline wear rate, modulated by the model's own predicted
  Brake Health (degraded systems wear pads faster),
- an engineering safety margin (conservative, never optimistic),

and is always clamped to be `>= 0`. See `calculate_remaining_pad_life_km`
in `predict.py` and the constants in `config.py`.

---

## The 9 input features (fixed order, contractually enforced)

1. Brake Pad Thickness (mm)
2. Brake Disc Temperature (°C)
3. Brake Fluid Level (%)
4. Brake Fluid Temperature (°C)
5. Hydraulic Pressure (bar)
6. Vehicle Speed (km/h)
7. Brake Pedal Force (N)
8. Ambient Temperature (°C)
9. Vehicle Mileage (km)

Every dataset CSV and every live prediction is validated against exactly
these 9 columns (no more, no fewer) before anything touches the model.

---

## Project structure

```
config.py       Single source of truth: feature/label definitions, physical
                validation ranges, engineering constants (pad-life formula),
                model & training hyperparameters, paths.
dataset.py      CSV loading, schema validation, duplicate/NaN/outlier
                cleaning, cross-field plausibility rules, stratified
                train/val/test split, scaler/encoder fitting.
model.py        EdgeGuardNet: shared Residual-FC backbone (BatchNorm,
                Dropout, Kaiming init) + 1 regression head + 2 classification
                heads.
losses.py       Weighted multi-task loss (Huber regression + 2x
                cross-entropy), balanced by MinMax-scaling the regression
                target so it doesn't dominate the combined loss.
metrics.py      Regression (MAE/RMSE/R2) and classification (accuracy,
                macro/weighted F1, full report) metrics, computed against
                the FULL fixed label set to avoid class-mismatch errors on
                small/imbalanced splits.
utils.py        Logging, seeding, device selection, checkpoint save/resume
                (robust to PyTorch 2.6+ `weights_only` default change),
                training-curve plotting.
train.py        CLI training entry point: AdamW + Cosine Annealing LR,
                gradient clipping, AMP, early stopping, checkpoint
                resume (latest/best/fresh), TensorBoard logging, automatic
                history + graph generation.
predict.py      CLI inference entry point: input validation, scaling,
                inference, output clamping, engineering post-processing,
                Remaining Pad Life calculation, Monte Carlo Dropout +
                Softmax confidence, formatted report, inference timing.
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is used automatically if available (`torch.cuda.is_available()`);
otherwise the framework falls back to CPU with no code changes required.

---

## Dataset format

A CSV with exactly these 12 columns (9 features + 3 targets):

```
brake_pad_thickness_mm, brake_disc_temp_c, brake_fluid_level_pct,
brake_fluid_temp_c, hydraulic_pressure_bar, vehicle_speed_kmh,
brake_pedal_force_n, ambient_temp_c, vehicle_mileage_km,
brake_health_pct, brake_fade_risk, maintenance_recommendation
```

The dataset path is **always supplied at runtime** — never hardcoded.

---

## Training

```bash
# Fresh training run
python train.py --dataset /path/to/data.csv

# Resume from the most recent checkpoint (default when a checkpoint exists)
python train.py --dataset /path/to/more_data.csv --resume latest

# Resume from the best-validation-loss checkpoint (e.g. for fine-tuning
# on new data without carrying forward a possibly-overfit recent state)
python train.py --dataset /path/to/more_data.csv --resume best

# Force training from scratch, ignoring any existing checkpoints
# (existing checkpoints are NOT deleted)
python train.py --dataset /path/to/data.csv --fresh

# Override hyperparameters
python train.py --dataset /path/to/data.csv --epochs 300 --batch-size 256 --lr 1e-4
```

Artifacts produced under `artifacts/`:

```
artifacts/checkpoints/latest_checkpoint.pt
artifacts/checkpoints/best_checkpoint.pt
artifacts/scalers/*.joblib
artifacts/logs/train.log
artifacts/tensorboard/...
artifacts/plots/loss_curve.png
artifacts/plots/regression_mae_curve.png
artifacts/plots/classification_accuracy_curve.png
artifacts/plots/lr_schedule.png
artifacts/training_history.json
```

View live training curves with:

```bash
tensorboard --logdir artifacts/tensorboard
```

Early stopping's patience counter is persisted inside each checkpoint, so
resuming training reflects true history rather than resetting patience.

---

## Prediction

Interactive terminal mode (prompts for all 9 readings, validates them,
prints a formatted report):

```bash
python predict.py
```

Batch mode from a CSV of readings (feature columns only):

```bash
python predict.py --input readings.csv --output report.csv
```

Example single-reading report:

```
==============================================================
EdgeGuard AI - Brake Health Prediction Report
==============================================================
Brake Health:                 78.42 %  (confidence std: 1.2031)
Remaining Pad Life:            9832.4 km (analytically calculated, not predicted)
Brake Fade Risk:             Medium       (confidence:  91.4%)
Maintenance Recommendation:  Inspect Soon         (confidence:  88.7%)
--------------------------------------------------------------
No sanity-check warnings.
--------------------------------------------------------------
Inference time: 4.87 ms
==============================================================
```

Every input reading is validated **before** inference:
- per-feature physical range checks (`config.FEATURE_RANGES`)
- cross-field plausibility checks (e.g. a 700°C disc at 15 km/h, or 100%
  fluid level with near-zero hydraulic pressure) are rejected with a
  clear error message rather than silently fed to the model.

---

## Why this avoids the failure modes of earlier versions

| Previous failure mode | How v3 prevents it |
|---|---|
| Negative brake health | Sigmoid-bounded regression head + explicit clamp to [0,100] at every output stage |
| Negative / fixed pad life | Pad life is never predicted; it's derived analytically and clamped to `>= 0` |
| Always predicting one class | Balanced multi-task loss (MinMax-scaled regression target) prevents one loss term from starving the classification heads of gradient signal |
| Confidence always 100% | Regression confidence from Monte Carlo Dropout std; classification confidence from real softmax probabilities, blended with MC-averaged probabilities |
| Checkpoint / `weights_only` load errors | `torch.load(..., weights_only=False)` used deliberately on self-produced checkpoints, with atomic tmp-file writes |
| Early stopping blocking resume | `epochs_without_improvement` persisted in the checkpoint and restored on resume |
| classification_report class mismatch | All classification metrics computed with an explicit `labels=` argument spanning the full fixed class set |
| Hardcoded feature names / magic numbers | All feature names, ranges, and engineering constants live in `config.py` only |

---

## Extending with a better dataset

Because every architectural choice (input dimensionality, output heads,
scaling strategy, loss weighting) is driven by `config.py` rather than
hardcoded, swapping in a larger or higher-quality real-world CSV (same 12
fixed columns) and re-running `train.py` is the only step required to
improve model quality — no architectural changes needed.
