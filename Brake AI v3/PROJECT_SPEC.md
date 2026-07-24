# EdgeGuard AI — Brake Health Prediction System
## Master Project Specification Document (Version 3)

**Document Type:** Master Architecture & Engineering Specification
**Intended Use:** Single source of truth to be uploaded into any Claude AI conversation so that every generated file (`config.py`, `dataset.py`, `model.py`, `train.py`, `predict.py`, `metrics.py`, `losses.py`, `utils.py`) is architecturally consistent with every other file, without needing to re-derive design decisions each time.
**Status:** Frozen for implementation. Any change to this document should be treated as a version bump (v3 → v3.1, etc.) and re-distributed to all chats before further code generation.

---

## 0. How to Use This Document

This specification is written the way a systems architecture document is written inside a Tier-1 automotive supplier (Bosch, Continental, Valeo) or an OEM ADAS/telematics group (Tesla, Mobileye) before a single line of production code is written. It is deliberately implementation-agnostic in places (it does not lock in a specific ML framework), but it is exact about:

- what data goes in and what predictions come out,
- what is learned by a neural network vs. what is computed by a deterministic engineering formula,
- the exact architecture, losses, training procedure, evaluation procedure, and file responsibilities.

Any Claude conversation generating code for this project should treat every section below as a **binding constraint**, not a suggestion. Where a future prompt conflicts with this document, this document wins unless the user explicitly says otherwise in that conversation.

---

## 1. Project Overview

**Project Name:** EdgeGuard AI – Brake Health Prediction System (Version 3)

**Goal:** Build a production-quality, edge-deployable AI system that ingests live brake-system telemetry from a vehicle and predicts:

1. **Brake Health (%)** — a continuous 0–100 regression score describing overall brake system condition.
2. **Brake Fade Risk** — a 5-class classification of how likely the brakes are to lose stopping power due to heat/pressure/speed conditions *right now*.
3. **Maintenance Action** — a 7-class classification recommending the appropriate workshop/driver action.

A fourth quantity, **Remaining Brake Pad Life (km)**, is reported to the driver/fleet operator alongside the three AI outputs above, but it is **never predicted by the neural network**. It is always derived afterward using deterministic engineering equations that take the *measured* Brake Pad Thickness and the *predicted* Brake Health as inputs (Section 5.1). This is a deliberate architectural decision, explained in Section 2.

**Target deployment:** Automotive edge compute (infotainment SoC, telematics control unit, or a dedicated ADAS/body-domain edge AI accelerator), running fully offline, with inference budgets suitable for embedded Linux + NPU/GPU-accelerated inference (see Section 14).

---

## 2. Design Philosophy

> **"Machine Learning + Physics + Automotive Engineering — never a black box where an equation already exists."**

EdgeGuard AI v3 follows three governing principles:

### 2.1 Learn only what cannot be reliably computed

Some brake-system quantities (e.g., how pad thickness relates to remaining kilometers) are governed by well-understood, publishable engineering relationships — friction material wear curves, thermal transfer, hydraulic principles. Where such a relationship exists and is trustworthy, **it is implemented as a deterministic formula, not learned**. The network is reserved for the genuinely difficult, multi-factor, non-linear pattern-recognition problem: fusing 9 simultaneous sensor channels into a single trustworthy health score and two risk/action classifications — a problem where hand-written rules would be brittle and where historical fleet data carries information a static formula cannot capture (sensor drift, component interaction effects, aging non-linearities).

Concretely, in EdgeGuard AI v3:

| Quantity | Computed by |
|---|---|
| Brake Health (%) | **Neural network** (regression head) |
| Brake Fade Risk | **Neural network** (classification head) |
| Maintenance Action | **Neural network** (classification head) |
| Remaining Brake Pad Life (km) | **Engineering equation** (Section 5.1), post-processing only |
| Brake Wear rate | **Engineering equation** (used only during synthetic dataset generation, Section 8) |

### 2.2 Why Remaining Pad Life is NOT predicted by the network (v3 change from v2)

In earlier iterations of this project, Remaining Pad Life was treated as one of several outputs the model could estimate directly. Version 3 deliberately removes it as a learned target for three engineering reasons:

1. **It is analytically well-defined.** Remaining pad life is fundamentally a function of how much friction material is physically left (Brake Pad Thickness) and the overall condition of the system (Brake Health, which already folds in disc heat, fluid condition, and usage fatigue). Nothing about the sensor-fusion problem the network is good at (finding a subtle joint pattern across 9 noisy channels) is needed to compute a value that mostly falls out of thickness and health directly.
2. **It removes an unnecessary error-compounding path.** If the network predicted Remaining Pad Life directly, its error would be the sum of (a) sensor-fusion error and (b) the network's own approximation of a formula it must implicitly re-derive from data. By instead deriving it deterministically from Brake Health (which the network *does* predict) and Brake Pad Thickness (measured directly, not predicted), the only error that can propagate into the reported life estimate is the Brake Health regression error — smaller, bounded, and easier to reason about at deployment time.
3. **It is safety-auditable.** A regulator, fleet safety officer, or dealership technician can be shown the exact equation used to compute Remaining Pad Life and verify it against known engineering references. They cannot be shown "the equation the neural network learned" in the same way. Keeping the safety-relevant, easily-gameable number as a transparent formula rather than a learned quantity is a defensible design choice in a functional-safety review.

### 2.3 Modularity, scalability, maintainability

The framework is split into single-responsibility files (Section 12) so that:
- the dataset generation logic can evolve (new scenarios, new sensor noise models) without touching the model,
- the model architecture can be swapped (e.g., adding a residual block, or migrating hidden sizes) without touching the training loop,
- the training loop can be modified (new schedulers, new logging) without touching loss definitions,
- inference/post-processing logic can be updated (new maintenance thresholds) without retraining being required, since the engineering-equation layer is decoupled from the network's weights.

---

## 3. Input Features — Exactly 9, No More, No Less

The neural network's input layer accepts **exactly 9 features**, in this fixed order. No additional column may be added to the model's input tensor; no column may be removed. Any additional telemetry present in the raw CSV (see Section 8.5 for the note on this) is used only during dataset generation / label computation, never fed to the network.

| # | Feature | Units | Range | Sensor Type (typical) |
|---|---|---|---|---|
| 1 | Brake Pad Thickness | mm | 0 – 12 | Wear sensor / eddy-current pad sensor |
| 2 | Brake Disc Temperature | °C | 20 – 800 | Infrared / thermocouple |
| 3 | Brake Fluid Level | % | 0 – 100 | Reservoir float sensor |
| 4 | Brake Fluid Temperature | °C | 20 – 250 | Immersed thermistor |
| 5 | Hydraulic Pressure | bar | 0 – 200 | Master-cylinder pressure transducer |
| 6 | Vehicle Speed | km/h | 0 – 220 | Wheel-speed-derived / GPS-fused |
| 7 | Brake Pedal Force | % | 0 – 100 | Pedal-effort / brake-by-wire sensor |
| 8 | Ambient Temperature | °C | -10 – 60 | Cabin/exterior thermometer |
| 9 | Total Mileage | km | 0 – 300,000 | Odometer / ECU mileage counter |

### 3.1 Why each feature matters, and how it affects brake health

**1. Brake Pad Thickness (mm)**
The single most direct physical indicator of remaining brake capability. Friction material is consumed irreversibly; once it approaches the backing plate, stopping distance increases and metal-on-metal contact risk (disc damage, total brake failure) rises sharply. This is why it dominates the Brake Health formula (Section 5.3) and is the primary driver of Remaining Pad Life (Section 5.1). It is monotonic — it can only decrease over a vehicle's life (barring a physical pad replacement event).

**2. Brake Disc Temperature (°C)**
Braking converts kinetic energy to heat at the disc/pad interface. Excessive disc temperature (>500–600 °C) degrades the friction coefficient of the pad material (brake fade), can warp the disc, and accelerates pad wear. It is the primary driver of Brake Fade Risk and interacts with Brake Fluid Temperature (heat conducts from disc → caliper → fluid).

**3. Brake Fluid Level (%)**
Brake fluid is the incompressible medium that transmits pedal force to the calipers. A falling fluid level indicates a leak, worn caliper seals, or (indirectly) pad wear (as pads wear, caliper pistons extend further, consuming fluid from the reservoir). Critically low fluid level risks a spongy or completely unresponsive pedal — one of the few brake failure modes that can be total and sudden rather than gradual.

**4. Brake Fluid Temperature (°C)**
Brake fluid absorbs moisture over time (hygroscopic), which lowers its boiling point. Combined with heat conducted from the disc, high fluid temperature raises the risk of fluid vaporization inside the lines — this produces compressible gas bubbles in an otherwise incompressible hydraulic system, which is the textbook mechanism of catastrophic brake fade (the pedal goes soft or to the floor with no stopping power). Fluid temperature lags disc temperature (thermal mass + distance), which is why the two are modeled with a lag relationship during data generation (Section 8.4).

**5. Hydraulic Pressure (bar)**
Directly reflects how hard the driver (or ADAS emergency braking system) is commanding the brakes. Sustained high pressure indicates aggressive or emergency braking, both a wear accelerant and a fade-risk accelerant. It is also a diagnostic signal: abnormally low pressure for a given pedal force can indicate a hydraulic system fault (though full fault diagnosis is out of scope for this model; that pattern is left for the network to partially learn as an anomaly signal, not explicitly rule-coded).

**6. Vehicle Speed (km/h)**
Determines the kinetic energy that must be dissipated during a braking event (energy ∝ speed²), which is why speed enters the heat-generation and wear equations non-linearly. It also affects airflow-based cooling of the disc between events (higher speed → more airflow → different cooling behavior than a stationary vehicle).

**7. Brake Pedal Force (%)**
The direct driver input. Combined with hydraulic pressure, this is one of the two "how hard is the driver braking right now" signals, and is central to both wear-rate and fade-risk calculations. The relationship between pedal force and hydraulic pressure is itself a diagnostic signal (a mismatch between the two, beyond a normal master-cylinder gain curve, can indicate a booster or line problem).

**8. Ambient Temperature (°C)**
Sets the thermal baseline the whole brake system cools toward. Very hot climates reduce the thermal headroom before fade risk becomes critical; very cold climates can affect brake fluid viscosity and pad bite characteristics on the first few brake applications of a trip ("cold bite" phenomena). It also normalizes disc/fluid temperature readings so the model does not confuse "hot climate, healthy brakes" with "cold climate, overheating brakes."

**9. Total Mileage (km)**
A long-horizon fatigue proxy. Beyond pad wear (already captured directly by Feature 1), total mileage correlates with age-related degradation the other 8 instantaneous sensor features don't directly capture: caliper slide-pin stiction, disc surface glazing/scoring, rubber hose stiffening, and general seal fatigue. It lets the model apply a slow, monotonic health discount that pad thickness and fluid level alone would miss for an aging-but-still-thick-padded vehicle.

---

## 4. Outputs

### 4.1 Brake Health (%) — Regression

- **Type:** Continuous regression.
- **Range:** 0–100 (0 = total brake system failure, 100 = brand-new/perfect condition).
- **Output activation:** Sigmoid scaled to [0, 100] (i.e., `100 * sigmoid(x)`), guaranteeing the network can never output a physically impossible value outside the valid range without needing post-hoc clipping. (A plain linear output head is acceptable only if post-processing always clips to [0, 100]; the sigmoid-scaled head is preferred because it removes the need for that safety net and gives smoother gradients near the bounds than hard clipping would.)
- **Semantics:** A single composite indicator of overall brake condition, folding in pad wear, disc heat, fluid condition, and long-term fatigue (mirrors the deterministic formula used to *label* the training data — see Section 5.3 — the network's job is to learn to reproduce and generalize that same real-world relationship from noisy live sensor readings, not to invent a new definition of "health").

### 4.2 Brake Fade Risk — Classification (5 classes)

| Class | Meaning |
|---|---|
| Low | Normal driving, no fade risk |
| Medium | Elevated thermal/pressure load, monitor |
| High | Sustained hard braking (e.g., mountain descent), fade risk building |
| Very High | Near-fade conditions, pedal feel degradation likely |
| Critical | Fade actively occurring or imminent — emergency-level heat/pressure |

- **Output activation:** Softmax over 5 logits.
- **Class balance note:** This is an intentionally imbalanced, safety-oriented class set — "Critical" should be rare in the real world but must never be missed when it occurs. See Section 9.6 for how this is handled in training (class weighting) and Section 10 for why per-class recall on "Critical" is tracked separately from overall accuracy.

### 4.3 Maintenance Action — Classification (7 classes)

| Class | Meaning | Typical trigger pattern |
|---|---|---|
| No Action | Everything nominal | High health, thick pads, good fluid |
| Inspect Soon | Early warning, not urgent | Moderate pad wear or fluid drop, health starting to trend down |
| Replace Brake Pads | Pad material below safe working range | Pad thickness in replacement band, health degraded mainly by thickness |
| Replace Brake Fluid | Fluid degraded/low, not yet a hydraulic emergency | Fluid level or fluid-temperature-driven degradation |
| Replace Brake Disc | Disc-specific thermal/wear damage suspected | Sustained high disc temperature short of failure, independent of pad thickness |
| Immediate Service | Multiple compounding issues, or single issue in the danger band | Pad thickness in danger band, or fluid critically low |
| Emergency Stop | Brakes are in, or about to enter, an unsafe-to-continue-driving state | Pad essentially gone, or disc temperature past failure threshold |

- **Output activation:** Softmax over 7 logits.
- **Design note:** These 7 classes are ordered from least to most severe by convention in all logging/UX, but the classifier itself is a standard multi-class (not ordinal-regression) head — see Section 6.5 for why a plain categorical softmax is preferred over an ordinal formulation for this project.

### 4.4 Remaining Brake Pad Life (km) — NOT an AI Output

Reported to the user, but computed **after** inference completes, using the deterministic formula in Section 5.1. It takes the network's Brake Health prediction and the raw (measured) Brake Pad Thickness as its two inputs. It is never part of the loss function, never part of the network's output layer, and never present in the model's `state_dict`.

---

## 5. Engineering Calculations

All constants below are named exactly as shown, so any `config.py` implementing them uses these identifiers verbatim, keeping every file that references them consistent.

```
NEW_PAD_MM        = 12.0     # brand-new pad thickness
CRITICAL_PAD_MM   = 1.0      # below this, pad considered fully failed
DANGER_PAD_MM     = 2.0      # dangerous zone lower bound
REPLACE_PAD_MM    = 3.0      # "replace soon" threshold
TYPICAL_LIFE_KM   = 50000.0  # reference full remaining-life ceiling, new pad, average driving
```

### 5.1 Remaining Brake Pad Life (km) — Post-Inference Engineering Formula

**Inputs:** Brake Pad Thickness (measured sensor value, mm), Brake Health (%) (the network's *prediction*, not a sensor value).

**Step 1 — Base life from thickness (reference curve).**
A calibrated piecewise curve maps pad thickness to a realistic remaining-life band. This curve is the same reference table used to generate/label the synthetic training data (Section 8.3), so the deployed formula and the data the network was trained on are self-consistent:

| Pad Thickness (mm) | Remaining Life Band (km) |
|---|---|
| 12 | 49,000 – 50,000 |
| 11 | 44,000 – 47,000 |
| 10 | 39,000 – 42,000 |
| 9 | 34,000 – 37,000 |
| 8 | 29,000 – 32,000 |
| 7 | 24,000 – 27,000 |
| 6 | 18,000 – 22,000 |
| 5 | 12,000 – 16,000 |
| 4 | 6,000 – 10,000 |
| 3 | 1,000 – 3,000 |
| ≤ 2 | 0 |

At inference time (a single live vehicle, not a batch of synthetic samples), the midpoint of the interpolated band is used — there is no reason to inject randomness into a number reported to a real driver. Values between table rows are obtained by linear interpolation (monotonic non-decreasing in thickness by construction), exactly as `numpy.interp` would compute it.

**Step 2 — Health-based adjustment.**
The predicted Brake Health further scales this base life within a bounded, mild range, so that two vehicles with identical pad thickness but different overall condition (fluid, heat, fatigue) are not reported as having identical remaining life:

```
health_factor = clip(0.85 + 0.15 * (brake_health / 100), 0.85, 1.00)
remaining_life_km = base_life_km(pad_thickness) * health_factor
```

Interpretation: a pad at a given thickness on an otherwise perfectly healthy brake system (Brake Health = 100) gets the full reference life; the same pad thickness on a system whose Brake Health is being dragged down by other factors (hot discs, low fluid, high fatigue) gets a life estimate discounted by up to 15%. The adjustment is deliberately small and one-sided (it can only reduce life, never inflate it above the thickness-based ceiling) so pad thickness — the ground-truth physical quantity — always remains the dominant driver, and Brake Health only fine-tunes around it.

**Step 3 — Hard bounds (always enforced, no exceptions).**
```
remaining_life_km = clip(remaining_life_km, 0, TYPICAL_LIFE_KM)
```
Remaining life must never be negative and must never exceed 50,000 km, regardless of any upstream prediction error.

### 5.2 Brake Wear Equation (used during synthetic dataset generation only)

Wear is modeled as a per-braking-event material loss, driven by a weighted "braking intensity" composite of the four sensors most physically responsible for frictional energy dissipation:

```
intensity = 0.30*(pedal_force/100) + 0.30*(pressure/190) + 0.20*(speed/140) + 0.20*(disc_temp/650)
wear_mm   = BASE_WEAR_RATE * scenario_wear_multiplier * vehicle_wear_scale * intensity
pad_thickness_new = max(0, pad_thickness_old - wear_mm)
```
- `scenario_wear_multiplier` — differs by driving scenario (city, highway, mountain, emergency braking, etc.), reflecting that identical pedal force produces different real wear depending on context (e.g., mountain descents sustain wear far longer per kilometer than city stop-and-go).
- `vehicle_wear_scale` — differs by simulated driver personality (gentle / average / aggressive), reflecting real-world variance in how different drivers wear identical hardware.
- Pad thickness is guaranteed monotonic non-increasing by construction (each step only subtracts, and the running minimum is enforced as a hard constraint downstream — see the companion dataset generator).

### 5.3 Brake Health (%) Formula (data-generation label; the network learns to reproduce this relationship)

Brake Health is a calibrated band keyed primarily off pad thickness (dominant physical driver), with fluid condition, disc heat, and long-term fatigue determining *where within that band* a given vehicle sits:

| Pad Thickness (mm) | Health Band (%) |
|---|---|
| 12 | 95 – 100 |
| 10 | 85 – 95 |
| 8 | 70 – 85 |
| 6 | 55 – 70 |
| 4 | 35 – 55 |
| 3 | 15 – 35 |
| 2 | 5 – 15 |
| 1 | 0 – 5 |
| 0 | 0 |

```
lo, hi = interpolate_band(pad_thickness)               # from the table above
fluid_frac    = clip(fluid_level, 0, 100) / 100
disc_frac     = 1.0 if disc_temp <= 250 else exp(-(disc_temp - 250) / 300)
usage_frac    = exp(-usage_count / 15000)
mileage_frac  = exp(-mileage / 300000)

position = 0.45*fluid_frac + 0.25*disc_frac + 0.18*usage_frac + 0.12*mileage_frac   # in [0, 1]
brake_health = lo + (hi - lo) * position
brake_health = clip(brake_health, 0, 100)
```
This is the exact formula used to generate ground-truth labels for training (see the companion dataset generator script). The network is trained to approximate this function directly from noisy sensor inputs — but note the network's 9 inputs do **not** include `usage_count` (Brake Usage Count is intentionally excluded from the model's input feature list per Section 3; it is used only at label-generation time as an engineering proxy, folded into `Total Mileage` correlation for the live model). This is discussed further in Section 8.5.

### 5.4 Brake Fade Risk Logic (data-generation label; the network learns to reproduce this relationship)

Fade risk is a weighted composite of five normalized sub-scores, each representing how close a physical quantity is to its fade-relevant danger zone:

```
disc_c  = clip((disc_temp - 150) / (650 - 150), 0, 1.3)
fluid_c = clip((fluid_temp - 90) / (220 - 90), 0, 1.3)
press_c = clip((pressure - 60) / (190 - 60), 0, 1.2)
speed_c = clip((speed - 40) / (140 - 40), 0, 1.0)
pedal_c = clip((pedal_force - 40) / (100 - 40), 0, 1.0)

score = 100 * (0.35*disc_c + 0.25*fluid_c + 0.15*press_c + 0.15*speed_c + 0.10*pedal_c)

if disc_temp >= 600:      risk = "Critical"
elif score < 15:          risk = "Low"
elif score < 35:          risk = "Medium"
elif score < 60:          risk = "High"
elif score < 90:          risk = "Very High"
else:                     risk = "Critical"
```
Disc temperature is weighted highest (0.35) because it is the direct physical cause of pad-material fade; fluid temperature is second (0.25) because it governs the separate (and more dangerous) fluid-boiling fade mechanism; pressure, speed, and pedal force contribute the remaining weight as indicators of how hard the system is currently working (fade risk is inherently forward-looking/contextual, not just a function of current temperature).

### 5.5 Maintenance Action Decision Logic (data-generation label; the network learns to reproduce this relationship)

A rule-based decision tree where the *most severe* matching condition always wins (later checks override earlier ones):

```
action = "No Action"
if pad_thickness < 5.0 or fluid_level < 65 or brake_health < 75:
    action = "Inspect Soon"
if fluid_level < 45 or (disc_temp > 130 and fluid_level < 55):
    action = "Replace Brake Fluid"
if DANGER_PAD_MM <= pad_thickness < REPLACE_PAD_MM:      # [2, 3) mm
    action = "Replace Brake Pads"
if 500 < disc_temp <= 600:
    action = "Replace Brake Disc"
if (CRITICAL_PAD_MM <= pad_thickness < DANGER_PAD_MM) or fluid_level < 35:   # [1, 2) mm
    action = "Immediate Service"
if pad_thickness < CRITICAL_PAD_MM or disc_temp > 600:    # < 1 mm
    action = "Emergency Stop"
```
Note that "remaining life ≤ 0" is deliberately **not** used as an Emergency Stop trigger condition, even though the Section 5.1 formula returns 0 for the entire pad range below 2 mm. If it were used, it would make "Immediate Service" (which is meant to represent the 1–2 mm band) unreachable, since remaining life is already 0 throughout that band. Emergency Stop is reserved for the genuinely worst physical states only: pad below 1 mm, or disc temperature above 600 °C.

---

## 6. Model Architecture

### 6.1 High-Level Shape

A **shared-trunk, multi-head neural network** (hard parameter sharing multi-task learning). One feature extractor consumes the 9 input features; three independent heads branch off the shared representation — one regression head (Brake Health) and two classification heads (Brake Fade Risk, Maintenance Action).

```
Input (9 features, standardized)
        │
        ▼
┌───────────────────────────┐
│  Shared Feature Extractor │   (Section 6.2)
└───────────────────────────┘
        │
        ├──────────────┬───────────────────┐
        ▼              ▼                   ▼
 Regression Head   Fade Risk Head   Maintenance Action Head
 (Section 6.3)     (Section 6.4)    (Section 6.4)
        │              │                   │
        ▼              ▼                   ▼
 Brake Health (%)  5-class softmax    7-class softmax
```

### 6.2 Shared Feature Extractor

| Layer | Output size | Activation | Notes |
|---|---|---|---|
| Input | 9 | — | Standardized (zero mean, unit variance) per feature — see Section 11.2 |
| Dense 1 | 128 | ReLU | + BatchNorm1d, + Dropout(0.20) |
| Dense 2 | 128 | ReLU | + BatchNorm1d, + Dropout(0.20), **residual add** from Dense 1 output (both are width 128, so the residual connection is a direct identity add — see 6.2.1) |
| Dense 3 | 64 | ReLU | + BatchNorm1d, + Dropout(0.15) |
| Shared representation | 64 | — | Fed to all three heads |

**6.2.1 Why a residual block here:** with only 9 inputs and a moderately deep trunk (3 dense blocks), vanishing-gradient risk is low in absolute terms, but the residual connection between Dense 1 and Dense 2 is included anyway because (a) it costs nothing computationally at this scale, (b) it measurably stabilizes training when three loss terms of different scales (Section 7) are being backpropagated through the same shared trunk simultaneously — the residual path gives the optimizer an easy "do nothing extra" route when one task's gradient is noisy, preventing that noise from as easily corrupting the features the other two tasks depend on. No residual connection is used between Dense 2 (128) and Dense 3 (64) since the width changes and a projection shortcut is not judged worth the added complexity at this model scale.

**6.2.2 Why BatchNorm + Dropout together:** BatchNorm stabilizes the input distribution to each layer, which matters here because the 9 raw features have very different natural ranges (e.g., Total Mileage 0–300,000 vs. Brake Fluid Level 0–100) even after standardization at the input, later activations can still drift. Dropout is included specifically because brake telemetry is fleet data with correlated sensor readings within a single vehicle's trip (many consecutive rows look alike) — without dropout, the model can overfit to specific vehicle/trip signatures rather than the general physical relationship, which is the single biggest generalization risk for this dataset (Section 8.6).

### 6.3 Regression Head — Brake Health (%)

| Layer | Output size | Activation |
|---|---|---|
| Dense | 32 | ReLU |
| Dense (output) | 1 | Sigmoid, scaled ×100 |

### 6.4 Classification Heads — Brake Fade Risk & Maintenance Action

Each head is structured identically (independent weights, same shape):

| Layer | Output size | Activation |
|---|---|---|
| Dense | 32 | ReLU + Dropout(0.10) |
| Dense (output) | 5 (Fade Risk) or 7 (Maintenance Action) | Softmax |

**6.4.1 Why separate heads rather than one combined classification head:** Brake Fade Risk and Maintenance Action are correlated but answer different questions (fade risk is a *current instantaneous thermal/hydraulic* state; maintenance action is a *longer-horizon component condition* recommendation). Combining them into a single 35-class head (5×7 cross-product) would need far more training data per combination to learn well and would make it impossible for the model to express, e.g., "Low fade risk right now, but Replace Brake Pads is still the correct recommendation" (a very common and important real combination — a car can be driving gently, with no fade risk at all, on pads that are nonetheless critically worn).

### 6.5 Why plain categorical softmax, not ordinal regression, for Maintenance Action

Although the 7 classes have a natural "severity order," an ordinal-regression formulation (e.g., a single scalar threshold-based head) was considered and rejected, because the true decision boundaries are not one-dimensional: "Replace Brake Fluid" and "Replace Brake Pads" are not strictly orderable against each other (a fluid problem and a pad problem are different failure modes, not different severities of the same failure mode) even though each is individually orderable against "No Action" and "Emergency Stop." A categorical softmax lets the model represent this branching structure; a strict ordinal model would force an artificial total ordering onto what is really a partial order.

### 6.6 Weight Initialization

- All Dense layers: **He (Kaiming) normal initialization**, matching the ReLU activations used throughout the trunk and heads (He initialization is derived specifically for ReLU-family activations and avoids the systematic variance shrinkage that Xavier/Glorot initialization would introduce here).
- Output layer biases:
  - Regression head output bias initialized to `0` (sigmoid center ≈ 50%, a neutral starting guess before training).
  - Classification head output biases initialized to `0` (uniform prior across classes before training begins; class imbalance is instead handled via loss weighting, Section 7.3, not via biased initialization, to keep the model's initial state auditable and simple).
- BatchNorm layers: scale (`gamma`) initialized to `1`, shift (`beta`) initialized to `0` (standard).

---

## 7. Loss Functions

### 7.1 Per-Task Losses

| Task | Loss | Why |
|---|---|---|
| Brake Health (%) | **Huber loss** (a.k.a. Smooth L1), delta = 5.0 | Behaves like MSE near-zero error (smooth gradients for fine-tuning close predictions) but like MAE for large errors (robust to the occasional synthetic-data outlier or sensor-noise spike, which would otherwise dominate an MSE loss due to the squared penalty) |
| Brake Fade Risk | **Weighted categorical cross-entropy** | Standard multi-class loss; weighted to counter the natural rarity of "Critical" (Section 7.3) |
| Maintenance Action | **Weighted categorical cross-entropy** | Same reasoning; weighted to counter the natural rarity of "Immediate Service" / "Replace Brake Disc" / "Emergency Stop" relative to "No Action" |

### 7.2 Combined Multi-Task Loss

```
L_total = w_health * L_health + w_fade * L_fade + w_maint * L_maint

w_health = 2.0
w_fade   = 1.0
w_maint  = 1.0
```

**Why regression receives a higher weight (2.0 vs. 1.0):** Brake Health is not just one of three independent outputs — it is the upstream input to the Section 5.1 Remaining Pad Life formula, and the Maintenance Action ground-truth labels themselves are partially defined in terms of Brake Health (Section 5.5). An error in the Brake Health regression therefore has a larger downstream blast radius than an equivalent-sized error confined to one classification head: it silently degrades the reported Remaining Pad Life *and* correlates with mistakes on Maintenance Action, whereas a Fade Risk misclassification stays contained to that single output. Weighting the regression loss higher tells the optimizer to prioritize getting this foundational, multi-consumer quantity right before fine-tuning the classification boundaries around it. This also compensates for a well-known multi-task-learning failure mode where a scalar regression loss (naturally small in magnitude once training progresses) gets numerically drowned out by cross-entropy losses (which stay larger in magnitude across a wider portion of training) unless explicitly up-weighted.

### 7.3 Class Weights (Fade Risk & Maintenance Action)

Class weights are computed once from the training set as **inverse square-root frequency**, not plain inverse frequency:

```
w_c = 1 / sqrt(count_c)
w_c = w_c / mean(w_c)     # renormalize so weights average to 1.0
```

Plain inverse-frequency weighting is deliberately avoided because, combined with the intentionally imbalanced classes described in Sections 4.2/4.3, it would overcorrect and push the model to over-predict rare classes (e.g., "Critical" fade risk on borderline-normal inputs), which is its own safety problem (alert fatigue causes drivers/fleets to start ignoring warnings). The square-root dampening keeps rare, safety-critical classes meaningfully up-weighted without causing this overcorrection.

### 7.4 Loss Weight Tuning Process

`w_health`, `w_fade`, `w_maint` are treated as hyperparameters, not fixed forever — `config.py` exposes them as named constants so they can be swept during early experimentation, but 2.0 / 1.0 / 1.0 is the frozen v3 default and should be the starting point for any new training run unless a documented experiment justifies changing it.

---

## 8. Dataset Specification

### 8.1 CSV Column Order (exact, do not reorder/rename/add/remove)

```
Brake Pad Thickness (mm)
Brake Disc Temperature (C)
Brake Fluid Level (%)
Brake Fluid Temperature (C)
Hydraulic Pressure (bar)
Vehicle Speed (km/h)
Wheel Speed (km/h)
Brake Pedal Force (%)
Ambient Temperature (C)
Total Mileage (km)
Brake Usage Count
Brake Health (%)
Remaining Pad Life (km)
Brake Fade Risk
Maintenance Action
```

### 8.2 Units, Ranges, and Types

| Column | Type | Units | Valid Range |
|---|---|---|---|
| Brake Pad Thickness | float | mm | 0 – 12 |
| Brake Disc Temperature | float | °C | ambient – 800 |
| Brake Fluid Level | float | % | 0 – 100 |
| Brake Fluid Temperature | float | °C | ambient – 250 |
| Hydraulic Pressure | float | bar | 0 – 200 |
| Vehicle Speed | float | km/h | 0 – 220 |
| Wheel Speed | float | km/h | 0 – 220 |
| Brake Pedal Force | float | % | 0 – 100 |
| Ambient Temperature | float | °C | -10 – 60 |
| Total Mileage | float | km | 0 – 300,000 |
| Brake Usage Count | int | count | 0 – unbounded (monotonic per vehicle) |
| Brake Health | float | % | 0 – 100 |
| Remaining Pad Life | int | km | 0 – 50,000 |
| Brake Fade Risk | categorical | — | {Low, Medium, High, Very High, Critical} |
| Maintenance Action | categorical | — | {No Action, Inspect Soon, Replace Brake Pads, Replace Brake Fluid, Replace Brake Disc, Immediate Service, Emergency Stop} |

### 8.3 Data Validation Rules

Every row must satisfy, and every generator/loader must actively enforce or reject-and-repair:

1. `0 <= Brake Health <= 100`
2. `0 <= Remaining Pad Life <= 50000`
3. `0 <= Brake Pad Thickness <= 12`
4. `Brake Disc Temperature >= Ambient Temperature`
5. `Brake Fluid Temperature >= Ambient Temperature`
6. Within a single vehicle's row sequence: `Brake Usage Count` never decreases, `Total Mileage` never decreases, `Brake Pad Thickness` never increases, `Brake Fluid Level` never increases, `Remaining Pad Life` never increases while thickness is falling.

### 8.4 Sensor Noise Model

Gaussian noise applied to *reported* values only (the underlying simulated physical state used to derive labels stays exact, then noise is layered on top, then values are re-clipped into valid ranges):

| Signal | Noise (σ) |
|---|---|
| Pad Thickness | ±0.03 mm |
| Disc / Fluid / Ambient Temperature | ±2 °C (±1 °C for Ambient) |
| Hydraulic Pressure | ±2 bar |
| Vehicle / Wheel Speed | ±1 km/h |
| Brake Fluid Level | ±1 % |

### 8.5 Note on the 9 Model Inputs vs. the 15 CSV Columns

The CSV intentionally contains more columns (15) than the model consumes as input (9). `Wheel Speed (km/h)` and `Brake Usage Count` are present in the CSV for two reasons: (a) they are genuinely useful engineering/diagnostic telemetry (wheel-speed divergence from vehicle speed is a slip indicator; usage count is a fatigue proxy used when *labeling* Brake Health during data generation, Section 5.3), and (b) keeping them in the CSV preserves compatibility with the existing dataset generator and any downstream analytics/telemetry tooling that already expects this 15-column schema. `dataset.py` is responsible for selecting the exact 9 modeling columns (Section 3) out of the 15 when constructing the model's input tensor — the other 6 columns (Wheel Speed, Brake Usage Count, and the 4 output/label columns) are never concatenated into the input feature vector.

### 8.6 Realistic Distributions: Driver & Vehicle Profiles

Synthetic data is generated by simulating a fleet of virtual vehicles **event-by-event** (each row = one braking event), carrying forward a continuous physical state per vehicle (pad thickness, disc/fluid temperature, fluid level, mileage, usage count), rather than sampling each row independently and identically. This is essential for realism: real brake telemetry is autocorrelated within a trip and within a vehicle's life, and a model (or a human reviewer) can trivially tell independently-sampled synthetic rows apart from real fleet data.

**Driver personalities** (assigned once per simulated vehicle, bias which scenarios are chosen and how fast wear accumulates):
- **Gentle** — favors Eco/City/Highway driving, wears brakes slowly, stretches remaining life.
- **Average** — no scenario bias, baseline wear rate.
- **Aggressive** — favors Aggressive Driving/Emergency Braking/Mountain/Downhill scenarios, wears brakes fast, burns remaining life faster per mm of thickness lost.

**Driving scenarios** (govern the instantaneous ranges that speed, pedal force, and hydraulic pressure are drawn from for a given braking event, plus wear/heat multipliers): City Driving, Highway, Mountain Road, Downhill, Traffic Jam, Emergency Braking, Aggressive Driving, Eco Driving, Rain, Hot Weather, Cold Weather. Scenarios persist across consecutive events with high probability (real trips involve sustained stretches — a downhill run lasts many braking events, not one), which is what allows heat to genuinely build up rather than resetting every row.

**Vehicle initialization mix** (ensures the dataset is not solely "brand-new vehicles wearing down within the simulated window," which would under-represent mid-life and critical states): a portion of simulated vehicles start brand-new, a portion start already moderately used (mid-range pad thickness/fluid/mileage), and a small portion start already near end-of-life (pad thickness below 2 mm, low fluid) — guaranteeing the dataset contains a healthy volume of Inspect Soon / Replace / Immediate Service / Emergency Stop examples without needing an impractically large number of simulated events per vehicle.

### 8.7 Column-Level Generation Notes

- **Disc Temperature:** heats from the current event's braking energy, cools toward ambient since the last event using a speed-dependent cooling-retention factor (higher speed → more airflow → lower retention) — never resets instantly to ambient between events.
- **Fluid Temperature:** lags Disc Temperature (slow thermal conduction through the caliper/line), modeled as a small fractional pull toward the current disc temperature each event, plus a smaller pull toward ambient.
- **Pad Thickness:** decreases each event by an amount driven by the Section 5.2 wear formula; never increases.
- **Fluid Level:** decreases each event, driven by wear-proportional piston travel plus mileage aging plus (for a subset of vehicles) an independent slow seal-leak term; never increases.
- **Wheel Speed:** tracks Vehicle Speed, diverging slightly (slip) under high hydraulic pressure to simulate hard-braking wheel slip.

### 8.8 Scale

The dataset generator must be able to produce **100,000 rows within a few minutes** on ordinary hardware (no GPU requirement for data generation), since data generation is run far more frequently during iteration than full model training.

---

## 9. Training Framework

### 9.1 Random Seed

A single global seed (`SEED = 42`, matching the dataset generator's own seed constant for reproducibility across the whole pipeline) is set at the start of `train.py` for the framework's RNG, the data-shuffling RNG, and (where applicable) the deep-learning framework's own seed function. Every training run logs the seed used alongside its checkpoint/metrics so any result can be reproduced exactly.

### 9.2 Validation Strategy

- **Split:** 70% train / 15% validation / 15% test.
- **Split unit:** by **vehicle ID**, not by row. Because rows from the same simulated vehicle are highly autocorrelated (Section 8.6), a row-level random split would leak near-duplicate information between train and validation/test, producing an optimistic and misleading validation score. Splitting by vehicle guarantees the model is evaluated on entirely unseen vehicle life-histories.
- **Stratification:** the vehicle-level split is stratified by each vehicle's *dominant* Maintenance Action class (the most common label across that vehicle's rows), so all three splits contain a representative mix of healthy, worn, and critical vehicles rather than, say, all the "aggressive/critical" vehicles landing in one split by chance.

### 9.3 Transfer Learning

`train.py` supports initializing the shared feature extractor from a previously trained checkpoint (e.g., a v2 model, or a model trained on an earlier, smaller synthetic dataset), with the three task heads re-initialized fresh (Section 6.6). This lets future dataset revisions (new scenarios, new sensor noise characteristics) be incorporated without discarding everything the shared trunk has already learned about how the 9 raw sensor channels relate to each other.

### 9.4 GPU Support / Mixed Precision

Training runs on GPU when available, CPU otherwise, detected automatically (no manual flag required for the common case, but an explicit override flag is provided for forcing CPU during debugging). Mixed-precision training (FP16/BF16 compute with FP32 master weights) is enabled by default when a GPU is present — the model is small enough that mixed precision is primarily a training-speed optimization here rather than a memory necessity, but it's free performance with negligible accuracy risk at this model scale.

### 9.5 Checkpointing & Resume

- A checkpoint is written after every epoch that improves validation total loss, plus a final "last epoch" checkpoint regardless of whether it improved.
- Checkpoint filename convention: `edgeguard_v3_epoch{N:04d}_valloss{LOSS:.4f}.pt` (see Section 13 for the full naming convention).
- Each checkpoint stores: model weights, optimizer state, scheduler state, current epoch, best validation loss so far, the RNG seed, and the exact `config.py` values used for that run (so a checkpoint is always self-describing and never ambiguous about which hyperparameters produced it).
- `train.py --resume <checkpoint_path>` restores all of the above and continues training from the next epoch.

### 9.6 Early Stopping

Monitored metric: validation **total weighted loss** (Section 7.2). Patience: 15 epochs with no improvement beyond a minimum-delta threshold. On trigger, training stops and the best (not the last) checkpoint is restored as the final artifact — the last epoch before stopping is not assumed to be the best epoch.

### 9.7 Learning Rate Scheduler

**ReduceLROnPlateau**, monitoring the same validation total loss used for early stopping, factor 0.5, patience 5 epochs, minimum LR floor to prevent the scheduler from decaying the rate to a value so small that remaining training epochs become wasted compute.

### 9.8 Gradient Clipping

Global-norm gradient clipping at `max_norm = 5.0`, applied every step, before the optimizer step. This exists specifically because of the multi-task loss combination (Section 7.2) — three loss terms of different natural scales summed together can occasionally produce a large combined gradient spike early in training (before BatchNorm statistics have stabilized), and clipping prevents that spike from corrupting the shared trunk's weights.

### 9.9 Class Weight Computation Timing

Class weights (Section 7.3) are computed once, from the **training split only**, after the vehicle-level split (Section 9.2) is finalized — never from the full unsplit dataset, to avoid leaking test/validation class-frequency information into a training-time weighting decision.

---

## 10. Evaluation

### 10.1 Regression Metrics — Brake Health (%)

- **MAE** (mean absolute error, in percentage points) — the primary, most interpretable metric for stakeholders ("the model is off by X% on average").
- **RMSE** — reported alongside MAE specifically to surface whether large individual errors are occurring (RMSE grows faster than MAE when a few predictions are badly wrong), which matters because a single badly-wrong Brake Health prediction has outsized downstream impact (Section 7.2).
- **R²** — reported as a sanity-check goodness-of-fit summary, not the primary reported number (R² alone can look deceptively good on this kind of skewed, table-anchored target distribution).

### 10.2 Classification Metrics — Brake Fade Risk & Maintenance Action

- **Accuracy** — overall, reported but explicitly flagged as insufficient on its own given the intentional class imbalance (Sections 4.2/4.3).
- **Precision, Recall, F1 — per class**, not just macro-averaged. Recall on "Critical" (Fade Risk) and on "Emergency Stop" / "Immediate Service" (Maintenance Action) is reported separately and called out in every evaluation report, since missing these is the single most safety-relevant failure mode for this system — a false alarm ("No Action" vehicle flagged as "Inspect Soon") is a UX annoyance, but a missed "Emergency Stop" is a safety incident.
- **Confusion Matrix** — for both classification heads, always generated and saved as part of the evaluation artifact, since it shows *which* classes get confused with which (e.g., confirming that "Replace Brake Pads" vs. "Immediate Service" confusion, which are adjacent severity bands, is far more acceptable than "No Action" vs. "Emergency Stop" confusion, which should essentially never happen).
- **ROC curves** — one-vs-rest ROC/AUC per class, reported as a supplementary diagnostic (useful for threshold tuning if a future version wants to bias the decision boundary toward higher recall on "Critical"/"Emergency Stop" at the cost of some precision), not used as the primary go/no-go metric for model acceptance.

### 10.3 Acceptance Criteria (for promoting a checkpoint to "production candidate")

A checkpoint is only considered for production promotion if, on the held-out test split:
- Brake Health MAE is within an agreed engineering tolerance (tracked in the training run's report, not hard-coded here, since it should be revisited per dataset revision).
- Recall on Fade Risk "Critical" and Maintenance Action "Emergency Stop" both meet or exceed a minimum safety-recall floor agreed with the automotive safety review process for this project (again tracked per training run, not frozen in this document, since it is a safety-sign-off decision rather than a pure ML architecture decision).

---

## 11. Prediction Framework

### 11.1 Input Validation

Before any inference, `predict.py` validates every one of the 9 input features against the Section 3 range table. Out-of-range values are **not silently clipped** (unlike training-data generation, where clipping is appropriate for synthetic realism) — at inference time an out-of-range sensor reading is itself a diagnostic fact (sensor fault) and is surfaced to the caller as a validation error rather than masked. Missing/NaN values are rejected with a specific error identifying which feature was missing, never silently imputed.

### 11.2 Feature Scaling

The exact same standardization (per-feature mean/std) fitted on the **training split only** during `train.py` is persisted (e.g., as a small JSON/`.npz` alongside the checkpoint) and re-loaded by `predict.py`. Inference must never re-fit scaling statistics on live/incoming data — doing so would make predictions dependent on whatever batch of live data happened to arrive, which is both non-deterministic and physically nonsensical for a single-vehicle live inference call.

### 11.3 Inference

A single forward pass through the shared trunk and all three heads. Batch-of-one (single live vehicle reading) and batch-of-N (fleet/offline scoring) code paths are both supported behind the same function signature.

### 11.4 Post-Processing / Engineering Calculations

Immediately after the raw network outputs are produced:
1. Brake Health is already bounded to [0, 100] by the sigmoid output head (Section 6.3); no further clipping should be necessary, but a defensive clip is applied anyway as a final safety net.
2. Brake Fade Risk / Maintenance Action: `argmax` over the softmax outputs to obtain the predicted class label string.
3. **Remaining Pad Life is computed here** — this is the only point in the entire pipeline where the Section 5.1 formula runs at inference time, consuming the just-produced Brake Health prediction and the raw (validated) Brake Pad Thickness input.

### 11.5 Confidence Estimation

- **Regression (Brake Health):** reported alongside a simple prediction-interval heuristic derived from the validation-set residual distribution of the loaded checkpoint (e.g., ± the validation MAE, or ± 1 validation-residual standard deviation), not a learned uncertainty head — keeping this heuristic-based rather than adding a second learned uncertainty output avoids expanding the model's output surface for a v3 feature that is secondary to the core three predictions.
- **Classification (Fade Risk, Maintenance Action):** the raw softmax probability of the predicted (argmax) class is reported directly as a confidence score. Predictions where this confidence falls below a configurable threshold are flagged in the output as "low confidence" rather than being silently presented with the same visual weight as a high-confidence prediction — relevant for a system whose outputs may reach a driver-facing UI.

### 11.6 Output Formatting

A single structured result object/dict per prediction, containing at minimum: the 3 predicted quantities, the derived Remaining Pad Life, per-output confidence, a timestamp, and an echo of the input feature values actually used (post-validation) — sufficient for both a UI layer and an offline audit log to consume without needing to re-derive anything.

### 11.7 Error Handling

Three distinct, distinguishable error classes (not generic exceptions):
- **Input validation error** — a feature was out of range or missing (Section 11.1).
- **Model load error** — checkpoint file missing, corrupted, or architecture-mismatched with the currently loaded `model.py` (e.g., a checkpoint from a differently-shaped head).
- **Inference error** — any other failure during the forward pass or post-processing step (e.g., a NaN appearing mid-computation).

Each is logged with enough context (which feature, which file, which layer) to debug without needing to reproduce the failure interactively — important for a system that may eventually run unattended on an edge device far from a development machine.

---

## 12. Project Structure

```
edgeguard_ai_v3/
├── config.py            # All constants, hyperparameters, paths, class name lists — single source of truth
├── dataset.py            # PyTorch/TF Dataset & DataLoader construction, CSV loading, 9-feature selection,
│                          # standardization fit/apply, vehicle-level train/val/test split
├── model.py               # Shared trunk + 3 heads network definition (Section 6), weight init
├── losses.py               # Huber loss wrapper, weighted cross-entropy construction, combined multi-task loss
├── metrics.py               # MAE/RMSE/R², per-class precision/recall/F1, confusion matrix, ROC/AUC
├── train.py                  # Training loop: seeding, checkpointing, resume, early stopping, LR scheduler,
│                              # gradient clipping, mixed precision, calls into dataset.py/model.py/losses.py/metrics.py
├── predict.py                  # Inference entry point: validation, scaling, forward pass, post-processing,
│                                # confidence estimation, structured output (Section 11)
├── utils.py                     # Shared helpers: seeding utility, logging setup, checkpoint I/O helpers,
│                                # the Section 5.1/5.3/5.4/5.5 engineering-equation implementations (imported
│                                # by both the dataset generator's label logic and predict.py's post-processing,
│                                # so the two never drift apart into two different formulas)
├── requirements.txt              # Pinned dependency versions
├── README.md                      # Human-facing quickstart, mirrors (does not replace) this spec document
└── checkpoints/                    # Saved model artifacts (Section 9.5 naming convention)
```

### 12.1 File Responsibilities & Communication

- `config.py` is imported by every other file and by nothing else that isn't listed here — it has zero dependencies on the rest of the project, guaranteeing it can always be imported first without circular-import risk.
- `utils.py` holds the engineering formulas (Section 5) as the **single implementation** shared by the dataset generator (for label generation) and `predict.py` (for post-processing Remaining Pad Life). No file other than `utils.py` may contain its own copy of these formulas — if `dataset.py`/`predict.py` need them, they import from `utils.py`.
- `dataset.py` depends on `config.py` and `utils.py` only. It has no knowledge of the model architecture.
- `model.py` depends on `config.py` only. It has no knowledge of how data is loaded or how loss is computed — it only defines the forward pass and returns raw (regression, fade-logits, maintenance-logits) tuples.
- `losses.py` depends on `config.py` only. It is a pure function library: given predictions and targets, return the combined loss and its components for logging.
- `metrics.py` depends on `config.py` only. Same pure-function structure as `losses.py`.
- `train.py` is the only file that imports `dataset.py`, `model.py`, `losses.py`, and `metrics.py` together and orchestrates them — it contains no formulas, no architecture definitions, and no metric-computation logic of its own.
- `predict.py` imports `config.py`, `model.py`, and `utils.py` — deliberately **not** `dataset.py` (predict-time scaling statistics are loaded from the persisted artifact produced by `train.py`/`dataset.py`, not recomputed) and not `losses.py`/`metrics.py` (no loss or metric computation happens at inference time).

---

## 13. Coding Standards

### 13.1 Naming Conventions

- **Modules/files:** `snake_case.py`, matching Section 12 exactly.
- **Functions:** `snake_case`, verb-first (`compute_brake_health`, `validate_inputs`, `load_checkpoint`).
- **Classes:** `PascalCase` (`EdgeGuardModel`, `BrakeDataset`, `MultiTaskLoss`).
- **Constants:** `UPPER_SNAKE_CASE`, always defined in `config.py`, never redefined locally in another file (import, don't duplicate).
- **Folders:** lowercase, `snake_case` if multi-word (`checkpoints/`, `logs/`, `configs/`).
- **Checkpoint files:** `edgeguard_v3_epoch{N:04d}_valloss{LOSS:.4f}.pt` (Section 9.5); the "best" checkpoint is additionally symlinked/copied to `edgeguard_v3_best.pt` for unambiguous downstream reference by `predict.py`.

### 13.2 Logging

Standard library `logging` (or framework-equivalent), never bare `print()`, for anything beyond a short-lived interactive script. One logger per module (`logging.getLogger(__name__)`). Training logs include, at minimum, per-epoch: all three individual losses, the combined weighted loss, current learning rate, and the headline validation metrics (Section 10). Log level `INFO` for normal training progress, `WARNING` for recoverable anomalies (e.g., a row failing validation and being repaired), `ERROR` for anything that stops the run.

### 13.3 Documentation & Comments

Every public function/class gets a docstring stating: what it does, its inputs/outputs (including units for any physical quantity), and — for anything implementing an equation from Section 5 — which section of this document it implements, so a future reader can always trace code back to the engineering rationale rather than just the formula. Inline comments are reserved for explaining *why*, not *what* (the code itself should make the "what" clear through naming).

### 13.4 Exception Handling

No bare `except:`. Catch the most specific exception type available. Every raised exception in `predict.py` uses one of the three custom exception classes from Section 11.7 (`InputValidationError`, `ModelLoadError`, `InferenceError`), defined once in `utils.py` and imported everywhere they're raised or caught, so calling code (a future infotainment UI layer, a fleet dashboard backend) can reliably branch on exception type rather than parsing error message strings.

---

## 14. Production Deployment

The architecture in Section 6 is deliberately kept simple (plain Dense/BatchNorm/Dropout layers, no custom ops, no dynamic control flow inside the forward pass, fixed 9-element input shape, fixed 3-head output shape) specifically so it can be converted to an edge-inference format **without any architecture change**, only an export/conversion step:

### 14.1 ONNX Export

The trained model is exported to ONNX via the deep-learning framework's standard exporter, with the 9-feature input fixed at shape `[batch, 9]` (dynamic batch axis only — the feature axis is never dynamic, since it is architecturally fixed at exactly 9 by Section 3). The three outputs (Brake Health scalar, Fade Risk 5-logit vector, Maintenance Action 7-logit vector) are exported as three named ONNX graph outputs, preserving the multi-head structure rather than concatenating them into one opaque output tensor — this keeps the exported graph self-documenting for whichever edge runtime consumes it next.

### 14.2 TensorRT

The ONNX graph from 14.1 is converted to a TensorRT engine using `trtexec`/the TensorRT Python API, with FP16 (or INT8 with a small calibration set drawn from the training distribution) precision for inference-speed gains on NVIDIA-based automotive compute platforms (e.g., NVIDIA DRIVE). Because the network contains no custom layers, standard TensorRT ONNX-parser conversion is expected to succeed without needing custom plugin development.

### 14.3 Edge Deployment Targets

| Target | Path |
|---|---|
| **Embedded Linux (generic)** | ONNX Runtime (CPU or vendor NPU execution provider) |
| **NVIDIA Jetson** | ONNX → TensorRT engine (Section 14.2), running under JetPack |
| **Qualcomm Snapdragon (automotive cockpit/infotainment SoCs)** | ONNX → Qualcomm SNPE/QNN SDK conversion, targeting the Hexagon DSP/NPU for low-power always-on inference |
| **Generic automotive infotainment hardware** | ONNX Runtime as the common-denominator fallback runtime where a vendor-specific NPU toolchain isn't available |

### 14.4 What Never Changes Across Deployment Targets

- The 9-feature input contract (Section 3) and 3-output contract (Section 4.1–4.3).
- The Section 5 engineering formulas — these run as ordinary post-processing code on the host CPU (not inside the exported graph) on every target, since they are cheap, branchy, and not worth burning NPU/accelerator cycles or ONNX-graph complexity on. Only the neural network itself is exported/converted; Remaining Pad Life, and any input validation, are implemented as native code around the inference call on every target identically (reusing the same `utils.py`-equivalent logic, ported to whatever language the edge runtime's host application is written in).
- The feature standardization statistics (Section 11.2), persisted once at training time and baked into the edge application's pre-processing step identically across all targets.

### 14.5 What Is Allowed to Change Across Deployment Targets

- Numeric precision (FP32 during training/development → FP16/INT8 on constrained edge targets).
- The runtime/engine format (ONNX Runtime vs. TensorRT engine vs. SNPE/QNN container).
- Batching behavior (a live single-vehicle edge deployment runs batch-of-one; an offline fleet-scoring job on a server can batch arbitrarily) — the model and its ONNX export both already support this via the dynamic batch axis (Section 14.1), so no target-specific model change is required, only a different calling pattern.

---

## Appendix A — Summary Table of "Learned vs. Computed"

| Quantity | Learned (NN) | Computed (Engineering) | Section |
|---|---|---|---|
| Brake Health (%) | ✅ | — | 6.3 |
| Brake Fade Risk | ✅ | — | 6.4 |
| Maintenance Action | ✅ | — | 6.4 |
| Remaining Pad Life (km) | — | ✅ | 5.1 |
| Brake Wear (per event, dataset generation only) | — | ✅ | 5.2 |

## Appendix B — Frozen Constants Reference

```python
NEW_PAD_MM        = 12.0
CRITICAL_PAD_MM   = 1.0
DANGER_PAD_MM     = 2.0
REPLACE_PAD_MM    = 3.0
TYPICAL_LIFE_KM   = 50000.0
SEED              = 42

W_HEALTH = 2.0   # regression loss weight
W_FADE   = 1.0   # fade-risk classification loss weight
W_MAINT  = 1.0   # maintenance-action classification loss weight

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
```

---

*End of Master Project Specification, Version 3. Any future Claude AI conversation generating `config.py`, `dataset.py`, `model.py`, `losses.py`, `metrics.py`, `train.py`, `predict.py`, or `utils.py` for EdgeGuard AI should treat every numbered section above as binding unless the user explicitly instructs otherwise in that conversation.*

