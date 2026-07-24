"""
EdgeGuard AI - Brake Health Prediction Dataset Generator
==========================================================

Simulates a fleet of virtual vehicles brake-event-by-brake-event, so
each row of the resulting CSV is ONE braking event drawn from a
continuously evolving vehicle state (pad thickness, disc/fluid
temperature, fluid level, mileage, usage count). Nothing is randomly
labeled: every output column (Brake Health, Remaining Pad Life, Brake
Fade Risk, Maintenance Action) is computed deterministically from the
current physical state using engineering formulas.

Randomness is used ONLY to pick which driving scenario a given
braking event belongs to (drivers don't drive the same way every
time) and to apply small sensor measurement noise at the very end.
The underlying wear/thermal/fluid physics are fully deterministic
given the scenario.

Run:
    python dataset_generator.py

The script will interactively ask for:
    - number of virtual vehicles
    - number of braking events per vehicle
    - output CSV path
(press Enter on any prompt to accept the shown default)
"""

import sys
import numpy as np
import pandas as pd

# ==================================================================
# GLOBAL ENGINEERING CONSTANTS
# ==================================================================
NEW_PAD_MM = 12.0          # brand-new pad thickness
CRITICAL_PAD_MM = 1.0      # below this -> pad considered fully failed
REPLACE_PAD_MM = 3.0       # replacement threshold
DANGER_PAD_MM = 2.0        # dangerous zone
USABLE_MM = NEW_PAD_MM - CRITICAL_PAD_MM  # usable material range for life curve

TYPICAL_LIFE_KM = 50_000.0  # reference full pad life (new pad, average driving)

SEED = 42

# Sensor noise magnitudes (applied at the very end, to reported values only)
NOISE_TEMP_C = 2.0
NOISE_PRESSURE_BAR = 2.0
NOISE_FLUID_PCT = 1.0
NOISE_PAD_MM = 0.05
NOISE_SPEED_KMH = 1.0

# ==================================================================
# DRIVING SCENARIOS
# ==================================================================
# Each scenario defines the ranges braking-event inputs are drawn
# from, plus multipliers that drive the wear and heat models.
#
#   speed_range_kmh     : vehicle speed during this braking event
#   pedal_range_pct     : brake pedal force applied
#   pressure_range_bar  : hydraulic pressure produced
#   wear_mult           : relative pad-wear severity of this scenario
#   heat_mult           : relative heat-generation severity
#   ambient_delta_c     : typical ambient temperature shift for this
#                         scenario (weather-linked scenarios shift it;
#                         pure driving-style scenarios leave it at 0)
#   mileage_inc_km      : distance covered leading up to this braking
#                         event (varies with typical scenario speed)
SCENARIOS = {
    "City Driving":       dict(speed=(10, 50),  pedal=(15, 45), pressure=(20, 60),
                                wear_mult=1.00, heat_mult=0.80, ambient_delta=0.0,
                                mileage_inc=(0.10, 0.40), weight=0.18),
    "Highway":            dict(speed=(80, 130), pedal=(10, 35), pressure=(15, 45),
                                wear_mult=0.55, heat_mult=0.55, ambient_delta=0.0,
                                mileage_inc=(1.00, 3.00), weight=0.16),
    "Mountain Road":      dict(speed=(30, 80),  pedal=(40, 80), pressure=(60, 120),
                                wear_mult=1.60, heat_mult=1.40, ambient_delta=0.0,
                                mileage_inc=(0.50, 1.50), weight=0.07),
    "Downhill":           dict(speed=(40, 100), pedal=(50, 90), pressure=(70, 140),
                                wear_mult=1.85, heat_mult=1.70, ambient_delta=0.0,
                                mileage_inc=(0.50, 1.20), weight=0.05),
    "Traffic Jam":        dict(speed=(0, 20),   pedal=(20, 50), pressure=(15, 40),
                                wear_mult=1.10, heat_mult=0.45, ambient_delta=0.0,
                                mileage_inc=(0.02, 0.10), weight=0.12),
    "Emergency Braking":  dict(speed=(40, 130), pedal=(85, 100), pressure=(140, 190),
                                wear_mult=3.50, heat_mult=2.00, ambient_delta=0.0,
                                mileage_inc=(0.05, 0.20), weight=0.03),
    "Aggressive Driving": dict(speed=(50, 140), pedal=(55, 95), pressure=(90, 160),
                                wear_mult=2.00, heat_mult=1.50, ambient_delta=0.0,
                                mileage_inc=(0.30, 1.00), weight=0.10),
    "Eco Driving":        dict(speed=(20, 60),  pedal=(8, 25),  pressure=(10, 30),
                                wear_mult=0.40, heat_mult=0.40, ambient_delta=0.0,
                                mileage_inc=(0.20, 0.60), weight=0.12),
    "Rain":               dict(speed=(20, 70),  pedal=(30, 60), pressure=(40, 90),
                                wear_mult=1.20, heat_mult=0.70, ambient_delta=-3.0,
                                mileage_inc=(0.20, 0.60), weight=0.07),
    "Hot Weather":        dict(speed=(30, 90),  pedal=(20, 50), pressure=(30, 70),
                                wear_mult=1.30, heat_mult=1.30, ambient_delta=12.0,
                                mileage_inc=(0.20, 0.70), weight=0.05),
    "Cold Weather":       dict(speed=(20, 70),  pedal=(25, 55), pressure=(35, 80),
                                wear_mult=1.10, heat_mult=0.90, ambient_delta=-12.0,
                                mileage_inc=(0.20, 0.60), weight=0.05),
}
SCENARIO_NAMES = list(SCENARIOS.keys())
SCENARIO_WEIGHTS = np.array([SCENARIOS[s]["weight"] for s in SCENARIO_NAMES])
SCENARIO_WEIGHTS = SCENARIO_WEIGHTS / SCENARIO_WEIGHTS.sum()

# ==================================================================
# VEHICLE "PERSONALITIES"
# ==================================================================
# Each virtual vehicle is assigned a driving personality that biases
# which scenarios it tends toward and how fast it burns through pad
# material and remaining-life estimate. This is what produces a
# BALANCED dataset spanning brand-new brakes all the way to emergency
# failures within a fixed number of events per vehicle: gentle
# drivers stay mostly healthy, aggressive drivers wear out and hit
# critical/emergency states well before their 1000th event.
PERSONALITIES = {
    "Gentle":     dict(wear_scale=0.55, life_severity=1.20,
                        scenario_bias={"Eco Driving": 2.2, "City Driving": 1.3,
                                       "Highway": 1.2, "Emergency Braking": 0.3,
                                       "Aggressive Driving": 0.3, "Downhill": 0.5,
                                       "Mountain Road": 0.5}),
    "Average":    dict(wear_scale=1.00, life_severity=1.00, scenario_bias={}),
    "Aggressive": dict(wear_scale=1.9, life_severity=0.70,
                        scenario_bias={"Aggressive Driving": 2.5, "Emergency Braking": 2.5,
                                       "Mountain Road": 1.8, "Downhill": 1.8,
                                       "Eco Driving": 0.2}),
}
PERSONALITY_NAMES = list(PERSONALITIES.keys())
PERSONALITY_PROBS = np.array([0.30, 0.40, 0.30])

# ==================================================================
# OUTPUT COLUMNS (EXACT NAMES, EXACT ORDER - do not change)
# ==================================================================
COLUMNS = [
    "Brake Pad Thickness (mm)",
    "Brake Disc Temperature (C)",
    "Brake Fluid Level (%)",
    "Brake Fluid Temperature (C)",
    "Hydraulic Pressure (bar)",
    "Vehicle Speed (km/h)",
    "Wheel Speed (km/h)",
    "Brake Pedal Force (%)",
    "Ambient Temperature (C)",
    "Total Mileage (km)",
    "Brake Usage Count",
    "Brake Health (%)",
    "Remaining Pad Life (km)",
    "Brake Fade Risk",
    "Maintenance Action",
]


def pick_scenario(rng, personality, previous_scenario):
    """Choose a driving scenario for one braking event, biased by the
    vehicle's personality (e.g. aggressive drivers pick 'Aggressive
    Driving' / 'Emergency Braking' far more often). Scenarios persist
    across consecutive events with 70% probability, since real trips
    involve sustained stretches (a downhill run or a highway cruise
    lasts many braking events, not just one) - this is what lets heat
    genuinely build up instead of resetting every single event."""
    if previous_scenario is not None and rng.random() < 0.70:
        return previous_scenario

    bias = PERSONALITIES[personality]["scenario_bias"]
    weights = SCENARIO_WEIGHTS.copy()
    for i, name in enumerate(SCENARIO_NAMES):
        if name in bias:
            weights[i] *= bias[name]
    weights = weights / weights.sum()
    return rng.choice(SCENARIO_NAMES, p=weights)


def compute_brake_health(pad_thickness, fluid_level, disc_temp, mileage, usage_count):
    """Composite, smooth 0-100 health score. Pad thickness dominates
    (it is the single biggest driver of stopping-performance margin);
    fluid condition, disc heat and long-term fatigue contribute the
    rest. No discontinuities - pure weighted exponential/linear terms."""
    pad_score = 100.0 * (pad_thickness / NEW_PAD_MM)
    fluid_score = np.clip(fluid_level, 0, 100)
    disc_score = 100.0 if disc_temp <= 250 else 100.0 * np.exp(-(disc_temp - 250) / 300.0)
    usage_score = 100.0 * np.exp(-usage_count / 15000.0)
    mileage_score = 100.0 * np.exp(-mileage / 300000.0)

    health = (
        0.55 * pad_score
        + 0.15 * fluid_score
        + 0.12 * disc_score
        + 0.10 * usage_score
        + 0.08 * mileage_score
    )
    return float(np.clip(health, 0, 100))


def compute_remaining_pad_life(pad_thickness, life_severity):
    """Nonlinear remaining-life curve calibrated against the reference
    table (12mm->50000km, 10->40000, 8->30000, 6->20000, 4->10000,
    3->3000, 2->500, 1->0), then scaled by the vehicle's own average
    wear severity (harder-driven vehicles burn remaining life faster
    for the same thickness; gentle drivers stretch it further).
    Guaranteed >= 0 and, for a fixed severity, strictly non-decreasing
    in thickness (so it can never rise while thickness falls)."""
    if pad_thickness <= CRITICAL_PAD_MM:
        return 0.0
    frac = np.clip((pad_thickness - CRITICAL_PAD_MM) / USABLE_MM, 0.0, 1.0)
    base_life = TYPICAL_LIFE_KM * (frac ** 1.3)
    life = base_life * life_severity
    return float(max(0.0, life))


def compute_fade_risk(disc_temp, fluid_temp, hydraulic_pressure, pedal_force, vehicle_speed):
    """Fade risk = how close the system is to boiling brake fluid /
    losing friction coefficient from excess heat, combined with how
    hard the brakes are currently working."""
    disc_c = np.clip((disc_temp - 150) / (650 - 150), 0, 1.3)
    fluid_c = np.clip((fluid_temp - 90) / (220 - 90), 0, 1.3)
    press_c = np.clip((hydraulic_pressure - 60) / (190 - 60), 0, 1.2)
    speed_c = np.clip((vehicle_speed - 40) / (140 - 40), 0, 1.0)
    pedal_c = np.clip((pedal_force - 40) / (100 - 40), 0, 1.0)

    score = 100.0 * (
        0.35 * disc_c + 0.25 * fluid_c + 0.15 * press_c + 0.15 * speed_c + 0.10 * pedal_c
    )

    if disc_temp >= 600:
        return "Critical", score
    if score < 15:
        return "Low", score
    if score < 35:
        return "Medium", score
    if score < 60:
        return "High", score
    if score < 90:
        return "Very High", score
    return "Critical", score


def compute_maintenance_action(brake_health, pad_thickness, fluid_level, disc_temp,
                                remaining_life):
    """Rule-based decision tree, most severe condition wins. Order of
    checks matters: emergency-level conditions are evaluated last so
    they always override softer recommendations."""
    action = "No Action"

    if pad_thickness < 5.0 or fluid_level < 65 or brake_health < 75:
        action = "Inspect Soon"

    if fluid_level < 45 or (disc_temp > 130 and fluid_level < 55):
        action = "Replace Brake Fluid"

    if REPLACE_PAD_MM > pad_thickness >= DANGER_PAD_MM:
        action = "Replace Brake Pads"

    if 500 < disc_temp <= 600:
        action = "Replace Brake Disc"

    if (DANGER_PAD_MM > pad_thickness >= CRITICAL_PAD_MM) or fluid_level < 35:
        action = "Immediate Service"

    if pad_thickness < CRITICAL_PAD_MM or disc_temp > 600 or remaining_life <= 0:
        action = "Emergency Stop"

    return action


def simulate_vehicle(vehicle_id, num_events, rng):
    """Run one vehicle through `num_events` sequential braking events,
    carrying physical state forward event-to-event, and return a list
    of row dicts (pre-noise, using EXACT column names)."""

    personality = rng.choice(PERSONALITY_NAMES, p=PERSONALITY_PROBS)
    wear_scale = PERSONALITIES[personality]["wear_scale"]
    life_severity = np.clip(PERSONALITIES[personality]["life_severity"], 0.5, 1.4)

    # Each vehicle lives in its own base climate.
    base_ambient = float(np.clip(rng.normal(22, 9), -10, 45))

    # ~10% of vehicles have a slow fluid seal leak independent of wear,
    # which is what lets fluid-level-triggered rules get exercised
    # even when pad wear alone would not (a leaky reservoir on an
    # otherwise gently-driven car).
    has_leak = rng.random() < 0.10
    leak_rate = rng.uniform(0.03, 0.09) if has_leak else 0.0

    # --- initial state: brand new vehicle ---
    pad_thickness = NEW_PAD_MM
    fluid_level = 100.0
    disc_temp = base_ambient
    fluid_temp = base_ambient
    mileage = 0.0
    usage_count = 0

    rows = []
    previous_scenario = None
    for _ in range(num_events):
        scenario_name = pick_scenario(rng, personality, previous_scenario)
        previous_scenario = scenario_name
        sc = SCENARIOS[scenario_name]

        # --- instantaneous braking-event inputs ---
        speed = rng.uniform(*sc["speed"])
        pedal = rng.uniform(*sc["pedal"])
        # hydraulic pressure is coupled to pedal force (physical
        # master-cylinder relationship) plus the scenario's own range
        pressure_base = rng.uniform(*sc["pressure"])
        pressure = 0.5 * pressure_base + 0.5 * (pedal / 100.0) * 190.0

        ambient = base_ambient + sc["ambient_delta"] + rng.normal(0, 1.0)
        mileage_inc = rng.uniform(*sc["mileage_inc"])

        # --- disc temperature: heats from this event, cools toward
        # ambient since the last event (never resets instantly) ---
        heat_generated = (
            165.0 * sc["heat_mult"]
            * (0.25 + 0.75 * (pedal / 100.0))
            * (0.25 + 0.75 * (pressure / 190.0))
        )
        # higher speed -> more airflow -> more cooling retained fraction is lower
        cooling_retention = np.clip(0.68 - 0.16 * (speed / 140.0), 0.42, 0.68)
        disc_temp = ambient + (disc_temp - ambient) * cooling_retention + heat_generated
        disc_temp = float(np.clip(disc_temp, ambient, 700))

        # --- fluid temperature lags disc temperature (slow conduction) ---
        fluid_temp = fluid_temp + 0.10 * (disc_temp - fluid_temp) + 0.02 * (ambient - fluid_temp)
        fluid_temp = float(np.clip(fluid_temp, ambient, 230))

        # --- pad wear this event (never increases pad thickness) ---
        intensity = (
            0.30 * (pedal / 100.0)
            + 0.30 * (pressure / 190.0)
            + 0.20 * (speed / 140.0)
            + 0.20 * (disc_temp / 650.0)
        )
        wear = 0.012 * sc["wear_mult"] * wear_scale * intensity
        pad_thickness = max(0.0, pad_thickness - wear)

        # --- fluid level: never increases; drained by pad wear (piston
        # travel, proportional to actual material consumed) plus
        # mileage aging plus optional seal leak ---
        fluid_drop = 2.0 * wear + 0.00006 * mileage_inc + leak_rate
        fluid_level = max(0.0, fluid_level - fluid_drop)

        # --- wheel speed: tracks vehicle speed, diverges slightly
        # (slip) under high hydraulic pressure / hard braking ---
        slip_frac = np.clip((pressure - 80) / 400.0, 0, 0.08)
        wheel_speed = max(0.0, speed * (1 - slip_frac))

        mileage += mileage_inc
        usage_count += 1

        brake_health = compute_brake_health(pad_thickness, fluid_level, disc_temp,
                                             mileage, usage_count)
        remaining_life = compute_remaining_pad_life(pad_thickness, life_severity)
        fade_risk, _ = compute_fade_risk(disc_temp, fluid_temp, pressure, pedal, speed)
        maint_action = compute_maintenance_action(brake_health, pad_thickness,
                                                   fluid_level, disc_temp, remaining_life)

        rows.append({
            "Brake Pad Thickness (mm)": pad_thickness,
            "Brake Disc Temperature (C)": disc_temp,
            "Brake Fluid Level (%)": fluid_level,
            "Brake Fluid Temperature (C)": fluid_temp,
            "Hydraulic Pressure (bar)": pressure,
            "Vehicle Speed (km/h)": speed,
            "Wheel Speed (km/h)": wheel_speed,
            "Brake Pedal Force (%)": pedal,
            "Ambient Temperature (C)": ambient,
            "Total Mileage (km)": mileage,
            "Brake Usage Count": usage_count,
            "Brake Health (%)": brake_health,
            "Remaining Pad Life (km)": remaining_life,
            "Brake Fade Risk": fade_risk,
            "Maintenance Action": maint_action,
            "_vehicle_id": vehicle_id,   # internal only, dropped before CSV export
        })

    return rows


def apply_sensor_noise(df, rng):
    """Apply small Gaussian sensor noise to the reported values only
    (the underlying simulation state used to drive the physics stays
    exact). Magnitudes match the requested tolerances."""
    n = len(df)

    def noisy(col, sigma):
        return df[col].to_numpy() + rng.normal(0, sigma, n)

    df["Brake Disc Temperature (C)"] = noisy("Brake Disc Temperature (C)", NOISE_TEMP_C)
    df["Brake Fluid Temperature (C)"] = noisy("Brake Fluid Temperature (C)", NOISE_TEMP_C)
    df["Ambient Temperature (C)"] = noisy("Ambient Temperature (C)", NOISE_TEMP_C / 2)
    df["Hydraulic Pressure (bar)"] = noisy("Hydraulic Pressure (bar)", NOISE_PRESSURE_BAR)
    df["Brake Fluid Level (%)"] = noisy("Brake Fluid Level (%)", NOISE_FLUID_PCT)
    df["Brake Pad Thickness (mm)"] = noisy("Brake Pad Thickness (mm)", NOISE_PAD_MM)
    df["Vehicle Speed (km/h)"] = noisy("Vehicle Speed (km/h)", NOISE_SPEED_KMH)
    df["Wheel Speed (km/h)"] = noisy("Wheel Speed (km/h)", NOISE_SPEED_KMH)

    # Clip back into physically valid ranges after noise
    df["Brake Pad Thickness (mm)"] = df["Brake Pad Thickness (mm)"].clip(0, NEW_PAD_MM)
    df["Brake Fluid Level (%)"] = df["Brake Fluid Level (%)"].clip(0, 100)
    df["Hydraulic Pressure (bar)"] = df["Hydraulic Pressure (bar)"].clip(0, None)
    df["Vehicle Speed (km/h)"] = df["Vehicle Speed (km/h)"].clip(0, None)
    df["Wheel Speed (km/h)"] = df["Wheel Speed (km/h)"].clip(0, None)
    df["Brake Disc Temperature (C)"] = df["Brake Disc Temperature (C)"].clip(-20, 720)
    df["Brake Fluid Temperature (C)"] = df["Brake Fluid Temperature (C)"].clip(-20, 240)

    return df


def enforce_hard_physical_constraints(df):
    """Belt-and-braces guarantee (survives sensor noise): within each
    vehicle's own event sequence, pad thickness and fluid level can
    only ever go down (or stay flat), mileage and usage count can
    only ever go up, and remaining pad life can only ever go down (or
    stay flat) alongside falling pad thickness."""
    df = df.sort_values(["_vehicle_id"], kind="stable").copy()

    def per_vehicle(group):
        group = group.copy()
        group["Brake Pad Thickness (mm)"] = group["Brake Pad Thickness (mm)"].cummin()
        group["Brake Fluid Level (%)"] = group["Brake Fluid Level (%)"].cummin()
        group["Remaining Pad Life (km)"] = group["Remaining Pad Life (km)"].cummin()
        group["Total Mileage (km)"] = group["Total Mileage (km)"].cummax()
        group["Brake Usage Count"] = group["Brake Usage Count"].cummax()
        return group

    df = df.groupby("_vehicle_id", group_keys=False).apply(per_vehicle)
    return df


def get_int_input(prompt, default):
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        raw = ""
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Not a valid integer, using default ({default}).")
        return default


def get_str_input(prompt, default):
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw if raw else default


def main():
    print("=" * 60)
    print("EdgeGuard AI - Brake Health Dataset Generator")
    print("=" * 60)

    num_vehicles = get_int_input("How many vehicles?", 100)
    events_per_vehicle = get_int_input("How many braking events per vehicle?", 1000)
    output_path = get_str_input("Output CSV path?", "brake_dataset.csv")

    if num_vehicles <= 0 or events_per_vehicle <= 0:
        print("Vehicle count and events per vehicle must be positive integers.")
        sys.exit(1)

    print(f"\nSimulating {num_vehicles} vehicles x {events_per_vehicle} braking events "
          f"= {num_vehicles * events_per_vehicle:,} rows ...")

    rng = np.random.default_rng(SEED)
    all_rows = []
    for vid in range(num_vehicles):
        all_rows.extend(simulate_vehicle(vid, events_per_vehicle, rng))

    df = pd.DataFrame(all_rows)

    # Apply realistic sensor noise to reported values
    df = apply_sensor_noise(df, rng)

    # Guarantee the hard monotonic physical constraints, even after noise
    df = enforce_hard_physical_constraints(df)

    # Recompute the four AI target labels from the FINAL (noisy,
    # constraint-enforced) physical state, so labels stay perfectly
    # consistent with the values actually present in the CSV.
    df["Brake Health (%)"] = [
        compute_brake_health(t, f, d, m, u)
        for t, f, d, m, u in zip(
            df["Brake Pad Thickness (mm)"], df["Brake Fluid Level (%)"],
            df["Brake Disc Temperature (C)"], df["Total Mileage (km)"],
            df["Brake Usage Count"],
        )
    ]
    df["Brake Fade Risk"] = [
        compute_fade_risk(d, ft, p, pf, s)[0]
        for d, ft, p, pf, s in zip(
            df["Brake Disc Temperature (C)"], df["Brake Fluid Temperature (C)"],
            df["Hydraulic Pressure (bar)"], df["Brake Pedal Force (%)"],
            df["Vehicle Speed (km/h)"],
        )
    ]
    df["Maintenance Action"] = [
        compute_maintenance_action(h, t, f, d, r)
        for h, t, f, d, r in zip(
            df["Brake Health (%)"], df["Brake Pad Thickness (mm)"],
            df["Brake Fluid Level (%)"], df["Brake Disc Temperature (C)"],
            df["Remaining Pad Life (km)"],
        )
    ]

    # Round for a clean, professional CSV
    round_map = {
        "Brake Pad Thickness (mm)": 2,
        "Brake Disc Temperature (C)": 1,
        "Brake Fluid Level (%)": 2,
        "Brake Fluid Temperature (C)": 1,
        "Hydraulic Pressure (bar)": 1,
        "Vehicle Speed (km/h)": 1,
        "Wheel Speed (km/h)": 1,
        "Brake Pedal Force (%)": 1,
        "Ambient Temperature (C)": 1,
        "Total Mileage (km)": 1,
        "Brake Health (%)": 2,
        "Remaining Pad Life (km)": 0,
    }
    for col, decimals in round_map.items():
        df[col] = df[col].round(decimals)
    df["Remaining Pad Life (km)"] = df["Remaining Pad Life (km)"].astype(int)
    df["Brake Usage Count"] = df["Brake Usage Count"].astype(int)

    # Drop the internal vehicle id - only the exact requested columns remain
    df = df[COLUMNS]

    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} rows to {output_path}")
    print("\nMaintenance Action distribution:")
    print(df["Maintenance Action"].value_counts())
    print("\nBrake Fade Risk distribution:")
    print(df["Brake Fade Risk"].value_counts())
    print("\nBrake Health (%) summary:")
    print(df["Brake Health (%)"].describe())


if __name__ == "__main__":
    main()
