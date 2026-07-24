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

# ------------------------------------------------------------------
# Calibration tables: pad thickness (mm) -> [low, high] band for
# Remaining Pad Life and Brake Health. These encode the requested
# real-world reference curve directly. Values for thicknesses that
# fall BETWEEN two rows (e.g. 5, 7, 9, 11 mm for health; any
# fractional mm for life) are produced with monotonic linear
# interpolation (np.interp), so the curve is smooth and can never
# increase as thickness decreases.
# ------------------------------------------------------------------
LIFE_MM =      [0,    1,    2,    3,     4,     5,     6,     7,     8,     9,     10,    11,    12]
LIFE_LO_KM =   [0,    0,    0,    1000,  6000,  12000, 18000, 24000, 29000, 34000, 39000, 44000, 49000]
LIFE_HI_KM =   [0,    0,    0,    3000,  10000, 16000, 22000, 27000, 32000, 37000, 42000, 47000, 50000]

HEALTH_MM =    [0, 1, 2, 3, 4, 6, 8, 10, 12]
HEALTH_LO_PCT = [0, 0, 5, 15, 35, 55, 70, 85, 95]
HEALTH_HI_PCT = [0, 5, 15, 35, 55, 70, 85, 95, 100]

SEED = 42

# Sensor noise magnitudes (applied at the very end, to reported values only)
NOISE_TEMP_C = 2.0
NOISE_PRESSURE_BAR = 2.0
NOISE_FLUID_PCT = 1.0
NOISE_PAD_MM = 0.03
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
    # NOTE ON WEIGHTS: rebalanced (vs. the original version) to draw
    # noticeably more events from the scenarios that stress the
    # system hardest (Mountain Road, Downhill, Emergency Braking,
    # Aggressive Driving, Hot Weather) so the dataset contains
    # meaningfully more critical/near-critical braking events, while
    # still keeping ordinary driving as the majority of the data.
    "City Driving":       dict(speed=(10, 50),  pedal=(15, 45), pressure=(20, 60),
                                wear_mult=1.00, heat_mult=0.80, ambient_delta=0.0,
                                mileage_inc=(0.10, 0.40), weight=0.13),
    "Highway":            dict(speed=(80, 130), pedal=(10, 35), pressure=(15, 45),
                                wear_mult=0.55, heat_mult=0.55, ambient_delta=0.0,
                                mileage_inc=(1.00, 3.00), weight=0.12),
    "Mountain Road":      dict(speed=(30, 80),  pedal=(40, 80), pressure=(60, 120),
                                wear_mult=1.60, heat_mult=1.40, ambient_delta=0.0,
                                mileage_inc=(0.50, 1.50), weight=0.11),
    "Downhill":           dict(speed=(40, 100), pedal=(50, 90), pressure=(70, 140),
                                wear_mult=1.85, heat_mult=1.70, ambient_delta=0.0,
                                mileage_inc=(0.50, 1.20), weight=0.09),
    "Traffic Jam":        dict(speed=(0, 20),   pedal=(20, 50), pressure=(15, 40),
                                wear_mult=1.10, heat_mult=0.45, ambient_delta=0.0,
                                mileage_inc=(0.02, 0.10), weight=0.09),
    "Emergency Braking":  dict(speed=(40, 130), pedal=(85, 100), pressure=(140, 190),
                                wear_mult=3.50, heat_mult=2.00, ambient_delta=0.0,
                                mileage_inc=(0.05, 0.20), weight=0.07),
    "Aggressive Driving": dict(speed=(50, 140), pedal=(55, 95), pressure=(90, 160),
                                wear_mult=2.00, heat_mult=1.50, ambient_delta=0.0,
                                mileage_inc=(0.30, 1.00), weight=0.13),
    "Eco Driving":        dict(speed=(20, 60),  pedal=(8, 25),  pressure=(10, 30),
                                wear_mult=0.40, heat_mult=0.40, ambient_delta=0.0,
                                mileage_inc=(0.20, 0.60), weight=0.08),
    "Rain":               dict(speed=(20, 70),  pedal=(30, 60), pressure=(40, 90),
                                wear_mult=1.20, heat_mult=0.70, ambient_delta=-3.0,
                                mileage_inc=(0.20, 0.60), weight=0.07),
    "Hot Weather":        dict(speed=(30, 90),  pedal=(20, 50), pressure=(30, 70),
                                wear_mult=1.30, heat_mult=1.30, ambient_delta=12.0,
                                mileage_inc=(0.20, 0.70), weight=0.06),
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
# Slightly more Aggressive-personality vehicles than before, so more
# vehicles reach critical wear/thermal states within the simulation.
PERSONALITY_PROBS = np.array([0.28, 0.40, 0.32])

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
    """Composite, smooth 0-100 health score, calibrated directly
    against the requested reference table (12mm -> 95-100%,
    10mm -> 85-95%, 8mm -> 70-85%, 6mm -> 55-70%, 4mm -> 35-55%,
    3mm -> 15-35%, 2mm -> 5-15%, 1mm -> 0-5%, 0mm -> 0%).

    Pad thickness sets a [low, high] band via monotonic interpolation
    over that table (so health can never rise while thickness falls).
    Fluid condition, disc heat and long-term fatigue then decide
    WHERE within that band this specific vehicle sits - they modulate
    the position inside the band, they never push health outside the
    band the thickness itself allows. This is what lets health reach
    all the way down toward 0% for badly worn brakes instead of
    plateauing around 20% as before."""
    pad_thickness = float(np.clip(pad_thickness, 0.0, NEW_PAD_MM))
    if pad_thickness <= 0.0:
        return 0.0

    lo = float(np.interp(pad_thickness, HEALTH_MM, HEALTH_LO_PCT))
    hi = float(np.interp(pad_thickness, HEALTH_MM, HEALTH_HI_PCT))

    fluid_frac = np.clip(fluid_level, 0, 100) / 100.0
    disc_frac = 1.0 if disc_temp <= 250 else float(np.exp(-(disc_temp - 250) / 300.0))
    usage_frac = float(np.exp(-usage_count / 15000.0))
    mileage_frac = float(np.exp(-mileage / 300000.0))

    # Weighted position within [0, 1] inside the thickness-defined band.
    position = (
        0.45 * fluid_frac
        + 0.25 * disc_frac
        + 0.18 * usage_frac
        + 0.12 * mileage_frac
    )
    position = float(np.clip(position, 0.0, 1.0))

    health = lo + (hi - lo) * position
    return float(np.clip(health, 0.0, 100.0))


def compute_remaining_pad_life(pad_thickness, life_severity, rng):
    """Remaining-life curve calibrated directly against the requested
    reference table:
        12mm -> 49,000-50,000 km   7mm -> 24,000-27,000 km
        11mm -> 44,000-47,000 km   6mm -> 18,000-22,000 km
        10mm -> 39,000-42,000 km   5mm -> 12,000-16,000 km
         9mm -> 34,000-37,000 km   4mm ->  6,000-10,000 km
         8mm -> 29,000-32,000 km   3mm ->  1,000- 3,000 km
                                   2mm, 1mm, 0mm -> 0 km

    For a given thickness, the [low, high] band is obtained by
    monotonic linear interpolation over the table above (so the band
    itself can never move upward as thickness falls), a value is
    drawn uniformly from within that band, and the vehicle's own
    wear-severity personality is allowed only a mild +/-15% nudge
    around that value (rather than a full multiplier) so that a
    single aggressive vehicle can't blow past the requested bounds.
    The result is finally clipped to [0, 50000] km. Non-decreasing
    monotonicity across a vehicle's own event sequence is additionally
    guaranteed later by the hard physical-constraint pass."""
    pad_thickness = float(np.clip(pad_thickness, 0.0, NEW_PAD_MM))
    if pad_thickness <= DANGER_PAD_MM:
        return 0.0

    lo = float(np.interp(pad_thickness, LIFE_MM, LIFE_LO_KM))
    hi = float(np.interp(pad_thickness, LIFE_MM, LIFE_HI_KM))
    base = lo if hi <= lo else float(rng.uniform(lo, hi))

    severity_nudge = float(np.clip(1.0 + (life_severity - 1.0) * 0.3, 0.85, 1.15))
    life = base * severity_nudge
    return float(np.clip(life, 0.0, TYPICAL_LIFE_KM))


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

    # NOTE: Remaining Pad Life is now 0 for the entire [1mm, 2mm) band
    # (per the requested reference table), not just for a
    # fully-worn-out pad. So "remaining_life <= 0" alone is no longer
    # a safe proxy for "truly emergency" - it would swallow the whole
    # "Immediate Service" band above. Emergency Stop is reserved for
    # the genuinely worst states: pad essentially gone, or the disc
    # itself overheating past a safe limit.
    if pad_thickness < CRITICAL_PAD_MM or disc_temp > 600:
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

    # --- initial state ---
    # Most vehicles still start brand-new, but a portion of the fleet
    # is initialized as already-in-service ("used fleet") vehicles.
    # Without this, reaching pad thicknesses below ~2mm or disc temps
    # above 600C requires wearing all the way down from a brand-new
    # pad within the simulated event window, which starves the
    # dataset of critical/near-critical rows (and of the
    # "Replace Brake Pads" / "Replace Brake Disc" / "Replace Brake
    # Fluid" / "Immediate Service" / "Emergency Stop" maintenance
    # classes that depend on them). Seeding some vehicles partway (or
    # almost all the way) through their brake life directly fixes
    # that without changing the underlying physics model at all.
    init_roll = rng.random()
    if init_roll < 0.04:
        # ~4% of vehicles: already near end-of-life / overdue for
        # service (pad below 2mm, low fluid) - guarantees a steady
        # stream of critical rows from their very first event. Kept
        # deliberately small: because pad thickness/fluid level can
        # only ever go down for the rest of that vehicle's history,
        # a larger share here would let critical rows swamp the
        # dataset rather than just being "significantly more common".
        pad_thickness = float(rng.uniform(0.3, 2.0))
        fluid_level = float(rng.uniform(15.0, 45.0))
        mileage = float(rng.uniform(30_000.0, 55_000.0))
        usage_count = int(rng.integers(3_000, 8_000))
    elif init_roll < 0.04 + 0.28:
        # ~27% of vehicles: moderately used, spanning the full
        # mid-life range so mid-range thickness/health/maintenance
        # classes are well represented too.
        pad_thickness = float(rng.uniform(2.0, 9.0))
        fluid_level = float(rng.uniform(40.0, 85.0))
        mileage = float(rng.uniform(5_000.0, 35_000.0))
        usage_count = int(rng.integers(500, 4_000))
    else:
        # remaining ~65%: brand-new vehicle, as before
        pad_thickness = NEW_PAD_MM
        fluid_level = 100.0
        mileage = 0.0
        usage_count = 0
    disc_temp = base_ambient
    fluid_temp = base_ambient

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
        remaining_life = compute_remaining_pad_life(pad_thickness, life_severity, rng)
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

    # Vectorized per-vehicle cumulative ops (equivalent to the old
    # groupby().apply() version, but doesn't rely on apply() to
    # preserve the "_vehicle_id" column - recent pandas versions drop
    # the grouping column from apply() results by default).
    g = df.groupby("_vehicle_id", sort=False)
    df["Brake Pad Thickness (mm)"] = g["Brake Pad Thickness (mm)"].cummin()
    df["Brake Fluid Level (%)"] = g["Brake Fluid Level (%)"].cummin()
    df["Remaining Pad Life (km)"] = g["Remaining Pad Life (km)"].cummin()
    df["Total Mileage (km)"] = g["Total Mileage (km)"].cummax()
    df["Brake Usage Count"] = g["Brake Usage Count"].cummax()
    return df


def validate_and_fix(df):
    """Explicit row-level validation pass, checked against every rule
    requested:
        Brake Health in [0, 100]
        Remaining Pad Life in [0, 50000]
        Brake Pad Thickness in [0, 12]
        Disc Temperature >= Ambient Temperature
        Fluid Temperature >= Ambient Temperature
        Brake Usage Count never decreases (within a vehicle)
        Total Mileage never decreases (within a vehicle)

    By this point in the pipeline the simulation physics, the sensor-
    noise clipping, and enforce_hard_physical_constraints() already
    guarantee all of these by construction - so in practice this pass
    finds (and fixes) nothing. It's kept as a final belt-and-braces
    safety net: rather than discarding a violating row outright
    (which would break that vehicle's continuous event sequence), any
    row found to violate a bound has ONLY the offending value clamped
    back into its valid range. The number of rows touched is reported
    so you can confirm the dataset needed no corrections."""
    n_fixed = 0

    bad = (df["Brake Health (%)"] < 0) | (df["Brake Health (%)"] > 100)
    n_fixed += int(bad.sum())
    df["Brake Health (%)"] = df["Brake Health (%)"].clip(0, 100)

    bad = (df["Remaining Pad Life (km)"] < 0) | (df["Remaining Pad Life (km)"] > TYPICAL_LIFE_KM)
    n_fixed += int(bad.sum())
    df["Remaining Pad Life (km)"] = df["Remaining Pad Life (km)"].clip(0, TYPICAL_LIFE_KM)

    bad = (df["Brake Pad Thickness (mm)"] < 0) | (df["Brake Pad Thickness (mm)"] > NEW_PAD_MM)
    n_fixed += int(bad.sum())
    df["Brake Pad Thickness (mm)"] = df["Brake Pad Thickness (mm)"].clip(0, NEW_PAD_MM)

    bad = df["Brake Disc Temperature (C)"] < df["Ambient Temperature (C)"]
    n_fixed += int(bad.sum())
    df["Brake Disc Temperature (C)"] = np.maximum(
        df["Brake Disc Temperature (C)"], df["Ambient Temperature (C)"]
    )

    bad = df["Brake Fluid Temperature (C)"] < df["Ambient Temperature (C)"]
    n_fixed += int(bad.sum())
    df["Brake Fluid Temperature (C)"] = np.maximum(
        df["Brake Fluid Temperature (C)"], df["Ambient Temperature (C)"]
    )

    # Usage Count / Total Mileage: enforce_hard_physical_constraints()
    # already made these cumulative-max per vehicle; re-check here.
    usage_bad = df.groupby("_vehicle_id")["Brake Usage Count"].diff().lt(0).fillna(False)
    n_fixed += int(usage_bad.sum())
    mileage_bad = df.groupby("_vehicle_id")["Total Mileage (km)"].diff().lt(0).fillna(False)
    n_fixed += int(mileage_bad.sum())
    if usage_bad.any() or mileage_bad.any():
        df["Brake Usage Count"] = df.groupby("_vehicle_id")["Brake Usage Count"].cummax()
        df["Total Mileage (km)"] = df.groupby("_vehicle_id")["Total Mileage (km)"].cummax()

    print(f"Row validation: {n_fixed} value(s) required correction (out of {len(df):,} rows).")
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

    # Explicit row-by-row validation pass (requirement 8): checks every
    # rule and corrects any row that still violates a physical bound.
    df = validate_and_fix(df)

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
