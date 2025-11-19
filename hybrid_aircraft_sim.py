import numpy as np
import logging
from copy import deepcopy
import math
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ================================================================
# PIPER PA-28R-200 ARROW II â€” AERODYNAMIC & PROPULSION REFERENCE
# Aircraft and aero reference data
arrow2_data = {
    "aircraft": {
        "model": "Piper PA-28R-200 Arrow II",
        "wing_area_m2": 15.8,          # From POH
        "aspect_ratio": 7.2,
        "empty_mass_kg": 550.0,
        "mtow_kg": 1200.0,
        "airfoil": "NACA 64-415",      # Confirmed for Arrow II
    },

    # Propeller data
    "propeller": {
        "manufacturer": "Hartzell",
        "model": "HC-C2YK-1 / FC7666A-2",
        "type": "2-blade constant-speed, metal",
        "diameter_m": 1.98,
        "efficiency_curve": [
            # Approximate Î·_prop vs. advance ratio (J = V / (n*D))
            # Based on generic Hartzell 2-blade cruise prop curves.
            {"J": 0.0,  "eta": 0.00},
            {"J": 0.2,  "eta": 0.55},
            {"J": 0.4,  "eta": 0.73},
            {"J": 0.6,  "eta": 0.80},
            {"J": 0.8,  "eta": 0.83},
            {"J": 1.0,  "eta": 0.78},
            {"J": 1.2,  "eta": 0.60}
        ],
        "j_points": [p["J"] for p in [
            {"J": 0.0, "eta": 0.00}, {"J": 0.2, "eta": 0.55}, {"J": 0.4, "eta": 0.73},
            {"J": 0.6, "eta": 0.80}, {"J": 0.8, "eta": 0.83}, {"J": 1.0, "eta": 0.78},
            {"J": 1.2, "eta": 0.60}
        ]],
        "eta_points": [p["eta"] for p in [
            {"J": 0.0, "eta": 0.00}, {"J": 0.2, "eta": 0.55}, {"J": 0.4, "eta": 0.73},
            {"J": 0.6, "eta": 0.80}, {"J": 0.8, "eta": 0.83}, {"J": 1.0, "eta": 0.78},
            {"J": 1.2, "eta": 0.60}
        ]]
    },

    # Flap aerodynamics
    "flap_data": {
        0:  {"CL_max": 1.21, "CD0": 0.027, "k": 0.059}, 
        10: {"CL_max": 1.28, "CD0": 0.037, "k": 0.059},
        25: {"CL_max": 1.38, "CD0": 0.052, "k": 0.059},
        40: {"CL_max": 1.49, "CD0": 0.077, "k": 0.059},
    },
    
    "other_drag": {
        "gear_down_cd0_penalty": 0.015
    }
}


# Baseline aircraft constants
P_BASELINE_HP = 200
MTOW_KG = arrow2_data["aircraft"]["mtow_kg"]
BASE_AIRFRAME_KG = arrow2_data["aircraft"]["empty_mass_kg"]
S_W_M2 = arrow2_data["aircraft"]["wing_area_m2"]

# Propulsion efficiencies
ETA_MOTOR = 0.92
ETA_GEN = 0.90
ETA_BATT = 0.95

# Fuel and emission constants
JET_A_DENSITY = 0.804
AVGAS_DENSITY = 0.721
CO2_AVGAS_KG = 3.10
CO2_JETA_KG = 3.16
NOX_AVGAS_KG = 0.015
NOX_JETA_KG = 0.020
PM_AVGAS_KG = 0.0008
PM_JETA_KG = 0.0015

# Fuel property lookup
FUEL_PROPERTIES = {
    "Jet-A": {
        "density": JET_A_DENSITY,
        "specific_energy_mj_per_kg": 43.15,
    },
    "Avgas": {
        "density": AVGAS_DENSITY,
        "specific_energy_mj_per_kg": 44.40,
    },
}

# Scenario defaults and weather state
BASELINE_CO2_PER_KM = None
TECH_SCENARIO = "realistic"
FUEL_SCENARIO = "fossil"
weather_config = {
    "mode": "none",
    "lat": 52.24,
    "lon": 6.05,
    "datetime_utc": None,
    "altitudes_ft": [0, 3000, 6000, 9000, 12000],
    "wind_profile": [],
}
# Regulatory reserve definition
RESERVE_POLICY = {
    "name": "EASA_VFR_DAY_30MIN",
    "extra_time_min": 30.0,
    "reserve_alt_ft": 3000,
    "reserve_power_frac": 0.65,
}

# ICE SFC reference curve
ICE_SFC_DATA = {
    1.00: 0.245,
    0.75: 0.191,
    0.65: 0.191,
    0.55: 0.195
}

# ECMS parameters
ECMS_EQUIVALENCE_FACTOR = 0.04
ECMS_SOC_MAX = 0.995
ECMS_SOC_LOW_PENALTY = 0.05
SPECIFIC_ENERGY_WH_PER_KG = 300.0
SPECIFIC_ENERGY_WH_PER_KG_FUTURE = 600.0

BSE_PROXIES = {
    "GT": {"NOx": 12.0, "PM": 0.006},
    "ICE": {"NOx": 8.0, "PM": 0.030},
}


# Concept configuration templates
BASE_CONCEPTS = {
    "Baseline (Avgas)": {
        "type": "ICE",
        "p_ice_hp": 200, "wt_ice": 180, "sfc_design": 0.20,
        "p_em_hp": 0,    "wt_em": 0,    "batt_kwh": 0, "wt_batt": 15,
        "p_gt_hp": 0,    "wt_gt": 0,    "gt_sfc_design": 0,
        "fuel_type": "Avgas", "fuel_vol_L": 182,
        "base_mass_adder": 0,
        "cd0_adder": 0.000,
        "cruise_mode": "economy",
    },
    "4.1.2 Parallel Hybrid": {
        "type": "Parallel",
        "p_gt_hp": 240,  "wt_gt": 98,   "gt_sfc_design": 0.35,
        "p_gen_kw": 50,  "wt_gen": 15,
        "p_em_hp": 50,   "wt_em": 15,   "batt_kwh": 30, "wt_batt": 0,
        "fuel_type": "Jet-A", "fuel_vol_L": 182,
        "base_mass_adder": 30,
        "cd0_adder": 0.002,
        "batt_c_max": 3.0,
        "batt_c_chg_max": 1.0,
        "cruise_mode": "economy",
    },  
    "4.1.3 Series Hybrid": {
        "type": "Series",
        "p_gt_hp": 290,  "wt_gt": 110,  "gt_sfc_design": 0.32,
        "p_gen_kw": 180, "wt_gen": 30,
        "p_em_hp": 200,  "wt_em": 50,   "batt_kwh": 40  , "wt_batt": 0,
        "fuel_type": "Jet-A", "fuel_vol_L": 182,
        "base_mass_adder": 20,
        "cd0_adder": 0.003,
        "batt_c_max": 3.0,
        "batt_c_chg_max": 1.0,
        "cruise_mode": "economy",
    },
    "4.1.4 Parallel-Series": {
        "type": "Parallel-Series",
        "p_gt_hp": 200,  "wt_gt": 90,   "gt_sfc_design": 0.42,
        "p_gen_kw": 150, "wt_gen": 25,
        "p_em_hp": 200,  "wt_em": 50,   "batt_kwh": 15, "wt_batt": 0,
        "fuel_type": "Jet-A", "fuel_vol_L": 182,
        "base_mass_adder": 80,
        "cd0_adder": 0.004,
        "batt_c_max": 2.5,
        "batt_c_chg_max": 1.0,
        "cruise_mode": "economy",
    },
    "4.1.5 Turboprop": {
        "type": "Turboprop",
        "p_gt_hp": 250,  "wt_gt": 100,  "gt_sfc_design": 0.34,
        "p_em_hp": 0,    "wt_em": 0,    "batt_kwh": 0, "wt_batt": 15, # 15kg for starter batt
        "fuel_type": "Jet-A", "fuel_vol_L": 182,
        "base_mass_adder": -20,
        "cd0_adder": 0.0015, # Sleeker, but still different, nacelle
        "cruise_mode": "economy",
    }
}

def configure_efficiencies(scenario):
    """Set global efficiency constants based on tech scenario."""
    global ETA_MOTOR, ETA_GEN, ETA_BATT
    if scenario == "optimistic_future":
        ETA_MOTOR = 0.97   # better motors/inverters
        ETA_GEN = 0.96     # high-efficiency turbogenerator
        ETA_BATT = 0.98    # improved pack round-trip
    else:
        ETA_MOTOR = 0.92
        ETA_GEN = 0.90
        ETA_BATT = 0.95


def build_concepts_for_scenario(scenario):
    """Return a deep-copied concept dictionary with scenario-specific overrides."""
    scenario_concepts = deepcopy(BASE_CONCEPTS)
    specific_energy = (
        SPECIFIC_ENERGY_WH_PER_KG_FUTURE
        if scenario == "optimistic_future"
        else SPECIFIC_ENERGY_WH_PER_KG
    )

    for concept in scenario_concepts.values():
        if concept["batt_kwh"] > 0 and concept["type"] not in ["Baseline (Avgas)", "Turboprop"]:
            concept["wt_batt"] = concept["batt_kwh"] * 1000 / specific_energy

    if scenario == "optimistic_future":
        if "4.1.2 Parallel Hybrid" in scenario_concepts:
            concept = scenario_concepts["4.1.2 Parallel Hybrid"]
            concept["gt_sfc_design"] = 0.22
            concept["wt_gt"] = 90
            concept["cd0_adder"] = 0.0015
        if "4.1.3 Series Hybrid" in scenario_concepts:
            concept = scenario_concepts["4.1.3 Series Hybrid"]
            concept["gt_sfc_design"] = 0.20
            concept["wt_gt"] = 100
            concept["cd0_adder"] = 0.0025
        if "4.1.4 Parallel-Series" in scenario_concepts:
            concept = scenario_concepts["4.1.4 Parallel-Series"]
            concept["gt_sfc_design"] = 0.22
            concept["wt_gt"] = 85
            concept["cd0_adder"] = 0.003
        if "4.1.5 Turboprop" in scenario_concepts:
            concept = scenario_concepts["4.1.5 Turboprop"]
            concept["gt_sfc_design"] = 0.24
            concept["wt_gt"] = 90

    return scenario_concepts

# =========================================
# 3. PHYSICS HELPERS
# =========================================

def get_air_density(alt_ft):
    """Calculates air density at a given altitude using a simple ISA model."""
    alt_m = alt_ft * 0.3048
    if alt_m < 11000:
        temp = 288.15 - 0.0065 * alt_m
        return (101325 * (1 - 0.0065 * alt_m / 288.15)**5.2561) / (287.05 * temp)
    return 0.4 # Simplified high-altitude density

RHO0 = get_air_density(0)


def kias_to_tas(kias, alt_ft):
    """Convert IAS to TAS using density ratio."""
    rho = get_air_density(alt_ft)
    if rho <= 0:
        return kias * 2.0
    return kias * math.sqrt(RHO0 / rho)


def ktas_to_kias(ktas, alt_ft):
    """Convert TAS to IAS using density ratio."""
    rho = get_air_density(alt_ft)
    if rho <= 0:
        return ktas * 0.5
    return ktas * math.sqrt(rho / RHO0)


def get_climb_speed_kias(mass_kg, alt_ft):
    """Approximate climb IAS schedule that varies with mass and altitude."""
    base_vy = 90.0
    alt_correction = -0.0005 * alt_ft  # -0.5 kt per 1000 ft
    weight_factor = mass_kg / MTOW_KG
    weight_correction = 5.0 * (weight_factor - 1.0)
    v_kias = base_vy + alt_correction + weight_correction
    return max(75.0, min(100.0, v_kias))


def get_descent_speed_kias(mass_kg, alt_ft):
    """Approximate descent IAS schedule."""
    base = 100.0
    weight_factor = mass_kg / MTOW_KG
    weight_correction = -3.0 * (1.0 - weight_factor)
    return max(80.0, min(120.0, base + weight_correction))


def pick_optimal_cruise_speed(c, current_mass, concept_cd0_base, alt_ft, cruise_mode):
    """Scan 110-150 KTAS and pick the speed with the lowest equivalent fuel per distance."""
    concept_type = c.get("type", "ICE")
    rho = get_air_density(alt_ft)
    system_limit_kw = P_BASELINE_HP * 0.7457
    gt_max_kw = (c.get("p_gt_hp", 0.0) + c.get("p_ice_hp", 0.0)) * 0.7457
    em_max_kw = c.get("p_em_hp", 0.0) * 0.7457
    gen_max_kw = c.get("p_gen_kw", gt_max_kw if gt_max_kw > 0 else 0.0)

    eng_type = "ICE" if concept_type == "ICE" else "GT"
    base_sfc = c.get("sfc_design", 0.2) if eng_type == "ICE" else c.get("gt_sfc_design", 0.45)
    speeds = range(110, 151, 2)
    best_speed = 130.0
    best_cost = float("inf")
    for v_kts in speeds:
        vel_ms = v_kts * 0.51444
        eta_prop = get_prop_efficiency(vel_ms, "Cruise")
        cl = (current_mass * 9.81) / (0.5 * rho * vel_ms**2 * S_W_M2)
        current_cd0, current_k = get_aero_coeffs("Cruise", concept_cd0_base)
        cd = current_cd0 + current_k * cl**2
        drag_n = 0.5 * rho * vel_ms**2 * S_W_M2 * cd
        p_prop_kw = (drag_n * vel_ms) / 1000.0
        p_shaft_kw = p_prop_kw / max(0.1, eta_prop)

        # Respect overall system cap
        available_kw = min(system_limit_kw, gt_max_kw + em_max_kw)
        if p_shaft_kw > available_kw + 1e-6:
            continue

        p_gt_kw = min(p_shaft_kw, gt_max_kw)
        p_em_kw = max(0.0, p_shaft_kw - p_gt_kw)
        if p_em_kw > em_max_kw + 1e-6:
            continue

        # Fuel consumption from GT/ICE portion
        fuel_rate = 0.0
        if p_gt_kw > 0 and gt_max_kw > 0:
            load_frac = max(0.0, min(1.0, p_gt_kw / gt_max_kw))
            sfc_kg_hp_hr = get_sfc(load_frac, eng_type, base_sfc)
            fuel_rate = (sfc_kg_hp_hr * (p_gt_kw / 0.7457)) / 3600.0  # kg/s

        batt_rate_kw = p_em_kw / max(ETA_MOTOR, 0.1)
        batt_rate_kwh_s = batt_rate_kw / 3600.0
        eq_rate = fuel_rate + ECMS_EQUIVALENCE_FACTOR * batt_rate_kwh_s
        gs_ms = max(1.0, vel_ms)  # TAS ≈ GS for optimizer
        cost = eq_rate / gs_ms

        if cost < best_cost - 1e-9:
            best_cost = cost
            best_speed = float(v_kts)

    if cruise_mode == "fast":
        best_speed = min(150.0, best_speed + 8.0)
    elif cruise_mode == "economy":
        best_speed = max(110.0, best_speed)
    return best_speed


def apply_weather_to_phase(phase, weather_cfg, default_track_deg):
    phase = phase.copy()
    phase.setdefault("track_deg", default_track_deg)
    wind_profile = weather_cfg.get("wind_profile", [])
    wind_kt, wind_dir = get_wind_at_alt(phase.get("alt", 0.0), wind_profile)
    phase["wind_kt"] = wind_kt
    phase["wind_dir_deg"] = wind_dir
    return phase


def _mock_wind_profile(config):
    profile = []
    base_speed = 40.0
    base_dir = 220.0
    for alt in config.get("altitudes_ft", [0, 3000, 6000, 9000, 12000]):
        shear_gain = (alt / 1000.0) * 4.0
        gust = 10.0 * math.sin(math.radians((alt / 500.0) * 20.0))
        speed = max(5.0, base_speed + shear_gain + gust)
        direction_shift = (alt / 1000.0) * 6.0 + 8.0 * math.sin(math.radians((alt / 1000.0) * 15.0))
        direction = (base_dir + direction_shift) % 360
        profile.append({"alt_ft": alt, "wind_kt": speed, "wind_dir_deg": direction})
    return profile


def fetch_wind_profile(config):
    """Fill wind_profile using Open-Meteo data, with graceful fallback."""
    mode = config.get("mode", "mock")
    altitudes = config.get("altitudes_ft", [0, 3000, 6000, 9000, 12000])

    if mode == "none":
        config["wind_profile"] = [{"alt_ft": alt, "wind_kt": 0.0, "wind_dir_deg": 0.0} for alt in altitudes]
        return config["wind_profile"]

    if mode == "mock":
        config["wind_profile"] = _mock_wind_profile(config)
        return config["wind_profile"]

    lat = float(config.get("lat", 52.24))
    lon = float(config.get("lon", 6.05))
    dt = config.get("datetime_utc")
    if isinstance(dt, str):
        try:
            dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None:
        dt = datetime.datetime.utcnow()
    dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_speed_80m",
                "wind_direction_80m",
                "wind_speed_120m",
                "wind_direction_120m",
            ]
        ),
        "start_date": dt.date().isoformat(),
        "end_date": dt.date().isoformat(),
        "timezone": "UTC",
    }

    try:
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise ValueError("weather data missing time axis")
        target_ts = dt.strftime("%Y-%m-%dT%H:00")
        if target_ts in times:
            idx = times.index(target_ts)
        else:
            idx = min(range(len(times)), key=lambda i: abs(datetime.datetime.fromisoformat(times[i]) - dt))

        level_data = []
        for height in (10, 80, 120):
            spd_key = f"wind_speed_{height}m"
            dir_key = f"wind_direction_{height}m"
            spd_list = hourly.get(spd_key)
            dir_list = hourly.get(dir_key)
            if spd_list and dir_list:
                speed = float(spd_list[idx])
                direction = float(dir_list[idx])
                level_data.append({"alt_m": height, "wind_kt": speed * 0.539957, "wind_dir_deg": direction})

        if not level_data:
            raise ValueError("weather response lacked usable levels")

        level_data.sort(key=lambda x: x["alt_m"])

        def interp_speed_dir(target_ft):
            target_m = target_ft * 0.3048
            lower = level_data[0]
            for entry in level_data:
                if target_m < entry["alt_m"]:
                    upper = entry
                    break
                lower = entry
            else:
                upper = None

            if upper is None:
                grad = 0.0006  # kt per meter
                delta = target_m - level_data[-1]["alt_m"]
                speed = level_data[-1]["wind_kt"] + grad * delta
                direction = level_data[-1]["wind_dir_deg"]
                return speed, direction

            span = upper["alt_m"] - lower["alt_m"]
            if span <= 0:
                return lower["wind_kt"], lower["wind_dir_deg"]
            frac = (target_m - lower["alt_m"]) / span
            speed = lower["wind_kt"] + frac * (upper["wind_kt"] - lower["wind_kt"])
            dir_diff = ((upper["wind_dir_deg"] - lower["wind_dir_deg"] + 180) % 360) - 180
            direction = (lower["wind_dir_deg"] + frac * dir_diff) % 360
            return speed, direction

        profile = []
        for alt in altitudes:
            speed, direction = interp_speed_dir(alt)
            profile.append({"alt_ft": alt, "wind_kt": max(0.0, speed), "wind_dir_deg": direction})
        config["wind_profile"] = profile
    except Exception as exc:
        logging.warning(f"Weather fetch failed ({exc}); using mock wind profile.")
        config["wind_profile"] = _mock_wind_profile(config)

    return config["wind_profile"]


def get_wind_at_alt(alt_ft, wind_profile):
    """Linear interpolation of wind profile."""
    if not wind_profile:
        return 0.0, 0.0
    lower = wind_profile[0]
    for entry in wind_profile:
        if alt_ft < entry["alt_ft"]:
            upper = entry
            break
        lower = entry
    else:
        return lower["wind_kt"], lower["wind_dir_deg"]
    span = upper["alt_ft"] - lower["alt_ft"]
    if span <= 0:
        return lower["wind_kt"], lower["wind_dir_deg"]
    frac = (alt_ft - lower["alt_ft"]) / span
    wind_speed = lower["wind_kt"] + frac * (upper["wind_kt"] - lower["wind_kt"])
    wind_dir = (lower["wind_dir_deg"] + frac * ((upper["wind_dir_deg"] - lower["wind_dir_deg"]) % 360)) % 360
    return wind_speed, wind_dir


def ground_speed_from_tas_and_wind(tas_kt, track_deg, wind_speed_kt, wind_from_deg):
    """Compute ground speed vector from TAS and wind."""
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    tr = math.radians(track_deg)
    wr = math.radians(wind_to_deg)
    vx_air = tas_kt * math.cos(tr)
    vy_air = tas_kt * math.sin(tr)
    vx_wind = wind_speed_kt * math.cos(wr)
    vy_wind = wind_speed_kt * math.sin(wr)
    vx_gnd = vx_air + vx_wind
    vy_gnd = vy_air + vy_wind
    gs = math.sqrt(vx_gnd ** 2 + vy_gnd ** 2)
    return max(1.0, gs)


def compute_vfr_day_reserve_fuel(c, mass_for_reserve, batt_kwh_for_reserve,
                                 batt_kwh_max, concept_cd0_base, total_dist_km):
    """Compute fuel required for EASA VFR day reserve (30 minutes cruise)."""
    dur_sec = int(RESERVE_POLICY["extra_time_min"] * 60.0)
    reserve_alt = RESERVE_POLICY["reserve_alt_ft"]
    reserve_power = RESERVE_POLICY["reserve_power_frac"]
    reserve_v_kts = 110  # approximate endurance-oriented IAS/TAS for reserve cruise
    reserve_phase = {
        "name": "VFR_Reserve_Cruise",
        "dur": dur_sec,
        "v_kts": reserve_v_kts,
        "p_req": reserve_power,
        "alt": reserve_alt,
        "force_power_frac": reserve_power,
    }
    reserve_cfg = deepcopy(c)
    original_em_hp = reserve_cfg.get("p_em_hp", 0.0)
    if reserve_cfg.get("type") != "Series":
        reserve_cfg["p_em_hp"] = 0.0
    fuel_kg, batt_delta_kwh, _, _, _, _, _, _, _ = calculate_phase_burns(
        reserve_phase,
        reserve_cfg,
        mass_for_reserve,
        batt_kwh_for_reserve,
        batt_kwh_max,
        concept_cd0_base,
        total_dist_km,
        is_reserve_calc=True,
    )
    reserve_cfg["p_em_hp"] = original_em_hp

    return max(0.0, fuel_kg)


def compute_directional_range(
    c,
    base_mass,
    base_batt_kwh,
    batt_kwh_max,
    concept_cd0_base,
    track_deg,
    weather_cfg,
    available_fuel_kg,
    max_minutes=600,
):
    """Simulate a cruise leg along a specific track to determine maximum distance under current weather."""
    cruise_cfg = deepcopy(c)
    cruise_cfg["mission_track_deg"] = track_deg
    current_mass = base_mass
    current_batt_kwh = base_batt_kwh
    total_dist_km = 0.0
    fuel_used = 0.0
    cruise_speed = pick_optimal_cruise_speed(cruise_cfg, current_mass, concept_cd0_base, 10000, cruise_cfg.get("cruise_mode", "economy"))
    cruise_ph = {
        "name": "Cruise",
        "dur": 60,
        "v_kts": cruise_speed,
        "alt": 10000,
    }
    minutes = 0
    while minutes < max_minutes:
        cruise_ph = apply_weather_to_phase(
            {
                "name": "Cruise",
                "dur": 60,
                "v_kts": cruise_ph["v_kts"],
                "alt": cruise_ph["alt"],
                "track_deg": track_deg,
            },
            weather_cfg,
            track_deg,
        )
        f, b, _, d_dist, _, _, _, _, _ = calculate_phase_burns(
            cruise_ph,
            cruise_cfg,
            current_mass,
            current_batt_kwh,
            batt_kwh_max,
            concept_cd0_base,
            total_dist_km,
        )
        if (f <= 0.0 and d_dist <= 0.0) or available_fuel_kg <= 0.0:
            break
        fuel_used += f
        current_mass -= f
        current_batt_kwh -= b
        current_batt_kwh = max(0.0, min(current_batt_kwh, batt_kwh_max))
        total_dist_km += d_dist
        minutes += 1
        if fuel_used >= available_fuel_kg:
            break
    return total_dist_km


def sample_directional_ranges(
    c,
    base_mass,
    base_batt_kwh,
    batt_kwh_max,
    concept_cd0_base,
    weather_cfg,
    available_fuel_kg,
    step_deg=10,
):
    cfg = deepcopy(weather_cfg)
    if not cfg.get("wind_profile"):
        fetch_wind_profile(cfg)
    samples = []
    for track in range(0, 360, step_deg):
        dist = compute_directional_range(
            c,
            base_mass,
            base_batt_kwh,
            batt_kwh_max,
            concept_cd0_base,
            track,
            cfg,
            available_fuel_kg,
        )
        samples.append({"track_deg": track, "range_km": dist})
    return samples

def get_sfc(pct, eng_type, base):
    """3. Improved off-design SFC model for turbine and ICE."""
    if eng_type == "GT":
        pct = max(0.01, min(1.0, pct)) # Clamp load factor
        if pct < 0.3:
            return base * (2.5 - 3.0 * pct)   # very poor at low power
        elif pct < 0.7:
            return base * (1.1 - 0.3 * pct)   # efficiency improves
        else:
            return base * (1.0 + 0.15 * (pct - 0.7))  # slightly worse at max
            
    if eng_type == "ICE" and ICE_SFC_DATA:
        # Find the closest power percentage in the data and use its SFC
        power_pct_clamped = max(min(pct, max(ICE_SFC_DATA.keys())), min(ICE_SFC_DATA.keys()))
        closest_pct = min(ICE_SFC_DATA.keys(), key=lambda x: abs(x - power_pct_clamped))
        return ICE_SFC_DATA[closest_pct]
        
    return base * (1.0 + 0.5 * (1 - pct)**2) # Fallback

def get_bse_g_kw_hr(load_frac, eng_type, pollutant):
    """
    Returns a proxy brake-specific emission (g/kWh) for NOx or PM based on engine type and load.
    NOx increases with load/temperature, PM gently decreases at higher loads due to hotter combustion.
    """
    eng_key = "GT" if eng_type.upper() != "ICE" else "ICE"
    props = BSE_PROXIES.get(eng_key, BSE_PROXIES["GT"])
    base = props.get(pollutant, 0.0)
    lf = max(0.0, min(1.0, load_frac))
    if pollutant == "NOx":
        scale = 0.6 + 0.4 * lf  # 60% of base at idle, 100% near max power
    else:  # PM
        scale = 1.1 - 0.4 * lf  # Slightly higher at low load, lower at high load
        scale = max(0.3, scale)
    return base * scale


def get_jet_a_emission_factors(use_saf: bool):
    """
    Returns (co2_factor, nox_factor, pm_factor) in kg per kg fuel
    for Jet-A or Jet-A + SAF50.
    """
    base_co2 = CO2_JETA_KG
    base_nox = NOX_JETA_KG
    base_pm = PM_JETA_KG

    if TECH_SCENARIO == "optimistic_future":
        base_nox *= 0.85
        base_pm *= 0.8

    if not use_saf:
        return base_co2, base_nox, base_pm

    co2_factor = base_co2 * 0.5
    nox_factor = base_nox * 0.6
    pm_factor = base_pm * 0.4
    return co2_factor, nox_factor, pm_factor

def get_prop_efficiency(vel_ms, phase_name):
    """
    Calculates propeller efficiency based on Advance Ratio J = V / (n*D).
    Assumes a representative RPM 'n' for each flight phase.
    """
    D = arrow2_data["propeller"]["diameter_m"]
    
    if phase_name in ["Takeoff", "Climb", "Go-Around"]:
        n = 45.0 # 2700 RPM (max power)
    elif phase_name == "Cruise":
        n = 40.0 # 2400 RPM (cruise setting)
    elif phase_name in ["Descent", "Pattern"]:
        n = 35.0 # 2100 RPM (lower power)
    else: # Taxi
        n = 16.7 # 1000 RPM (idle)

    if vel_ms < 1.0 or n < 1:
        return 0.1 # Static thrust, avoid division by zero
        
    J = vel_ms / (n * D)
    
    # Interpolate from the efficiency curve
    eta = np.interp(J, arrow2_data["propeller"]["j_points"], arrow2_data["propeller"]["eta_points"])
    return max(0.1, eta) # Ensure a minimum efficiency

def get_aero_coeffs(phase_name, concept_cd0_base):
    """
    Gets the correct CD0 and K for a given flight phase and concept.
    Includes penalties for flaps and landing gear.
    """
    k = arrow2_data["flap_data"][0]["k"] # K is assumed constant
    delta_cd0_flaps = 0
    delta_cd0_gear = 0

    base_clean_cd0 = arrow2_data["flap_data"][0]["CD0"]
    gear_penalty = arrow2_data["other_drag"]["gear_down_cd0_penalty"]

    if phase_name in ["Takeoff", "Go-Around"]:
        # 25 deg flaps, gear down
        delta_cd0_flaps = arrow2_data["flap_data"][25]["CD0"] - base_clean_cd0
        delta_cd0_gear = gear_penalty
    elif phase_name == "Climb":
        # 10 deg flaps, gear up
        delta_cd0_flaps = arrow2_data["flap_data"][10]["CD0"] - base_clean_cd0
        # Gear is up
    elif phase_name == "Pattern":
        # 40 deg flaps, gear down
        delta_cd0_flaps = arrow2_data["flap_data"][40]["CD0"] - base_clean_cd0
        delta_cd0_gear = gear_penalty
    elif phase_name in ["Taxi-Out", "Taxi-In"]:
        delta_cd0_gear = gear_penalty
    elif phase_name == "Descent":
        # Clean configuration, gear up
        pass
    
    # Total CD0 = Concept's fuselage drag + flap drag + gear drag
    current_cd0 = concept_cd0_base + delta_cd0_flaps + delta_cd0_gear
    return current_cd0, k


# =========================================
# 4. CORE SIMULATION
# =========================================

def calculate_phase_burns(phase, c, current_mass, current_batt_kwh, batt_kwh_max, concept_cd0_base, total_dist_km, is_reserve_calc=False, phase_log_list=None):
    """
    Calculates the fuel and battery consumption for a single mission phase.
    Returns:
        fuel_kg,
        batt_delta_kwh,
        duration_s,
        distance_km,
        ecms_failsafe_triggered,
        fuel_mj,
        fuel_liters,
        nox_g,
        pm_g
    """
    dur = phase["dur"]
    if dur <= 0:
        logging.warning(f"Phase duration <= 0 ({dur:.2f}s) for {phase['name']}. Using 60s fallback.")
        dur = 60.0

    tas_kt = phase.get("v_kts", 0.0)
    vel_ms = tas_kt * 0.51444
    alt_ft = phase.get("alt", 0)
    rho = get_air_density(alt_ft)
    track_deg = phase.get("track_deg", 0.0)
    wind_kt = phase.get("wind_kt", 0.0)
    wind_dir_deg = phase.get("wind_dir_deg", 0.0)
    gs_kt = ground_speed_from_tas_and_wind(tas_kt, track_deg, wind_kt, wind_dir_deg)
    gs_ms = gs_kt * 0.51444
    distance_km = (gs_ms * dur) / 1000.0
    
    # Aerodynamic coefficients for this phase
    eta_prop = get_prop_efficiency(vel_ms, phase["name"])
    current_cd0, current_k = get_aero_coeffs(phase["name"], concept_cd0_base)

    force_power_frac = phase.get("force_power_frac")

    # Determine shaft power demand for the phase
    if force_power_frac is not None:
        avail_hp = c.get("p_gt_hp", 0) + c.get("p_ice_hp", 0)
        if avail_hp <= 0:
            avail_hp = P_BASELINE_HP
        p_req_shaft_kw = avail_hp * 0.7457 * max(0.0, min(1.0, force_power_frac))
        p_req_prop_kw = p_req_shaft_kw * eta_prop
    elif phase["v_kts"] > 0 and "alt" in phase and phase["name"] not in ["Takeoff", "Climb", "Go-Around"]:
        cl = (current_mass * 9.81) / (0.5 * rho * vel_ms**2 * S_W_M2)
        cd = current_cd0 + current_k * cl**2
        drag_n = 0.5 * rho * vel_ms**2 * S_W_M2 * cd
        p_drag_prop_kw = (drag_n * vel_ms) / 1000.0
        if phase["name"] == "Descent":
            vsink_ms = phase.get("vsink_ms", 3.0)
            p_req_prop_kw = max(0.0, p_drag_prop_kw - (current_mass * 9.81 * vsink_ms) / 1000.0)
        else:
            p_req_prop_kw = p_drag_prop_kw
    else:
        # Percentage-based power for ground and high-power phases
        total_avail_hp = P_BASELINE_HP
        if total_avail_hp == 0: total_avail_hp = P_BASELINE_HP
        
        total_avail_kw = total_avail_hp * 0.7457
        
        weight_factor = (current_mass / MTOW_KG)**1.5
        p_req_shaft_kw = total_avail_kw * phase["p_req"] * max(0.8, weight_factor)
        p_req_prop_kw = p_req_shaft_kw * eta_prop


    # Climb power requirement with ECMS integration
    if phase["name"] == "Climb":
        v_climb = max(60.0, phase.get("v_kts", 100.0)) * 0.51444
        climb_alt = max(0.0, phase.get("alt", 6000.0))
        rho_c = get_air_density(climb_alt)
        eta_prop_climb = get_prop_efficiency(v_climb, "Climb")
        climb_cd0, climb_k = get_aero_coeffs("Climb", concept_cd0_base)
        cl_climb = (current_mass * 9.81) / max(1e-6, (0.5 * rho_c * v_climb**2 * S_W_M2))
        drag_n = 0.5 * rho_c * v_climb**2 * S_W_M2 * (climb_cd0 + climb_k * (cl_climb**2))

        gt_component_kw = (c.get("p_gt_hp", 0.0) + c.get("p_ice_hp", 0.0)) * 0.7457
        em_component_kw = c.get("p_em_hp", 0.0) * 0.7457
        rho_ratio = rho_c / get_air_density(0)
        if gt_component_kw > 0:
            if c["type"] == "ICE":
                gt_component_kw *= rho_ratio
            else:
                gt_component_kw *= rho_ratio**0.7
        system_limit_kw = P_BASELINE_HP * 0.7457
        total_available_kw = min(system_limit_kw, gt_component_kw + em_component_kw)

        p_avail_prop_kw = total_available_kw * 0.95
        p_avail_thrust_kw = p_avail_prop_kw * eta_prop_climb
        roc_ms = ((p_avail_thrust_kw * 1000.0) - (drag_n * v_climb)) / max(1.0, (current_mass * 9.81))
        target_alt = phase.get("target_alt", 8000)
        if roc_ms <= 0.5:
            logging.warning(f"Climb rate <= 0.5 m/s ({roc_ms:.2f} m/s) for {c['type']}. Using fallback duration (3600s).")
            dur = 3600
            roc_ms = max(0.1, (target_alt / dur) * 0.00508)
        else:
            dur = (target_alt / (roc_ms * 3.28084))

        climb_energy_kw = max(0.0, (current_mass * 9.81 * roc_ms) / 1000.0)
        p_req_prop_kw = max(0.0, ((drag_n * v_climb) / 1000.0) + climb_energy_kw)

    # Hybrid power split logic
    p_gt_load, p_batt_load = 0, 0
    
    gt_max_hp = c.get("p_gt_hp", 0) + c.get("p_ice_hp", 0)
    gt_max_kw = gt_max_hp * 0.7457
    gen_max_kw = c.get("p_gen_kw", gt_max_kw if gt_max_kw > 0 else 0.0)
    em_max_kw = c.get("p_em_hp", 0) * 0.7457
    
    # De-rate max GT power by altitude
    if c["type"] == "ICE" and alt_ft > 0:
        gt_max_kw *= (rho / get_air_density(0)) # Piston power loss at alt
    elif c["type"] != "ICE" and gt_max_hp > 0 and alt_ft > 0: # Turbines
        gt_max_kw *= (rho / get_air_density(0))**0.7 # Less severe power loss
        
    # Guard against invalid derated power
    if gt_max_hp > 0 and gt_max_kw <= 0.01:
        logging.warning(f"GT max power (de-rated) is <= 0 ({gt_max_kw:.2f} kW) for {c['type']} @ {alt_ft} ft. Using 1kW fallback.")
        gt_max_kw = 1.0

    current_soc = current_batt_kwh / batt_kwh_max if batt_kwh_max > 0 else 0
    if batt_kwh_max > 0:
        batt_c_max = c.get("batt_c_max", 3.0)
        batt_c_chg_max = c.get("batt_c_chg_max", 1.0)
        batt_max_kw = batt_c_max * batt_kwh_max
        batt_chg_max_kw = batt_c_chg_max * batt_kwh_max
    else:
        batt_max_kw = 0.0
        batt_chg_max_kw = 0.0
    is_hybrid = c["type"] in ["Parallel", "Series", "Parallel-Series"]
    gt_mode = None  # For Series logging: OFF/MAX/SWEET
    fuel_type = c.get("fuel_type", "Jet-A")
    soc_frac_for_penalty = current_batt_kwh / batt_kwh_max if batt_kwh_max > 0 else 0.0
    if soc_frac_for_penalty >= ECMS_SOC_LOW_PENALTY:
        low_soc_penalty = 0.0
    else:
        deficit = (ECMS_SOC_LOW_PENALTY - soc_frac_for_penalty) / max(ECMS_SOC_LOW_PENALTY, 1e-6)
        low_soc_penalty = (deficit ** 3) * 50.0
    
    # Power required *at the shaft*
    p_req_shaft = p_req_prop_kw / eta_prop if eta_prop > 0.1 else p_req_prop_kw / 0.1
    
    # Ensure shaft demand stays non-negative
    if p_req_shaft <= 0:
        if phase["name"] in ["Taxi-Out", "Taxi-In"]:
                fallback_power = 0.05 * max(1.0, gt_max_kw)
                logging.info(f"Shaft power <= 0 ({p_req_shaft:.2f} kW) for {phase['name']}. Using {fallback_power:.1f} kW fallback.")
                p_req_shaft = fallback_power
        elif p_req_shaft < 0:
                p_req_shaft = 0
            
    # ECMS optimization logic
    ecms_failsafe_triggered = False
    if is_hybrid:
        if c["type"] == "Series":
            # Series hybrid ECMS search
            p_shaft_req = p_req_shaft
            p_motor_elec_kw = p_shaft_req / ETA_MOTOR
            p_gt_load = 0.0
            p_batt_load = 0.0
            gt_mode = "OFF"

            dynamic_s_factor = ECMS_EQUIVALENCE_FACTOR * (1.0 + low_soc_penalty)

            fractions = np.linspace(0, 1.0, 21)  # 0% ... 100% GT power
            best_cost = 1e9
            best_p_gt_kw = 0.0
            best_batt_rate_kw = 0.0

            base_sfc = c.get("gt_sfc_design", 0.45)
            eng_type = "GT" if c.get("p_gt_hp", 0) > 0 else "ICE"
            if eng_type == "ICE":
                base_sfc = c.get("sfc_design", 0.2)

            for f in fractions:
                p_gt_kw = f * gt_max_kw
                if p_gt_kw < 0 or (gt_max_kw > 0 and p_gt_kw > gt_max_kw):
                    continue
                if gt_max_kw <= 0 and p_gt_kw > 0:
                    continue

                load_frac = f if gt_max_kw > 0 else 0.0
                sfc_kg_hp_hr = get_sfc(load_frac, eng_type, base_sfc)
                fuel_rate = (sfc_kg_hp_hr * (p_gt_kw / 0.7457)) / 3600.0  # kg/s

                p_gen_bus_kw = min(p_gt_kw, gen_max_kw if gen_max_kw > 0 else p_gt_kw) * ETA_GEN

                # Power balance on the DC bus (generator first, battery handles remainder)
                if p_gen_bus_kw >= p_motor_elec_kw:
                    p_gen_to_batt_kw = p_gen_bus_kw - p_motor_elec_kw
                    batt_rate_kw = -p_gen_to_batt_kw * ETA_BATT  # charging (negative)
                else:
                    p_batt_to_motor_kw = p_motor_elec_kw - p_gen_bus_kw
                    batt_rate_kw = p_batt_to_motor_kw / ETA_BATT  # discharging (positive)

                if batt_kwh_max <= 0 and abs(batt_rate_kw) > 1e-6:
                    continue  # No battery available for charge/discharge

                batt_delta_kwh_candidate = batt_rate_kw * (dur / 3600.0)
                new_batt_kwh = current_batt_kwh - batt_delta_kwh_candidate

                if batt_rate_kw > 0 and is_reserve_calc:
                    continue  # Engine-only reserve calculation
                if batt_rate_kw > 0 and new_batt_kwh < -1e-3:
                    continue
                if batt_rate_kw < 0 and batt_kwh_max > 0:
                    charge_limit_kwh = batt_kwh_max * ECMS_SOC_MAX
                    if new_batt_kwh > charge_limit_kwh:
                        continue

                batt_rate_kwh_s = batt_rate_kw / 3600.0
                total_cost = fuel_rate + dynamic_s_factor * batt_rate_kwh_s

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_p_gt_kw = p_gt_kw
                    best_batt_rate_kw = batt_rate_kw

            if best_cost == 1e9:
                p_gt_load = min(gt_max_kw, max(0.0, p_shaft_req))
                p_batt_load = 0.0
                best_batt_rate_kw = 0.0
                ecms_failsafe_triggered = True
            else:
                p_gt_load = best_p_gt_kw
                p_batt_load = best_batt_rate_kw

            if gt_max_kw > 0:
                frac = p_gt_load / gt_max_kw
                if frac < 0.05:
                    gt_mode = "OFF"
                elif frac > 0.95:
                    gt_mode = "MAX"
                else:
                    gt_mode = "ECMS" if not ecms_failsafe_triggered else "FAIL"
            else:
                gt_mode = "OFF"
        
        elif c["type"] in ["Parallel", "Parallel-Series"]:
            
            dynamic_s_factor = ECMS_EQUIVALENCE_FACTOR * (1.0 + low_soc_penalty)

            p_shaft_req = p_req_shaft # Renamed for clarity
            fractions = np.linspace(0, 1.0, 21) # Define search grid
            best_cost, best_split = 1e9, (0, 0) # (p_gt_kw, p_batt_kw)

            base_sfc = c.get("gt_sfc_design", 0.45)
            eng_type = "GT" if c.get("p_gt_hp", 0) > 0 else "ICE"
            if eng_type == "ICE":
                base_sfc = c.get("sfc_design", 0.2)
            
            for f in fractions:
                p_gt_kw = f * gt_max_kw # Proposed GT shaft power
                
                p_em_shaft_req = max(0.0, p_shaft_req - p_gt_kw)

                # Apply EM Power Limit
                if p_em_shaft_req > em_max_kw:
                    p_gt_kw += (p_em_shaft_req - em_max_kw) # GT must provide the difference
                    p_em_shaft_req = em_max_kw
                
                # Apply GT Power Limit
                if p_gt_kw > gt_max_kw:
                    continue # This split is impossible

                # Calculate fuel consumption
                sfc_kg_hp_hr = get_sfc(f, eng_type, base_sfc)
                fuel_rate = (sfc_kg_hp_hr * (p_gt_kw / 0.7457)) / 3600.0

                # Calculate battery power draw (discharge is +, charge is -)
                batt_rate_kw = p_em_shaft_req / ETA_MOTOR  # discharge (+)
                if p_gt_kw > p_shaft_req and current_batt_kwh < (batt_kwh_max * ECMS_SOC_MAX):
                    surplus_shaft_kw = p_gt_kw - p_shaft_req
                    gen_input_kw = min(surplus_shaft_kw, gen_max_kw if gen_max_kw > 0 else surplus_shaft_kw)
                    batt_rate_kw = -gen_input_kw * ETA_GEN # recharge (-)
                
                if batt_rate_kw > 0 and is_reserve_calc:
                    continue
                new_batt_kwh = current_batt_kwh - (batt_rate_kw * (dur / 3600.0))
                if batt_rate_kw > 0 and new_batt_kwh < -1e-3:
                    continue
                if batt_rate_kw < 0 and batt_kwh_max > 0:
                    charge_limit_kwh = batt_kwh_max * ECMS_SOC_MAX
                    if new_batt_kwh > charge_limit_kwh:
                        continue
                
                batt_rate_kwh_s = batt_rate_kw / 3600.0
                total_cost = fuel_rate + dynamic_s_factor * batt_rate_kwh_s

                # Check if this is the best cost so far
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_split = (p_gt_kw, batt_rate_kw) # (GT shaft kW, Batt draw kW)


            if best_cost == 1e9:
                p_gt_load = gt_max_kw
                p_batt_load = 0
                if current_batt_kwh > 0 and em_max_kw > 0:
                    p_batt_load = em_max_kw / max(ETA_MOTOR, 0.1)
                best_split = (p_gt_load, p_batt_load)
                logging.warning(f"ECMS failsafe triggered for {phase['name']}. Demand may exceed total power.")
                ecms_failsafe_triggered = True

            p_gt_load, p_batt_load = best_split # Assign the best split
    
    elif not is_hybrid: # ICE/Turboprop
        p_gt_load = p_req_shaft # Engine provides all shaft power
        p_batt_load = 0

    fuel_kg, batt_delta = 0, 0
    if p_gt_load > 0:
        eng = "GT" if c.get("p_gt_hp", 0) > 0 else "ICE"
        base_sfc = c.get("gt_sfc_design", 0.45)
        
        if eng == "ICE": base_sfc = c.get("sfc_design", 0.2)
        
        if gt_max_kw <= 0.01:
            fuel_kg = 0
            if not is_hybrid: # Don't spam for hybrids
                logging.warning(f"!! CRITICAL WARNING: p_gt_load > 0 but gt_max_kw <= 0 for {c['type']} !!")
        else:
            load_frac = p_gt_load / gt_max_kw
            sfc_kg_hp_hr = get_sfc(load_frac, eng, base_sfc)
            p_gt_load_hp = p_gt_load / 0.7457 # Convert kW to HP for SFC calc
            fuel_kg = sfc_kg_hp_hr * p_gt_load_hp * (dur/3600)
            
    fuel_props = FUEL_PROPERTIES.get(fuel_type, FUEL_PROPERTIES["Jet-A"])
    fuel_mj = fuel_kg * fuel_props["specific_energy_mj_per_kg"]
    fuel_liters = fuel_kg / fuel_props["density"] if fuel_props["density"] > 0 else 0.0

    engine_type_for_emissions = "ICE" if c.get("type") == "ICE" else "GT"
    load_frac_for_emissions = 0.0
    if p_gt_load > 0 and gt_max_kw > 0:
        load_frac_for_emissions = max(0.0, min(1.0, p_gt_load / gt_max_kw))
    shaft_energy_kwh = (p_gt_load * dur) / 3600.0
    if shaft_energy_kwh > 0:
        nox_bse = get_bse_g_kw_hr(load_frac_for_emissions, engine_type_for_emissions, "NOx")
        pm_bse = get_bse_g_kw_hr(load_frac_for_emissions, engine_type_for_emissions, "PM")
        nox_g = nox_bse * shaft_energy_kwh
        pm_g = pm_bse * shaft_energy_kwh
    else:
        nox_g = 0.0
        pm_g = 0.0
            
    # Enforce battery charge/discharge power limits
    if batt_kwh_max > 0:
        if p_batt_load > batt_max_kw:
            p_batt_load = batt_max_kw
        if p_batt_load < -batt_chg_max_kw:
            p_batt_load = -batt_chg_max_kw

    # Calculate raw battery delta (kWh at battery terminals)
    batt_delta_raw = 0
    if batt_kwh_max > 0:
        batt_delta_raw = p_batt_load * (dur / 3600.0)
    
    allow_regen = (
        is_hybrid
        and batt_kwh_max > 0
        and phase["name"] in ["Descent", "Pattern"]
        and phase.get("alt", 0) > 1000
        and current_batt_kwh < batt_kwh_max * ECMS_SOC_MAX
    )
    if allow_regen:
        regen_shaft_kw = min(30.0, gen_max_kw if gen_max_kw > 0 else 30.0)
        if batt_chg_max_kw > 0:
            regen_shaft_kw = min(regen_shaft_kw, batt_chg_max_kw / max(ETA_GEN, 0.1))
        else:
            regen_shaft_kw = 0.0
        if regen_shaft_kw > 0:
            regen_delta_kwh_at_batt = -regen_shaft_kw * ETA_GEN * (dur / 3600.0)
            batt_delta_raw += regen_delta_kwh_at_batt # Add regen charge
    
    if batt_delta_raw > 0 and not is_reserve_calc and batt_kwh_max > 0:
        max_drawable_kwh = current_batt_kwh
        if batt_delta_raw > max_drawable_kwh:
            logging.warning(f"  Phase {phase['name']:<10}: Capping batt draw. Tried: {batt_delta_raw:.1f} kWh, Max: {max_drawable_kwh:.1f} kWh")
            batt_delta_raw = max_drawable_kwh

    if batt_kwh_max > 0:
        # Prevent discharging below zero
        if batt_delta_raw > current_batt_kwh:
            logging.warning(f"  Phase {phase['name']:<10}: Limiting batt draw to remaining SOC ({current_batt_kwh:.1f} kWh).")
            batt_delta_raw = current_batt_kwh
        # Prevent charging beyond capacity
        elif batt_delta_raw < 0:
            max_charge_kwh = batt_kwh_max - current_batt_kwh
            if -batt_delta_raw > max_charge_kwh:
                batt_delta_raw = -max_charge_kwh

    # Apply storage efficiency to final battery delta
    batt_delta = 0
    if batt_delta_raw < 0: # Charging
        batt_delta = batt_delta_raw * ETA_BATT # 10kWh sent -> 9.5kWh stored
    else: # Discharging
        batt_delta = batt_delta_raw # 10kWh needed -> 10kWh drawn
        # (Discharge efficiency was already handled in ETA_MOTOR)

    if is_hybrid:
        mode_str = f" | GTmode={gt_mode}" if (c.get("type") == "Series" and gt_mode is not None) else ""
        logging.info(f"  Phase {phase['name']:<10}: Dist={total_dist_km:6.0f} km | GT={p_gt_load:6.1f} kW | Batt={p_batt_load:6.1f} kW | SOC={current_soc:.2f}{mode_str}")

    if phase_log_list is not None:
        phase_log_list.append({
            "concept": c.get("name", "UNKNOWN"),
            "tech_scenario": globals().get("TECH_SCENARIO", "default"),
            "fuel_scenario": globals().get("FUEL_SCENARIO", "default"),
            "phase": phase["name"],
            "alt_ft": alt_ft,
            "v_kts": phase["v_kts"],
            "duration_s": dur,
            "dist_km": distance_km,
            "gt_power_kw": p_gt_load,
            "batt_power_kw": p_batt_load,
            "soc": current_batt_kwh / batt_kwh_max if batt_kwh_max > 0 else 0.0,
            "fuel_kg": fuel_kg,
        })

    return (
        fuel_kg,
        batt_delta,
        dur,
        distance_km,
        ecms_failsafe_triggered,
        fuel_mj,
        fuel_liters,
        nox_g,
        pm_g,
    )

# =========================================
# 5. MISSION EXECUTION
# =========================================

def run_mission(name, c):
    global BASELINE_CO2_PER_KM
    fuel_dens = JET_A_DENSITY if c["fuel_type"] == "Jet-A" else AVGAS_DENSITY
    current_fuel_kg = c["fuel_vol_L"] * fuel_dens
    start_fuel_kg = current_fuel_kg
    batt_max = c["batt_kwh"]
    current_batt_kwh = batt_max * 1.0 # Start fully charged
    
    # Get the concept's specific "clean" drag (fuselage + concept penalty)
    concept_cd0_base = arrow2_data["flap_data"][0]["CD0"] + c.get("cd0_adder", 0)
    
    # FIXED MASS CALCULATION
    mass_empty = (
        BASE_AIRFRAME_KG
        + c.get("wt_ice", 0)
        + c.get("wt_gt", 0)
        + c.get("wt_gen", 0)
        + c.get("wt_em", 0)
        + c["wt_batt"]
        + c["base_mass_adder"]
    )
    payload_kg = c.get("payload_kg", 150.0)
    current_mass = mass_empty + current_fuel_kg + payload_kg
    takeoff_mass = current_mass
    
    logging.info(f"Takeoff Mass: {current_mass:.0f} kg (MTOW: {MTOW_KG:.0f} kg)")
    logging.info(f"  (Empty: {mass_empty:.0f} kg, Fuel: {current_fuel_kg:.0f} kg, Batt: {c['wt_batt']:.0f} kg, Payload: {payload_kg:.0f} kg)")
    phase_log = []
    if current_mass > MTOW_KG:
        logging.warning(f"!! [FAILED: OVERWEIGHT] By {current_mass - MTOW_KG:.0f} kg !!")
        return None, phase_log

    total_dist_km = 0
    is_gas_turbine = (c.get("p_gt_hp", 0) > 0) and (c.get("fuel_type", "") == "Jet-A")
    use_saf = (FUEL_SCENARIO == "saf50") and is_gas_turbine
    total_fuel_mj = 0.0
    total_fuel_l = 0.0
    total_nox_g = 0.0
    total_pm_g = 0.0
    cruise_mode = c.get("cruise_mode", "economy")
    mission_track_deg = c.get("mission_track_deg", 90.0)
    if not weather_config.get("wind_profile"):
        fetch_wind_profile(weather_config)

    def adjust_for_remaining_fuel(available_fuel, burn_kg, fuel_mj, fuel_liters, nox_g, pm_g, phase_name):
        """Prevent negative landing fuel by scaling consumption to what remains."""
        if burn_kg <= available_fuel + 1e-6 or burn_kg <= 0:
            return burn_kg, fuel_mj, fuel_liters, nox_g, pm_g
        if available_fuel <= 0:
            logging.warning(f"!! [FAILED: OUT OF FUEL] Attempted {burn_kg:.2f} kg during {phase_name} but none left. Clamping to zero.")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        scale = available_fuel / burn_kg
        logging.warning(f"!! [WARNING: FUEL BELOW ZERO] Capping {phase_name} burn from {burn_kg:.2f} kg to available {available_fuel:.2f} kg.")
        return available_fuel, fuel_mj * scale, fuel_liters * scale, nox_g * scale, pm_g * scale
    
    dummy_mass_for_res = current_mass - (start_fuel_kg * 0.5) # Estimate landing mass
    res_soc_dummy_precalc = (
        batt_max * min(0.90, max(0.0, ECMS_SOC_MAX - 0.01)) if batt_max > 0 else 0.0
    )

    reserve_phases = [
        {"name": "Pattern", "dur": 240, "v_kts": 90, "p_req": 0.35, "alt": 1000},
        {"name": "Taxi-In", "dur": 300, "v_kts": 0, "p_req": 0.15, "alt": 0},
        {"name": "Go-Around", "dur": 120, "v_kts": 85, "p_req": 1.0, "alt": 1000},
    ]
    extra_reserve_fuel = 0.0
    reserve_cfg = deepcopy(c)
    if reserve_cfg.get("type") == "Parallel":
        reserve_cfg["p_em_hp"] = 0.0
    for rp in reserve_phases:
        ph = apply_weather_to_phase(rp.copy(), weather_config, mission_track_deg)
        f, _, _, _, _, _, _, _, _ = calculate_phase_burns(
            ph,
            reserve_cfg,
            dummy_mass_for_res,
            res_soc_dummy_precalc,
            batt_max,
            concept_cd0_base,
            total_dist_km,
            is_reserve_calc=True,
        )
        if f > 0:
            extra_reserve_fuel += f

    vfr_reserve_fuel_kg = compute_vfr_day_reserve_fuel(
        c,
        dummy_mass_for_res,
        current_batt_kwh,
        batt_max,
        concept_cd0_base,
        total_dist_km
    )
    res_fuel_precalc = extra_reserve_fuel + vfr_reserve_fuel_kg

    # DEPARTURE (Climb to 10k)
    for ph_name in ["Taxi-Out", "Takeoff", "Climb"]:
        if ph_name == "Climb":
            climb_alt_ft = 5000
            v_kias = get_climb_speed_kias(current_mass, climb_alt_ft)
            ph_v_kts = kias_to_tas(v_kias, climb_alt_ft)
            ph_alt = climb_alt_ft
        elif "Taxi" in ph_name:
            ph_v_kts = 0
            ph_alt = 0
        elif "Take" in ph_name:
            ph_v_kts = 75
            ph_alt = 0
        else:
            ph_v_kts = 100
            ph_alt = 0
        ph = {"name":ph_name, "dur":480 if "Taxi" in ph_name else 40,
                "v_kts":ph_v_kts,
                "p_req":0.15 if "Taxi" in ph_name else (1.0 if "Take" in ph_name else 0.95),
                "target_alt": 10000 if ph_name == "Climb" else 0,
                "alt": ph_alt} # Use avg alt for climb calc
        ph = apply_weather_to_phase(ph, weather_config, mission_track_deg)
        
        # Pass the pre-calculated reserve battery requirement
        f, b, dur, d_dist, _, fuel_mj, fuel_liters, nox_g, pm_g = calculate_phase_burns(
            ph,
            c,
            current_mass,
            current_batt_kwh,
            batt_max,
            concept_cd0_base,
            total_dist_km,
            phase_log_list=phase_log,
        )
        f, fuel_mj, fuel_liters, nox_g, pm_g = adjust_for_remaining_fuel(
            current_fuel_kg, f, fuel_mj, fuel_liters, nox_g, pm_g, ph_name
        )
        
        current_fuel_kg -= f; current_mass -= f; current_batt_kwh -= b; total_dist_km += d_dist
        current_batt_kwh = max(0.0, min(current_batt_kwh, batt_max)) # Cap SOC within bounds
        total_fuel_mj += fuel_mj
        total_fuel_l += fuel_liters
        total_nox_g += nox_g
        total_pm_g += pm_g

    # RESERVES
    # Use the pre-calculated values
    res_fuel = res_fuel_precalc
    if current_fuel_kg < res_fuel:
        logging.warning(f"!! [FAILED: INSUFFICIENT FUEL FOR RESERVES] Fuel Req: {res_fuel:.1f}kg, Have: {current_fuel_kg:.1f}kg !!"); return None, phase_log


    # CRUISE
    current_cruise_alt = 10000
    cruise_speed = pick_optimal_cruise_speed(c, current_mass, concept_cd0_base, current_cruise_alt, cruise_mode)
    cruise_ph = {"name":"Cruise", "dur":60, "v_kts":cruise_speed, "alt": current_cruise_alt}
    cruise_mins = 0
    while True:
        cruise_ph["alt"] = current_cruise_alt # Update altitude for this loop iteration
        cruise_ph["v_kts"] = pick_optimal_cruise_speed(c, current_mass, concept_cd0_base, current_cruise_alt, cruise_mode)
        cruise_ph = apply_weather_to_phase(cruise_ph, weather_config, mission_track_deg)
        
        if (current_fuel_kg - res_fuel) < 0.5:
            break
        
        f, b, _, d_dist, failsafe_triggered, fuel_mj, fuel_liters, nox_g, pm_g = calculate_phase_burns(
            cruise_ph,
            c,
            current_mass,
            current_batt_kwh,
            batt_max,
            concept_cd0_base,
            total_dist_km,
            phase_log_list=phase_log,
        )
        f, fuel_mj, fuel_liters, nox_g, pm_g = adjust_for_remaining_fuel(
            current_fuel_kg, f, fuel_mj, fuel_liters, nox_g, pm_g, "Cruise"
        )
        
        if failsafe_triggered:
            if current_cruise_alt == 10000:
                logging.warning(f"!! ECMS Failsafe at 10,000 ft. Descending to 6,000 ft to continue cruise.")
                current_cruise_alt = 6000
                continue # Restart loop at new altitude
            elif current_cruise_alt == 6000:
                logging.warning(f"!! ECMS Failsafe triggered again at 6,000 ft. Ending cruise.")
                break # Exit cruise loop
            
        # Check if this minute of cruise would put us BELOW our reserves
        # This check is now a redundant failsafe, as the ECMS should handle it
        if (current_fuel_kg - f) < res_fuel:
            logging.info(f"Cruise ended at {current_cruise_alt} ft: Reached reserve limits.")
            break # Stop cruising, we're at our reserve limit
            
        current_fuel_kg -= f; current_mass -= f; current_batt_kwh -= b; total_dist_km += d_dist; cruise_mins += 1
        current_batt_kwh = max(0.0, min(current_batt_kwh, batt_max)) # Cap SOC within bounds
        total_fuel_mj += fuel_mj
        total_fuel_l += fuel_liters
        total_nox_g += nox_g
        total_pm_g += pm_g
        if cruise_mins > 600: break # Safety exit after 10 hrs

    # DESCENT
    descent_mid_alt = max(1000.0, current_cruise_alt / 2.0)
    v_desc_kias = get_descent_speed_kias(current_mass, descent_mid_alt)
    v_desc_ktas = kias_to_tas(v_desc_kias, descent_mid_alt)
    descent_dur_s = max(120.0, (current_cruise_alt / 1000) * 100)
    descent_ph = {
        "name": "Descent",
        "dur": descent_dur_s,
        "v_kts": v_desc_ktas,
        "p_req": 0.35,
        "alt": descent_mid_alt,
        "vsink_ms": 3.0,
    }
    descent_ph = apply_weather_to_phase(descent_ph, weather_config, mission_track_deg)
    f, b, _, d_dist, _, fuel_mj, fuel_liters, nox_g, pm_g = calculate_phase_burns(
        descent_ph,
        c,
        current_mass,
        current_batt_kwh,
        batt_max,
        concept_cd0_base,
        total_dist_km,
        phase_log_list=phase_log,
    )
    f, fuel_mj, fuel_liters, nox_g, pm_g = adjust_for_remaining_fuel(
        current_fuel_kg, f, fuel_mj, fuel_liters, nox_g, pm_g, "Descent"
    )
    current_fuel_kg -= f
    current_mass -= f
    current_batt_kwh -= b
    current_batt_kwh = max(0.0, min(current_batt_kwh, batt_max))
    total_dist_km += d_dist
    total_fuel_mj += fuel_mj
    total_fuel_l += fuel_liters
    total_nox_g += nox_g
    total_pm_g += pm_g
    
    # ENVIRON
    if total_dist_km > 1.0:
        total_fuel_burned = start_fuel_kg - current_fuel_kg
        if c["fuel_type"] == "Avgas":
            co2_factor = CO2_AVGAS_KG
            nox_factor = NOX_AVGAS_KG
            pm_factor = PM_AVGAS_KG
        else:
            co2_factor, nox_factor, pm_factor = get_jet_a_emission_factors(use_saf)

        co2_per_km_g = (total_fuel_burned * co2_factor * 1000.0) / total_dist_km
        nox_per_km_g = (total_fuel_burned * nox_factor * 1000.0) / total_dist_km
        pm_per_km_mg = (total_fuel_burned * pm_factor * 1e6) / total_dist_km

        if name == "Baseline (Avgas)":
            BASELINE_CO2_PER_KM = co2_per_km_g
            env_str = f"{co2_per_km_g:.0f} g/km CO2 (Baseline)"
        elif BASELINE_CO2_PER_KM is not None:
            diff_pct = ((co2_per_km_g - BASELINE_CO2_PER_KM) / BASELINE_CO2_PER_KM) * 100
            env_str = f"{co2_per_km_g:.0f} g/km CO2 ({abs(diff_pct):.1f}% {'worse' if diff_pct > 0 else 'better'})"
        else:
            env_str = f"{co2_per_km_g:.0f} g/km CO2"
    else:
        co2_per_km_g = 0.0
        nox_per_km_g = 0.0
        pm_per_km_mg = 0.0
        env_str = "N"

    final_soc = current_batt_kwh / batt_max if batt_max > 0 else 0
    logging.info(f"Cruise: {cruise_mins/60:.1f} hrs | Range: {total_dist_km:.0f} km | Land Fuel: {current_fuel_kg:.1f} kg | Fuel Res: {res_fuel:.1f} kg | Land SOC: {final_soc:.2f}")
    logging.info(f"ENVIRON: {env_str} | NOx: {nox_per_km_g:.2f} g/km | PM: {pm_per_km_mg:.1f} mg/km")
    if total_dist_km > 0:
        energy_per_km_mj = total_fuel_mj / total_dist_km
        volume_per_km_l = total_fuel_l / total_dist_km
        nox_per_km = total_nox_g / total_dist_km
        pm_per_km = total_pm_g / total_dist_km
    else:
        energy_per_km_mj = volume_per_km_l = nox_per_km = pm_per_km = 0.0
    logging.info(
        f"Intensity: {energy_per_km_mj:.2f} MJ/km | Fuel Vol: {volume_per_km_l:.2f} L/km | NOx: {nox_per_km:.2f} g/km | PM: {pm_per_km:.3f} g/km"
    )
    marketable = total_dist_km >= 330
    if not marketable:
        logging.warning(">> RESULT: [FAILED] NOT MARKETABLE (< 330 km)")
    else:
        logging.info(">> RESULT: [SUCCESS] MARKETABLE")

    result = {
        "concept": name,
        "tech_scenario": globals().get("TECH_SCENARIO", "default"),
        "fuel_scenario": globals().get("FUEL_SCENARIO", "default"),
        "range_km": total_dist_km,
        "cruise_hours": cruise_mins / 60.0,
        "land_fuel_kg": current_fuel_kg,
        "fuel_reserve_kg": res_fuel,
        "land_soc": final_soc,
        "co2_g_per_km": co2_per_km_g if total_dist_km > 0 else 0.0,
        "nox_g_per_km": nox_per_km_g if total_dist_km > 0 else 0.0,
        "pm_mg_per_km": pm_per_km_mg if total_dist_km > 0 else 0.0,
        "marketable": marketable,
        "takeoff_mass_kg": takeoff_mass,
        "payload_kg": payload_kg,
    }
    return result, phase_log

def run_all_concepts():
    global TECH_SCENARIO, FUEL_SCENARIO, BASELINE_CO2_PER_KM
    summary_rows = []
    phase_rows = []

    for scenario in ["realistic", "optimistic_future"]:
        TECH_SCENARIO = scenario
        configure_efficiencies(scenario)
        scenario_concepts = build_concepts_for_scenario(scenario)

        for fuel_scenario in ["fossil", "saf50"]:
            FUEL_SCENARIO = fuel_scenario
            BASELINE_CO2_PER_KM = None
            print(f"\n\n===== TECH: {scenario.upper()} | FUEL: {fuel_scenario.upper()} =====\n")

            for name, data in scenario_concepts.items():
                concept_instance = deepcopy(data)
                concept_instance["name"] = name
                result, phase_log = run_mission(name, concept_instance)
                phase_rows.extend(phase_log)
                if result is not None:
                    summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)
    phases_df = pd.DataFrame(phase_rows)

    if not summary_df.empty:
        print("\n=== SUMMARY ===")
        cols = ["concept", "tech_scenario", "fuel_scenario", "range_km",
                "co2_g_per_km", "nox_g_per_km", "pm_mg_per_km", "land_soc"]
        print(summary_df[cols])
        summary_df.to_csv("hybrid_summary.csv", index=False)
    if not phases_df.empty:
        phases_df.to_csv("hybrid_phases.csv", index=False)
    if not summary_df.empty or not phases_df.empty:
        print("\nWrote hybrid_summary.csv and hybrid_phases.csv")

    make_plots(summary_df, phases_df)
    return summary_df, phases_df


def make_plots(summary_df, phases_df):
    if summary_df.empty:
        return

    plt.figure()
    for concept, sub in summary_df.groupby("concept"):
        plt.scatter(sub["range_km"], sub["co2_g_per_km"], label=concept)
    plt.xlabel("Range [km]")
    plt.ylabel("COâ‚‚ [g/km]")
    plt.legend()
    plt.title("Range vs COâ‚‚")
    plt.tight_layout()
    plt.savefig("plot_range_vs_co2.png")

    plt.figure()
    for concept, sub in summary_df.groupby("concept"):
        plt.scatter(sub["range_km"], sub["nox_g_per_km"], label=concept)
    plt.xlabel("Range [km]")
    plt.ylabel("NOx [g/km]")
    plt.legend()
    plt.title("Range vs NOx")
    plt.tight_layout()
    plt.savefig("plot_range_vs_nox.png")

    if phases_df.empty:
        return

    combos = phases_df[["concept", "tech_scenario", "fuel_scenario"]].drop_duplicates().head(3)
    for _, row in combos.iterrows():
        plot_soc_profile(
            phases_df,
            row["concept"],
            row["tech_scenario"],
            row["fuel_scenario"],
        )


def plot_soc_profile(phases_df, concept, tech, fuel):
    sub = phases_df[
        (phases_df["concept"] == concept) &
        (phases_df["tech_scenario"] == tech) &
        (phases_df["fuel_scenario"] == fuel)
    ].copy()

    if sub.empty:
        return

    sub = sub.reset_index(drop=True)
    sub["cum_dist_km"] = sub["dist_km"].cumsum()

    plt.figure()
    plt.plot(sub["cum_dist_km"], sub["soc"])
    plt.xlabel("Cumulative Distance [km]")
    plt.ylabel("SOC [-]")
    plt.title(f"SOC profile â€“ {concept} | {tech} | {fuel}")
    plt.tight_layout()
    fname = f"soc_{concept}_{tech}_{fuel}.png".replace(" ", "_")
    plt.savefig(fname)


if __name__ == "__main__":
    run_all_concepts()

