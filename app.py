import math
from copy import deepcopy
import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import hybrid_aircraft_sim as sim

st.set_page_config(page_title="Hybrid Piper Sim", layout="wide")
st.title("Hybrid Piper Sim")

TEUGE_CENTER = [6.05, 52.24]  # [lon, lat]
CONFIG_FILE = Path(__file__).with_name("user_concept_configs.json")
CONCEPT_COLORS = {
    "Baseline (Avgas)": [0, 102, 204],
    "4.1.2 Parallel Hybrid": [220, 53, 69],
    "4.1.3 Series Hybrid": [25, 135, 84],
    "4.1.4 Parallel-Series": [255, 159, 64],
    "4.1.5 Turboprop": [111, 66, 193],
}


def init_state():
    defaults = {
        "summary_df": None,
        "phases_df": None,
        "last_result": None,
        "concept_cfg": None,
        "concept_key": None,
        "baseline_refs": {},
        "saved_configs": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def load_saved_configs():
    if st.session_state["saved_configs"] is None:
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except Exception:
                data = {}
        else:
            data = {}
        st.session_state["saved_configs"] = data


def persist_configs():
    try:
        CONFIG_FILE.write_text(json.dumps(st.session_state["saved_configs"], indent=2))
    except Exception as exc:
        st.warning(f"Could not persist configs: {exc}")


load_saved_configs()


def get_saved_config(concept_label):
    saved = st.session_state.get("saved_configs") or {}
    return deepcopy(saved.get(concept_label, {}))


def save_concept_config(concept_label, cfg):
    saved = st.session_state.get("saved_configs") or {}
    saved[concept_label] = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str))}
    st.session_state["saved_configs"] = saved
    persist_configs()


def make_circle_polygon(center_lon, center_lat, radius_km, n_points=180):
    """Return a polygon approximating a range circle on Earth."""
    earth_radius_km = 6371.0
    coords = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        d_by_r = radius_km / earth_radius_km
        lat_rad = math.asin(
            math.sin(math.radians(center_lat)) * math.cos(d_by_r)
            + math.cos(math.radians(center_lat)) * math.sin(d_by_r) * math.cos(angle)
        )
        lon_rad = math.radians(center_lon) + math.atan2(
            math.sin(angle) * math.sin(d_by_r) * math.cos(math.radians(center_lat)),
            math.cos(d_by_r) - math.sin(math.radians(center_lat)) * math.sin(lat_rad),
        )
        coords.append([math.degrees(lon_rad), math.degrees(lat_rad)])
    return [coords]


def make_directional_polygon(center_lon, center_lat, samples):
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: s["track_deg"])
    coords = []
    lat1 = math.radians(center_lat)
    lon1 = math.radians(center_lon)
    earth_radius_km = 6371.0
    for sample in samples:
        dist_ratio = sample["range_km"] / earth_radius_km
        bearing = math.radians(sample["track_deg"])
        lat2 = math.asin(
            math.sin(lat1) * math.cos(dist_ratio)
            + math.cos(lat1) * math.sin(dist_ratio) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(dist_ratio) * math.cos(lat1),
            math.cos(dist_ratio) - math.sin(lat1) * math.sin(lat2),
        )
        coords.append([math.degrees(lon2), math.degrees(lat2)])
    coords.append(coords[0])
    return [coords]


def get_baseline_reference(tech_scenario, fuel_scenario):
    """Run or retrieve the reference Baseline (Avgas) mission for comparison."""
    cache_key = f"{tech_scenario}_{fuel_scenario}"
    refs = st.session_state.get("baseline_refs", {})
    if cache_key in refs:
        return refs[cache_key]
    concepts = sim.build_concepts_for_scenario(tech_scenario)
    baseline_cfg = concepts.get("Baseline (Avgas)")
    if baseline_cfg is None:
        return None
    baseline_cfg = deepcopy(baseline_cfg)
    baseline_cfg["name"] = "Baseline (Avgas)"

    prev_tech = sim.TECH_SCENARIO
    prev_fuel = sim.FUEL_SCENARIO
    if prev_tech != tech_scenario:
        sim.TECH_SCENARIO = tech_scenario
        sim.configure_efficiencies(tech_scenario)
    sim.FUEL_SCENARIO = fuel_scenario

    result, _ = sim.run_mission("Baseline (Avgas)", baseline_cfg)
    refs[cache_key] = result
    st.session_state["baseline_refs"] = refs

    if prev_tech != tech_scenario:
        sim.TECH_SCENARIO = prev_tech
        sim.configure_efficiencies(prev_tech)
    sim.FUEL_SCENARIO = prev_fuel
    return result


@st.cache_data(show_spinner=False)
def run_concepts_for_map(
    tech_scenario,
    fuel_scenario,
    weather_signature,
    weather_snapshot,
    overrides_signature,
):
    """Run all default concepts for the selected scenario/fuel with specified weather."""
    prev_tech = sim.TECH_SCENARIO
    prev_fuel = sim.FUEL_SCENARIO
    prev_eta = (sim.ETA_MOTOR, sim.ETA_GEN, sim.ETA_BATT)
    prev_weather = deepcopy(sim.weather_config)

    sim.TECH_SCENARIO = tech_scenario
    sim.configure_efficiencies(tech_scenario)
    sim.FUEL_SCENARIO = fuel_scenario
    sim.weather_config = deepcopy(weather_snapshot)

    overrides = {}
    if overrides_signature:
        try:
            overrides = json.loads(overrides_signature)
        except Exception:
            overrides = {}

    concepts = sim.build_concepts_for_scenario(tech_scenario)
    rows = []
    skipped = []
    for name, cfg in concepts.items():
        cfg = deepcopy(cfg)
        cfg["name"] = name
        if name in overrides:
            cfg.update(overrides[name])
        try:
            result, _ = sim.run_mission(name, cfg)
        except Exception as exc:
            logging.warning(f"run_mission failed for {name}: {exc}")
            skipped.append(name)
            continue
        if result is None:
            skipped.append(name)
            continue

        takeoff_mass = compute_takeoff_mass(cfg)
        batt_max = cfg.get("batt_kwh", 0.0)
        base_cd0 = (
            getattr(sim, "arrow2_data", {})
            .get("flap_data", {})
            .get(0, {})
            .get("CD0", 0.027)
        )
        concept_cd0_base = base_cd0 + cfg.get("cd0_adder", 0.0)
        start_fuel_kg = compute_start_fuel_kg(cfg)
        reserve_fuel_kg = result.get("fuel_reserve_kg", 0.0)
        available_fuel_kg = max(start_fuel_kg - reserve_fuel_kg, 0.0)

        samples = []
        try:
            samples = sim.sample_directional_ranges(
                cfg,
                takeoff_mass,
                batt_max,
                batt_max,
                concept_cd0_base,
                sim.weather_config,
                available_fuel_kg,
                step_deg=15,
            )
        except Exception as exc:
            logging.warning(f"Directional range sampling failed for {name}: {exc}")
            samples = []

        ranges = [max(s["range_km"], 0.0) for s in samples]
        row_range = max(result.get("range_km", 0.0), 0.0)
        result["range_mission_km"] = row_range
        if not ranges:
            samples = []
            ranges = []
        else:
            avg_sample = sum(ranges) / len(ranges)
            target_range = row_range if row_range > 0 else avg_sample
            if avg_sample > 0 and target_range > 0:
                scale = target_range / avg_sample
                ranges = [max(r * scale, 0.0) for r in ranges]
            for sample, r in zip(samples, ranges):
                sample["range_km"] = r

        if samples and ranges:
            result["range_samples"] = samples
            result["range_best_km"] = max(ranges)
            result["range_worst_km"] = min(ranges)
            result["range_avg_km"] = sum(ranges) / len(ranges)
        else:
            result["range_samples"] = []
            result["range_best_km"] = row_range
            result["range_worst_km"] = row_range
            result["range_avg_km"] = row_range

        rows.append(result)

    sim.TECH_SCENARIO = prev_tech
    sim.configure_efficiencies(prev_tech)
    sim.FUEL_SCENARIO = prev_fuel
    sim.ETA_MOTOR, sim.ETA_GEN, sim.ETA_BATT = prev_eta
    sim.weather_config = prev_weather

    if skipped:
        logging.warning(f"Skipped concepts in overlay (failed mission): {', '.join(skipped)}")

    df = pd.DataFrame(rows)
    if not df.empty:
        if "range_mission_km" not in df.columns:
            df["range_mission_km"] = df.get("range_km", 0.0)
        if "range_avg_km" in df.columns:
            df["range_km"] = df["range_avg_km"].where(~df["range_avg_km"].isna(), df["range_km"])
    return df


def update_weather_configuration(mode_label, lat, lon, date_value, time_value):
    mode_map = {
        "None": "none",
        "Live weather": "live",
        "Mock": "mock",
        "Select by date/time": "by_date",
    }
    mode = mode_map.get(mode_label, "mock")
    config = sim.weather_config
    config["mode"] = mode
    config["lat"] = lat
    config["lon"] = lon
    if mode == "by_date":
        if date_value is not None and time_value is not None:
            dt = datetime.datetime.combine(date_value, time_value, tzinfo=datetime.timezone.utc)
            config["datetime_utc"] = dt.isoformat()
        else:
            config["datetime_utc"] = None
    else:
        config["datetime_utc"] = None
    config["wind_profile"] = []
    sim.weather_config = config
    sim.fetch_wind_profile(sim.weather_config)
    signature = f"{mode}|{lat:.3f}|{lon:.3f}|{config.get('datetime_utc','')}"
    return signature, deepcopy(sim.weather_config)


def build_wind_arrows(weather_config, altitude_ft, spacing_deg, max_arrows=100):
    if not weather_config.get("wind_profile"):
        return []
    arrow_data = []
    lat_min, lat_max = 45.0, 60.0
    lon_min, lon_max = -5.0, 15.0
    spacing = max(1.0, spacing_deg)
    lat = lat_min
    while lat <= lat_max and len(arrow_data) < max_arrows:
        lon = lon_min
        while lon <= lon_max and len(arrow_data) < max_arrows:
            wind_kt, wind_dir = sim.get_wind_at_alt(altitude_ft, weather_config["wind_profile"])
            wind_to_deg = (wind_dir + 180.0) % 360.0
            length_deg = 0.3 * (wind_kt / 30.0)
            dx = math.cos(math.radians(wind_to_deg)) * length_deg
            dy = math.sin(math.radians(wind_to_deg)) * length_deg
            arrow_data.append(
                {
                    "source": [lon, lat],
                    "target": [lon + dx, lat + dy],
                    "speed": wind_kt,
                }
            )
            lon += spacing
        lat += spacing
    return arrow_data


def compute_takeoff_mass(cfg):
    fuel_dens = sim.JET_A_DENSITY if cfg.get("fuel_type", "Jet-A") == "Jet-A" else sim.AVGAS_DENSITY
    fuel_kg = cfg.get("fuel_vol_L", 0.0) * fuel_dens
    mass_empty = (
        sim.BASE_AIRFRAME_KG
        + cfg.get("wt_ice", 0.0)
        + cfg.get("wt_gt", 0.0)
        + cfg.get("wt_gen", 0.0)
        + cfg.get("wt_em", 0.0)
        + cfg.get("wt_batt", 0.0)
        + cfg.get("base_mass_adder", 0.0)
    )
    payload = cfg.get("payload_kg", 150.0)
    return mass_empty + fuel_kg + payload


def compute_start_fuel_kg(cfg):
    fuel_dens = sim.JET_A_DENSITY if cfg.get("fuel_type", "Jet-A") == "Jet-A" else sim.AVGAS_DENSITY
    return cfg.get("fuel_vol_L", 0.0) * fuel_dens


def style_summary_vs_baseline(df, baseline_row):
    compare_cols = {
        "range_km": "higher",
        "range_best_km": "higher",
        "range_avg_km": "higher",
        "range_worst_km": "higher",
        "co2_g_per_km": "lower",
        "nox_g_per_km": "lower",
        "pm_mg_per_km": "lower",
        "land_fuel_kg": "higher",
        "landing_soc": "higher",
    }
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col, preference in compare_cols.items():
        if col not in df or pd.isna(baseline_row.get(col)):
            continue
        base_val = baseline_row[col]
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            if preference == "higher":
                styles.loc[idx, col] = "color: green" if val >= base_val else "color: red"
            else:
                styles.loc[idx, col] = "color: green" if val <= base_val else "color: red"
    return styles


def bounded(value, *, min_value, max_value, default):
    """Clamp values so Streamlit inputs always start inside allowed bounds."""
    if value is None:
        value = default
    try:
        value = float(value)
    except (ValueError, TypeError):
        value = default
    return max(min_value, min(max_value, value))


def render_parameter_controls(concept_cfg):
    st.markdown("### Propulsion parameters")
    prop_cols = st.columns(3)

    concept_cfg["p_gt_hp"] = prop_cols[0].number_input(
        "GT power (hp)",
        min_value=0.0,
        max_value=1000.0,
        value=float(concept_cfg.get("p_gt_hp", 0.0)),
        step=5.0,
    )
    concept_cfg["wt_gt"] = prop_cols[1].number_input(
        "GT mass (kg)",
        min_value=0.0,
        max_value=500.0,
        value=float(concept_cfg.get("wt_gt", 0.0)),
        step=5.0,
    )
    default_sfc = bounded(
        concept_cfg.get("gt_sfc_design"),
        min_value=0.1,
        max_value=1.0,
        default=0.3,
    )
    concept_cfg["gt_sfc_design"] = prop_cols[2].number_input(
        "GT SFC (kg/hp-hr)",
        min_value=0.1,
        max_value=1.0,
        value=default_sfc,
        step=0.01,
    )

    prop_cols2 = st.columns(3)
    concept_cfg["p_em_hp"] = prop_cols2[0].number_input(
        "Motor power (hp)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("p_em_hp", 0.0)),
        step=5.0,
    )
    concept_cfg["wt_em"] = prop_cols2[1].number_input(
        "Motor mass (kg)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("wt_em", 0.0)),
        step=5.0,
    )
    concept_cfg["batt_kwh"] = prop_cols2[2].number_input(
        "Battery capacity (kWh)",
        min_value=0.0,
        max_value=500.0,
        value=float(concept_cfg.get("batt_kwh", 0.0)),
        step=5.0,
    )
    prop_cols3 = st.columns(3)
    concept_cfg["p_gen_kw"] = prop_cols3[0].number_input(
        "Generator power (kW)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("p_gen_kw", concept_cfg.get("p_gt_hp", 0) * 0.7457)),
        step=5.0,
    )
    concept_cfg["wt_gen"] = prop_cols3[1].number_input(
        "Generator mass (kg)",
        min_value=0.0,
        max_value=200.0,
        value=float(concept_cfg.get("wt_gen", 0.0)),
        step=2.0,
    )
    if concept_cfg.get("batt_kwh", 0) > 0:
        batt_cols = st.columns(2)
        concept_cfg["batt_c_max"] = batt_cols[0].number_input(
            "Battery max discharge C-rate",
            min_value=0.1,
            max_value=6.0,
            value=float(concept_cfg.get("batt_c_max", 3.0)),
            step=0.1,
        )
        concept_cfg["batt_c_chg_max"] = batt_cols[1].number_input(
            "Battery max charge C-rate",
            min_value=0.1,
            max_value=3.0,
            value=float(concept_cfg.get("batt_c_chg_max", 1.0)),
            step=0.1,
        )

    st.markdown("### Airframe modifiers")
    air_cols = st.columns(3)
    concept_cfg["fuel_vol_L"] = air_cols[0].number_input(
        "Fuel volume (L)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("fuel_vol_L", 0.0)),
        step=5.0,
    )
    concept_cfg["cd0_adder"] = air_cols[1].number_input(
        "Extra drag (cd0_adder)",
        min_value=0.0,
        max_value=0.05,
        value=float(concept_cfg.get("cd0_adder", 0.0)),
        step=0.0005,
        format="%.4f",
    )
    concept_cfg["base_mass_adder"] = air_cols[2].number_input(
        "Base mass adder (kg)",
        min_value=-200.0,
        max_value=200.0,
        value=float(concept_cfg.get("base_mass_adder", 0.0)),
        step=5.0,
    )
    mission_cols = st.columns(2)
    concept_cfg["payload_kg"] = mission_cols[0].number_input(
        "Payload mass (kg)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("payload_kg", 150.0)),
        step=5.0,
    )

    st.markdown("### Mission options")
    cruise_mode_display = {
        "economy": "Economy (best range)",
        "fast": "Fast (high speed)",
    }
    current_mode = concept_cfg.get("cruise_mode", "economy")
    mode_options = list(cruise_mode_display.keys())
    selected_mode = st.selectbox(
        "Cruise mode",
        options=mode_options,
        format_func=lambda x: cruise_mode_display.get(x, x),
        index=mode_options.index(current_mode) if current_mode in mode_options else 0,
    )
    concept_cfg["cruise_mode"] = selected_mode

    return concept_cfg


# Sidebar for scenario choices
st.sidebar.header("Scenario")
concept_label = st.sidebar.selectbox(
    "Concept",
    [
        "Baseline (Avgas)",
        "4.1.2 Parallel Hybrid",
        "4.1.3 Series Hybrid",
        "4.1.4 Parallel-Series",
        "4.1.5 Turboprop",
    ],
)
tech_scenario = st.sidebar.selectbox(
    "Tech scenario",
    ["realistic", "optimistic_future"],
)
fuel_scenario = st.sidebar.selectbox(
    "Fuel scenario",
    ["fossil", "saf50"],
)

st.sidebar.header("Weather")
weather_mode_label = st.sidebar.selectbox(
    "Weather mode",
    ["Mock", "Live weather", "Select by date/time", "None"],
    index=0 if sim.weather_config.get("mode") in (None, "mock") else
    (1 if sim.weather_config.get("mode") == "live" else 2 if sim.weather_config.get("mode") == "by_date" else 3),
)
weather_lat = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=float(sim.weather_config.get("lat", 52.24)),
)
weather_lon = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=float(sim.weather_config.get("lon", 6.05)),
)
date_value = None
time_value = None
if weather_mode_label == "Select by date/time":
    default_date = datetime.date.today()
    default_time = datetime.time(hour=datetime.datetime.utcnow().hour, minute=0)
    date_value = st.sidebar.date_input("Date (UTC)", value=default_date)
    time_value = st.sidebar.time_input("Time (UTC)", value=default_time)

arrow_toggle = st.sidebar.checkbox("Show wind arrows", value=True)
arrow_spacing = st.sidebar.slider("Arrow spacing (deg)", 1, 6, 3)
arrow_altitude = st.sidebar.slider("Arrow altitude (ft)", 0, 12000, 6000, step=500)

weather_signature, weather_snapshot = update_weather_configuration(
    weather_mode_label, weather_lat, weather_lon, date_value, time_value
)
st.session_state["wind_display"] = {
    "show_arrows": arrow_toggle,
    "arrow_spacing": arrow_spacing,
    "arrow_altitude": arrow_altitude,
}
if st.sidebar.button("Reset all parameters to defaults"):
    st.session_state["saved_configs"] = {}
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    st.session_state["concept_cfg"] = None
    st.session_state["concept_key"] = None
    rerunner = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerunner:
        rerunner()
saved_configs_snapshot = st.session_state.get("saved_configs") or {}
overrides_signature = json.dumps(saved_configs_snapshot, sort_keys=True)
summary_df = st.session_state["summary_df"]
phases_df = st.session_state["phases_df"]
summary_all_data = run_concepts_for_map(
    tech_scenario,
    fuel_scenario,
    weather_signature,
    weather_snapshot,
    overrides_signature,
)
if summary_df is not None and not summary_df.empty:
    mask = summary_all_data["concept"] == summary_df.iloc[0]["concept"]
    if any(mask):
        for key, val in summary_df.iloc[0].items():
            if key in summary_all_data.columns:
                summary_all_data.loc[mask, key] = val
        if "range_mission_km" in summary_all_data.columns:
            summary_all_data.loc[mask, "range_mission_km"] = summary_df.iloc[0].get(
                "range_km", summary_all_data.loc[mask, "range_mission_km"]
            )
        if "range_avg_km" in summary_all_data.columns:
            summary_all_data.loc[mask, "range_km"] = summary_all_data.loc[mask, "range_avg_km"].fillna(
                summary_all_data.loc[mask, "range_km"]
            )

# Configure backend for selections
sim.TECH_SCENARIO = tech_scenario
sim.FUEL_SCENARIO = fuel_scenario
sim.configure_efficiencies(tech_scenario)
concepts = sim.build_concepts_for_scenario(tech_scenario)
concept_cfg_base = deepcopy(concepts[concept_label])
concept_cfg_base["name"] = concept_label
saved_override = get_saved_config(concept_label)
if saved_override:
    concept_cfg_base.update(saved_override)

concept_key = (concept_label, tech_scenario, fuel_scenario)
if st.session_state["concept_key"] != concept_key or st.session_state["concept_cfg"] is None:
    st.session_state["concept_cfg"] = concept_cfg_base
    st.session_state["concept_key"] = concept_key

concept_cfg_state = st.session_state["concept_cfg"]

tab_run, tab_edit, tab_graphs, tab_map = st.tabs(
    ["Run simulation", "Edit parameters", "View graphs & data", "See map from Teuge"]
)


with tab_edit:
    st.markdown("#### Adjust concept parameters")
    if st.button(f"Reset '{concept_label}' to defaults"):
        saved = st.session_state.get("saved_configs") or {}
        if concept_label in saved:
            saved.pop(concept_label)
            st.session_state["saved_configs"] = saved
            persist_configs()
        reset_cfg = deepcopy(concepts[concept_label])
        reset_cfg["name"] = concept_label
        st.session_state["concept_cfg"] = reset_cfg
        st.session_state["concept_key"] = None
        rerunner = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if rerunner:
            rerunner()
    st.session_state["concept_cfg"] = render_parameter_controls(concept_cfg_state)
    save_concept_config(concept_label, st.session_state["concept_cfg"])


with tab_run:
    st.markdown("#### Execute the mission")
    takeoff_mass = compute_takeoff_mass(st.session_state["concept_cfg"])
    mtow = sim.MTOW_KG
    over_mtow = takeoff_mass > mtow
    st.metric("Takeoff mass vs MTOW", f"{takeoff_mass:.0f} / {mtow:.0f} kg", delta=f"{takeoff_mass-mtow:.0f} kg" if over_mtow else f"{mtow-takeoff_mass:.0f} kg margin")
    if over_mtow:
        st.error("Exceeds MTOW. Reduce payload, fuel, or mass modifiers before running the simulation.")
    wind_alt_ft = st.session_state["wind_display"].get("arrow_altitude", 6000)
    wind_profile = sim.weather_config.get("wind_profile", [])
    if not wind_profile:
        sim.fetch_wind_profile(sim.weather_config)
        wind_profile = sim.weather_config.get("wind_profile", [])
    wind_speed, wind_dir = sim.get_wind_at_alt(wind_alt_ft, wind_profile)
    wind_cols = st.columns(2)
    wind_cols[0].metric(f"Wind @ {wind_alt_ft} ft", f"{wind_speed:.0f} kt")
    wind_cols[1].metric("Direction (from)", f"{wind_dir:.0f} deg")
    run_clicked = st.button("Run simulation now", use_container_width=True, disabled=over_mtow)

    if run_clicked:
        with st.spinner("Running simulation..."):
            payload_cfg = deepcopy(st.session_state["concept_cfg"])
            result_dict, phase_log = sim.run_mission(concept_label, payload_cfg)
        if result_dict is None:
            st.error("Simulation failed (check mass/reserve constraints).")
        else:
            summary_df = pd.DataFrame([result_dict])
            phases_df = pd.DataFrame(phase_log)
            st.session_state["summary_df"] = summary_df
            st.session_state["phases_df"] = phases_df
            st.session_state["last_result"] = result_dict
            st.success("Simulation completed.")

    summary_df = st.session_state["summary_df"]
    if summary_df is None or summary_df.empty:
        st.info("Run a simulation to see key metrics.")
    else:
        row = summary_df.iloc[0]
        cache_key = f"{tech_scenario}_{fuel_scenario}"
        if concept_label == "Baseline (Avgas)":
            baseline_result = row.to_dict()
            st.session_state["baseline_refs"][cache_key] = baseline_result
        else:
            baseline_result = get_baseline_reference(tech_scenario, fuel_scenario)
        co2_delta_label = None
        nox_delta_label = None
        if baseline_result:
            base_co2 = baseline_result.get("co2_g_per_km", 0)
            base_nox = baseline_result.get("nox_g_per_km", 0)
            if base_co2:
                delta = ((row["co2_g_per_km"] - base_co2) / base_co2) * 100
                co2_delta_label = f"{delta:+.1f}% vs base"
            if base_nox:
                delta = ((row["nox_g_per_km"] - base_nox) / base_nox) * 100
                nox_delta_label = f"{delta:+.1f}% vs base"
        st.markdown("#### Key metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Range (km)", f"{row.get('range_avg_km', row['range_km']):.0f}")
            st.metric("Cruise time (h)", f"{row['cruise_hours']:.2f}")
        with col2:
            st.metric("Landing fuel (kg)", f"{row['land_fuel_kg']:.1f}")
            st.metric("Fuel reserve (kg)", f"{row['fuel_reserve_kg']:.1f}")
        with col3:
            st.metric("CO2 (g/km)", f"{row['co2_g_per_km']:.0f}", delta=co2_delta_label)
            st.metric("NOx (g/km)", f"{row['nox_g_per_km']:.2f}", delta=nox_delta_label)
            st.metric("Landing SOC", f"{row['land_soc']:.2f}")
        range_best = row.get("range_best_km", row["range_km"])
        range_avg = row.get("range_avg_km", row["range_km"])
        range_worst = row.get("range_worst_km", row["range_km"])
        dir_cols = st.columns(3)
        dir_cols[0].metric("Best range (km)", f"{range_best:.0f}")
        dir_cols[1].metric("Average range (km)", f"{range_avg:.0f}")
        dir_cols[2].metric("Worst range (km)", f"{range_worst:.0f}")


with tab_graphs:
    st.markdown("#### Plots and data")
    if summary_all_data.empty:
        st.info("No simulation data available for this scenario.")
    else:
        st.subheader("Summary across all concepts")
        baseline = summary_all_data[summary_all_data["concept"] == "Baseline (Avgas)"]
        baseline_row = baseline.iloc[0] if not baseline.empty else summary_all_data.iloc[0]
        display_cols = [
            "concept",
            "range_km",
            "range_best_km",
            "range_avg_km",
            "range_worst_km",
            "co2_g_per_km",
            "nox_g_per_km",
            "pm_mg_per_km",
            "land_fuel_kg",
            "fuel_reserve_kg",
            "land_soc",
        ]
        table = summary_all_data[display_cols].copy()
        styles = style_summary_vs_baseline(table, baseline_row)
        styler = table.style.format(
            {
                "range_km": "{:.0f}",
                "range_best_km": "{:.0f}",
                "range_avg_km": "{:.0f}",
                "range_worst_km": "{:.0f}",
                "co2_g_per_km": "{:.0f}",
                "nox_g_per_km": "{:.2f}",
                "pm_mg_per_km": "{:.1f}",
                "land_fuel_kg": "{:.1f}",
                "fuel_reserve_kg": "{:.1f}",
                "land_soc": "{:.2f}",
            }
        ).set_properties(**{"font-weight": "bold"})\
         .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])\
         .apply(lambda _: styles, axis=None)
        st.dataframe(styler, use_container_width=True)
        policy_name = getattr(sim, "RESERVE_POLICY", {}).get("name", "EASA VFR day (30 min)")
        st.info(f"Reserve policy: {policy_name} (30 min at normal cruise)")

    if summary_df is None or summary_df.empty or phases_df is None or phases_df.empty:
        st.info("Run a simulation to view detailed mission profiles.")
    else:
        df = phases_df.copy()
        df["cum_dist_km"] = df["dist_km"].cumsum()
        total_fuel_kg = df.get("fuel_kg", pd.Series(dtype=float)).sum()
        row = summary_df.iloc[0]
        fuel_per_100km = (total_fuel_kg / row["range_km"] * 100.0) if row["range_km"] > 0 else float("nan")

        st.subheader("Reserve and efficiency overview")
        eff_cols = st.columns(3)
        eff_cols[0].metric("Fuel per 100 km (kg)", f"{fuel_per_100km:.2f}" if row["range_km"] > 0 else "N/A")
        eff_cols[1].metric("Total fuel burned (kg)", f"{total_fuel_kg:.2f}")
        reserve_margin = row["land_fuel_kg"] - row["fuel_reserve_kg"]
        eff_cols[2].metric("Fuel reserve margin (kg)", f"{reserve_margin:.1f}")

        st.subheader("SOC profile over mission")
        fig_soc = px.line(
            df,
            x="cum_dist_km",
            y="soc",
            title="SOC vs distance",
        )
        st.plotly_chart(fig_soc, use_container_width=True)

        st.subheader("Average GT vs Battery power per phase")
        power_df = (
            df.groupby("phase")[["gt_power_kw", "batt_power_kw"]]
            .mean()
            .reset_index()
        )
        fig_power = px.bar(
            power_df,
            x="phase",
            y=["gt_power_kw", "batt_power_kw"],
            barmode="stack",
            labels={"value": "Power (kW)", "phase": "Phase"},
            title="Average power per phase (GT + Battery)",
        )
        st.plotly_chart(fig_power, use_container_width=True)

        if "fuel_kg" in df.columns:
            st.subheader("Fuel burn per phase")
            phase_fuel_df = df.groupby("phase")["fuel_kg"].sum().reset_index()
            fig_phase_fuel = px.bar(
                phase_fuel_df,
                x="phase",
                y="fuel_kg",
                labels={"fuel_kg": "Fuel (kg)", "phase": "Phase"},
                title="Fuel burn per mission phase",
            )
            st.plotly_chart(fig_phase_fuel, use_container_width=True)

        st.subheader("Emissions breakdown")
        emissions_rows = [
            {"Pollutant": "CO2", "value": row["co2_g_per_km"], "unit": "g/km"},
            {"Pollutant": "NOx", "value": row["nox_g_per_km"], "unit": "g/km"},
            {"Pollutant": "PM", "value": row["pm_mg_per_km"] / 1000.0, "unit": "g/km"},
        ]
        emissions_df = pd.DataFrame(emissions_rows)
        fig_emissions = px.bar(
            emissions_df,
            x="Pollutant",
            y="value",
            color="Pollutant",
            labels={"value": "Intensity (g/km)"},
            title="Emissions intensity per km",
        )
        st.plotly_chart(fig_emissions, use_container_width=True)


with tab_map:
    st.markdown("#### Range map from Teuge (EHTE)")
    summary_all = summary_all_data.copy()
    if summary_all.empty:
        st.info("No simulation data available for this scenario.")
    else:
        polygon_data = []
        legend_entries = []
        for _, row in summary_all.iterrows():
            concept = row["concept"]
            range_km = float(row.get("range_km", 0))
            color_rgb = CONCEPT_COLORS.get(concept, [80, 80, 80])
            samples = row.get("range_samples")
            if isinstance(samples, list) and samples:
                polygon_coords = make_directional_polygon(TEUGE_CENTER[0], TEUGE_CENTER[1], samples)
            else:
                polygon_coords = make_circle_polygon(
                    TEUGE_CENTER[0], TEUGE_CENTER[1], max(range_km, 0)
                )
            polygon_data.append(
                {
                    "polygon": polygon_coords,
                    "name": f"{concept} ({range_km:.0f} km)",
                    "range_km": range_km,
                    "color": color_rgb + [50],
                    "line_color": color_rgb + [200],
                }
            )
            legend_entries.append(
                (
                    concept,
                    range_km,
                    color_rgb,
                    row.get("range_best_km", range_km),
                    row.get("range_avg_km", range_km),
                    row.get("range_worst_km", range_km),
                )
            )

        polygon_layer = pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="polygon",
            get_fill_color="color",
            get_line_color="line_color",
            line_width_min_pixels=1,
            pickable=True,
        )
        wind_display = st.session_state.get("wind_display", {})
        arrow_layers = []
        if wind_display.get("show_arrows", False):
            arrow_data = build_wind_arrows(
                sim.weather_config,
                wind_display.get("arrow_altitude", 6000),
                wind_display.get("arrow_spacing", 3),
            )
            if arrow_data:
                arrow_layers.append(
                    pdk.Layer(
                        "LineLayer",
                        data=arrow_data,
                        get_source_position="source",
                        get_target_position="target",
                        get_width=2,
                        get_color=[50, 50, 50, 200],
                    )
                )
        base_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": TEUGE_CENTER, "name": "Teuge Airport"}],
            get_position="position",
            get_radius=10000,
            get_fill_color="[0, 0, 0, 255]",
            pickable=True,
        )
        view_state = pdk.ViewState(
            longitude=TEUGE_CENTER[0],
            latitude=TEUGE_CENTER[1],
            zoom=4,
        )
        deck = pdk.Deck(
            layers=[polygon_layer, base_layer] + arrow_layers,
            initial_view_state=view_state,
            tooltip={"text": "{name}"},
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("**Legend**")
        for concept, rng, color, best, avg, worst in legend_entries:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            st.markdown(
                f"<span style='display:inline-block;width:16px;height:16px;background:{hex_color};border:1px solid #222;margin-right:8px;'></span>"
                f"{concept}: best {best:.0f} km / avg {avg:.0f} km / worst {worst:.0f} km",
                unsafe_allow_html=True,
            )






