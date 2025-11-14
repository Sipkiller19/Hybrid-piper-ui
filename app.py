import math
from copy import deepcopy

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import hybrid_aircraft_sim as sim


st.set_page_config(page_title="Hybrid Piper Sim", layout="wide")
st.title("Hybrid Piper Sim")

TEUGE_CENTER = [6.05, 52.24]  # [lon, lat]
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


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
def run_concepts_for_map(tech_scenario, fuel_scenario):
    """Run all default concepts for the selected scenario/fuel."""
    prev_tech = sim.TECH_SCENARIO
    prev_fuel = sim.FUEL_SCENARIO
    prev_eta = (sim.ETA_MOTOR, sim.ETA_GEN, sim.ETA_BATT)

    sim.TECH_SCENARIO = tech_scenario
    sim.configure_efficiencies(tech_scenario)
    sim.FUEL_SCENARIO = fuel_scenario

    concepts = sim.build_concepts_for_scenario(tech_scenario)
    rows = []
    for name, cfg in concepts.items():
        cfg = deepcopy(cfg)
        cfg["name"] = name
        result, _ = sim.run_mission(name, cfg)
        if result is not None:
            rows.append(result)

    sim.TECH_SCENARIO = prev_tech
    sim.configure_efficiencies(prev_tech)
    sim.FUEL_SCENARIO = prev_fuel
    sim.ETA_MOTOR, sim.ETA_GEN, sim.ETA_BATT = prev_eta

    return pd.DataFrame(rows)


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

# Configure backend for selections
sim.TECH_SCENARIO = tech_scenario
sim.FUEL_SCENARIO = fuel_scenario
sim.configure_efficiencies(tech_scenario)
concepts = sim.build_concepts_for_scenario(tech_scenario)
concept_cfg_base = deepcopy(concepts[concept_label])
concept_cfg_base["name"] = concept_label

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
    if concept_label == "Baseline (Avgas)":
        st.info("Baseline aircraft parameters are fixed and cannot be edited.")
        st.session_state["concept_cfg"] = concept_cfg_base
    else:
        st.session_state["concept_cfg"] = render_parameter_controls(concept_cfg_state)


with tab_run:
    st.markdown("#### Execute the mission")
    run_clicked = st.button("Run simulation now", use_container_width=True)

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
            st.metric("Range (km)", f"{row['range_km']:.0f}")
            st.metric("Cruise time (h)", f"{row['cruise_hours']:.2f}")
        with col2:
            st.metric("Landing fuel (kg)", f"{row['land_fuel_kg']:.1f}")
            st.metric("Fuel reserve (kg)", f"{row['fuel_reserve_kg']:.1f}")
        with col3:
            st.metric("COâ‚‚ (g/km)", f"{row['co2_g_per_km']:.0f}", delta=co2_delta_label)
            st.metric("NOx (g/km)", f"{row['nox_g_per_km']:.2f}", delta=nox_delta_label)
            st.metric("Landing SOC", f"{row['land_soc']:.2f}")


with tab_graphs:
    st.markdown("#### Plots and data")
    summary_df = st.session_state["summary_df"]
    phases_df = st.session_state["phases_df"]
    if summary_df is None or phases_df is None or summary_df.empty or phases_df.empty:
        st.info("Run a simulation first (tab: 'Run simulation').")
    else:
        st.subheader("Summary data")
        st.dataframe(summary_df, use_container_width=True)

        df = phases_df.copy()
        df["cum_dist_km"] = df["dist_km"].cumsum()

        total_fuel_kg = df.get("fuel_kg", pd.Series(dtype=float)).sum()
        row = summary_df.iloc[0]
        fuel_per_100km = (total_fuel_kg / row["range_km"] * 100.0) if row["range_km"] > 0 else float("nan")
        st.subheader("Efficiency overview")
        eff_cols = st.columns(2)
        eff_cols[0].metric("Fuel per 100 km (kg)", f"{fuel_per_100km:.2f}" if row["range_km"] > 0 else "N/A")
        eff_cols[1].metric("Total fuel burned (kg)", f"{total_fuel_kg:.2f}")

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
            {"Pollutant": "CO₂", "value": row["co2_g_per_km"], "unit": "g/km"},
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
    summary_all = run_concepts_for_map(tech_scenario, fuel_scenario)
    summary_df = st.session_state["summary_df"]
    if summary_df is not None and not summary_df.empty:
        current_row = summary_df.iloc[0].to_dict()
        mask = summary_all["concept"] == current_row["concept"]
        if any(mask):
            for key, val in current_row.items():
                summary_all.loc[mask, key] = val
        else:
            summary_all = pd.concat([summary_all, summary_df], ignore_index=True)
    if summary_all.empty:
        st.info("No simulation data available for this scenario.")
    else:
        polygon_data = []
        legend_entries = []
        for _, row in summary_all.iterrows():
            concept = row["concept"]
            range_km = float(row.get("range_km", 0))
            color_rgb = CONCEPT_COLORS.get(concept, [80, 80, 80])
            circle_polygon = make_circle_polygon(
                TEUGE_CENTER[0], TEUGE_CENTER[1], max(range_km, 0)
            )
            polygon_data.append(
                {
                    "polygon": circle_polygon,
                    "name": f"{concept} ({range_km:.0f} km)",
                    "range_km": range_km,
                    "color": color_rgb + [50],
                    "line_color": color_rgb + [200],
                }
            )
            legend_entries.append((concept, range_km, color_rgb))

        polygon_layer = pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="polygon",
            get_fill_color="color",
            get_line_color="line_color",
            line_width_min_pixels=1,
            pickable=True,
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
            layers=[polygon_layer, base_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}"},
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("**Legend**")
        for concept, rng, color in legend_entries:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            st.markdown(
                f"<span style='display:inline-block;width:16px;height:16px;background:{hex_color};border:1px solid #222;margin-right:8px;'></span>"
                f"{concept}: {rng:.0f} km",
                unsafe_allow_html=True,
            )

