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


def init_state():
    defaults = {
        "summary_df": None,
        "phases_df": None,
        "last_result": None,
        "concept_cfg": None,
        "concept_key": None,
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
        "GT SFC (kg/hp·hr)",
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
        st.markdown("#### Key metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Range (km)", f"{row['range_km']:.0f}")
            st.metric("Cruise time (h)", f"{row['cruise_hours']:.2f}")
        with col2:
            st.metric("Landing fuel (kg)", f"{row['land_fuel_kg']:.1f}")
            st.metric("Fuel reserve (kg)", f"{row['fuel_reserve_kg']:.1f}")
        with col3:
            st.metric("CO₂ (g/km)", f"{row['co2_g_per_km']:.0f}")
            st.metric("NOx (g/km)", f"{row['nox_g_per_km']:.2f}")
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


with tab_map:
    st.markdown("#### Range map from Teuge (EHTE)")
    summary_df = st.session_state["summary_df"]
    if summary_df is None or summary_df.empty:
        st.info("Run a simulation first (tab: 'Run simulation').")
    else:
        row = summary_df.iloc[0]
        range_km = float(row["range_km"])
        circle_polygon = make_circle_polygon(
            TEUGE_CENTER[0], TEUGE_CENTER[1], range_km
        )
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": circle_polygon, "name": f"Range {range_km:.0f} km"}],
            get_polygon="polygon",
            get_fill_color="[255, 0, 0, 40]",
            get_line_color="[255, 0, 0, 200]",
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
