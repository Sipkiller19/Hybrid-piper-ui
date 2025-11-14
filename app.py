import math

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

import hybrid_aircraft_sim as sim

st.set_page_config(page_title="Hybrid Aircraft Explorer", layout="wide")
st.title("Hybrid Aircraft Concept Explorer")

TEUGE_CENTER = [6.05, 52.24]  # [lon, lat]


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


def edit_concept_parameters(concept_name, concept_cfg):
    st.sidebar.subheader("Propulsion parameters")

    concept_cfg["p_gt_hp"] = st.sidebar.number_input(
        "GT power (hp)",
        min_value=0.0,
        max_value=1000.0,
        value=float(concept_cfg.get("p_gt_hp", 0.0)),
    )

    concept_cfg["wt_gt"] = st.sidebar.number_input(
        "GT mass (kg)",
        min_value=0.0,
        max_value=500.0,
        value=float(concept_cfg.get("wt_gt", 0.0)),
    )

    concept_cfg["gt_sfc_design"] = st.sidebar.number_input(
        "GT SFC (kg/hp·hr)",
        min_value=0.1,
        max_value=1.0,
        value=float(concept_cfg.get("gt_sfc_design", 0.3)),
        step=0.01,
    )

    concept_cfg["p_em_hp"] = st.sidebar.number_input(
        "Motor power (hp)",
        min_value=0.0,
        max_value=300.0,
        value=float(concept_cfg.get("p_em_hp", 0.0)),
    )

    concept_cfg["wt_em"] = st.sidebar.number_input(
        "Motor mass (kg)",
        min_value=0.0,
        max_value=300.0,
        value=float(concept_cfg.get("wt_em", 0.0)),
    )

    concept_cfg["batt_kwh"] = st.sidebar.number_input(
        "Battery capacity (kWh)",
        min_value=0.0,
        max_value=500.0,
        value=float(concept_cfg.get("batt_kwh", 0.0)),
    )

    st.sidebar.subheader("Airframe modifiers")

    concept_cfg["fuel_vol_L"] = st.sidebar.number_input(
        "Fuel volume (L)",
        min_value=0.0,
        max_value=400.0,
        value=float(concept_cfg.get("fuel_vol_L", 0.0)),
    )

    concept_cfg["cd0_adder"] = st.sidebar.number_input(
        "Extra drag (cd0_adder)",
        min_value=0.0,
        max_value=0.05,
        value=float(concept_cfg.get("cd0_adder", 0.0)),
        step=0.0005,
    )

    concept_cfg["base_mass_adder"] = st.sidebar.number_input(
        "Base mass adder (kg)",
        min_value=-200.0,
        max_value=200.0,
        value=float(concept_cfg.get("base_mass_adder", 0.0)),
        step=5.0,
    )

    return concept_cfg


# Sidebar controls
st.sidebar.header("Configuration")

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

run_button = st.sidebar.button("Run simulation")
compare_button = st.sidebar.button("Compare all concepts")

# Configure backend scenarios and build concepts once per selection
sim.TECH_SCENARIO = tech_scenario
sim.FUEL_SCENARIO = fuel_scenario
sim.configure_efficiencies(tech_scenario)
concepts = sim.build_concepts_for_scenario(tech_scenario)
concept_cfg = concepts.get(concept_label)

if concept_cfg is None:
    st.error("Concept not found in backend configuration.")
else:
    concept_cfg = edit_concept_parameters(concept_label, concept_cfg)
    concept_cfg["name"] = concept_label

    if run_button:
        st.info("Running simulation...")
        result_dict, phase_log = sim.run_mission(concept_label, concept_cfg)

        if result_dict is None:
            st.error("Simulation failed (check logs or parameters).")
        else:
            summary_df = pd.DataFrame([result_dict])
            phases_df = pd.DataFrame(phase_log)
            row = summary_df.iloc[0]

            st.subheader("Key metrics")
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

            st.subheader("Summary (full)")
            st.dataframe(summary_df)

            st.subheader("Phase log")
            st.dataframe(phases_df)

            if not phases_df.empty:
                phases_df = phases_df.copy()
                phases_df["cum_dist_km"] = phases_df["dist_km"].cumsum()

                st.subheader("SOC profile over mission")
                fig_soc = px.line(
                    phases_df,
                    x="cum_dist_km",
                    y="soc",
                    title="SOC vs distance",
                )
                st.plotly_chart(fig_soc, use_container_width=True)

                st.subheader("GT vs Battery power per phase")
                power_df = (
                    phases_df.groupby("phase")[["gt_power_kw", "batt_power_kw"]]
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

                st.subheader("Range map from Teuge (EHTE)")
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
                st.pydeck_chart(deck)

if compare_button:
    with st.spinner("Running all concepts across scenarios..."):
        summary_all, _ = sim.run_all_concepts()

    if summary_all is None or summary_all.empty:
        st.warning("No comparison data available.")
    else:
        st.subheader("Compare all concepts (range vs CO₂)")
        fig_compare = px.scatter(
            summary_all,
            x="range_km",
            y="co2_g_per_km",
            color="concept",
            symbol="fuel_scenario",
            facet_col="tech_scenario",
            hover_data=["fuel_scenario"],
            title="Range vs CO₂ per km",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("All concept summary")
        st.dataframe(summary_all.sort_values(["tech_scenario", "concept", "fuel_scenario"]))
