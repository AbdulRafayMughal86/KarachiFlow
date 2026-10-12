"""
KarachiFlow — Decoupled Geospatial Traffic Visualization Dashboard
===================================================================
Interactive Streamlit and PyDeck dashboard for microscopic traffic simulation
replay, congestion shockwave analysis, and parallel computing scaling benchmarks.
"""

import os
import glob
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# Import modular components
import visualizer
import metrics_plotter

# Page Configuration
st.set_page_config(
    page_title="KarachiFlow — Traffic Simulation Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme & Glassmorphism Styling
st.markdown(
    """
    <style>
    /* Dark background styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric card container */
    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    /* Status badge pill */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-online {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* Header styling */
    .dashboard-header {
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dashboard-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Legend pill container */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 10px 14px;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        margin-bottom: 14px;
        font-size: 0.85rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Karachi road network topology...")
def get_cached_network() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and caches the static road network graph geometry."""
    bin_path = "data/karachi_network.bin"
    metadata_path = "data/karachi_network_metadata.json"

    # Fallback to synthetic if full metropolitan binary not yet generated
    if not os.path.exists(bin_path):
        bin_path = "data/karachi_synthetic.bin"
        metadata_path = "data/karachi_synthetic_metadata.json"

    return visualizer.load_network_geometry(bin_path, metadata_path)


def discover_available_snapshots(snapshot_dir: str = "data/snapshots") -> List[Tuple[int, str]]:
    """Discovers all available snapshot CSVs and sorts them chronologically by time step."""
    if not os.path.exists(snapshot_dir):
        return []

    pattern = os.path.join(snapshot_dir, "snapshot_t*.csv")
    files = glob.glob(pattern)
    snapshots = []

    for fpath in files:
        basename = os.path.basename(fpath)
        # Extract timestep integer: snapshot_t{step}.csv
        name_no_ext = os.path.splitext(basename)[0]
        try:
            step_str = name_no_ext.replace("snapshot_t", "")
            step_int = int(step_str)
            snapshots.append((step_int, fpath))
        except ValueError:
            continue

    snapshots.sort(key=lambda x: x[0])
    return snapshots


@st.cache_data(show_spinner=False)
def get_cached_snapshot(snapshot_path: str) -> pd.DataFrame:
    """Loads and caches individual simulation snapshot telemetry files."""
    return visualizer.load_snapshot(snapshot_path)


def main():
    header_placeholder = st.empty()

    # Load Road Network Geometry
    try:
        network_df, metadata = get_cached_network()
    except Exception as e:
        st.error(f"Failed to load road network geometry: {e}")
        return

    # Discover Snapshots
    snapshots = discover_available_snapshots()
    if not snapshots:
        st.warning(
            "No snapshot telemetry files detected in `data/snapshots/`. "
            "Please run a simulation run first to produce `snapshot_t*.csv` records."
        )
        return

    step_numbers = [s[0] for s in snapshots]
    step_to_path = {s[0]: s[1] for s in snapshots}

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Simulation Controls")

        # Operational Mode Selector
        sim_mode = st.radio(
            "Operation Mode",
            ["Historical Replay", "Live Simulation Monitor"],
            index=0,
            help="Choose between scrubbing through completed snapshots or monitoring ongoing simulation runs.",
        )

        # Dedicated sidebar placeholder for real-time City KPIs
        sidebar_kpi_placeholder = st.container()

        st.markdown("---")
        if sim_mode == "Historical Replay":
            st.markdown("### ⏱️ Time Scrubber")

            # Session state for playback control
            if "step_slider" not in st.session_state:
                st.session_state.step_slider = step_numbers[-1]
            if "is_playing" not in st.session_state:
                st.session_state.is_playing = False

            # Step slider
            selected_step = st.select_slider(
                "Simulation Step ($t$ seconds)",
                options=step_numbers,
                key="step_slider",
                help="Select the exact second of simulated traffic flow.",
            )
            current_idx = step_numbers.index(selected_step)

            # Step jump buttons
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            with col_btn1:
                if st.button("⏮ First", width="stretch"):
                    st.session_state.step_slider = step_numbers[0]
                    st.rerun()
            with col_btn2:
                if st.button("◀ Prev", width="stretch"):
                    st.session_state.step_slider = step_numbers[max(0, current_idx - 1)]
                    st.rerun()
            with col_btn3:
                if st.button("Next ▶", width="stretch"):
                    st.session_state.step_slider = step_numbers[min(len(step_numbers) - 1, current_idx + 1)]
                    st.rerun()
            with col_btn4:
                if st.button("Latest ⏭", width="stretch"):
                    st.session_state.step_slider = step_numbers[-1]
                    st.rerun()

            # Replay Auto-Play Toggle
            auto_play = st.toggle("Auto-Play Timeline", value=st.session_state.is_playing)
            play_speed = st.slider("Playback Speed (s/step)", 0.2, 2.0, 0.8, 0.2)
            st.session_state.is_playing = auto_play
            poll_interval = 2.0
        else:
            st.info("Tracking `data/snapshots/` for incoming simulation frames.")
            selected_step = step_numbers[-1]
            current_idx = len(step_numbers) - 1
            auto_play = False
            play_speed = 0.8
            auto_poll = st.toggle("Auto-Refresh Telemetry", value=False)
            poll_interval = st.slider("Live Poll Interval (s)", 1.0, 5.0, 2.0, 0.5, key="poll_slider", disabled=not auto_poll)
            if st.button("🔄 Refresh Telemetry Now", width="stretch"):
                st.rerun()
            st.caption(f"Currently tracking newest step: **t = {selected_step} s**")

        st.markdown("---")
        st.markdown("### 🗺️ Map & Visual Viewport")

        viewport_presets = visualizer.get_viewport_presets(metadata)
        preset_name = st.selectbox(
            "Camera Preset",
            list(viewport_presets.keys()),
            index=0,
            help="Center camera on key Karachi arteries and corridors.",
        )
        selected_view_state = viewport_presets[preset_name]

        display_mode = st.selectbox(
            "Network Rendering Detail",
            [
                "Active Traffic Only",
                "Congested Corridors Only",
                "Severe Bottlenecks Only",
                "Major Arteries + Active Traffic",
                "Complete Road Network",
            ],
            index=0,
            help="Optimizes rendering speed by filtering empty road segments.",
        )

        col_lay1, col_lay2 = st.columns(2)
        with col_lay1:
            show_zones = st.checkbox("Show OD Zones", value=True)
        with col_lay2:
            show_heatmap = st.checkbox("Bottleneck Heatmap", value=False)

        pitch_deg = st.slider("Camera Pitch", 0, 60, int(selected_view_state.pitch), 5)
        bearing_deg = st.slider("Camera Bearing", -180, 180, int(selected_view_state.bearing), 5)
        selected_view_state.pitch = pitch_deg
        selected_view_state.bearing = bearing_deg

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
                <b>KarachiFlow Engine</b><br>
                Model: 7.5m Nagel–Schreckenberg CA<br>
                OSM Road Graph: 74,914 segments<br>
                Parallel Frameworks: OpenMP, MS-MPI
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Dynamic Header Banner Rendering
    status_label = (
        f"● Live Monitoring (t = {selected_step}s)"
        if sim_mode == "Live Simulation Monitor"
        else f"● Telemetry Replay (t = {selected_step}s)"
    )
    header_placeholder.markdown(
        f"""
        <div class="dashboard-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="dashboard-title">KarachiFlow — High-Performance Traffic Simulation Dashboard</h1>
                    <div class="dashboard-subtitle">
                        Microscopic Nagel–Schreckenberg Cellular Automaton on OpenStreetMap Network with Multi-Core & Distributed PDC Acceleration
                    </div>
                </div>
                <div>
                    <span class="status-badge badge-online">{status_label}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load and merge active snapshot telemetry
    active_snapshot_path = step_to_path[selected_step]
    snapshot_df = get_cached_snapshot(active_snapshot_path)
    merged_df = visualizer.merge_telemetry(network_df, snapshot_df)

    # Compute Executive Metrics
    active_cars = int(merged_df["vehicle_count"].sum())
    active_mask = merged_df["vehicle_count"] > 0
    active_segments = int(active_mask.sum())

    if active_segments > 0:
        avg_speed_cells = float(merged_df.loc[active_mask, "avg_speed"].mean())
        avg_speed_kmh = avg_speed_cells * visualizer.KMH_PER_CELL_PER_SEC
    else:
        avg_speed_cells = 0.0
        avg_speed_kmh = 0.0

    bottleneck_count = int((merged_df["congestion_status"] == "Severe Bottleneck (Queue)").sum())
    moderate_count = int((merged_df["congestion_status"] == "Moderate Flow (Slowing)").sum())
    free_flow_count = int((merged_df["congestion_status"] == "Free Flow (Normal)").sum())
    network_occupancy = (active_cars / metadata["counts"]["total_cells"]) * 100.0

    # Calculate completed trips based on vehicle demand conservation (injected == active + completed)
    total_target_vehicles = 5000
    max_step = max(step_numbers[-1], 1)
    cumulative_injected = min(total_target_vehicles, int(round(total_target_vehicles * (selected_step / max_step))))
    completed_trips = max(0, cumulative_injected - active_cars)

    # Populate Sidebar KPI Cards
    with sidebar_kpi_placeholder:
        st.markdown("### 📊 City Telemetry KPIs")
        st.metric("🚗 Active Vehicles", f"{active_cars:,}")
        st.metric("🏁 Completed Trips", f"{completed_trips:,}")
        st.metric("⚡ Average City Speed", f"{avg_speed_kmh:.1f} km/h")

    # Executive KPI Metric Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("🚗 Active Vehicles", f"{active_cars:,}")
    with kpi_col2:
        st.metric("🏁 Completed Trips", f"{completed_trips:,}")
    with kpi_col3:
        st.metric("⚡ Average Speed", f"{avg_speed_kmh:.1f} km/h", f"{avg_speed_cells:.2f} cells/s")
    with kpi_col4:
        st.metric("🔴 Bottleneck Queues", f"{bottleneck_count}", f"{(bottleneck_count/max(active_segments,1))*100:.1f}%")
    with kpi_col5:
        st.metric("🟢 Free Flow Roads", f"{free_flow_count}")
    with kpi_col6:
        st.metric("🚀 Peak Speedup", "8.29×", "Hybrid MPI+OpenMP")

    # Main Tabbed Layout
    tab_map, tab_benchmarks, tab_architecture = st.tabs([
        "🗺️ Geospatial Congestion Flow",
        "📊 Parallel Computing Benchmarks",
        "🔬 Cellular Automaton Model & Graph",
    ])

    # TAB 1: GEOSPATIAL MAP & TRAFFIC FLOW
    with tab_map:
        # Visual Congestion Legend
        st.markdown(
            """
            <div class="legend-container">
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #00ff88; box-shadow: 0 0 8px #00ff88;"></span>
                    <span><b>Free Flow:</b> High speed (v ≥ 0.75 vmax, low density)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ffc300; box-shadow: 0 0 8px #ffc300;"></span>
                    <span><b>Moderate Flow:</b> Traffic slowing down (0.4 vmax ≤ v &lt; 0.75 vmax)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ff1e3c; box-shadow: 0 0 8px #ff1e3c;"></span>
                    <span><b>Severe Bottleneck:</b> Queue buildup (v &lt; 0.4 vmax or density ≥ 40%)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #00d2ff;"></span>
                    <span><b>Origin Zones:</b> Residential Hubs</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ff9600;"></span>
                    <span><b>Destination Zones:</b> Commercial &amp; Employment Centers</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PyDeck Map Deck
        deck = visualizer.build_traffic_deck(
            merged_df=merged_df,
            metadata=metadata,
            view_state=selected_view_state,
            display_mode=display_mode,
            show_zones=show_zones,
            show_heatmap=show_heatmap,
        )
        st.pydeck_chart(deck, width="stretch")

        # Secondary Analytics Grid Below Map
        col_flow1, col_flow2 = st.columns([1, 1])

        with col_flow1:
            st.markdown("#### 🚦 Citywide Congestion Distribution")
            if active_segments > 0:
                p_free = (free_flow_count / active_segments) * 100.0
                p_mod = (moderate_count / active_segments) * 100.0
                p_bot = (bottleneck_count / active_segments) * 100.0

                dist_df = pd.DataFrame({
                    "Congestion State": ["Free Flow (🟢)", "Moderate Slowing (🟡)", "Severe Bottleneck (🔴)"],
                    "Segment Count": [free_flow_count, moderate_count, bottleneck_count],
                    "Percentage (%)": [f"{p_free:.1f}%", f"{p_mod:.1f}%", f"{p_bot:.1f}%"],
                })
                st.dataframe(dist_df, width="stretch", hide_index=True)

                st.progress(
                    int(p_free),
                    text=f"Free-Flow Arteries ({p_free:.1f}% of active traffic network)",
                )
            else:
                st.info("No vehicles currently circulating on the network at t = 0.")

        with col_flow2:
            st.markdown("#### 🛑 Top Critical Congestion Bottlenecks")
            critical_df = (
                merged_df[merged_df["vehicle_count"] > 0]
                .sort_values(by=["density_ratio", "vehicle_count"], ascending=[False, False])
                .head(5)[
                    [
                        "segment_id",
                        "lanes",
                        "vehicle_count",
                        "avg_speed_kmh",
                        "density_pct",
                        "congestion_status",
                    ]
                ]
                .rename(
                    columns={
                        "segment_id": "Segment ID",
                        "lanes": "Lanes",
                        "vehicle_count": "Cars",
                        "avg_speed_kmh": "Speed (km/h)",
                        "density_pct": "Density (%)",
                        "congestion_status": "Status",
                    }
                )
            )
            st.dataframe(critical_df, width="stretch", hide_index=True)

    # TAB 2: PARALLEL COMPUTING BENCHMARKS (PDC)
    with tab_benchmarks:
        st.markdown("### ⚡ Parallel Computing Performance & Scaling Analysis")
        st.markdown(
            """
            Empirical runtime performance and scalability measurements evaluated on
            the discrete Karachi road network (5,000 vehicles, 60s dev simulation duration).
            Comparing **Sequential Baseline**, **OpenMP Shared-Memory**, **Distributed MS-MPI**, and **Hybrid MPI + OpenMP**.
            """
        )

        benchmarks = metrics_plotter.get_empirical_benchmarks()

        # Benchmark Sub-Controls
        col_scale_ctl1, col_scale_ctl2 = st.columns([1, 2])
        with col_scale_ctl1:
            baseline_mode = st.radio(
                "Speedup Baseline Reference",
                ["Self-Relative (T1/Tp)", "Speedup vs Sequential Baseline"],
                horizontal=True,
                help="Switch between relative speedup within each paradigm or absolute speedup over single-core C++ reference.",
            )

        # Plotly Charts: Row 1
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_speedup = metrics_plotter.plot_speedup_curves(benchmarks, baseline_mode)
            st.plotly_chart(fig_speedup, width="stretch")

        with col_chart2:
            fig_efficiency = metrics_plotter.plot_efficiency_curves(benchmarks)
            st.plotly_chart(fig_efficiency, width="stretch")

        # Plotly Charts: Row 2
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            fig_runtime = metrics_plotter.plot_runtime_comparison(benchmarks)
            st.plotly_chart(fig_runtime, width="stretch")

        with col_chart4:
            fig_imbalance = metrics_plotter.plot_load_imbalance(benchmarks)
            st.plotly_chart(fig_imbalance, width="stretch")

        # Plotly Charts: Row 3
        col_chart5, col_chart6 = st.columns([1, 1])
        with col_chart5:
            fig_comm = metrics_plotter.plot_mpi_communication_overhead(benchmarks)
            st.plotly_chart(fig_comm, width="stretch")

        with col_chart6:
            st.markdown("#### 🌐 Distributed Scaling & Halo Exchange Physics")
            st.markdown(
                """
                - **Non-Blocking Halo Overlap:** Boundary vehicle migrations overlap with interior road segment updates via `MPI_Isend` and `MPI_Irecv`.
                - **Negligible Overhead ($R_{\\text{comm}} = 3.3\\%$):** Communication is 6.97 ms vs 204.33 ms compute, well below the 10% target threshold.
                - **Uniform Load Balance ($\\lambda = 1.005$):** Uniform spatial domain decomposition keeps worker imbalance at just 0.5% ($t_{\\max} / t_{\\text{avg}}$).
                - **Peak Hybrid Throughput:** Hybrid MPI + OpenMP unlocks **170.85 ms** (**8.29× speedup** over single-thread sequential reference).
                """
            )

        # Scaling Comparison Table
        st.markdown("#### 📋 Comprehensive Architectural Benchmark Summary")
        summary_table = metrics_plotter.create_scaling_summary_table(benchmarks)
        st.dataframe(summary_table, width="stretch", hide_index=True)

        # Key Takeaways Box
        st.info(
            """
            **PDC Engineering Insights:**
            - **OpenMP Scaling:** Scales from 242.53 ms down to 187.22 ms (1.30× self speedup, 7.57× over sequential). Loop iterations partition uniformly across road segments via static scheduling.
            - **Distributed MPI Scaling:** Non-blocking halo exchanges (`MPI_Isend` / `MPI_Irecv`) achieve a lightweight communication overhead of **R_comm = 3.3%** and load imbalance **λ = 1.005**.
            - **Hybrid MPI + OpenMP Synergy:** Combines inter-node domain decomposition with intra-node multi-threading, yielding the peak throughput of **170.85 ms** (**8.29× speedup** over single-thread sequential reference).
            """
        )

    # TAB 3: CELLULAR AUTOMATON & NETWORK TOPOLOGY
    with tab_architecture:
        st.markdown("### 🔬 Nagel–Schreckenberg Cellular Automaton Physics & Topology")

        col_model1, col_model2 = st.columns([1, 1])

        with col_model1:
            st.markdown(
                r"""
                #### 1. Discrete 7.5m Cell Physics Model
                In KarachiFlow, road segments are discretized into uniform **7.5-meter cells** ($\Delta t = 1.0$ s):
                - **Standard Car Length:** $4.5$ m
                - **Bumper-to-Bumper Gap:** $3.0$ m
                - **Total Space per Stationary Vehicle:** $7.5$ m

                **The 4-Step Velocity & Position Update Loop:**
                1. **Acceleration:** $v \leftarrow \min(v + 1, v_{\max})$
                2. **Safe Braking:** $v \leftarrow \min(v, d_{\text{gap}})$
                3. **Random Slowing:** With probability $p_{\text{slow}}$, $v \leftarrow \max(v - 1, 0)$
                4. **Position Advance:** $x \leftarrow x + v$
                """
            )

            # Speed Mapping Table
            st.markdown("#### 2. Discrete Speed to Real-World Velocity Mapping")
            speed_map_df = pd.DataFrame({
                "Discrete Speed (v)": ["0 cells/s", "1 cell/s", "2 cells/s", "3 cells/s", "4 cells/s"],
                "Velocity (m/s)": ["0.0 m/s", "7.5 m/s", "15.0 m/s", "22.5 m/s", "30.0 m/s"],
                "Real Velocity (km/h)": ["0.0 km/h", "27.0 km/h", "54.0 km/h", "81.0 km/h", "108.0 km/h"],
                "Representative Karachi Context": [
                    "Gridlocked / Bottleneck Queue",
                    "Dense Downtown Crawl (Saddar, Bolton Market)",
                    "Arterial Limit (M.A. Jinnah, University Road)",
                    "Cruising Expressway (Shahrah-e-Faisal)",
                    "Open Highway / Motorway Upper Bound",
                ],
            })
            st.dataframe(speed_map_df, width="stretch", hide_index=True)

        with col_model2:
            st.markdown("#### 3. Karachi Road Graph Ingestion Metadata")
            counts = metadata.get("counts", {})
            bbox = metadata.get("bounding_box", {})

            meta_df = pd.DataFrame({
                "Graph Metric": [
                    "Total Intersections (Nodes)",
                    "Total Directed Road Segments",
                    "Total Discretized Cells (7.5m)",
                    "Network Bounding Box (Lat)",
                    "Network Bounding Box (Lon)",
                    "Residential Origin Zones",
                    "Commercial Destination Hubs",
                    "Binary Serialization Format",
                ],
                "Value": [
                    f"{counts.get('node_count', 0):,}",
                    f"{counts.get('segment_count', 0):,}",
                    f"{counts.get('total_cells', 0):,}",
                    f"[{bbox.get('min_lat', 0):.4f}, {bbox.get('max_lat', 0):.4f}]",
                    f"[{bbox.get('min_lon', 0):.4f}, {bbox.get('max_lon', 0):.4f}]",
                    f"{len(metadata.get('origin_zones', {}))} Zones (Gulshan, Johar, Nazimabad, Korangi)",
                    f"{len(metadata.get('destination_zones', {}))} Zones (Chundrigar, Saddar, Clifton, SITE, Airport)",
                    "KarachiFlow Flat Binary v1 (Zero-Copy Mmap Ingestion)",
                ],
            })
            st.dataframe(meta_df, width="stretch", hide_index=True)

            st.markdown("#### 4. Lane Discretization Taxonomy")
            lane_dist = metadata.get("lane_distribution", {})
            lane_df = pd.DataFrame([
                {"Lanes": f"{k} Lanes", "Segment Count": f"{v:,}"}
                for k, v in sorted(lane_dist.items(), key=lambda x: int(x[0]))
            ])
            st.dataframe(lane_df, width="stretch", hide_index=True)

    # Dynamic execution loops (Timeline Auto-Play & Live Polling Monitor)
    if sim_mode == "Historical Replay" and auto_play and st.session_state.is_playing:
        time.sleep(play_speed)
        next_idx = (current_idx + 1) % len(step_numbers)
        st.session_state.step_slider = step_numbers[next_idx]
        st.rerun()
    elif sim_mode == "Live Simulation Monitor" and auto_poll:
        time.sleep(poll_interval)
        st.rerun()


if __name__ == "__main__":
    main()
