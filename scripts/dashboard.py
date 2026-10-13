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
    /* Dark charcoal/slate application background (#090d16) */
    .stApp {
        background-color: #090d16;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar container styling (#0b111e) */
    section[data-testid="stSidebar"] {
        background-color: #0b111e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
    }

    /* High-contrast monospaced typography for telemetry KPIs, speeds, and latency */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"],
    .telemetry-mono {
        font-family: "JetBrains Mono", "SF Mono", Consolas, monospace !important;
    }

    /* Refined glassmorphism metric cards (#101726 surface containers) */
    div[data-testid="metric-container"] {
        background: #101726 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(8px) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(56, 189, 248, 0.28) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45) !important;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.74rem !important;
        font-weight: 500 !important;
    }

    /* Status indicator badges & pills */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    .badge-online {
        background-color: rgba(0, 230, 118, 0.12);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .badge-pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #00e676;
        box-shadow: 0 0 6px #00e676;
    }
    .badge-spec {
        background-color: rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Engine Specification Card */
    .engine-spec-card {
        background: #101726;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 6px;
        margin-bottom: 12px;
        font-size: 0.76rem;
        line-height: 1.5;
    }
    .spec-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
        color: #cbd5e1;
    }
    .spec-row b {
        font-family: "JetBrains Mono", Consolas, monospace;
        font-weight: 600;
    }

    /* Enterprise-grade Header Styling */
    .dashboard-header {
        margin-bottom: 1.15rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dashboard-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .dashboard-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 3px;
        font-weight: 400;
    }

    /* Compact Telemetry Status Legend */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 14px;
        padding: 8px 14px;
        background: #101726;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.78rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #cbd5e1;
    }
    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Button styling */
    div.stButton > button {
        background: #101726 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        background: #162035 !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Tabs styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Sidebar Headings */
    .sidebar-section-title {
        font-size: 0.74rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 14px;
        margin-bottom: 6px;
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

    # Sidebar Controls & Navigation Overhaul
    with st.sidebar:
        # Header with KarachiFlow branding and clean status badge
        st.markdown(
            """
            <div style="padding: 2px 0 14px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em; color: #f8fafc;">KARACHIFLOW</span>
                    <span class="status-badge badge-online"><span class="badge-pulse"></span> ONLINE</span>
                </div>
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 4px; font-family: 'JetBrains Mono', Consolas, monospace;">
                    SIMULATION ENGINE ONLINE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Execution Engine & Architecture
        st.markdown('<div class="sidebar-section-title">Execution Engine & Architecture</div>', unsafe_allow_html=True)
        engine_choice = st.selectbox(
            "Execution Engine",
            [
                "Hybrid MS-MPI + OpenMP (Peak 8.29×)",
                "OpenMP Multi-Core (8 Threads)",
                "Distributed MS-MPI (4 Ranks)",
                "CUDA GPU Accelerated (RTX 4000 Ada)",
                "Sequential Baseline (Single-Thread)",
            ],
            index=0,
            help="Display current engine with technical hardware specifications and parallel scaling details.",
            label_visibility="collapsed",
        )

        engine_specs = {
            "Hybrid MS-MPI + OpenMP (Peak 8.29×)": {
                "workers": "2 Ranks × 4 Threads (8 Workers)",
                "speedup": "8.29× vs Sequential",
                "runtime": "170.85 ms / step",
                "comm": "Non-blocking Halo (R_comm=3.3%)",
            },
            "OpenMP Multi-Core (8 Threads)": {
                "workers": "8 Threads (Shared Memory)",
                "speedup": "7.57× vs Sequential",
                "runtime": "187.22 ms / step",
                "comm": "Static Loop Scheduling",
            },
            "Distributed MS-MPI (4 Ranks)": {
                "workers": "4 Domain Decomposition Ranks",
                "speedup": "7.48× vs Sequential",
                "runtime": "189.40 ms / step",
                "comm": "Load Imbalance λ = 1.005",
            },
            "CUDA GPU Accelerated (RTX 4000 Ada)": {
                "workers": "Massive Cellular Grid Warps",
                "speedup": "RTX 4000 Ada (20 GB GDDR6)",
                "runtime": "< 50 ms / step (Target)",
                "comm": "Unified Host/Device Memory",
            },
            "Sequential Baseline (Single-Thread)": {
                "workers": "1 Thread (C++17 Single-Core)",
                "speedup": "1.00× Reference",
                "runtime": "1,416.59 ms / step",
                "comm": "Zero Parallel Overhead",
            },
        }
        spec = engine_specs[engine_choice]
        st.markdown(
            f"""
            <div class="engine-spec-card">
                <div class="spec-row"><span>Parallel Workers:</span> <b>{spec['workers']}</b></div>
                <div class="spec-row"><span>Measured Speedup:</span> <b style="color: #4ade80;">{spec['speedup']}</b></div>
                <div class="spec-row"><span>Compute Latency:</span> <b>{spec['runtime']}</b></div>
                <div class="spec-row"><span>Scalability:</span> <b>{spec['comm']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Operational Mode
        st.markdown('<div class="sidebar-section-title">Operation Mode</div>', unsafe_allow_html=True)
        sim_mode = st.radio(
            "Operation Mode",
            ["Historical Replay", "Live Simulation Monitor"],
            index=0,
            help="Choose between scrubbing through completed snapshots or monitoring ongoing simulation runs.",
            label_visibility="collapsed",
        )

        # 3. Dedicated sidebar placeholder for real-time City KPIs
        sidebar_kpi_placeholder = st.container()

        # 4. Replay & Telemetry Time Controls
        if sim_mode == "Historical Replay":
            st.markdown('<div class="sidebar-section-title">Timeline Replay Controls</div>', unsafe_allow_html=True)

            if "active_step" not in st.session_state or st.session_state.active_step not in step_numbers:
                st.session_state.active_step = step_numbers[-1]
            if "is_playing" not in st.session_state:
                st.session_state.is_playing = False

            def _jump_step(target: int):
                st.session_state.active_step = target
                st.session_state.step_slider = target

            def _on_slider_change():
                st.session_state.active_step = st.session_state.step_slider

            # Synchronize widget state with active_step before widget instantiation
            st.session_state.step_slider = st.session_state.active_step

            # Step jump buttons with safe on_click callbacks (executed before widget instantiation)
            current_val = st.session_state.active_step
            current_idx = step_numbers.index(current_val) if current_val in step_numbers else len(step_numbers) - 1

            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            with col_btn1:
                st.button("⏮ First", on_click=_jump_step, args=(step_numbers[0],), use_container_width=True)
            with col_btn2:
                prev_step = step_numbers[max(0, current_idx - 1)]
                st.button("◀ Prev", on_click=_jump_step, args=(prev_step,), use_container_width=True)
            with col_btn3:
                next_step = step_numbers[min(len(step_numbers) - 1, current_idx + 1)]
                st.button("Next ▶", on_click=_jump_step, args=(next_step,), use_container_width=True)
            with col_btn4:
                st.button("Latest ⏭", on_click=_jump_step, args=(step_numbers[-1],), use_container_width=True)

            # Step slider
            selected_step = st.select_slider(
                "Simulation Step ($t$ seconds)",
                options=step_numbers,
                key="step_slider",
                on_change=_on_slider_change,
                help="Select the exact second of simulated traffic flow.",
            )
            current_idx = step_numbers.index(selected_step)

            # Replay Auto-Play Toggle
            auto_play = st.toggle("Auto-Play Timeline", value=st.session_state.is_playing)
            play_speed = st.slider("Playback Speed (s/step)", 0.2, 2.0, 0.8, 0.2)
            st.session_state.is_playing = auto_play
            poll_interval = 2.0
        else:
            st.markdown('<div class="sidebar-section-title">Live Telemetry Monitor</div>', unsafe_allow_html=True)
            st.info("Tracking `data/snapshots/` for incoming simulation frames.")
            selected_step = step_numbers[-1]
            current_idx = len(step_numbers) - 1
            auto_play = False
            play_speed = 0.8
            auto_poll = st.toggle("Auto-Refresh Telemetry", value=False)
            poll_interval = st.slider("Live Poll Interval (s)", 1.0, 5.0, 2.0, 0.5, key="poll_slider", disabled=not auto_poll)
            if st.button("🔄 Refresh Telemetry Now", use_container_width=True):
                st.rerun()
            st.caption(f"Currently tracking newest step: **t = {selected_step} s**")

        # 5. Geospatial Camera Presets
        st.markdown('<div class="sidebar-section-title">Geospatial Camera Presets</div>', unsafe_allow_html=True)
        viewport_presets = visualizer.get_viewport_presets(metadata)
        corridor_options = [
            "Metropolitan Overview",
            "Shahrah-e-Faisal Expressway",
            "Saddar Commercial Core",
            "University Road Arterial",
            "Clifton Marine Drive",
            "Superhighway M-9",
            "Port Qasim Freight Corridor",
        ]
        available_presets = [c for c in corridor_options if c in viewport_presets]
        preset_name = st.selectbox(
            "Corridor Preset",
            available_presets,
            index=0,
            help="Focus camera on key arterial corridors across Karachi.",
        )
        selected_view_state = viewport_presets[preset_name]

        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            pitch_deg = st.slider("Pitch (°)", 0, 60, int(selected_view_state.pitch), 5)
        with col_cam2:
            bearing_deg = st.slider("Bearing (°)", -180, 180, int(selected_view_state.bearing), 5)
        selected_view_state.pitch = pitch_deg
        selected_view_state.bearing = bearing_deg

        # 6. Telemetry Layer Controls
        st.markdown('<div class="sidebar-section-title">Telemetry Layer Controls</div>', unsafe_allow_html=True)
        col_lay1, col_lay2 = st.columns(2)
        with col_lay1:
            show_heatmap = st.checkbox("Congestion Heatmap", value=False)
            scale_widths = st.checkbox("Dynamic Road Width", value=True)
        with col_lay2:
            show_zones = st.checkbox("Destination Hubs", value=False)
            highlight_deadlocks = st.checkbox("Deadlock Highlighter", value=True)

        display_mode = st.selectbox(
            "Network Rendering Detail",
            [
                "Active Traffic Only",
                "Congested Corridors Only",
                "Severe Bottlenecks Only",
                "Standstill Deadlocks Only",
                "Major Arteries + Active Traffic",
                "Complete Road Network",
            ],
            index=0,
            help="Filter road network segments to optimize WebGL rendering throughput.",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4; font-family: 'JetBrains Mono', Consolas, monospace;">
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
        f"LIVE MONITOR • t = {selected_step}s"
        if sim_mode == "Live Simulation Monitor"
        else f"TELEMETRY REPLAY • t = {selected_step}s"
    )
    header_placeholder.markdown(
        f"""
        <div class="dashboard-header">
            <div class="dashboard-title-row">
                <div>
                    <h1 class="dashboard-title">KarachiFlow — High-Performance Traffic Simulation Dashboard</h1>
                    <div class="dashboard-subtitle">
                        Microscopic Nagel–Schreckenberg Cellular Automaton on OpenStreetMap Network with Multi-Core &amp; Distributed PDC Acceleration
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="status-badge badge-spec">{engine_choice.split(' ')[0]}</span>
                    <span class="status-badge badge-online"><span class="badge-pulse"></span> {status_label}</span>
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
        jam_density_pct = float(merged_df.loc[active_mask, "density_pct"].mean())
    else:
        avg_speed_cells = 0.0
        avg_speed_kmh = 0.0
        jam_density_pct = 0.0

    deadlock_count = int((merged_df["congestion_status"] == "Standstill Gridlock (Deadlock)").sum())
    bottleneck_count = int((merged_df["congestion_status"] == "Severe Bottleneck (Queue)").sum())
    moderate_count = int((merged_df["congestion_status"] == "Moderate Flow (Slowing)").sum())
    free_flow_count = int((merged_df["congestion_status"] == "Free Flow (Normal)").sum())
    congested_count = deadlock_count + bottleneck_count
    network_occupancy = (active_cars / metadata["counts"]["total_cells"]) * 100.0

    # Calculate completed trips based on vehicle demand conservation (injected == active + completed)
    total_target_vehicles = 5000
    max_step = max(step_numbers[-1], 1)
    cumulative_injected = min(total_target_vehicles, int(round(total_target_vehicles * (selected_step / max_step))))
    completed_trips = max(0, cumulative_injected - active_cars)

    # Populate Sidebar KPI Cards
    with sidebar_kpi_placeholder:
        st.markdown('<div class="sidebar-section-title">City Telemetry KPIs</div>', unsafe_allow_html=True)
        st.metric("🚗 Active Vehicles", f"{active_cars:,}")
        st.metric("🏁 Completed Trips", f"{completed_trips:,}")
        st.metric("⚡ Average City Speed", f"{avg_speed_kmh:.1f} km/h")
        st.metric("🛑 Standstill Deadlocks", f"{deadlock_count}")

    # Executive KPI Metric Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("🚗 Active Vehicles", f"{active_cars:,}", f"{network_occupancy:.1f}% network capacity")
    with kpi_col2:
        st.metric("🏁 Completed Trips", f"{completed_trips:,}", f"{(completed_trips / max(total_target_vehicles, 1)) * 100:.1f}% completed")
    with kpi_col3:
        st.metric("⚡ Network Average Speed", f"{avg_speed_kmh:.1f} km/h", f"{avg_speed_cells:.2f} cells/s")
    with kpi_col4:
        st.metric("📊 Jam Density", f"{jam_density_pct:.1f}%", f"{congested_count} congested segments")
    with kpi_col5:
        st.metric("🛑 Detected Deadlocks", f"{deadlock_count}", f"{(deadlock_count / max(active_segments, 1)) * 100:.1f}% of active" if active_segments > 0 else "0.0% of active")
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
        # Visual Congestion Legend (4-Tier Physics + Zones)
        st.markdown(
            """
            <div class="legend-container">
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #00e676; box-shadow: 0 0 8px #00e676;"></span>
                    <span><b>Free Flow:</b> v ≥ 0.70 v<sub>max</sub>, ρ &lt; 0.25</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ffd600; box-shadow: 0 0 8px #ffd600;"></span>
                    <span><b>Moderate Slowing:</b> 0.35 ≤ v &lt; 0.70 or 0.25 ≤ ρ &lt; 0.45</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ff1744; box-shadow: 0 0 8px #ff1744;"></span>
                    <span><b>Severe Bottleneck:</b> 0.05 &lt; v &lt; 0.35 or 0.45 ≤ ρ &lt; 0.65</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ff0080; box-shadow: 0 0 10px #ff0080;"></span>
                    <span><b>Standstill Deadlock:</b> v ≤ 0.05, n ≥ 2, ρ ≥ 0.65</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #00d2ff;"></span>
                    <span><b>Origin Zones:</b> Residential Hubs</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot" style="background-color: #ff9600;"></span>
                    <span><b>Destination Hubs:</b> Commercial &amp; Freight Ports</span>
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
            highlight_deadlocks=highlight_deadlocks,
            scale_widths=scale_widths,
        )
        st.pydeck_chart(deck, width="stretch")

        # Secondary Analytics Grid Below Map
        col_flow1, col_flow2 = st.columns([1, 1])

        with col_flow1:
            st.markdown("#### 🚦 Citywide Congestion & Deadlock Distribution")
            if active_segments > 0:
                p_free = (free_flow_count / active_segments) * 100.0
                p_mod = (moderate_count / active_segments) * 100.0
                p_bot = (bottleneck_count / active_segments) * 100.0
                p_dead = (deadlock_count / active_segments) * 100.0

                dist_df = pd.DataFrame({
                    "Congestion State": [
                        "Free Flow (🟢)",
                        "Moderate Slowing (🟡)",
                        "Severe Bottleneck (🔴)",
                        "Standstill Deadlock (🟣)",
                    ],
                    "Segment Count": [free_flow_count, moderate_count, bottleneck_count, deadlock_count],
                    "Percentage (%)": [f"{p_free:.1f}%", f"{p_mod:.1f}%", f"{p_bot:.1f}%", f"{p_dead:.1f}%"],
                })
                st.dataframe(dist_df, width="stretch", hide_index=True)

                st.progress(
                    int(p_free),
                    text=f"Free-Flow Arteries ({p_free:.1f}% of active traffic network)",
                )
            else:
                st.info("No vehicles currently circulating on the network at t = 0.")

        with col_flow2:
            st.markdown("#### 🛑 Top Critical Congestion & Deadlock Hotspots")
            critical_df = (
                merged_df[merged_df["vehicle_count"] > 0]
                .sort_values(by=["density_ratio", "vehicle_count"], ascending=[False, False])
                .head(6)[
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
                        "vehicle_count": "Active Cars",
                        "avg_speed_kmh": "Speed (km/h)",
                        "density_pct": "Density (%)",
                        "congestion_status": "Traffic Status",
                    }
                )
            )
            st.dataframe(critical_df, width="stretch", hide_index=True)

    # TAB 2: PARALLEL COMPUTING BENCHMARKS (PDC)
    with tab_benchmarks:
        st.markdown("### ⚡ Parallel Computing Performance & Scaling Analysis")
        
        benchmarks = metrics_plotter.get_empirical_benchmarks()
        is_live = benchmarks.get("metadata", {}).get("is_live_data", False)
        hw_cpu = benchmarks.get("metadata", {}).get("hardware_cpu", "Intel Core i7-14700")
        hw_gpu = benchmarks.get("metadata", {}).get("hardware_gpu", "NVIDIA RTX 4000 Ada")

        if is_live:
            st.success(f"🟢 **Live Benchmark Records Loaded:** Evaluated on host ({hw_cpu} | {hw_gpu})")
        else:
            st.info(f"ℹ️ **Empirical HPC Reference Profile:** Evaluated on University Workstation ({hw_cpu} | {hw_gpu})")

        # Benchmark Sub-Controls
        col_scale_ctl1, col_scale_ctl2 = st.columns([1, 2])
        with col_scale_ctl1:
            baseline_mode = st.radio(
                "Speedup Baseline Reference",
                ["Self-Relative (T1/Tp)", "Speedup vs Sequential Baseline"],
                horizontal=True,
                help="Switch between relative speedup within each paradigm or absolute speedup over single-core C++ reference.",
            )

        # Plotly Charts: Row 1 (Strong Scaling & Efficiency)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_speedup = metrics_plotter.plot_speedup_curves(benchmarks, baseline_mode)
            st.plotly_chart(fig_speedup, width="stretch")

        with col_chart2:
            fig_efficiency = metrics_plotter.plot_efficiency_curves(benchmarks)
            st.plotly_chart(fig_efficiency, width="stretch")

        # Plotly Charts: Row 2 (Wall-Clock Runtime & Load Balance)
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            fig_runtime = metrics_plotter.plot_runtime_comparison(benchmarks)
            st.plotly_chart(fig_runtime, width="stretch")

        with col_chart4:
            fig_imbalance = metrics_plotter.plot_load_imbalance(benchmarks)
            st.plotly_chart(fig_imbalance, width="stretch")

        # Plotly Charts: Row 3 (GPU Acceleration & MPI Halo Overhead)
        col_chart5, col_chart6 = st.columns([1, 1])
        with col_chart5:
            fig_comm = metrics_plotter.plot_mpi_communication_overhead(benchmarks)
            st.plotly_chart(fig_comm, width="stretch")

        with col_chart6:
            fig_gpu = metrics_plotter.plot_gpu_vs_cpu_comparison(benchmarks)
            st.plotly_chart(fig_gpu, width="stretch")

        # Plotly Charts: Row 4 (What-If Urban Mobility Shockwave Analysis)
        fig_what_if = metrics_plotter.plot_what_if_bottleneck_scenario(benchmarks)
        st.plotly_chart(fig_what_if, width="stretch")

        # Scaling Comparison Table
        st.markdown("#### 📋 Comprehensive Architectural Benchmark Summary")
        summary_table = metrics_plotter.create_scaling_summary_table(benchmarks)
        st.dataframe(summary_table, width="stretch", hide_index=True)

        # Key Takeaways Box
        st.info(
            """
            **PDC Engineering Insights:**
            - **OpenMP Scaling:** Loop iterations partition dynamically across road segments via `schedule(dynamic, 64)`, balancing urban network load across heterogeneous core types.
            - **Distributed MS-MPI Scaling:** Non-blocking halo exchanges (`MPI_Isend` / `MPI_Irecv`) achieve a lightweight communication overhead of **R_comm ≤ 3.3%** and load imbalance **λ ≈ 1.005**.
            - **Hybrid MPI + OpenMP Synergy:** Combines inter-node domain decomposition with intra-node multi-threading, maximizing throughput across physical compute nodes.
            - **GPU Acceleration Leap:** SoA resident memory layout coupled with deterministic `atomicMin` collision arbitration delivers massive SIMT parallelism, scaling into billions of cell updates per second on NVIDIA Ada Lovelace.
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
        st.session_state.active_step = step_numbers[next_idx]
        st.rerun()
    elif sim_mode == "Live Simulation Monitor" and auto_poll:
        time.sleep(poll_interval)
        st.rerun()


if __name__ == "__main__":
    main()
