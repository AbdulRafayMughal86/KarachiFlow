"""
KarachiFlow — High-Performance Traffic Simulation & GIS Telemetry Dashboard
=============================================================================
Bespoke executive single-page GIS telemetry web application for microscopic
traffic simulation replay, congestion shockwave analysis, and HPC scaling benchmarks.
Zero sidebar clutter, unified floating controls, and high-contrast dark telemetry theme.
"""

import os
import glob
import json
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# Import modular components
import visualizer
import metrics_plotter

# Page Configuration — Immersive Full-Width Canvas, Sidebar Hidden
st.set_page_config(
    page_title="KarachiFlow — Traffic Simulation Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Obsidian & Glassmorphism Theme System (Overrides default Streamlit chrome)
st.markdown(
    """
    <style>
    /* 1. Global Reset & Obsidian Theme (#06080d with subtle radial depth) */
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #0d1527 0%, #06080d 100%) !important;
        color: #f1f5f9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* 2. Completely Erase Streamlit Sidebar & Chrome */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        display: none !important;
    }
    footer {
        display: none !important;
    }
    .stDeployButton,
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* 3. Optimal Viewport Padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* 4. Top Executive Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(16, 23, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 10px 20px;
        margin-bottom: 1rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0;
    }
    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(0, 230, 118, 0.12);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #00e676;
        box-shadow: 0 0 8px #00e676;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }
    .spec-chip {
        font-size: 0.75rem;
        color: #64748b;
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    .nav-meta {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .engine-chip {
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        font-family: "JetBrains Mono", Consolas, monospace;
    }

    /* 5. Modern Segmented Tab Bar */
    div[data-testid="stTabs"] {
        margin-bottom: 1rem;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        background: rgba(16, 23, 38, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 4px 6px !important;
        gap: 6px !important;
        backdrop-filter: blur(12px) !important;
    }
    div[data-testid="stTabs"] [role="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: rgba(56, 189, 248, 0.16) !important;
        color: #38bdf8 !important;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.2) !important;
    }

    /* 6. High-Contrast Frosted Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(16, 23, 38, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        font-family: "JetBrains Mono", Consolas, monospace !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        font-family: "JetBrains Mono", Consolas, monospace !important;
        color: #94a3b8 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }

    /* 7. Floating Media & Command Dock */
    .control-dock {
        background: rgba(16, 23, 38, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 12px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .dock-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dock-title {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin: 0;
    }
    .compact-legend {
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 0.74rem;
        font-family: "JetBrains Mono", Consolas, monospace;
    }
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #cbd5e1;
    }
    .dot-green { width: 8px; height: 8px; border-radius: 50%; background: #00e676; box-shadow: 0 0 6px #00e676; }
    .dot-amber { width: 8px; height: 8px; border-radius: 50%; background: #ffd600; box-shadow: 0 0 6px #ffd600; }
    .dot-red   { width: 8px; height: 8px; border-radius: 50%; background: #ff1744; box-shadow: 0 0 6px #ff1744; }
    .dot-deadlock { width: 8px; height: 8px; border-radius: 50%; background: #ff0080; box-shadow: 0 0 8px #ff0080; }

    /* Button and Toggle styling */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(30, 41, 59, 0.6) !important;
        color: #f1f5f9 !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        background: rgba(56, 189, 248, 0.2) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stToggle"] label {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Paths
DATA_DIR = "data"
BIN_PATH = os.path.join(DATA_DIR, "karachi_network.bin")
METADATA_PATH = os.path.join(DATA_DIR, "karachi_network_metadata.json")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")


@st.cache_data(show_spinner=False)
def get_cached_network_geometry() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and caches the 74,914-segment binary road network."""
    return visualizer.load_network_geometry(BIN_PATH, METADATA_PATH)


@st.cache_data(show_spinner=False)
def get_cached_snapshot(snapshot_path: str) -> pd.DataFrame:
    """Loads and caches a single microscopic simulation snapshot."""
    return visualizer.load_snapshot(snapshot_path)


@st.cache_data(show_spinner=False)
def get_cached_merged_telemetry(snapshot_path: str) -> pd.DataFrame:
    """Loads snapshot and merges with cached road network geometry."""
    network_df, _ = get_cached_network_geometry()
    snapshot_df = get_cached_snapshot(snapshot_path)
    return visualizer.merge_telemetry(network_df, snapshot_df)


def discover_available_snapshots() -> Tuple[List[int], Dict[int, str]]:
    """Scans snapshot directory and returns sorted available timesteps."""
    files = glob.glob(os.path.join(SNAPSHOT_DIR, "snapshot_t*.csv"))
    step_to_path: Dict[int, str] = {}
    for f in files:
        basename = os.path.basename(f)
        try:
            step_str = basename.replace("snapshot_t", "").replace(".csv", "")
            step_num = int(step_str)
            step_to_path[step_num] = f
        except ValueError:
            continue
    sorted_steps = sorted(step_to_path.keys())
    return sorted_steps, step_to_path


def main():
    # Verify prerequisites
    if not os.path.exists(BIN_PATH) or not os.path.exists(METADATA_PATH):
        st.error(
            f"❌ Karachi road network assets missing in `{DATA_DIR}/`. "
            f"Please run `python scripts/preprocess_graph.py` first."
        )
        return

    network_df, metadata = get_cached_network_geometry()
    step_numbers, step_to_path = discover_available_snapshots()

    if not step_numbers:
        st.warning("⚠️ No simulation snapshots found in `data/snapshots/`. Launch a run with `.\\bin\\karachiflow.exe --dev`.")
        return

    # Load dynamic benchmark dataset with empirical reference fallback
    benchmarks = metrics_plotter.load_benchmark_data()
    hyb_bench = benchmarks.get("hybrid", {})
    omp_bench = benchmarks.get("openmp", {})
    mpi_bench = benchmarks.get("mpi", {})
    cuda_bench = benchmarks.get("cuda", {})
    seq_bench = benchmarks.get("sequential", {})

    def _extract_metric(bench: dict, key: str, default: float = 1.0, use_max: bool = True) -> float:
        val = bench.get(key)
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(max(val)) if use_max else float(val[-1])
        return default

    hyb_sp = _extract_metric(hyb_bench, "speedup_vs_seq", default=1.0, use_max=True)
    omp_sp = _extract_metric(omp_bench, "speedup_vs_seq", default=1.0, use_max=True)
    mpi_sp = _extract_metric(mpi_bench, "speedup_vs_seq", default=1.0, use_max=True)
    cuda_sp = _extract_metric(cuda_bench, "speedup_vs_seq", default=1.0, use_max=True)
    seq_sp = 1.00

    hyb_tp = _extract_metric(hyb_bench, "throughput_mcells", default=1.0, use_max=True)
    omp_tp = _extract_metric(omp_bench, "throughput_mcells", default=1.0, use_max=True)
    mpi_tp = _extract_metric(mpi_bench, "throughput_mcells", default=1.0, use_max=True)
    cuda_tp = _extract_metric(cuda_bench, "throughput_mcells", default=1.0, use_max=True)
    seq_tp = _extract_metric(seq_bench, "throughput_mcells", default=1.0, use_max=True)

    hyb_eff = _extract_metric(hyb_bench, "efficiency_self", default=1.0, use_max=True)
    omp_eff = _extract_metric(omp_bench, "efficiency_self", default=1.0, use_max=False)
    mpi_eff = _extract_metric(mpi_bench, "efficiency_self", default=1.0, use_max=False)

    ENGINE_OPTIONS = [
        f"Hybrid MS-MPI + OpenMP (Peak {hyb_sp:.2f}×)",
        f"OpenMP Multi-Core ({omp_sp:.2f}×)",
        f"Distributed MS-MPI ({mpi_sp:.2f}×)",
        f"CUDA RTX 4000 Ada ({cuda_sp:.2f}×)",
        "Sequential Baseline T1",
    ]

    ENGINE_METRICS = {
        ENGINE_OPTIONS[0]: {
            "speedup": f"{hyb_sp:.2f}×",
            "chip": f"⚡ Hybrid MPI+OpenMP {hyb_sp:.2f}×",
            "desc": "MPI + OpenMP Hybrid Scaling",
            "throughput": f"{hyb_tp:.1f} Mcells/s",
            "efficiency": f"{hyb_eff * 100.0:.1f}%",
        },
        ENGINE_OPTIONS[1]: {
            "speedup": f"{omp_sp:.2f}×",
            "chip": f"⚡ OpenMP {omp_sp:.2f}×",
            "desc": "Shared-Memory OpenMP Threads",
            "throughput": f"{omp_tp:.1f} Mcells/s",
            "efficiency": f"{omp_eff * 100.0:.1f}%",
        },
        ENGINE_OPTIONS[2]: {
            "speedup": f"{mpi_sp:.2f}×",
            "chip": f"⚡ MS-MPI {mpi_sp:.2f}×",
            "desc": "Domain Decomposition Ranks",
            "throughput": f"{mpi_tp:.1f} Mcells/s",
            "efficiency": f"{mpi_eff * 100.0:.1f}%",
        },
        ENGINE_OPTIONS[3]: {
            "speedup": f"{cuda_sp:.2f}×",
            "chip": f"⚡ CUDA {cuda_sp:.2f}×",
            "desc": "Massively Parallel GPU Kernels",
            "throughput": f"{cuda_tp:.1f} Mcells/s",
            "efficiency": "N/A (GPU Acceleration)",
        },
        ENGINE_OPTIONS[4]: {
            "speedup": "1.00×",
            "chip": "⏱ Sequential 1.00×",
            "desc": "Single-Thread Reference Baseline",
            "throughput": f"{seq_tp:.1f} Mcells/s",
            "efficiency": "100.0%",
        },
    }

    # Initialize Session State
    if "active_step" not in st.session_state:
        st.session_state.active_step = step_numbers[-1]
    if "step_slider" not in st.session_state:
        st.session_state.step_slider = step_numbers[-1]
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "corridor_preset" not in st.session_state:
        st.session_state.corridor_preset = "Metropolitan Overview"
    if "engine_choice" not in st.session_state or st.session_state.engine_choice not in ENGINE_METRICS:
        st.session_state.engine_choice = ENGINE_OPTIONS[0]
    if "display_mode" not in st.session_state:
        st.session_state.display_mode = "Major Arteries + Active Traffic"
    if "show_zones" not in st.session_state:
        st.session_state.show_zones = False
    if "show_heatmap" not in st.session_state:
        st.session_state.show_heatmap = False
    if "highlight_deadlocks" not in st.session_state:
        st.session_state.highlight_deadlocks = True
    if "scale_widths" not in st.session_state:
        st.session_state.scale_widths = True

    # Dedicated transport callbacks
    def _step_prev():
        st.session_state.is_playing = False
        curr = st.session_state.active_step
        if curr in step_numbers:
            idx = max(0, step_numbers.index(curr) - 1)
        else:
            idx = 0
        if step_numbers:
            st.session_state.active_step = step_numbers[idx]
            st.session_state.step_slider = step_numbers[idx]

    def _step_next():
        st.session_state.is_playing = False
        curr = st.session_state.active_step
        if curr in step_numbers:
            idx = min(len(step_numbers) - 1, step_numbers.index(curr) + 1)
        else:
            idx = 0
        if step_numbers:
            st.session_state.active_step = step_numbers[idx]
            st.session_state.step_slider = step_numbers[idx]

    def _toggle_play():
        st.session_state.is_playing = not st.session_state.is_playing

    def _on_slider_change():
        st.session_state.is_playing = False
        st.session_state.active_step = st.session_state.step_slider

    # Determine Active Step
    selected_step = st.session_state.active_step
    if selected_step not in step_to_path:
        selected_step = step_numbers[-1]
        st.session_state.active_step = selected_step

    # Synchronize step_slider before widget instantiation
    if "step_slider" not in st.session_state or st.session_state.step_slider != selected_step:
        st.session_state.step_slider = selected_step

    # Load and merge active snapshot telemetry via cache
    active_snapshot_path = step_to_path[selected_step]
    merged_df = get_cached_merged_telemetry(active_snapshot_path)

    # Compute Core Executive Metrics
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
    congested_count = deadlock_count + bottleneck_count
    network_occupancy = (active_cars / metadata["counts"]["total_cells"]) * 100.0

    total_target_vehicles = 5000
    meta_path = "data/snapshots/snapshot_metadata.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                s_meta = json.load(mf)
                total_target_vehicles = int(s_meta.get("total_vehicles", 5000))
        except Exception:
            pass
    total_target_vehicles = max(total_target_vehicles, active_cars)
    max_step = max(step_numbers[-1], 1)
    cumulative_injected = min(total_target_vehicles, int(round(total_target_vehicles * (selected_step / max_step))))
    completed_trips = max(0, cumulative_injected - active_cars)

    # Active engine profile metadata
    selected_engine = st.session_state.engine_choice
    if selected_engine not in ENGINE_METRICS:
        selected_engine = ENGINE_OPTIONS[0]
        st.session_state.engine_choice = selected_engine
    active_profile = ENGINE_METRICS[selected_engine]

    # 1. Top Executive Navigation Bar
    st.markdown(
        f"""
        <div class="top-nav">
            <div class="nav-brand">
                <h1 class="brand-title">KarachiFlow</h1>
                <span class="brand-badge"><span class="pulse-dot"></span> Telemetry Active</span>
                <span class="spec-chip">74,914 Segments • 1.07M Cells (7.5m)</span>
            </div>
            <div class="nav-meta">
                <span class="engine-chip">{active_profile["chip"]}</span>
                <span class="spec-chip">Step t = {selected_step}s</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive Metric Ribbon (6 High-Impact Telemetry Cards)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("🚗 Active Vehicles", f"{active_cars:,}", f"{network_occupancy:.1f}% network capacity", delta_color="off")
    with kpi_col2:
        st.metric("🏁 Completed Trips", f"{completed_trips:,}", f"{(completed_trips / max(total_target_vehicles, 1)) * 100:.1f}% completed", delta_color="off")
    with kpi_col3:
        st.metric("⚡ Network Average Speed", f"{avg_speed_kmh:.1f} km/h", f"{avg_speed_cells:.2f} cells/s", delta_color="off")
    with kpi_col4:
        st.metric("📊 Jam Density", f"{jam_density_pct:.1f}%", f"{congested_count:,} congested segments", delta_color="off")
    with kpi_col5:
        st.metric("🛑 Detected Deadlocks", f"{deadlock_count:,}", f"{(deadlock_count / max(active_segments, 1)) * 100:.1f}% of active", delta_color="off")
    with kpi_col6:
        st.metric("🚀 Peak Speedup", active_profile["speedup"], active_profile["desc"], delta_color="off")

    # 3. Main Stage Presentation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🗺️ City Replay & Congestion Flow",
        "📊 Parallel Computing Benchmarks",
        "🚨 What-If Incident Simulation & Physics",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: City Replay & Congestion Flow (Primary Stage)
    # -------------------------------------------------------------------------
    with tab1:
        # PyDeck WebGL Map
        viewports = visualizer.get_viewport_presets(metadata)
        active_viewport = viewports.get(st.session_state.corridor_preset, viewports["Metropolitan Overview"])

        deck_cache_key = (
            selected_step,
            st.session_state.corridor_preset,
            st.session_state.display_mode,
            st.session_state.show_zones,
            st.session_state.show_heatmap,
            st.session_state.highlight_deadlocks,
            st.session_state.scale_widths,
        )
        deck = visualizer.get_cached_traffic_deck(
            merged_df=merged_df,
            metadata=metadata,
            view_state=active_viewport,
            display_mode=st.session_state.display_mode,
            show_zones=st.session_state.show_zones,
            show_heatmap=st.session_state.show_heatmap,
            highlight_deadlocks=st.session_state.highlight_deadlocks,
            scale_widths=st.session_state.scale_widths,
            cache_key=deck_cache_key,
        )
        st.pydeck_chart(deck, width="stretch", height=490)

        # Integrated Simulation Transport & Layer Deck (Unified Dock)
        st.markdown(
            """
            <div class="control-dock">
                <div class="dock-header-row">
                    <div class="dock-title">Simulation Transport & Layer Deck</div>
                    <div class="compact-legend">
                        <span class="legend-chip"><span class="dot-green"></span> Free Flow (≥70%)</span>
                        <span class="legend-chip"><span class="dot-amber"></span> Moderate (35–70%)</span>
                        <span class="legend-chip"><span class="dot-red"></span> Bottleneck (5–35%)</span>
                        <span class="legend-chip"><span class="dot-deadlock"></span> Deadlock (≤5%)</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Row 1: Unified Playback Controls & Timeline Scrubber
        t_col1, t_col2, t_col3, t_col4 = st.columns([0.8, 1.1, 0.8, 5.5])
        with t_col1:
            st.button("◀ Prev", on_click=_step_prev, key="btn_prev", width="stretch")
        with t_col2:
            play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
            st.button(play_label, on_click=_toggle_play, key="btn_play", width="stretch")
        with t_col3:
            st.button("Next ▶", on_click=_step_next, key="btn_next", width="stretch")
        with t_col4:
            st.select_slider(
                "Simulation Timeline Step",
                options=step_numbers,
                format_func=lambda s: f"t = {s}s",
                key="step_slider",
                on_change=_on_slider_change,
                label_visibility="collapsed",
            )

        # Row 2: Architecture, Corridor Focus, and Filter Selectors
        c_engine, c_corridor, c_filter = st.columns([1.1, 1.0, 1.0])
        with c_engine:
            def _on_engine_change():
                st.session_state.engine_choice = st.session_state.engine_select

            st.selectbox(
                "Execution Architecture",
                ENGINE_OPTIONS,
                index=ENGINE_OPTIONS.index(st.session_state.engine_choice) if st.session_state.engine_choice in ENGINE_OPTIONS else 0,
                key="engine_select",
                on_change=_on_engine_change,
                label_visibility="collapsed",
            )
        with c_corridor:
            def _on_corridor_change():
                st.session_state.corridor_preset = st.session_state.corridor_select

            corridor_keys = list(viewports.keys())
            st.selectbox(
                "Corridor Focus",
                corridor_keys,
                index=corridor_keys.index(st.session_state.corridor_preset) if st.session_state.corridor_preset in corridor_keys else 0,
                key="corridor_select",
                on_change=_on_corridor_change,
                label_visibility="collapsed",
            )
        with c_filter:
            def _on_filter_change():
                st.session_state.display_mode = st.session_state.filter_select

            filter_options = [
                "Major Arteries + Active Traffic",
                "Active Traffic Only",
                "Congested Corridors Only",
                "Severe Bottlenecks Only",
                "Standstill Deadlocks Only",
                "Complete Road Network",
            ]
            st.selectbox(
                "Display Filter",
                filter_options,
                index=filter_options.index(st.session_state.display_mode) if st.session_state.display_mode in filter_options else 0,
                key="filter_select",
                on_change=_on_filter_change,
                label_visibility="collapsed",
            )

        # Row 3: Visualization Overlay Toggles
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        with l_col1:
            st.toggle("🏙️ OD Zones", key="show_zones")
        with l_col2:
            st.toggle("🔥 Bottleneck Heatmap", key="show_heatmap")
        with l_col3:
            st.toggle("🛑 Deadlock Halos", key="highlight_deadlocks")
        with l_col4:
            st.toggle("📏 Dynamic Width Scaling", key="scale_widths")

        # Optional Collapsible Technical Diagnostics (Cleanly hidden by default!)
        with st.expander("🔬 Detailed Network Diagnostics & Raw Hotspots (Developer Mode)", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Citywide Congestion Distribution**")
                dist_data = {
                    "Congestion State": ["Free Flow (🟢)", "Moderate Slowing (🟡)", "Severe Bottleneck (🔴)", "Standstill Deadlock (🟣)"],
                    "Segment Count": [
                        int((merged_df["congestion_status"] == "Free Flow (Normal)").sum()),
                        int((merged_df["congestion_status"] == "Moderate Flow (Slowing)").sum()),
                        int((merged_df["congestion_status"] == "Severe Bottleneck (Queue)").sum()),
                        deadlock_count,
                    ],
                }
                summary_df = pd.DataFrame(dist_data)
                st.dataframe(summary_df, width="stretch")
            with t_col2:
                st.markdown("**Top Critical Congestion & Deadlock Hotspots**")
                hotspots = (
                    merged_df[merged_df["vehicle_count"] > 0]
                    .sort_values(by=["density_ratio", "vehicle_count"], ascending=[False, False])
                    .head(10)[["segment_id", "lanes", "vehicle_count", "avg_speed_kmh", "density_pct", "congestion_status"]]
                )
                st.dataframe(hotspots, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 2: Parallel Computing Benchmarks
    # -------------------------------------------------------------------------
    with tab2:
        benchmarks = metrics_plotter.load_benchmark_data()
        is_live = benchmarks.get("metadata", {}).get("is_live_data", False)
        
        status_text = "Live Benchmark Records Loaded" if is_live else "High-Performance Empirical Reference Data"
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <span class="brand-badge"><span class="pulse-dot"></span> {status_text}</span>
                <span class="spec-chip">Hardware: Intel Core i7 14th Gen • NVIDIA RTX 4000 Ada (20 GB)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Highlight card for selected execution architecture
        st.markdown(
            f"""
            <div style="background: rgba(16, 23, 38, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Active Architecture Focus</div>
                    <div style="font-size: 1.12rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', Consolas, monospace;">{st.session_state.engine_choice}</div>
                </div>
                <div style="display: flex; gap: 24px;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.70rem; color: #64748b; font-weight: 600;">SPEEDUP</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #00e676; font-family: 'JetBrains Mono', Consolas, monospace;">{active_profile['speedup']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.70rem; color: #64748b; font-weight: 600;">THROUGHPUT</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', Consolas, monospace;">{active_profile['throughput']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.70rem; color: #64748b; font-weight: 600;">EFFICIENCY</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #ffd600; font-family: 'JetBrains Mono', Consolas, monospace;">{active_profile['efficiency']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_sp = metrics_plotter.plot_speedup_curves(benchmarks, "Speedup vs Sequential Baseline")
            st.plotly_chart(fig_sp, width="stretch")
        with b_col2:
            fig_eff = metrics_plotter.plot_efficiency_curves(benchmarks)
            st.plotly_chart(fig_eff, width="stretch")

        b_col3, b_col4 = st.columns(2)
        with b_col3:
            fig_gpu = metrics_plotter.plot_gpu_vs_cpu_comparison(benchmarks)
            st.plotly_chart(fig_gpu, width="stretch")
        with b_col4:
            fig_overhead = metrics_plotter.plot_mpi_communication_overhead(benchmarks)
            st.plotly_chart(fig_overhead, width="stretch")

        st.markdown("### Executive Performance Summary Matrix")
        summary_table = metrics_plotter.create_scaling_summary_table(benchmarks)
        st.dataframe(summary_table, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 3: What-If Incident Simulation & Physics
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown(
            """
            <div style="margin-bottom: 16px;">
                <h3 style="margin: 0; color: #f8fafc;">🚨 Shahrah-e-Faisal Expressway Bottleneck Shockwave Analysis</h3>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    Simulated multi-vehicle incident near Karsaz Flyover: 2-lane blockage during morning rush-hour peak, measuring backward shockwave propagation velocity and arterial delay spillover.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            fig_whatif = metrics_plotter.plot_what_if_bottleneck_scenario(benchmarks)
            st.plotly_chart(fig_whatif, width="stretch")
        with w_col2:
            fig_weak = metrics_plotter.plot_weak_scaling_curves(benchmarks)
            st.plotly_chart(fig_weak, width="stretch")

        # Execution Runtime Comparison
        st.markdown("### Execution Runtime & Latency Profile")
        fig_runtime = metrics_plotter.plot_runtime_comparison(benchmarks)
        st.plotly_chart(fig_runtime, width="stretch")

    # Auto-play tick loop (executed after full render pass)
    if st.session_state.is_playing:
        time.sleep(0.8)
        curr_step = st.session_state.active_step
        next_idx = (step_numbers.index(curr_step) + 1) % len(step_numbers) if curr_step in step_numbers else 0
        st.session_state.active_step = step_numbers[next_idx]
        try:
            st.session_state.step_slider = step_numbers[next_idx]
        except Exception:
            pass
        st.rerun()


if __name__ == "__main__":
    main()
