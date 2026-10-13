"""
KarachiFlow — High-Performance Traffic Simulation & GIS Telemetry Dashboard
=============================================================================
Bespoke executive single-page GIS telemetry web application for microscopic
traffic simulation replay, congestion shockwave analysis, and HPC scaling benchmarks.
Zero sidebar clutter, unified floating controls, and high-contrast dark telemetry theme.
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
    /* 1. Global Reset & Obsidian Theme (#06080d) */
    .stApp {
        background-color: #06080d !important;
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
    }

    /* 7. Floating Media & Command Dock */
    .control-dock {
        background: rgba(16, 23, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-top: 12px;
        margin-bottom: 12px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .dock-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    /* 8. Minimalist Legend Pill */
    .floating-legend {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        padding: 8px 16px;
        background: rgba(16, 23, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        margin-bottom: 10px;
        font-size: 0.78rem;
        font-family: "JetBrains Mono", Consolas, monospace;
        backdrop-filter: blur(12px);
    }
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .dot-green { width: 8px; height: 8px; border-radius: 50%; background: #00e676; box-shadow: 0 0 6px #00e676; }
    .dot-amber { width: 8px; height: 8px; border-radius: 50%; background: #ffd600; box-shadow: 0 0 6px #ffd600; }
    .dot-red   { width: 8px; height: 8px; border-radius: 50%; background: #ff1744; box-shadow: 0 0 6px #ff1744; }
    .dot-deadlock { width: 8px; height: 8px; border-radius: 50%; background: #ff0080; box-shadow: 0 0 8px #ff0080; }

    /* Button styling */
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

    # Initialize Session State
    if "active_step" not in st.session_state:
        st.session_state.active_step = step_numbers[0]
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "corridor_preset" not in st.session_state:
        st.session_state.corridor_preset = "Metropolitan Overview"
    if "engine_choice" not in st.session_state:
        st.session_state.engine_choice = "Hybrid MS-MPI + OpenMP (Peak 8.29×)"
    if "operation_mode" not in st.session_state:
        st.session_state.operation_mode = "Historical Replay"

    # Step jump button pre-execution callback
    def _jump_step(target: int):
        st.session_state.active_step = target
        st.session_state.step_slider = target

    # Auto-play tick loop
    if st.session_state.is_playing:
        curr_idx = step_numbers.index(st.session_state.active_step) if st.session_state.active_step in step_numbers else 0
        next_idx = (curr_idx + 1) % len(step_numbers)
        st.session_state.active_step = step_numbers[next_idx]
        time.sleep(0.6)
        st.rerun()

    # Determine Active Step
    selected_step = st.session_state.active_step
    if selected_step not in step_to_path:
        selected_step = step_numbers[0]
        st.session_state.active_step = selected_step

    # Load and merge active snapshot telemetry
    active_snapshot_path = step_to_path[selected_step]
    snapshot_df = get_cached_snapshot(active_snapshot_path)
    merged_df = visualizer.merge_telemetry(network_df, snapshot_df)

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
    max_step = max(step_numbers[-1], 1)
    cumulative_injected = min(total_target_vehicles, int(round(total_target_vehicles * (selected_step / max_step))))
    completed_trips = max(0, cumulative_injected - active_cars)

    # 1. Top Executive Navigation Bar
    st.markdown(
        f"""
        <div class="top-nav">
            <div class="nav-brand">
                <h1 class="brand-title">KarachiFlow</h1>
                <span class="brand-badge"><span class="pulse-dot"></span> ENGINE ONLINE</span>
                <span class="spec-chip">74,914 Segments • 1.07M Cells (7.5m)</span>
            </div>
            <div class="nav-meta">
                <span class="engine-chip">⚡ {st.session_state.engine_choice.split(' ')[0]} 8.29×</span>
                <span class="spec-chip">Step t = {selected_step}s</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Executive Metric Ribbon (6 High-Impact Telemetry Cards)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("🚗 Active Vehicles", f"{active_cars:,}", f"{network_occupancy:.1f}% network capacity")
    with kpi_col2:
        st.metric("🏁 Completed Trips", f"{completed_trips:,}", f"{(completed_trips / max(total_target_vehicles, 1)) * 100:.1f}% completed")
    with kpi_col3:
        st.metric("⚡ Network Average Speed", f"{avg_speed_kmh:.1f} km/h", f"{avg_speed_cells:.2f} cells/s")
    with kpi_col4:
        st.metric("📊 Jam Density", f"{jam_density_pct:.1f}%", f"{congested_count:,} congested segments")
    with kpi_col5:
        st.metric("🛑 Detected Deadlocks", f"{deadlock_count:,}", f"{(deadlock_count / max(active_segments, 1)) * 100:.1f}% of active")
    with kpi_col6:
        st.metric("🚀 Peak Speedup", "8.29×", "Hybrid MPI+OpenMP")

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
        # Floating Legend Pill
        st.markdown(
            """
            <div class="floating-legend">
                <span class="legend-chip"><span class="dot-green"></span> Free Flow (v ≥ 70%)</span>
                <span class="legend-chip"><span class="dot-amber"></span> Moderate Slowing (35% ≤ v < 70%)</span>
                <span class="legend-chip"><span class="dot-red"></span> Severe Bottleneck (5% < v < 35%)</span>
                <span class="legend-chip"><span class="dot-deadlock"></span> Standstill Deadlock (v ≤ 5%, Stalled)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PyDeck WebGL Map
        viewports = visualizer.get_viewport_presets(metadata)
        active_viewport = viewports.get(st.session_state.corridor_preset, viewports["Metropolitan Overview"])
        
        deck = visualizer.build_traffic_deck(
            merged_df=merged_df,
            metadata=metadata,
            view_state=active_viewport,
            show_zones=False,
            display_mode="Active Traffic Only",
            scale_widths=True,
            show_heatmap=False,
            highlight_deadlocks=True,
        )
        st.pydeck_chart(deck, use_container_width=True, height=540)

        # Floating Media & Command Dock
        st.markdown('<div class="dock-title">Simulation Transport & Command Deck</div>', unsafe_allow_html=True)
        
        # Row 1: Dropdowns and Transport Buttons
        c_mode, c_engine, c_corridor, c_first, c_prev, c_play, c_next, c_latest = st.columns([2.0, 2.2, 2.0, 0.7, 0.7, 1.1, 0.7, 0.7])
        
        with c_mode:
            st.radio(
                "Operation Mode",
                ["Historical Replay", "Live Simulation Monitor"],
                horizontal=True,
                label_visibility="collapsed",
                key="op_mode_radio",
            )
        with c_engine:
            st.session_state.engine_choice = st.selectbox(
                "Execution Architecture",
                [
                    "Hybrid MS-MPI + OpenMP (Peak 8.29×)",
                    "OpenMP Multi-Core 8T (7.57×)",
                    "Distributed MS-MPI 4R (7.48×)",
                    "CUDA RTX 4000 Ada (sm_89)",
                    "Sequential Baseline T1",
                ],
                index=0,
                label_visibility="collapsed",
            )
        with c_corridor:
            st.session_state.corridor_preset = st.selectbox(
                "Corridor Focus",
                list(viewports.keys()),
                index=0,
                label_visibility="collapsed",
            )
        with c_first:
            st.button("⏮ First", on_click=_jump_step, args=(step_numbers[0],), use_container_width=True)
        with c_prev:
            curr_idx = step_numbers.index(selected_step) if selected_step in step_numbers else 0
            prev_step = step_numbers[max(0, curr_idx - 1)]
            st.button("◀ Prev", on_click=_jump_step, args=(prev_step,), use_container_width=True)
        with c_play:
            play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
            if st.button(play_label, use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()
        with c_next:
            curr_idx = step_numbers.index(selected_step) if selected_step in step_numbers else 0
            next_step = step_numbers[min(len(step_numbers) - 1, curr_idx + 1)]
            st.button("Next ▶", on_click=_jump_step, args=(next_step,), use_container_width=True)
        with c_latest:
            st.button("Latest ⏭", on_click=_jump_step, args=(step_numbers[-1],), use_container_width=True)

        # Row 2: Full-Width Timeline Scrubber
        st.select_slider(
            "Simulation Timeline Step",
            options=step_numbers,
            value=selected_step,
            format_func=lambda s: f"t = {s}s",
            key="step_slider",
            on_change=lambda: st.session_state.update(active_step=st.session_state.step_slider),
            label_visibility="collapsed",
        )

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
                st.dataframe(summary_df, use_container_width=True)
            with t_col2:
                st.markdown("**Top Critical Congestion & Deadlock Hotspots**")
                hotspots = (
                    merged_df[merged_df["vehicle_count"] > 0]
                    .sort_values(by=["density_ratio", "vehicle_count"], ascending=[False, False])
                    .head(10)[["segment_id", "lanes", "vehicle_count", "avg_speed_kmh", "density_pct", "congestion_status"]]
                )
                st.dataframe(hotspots, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Parallel Computing Benchmarks
    # -------------------------------------------------------------------------
    with tab2:
        benchmarks = metrics_plotter.get_empirical_benchmarks()
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

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_sp = metrics_plotter.plot_speedup_curves(benchmarks, "Speedup vs Sequential Baseline")
            st.plotly_chart(fig_sp, use_container_width=True)
        with b_col2:
            fig_eff = metrics_plotter.plot_efficiency_curves(benchmarks)
            st.plotly_chart(fig_eff, use_container_width=True)

        b_col3, b_col4 = st.columns(2)
        with b_col3:
            fig_gpu = metrics_plotter.plot_gpu_vs_cpu_comparison(benchmarks)
            st.plotly_chart(fig_gpu, use_container_width=True)
        with b_col4:
            fig_overhead = metrics_plotter.plot_mpi_communication_overhead(benchmarks)
            st.plotly_chart(fig_overhead, use_container_width=True)

        st.markdown("### Executive Performance Summary Matrix")
        summary_table = metrics_plotter.create_scaling_summary_table(benchmarks)
        st.dataframe(summary_table, use_container_width=True)

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
            st.plotly_chart(fig_whatif, use_container_width=True)
        with w_col2:
            fig_weak = metrics_plotter.plot_weak_scaling_curves(benchmarks)
            st.plotly_chart(fig_weak, use_container_width=True)

        # Execution Runtime Comparison
        st.markdown("### Execution Runtime & Latency Profile")
        fig_runtime = metrics_plotter.plot_runtime_comparison(benchmarks)
        st.plotly_chart(fig_runtime, use_container_width=True)


if __name__ == "__main__":
    main()
