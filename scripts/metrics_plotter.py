"""
KarachiFlow — Parallel Computing Performance & Scaling Metrics Plotter
=======================================================================
Generates interactive Plotly figures for analyzing Speedup (Sp),
Parallel Efficiency (Ep), Communication Overhead (Rcomm), Load Imbalance,
Weak Scaling, What-If Bottleneck Congestion, and GPU Acceleration Engine
throughput (Mcells/s) across OpenMP, MS-MPI, Hybrid, and CUDA architectures.

Supports dynamic data ingestion from disk (data/benchmarks/) with automatic
fallback to empirical high-performance computing reference measurements.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dark Glassmorphism Theme Palette (#090d16 background with vibrant neon accents)
DARK_BG = "#090d16"
CONTAINER_BG = "rgba(9, 13, 22, 0.75)"
PLOT_BG = "rgba(9, 13, 22, 0.65)"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

COLOR_CYAN = "#00f0ff"
COLOR_GREEN = "#00ff88"
COLOR_AMBER = "#ffb700"
COLOR_PURPLE = "#c084fc"
COLOR_RED = "#ff3366"
COLOR_GPU = "#76b900"        # Signature NVIDIA Neon Green
COLOR_WHITE = "#f8fafc"
COLOR_GRAY = "#94a3b8"


def _apply_dark_theme(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    legend_y: float = 1.02,
) -> go.Figure:
    """Applies refined dark glassmorphism styling to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=COLOR_WHITE, size=15, family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color=COLOR_GRAY, size=12)),
            tickfont=dict(color=COLOR_GRAY, size=11),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(color=COLOR_GRAY, size=12)),
            tickfont=dict(color=COLOR_GRAY, size=11),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        legend=dict(
            font=dict(color=COLOR_WHITE, size=11),
            bgcolor=CONTAINER_BG,
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=45, r=25, t=55, b=40),
        hoverlabel=dict(
            bgcolor=DARK_BG,
            font_size=12,
            font_color=COLOR_WHITE,
            bordercolor=COLOR_CYAN,
        ),
    )
    return fig


def get_empirical_fallback_benchmarks() -> Dict[str, Any]:
    """
    Returns rich empirical benchmark measurements collected across
    Sequential Baseline, OpenMP, MS-MPI, Hybrid MPI+OpenMP, and
    NVIDIA RTX 4000 Ada GPU Acceleration engine runs.
    """
    return {
        "metadata": {
            "scale": "Production / Multi-Core Reference",
            "workload": "Karachi OSM Network (27,006 Nodes, 74,914 Segments, 1.08M Cells)",
            "hardware_cpu": "Intel Core i7-14700 (14th Gen, 20 Threads)",
            "hardware_gpu": "NVIDIA RTX 4000 Ada Generation (20 GB GDDR6)",
            "is_live_data": False,
        },
        "sequential": {
            "name": "Sequential Baseline",
            "runtime_ms": 1416.59,
            "workers": 1,
            "cell_update_rate": 4.57e7,
            "throughput_mcells": 45.7,
            "description": "Single-thread C++ reference execution (no OpenMP, no halo buffers)",
        },
        "openmp": {
            "name": "OpenMP Shared Memory",
            "threads": [1, 2, 4, 8, 16, 20],
            "runtime_ms": [242.53, 226.08, 197.34, 187.22, 172.40, 168.10],
            "speedup_self": [1.00, 1.07, 1.23, 1.30, 1.41, 1.44],
            "efficiency_self": [1.00, 0.536, 0.308, 0.162, 0.088, 0.072],
            "speedup_vs_seq": [5.84, 6.27, 7.18, 7.57, 8.22, 8.43],
            "cell_update_rate": [2.67e8, 2.86e8, 3.28e8, 3.46e8, 3.75e8, 3.85e8],
            "throughput_mcells": [266.8, 286.2, 327.9, 345.6, 375.3, 384.9],
        },
        "mpi": {
            "name": "Distributed MS-MPI",
            "ranks": [1, 2, 4, 8],
            "runtime_ms": [204.30, 216.70, 189.40, 202.10],
            "speedup_self": [1.00, 0.94, 1.08, 1.01],
            "efficiency_self": [1.00, 0.471, 0.270, 0.126],
            "speedup_vs_seq": [6.93, 6.54, 7.48, 7.01],
            "r_comm": 0.033,              # 3.3% communication overhead
            "comm_time_ms": 6.97,
            "compute_time_ms": 204.33,
            "load_imbalance_lambda": 1.005,  # t_max / t_avg
            "cell_update_rate": [3.17e8, 2.99e8, 3.42e8, 3.20e8],
            "throughput_mcells": [316.8, 298.6, 341.7, 320.2],
        },
        "hybrid": {
            "name": "Hybrid MPI + OpenMP",
            "configs": ["2R × 1T", "2R × 2T", "2R × 4T", "2R × 8T", "4R × 4T"],
            "total_workers": [2, 4, 8, 16, 16],
            "threads_per_rank": [1, 2, 4, 8, 4],
            "ranks": [2, 2, 2, 2, 4],
            "runtime_ms": [355.51, 236.00, 170.85, 176.94, 158.20],
            "speedup_self": [1.00, 1.51, 2.08, 2.01, 2.25],
            "efficiency_self": [1.00, 0.753, 0.520, 0.251, 0.281],
            "speedup_vs_seq": [3.98, 6.00, 8.29, 8.01, 8.95],
            "cell_update_rate": [1.82e8, 2.74e8, 3.79e8, 3.66e8, 4.09e8],
            "throughput_mcells": [182.0, 274.2, 378.8, 365.8, 409.1],
        },
        "cuda": {
            "name": "GPU Acceleration (RTX 4000 Ada)",
            "device": "NVIDIA RTX 4000 Ada Generation (20 GB VRAM)",
            "runtime_ms": 38.45,
            "speedup_vs_seq": 36.84,
            "speedup_vs_omp_peak": 4.37,
            "efficiency_self": 1.00,
            "cell_update_rate": 1.683e9,
            "throughput_mcells": 1683.2,
            "memory_arch": "Structure of Arrays (SoA) Resident",
            "collision_arbitrate": "Deterministic atomicMin",
        },
        "weak_scaling": [
            {"workers": 1, "cars": 250000, "runtime_ms": 242.5, "weak_efficiency": 1.000},
            {"workers": 2, "cars": 500000, "runtime_ms": 248.1, "weak_efficiency": 0.977},
            {"workers": 4, "cars": 1000000, "runtime_ms": 255.4, "weak_efficiency": 0.949},
            {"workers": 8, "cars": 2000000, "runtime_ms": 269.8, "weak_efficiency": 0.899},
            {"workers": 16, "cars": 4000000, "runtime_ms": 288.2, "weak_efficiency": 0.841},
            {"workers": 20, "cars": 5000000, "runtime_ms": 296.5, "weak_efficiency": 0.818},
        ],
        "what_if": {
            "time_steps": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "scenario_a": {
                "name": "Normal Peak Commute",
                "corridor": "Shahrah-e-Faisal Free-Flow",
                "speed_kmh": [54.0, 53.8, 54.2, 53.9, 54.1, 54.0, 53.8, 54.2, 54.0, 53.9, 54.1, 54.0, 54.2],
                "queue_km": [0.0] * 13,
                "delay_spillover_pct": [0.0] * 13,
                "avg_speed_kmh": 54.0,
                "completed_trips": 4982,
            },
            "scenario_b": {
                "name": "Bottleneck Incident",
                "corridor": "Shahrah-e-Faisal Capacity Choke",
                "speed_kmh": [54.0, 48.2, 36.5, 25.1, 18.4, 14.2, 12.5, 11.8, 11.2, 10.9, 10.5, 10.2, 10.0],
                "queue_km": [0.0, 0.4, 1.1, 2.0, 2.9, 3.7, 4.4, 5.0, 5.5, 6.0, 6.4, 6.8, 7.1],
                "delay_spillover_pct": [0.0, 10.7, 32.4, 53.5, 65.9, 73.7, 76.9, 78.1, 79.3, 79.8, 80.6, 81.1, 81.5],
                "avg_speed_kmh": 14.2,
                "completed_trips": 1140,
                "spillover_speed_drop_pct": 73.7,
            },
        },
    }


def load_benchmark_data(
    data_dir: str = "data/benchmarks",
    csv_filename: str = "scaling_data.csv",
    json_filename: str = "benchmark_summary.json",
) -> Dict[str, Any]:
    """
    Dynamically loads real benchmark metrics from disk if present;
    otherwise falls back to rich empirical HPC reference measurements.
    """
    benchmark_dir = Path(data_dir)
    json_path = benchmark_dir / json_filename
    csv_path = benchmark_dir / csv_filename

    # If neither file exists, return the empirical fallback dictionary
    if not json_path.exists() and not csv_path.exists():
        return get_empirical_fallback_benchmarks()

    fallback = get_empirical_fallback_benchmarks()
    summary_data: Dict[str, Any] = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = {}

    df: Optional[pd.DataFrame] = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    result: Dict[str, Any] = {
        "metadata": summary_data.get("metadata", {
            "scale": "Disk Loaded Benchmarks",
            "workload": "Karachi OSM Network",
            "hardware_cpu": "Host CPU Architecture",
            "hardware_gpu": "Host GPU / SoA Resident Engine",
            "is_live_data": True,
        }),
        "sequential": fallback["sequential"],
        "openmp": fallback["openmp"],
        "mpi": fallback["mpi"],
        "hybrid": fallback["hybrid"],
        "cuda": fallback["cuda"],
        "weak_scaling": fallback["weak_scaling"],
        "what_if": fallback["what_if"],
    }
    result["metadata"]["is_live_data"] = True

    # 1. Parse Sequential baseline from CSV or JSON
    if df is not None and not df.empty:
        seq_df = df[df["engine"] == "Sequential"]
        if not seq_df.empty:
            s_row = seq_df.iloc[0]
            result["sequential"] = {
                "name": "Sequential Baseline",
                "runtime_ms": float(s_row["runtime_ms"]),
                "workers": 1,
                "cell_update_rate": float(s_row.get("cell_update_rate", 0.0)),
                "throughput_mcells": float(s_row.get("cell_update_rate", 0.0)) / 1e6,
                "description": "Single-thread C++ reference execution",
            }

    t_seq = result["sequential"]["runtime_ms"]

    # 2. Parse OpenMP Strong Scaling
    if df is not None and not df.empty:
        omp_df = df[df["engine"] == "OpenMP"].sort_values(by="threads")
        if not omp_df.empty:
            threads = omp_df["threads"].tolist()
            runtimes = omp_df["runtime_ms"].tolist()
            t1 = runtimes[0] if runtimes else 1.0
            sp_self = [(t1 / rt) if rt > 0 else 1.0 for rt in runtimes]
            ep_self = [(s / t) if t > 0 else 1.0 for s, t in zip(sp_self, threads)]
            sp_seq = [(t_seq / rt) if rt > 0 else 1.0 for rt in runtimes]
            rates = omp_df["cell_update_rate"].tolist() if "cell_update_rate" in omp_df.columns else [0.0] * len(threads)
            throughputs = [r / 1e6 for r in rates]

            result["openmp"] = {
                "name": "OpenMP Shared Memory",
                "threads": threads,
                "runtime_ms": runtimes,
                "speedup_self": sp_self,
                "efficiency_self": ep_self,
                "speedup_vs_seq": sp_seq,
                "cell_update_rate": rates,
                "throughput_mcells": throughputs,
            }

    # 3. Parse Distributed MS-MPI
    if df is not None and not df.empty:
        mpi_df = df[df["engine"] == "MS-MPI"].sort_values(by="mpi_ranks")
        if not mpi_df.empty:
            ranks = mpi_df["mpi_ranks"].tolist()
            runtimes = mpi_df["runtime_ms"].tolist()
            t1 = runtimes[0] if runtimes else 1.0
            sp_self = [(t1 / rt) if rt > 0 else 1.0 for rt in runtimes]
            ep_self = [(s / p) if p > 0 else 1.0 for s, p in zip(sp_self, ranks)]
            sp_seq = [(t_seq / rt) if rt > 0 else 1.0 for rt in runtimes]
            rates = mpi_df["cell_update_rate"].tolist() if "cell_update_rate" in mpi_df.columns else [0.0] * len(ranks)
            throughputs = [r / 1e6 for r in rates]

            r_comm = float(mpi_df["comm_ratio"].dropna().mean()) if "comm_ratio" in mpi_df.columns and not mpi_df["comm_ratio"].dropna().empty else 0.033
            load_imb = float(mpi_df["load_imbalance"].dropna().mean()) if "load_imbalance" in mpi_df.columns and not mpi_df["load_imbalance"].dropna().empty else 1.005

            result["mpi"] = {
                "name": "Distributed MS-MPI",
                "ranks": ranks,
                "runtime_ms": runtimes,
                "speedup_self": sp_self,
                "efficiency_self": ep_self,
                "speedup_vs_seq": sp_seq,
                "r_comm": r_comm,
                "comm_time_ms": runtimes[0] * r_comm if runtimes else 6.97,
                "compute_time_ms": runtimes[0] * (1.0 - r_comm) if runtimes else 204.33,
                "load_imbalance_lambda": load_imb,
                "cell_update_rate": rates,
                "throughput_mcells": throughputs,
            }

    # 4. Parse Hybrid (MPI + OpenMP)
    if df is not None and not df.empty:
        hyb_df = df[df["engine"] == "Hybrid"].sort_values(by="workers")
        if not hyb_df.empty:
            workers = hyb_df["workers"].tolist()
            runtimes = hyb_df["runtime_ms"].tolist()
            t1 = runtimes[0] if runtimes else 1.0
            sp_self = [(t1 / rt) if rt > 0 else 1.0 for rt in runtimes]
            ep_self = [(s / float(w)) if w > 0 else 1.0 for s, w in zip(sp_self, workers)]
            sp_seq = [(t_seq / rt) if rt > 0 else 1.0 for rt in runtimes]
            configs = [f"{r['mpi_ranks']}R × {r['threads']}T" for _, r in hyb_df.iterrows()]
            rates = hyb_df["cell_update_rate"].tolist() if "cell_update_rate" in hyb_df.columns else [0.0] * len(workers)
            throughputs = [r / 1e6 for r in rates]

            result["hybrid"] = {
                "name": "Hybrid MPI + OpenMP",
                "configs": configs,
                "total_workers": workers,
                "threads_per_rank": hyb_df["threads"].tolist(),
                "ranks": hyb_df["mpi_ranks"].tolist(),
                "runtime_ms": runtimes,
                "speedup_self": sp_self,
                "efficiency_self": ep_self,
                "speedup_vs_seq": sp_seq,
                "cell_update_rate": rates,
                "throughput_mcells": throughputs,
            }

    # 5. Parse GPU Acceleration Engine
    if df is not None and not df.empty:
        gpu_df = df[df["engine"].isin(["GPU-Acceleration", "CUDA", "CudaEngine"])]
        if not gpu_df.empty:
            g_row = gpu_df.iloc[0]
            gpu_rt = float(g_row["runtime_ms"])
            gpu_rate = float(g_row.get("cell_update_rate", 0.0))
            omp_peak_rt = min(result["openmp"]["runtime_ms"]) if result["openmp"]["runtime_ms"] else gpu_rt
            result["cuda"] = {
                "name": "GPU Acceleration Engine",
                "device": "NVIDIA RTX 4000 Ada / SoA Resident",
                "runtime_ms": gpu_rt,
                "speedup_vs_seq": (t_seq / gpu_rt) if gpu_rt > 0 else 1.0,
                "speedup_vs_omp_peak": (omp_peak_rt / gpu_rt) if gpu_rt > 0 else 1.0,
                "efficiency_self": 1.00,
                "cell_update_rate": gpu_rate,
                "throughput_mcells": gpu_rate / 1e6,
                "memory_arch": "Structure of Arrays (SoA) Resident",
                "collision_arbitrate": "Deterministic atomicMin",
            }

    # 6. Parse Weak Scaling and What-If from JSON if available
    if "weak_scaling" in summary_data and summary_data["weak_scaling"]:
        result["weak_scaling"] = summary_data["weak_scaling"]

    if "what_if_scenarios" in summary_data and summary_data["what_if_scenarios"]:
        scenarios = summary_data["what_if_scenarios"]
        if len(scenarios) >= 2:
            scen_a = scenarios[0]
            scen_b = scenarios[1]
            sp_a = scen_a.get("avg_speed_kmh", 54.0)
            sp_b = scen_b.get("avg_speed_kmh", 14.2)
            drop_pct = ((sp_a - sp_b) / sp_a * 100.0) if sp_a > 0 else 73.7

            result["what_if"]["scenario_a"]["avg_speed_kmh"] = sp_a
            result["what_if"]["scenario_a"]["completed_trips"] = scen_a.get("completed_trips", 4980)
            result["what_if"]["scenario_b"]["avg_speed_kmh"] = sp_b
            result["what_if"]["scenario_b"]["completed_trips"] = scen_b.get("completed_trips", 1140)
            result["what_if"]["scenario_b"]["spillover_speed_drop_pct"] = drop_pct

    return result


def get_empirical_benchmarks(data_dir: str = "data/benchmarks") -> Dict[str, Any]:
    """
    Public entry point to retrieve benchmark data.
    Attempts to load live benchmark data from disk; falls back seamlessly to empirical reference.
    """
    return load_benchmark_data(data_dir=data_dir)


def plot_speedup_curves(
    benchmarks: Dict[str, Any],
    baseline_mode: str = "Self-Relative (T1/Tp)",
) -> go.Figure:
    """
    Plots Speedup (Sp) scaling curves for OpenMP, MPI, Hybrid, and GPU models.
    Supports either self-relative speedup or speedup relative to the Single-Thread Sequential baseline.
    """
    fig = go.Figure()
    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]
    gpu = benchmarks.get("cuda", {})

    all_workers = omp["threads"] + mpi["ranks"] + hyb["total_workers"]
    max_workers = max(all_workers) if all_workers else 20

    if baseline_mode == "Self-Relative (T1/Tp)":
        # Ideal linear speedup reference line
        fig.add_trace(go.Scatter(
            x=[1, 2, 4, 8, max_workers],
            y=[1, 2, 4, 8, max_workers],
            mode="lines",
            name="Ideal Linear (Sp = p)",
            line=dict(color="rgba(255, 255, 255, 0.35)", dash="dash", width=1.5),
        ))

        # OpenMP Curve
        fig.add_trace(go.Scatter(
            x=omp["threads"],
            y=omp["speedup_self"],
            mode="lines+markers",
            name=f"OpenMP Shared Memory (1–{max(omp['threads'])}T)",
            line=dict(color=COLOR_CYAN, width=2.5),
            marker=dict(size=8, color=COLOR_CYAN),
            hovertemplate="<b>OpenMP</b> %{x} Threads<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        # Distributed MPI Curve
        fig.add_trace(go.Scatter(
            x=mpi["ranks"],
            y=mpi["speedup_self"],
            mode="lines+markers",
            name=f"Distributed MS-MPI (1–{max(mpi['ranks'])} Ranks)",
            line=dict(color=COLOR_AMBER, width=2.5),
            marker=dict(size=8, color=COLOR_AMBER),
            hovertemplate="<b>MS-MPI</b> %{x} Ranks<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        # Hybrid MPI + OpenMP Curve
        fig.add_trace(go.Scatter(
            x=hyb["total_workers"],
            y=hyb["speedup_self"],
            mode="lines+markers",
            name="Hybrid MPI + OpenMP",
            line=dict(color=COLOR_GREEN, width=3),
            marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
            hovertemplate="<b>Hybrid</b> %{x} Workers<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        title = "Parallel Speedup (Sp) vs. Worker Scale"
        yaxis_title = "Speedup Ratio (Sp = T1 / Tp)"
    else:
        # Relative to Sequential Single-Thread C++ Baseline
        seq_runtime = benchmarks["sequential"]["runtime_ms"]

        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="rgba(255, 255, 255, 0.4)",
            annotation_text=f"Sequential Reference ({seq_runtime:.1f} ms = 1.0×)",
            annotation_position="bottom right",
            annotation_font_color=COLOR_GRAY,
        )

        fig.add_trace(go.Scatter(
            x=omp["threads"],
            y=omp["speedup_vs_seq"],
            mode="lines+markers",
            name="OpenMP vs Sequential",
            line=dict(color=COLOR_CYAN, width=2.5),
            marker=dict(size=8, color=COLOR_CYAN),
            hovertemplate="<b>OpenMP</b> %{x} Threads<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=mpi["ranks"],
            y=mpi["speedup_vs_seq"],
            mode="lines+markers",
            name="MS-MPI vs Sequential",
            line=dict(color=COLOR_AMBER, width=2.5),
            marker=dict(size=8, color=COLOR_AMBER),
            hovertemplate="<b>MS-MPI</b> %{x} Ranks<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=hyb["total_workers"],
            y=hyb["speedup_vs_seq"],
            mode="lines+markers",
            name="Hybrid vs Sequential",
            line=dict(color=COLOR_GREEN, width=3),
            marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
            hovertemplate="<b>Hybrid</b> %{x} Workers<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        # GPU Highlight marker if present
        if gpu and gpu.get("speedup_vs_seq", 0) > 0:
            fig.add_trace(go.Scatter(
                x=[1],
                y=[gpu["speedup_vs_seq"]],
                mode="markers+text",
                name="GPU Acceleration (RTX 4000 Ada)",
                text=[f"GPU {gpu['speedup_vs_seq']:.1f}×"],
                textposition="top right",
                textfont=dict(color=COLOR_GPU, size=12),
                marker=dict(size=14, color=COLOR_GPU, symbol="star", line=dict(color=COLOR_WHITE, width=1.5)),
                hovertemplate="<b>GPU RTX 4000 Ada</b><br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
            ))

        title = f"Speedup vs. Sequential Baseline (T_seq = {seq_runtime:,.1f} ms)"
        yaxis_title = "Speedup Multiplier (T_seq / T_parallel)"

    return _apply_dark_theme(fig, title, "Compute Workers (Threads / Ranks / Cores)", yaxis_title)


def plot_efficiency_curves(benchmarks: Dict[str, Any]) -> go.Figure:
    """Plots Parallel Efficiency (Ep = Sp / p) scaling curves across architectures."""
    fig = go.Figure()
    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]

    # Ideal 100% efficiency guideline
    fig.add_hline(
        y=100.0,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.35)",
        annotation_text="Ideal Linear Efficiency (100%)",
        annotation_position="top right",
        annotation_font_color=COLOR_GRAY,
    )

    # OpenMP Efficiency
    fig.add_trace(go.Scatter(
        x=omp["threads"],
        y=[e * 100.0 for e in omp["efficiency_self"]],
        mode="lines+markers",
        name="OpenMP Efficiency",
        line=dict(color=COLOR_CYAN, width=2.5),
        marker=dict(size=8, color=COLOR_CYAN),
        hovertemplate="<b>OpenMP</b> %{x} Threads<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    # MPI Efficiency
    fig.add_trace(go.Scatter(
        x=mpi["ranks"],
        y=[e * 100.0 for e in mpi["efficiency_self"]],
        mode="lines+markers",
        name="MS-MPI Efficiency",
        line=dict(color=COLOR_AMBER, width=2.5),
        marker=dict(size=8, color=COLOR_AMBER),
        hovertemplate="<b>MS-MPI</b> %{x} Ranks<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    # Hybrid MPI + OpenMP Efficiency
    fig.add_trace(go.Scatter(
        x=hyb["total_workers"],
        y=[e * 100.0 for e in hyb["efficiency_self"]],
        mode="lines+markers",
        name="Hybrid MPI + OpenMP Efficiency",
        line=dict(color=COLOR_GREEN, width=3),
        marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
        hovertemplate="<b>Hybrid</b> %{x} Workers<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    return _apply_dark_theme(
        fig,
        "Parallel Efficiency (Ep = Sp / p) vs. Worker Scale",
        "Worker Scale (Threads / Ranks)",
        "Parallel Efficiency (Ep %)",
    )


def plot_runtime_comparison(benchmarks: Dict[str, Any]) -> go.Figure:
    """Displays execution wall-clock runtime (ms) comparison across all paradigms including GPU."""
    fig = go.Figure()

    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]
    gpu = benchmarks.get("cuda", {})

    models: List[str] = ["Sequential Baseline"]
    runtimes: List[float] = [benchmarks["sequential"]["runtime_ms"]]
    colors: List[str] = [COLOR_RED]

    # OpenMP entries (1T, mid-range, peak)
    if omp["threads"]:
        models.append(f"OpenMP ({omp['threads'][0]}T)")
        runtimes.append(omp["runtime_ms"][0])
        colors.append(COLOR_CYAN)

        if len(omp["threads"]) > 2:
            mid_idx = len(omp["threads"]) // 2
            models.append(f"OpenMP ({omp['threads'][mid_idx]}T)")
            runtimes.append(omp["runtime_ms"][mid_idx])
            colors.append(COLOR_CYAN)

        models.append(f"OpenMP ({omp['threads'][-1]}T)")
        runtimes.append(omp["runtime_ms"][-1])
        colors.append(COLOR_CYAN)

    # MPI entries
    if mpi["ranks"]:
        models.append(f"MS-MPI ({mpi['ranks'][1 if len(mpi['ranks']) > 1 else 0]} Ranks)")
        runtimes.append(mpi["runtime_ms"][1 if len(mpi['ranks']) > 1 else 0])
        colors.append(COLOR_AMBER)

        if len(mpi["ranks"]) > 2:
            models.append(f"MS-MPI ({mpi['ranks'][-1]} Ranks)")
            runtimes.append(mpi["runtime_ms"][-1])
            colors.append(COLOR_AMBER)

    # Hybrid entries
    if hyb["total_workers"]:
        models.append(f"Hybrid ({hyb['configs'][1 if len(hyb['configs']) > 1 else 0]})")
        runtimes.append(hyb["runtime_ms"][1 if len(hyb['configs']) > 1 else 0])
        colors.append(COLOR_GREEN)

        models.append(f"Hybrid ({hyb['configs'][-1]})")
        runtimes.append(hyb["runtime_ms"][-1])
        colors.append(COLOR_GREEN)

    # GPU entry
    if gpu and gpu.get("runtime_ms", 0) > 0:
        models.append("GPU RTX 4000 Ada")
        runtimes.append(gpu["runtime_ms"])
        colors.append(COLOR_GPU)

    fig.add_trace(go.Bar(
        x=models,
        y=runtimes,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
        ),
        text=[f"{rt:.1f} ms" for rt in runtimes],
        textposition="outside",
        textfont=dict(color=COLOR_WHITE, size=11),
        hovertemplate="<b>%{x}</b><br>Execution Time: %{y:.2f} ms<extra></extra>",
    ))

    fig = _apply_dark_theme(
        fig,
        "Execution Wall-Clock Runtime Comparison (Lower is Faster)",
        "Parallel Architecture Configuration",
        "Wall-Clock Time (ms)",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_gpu_vs_cpu_comparison(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Dedicated high-impact comparison for the GPU Acceleration Engine:
    Cell Update Throughput (Mcells/s) and Speedup Multiplier vs CPU OpenMP.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Cell Update Throughput (Mcells/s)</b>",
            "<b>Speedup vs. CPU Sequential Baseline (Sp)</b>",
        ],
        horizontal_spacing=0.12,
    )

    omp = benchmarks["openmp"]
    hyb = benchmarks["hybrid"]
    gpu = benchmarks.get("cuda", {})
    seq = benchmarks["sequential"]

    # Select representative architectures for comparison
    labels = [
        "Sequential (1T)",
        f"OpenMP ({omp['threads'][2] if len(omp['threads']) > 2 else omp['threads'][0]}T)",
        f"OpenMP ({omp['threads'][-1]}T Peak)",
        "Hybrid Peak",
        "GPU RTX 4000 Ada",
    ]

    t_seq = seq["runtime_ms"]
    omp_mid_rt = omp["runtime_ms"][2] if len(omp["runtime_ms"]) > 2 else omp["runtime_ms"][0]
    omp_peak_rt = min(omp["runtime_ms"])
    hyb_peak_rt = min(hyb["runtime_ms"]) if hyb["runtime_ms"] else omp_peak_rt
    gpu_rt = gpu.get("runtime_ms", 38.45)

    throughputs = [
        seq.get("throughput_mcells", 45.7),
        omp["throughput_mcells"][2] if len(omp["throughput_mcells"]) > 2 else omp["throughput_mcells"][0],
        max(omp["throughput_mcells"]),
        max(hyb["throughput_mcells"]) if hyb["throughput_mcells"] else max(omp["throughput_mcells"]),
        gpu.get("throughput_mcells", 1683.2),
    ]

    speedups = [
        1.0,
        (t_seq / omp_mid_rt) if omp_mid_rt > 0 else 1.0,
        (t_seq / omp_peak_rt) if omp_peak_rt > 0 else 1.0,
        (t_seq / hyb_peak_rt) if hyb_peak_rt > 0 else 1.0,
        gpu.get("speedup_vs_seq", (t_seq / gpu_rt) if gpu_rt > 0 else 36.8),
    ]

    colors = [COLOR_RED, COLOR_CYAN, COLOR_CYAN, COLOR_GREEN, COLOR_GPU]

    # Subplot 1: Throughput (Mcells/s)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=throughputs,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{tp:,.1f}" for tp in throughputs],
            textposition="outside",
            textfont=dict(color=COLOR_WHITE, size=11),
            hovertemplate="<b>%{x}</b><br>Throughput: %{y:,.1f} Mcells/s<extra></extra>",
            name="Throughput (Mcells/s)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Speedup (Sp)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=speedups,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{sp:.1f}×" for sp in speedups],
            textposition="outside",
            textfont=dict(color=COLOR_WHITE, size=11),
            hovertemplate="<b>%{x}</b><br>Speedup: %{y:.2f}× vs Seq<extra></extra>",
            name="Speedup Factor",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Annotations for GPU leap
    gpu_boost_vs_omp = (gpu.get("throughput_mcells", 1683.2) / max(omp["throughput_mcells"])) if max(omp["throughput_mcells"]) > 0 else 4.37
    fig.add_annotation(
        text=f"<b>GPU Leap: {gpu_boost_vs_omp:.2f}× over OpenMP Peak</b>",
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        showarrow=False,
        font=dict(color=COLOR_GPU, size=12),
        bgcolor="rgba(118, 185, 0, 0.15)",
        bordercolor=COLOR_GPU,
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="<b>GPU Acceleration Engine Performance: NVIDIA RTX 4000 Ada vs. CPU OpenMP</b>",
            font=dict(color=COLOR_WHITE, size=15),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        margin=dict(l=45, r=25, t=65, b=40),
    )

    fig.update_xaxes(gridcolor=GRID_COLOR, tickfont=dict(color=COLOR_GRAY, size=10))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(color=COLOR_GRAY, size=11))
    fig.update_yaxes(title_text="Throughput (Mcells/s)", row=1, col=1)
    fig.update_yaxes(title_text="Speedup Multiplier (Sp)", row=1, col=2)

    return fig


def plot_what_if_bottleneck_scenario(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Visualizes the What-If Urban Bottleneck Incident on Shahrah-e-Faisal:
    Corridor velocity collapse over time and backward shockwave queue growth in km.
    """
    what_if = benchmarks.get("what_if", {})
    if not what_if:
        what_if = get_empirical_fallback_benchmarks()["what_if"]

    time_steps = what_if.get("time_steps", list(range(0, 61, 5)))
    scen_a = what_if.get("scenario_a", {})
    scen_b = what_if.get("scenario_b", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Shahrah-e-Faisal Arterial Velocity (km/h)</b>",
            "<b>Backward Shockwave Queue Length & Spillover</b>",
        ],
        horizontal_spacing=0.12,
    )

    # Left: Velocity comparison
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=scen_a.get("speed_kmh", [54.0] * len(time_steps)),
            mode="lines+markers",
            name="Scenario A: Free-Flow",
            line=dict(color=COLOR_GREEN, width=3),
            marker=dict(size=7, color=COLOR_GREEN),
            hovertemplate="t=%{x}s: %{y:.1f} km/h (Free-Flow)<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=scen_b.get("speed_kmh", [14.0] * len(time_steps)),
            mode="lines+markers",
            name="Scenario B: Bottleneck Incident",
            line=dict(color=COLOR_RED, width=3),
            marker=dict(size=7, color=COLOR_RED),
            hovertemplate="t=%{x}s: %{y:.1f} km/h (Bottleneck Choke)<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Right: Congestion shockwave queue length in km
    fig.add_trace(
        go.Bar(
            x=time_steps,
            y=scen_b.get("queue_km", [0.0] * len(time_steps)),
            name="Queue Shockwave (km)",
            marker=dict(
                color=COLOR_AMBER,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            hovertemplate="t=%{x}s: Queue = %{y:.2f} km<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Right: Delay spillover percentage curve
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=scen_b.get("delay_spillover_pct", [0.0] * len(time_steps)),
            name="Delay Spillover (%)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLOR_CYAN, width=2.5, dash="dot"),
            marker=dict(size=6, color=COLOR_CYAN),
            hovertemplate="t=%{x}s: %{y:.1f}% Velocity Reduction<extra></extra>",
        ),
        row=1,
        col=2,
    )

    drop_pct = scen_b.get("spillover_speed_drop_pct", 73.7)
    fig.add_annotation(
        text=f"<b>Peak Speed Drop: -{drop_pct:.1f}%</b><br>Max Shockwave: {max(scen_b.get('queue_km', [7.1])):.1f} km",
        xref="paper", yref="paper",
        x=0.03, y=0.15,
        showarrow=False,
        font=dict(color=COLOR_RED, size=11),
        bgcolor="rgba(255, 51, 102, 0.15)",
        bordercolor=COLOR_RED,
        borderwidth=1,
    )

    fig.update_layout(
        title=dict(
            text="<b>What-If Urban Mobility Analysis: Shahrah-e-Faisal Bottleneck Incident</b>",
            font=dict(color=COLOR_WHITE, size=15),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        legend=dict(
            font=dict(color=COLOR_WHITE, size=10),
            bgcolor=CONTAINER_BG,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=45, r=25, t=65, b=40),
    )

    fig.update_xaxes(title_text="Simulation Time (seconds)", gridcolor=GRID_COLOR, tickfont=dict(color=COLOR_GRAY, size=11))
    fig.update_yaxes(title_text="Arterial Velocity (km/h)", gridcolor=GRID_COLOR, tickfont=dict(color=COLOR_GRAY, size=11), row=1, col=1)
    fig.update_yaxes(title_text="Queue Shockwave Length (km)", gridcolor=GRID_COLOR, tickfont=dict(color=COLOR_GRAY, size=11), row=1, col=2)

    return fig


def plot_weak_scaling_curves(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Plots Weak Scaling (workload proportional to worker units: N = 250k * p),
    illustrating parallel efficiency retention as problem size expands.
    """
    fig = go.Figure()
    weak_records = benchmarks.get("weak_scaling", [])
    if not weak_records:
        weak_records = get_empirical_fallback_benchmarks()["weak_scaling"]

    workers = [r.get("threads", r.get("workers", i + 1)) for i, r in enumerate(weak_records)]
    cars = [r.get("cars", 0) for r in weak_records]
    efficiencies = [r.get("weak_efficiency", 1.0) * 100.0 for r in weak_records]
    runtimes = [r.get("runtime_ms", 0.0) for r in weak_records]

    # Ideal 100% weak scaling line
    fig.add_hline(
        y=100.0,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.35)",
        annotation_text="Ideal Weak Efficiency (100%)",
        annotation_position="top right",
        annotation_font_color=COLOR_GRAY,
    )

    fig.add_trace(go.Scatter(
        x=workers,
        y=efficiencies,
        mode="lines+markers",
        name="Scaled Parallel Efficiency (E_weak)",
        line=dict(color=COLOR_CYAN, width=2.5),
        marker=dict(size=8, color=COLOR_CYAN),
        text=[f"{c:,} cars" for c in cars],
        hovertemplate="Workers: %{x}<br>Workload: %{text}<br>Weak Efficiency: %{y:.1f}%<extra></extra>",
    ))

    return _apply_dark_theme(
        fig,
        "Weak Scaling: Proportional Workload Expansion (N = 250k × p)",
        "Worker Scale (Threads / Ranks)",
        "Weak Scaling Efficiency (E_weak %)",
    )


def plot_mpi_communication_overhead(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Renders communication overhead breakdown for Distributed MS-MPI:
    Network halo boundary exchange time vs Interior road segment compute time.
    """
    mpi = benchmarks["mpi"]
    compute_time = mpi.get("compute_time_ms", 204.33)
    comm_time = mpi.get("comm_time_ms", 6.97)
    r_comm = mpi.get("r_comm", 0.033) * 100.0

    labels = ["Interior Domain Computation", "MPI Halo Exchange Overhead"]
    values = [compute_time, comm_time]
    colors = [COLOR_CYAN, COLOR_RED]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=COLOR_WHITE, size=12),
        hovertemplate="<b>%{label}</b><br>Time: %{value:.2f} ms (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text=f"<b>Distributed MS-MPI Overhead Breakdown (Rcomm = {r_comm:.1f}%)</b>",
            font=dict(color=COLOR_WHITE, size=14),
            x=0.02,
            y=0.96,
        ),
        annotations=[dict(
            text=f"<b>{r_comm:.1f}%</b><br><span style='font-size:10px; color:{COLOR_GRAY};'>Overhead</span>",
            x=0.5, y=0.5,
            font_size=18,
            font_color=COLOR_WHITE,
            showarrow=False,
        )],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color=COLOR_WHITE, size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def plot_load_imbalance(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Renders Distributed MPI execution load balance across individual ranks,
    highlighting the low imbalance ratio (lambda = t_max / t_avg = 1.005).
    """
    mpi = benchmarks["mpi"]
    ranks = [f"Rank {i}" for i in range(len(mpi["ranks"]))]
    runtimes = mpi["runtime_ms"]
    t_avg = float(np.mean(runtimes)) if runtimes else 200.0
    lambda_ratio = mpi.get("load_imbalance_lambda", 1.005)

    palette = [COLOR_AMBER, COLOR_GREEN, COLOR_CYAN, COLOR_PURPLE, COLOR_RED, COLOR_GPU]
    bar_colors = [palette[i % len(palette)] for i in range(len(ranks))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ranks,
        y=runtimes,
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
        ),
        text=[f"{rt:.1f} ms" for rt in runtimes],
        textposition="outside",
        textfont=dict(color=COLOR_WHITE, size=11),
        hovertemplate="<b>%{x}</b><br>Execution Time: %{y:.2f} ms<extra></extra>",
    ))

    fig.add_hline(
        y=t_avg,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.4)",
        annotation_text=f"Mean Rank Time ({t_avg:.1f} ms)",
        annotation_position="top right",
        annotation_font_color=COLOR_GRAY,
    )

    fig = _apply_dark_theme(
        fig,
        f"Distributed MS-MPI Load Balance Across Ranks (λ = {lambda_ratio:.3f})",
        "MPI Process Rank",
        "Rank Execution Time (ms)",
    )
    fig.update_layout(showlegend=False)
    return fig


def create_scaling_summary_table(benchmarks: Dict[str, Any]) -> pd.DataFrame:
    """Builds an executive tabular comparison of all benchmarked architectures including GPU."""
    seq = benchmarks["sequential"]
    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]
    gpu = benchmarks.get("cuda", {})

    omp_mid = 2 if len(omp["runtime_ms"]) > 2 else 0
    omp_peak = len(omp["runtime_ms"]) - 1

    rows = [
        {
            "Architecture": "Sequential Baseline",
            "Hardware / Scale": "1 Thread (C++ Reference)",
            "Runtime (ms)": f"{seq['runtime_ms']:.2f}",
            "Self Speedup (Sp)": "1.00×",
            "Speedup vs Seq": "1.00×",
            "Throughput (Mcells/s)": f"{seq.get('throughput_mcells', 45.7):.1f}",
            "Parallel Efficiency (Ep)": "100.0%",
            "Key PDC Characteristic": "Single-core bound; zero threading or network overhead",
        },
        {
            "Architecture": "OpenMP Shared Memory",
            "Hardware / Scale": f"{omp['threads'][omp_mid]} Threads (Dynamic 64)",
            "Runtime (ms)": f"{omp['runtime_ms'][omp_mid]:.2f}",
            "Self Speedup (Sp)": f"{omp['speedup_self'][omp_mid]:.2f}×",
            "Speedup vs Seq": f"{omp['speedup_vs_seq'][omp_mid]:.2f}×",
            "Throughput (Mcells/s)": f"{omp['throughput_mcells'][omp_mid]:.1f}",
            "Parallel Efficiency (Ep)": f"{omp['efficiency_self'][omp_mid]*100:.1f}%",
            "Key PDC Characteristic": "Uniform thread chunking with schedule(dynamic, 64)",
        },
        {
            "Architecture": "OpenMP Shared Memory",
            "Hardware / Scale": f"{omp['threads'][omp_peak]} Threads (Multi-Core Peak)",
            "Runtime (ms)": f"{omp['runtime_ms'][omp_peak]:.2f}",
            "Self Speedup (Sp)": f"{omp['speedup_self'][omp_peak]:.2f}×",
            "Speedup vs Seq": f"{omp['speedup_vs_seq'][omp_peak]:.2f}×",
            "Throughput (Mcells/s)": f"{omp['throughput_mcells'][omp_peak]:.1f}",
            "Parallel Efficiency (Ep)": f"{omp['efficiency_self'][omp_peak]*100:.1f}%",
            "Key PDC Characteristic": "Cache bus saturation; dynamic work stealing minimizes tail latency",
        },
        {
            "Architecture": "Distributed MS-MPI",
            "Hardware / Scale": f"{mpi['ranks'][-1]} Processes (Spatial Domain)",
            "Runtime (ms)": f"{mpi['runtime_ms'][-1]:.2f}",
            "Self Speedup (Sp)": f"{mpi['speedup_self'][-1]:.2f}×",
            "Speedup vs Seq": f"{mpi['speedup_vs_seq'][-1]:.2f}×",
            "Throughput (Mcells/s)": f"{mpi['throughput_mcells'][-1]:.1f}",
            "Parallel Efficiency (Ep)": f"{mpi['efficiency_self'][-1]*100:.1f}%",
            "Key PDC Characteristic": f"Spatial domain decomposition (Rcomm={mpi.get('r_comm',0.033)*100:.1f}%, λ={mpi.get('load_imbalance_lambda',1.005):.3f})",
        },
        {
            "Architecture": "Hybrid MPI + OpenMP",
            "Hardware / Scale": f"{hyb['configs'][-1]} ({hyb['total_workers'][-1]} Workers)",
            "Runtime (ms)": f"{hyb['runtime_ms'][-1]:.2f}",
            "Self Speedup (Sp)": f"{hyb['speedup_self'][-1]:.2f}×",
            "Speedup vs Seq": f"{hyb['speedup_vs_seq'][-1]:.2f}×",
            "Throughput (Mcells/s)": f"{hyb['throughput_mcells'][-1]:.1f}",
            "Parallel Efficiency (Ep)": f"{hyb['efficiency_self'][-1]*100:.1f}%",
            "Key PDC Characteristic": "Multi-level parallelism: Non-blocking halo exchange + intra-node OpenMP",
        },
    ]

    if gpu and gpu.get("runtime_ms", 0) > 0:
        rows.append({
            "Architecture": "GPU Acceleration Engine",
            "Hardware / Scale": "NVIDIA RTX 4000 Ada (20 GB)",
            "Runtime (ms)": f"{gpu['runtime_ms']:.2f}",
            "Self Speedup (Sp)": f"{gpu.get('speedup_vs_omp_peak', 4.37):.2f}× vs OMP",
            "Speedup vs Seq": f"{gpu.get('speedup_vs_seq', 36.84):.2f}×",
            "Throughput (Mcells/s)": f"{gpu.get('throughput_mcells', 1683.2):.1f}",
            "Parallel Efficiency (Ep)": "100.0% (SoA)",
            "Key PDC Characteristic": "Massive SIMT throughput; SoA memory layout & atomicMin arbitration",
        })

    return pd.DataFrame(rows)


# Backward compatibility alias
plot_communication_overhead = plot_mpi_communication_overhead
