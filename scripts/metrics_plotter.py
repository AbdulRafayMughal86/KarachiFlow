"""
KarachiFlow — Parallel Computing Performance & Scaling Metrics Plotter
=======================================================================
Generates interactive Plotly figures for analyzing Speedup (Sp),
Parallel Efficiency (Ep), Communication Overhead (Rcomm), and Load Imbalance
across OpenMP, Distributed MPI, and Hybrid MPI+OpenMP configurations.
"""

from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Theme palette matching the dashboard's dark neon aesthetic
DARK_BG = "rgba(15, 23, 42, 0.6)"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
COLOR_CYAN = "#00d2ff"
COLOR_GREEN = "#00ff88"
COLOR_AMBER = "#ffb700"
COLOR_PURPLE = "#c084fc"
COLOR_RED = "#ff4466"
COLOR_WHITE = "#f8fafc"
COLOR_GRAY = "#94a3b8"


def get_empirical_benchmarks() -> Dict[str, Any]:
    """
    Returns empirical benchmark measurements collected across
    Sequential Baseline, OpenMP, MS-MPI, and Hybrid MPI+OpenMP runs.
    Workload: 5,000 vehicles, 60 seconds duration, Karachi road graph.
    """
    return {
        "sequential": {
            "name": "Sequential Baseline",
            "runtime_ms": 1416.59,
            "workers": 1,
            "description": "Single-thread C++ reference execution (no threading, no halo buffers)",
        },
        "openmp": {
            "name": "OpenMP Shared Memory",
            "threads": [1, 2, 4, 8],
            "runtime_ms": [242.53, 226.08, 197.34, 187.22],
            "speedup_self": [1.00, 1.07, 1.23, 1.30],
            "efficiency_self": [1.00, 0.536, 0.308, 0.162],
            "speedup_vs_seq": [5.84, 6.27, 7.18, 7.57],
        },
        "mpi": {
            "name": "Distributed MS-MPI",
            "ranks": [1, 2, 3, 4],
            "runtime_ms": [204.30, 216.70, 189.40, 211.30],
            "speedup_self": [1.00, 0.94, 1.08, 0.97],
            "efficiency_self": [1.00, 0.471, 0.360, 0.242],
            "speedup_vs_seq": [6.93, 6.54, 7.48, 6.70],
            "r_comm": 0.033,            # 3.3% communication overhead ratio
            "comm_time_ms": 6.97,
            "compute_time_ms": 204.33,
            "load_imbalance_lambda": 1.005, # t_max / t_avg
        },
        "hybrid": {
            "name": "Hybrid MPI + OpenMP",
            "configs": ["2R × 1T", "2R × 2T", "2R × 4T", "2R × 8T"],
            "total_workers": [2, 4, 8, 16],
            "threads_per_rank": [1, 2, 4, 8],
            "ranks": 2,
            "runtime_ms": [355.51, 236.00, 170.85, 176.94],
            "speedup_self": [1.00, 1.51, 2.08, 2.01],
            "efficiency_self": [1.00, 0.753, 0.520, 0.251],
            "speedup_vs_seq": [3.98, 6.00, 8.29, 8.01],
        },
    }


def _apply_dark_theme(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str) -> go.Figure:
    """Applies clean dark glassmorphism styling to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=COLOR_WHITE, size=15),
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
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color=COLOR_WHITE, size=11),
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=45, r=25, t=55, b=40),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_color="#f8fafc",
            bordercolor="#38bdf8",
        ),
    )
    return fig


def plot_speedup_curves(benchmarks: Dict[str, Any], baseline_mode: str = "Self-Relative (T1/Tp)") -> go.Figure:
    """
    Plots Speedup (Sp) scaling curves for OpenMP, MPI, and Hybrid models.
    Supports either self-relative speedup or speedup relative to the Single-Thread Sequential baseline.
    """
    fig = go.Figure()
    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]

    if baseline_mode == "Self-Relative (T1/Tp)":
        # Ideal linear speedup reference line
        max_workers = 16
        fig.add_trace(go.Scatter(
            x=[1, 2, 4, 8, 16],
            y=[1, 2, 4, 8, 16],
            mode="lines",
            name="Ideal Linear (Sp = p)",
            line=dict(color="rgba(255, 255, 255, 0.35)", dash="dash", width=1.5),
        ))

        # OpenMP Curve
        fig.add_trace(go.Scatter(
            x=omp["threads"],
            y=omp["speedup_self"],
            mode="lines+markers",
            name="OpenMP Threads (1–8)",
            line=dict(color=COLOR_CYAN, width=2.5),
            marker=dict(size=8, color=COLOR_CYAN),
            hovertemplate="OpenMP %{x} Threads<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        # Distributed MPI Curve
        fig.add_trace(go.Scatter(
            x=mpi["ranks"],
            y=mpi["speedup_self"],
            mode="lines+markers",
            name="Distributed MPI (1–4 Ranks)",
            line=dict(color=COLOR_AMBER, width=2.5),
            marker=dict(size=8, color=COLOR_AMBER),
            hovertemplate="MPI %{x} Ranks<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        # Hybrid MPI + OpenMP Curve
        fig.add_trace(go.Scatter(
            x=hyb["total_workers"],
            y=hyb["speedup_self"],
            mode="lines+markers",
            name="Hybrid 2-Rank MPI + OpenMP",
            line=dict(color=COLOR_GREEN, width=3),
            marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
            hovertemplate="Hybrid %{x} Total Workers<br>Speedup: %{y:.2f}×<extra></extra>",
        ))

        title = "Parallel Speedup ($S_p$) vs. Worker Count"
        yaxis_title = "Speedup Ratio ($S_p = T_1 / T_p$)"
    else:
        # Relative to Sequential Single-Thread C++ Baseline (T1 = 1416.59 ms)
        seq_runtime = benchmarks["sequential"]["runtime_ms"]

        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="rgba(255, 255, 255, 0.4)",
            annotation_text=f"Sequential Baseline ({seq_runtime:.1f} ms = 1.0×)",
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
            hovertemplate="OpenMP %{x} Threads<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=mpi["ranks"],
            y=mpi["speedup_vs_seq"],
            mode="lines+markers",
            name="MPI vs Sequential",
            line=dict(color=COLOR_AMBER, width=2.5),
            marker=dict(size=8, color=COLOR_AMBER),
            hovertemplate="MPI %{x} Ranks<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=hyb["total_workers"],
            y=hyb["speedup_vs_seq"],
            mode="lines+markers",
            name="Hybrid MPI+OpenMP vs Sequential",
            line=dict(color=COLOR_GREEN, width=3),
            marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
            hovertemplate="Hybrid %{x} Workers<br>Speedup vs Seq: %{y:.2f}×<extra></extra>",
        ))

        title = "Speedup vs. Sequential Baseline ($T_{seq} = 1,416.59$ ms)"
        yaxis_title = "Speedup Multiplier ($T_{seq} / T_{parallel}$)"

    return _apply_dark_theme(fig, title, "Worker Units (Threads / Ranks / Cores)", yaxis_title)


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
        hovertemplate="OpenMP %{x} Threads<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    # MPI Efficiency
    fig.add_trace(go.Scatter(
        x=mpi["ranks"],
        y=[e * 100.0 for e in mpi["efficiency_self"]],
        mode="lines+markers",
        name="MPI Efficiency",
        line=dict(color=COLOR_AMBER, width=2.5),
        marker=dict(size=8, color=COLOR_AMBER),
        hovertemplate="MPI %{x} Ranks<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    # Hybrid MPI + OpenMP Efficiency
    fig.add_trace(go.Scatter(
        x=hyb["total_workers"],
        y=[e * 100.0 for e in hyb["efficiency_self"]],
        mode="lines+markers",
        name="Hybrid MPI + OpenMP Efficiency",
        line=dict(color=COLOR_GREEN, width=3),
        marker=dict(size=9, color=COLOR_GREEN, symbol="diamond"),
        hovertemplate="Hybrid %{x} Workers<br>Efficiency: %{y:.1f}%<extra></extra>",
    ))

    return _apply_dark_theme(
        fig,
        "Parallel Efficiency ($E_p = S_p / p$) vs. Worker Scale",
        "Worker Units (Threads / Ranks)",
        "Efficiency Percentage ($E_p$ %)"
    )


def plot_runtime_comparison(benchmarks: Dict[str, Any]) -> go.Figure:
    """Displays execution runtime (ms) comparison across all paradigms."""
    fig = go.Figure()

    models = [
        "Sequential Baseline",
        "OpenMP (1T)",
        "OpenMP (4T)",
        "OpenMP (8T)",
        "MS-MPI (2 Ranks)",
        "MS-MPI (3 Ranks)",
        "Hybrid (2R × 2T)",
        "Hybrid (2R × 4T)",
    ]

    runtimes = [
        benchmarks["sequential"]["runtime_ms"],
        benchmarks["openmp"]["runtime_ms"][0],
        benchmarks["openmp"]["runtime_ms"][2],
        benchmarks["openmp"]["runtime_ms"][3],
        benchmarks["mpi"]["runtime_ms"][1],
        benchmarks["mpi"]["runtime_ms"][2],
        benchmarks["hybrid"]["runtime_ms"][1],
        benchmarks["hybrid"]["runtime_ms"][2],
    ]

    colors = [
        COLOR_RED,
        COLOR_CYAN,
        COLOR_CYAN,
        COLOR_CYAN,
        COLOR_AMBER,
        COLOR_AMBER,
        COLOR_GREEN,
        COLOR_GREEN,
    ]

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
        hovertemplate="<b>%{x}</b><br>Runtime: %{y:.2f} ms<extra></extra>",
    ))

    fig = _apply_dark_theme(
        fig,
        "Execution Wall-Clock Runtime Comparison (Lower is Faster)",
        "Parallel Architecture Configuration",
        "Wall-Clock Time (ms)"
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_mpi_communication_overhead(benchmarks: Dict[str, Any]) -> go.Figure:
    """
    Renders communication overhead breakdown for Distributed MPI:
    Network halo boundary exchange time vs Interior road segment compute time.
    """
    mpi = benchmarks["mpi"]
    compute_time = mpi["compute_time_ms"]
    comm_time = mpi["comm_time_ms"]
    r_comm = mpi["r_comm"] * 100.0

    labels = ["Interior Domain Computation", "MPI Halo Exchange Overhead"]
    values = [compute_time, comm_time]
    colors = [COLOR_CYAN, COLOR_RED]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(color=COLOR_WHITE, size=12),
        hovertemplate="<b>%{label}</b><br>Time: %{value:.2f} ms (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text=f"<b>Distributed MPI Overhead Breakdown (Rcomm = {r_comm:.1f}%)</b>",
            font=dict(color=COLOR_WHITE, size=14),
            x=0.02,
            y=0.96,
        ),
        annotations=[dict(
            text=f"<b>{r_comm:.1f}%</b><br><span style='font-size:10px; color:#94a3b8;'>Overhead</span>",
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
    ranks = ["Rank 0", "Rank 1", "Rank 2", "Rank 3"]
    runtimes = mpi["runtime_ms"]
    t_avg = float(np.mean(runtimes))
    lambda_ratio = mpi["load_imbalance_lambda"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ranks,
        y=runtimes,
        marker=dict(
            color=[COLOR_AMBER, COLOR_GREEN, COLOR_CYAN, COLOR_PURPLE],
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
        f"Distributed MPI Load Balance Across Ranks (λ = {lambda_ratio:.3f})",
        "MPI Process Rank",
        "Rank Execution Time (ms)",
    )
    fig.update_layout(showlegend=False)
    return fig


def create_scaling_summary_table(benchmarks: Dict[str, Any]) -> pd.DataFrame:
    """Builds an executive tabular comparison of all benchmarked architectures."""
    seq = benchmarks["sequential"]
    omp = benchmarks["openmp"]
    mpi = benchmarks["mpi"]
    hyb = benchmarks["hybrid"]

    rows = [
        {
            "Architecture": "Sequential Baseline",
            "Config / Scale": "1 Thread (C++ reference)",
            "Runtime (ms)": f"{seq['runtime_ms']:.2f}",
            "Self Speedup (Sp)": "1.00×",
            "Efficiency (Ep)": "100.0%",
            "Speedup vs Seq": "1.00×",
            "Key Characteristic": "Zero parallel overhead, CPU single-core bound",
        },
        {
            "Architecture": "OpenMP Shared Memory",
            "Config / Scale": "4 Threads (OMP)",
            "Runtime (ms)": f"{omp['runtime_ms'][2]:.2f}",
            "Self Speedup (Sp)": f"{omp['speedup_self'][2]:.2f}×",
            "Efficiency (Ep)": f"{omp['efficiency_self'][2]*100:.1f}%",
            "Speedup vs Seq": f"{omp['speedup_vs_seq'][2]:.2f}×",
            "Key Characteristic": "Uniform thread chunking with schedule(static)",
        },
        {
            "Architecture": "OpenMP Shared Memory",
            "Config / Scale": "8 Threads (OMP)",
            "Runtime (ms)": f"{omp['runtime_ms'][3]:.2f}",
            "Self Speedup (Sp)": f"{omp['speedup_self'][3]:.2f}×",
            "Efficiency (Ep)": f"{omp['efficiency_self'][3]*100:.1f}%",
            "Speedup vs Seq": f"{omp['speedup_vs_seq'][3]:.2f}×",
            "Key Characteristic": "Cache bandwidth saturation on memory bus",
        },
        {
            "Architecture": "Distributed MS-MPI",
            "Config / Scale": "3 Ranks (MPI)",
            "Runtime (ms)": f"{mpi['runtime_ms'][2]:.2f}",
            "Self Speedup (Sp)": f"{mpi['speedup_self'][2]:.2f}×",
            "Efficiency (Ep)": f"{mpi['efficiency_self'][2]*100:.1f}%",
            "Speedup vs Seq": f"{mpi['speedup_vs_seq'][2]:.2f}×",
            "Key Characteristic": "Domain decomposition (Rcomm = 3.3%, λ = 1.005)",
        },
        {
            "Architecture": "Hybrid MPI + OpenMP",
            "Config / Scale": "2 Ranks × 4 Threads",
            "Runtime (ms)": f"{hyb['runtime_ms'][2]:.2f}",
            "Self Speedup (Sp)": f"{hyb['speedup_self'][2]:.2f}×",
            "Efficiency (Ep)": f"{hyb['efficiency_self'][2]*100:.1f}%",
            "Speedup vs Seq": f"{hyb['speedup_vs_seq'][2]:.2f}×",
            "Key Characteristic": "Peak throughput: 8.29× speedup over sequential",
        },
        {
            "Architecture": "Hybrid MPI + OpenMP",
            "Config / Scale": "2 Ranks × 8 Threads",
            "Runtime (ms)": f"{hyb['runtime_ms'][3]:.2f}",
            "Self Speedup (Sp)": f"{hyb['speedup_self'][3]:.2f}×",
            "Efficiency (Ep)": f"{hyb['efficiency_self'][3]*100:.1f}%",
            "Speedup vs Seq": f"{hyb['speedup_vs_seq'][3]:.2f}×",
            "Key Characteristic": "Hyper-threading tail (8.01× speedup over seq)",
        },
    ]

    return pd.DataFrame(rows)
