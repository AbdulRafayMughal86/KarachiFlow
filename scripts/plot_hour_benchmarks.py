#!/usr/bin/env python3
"""
KarachiFlow — Generate Publication-Quality Plots for 3,600-Step & Multi-Tier Benchmarks
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = Path("data/benchmarks/hour_3600s/summary.json")
    out_dir = Path("data/benchmarks/hour_3600s")
    if not json_path.exists():
        print(f"[ERROR] {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Plot: Speedup Curves for 3,600-Step 1M Car Workload (OpenMP vs MPI vs Hybrid)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    
    omp_workers = [1, 8, 16, 20]
    omp_speedups = [1.0, 1.597, 2.171, 2.998]

    mpi_workers = [1, 2, 4, 8]
    mpi_speedups = [1.0, 1.008, 1.356, 2.173]

    hyb_workers = [1, 16, 16]
    hyb_speedups = [1.0, 1.558, 1.753] # 2Rx8T, 4Rx4T

    ax.plot([1, 20], [1, 20], "k--", alpha=0.4, label="Ideal Linear Speedup ($S_p = p$)")
    ax.plot(omp_workers, omp_speedups, "o-", color="#1f77b4", linewidth=2.2, markersize=7, label="OpenMP (Intel 14th Gen P/E)")
    ax.plot(mpi_workers, mpi_speedups, "s-", color="#2ca02c", linewidth=2.2, markersize=7, label="MS-MPI (Domain Decomposition)")
    ax.plot([1, 16], [1.0, 1.753], "^-", color="#ff7f0e", linewidth=2.2, markersize=7, label="Hybrid MPI+OpenMP (4R x 4T)")

    ax.set_title("KarachiFlow: 1-Simulated-Hour (3,600 Steps) Parallel Strong Scaling (1M Cars)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Compute Workers / Hardware Threads ($p$)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Speedup vs. Single-Threaded ($S_p = T_1 / T_p$)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 4.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "hour_parallel_speedup.png")
    plt.close(fig)
    print(f"[SAVED] {out_dir / 'hour_parallel_speedup.png'}")

    # 2. Plot: RTX 4000 Ada GPU Throughput Across Vehicle Counts (1M to 10M Cars)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    gpu_labels = ["1M Cars\n(Tier 2)", "2M Cars\n(Tier 2)", "3M Cars\n(Tier 2)", "3M Cars\n(Tier 3)", "5M Cars\n(Tier 3)", "10M Stress\n(Tier 3)"]
    gpu_throughput = [7.89, 5.07, 3.94, 5.44, 4.06, 1.38] # Billions cells/s
    gpu_times = [2.71, 4.21, 5.43, 6.02, 8.06, 23.66] # Seconds

    bars = ax.bar(gpu_labels, gpu_throughput, color="#2ca02c", alpha=0.85, edgecolor="#1b611b", width=0.55)
    ax.set_title("NVIDIA RTX 4000 Ada (20GB): 1-Hour Simulation Throughput (3,600 Steps)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Throughput (Billion Cell Updates / sec)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, t in zip(bars, gpu_times):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.2f} Gcells/s\n({t:.1f}s total)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "hour_gpu_throughput.png")
    plt.close(fig)
    print(f"[SAVED] {out_dir / 'hour_gpu_throughput.png'}")

    # 3. Plot: Comparative Wall-Clock Execution Time (Sequential vs OpenMP vs MPI vs CUDA at 1M Cars 3600 Steps)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    engines = ["Sequential (1T)", "MS-MPI (4 Ranks)", "OpenMP (8T)", "Hybrid (4Rx4T)", "OpenMP (20T)", "RTX 4000 Ada (GPU)"]
    times_sec = [68.27, 50.34, 42.74, 38.96, 22.78, 2.71]
    colors = ["#d62728", "#9467bd", "#1f77b4", "#ff7f0e", "#17becf", "#2ca02c"]

    bars = ax.barh(engines, times_sec, color=colors, alpha=0.85, edgecolor="#333333", height=0.55)
    ax.set_title("Execution Wall-Clock Runtime: 1-Simulated-Hour (1M Cars, 3600 Steps)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Execution Time (Seconds) — Lower is Better", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, times_sec):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}s ({68.27/val:.1f}x)", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 80)
    plt.tight_layout()
    fig.savefig(out_dir / "hour_engine_runtime_comparison.png")
    plt.close(fig)
    print(f"[SAVED] {out_dir / 'hour_engine_runtime_comparison.png'}")

if __name__ == "__main__":
    main()
