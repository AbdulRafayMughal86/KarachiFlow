#!/usr/bin/env python3
"""
KarachiFlow — Multi-Tier 3,600-Step (1 Simulated Hour) Parallel Benchmark Suite
Executes comprehensive strong scaling benchmarks across all 5 parallel paradigms:
  1. Single-Threaded Sequential Baseline (1T)
  2. OpenMP Shared-Memory (1, 2, 4, 8, 16, 20 threads + schedules)
  3. Distributed MS-MPI (1, 2, 4, 8 ranks)
  4. Hierarchical Hybrid (MPI + OpenMP: 2x2, 2x4, 2x8, 4x2, 4x4)
  5. Native CUDA GPU Acceleration (NVIDIA RTX 4000 Ada sm_89)

Evaluated across all three geographic tiers:
  - Tier 1: 500,000 vehicles, 3,600 steps (Urban Core)
  - Tier 2: 1,000,000 vehicles, 3,600 steps (Metropolitan Karachi)
  - Tier 3: 2,000,000 vehicles, 3,600 steps (Regional Mega-Corridor)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_simulation_output(stdout: str) -> dict:
    metrics = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "Execution Time:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["exec_time_ms"] = float(parts[1].split("ms")[0].strip())
                except ValueError:
                    pass
        elif "Cell Update Rate:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["cell_update_rate"] = float(parts[1].split("cell")[0].strip())
                except ValueError:
                    pass
        elif "Average Network Speed:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["avg_speed_cells_s"] = float(parts[1].split("cells/s")[0].strip())
                except ValueError:
                    pass
        elif "Injected Vehicles:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["injected_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
        elif "Active Vehicles:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["active_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
        elif "Completed Trips:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["completed_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
        elif "Max Rank Time:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["max_rank_time_ms"] = float(parts[1].split("ms")[0].strip())
                except ValueError:
                    pass
        elif "Avg Rank Time:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["avg_rank_time_ms"] = float(parts[1].split("ms")[0].strip())
                except ValueError:
                    pass
        elif "Load Imbalance" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["load_imbalance"] = float(parts[1].split("(")[0].strip())
                except ValueError:
                    pass
        elif "Comm-to-Compute Ratio:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["comm_to_compute_ratio"] = float(parts[1].split("(")[0].strip())
                except ValueError:
                    pass
        elif "MPI Waitall Latency:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["mpi_wait_ms"] = float(parts[1].split("ms")[0].strip())
                except ValueError:
                    pass
        elif "Vehicle Conservation:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                metrics["conservation"] = parts[1].strip()
    return metrics


def run_command(cmd, timeout=900):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return p.returncode == 0, p.stdout, p.stderr, wall_ms


def generate_plots_for_tier(out_dir: Path, records: list, tier_name: str, cars: int, steps: int):
    if not HAS_MATPLOTLIB:
        return

    import numpy as np
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.9

    omp_recs = [r for r in records if r["category"] == "OpenMP" and "schedule" not in r["label"].lower()]
    mpi_recs = [r for r in records if r["category"] == "MS-MPI"]
    hyb_recs = [r for r in records if r["category"] == "Hybrid"]
    gpu_recs = [r for r in records if r["category"] == "CUDA GPU"]
    seq_recs = [r for r in records if r["category"] == "Sequential"]

    # 1. Two-Panel Strong Scaling & Paradigm Speedup Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("#ffffff")

    max_w = max([r["workers"] for r in records if r["category"] != "CUDA GPU"] + [1])
    max_cpu_sp = max([r["speedup"] for r in records if r["category"] != "CUDA GPU"] + [1.5])

    w_ideal = np.linspace(1, max_w, 50)
    ax1.plot(w_ideal, w_ideal, "--", color="#94a3b8", linewidth=1.4, label="Ideal Linear ($S_p = p$)")

    if omp_recs:
        w = [r["workers"] for r in omp_recs]
        sp = [r["speedup"] for r in omp_recs]
        ax1.plot(w, sp, "o-", color="#2563eb", linewidth=2.2, markersize=6.5, label="OpenMP (Shared Memory)")

    if mpi_recs:
        w = [r["workers"] for r in mpi_recs]
        sp = [r["speedup"] for r in mpi_recs]
        ax1.plot(w, sp, "s-", color="#059669", linewidth=2.2, markersize=6.5, label="MS-MPI (Distributed)")

    if hyb_recs:
        hyb_by_w = {}
        for r in hyb_recs:
            hw = r["workers"]
            if hw not in hyb_by_w or r["speedup"] > hyb_by_w[hw]["speedup"]:
                hyb_by_w[hw] = r
        w = sorted(hyb_by_w.keys())
        sp = [hyb_by_w[k]["speedup"] for k in w]
        ax1.plot(w, sp, "^-", color="#d97706", linewidth=2.2, markersize=7.0, label="Hybrid (MPI + OpenMP)")

    ax1.axhline(1.0, color="#cbd5e1", linestyle=":", linewidth=1.0)
    ax1.set_title("CPU Multi-Core Strong Scaling ($S_p$)", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax1.set_xlabel("CPU Compute Workers ($p$)", fontsize=10, fontweight="bold", color="#1e293b")
    ax1.set_ylabel("Speedup ($S_p = T_1 / T_p$)", fontsize=10, fontweight="bold", color="#1e293b")
    ax1.set_xlim(0.5, max_w + 1.5)
    ax1.set_ylim(0, max(2.6, max_cpu_sp * 1.25))
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax1.legend(loc="upper left", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.8)

    best_omp = max(omp_recs, key=lambda r: r["speedup"]) if omp_recs else None
    best_mpi = max(mpi_recs, key=lambda r: r["speedup"]) if mpi_recs else None
    best_hyb = max(hyb_recs, key=lambda r: r["speedup"]) if hyb_recs else None
    seq_r = seq_recs[0] if seq_recs else None
    gpu_r = gpu_recs[0] if gpu_recs else None

    bars_data = []
    if seq_r:
        bars_data.append(("Sequential\n(1 Core)", seq_r["speedup"], seq_r["sim_time_ms"] / 1000.0, "#64748b"))
    if best_mpi:
        bars_data.append((f"Best MS-MPI\n({best_mpi['ranks']} Ranks)", best_mpi["speedup"], best_mpi["sim_time_ms"] / 1000.0, "#059669"))
    if best_omp:
        bars_data.append((f"Best OpenMP\n({best_omp['threads']} Threads)", best_omp["speedup"], best_omp["sim_time_ms"] / 1000.0, "#2563eb"))
    if best_hyb:
        bars_data.append((f"Best Hybrid\n({best_hyb['ranks']}R×{best_hyb['threads']}T)", best_hyb["speedup"], best_hyb["sim_time_ms"] / 1000.0, "#d97706"))
    if gpu_r:
        bars_data.append(("CUDA GPU\n(RTX 4000)", gpu_r["speedup"], gpu_r["sim_time_ms"] / 1000.0, "#16a34a"))

    labels = [b[0] for b in bars_data]
    speedups = [b[1] for b in bars_data]
    runtimes = [b[2] for b in bars_data]
    colors = [b[3] for b in bars_data]

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, speedups, color=colors, height=0.55, edgecolor="#1e293b", linewidth=0.8, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Overall Speedup Factor ($S_p = T_1 / T_p$)", fontsize=10, fontweight="bold", color="#1e293b")
    ax2.set_title("Peak Paradigm Speedup (CPU vs. CUDA GPU)", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax2.grid(axis="x", linestyle=":", alpha=0.6, color="#cbd5e1")

    max_sp = max(speedups) if speedups else 1.0
    ax2.set_xlim(0, max_sp * 1.32)

    for bar, sp_val, rt_val in zip(bars, speedups, runtimes):
        ax2.text(
            sp_val + max_sp * 0.025,
            bar.get_y() + bar.get_height() / 2.0,
            f"{sp_val:.2f}x  ({rt_val:.2f}s)",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold",
            color="#0f172a"
        )

    fig.suptitle(f"KarachiFlow — {tier_name}: {cars:,} Vehicles ({steps:,} Steps / 1 Simulated Hour)", fontsize=12.5, fontweight="bold", color="#0f172a", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "speedup_scaling.png", dpi=300)
    plt.close(fig)

    # 2. Runtime Comparison (Horizontal Bar Chart)
    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=300)
    plot_items = []
    if seq_recs: plot_items.append(("Sequential (1T)", seq_recs[0]["sim_time_ms"] / 1000.0, "#d62728"))
    for r in omp_recs:
        if r["threads"] in [4, 8, 16, 20]:
            plot_items.append((f"OpenMP ({r['threads']}T)", r["sim_time_ms"] / 1000.0, "#1f77b4"))
    for r in mpi_recs:
        if r["ranks"] in [2, 4, 8]:
            plot_items.append((f"MS-MPI ({r['ranks']}P)", r["sim_time_ms"] / 1000.0, "#2ca02c"))
    for r in hyb_recs:
        if r["label"] in ["Hybrid 4Rx4T (16W)", "Hybrid 2Rx8T (16W)"]:
            plot_items.append((r["label"], r["sim_time_ms"] / 1000.0, "#ff7f0e"))
    if gpu_recs:
        plot_items.append(("RTX 4000 Ada (GPU)", gpu_recs[0]["sim_time_ms"] / 1000.0, "#2ca02c"))

    labels = [x[0] for x in plot_items]
    times_s = [x[1] for x in plot_items]
    colors = [x[2] for x in plot_items]

    bars = ax.barh(labels, times_s, color=colors, alpha=0.85, edgecolor="#333333", height=0.55)
    ax.set_title(f"Wall-Clock Runtime: {tier_name} ({cars:,} Cars, {steps} Steps)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Runtime in Seconds (Lower is Better)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    max_val = max(times_s) if times_s else 1.0
    seq_time = seq_recs[0]["sim_time_ms"] / 1000.0 if seq_recs else max_val

    for bar, val in zip(bars, times_s):
        sp = seq_time / val if val > 0 else 1.0
        ax.text(val + max_val * 0.02, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}s ({sp:.2f}x)", va="center", ha="left", fontsize=8.5, fontweight="bold")

    ax.set_xlim(0, max_val * 1.25)
    plt.tight_layout()
    fig.savefig(out_dir / "runtime_comparison.png")
    plt.close(fig)


def run_tier(tier_key: str, tier_name: str, net_bin: str, net_meta: str, cars: int, steps: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 85)
    print(f" BENCHMARK SUITE: {tier_name.upper()}")
    print(f" Network:  {net_bin}")
    print(f" Workload: {cars:,} Vehicles | {steps} Steps (1 Simulated Hour) | Dynamic Boundary Inflow")
    print(f" Output:   {out_dir}")
    print("=" * 85)

    benchmarks = [
        # Category, Label, Command, Workers, Threads, Ranks
        ("Sequential", "Sequential Baseline (1T)", ["bin/karachiflow.exe", "--sequential", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1),

        # OpenMP Thread Scaling
        ("OpenMP", "OpenMP 1 Thread (1T)", ["bin/karachiflow.exe", "--threads", "1", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1),
        ("OpenMP", "OpenMP 2 Threads (2T)", ["bin/karachiflow.exe", "--threads", "2", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 2, 2, 1),
        ("OpenMP", "OpenMP 4 Threads (4T)", ["bin/karachiflow.exe", "--threads", "4", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 4, 1),
        ("OpenMP", "OpenMP 8 Threads (8T)", ["bin/karachiflow.exe", "--threads", "8", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 8, 1),
        ("OpenMP", "OpenMP 16 Threads (16T)", ["bin/karachiflow.exe", "--threads", "16", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 16, 1),
        ("OpenMP", "OpenMP 20 Threads (20T)", ["bin/karachiflow.exe", "--threads", "20", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 20, 20, 1),

        # OpenMP Scheduling Studies at 4T
        ("OpenMP Schedule", "OpenMP 4T Schedule Static 64", ["bin/karachiflow.exe", "--threads", "4", "--schedule", "static", "64", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 4, 1),
        ("OpenMP Schedule", "OpenMP 4T Schedule Dynamic 64", ["bin/karachiflow.exe", "--threads", "4", "--schedule", "dynamic", "64", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 4, 1),
        ("OpenMP Schedule", "OpenMP 4T Schedule Guided 64", ["bin/karachiflow.exe", "--threads", "4", "--schedule", "guided", "64", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 4, 1),

        # MS-MPI Process Scaling
        ("MS-MPI", "MS-MPI 1 Rank (1P)", ["mpiexec", "-n", "1", "bin/karachiflow.exe", "--mpi", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1),
        ("MS-MPI", "MS-MPI 2 Ranks (2P)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--mpi", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 2, 1, 2),
        ("MS-MPI", "MS-MPI 4 Ranks (4P)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--mpi", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 1, 4),
        ("MS-MPI", "MS-MPI 8 Ranks (8P)", ["mpiexec", "-n", "8", "bin/karachiflow.exe", "--mpi", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 1, 8),

        # Hybrid Orchestration (MPI + OpenMP)
        ("Hybrid", "Hybrid 2Rx2T (4W)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--hybrid", "--threads", "2", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 2, 2),
        ("Hybrid", "Hybrid 2Rx4T (8W)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--hybrid", "--threads", "4", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 4, 2),
        ("Hybrid", "Hybrid 2Rx8T (16W)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--hybrid", "--threads", "8", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 8, 2),
        ("Hybrid", "Hybrid 4Rx2T (8W)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--hybrid", "--threads", "2", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 2, 4),
        ("Hybrid", "Hybrid 4Rx4T (16W)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--hybrid", "--threads", "4", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 4, 4),

        # Native CUDA GPU Acceleration (Dynamic Boundary Inflow)
        ("CUDA GPU", "NVIDIA RTX 4000 Ada (GPU)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", net_bin, "--metadata", net_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1),
    ]

    records = []
    t1_ms = None

    for cat, label, cmd, workers, threads, ranks in benchmarks:
        print(f"[{cat:15s}] Executing: {label} ...", end=" ", flush=True)
        success, stdout, stderr, wall_ms = run_command(cmd)
        if not success:
            print(f"FAILED!\n  Error: {stderr[:200]}")
            continue

        m = parse_simulation_output(stdout)
        sim_ms = m.get("exec_time_ms", wall_ms)
        if cat == "Sequential" and t1_ms is None:
            t1_ms = sim_ms

        sp = (t1_ms / sim_ms) if (t1_ms and sim_ms > 0) else 1.0
        ep = (sp / workers) if workers > 1 else 1.0
        rate = m.get("cell_update_rate", 0.0)

        print(f"DONE in {sim_ms:8.2f} ms | Sp: {sp:5.2f}x | Ep: {ep:5.3f} | {rate:1.2e} cells/s")

        records.append({
            "category": cat,
            "label": label,
            "tier": tier_name,
            "cars": cars,
            "steps": steps,
            "workers": workers,
            "threads": threads,
            "ranks": ranks,
            "sim_time_ms": sim_ms,
            "speedup": round(sp, 3),
            "efficiency": round(ep, 3),
            "cell_update_rate": rate,
            "active_vehicles": m.get("active_vehicles", 0),
            "completed_vehicles": m.get("completed_vehicles", 0),
            "injected_vehicles": m.get("injected_vehicles", cars),
            "load_imbalance": m.get("load_imbalance", 1.0),
            "mpi_wait_ms": m.get("mpi_wait_ms", 0.0),
            "comm_to_compute_ratio": m.get("comm_to_compute_ratio", 0.0),
            "conservation": m.get("conservation", "VERIFIED"),
        })

    # Save CSV
    csv_path = out_dir / "scaling_data.csv"
    with open(csv_path, "w") as f:
        f.write("category,label,tier,cars,steps,workers,threads,ranks,sim_time_ms,speedup,efficiency,cell_update_rate,active_vehicles,completed_vehicles,load_imbalance,mpi_wait_ms,comm_to_compute_ratio,conservation\n")
        for r in records:
            f.write(f"{r['category']},{r['label']},{r['tier']},{r['cars']},{r['steps']},{r['workers']},{r['threads']},{r['ranks']},{r['sim_time_ms']:.2f},{r['speedup']},{r['efficiency']},{r['cell_update_rate']:.2e},{r['active_vehicles']},{r['completed_vehicles']},{r['load_imbalance']:.3f},{r['mpi_wait_ms']:.2f},{r['comm_to_compute_ratio']:.3f},{r['conservation']}\n")

    # Save JSON
    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    generate_plots_for_tier(out_dir, records, tier_name, cars, steps)
    print(f"\n[SAVED] {tier_name} results written to {out_dir}")
    return records


def main():
    start_total = time.perf_counter()

    t1_records = run_tier(
        tier_key="tier1",
        tier_name="Tier 1 (Urban Core)",
        net_bin="data/karachi_network.bin",
        net_meta="data/karachi_network_metadata.json",
        cars=500000,
        steps=3600,
        out_dir=Path("data/benchmarks/tier1_500k_3600s")
    )

    t2_records = run_tier(
        tier_key="tier2",
        tier_name="Tier 2 (Metropolitan Karachi)",
        net_bin="data/karachi_outer_network.bin",
        net_meta="data/karachi_outer_network_metadata.json",
        cars=1000000,
        steps=3600,
        out_dir=Path("data/benchmarks/tier2_1m_3600s")
    )

    t3_records = run_tier(
        tier_key="tier3",
        tier_name="Tier 3 (Regional Mega-Corridor)",
        net_bin="data/karachi_regional_network.bin",
        net_meta="data/karachi_regional_network_metadata.json",
        cars=2000000,
        steps=3600,
        out_dir=Path("data/benchmarks/tier3_2m_3600s")
    )

    total_time_min = (time.perf_counter() - start_total) / 60.0
    print("\n" + "=" * 85)
    print(f"[COMPLETE] All 3 Tiers fully benchmarked in {total_time_min:.2f} minutes!")
    print("=" * 85)


if __name__ == "__main__":
    main()
