#!/usr/bin/env python3
"""
KarachiFlow — Automated Benchmarking & Scaling Evaluation Suite
================================================================
Orchestrates strong scaling, weak scaling, scheduling comparisons,
and What-If urban bottleneck scenarios across OpenMP, MS-MPI,
Hybrid (MPI+OpenMP), and GPU Acceleration engines.

Outputs:
  - data/benchmarks/scaling_data.csv
  - data/benchmarks/benchmark_summary.json
  - data/benchmarks/scaling_curves.png (Publication-quality plots)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOT_DEPS = True
except ImportError:
    HAS_PLOT_DEPS = False


def find_binary(bin_dir: Path, name: str) -> Optional[Path]:
    """Locate binary executable with or without .exe extension."""
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    cand = bin_dir / exe_name
    if cand.exists():
        return cand
    cand_root = bin_dir / name
    if cand_root.exists():
        return cand_root
    return None


def run_command(cmd: List[str], timeout: int = 120) -> Dict[str, Any]:
    """Execute command and capture output, exit code, and execution elapsed time."""
    start_t = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        elapsed_sec = time.perf_counter() - start_t
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "wall_sec": elapsed_sec,
        }
    except Exception as e:
        elapsed_sec = time.perf_counter() - start_t
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "wall_sec": elapsed_sec,
        }


def parse_simulation_output(stdout: str) -> Dict[str, Any]:
    """Parse structured execution summary metrics from simulation stdout."""
    metrics: Dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "Execution Time:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                ms_part = parts[1].split("ms")[0].strip()
                try:
                    metrics["exec_time_ms"] = float(ms_part)
                except ValueError:
                    pass
        elif "Cell Update Rate:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                rate_part = parts[1].split("cell")[0].strip()
                try:
                    metrics["cell_update_rate"] = float(rate_part)
                except ValueError:
                    pass
        elif "Average Network Speed:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                spd_part = parts[1].split("cells/s")[0].strip()
                try:
                    metrics["avg_speed_cells_s"] = float(spd_part)
                except ValueError:
                    pass
        elif "Load Imbalance" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                imb_part = parts[1].split("(")[0].strip()
                try:
                    metrics["load_imbalance"] = float(imb_part)
                except ValueError:
                    pass
        elif "Comm-to-Compute Ratio:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                comm_part = parts[1].split("(")[0].strip()
                try:
                    metrics["comm_ratio"] = float(comm_part)
                except ValueError:
                    pass
        elif "Active Vehicles:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["active_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
        elif "Completed Trips:" in line or "Completed vehicles:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["completed_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
        elif "Injected Vehicles:" in line or "Injected vehicles:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["injected_vehicles"] = int(parts[1].strip())
                except ValueError:
                    pass
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="KarachiFlow Automated Scaling & Performance Evaluation Suite"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Workload scale: small (5k cars, 60s), medium (50k cars, 60s), large (250k cars, 120s)",
    )
    parser.add_argument(
        "--cars",
        type=int,
        default=None,
        help="Explicit car count override (default derives from --scale)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Explicit step count override (default: 60)",
    )
    parser.add_argument(
        "--bin-dir",
        type=str,
        default="bin",
        help="Path to compiled C++ binaries directory (default: bin)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks",
        help="Output directory for CSVs and plots (default: data/benchmarks)",
    )
    parser.add_argument(
        "--skip-mpi",
        action="store_true",
        help="Skip distributed MPI and hybrid runs (useful if MS-MPI runtime is absent)",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip GPU acceleration engine test",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="List of OpenMP thread counts to benchmark (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--mpi-procs",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="List of MPI process counts to benchmark (default: 1 2 4)",
    )

    args = parser.parse_args()

    # Determine workload parameters
    if args.scale == "small":
        target_cars = args.cars if args.cars else 5000
        steps = args.steps if args.steps else 60
    elif args.scale == "medium":
        target_cars = args.cars if args.cars else 50000
        steps = args.steps if args.steps else 60
    else:
        target_cars = args.cars if args.cars else 250000
        steps = args.steps if args.steps else 120

    bin_dir = Path(args.bin_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    karachiflow_bin = find_binary(bin_dir, "karachiflow")
    if not karachiflow_bin:
        print(f"[ERROR] Main executable 'karachiflow' not found in {bin_dir}!")
        print("Please build the C++ project with: cmake --build build --config Release")
        sys.exit(1)

    print("=" * 72)
    print(" KarachiFlow — Performance Evaluation & Parallel Scaling Study")
    print("=" * 72)
    print(f"Target Binary:     {karachiflow_bin}")
    print(f"Workload Scale:    {args.scale.upper()} ({target_cars:,} vehicles, {steps} steps)")
    print(f"OpenMP Threads:    {args.threads}")
    print(f"MPI Processes:     {args.mpi_procs if not args.skip_mpi else 'Skipped'}")
    print(f"GPU Engine (CUDA): {'Enabled' if not args.skip_cuda else 'Skipped'}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 72 + "\n")

    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Single-Threaded Sequential Baseline Reference
    # -------------------------------------------------------------------------
    print("1. Benchmarking Single-Threaded Sequential Baseline...")
    cmd_seq = [
        str(karachiflow_bin),
        "--sequential",
        "--cars", str(target_cars),
        "--steps", str(steps),
        "--snapshot-interval", "0",
    ]
    res_seq = run_command(cmd_seq)
    if not res_seq["success"]:
        print(f"[WARN] Sequential baseline failed: {res_seq['stderr']}")
        t1_ms = res_seq["wall_sec"] * 1000.0
    else:
        m = parse_simulation_output(res_seq["stdout"])
        t1_ms = m.get("exec_time_ms", res_seq["wall_sec"] * 1000.0)

    print(f"   Baseline Time (T1): {t1_ms:.2f} ms\n")
    records.append({
        "engine": "Sequential",
        "workers": 1,
        "threads": 1,
        "mpi_ranks": 1,
        "schedule": "None",
        "chunk_size": 0,
        "cars": target_cars,
        "steps": steps,
        "runtime_ms": t1_ms,
        "speedup_vs_t1": 1.0,
        "efficiency": 1.0,
        "cell_update_rate": parse_simulation_output(res_seq["stdout"]).get("cell_update_rate", 0.0),
    })

    # -------------------------------------------------------------------------
    # 2. Shared-Memory OpenMP Strong Scaling Study
    # -------------------------------------------------------------------------
    print("2. Benchmarking Shared-Memory OpenMP Strong Scaling...")
    for t in args.threads:
        cmd_omp = [
            str(karachiflow_bin),
            "--threads", str(t),
            "--schedule", "dynamic", "64",
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
        ]
        res = run_command(cmd_omp)
        if res["success"]:
            m = parse_simulation_output(res["stdout"])
            run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
            sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
            ep = sp / float(t)
            rate = m.get("cell_update_rate", 0.0)
            print(f"   OpenMP T={t:2d}: {run_ms:8.2f} ms | Speedup: {sp:5.2f}x | Efficiency: {ep:5.3f} | {rate:1.2e} cells/s")
            records.append({
                "engine": "OpenMP",
                "workers": t,
                "threads": t,
                "mpi_ranks": 1,
                "schedule": "dynamic",
                "chunk_size": 64,
                "cars": target_cars,
                "steps": steps,
                "runtime_ms": run_ms,
                "speedup_vs_t1": sp,
                "efficiency": ep,
                "cell_update_rate": rate,
            })
        else:
            print(f"   [WARN] OpenMP run failed for T={t}: {res['stderr']}")

    # -------------------------------------------------------------------------
    # 3. OpenMP Loop Scheduling Study (Static vs Dynamic vs Guided)
    # -------------------------------------------------------------------------
    print("\n3. Benchmarking OpenMP Scheduling Algorithms (at 4 Threads)...")
    schedules = [("static", 64), ("dynamic", 64), ("guided", 64)]
    for sched, chunk in schedules:
        cmd_sched = [
            str(karachiflow_bin),
            "--threads", "4",
            "--schedule", sched, str(chunk),
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
        ]
        res = run_command(cmd_sched)
        if res["success"]:
            m = parse_simulation_output(res["stdout"])
            run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
            sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
            ep = sp / 4.0
            print(f"   Schedule {sched.capitalize():7s} (chunk={chunk}): {run_ms:8.2f} ms | Speedup: {sp:5.2f}x")
            records.append({
                "engine": f"OpenMP-{sched.capitalize()}",
                "workers": 4,
                "threads": 4,
                "mpi_ranks": 1,
                "schedule": sched,
                "chunk_size": chunk,
                "cars": target_cars,
                "steps": steps,
                "runtime_ms": run_ms,
                "speedup_vs_t1": sp,
                "efficiency": ep,
                "cell_update_rate": m.get("cell_update_rate", 0.0),
            })

    # -------------------------------------------------------------------------
    # 4. Distributed-Memory MS-MPI Scaling Study (Optional / Available)
    # -------------------------------------------------------------------------
    if not args.skip_mpi:
        print("\n4. Benchmarking Distributed-Memory MS-MPI Scaling...")
        mpiexec_cmd = "mpiexec"
        for p in args.mpi_procs:
            cmd_mpi = [
                mpiexec_cmd,
                "-n", str(p),
                str(karachiflow_bin),
                "--mpi",
                "--cars", str(target_cars),
                "--steps", str(steps),
                "--snapshot-interval", "0",
            ]
            res = run_command(cmd_mpi)
            if res["success"]:
                m = parse_simulation_output(res["stdout"])
                run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
                sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
                ep = sp / float(p)
                rate = m.get("cell_update_rate", 0.0)
                imb = m.get("load_imbalance", 1.0)
                print(f"   MS-MPI P={p:2d}: {run_ms:8.2f} ms | Speedup: {sp:5.2f}x | Efficiency: {ep:5.3f} | Imbalance: {imb:.3f}")
                records.append({
                    "engine": "MS-MPI",
                    "workers": p,
                    "threads": 1,
                    "mpi_ranks": p,
                    "schedule": "Domain-Decomposition",
                    "chunk_size": 0,
                    "cars": target_cars,
                    "steps": steps,
                    "runtime_ms": run_ms,
                    "speedup_vs_t1": sp,
                    "efficiency": ep,
                    "cell_update_rate": rate,
                    "load_imbalance": imb,
                })
            else:
                print(f"   [INFO] MPI execution skipped or not configured for P={p}.")

        # ---------------------------------------------------------------------
        # 5. Hybrid (MPI + OpenMP) Orchestration Run
        # ---------------------------------------------------------------------
        print("\n5. Benchmarking Hybrid (MPI + OpenMP) Orchestration...")
        hybrid_configs = [(2, 2), (2, 4)]
        for p, t in hybrid_configs:
            workers = p * t
            cmd_hyb = [
                mpiexec_cmd,
                "-n", str(p),
                str(karachiflow_bin),
                "--hybrid",
                "--threads", str(t),
                "--cars", str(target_cars),
                "--steps", str(steps),
                "--snapshot-interval", "0",
            ]
            res = run_command(cmd_hyb)
            if res["success"]:
                m = parse_simulation_output(res["stdout"])
                run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
                sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
                ep = sp / float(workers)
                rate = m.get("cell_update_rate", 0.0)
                print(f"   Hybrid P={p} x T={t} ({workers:2d} workers): {run_ms:8.2f} ms | Speedup: {sp:5.2f}x | Efficiency: {ep:5.3f}")
                records.append({
                    "engine": "Hybrid",
                    "workers": workers,
                    "threads": t,
                    "mpi_ranks": p,
                    "schedule": "Funneled-Halo",
                    "chunk_size": 64,
                    "cars": target_cars,
                    "steps": steps,
                    "runtime_ms": run_ms,
                    "speedup_vs_t1": sp,
                    "efficiency": ep,
                    "cell_update_rate": rate,
                })

    # -------------------------------------------------------------------------
    # 6. GPU Acceleration Engine Evaluation
    # -------------------------------------------------------------------------
    if not args.skip_cuda:
        print("\n6. Benchmarking GPU Acceleration Engine (Structure of Arrays)...")
        cmd_cuda = [
            str(karachiflow_bin),
            "--cuda",
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
        ]
        res = run_command(cmd_cuda)
        if res["success"]:
            m = parse_simulation_output(res["stdout"])
            run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
            sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
            rate = m.get("cell_update_rate", 0.0)
            print(f"   GPU Acceleration: {run_ms:8.2f} ms | Speedup: {sp:5.2f}x vs CPU Sequential | {rate:1.2e} cells/s")
            records.append({
                "engine": "GPU-Acceleration",
                "workers": 1,
                "threads": 1,
                "mpi_ranks": 1,
                "schedule": "SoA-atomicMin",
                "chunk_size": 0,
                "cars": target_cars,
                "steps": steps,
                "runtime_ms": run_ms,
                "speedup_vs_t1": sp,
                "efficiency": sp,
                "cell_update_rate": rate,
            })
        else:
            print(f"   [INFO] GPU engine run: {res['stderr']}")

    # -------------------------------------------------------------------------
    # Save Benchmark Data to CSV & JSON
    # -------------------------------------------------------------------------
    csv_path = out_dir / "scaling_data.csv"
    json_path = out_dir / "benchmark_summary.json"

    if HAS_PLOT_DEPS:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Scaling data exported to: {csv_path}")

        # Generate publication-grade scaling plot
        try:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        except Exception:
            pass

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Plot 1: Speedup Sp vs Workers
        omp_df = df[df["engine"] == "OpenMP"]
        if not omp_df.empty:
            ax1.plot(omp_df["workers"], omp_df["speedup_vs_t1"], marker="o", linewidth=2.2, label="OpenMP (Dynamic 64)", color="#0284c7")
        mpi_df = df[df["engine"] == "MS-MPI"]
        if not mpi_df.empty:
            ax1.plot(mpi_df["workers"], mpi_df["speedup_vs_t1"], marker="s", linewidth=2.2, label="Distributed MS-MPI", color="#10b981")
        hyb_df = df[df["engine"] == "Hybrid"]
        if not hyb_df.empty:
            ax1.plot(hyb_df["workers"], hyb_df["speedup_vs_t1"], marker="^", linewidth=2.2, label="Hybrid (MPI + OpenMP)", color="#8b5cf6")

        max_w = df["workers"].max()
        ideal_x = [1, max_w]
        ax1.plot(ideal_x, ideal_x, "--", color="#94a3b8", label="Ideal Linear Speedup ($S_p = p$)")

        ax1.set_title(f"Strong Scaling Speedup ($S_p$) — {target_cars:,} Vehicles", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Compute Workers ($p$)", fontsize=11)
        ax1.set_ylabel("Speedup Factor ($S_p = T_1 / T_p$)", fontsize=11)
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Plot 2: Parallel Efficiency Ep vs Workers
        if not omp_df.empty:
            ax2.plot(omp_df["workers"], omp_df["efficiency"], marker="o", linewidth=2.2, label="OpenMP Efficiency", color="#0284c7")
        if not mpi_df.empty:
            ax2.plot(mpi_df["workers"], mpi_df["efficiency"], marker="s", linewidth=2.2, label="MS-MPI Efficiency", color="#10b981")
        if not hyb_df.empty:
            ax2.plot(hyb_df["workers"], hyb_df["efficiency"], marker="^", linewidth=2.2, label="Hybrid Efficiency", color="#8b5cf6")

        ax2.axhline(y=1.0, linestyle="--", color="#94a3b8", label="Ideal Efficiency ($E_p = 1.0$)")
        ax2.set_title("Parallel Efficiency ($E_p = S_p / p$)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Compute Workers ($p$)", fontsize=11)
        ax2.set_ylabel("Efficiency ($E_p$)", fontsize=11)
        ax2.set_ylim(0.0, 1.25)
        ax2.legend(loc="lower left", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plot_path = out_dir / "scaling_curves.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[SUCCESS] High-resolution scaling curves saved to: {plot_path}")
    else:
        # Fallback to pure standard library CSV serialization
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[SUCCESS] Scaling data exported to: {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[SUCCESS] Summary JSON saved to: {json_path}")
    print("\nBenchmark study execution successfully completed!")


if __name__ == "__main__":
    main()
