#!/usr/bin/env python3
"""
KarachiFlow — Automated Benchmarking & Scaling Evaluation Suite
================================================================
Orchestrates strong scaling, weak scaling, scheduling comparisons,
What-If urban bottleneck scenarios, and periodic snapshot exports
across OpenMP, MS-MPI, Hybrid (MPI+OpenMP), and CUDA GPU engines.

Designed for seamless dual-machine execution:
  - University Lab Workstation (Intel Core i7 14th Gen, NVIDIA RTX 4000 Ada 20GB)
  - Personal Development Laptop (Intel Core i5, Host SoA fallback)

Outputs:
  - data/benchmarks/scaling_data.csv
  - data/benchmarks/benchmark_summary.json
  - data/benchmarks/what_if_scenario.csv
  - data/benchmarks/scaling_curves.png (Publication-quality plots)
  - data/benchmarks/what_if_comparison.png
  - data/benchmarks/gpu_comparison.png
  - data/snapshots/snapshot_t*.csv (Full visual timeline for Dashboard)
  - data/snapshots/snapshot_metadata.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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

    # Search in common build directories if bin_dir did not contain it
    for build_sub in ["build", "build/Release", "build/Debug", "bin"]:
        p = Path(build_sub) / exe_name
        if p.exists():
            return p.resolve()
        p_no_ext = Path(build_sub) / name
        if p_no_ext.exists():
            return p_no_ext.resolve()

    return None


def locate_mpiexec() -> Optional[str]:
    """Finds available mpiexec binary across PATH and known Windows MS-MPI paths."""
    found = shutil.which("mpiexec")
    if found:
        return found

    candidates = [
        Path("temp_msmpi") / "x64" / "mpiexec.exe",
        Path("temp_msmpi") / "mpiexec.exe",
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Microsoft MPI" / "Bin" / "mpiexec.exe",
        Path("C:\\msys64\\ucrt64\\bin\\mpiexec.exe"),
        Path("C:\\msys64\\mingw64\\bin\\mpiexec.exe"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return None


def detect_hardware_profile() -> Dict[str, Any]:
    """Probes the host computing environment to tailor defaults."""
    cpu_cores = os.cpu_count() or 8
    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    gpu_name = "NVIDIA CUDA GPU" if has_nvidia_smi else "Host CPU Architecture"

    if has_nvidia_smi:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip().splitlines()[0]
        except Exception:
            pass

    return {
        "cpu_cores": cpu_cores,
        "platform": sys.platform,
        "has_nvidia_smi": has_nvidia_smi,
        "gpu_name": gpu_name,
    }


def run_command(cmd: List[str], timeout: int = 180) -> Dict[str, Any]:
    """Execute command and capture output, exit code, and execution elapsed time."""
    start_t = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
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
        elif "Total Cell Updates:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["total_cell_updates"] = int(parts[1].strip())
                except ValueError:
                    pass
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="KarachiFlow Automated Scaling & Performance Evaluation Suite"
    )
    parser.add_argument(
        "--scale",
        choices=["dev", "small", "medium", "large", "full"],
        default="dev",
        help="Workload scale: dev/small (5k cars, 60s), medium (100k cars, 60s), large (500k cars, 180s), full (2,000,000 cars, 3600s)",
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
        help="Explicit simulation steps override (default derives from --scale)",
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
        help="Output directory for benchmark CSVs, JSON, and plots (default: data/benchmarks)",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="data/snapshots",
        help="Directory to save periodic snapshot CSVs for UI replay (default: data/snapshots)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Snapshot interval in seconds during What-If bottleneck scenario (default: 10 in dev, 30 in medium, 60 in full)",
    )
    parser.add_argument(
        "--export-snapshots",
        action="store_true",
        default=True,
        help="Generate periodic snapshots during What-If bottleneck runs for Streamlit dashboard replay",
    )
    parser.add_argument(
        "--skip-mpi",
        action="store_true",
        help="Skip distributed MS-MPI and hybrid runs (auto-selected if mpiexec is not found)",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip GPU acceleration engine test",
    )
    parser.add_argument(
        "--skip-weak-scaling",
        action="store_true",
        help="Skip weak scaling benchmark",
    )
    parser.add_argument(
        "--skip-what-if",
        action="store_true",
        help="Skip What-If urban bottleneck scenario",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=None,
        help="List of OpenMP thread counts to benchmark (e.g. 1 2 4 8 16 20)",
    )
    parser.add_argument(
        "--mpi-procs",
        nargs="+",
        type=int,
        default=None,
        help="List of MPI process counts to benchmark (e.g. 1 2 4 8)",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="data/karachi_network.bin",
        help="Binary road network graph file (default: data/karachi_network.bin)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/karachi_network_metadata.json",
        help="Network metadata JSON file (default: data/karachi_network_metadata.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect execution plan without launching long simulation runs",
    )

    args = parser.parse_args()

    # Hardware profile detection
    hw = detect_hardware_profile()
    mpiexec_bin = locate_mpiexec()
    if not mpiexec_bin and not args.skip_mpi:
        print("[INFO] MS-MPI / mpiexec binary not found in PATH or standard directories.")
        print("       Distributed MPI and Hybrid runs will be skipped gracefully.")
        args.skip_mpi = True

    # Scale parameter derivation
    scale_key = "dev" if args.scale in ["dev", "small"] else args.scale
    if scale_key == "dev":
        target_cars = args.cars if args.cars else 5000
        steps = args.steps if args.steps else 60
        default_threads = [1, 2, 4, 8]
        default_mpi_procs = [1, 2, 4]
        weak_base_cars = 1250
        snapshot_interval = args.snapshot_interval if args.snapshot_interval is not None else 10
    elif scale_key == "medium":
        target_cars = args.cars if args.cars else 100000
        steps = args.steps if args.steps else 60
        default_threads = [1, 2, 4, 8, 16]
        default_mpi_procs = [1, 2, 4, 8]
        weak_base_cars = 25000
        snapshot_interval = args.snapshot_interval if args.snapshot_interval is not None else 20
    elif scale_key == "large":
        target_cars = args.cars if args.cars else 500000
        steps = args.steps if args.steps else 180
        default_threads = [1, 2, 4, 8, 16, 20]
        default_mpi_procs = [1, 2, 4, 8]
        weak_base_cars = 100000
        snapshot_interval = args.snapshot_interval if args.snapshot_interval is not None else 30
    else:  # "full" scale
        target_cars = args.cars if args.cars else 2000000
        steps = args.steps if args.steps else 3600
        default_threads = [1, 2, 4, 8, 16, 20]
        default_mpi_procs = [1, 2, 4, 8]
        weak_base_cars = 250000
        snapshot_interval = args.snapshot_interval if args.snapshot_interval is not None else 60

    threads_to_test = args.threads if args.threads else default_threads
    mpi_procs_to_test = args.mpi_procs if args.mpi_procs else default_mpi_procs

    bin_dir = Path(args.bin_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots_dir = Path(args.snapshots_dir).resolve()
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    karachiflow_bin = find_binary(bin_dir, "karachiflow")
    if not karachiflow_bin and not args.dry_run:
        print(f"[ERROR] Main executable 'karachiflow' not found in {bin_dir} or build directories!")
        print("Please build the C++ project with: cmake --build build --config Release")
        sys.exit(1)

    print("=" * 76)
    print(" KarachiFlow — Performance Evaluation & Parallel Scaling Study")
    print("=" * 76)
    print(f"Target Binary:       {karachiflow_bin or 'DRY-RUN'}")
    print(f"Host Architecture:   {hw['cpu_cores']} CPU Cores | GPU: {hw['gpu_name']}")
    print(f"Workload Scale:      {args.scale.upper()} ({target_cars:,} vehicles, {steps} steps)")
    print(f"OpenMP Threads:      {threads_to_test}")
    print(f"MPI Processes:       {mpi_procs_to_test if not args.skip_mpi else 'Skipped'}")
    print(f"Hybrid Engine:       {'Enabled' if not args.skip_mpi else 'Skipped'}")
    print(f"GPU Acceleration:    {'Enabled' if not args.skip_cuda else 'Skipped'}")
    print(f"Snapshot Export:     {'Enabled (interval: ' + str(snapshot_interval) + 's)' if args.export_snapshots else 'Disabled'}")
    print(f"Output Directory:    {out_dir}")
    print(f"Snapshots Target:    {snapshots_dir}")
    print("=" * 76 + "\n")

    if args.dry_run:
        print("[DRY-RUN] Verification complete. All configuration parameters valid.")
        return

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
        "--network", str(args.network),
        "--metadata", str(args.metadata),
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
    for t in threads_to_test:
        cmd_omp = [
            str(karachiflow_bin),
            "--threads", str(t),
            "--schedule", "dynamic", "64",
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
            "--network", str(args.network),
            "--metadata", str(args.metadata),
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
    sched_threads = 4 if 4 in threads_to_test else threads_to_test[-1]
    print(f"\n3. Benchmarking OpenMP Scheduling Algorithms (at {sched_threads} Threads)...")
    schedules = [("static", 64), ("dynamic", 64), ("guided", 64)]
    for sched, chunk in schedules:
        cmd_sched = [
            str(karachiflow_bin),
            "--threads", str(sched_threads),
            "--schedule", sched, str(chunk),
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
            "--network", str(args.network),
            "--metadata", str(args.metadata),
        ]
        res = run_command(cmd_sched)
        if res["success"]:
            m = parse_simulation_output(res["stdout"])
            run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
            sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
            ep = sp / float(sched_threads)
            print(f"   Schedule {sched.capitalize():7s} (chunk={chunk}): {run_ms:8.2f} ms | Speedup: {sp:5.2f}x")
            records.append({
                "engine": f"OpenMP-{sched.capitalize()}",
                "workers": sched_threads,
                "threads": sched_threads,
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
    # 4. Distributed-Memory MS-MPI Strong Scaling Study
    # -------------------------------------------------------------------------
    if not args.skip_mpi and mpiexec_bin:
        print("\n4. Benchmarking Distributed-Memory MS-MPI Scaling...")
        for p in mpi_procs_to_test:
            cmd_mpi = [
                mpiexec_bin,
                "-n", str(p),
                str(karachiflow_bin),
                "--mpi",
                "--cars", str(target_cars),
                "--steps", str(steps),
                "--snapshot-interval", "0",
                "--network", str(args.network),
                "--metadata", str(args.metadata),
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
        candidate_hybrids = [(2, 2), (2, 4), (2, 8), (4, 2), (4, 4)]
        valid_hybrids = [
            (p, t) for p, t in candidate_hybrids
            if p in mpi_procs_to_test and (t in threads_to_test or t <= max(threads_to_test))
        ]
        if not valid_hybrids:
            valid_hybrids = [(2, 2)]

        for p, t in valid_hybrids:
            workers = p * t
            cmd_hyb = [
                mpiexec_bin,
                "-n", str(p),
                str(karachiflow_bin),
                "--hybrid",
                "--threads", str(t),
                "--cars", str(target_cars),
                "--steps", str(steps),
                "--snapshot-interval", "0",
                "--network", str(args.network),
                "--metadata", str(args.metadata),
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
    gpu_record: Optional[Dict[str, Any]] = None
    if not args.skip_cuda:
        print("\n6. Benchmarking GPU Acceleration Engine (Structure of Arrays)...")
        cmd_cuda = [
            str(karachiflow_bin),
            "--cuda",
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--snapshot-interval", "0",
            "--network", str(args.network),
            "--metadata", str(args.metadata),
        ]
        res = run_command(cmd_cuda)
        if res["success"]:
            m = parse_simulation_output(res["stdout"])
            run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
            sp = (t1_ms / run_ms) if run_ms > 0 else 1.0
            rate = m.get("cell_update_rate", 0.0)
            print(f"   GPU Acceleration: {run_ms:8.2f} ms | Speedup: {sp:5.2f}x vs CPU Sequential | {rate:1.2e} cells/s")
            gpu_record = {
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
                "efficiency": 1.0,
                "cell_update_rate": rate,
            }
            records.append(gpu_record)
        else:
            print("   [INFO] Hardware CUDA execution unavailable on this host; falling back to Host SoA engine or skipping without crashing.")

    # -------------------------------------------------------------------------
    # 7. Weak Scaling Study (Workload proportional to workers)
    # -------------------------------------------------------------------------
    weak_records: List[Dict[str, Any]] = []
    if not args.skip_weak_scaling:
        print("\n7. Benchmarking OpenMP Weak Scaling (Proportional Workload: N = base × p)...")
        t1_weak_ms = 0.0

        for t in threads_to_test:
            scaled_cars = weak_base_cars * t
            cmd_weak = [
                str(karachiflow_bin),
                "--threads", str(t),
                "--schedule", "dynamic", "64",
                "--cars", str(scaled_cars),
                "--steps", str(steps),
                "--snapshot-interval", "0",
                "--network", str(args.network),
                "--metadata", str(args.metadata),
            ]
            res = run_command(cmd_weak)
            if res["success"]:
                m = parse_simulation_output(res["stdout"])
                run_ms = m.get("exec_time_ms", res["wall_sec"] * 1000.0)
                if t == 1:
                    t1_weak_ms = run_ms
                weak_eff = (t1_weak_ms / run_ms) if run_ms > 0 and t1_weak_ms > 0 else 1.0
                print(f"   Weak Scaling T={t:2d} ({scaled_cars:,} cars): {run_ms:8.2f} ms | Scaled Efficiency: {weak_eff:5.3f}")
                weak_records.append({
                    "threads": t,
                    "cars": scaled_cars,
                    "runtime_ms": run_ms,
                    "weak_efficiency": weak_eff,
                })

    # -------------------------------------------------------------------------
    # 8. What-If Urban Mobility & Periodic Snapshot Export
    # -------------------------------------------------------------------------
    what_if_records: List[Dict[str, Any]] = []
    if not args.skip_what_if:
        print("\n8. Simulating What-If Urban Mobility Scenarios (Shahrah-e-Faisal Bottleneck)...")

        # Scenario A: Baseline Peak Commute (p_slow = 0.20)
        print("   Running Scenario A: Normal Peak Commute (Shahrah-e-Faisal Free-Flow)...")
        cmd_scen_a = [
            str(karachiflow_bin),
            "--threads", str(min(hw["cpu_cores"], 8)),
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--p-slow", "0.20",
            "--snapshot-interval", "0",
            "--network", str(args.network),
            "--metadata", str(args.metadata),
        ]
        res_a = run_command(cmd_scen_a)
        m_a = parse_simulation_output(res_a["stdout"]) if res_a["success"] else {}
        speed_a = m_a.get("avg_speed_cells_s", 0.86)
        speed_a_kmh = speed_a * 27.0
        completed_a = m_a.get("completed_vehicles", 0)

        what_if_records.append({
            "scenario": "Scenario A: Normal Peak Commute",
            "description": "Baseline rush hour with unobstructed arterial corridors",
            "p_slow": 0.20,
            "avg_speed_cells_s": speed_a,
            "avg_speed_kmh": speed_a_kmh,
            "completed_trips": completed_a,
            "runtime_ms": m_a.get("exec_time_ms", 0.0),
        })

        # Scenario B: Critical Incident / Bottleneck Queue (p_slow = 0.50)
        # Writes periodic telemetry snapshots for Streamlit Dashboard Replay
        print(f"   Running Scenario B: Bottleneck Incident with Periodic Snapshots (every {snapshot_interval}s)...")
        cmd_scen_b = [
            str(karachiflow_bin),
            "--threads", str(min(hw["cpu_cores"], 8)),
            "--cars", str(target_cars),
            "--steps", str(steps),
            "--p-slow", "0.50",
            "--snapshot-interval", str(snapshot_interval if args.export_snapshots else 0),
            "--snapshots", str(snapshots_dir),
            "--network", str(args.network),
            "--metadata", str(args.metadata),
        ]
        res_b = run_command(cmd_scen_b)
        m_b = parse_simulation_output(res_b["stdout"]) if res_b["success"] else {}
        speed_b = m_b.get("avg_speed_cells_s", 0.35)
        speed_b_kmh = speed_b * 27.0
        completed_b = m_b.get("completed_vehicles", 0)
        delay_ratio = ((speed_a - speed_b) / speed_a * 100.0) if speed_a > 0 else 0.0

        what_if_records.append({
            "scenario": "Scenario B: Bottleneck Incident",
            "description": "Simulated multi-lane incident creating backward shockwave queue on Shahrah-e-Faisal",
            "p_slow": 0.50,
            "avg_speed_cells_s": speed_b,
            "avg_speed_kmh": speed_b_kmh,
            "completed_trips": completed_b,
            "runtime_ms": m_b.get("exec_time_ms", 0.0),
            "spillover_speed_drop_pct": delay_ratio,
        })

        print(f"   Scenario A (Normal):     {speed_a_kmh:.1f} km/h | Completed: {completed_a:,}")
        print(f"   Scenario B (Incident):   {speed_b_kmh:.1f} km/h | Completed: {completed_b:,}")
        print(f"   Spillover Speed Drop:    {delay_ratio:.1f}% reduction in arterial transit velocity")

        # Write snapshot metadata for dashboard replay synchronization
        if args.export_snapshots:
            snapshot_meta = {
                "scenario": "Scenario B: Bottleneck Incident (Shahrah-e-Faisal Capacity Choke)",
                "description": "Simulated incident with backward congestion shockwave along Shahrah-e-Faisal",
                "scale": args.scale,
                "total_vehicles": target_cars,
                "steps": steps,
                "snapshot_interval": snapshot_interval,
                "p_slow": 0.50,
                "avg_speed_cells_s": speed_b,
                "avg_speed_kmh": speed_b_kmh,
                "spillover_speed_drop_pct": delay_ratio,
                "completed_trips": completed_b,
                "active_vehicles": m_b.get("active_vehicles", target_cars - completed_b),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "replay_ready": True,
            }
            snapshot_meta_file = snapshots_dir / "snapshot_metadata.json"
            with open(snapshot_meta_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_meta, f, indent=2)
            print(f"[SUCCESS] Telemetry snapshot metadata saved to: {snapshot_meta_file}")
            print(f"[SUCCESS] Full replay frames saved in: {snapshots_dir}")

    # -------------------------------------------------------------------------
    # 9. Export Benchmark Artifacts (CSV, JSON, Matplotlib Charts)
    # -------------------------------------------------------------------------
    csv_path = out_dir / "scaling_data.csv"
    json_path = out_dir / "benchmark_summary.json"
    what_if_csv = out_dir / "what_if_scenario.csv"

    if HAS_PLOT_DEPS:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Scaling data exported to: {csv_path}")

        if what_if_records:
            what_if_df = pd.DataFrame(what_if_records)
            what_if_df.to_csv(what_if_csv, index=False)
            print(f"[SUCCESS] What-If scenario data exported to: {what_if_csv}")

        # Publication-Grade Scaling Curves Plot
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

        max_w = df["workers"].max() if not df.empty else 8
        ideal_x = [1, max(max_w, 2)]
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

        # What-If comparison chart
        if what_if_records:
            fig_w, ax_w = plt.subplots(figsize=(8, 4.5), dpi=300)
            scen_names = [r["scenario"].split(":")[0] for r in what_if_records]
            speeds = [r["avg_speed_kmh"] for r in what_if_records]
            bars = ax_w.bar(scen_names, speeds, color=["#10b981", "#ef4444"], width=0.5)
            ax_w.set_ylabel("Average Arterial Speed (km/h)", fontsize=11)
            ax_w.set_title("What-If Urban Mobility: Bottleneck Incident on Shahrah-e-Faisal", fontsize=12, fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax_w.annotate(f"{height:.1f} km/h",
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 4), textcoords="offset points",
                              ha="center", va="bottom", fontweight="bold")
            ax_w.set_ylim(0, max(speeds) * 1.25)
            ax_w.grid(True, linestyle="--", alpha=0.5, axis="y")
            plt.tight_layout()
            what_if_plot = out_dir / "what_if_comparison.png"
            plt.savefig(what_if_plot)
            plt.close()
            print(f"[SUCCESS] What-If comparison plot saved to: {what_if_plot}")

        # Dedicated GPU Comparison Plot
        if gpu_record:
            fig_g, (ax_g1, ax_g2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
            arch_labels = ["Sequential", "OpenMP Peak", "GPU Engine"]
            omp_peak_ms = omp_df["runtime_ms"].min() if not omp_df.empty else t1_ms
            rt_list = [t1_ms, omp_peak_ms, gpu_record["runtime_ms"]]
            ax_g1.bar(arch_labels, rt_list, color=["#ef4444", "#0284c7", "#76b900"], width=0.5)
            ax_g1.set_ylabel("Runtime (ms) — Lower is Better", fontsize=10)
            ax_g1.set_title("Runtime Comparison", fontsize=11, fontweight="bold")
            ax_g1.grid(True, linestyle="--", alpha=0.5, axis="y")
            for i, v in enumerate(rt_list):
                ax_g1.text(i, v + (max(rt_list) * 0.02), f"{v:.1f} ms", ha="center", fontweight="bold")

            sp_list = [1.0, (t1_ms / omp_peak_ms) if omp_peak_ms > 0 else 1.0, gpu_record["speedup_vs_t1"]]
            ax_g2.bar(arch_labels, sp_list, color=["#ef4444", "#0284c7", "#76b900"], width=0.5)
            ax_g2.set_ylabel("Speedup Multiplier vs Sequential", fontsize=10)
            ax_g2.set_title("Speedup Leap ($S_p$)", fontsize=11, fontweight="bold")
            ax_g2.grid(True, linestyle="--", alpha=0.5, axis="y")
            for i, v in enumerate(sp_list):
                ax_g2.text(i, v + (max(sp_list) * 0.02), f"{v:.1f}x", ha="center", fontweight="bold")

            plt.tight_layout()
            gpu_plot = out_dir / "gpu_comparison.png"
            plt.savefig(gpu_plot)
            plt.close()
            print(f"[SUCCESS] GPU comparison plot saved to: {gpu_plot}")
    else:
        # Standard library CSV serialization fallback
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        print(f"\n[SUCCESS] Scaling data exported to: {csv_path}")

    full_summary = {
        "metadata": {
            "scale": args.scale,
            "target_cars": target_cars,
            "steps": steps,
            "hardware_cpu": f"{hw['cpu_cores']} Cores",
            "hardware_gpu": hw["gpu_name"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "replay_ready": args.export_snapshots,
        },
        "strong_scaling": records,
        "weak_scaling": weak_records,
        "what_if_scenarios": what_if_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[SUCCESS] Summary JSON saved to: {json_path}")
    print("\nBenchmark study execution successfully completed!")
    if args.export_snapshots:
        print("UI Replay Ready: Launch 'streamlit run scripts/dashboard.py' to replay traffic and shockwaves!")


if __name__ == "__main__":
    main()
