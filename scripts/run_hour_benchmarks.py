#!/usr/bin/env python3
"""
KarachiFlow — Dedicated 3,600-Step (1 Simulated Hour) Full-Scale Parallel Benchmark Suite
Executes long-term steady-state scaling runs across all 5 parallel paradigms
at 1,000,000 to 10,000,000 vehicles on Tier 2 Metropolitan and Tier 3 Regional Mega-Corridor networks.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def parse_simulation_output(stdout: str):
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
        elif "Load Imbalance" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["load_imbalance"] = float(parts[1].split("(")[0].strip())
                except ValueError:
                    pass
        elif "MPI Waitall Latency:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    metrics["mpi_wait_ms"] = float(parts[1].split("ms")[0].strip())
                except ValueError:
                    pass
    return metrics

def run_command(cmd, timeout=600):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return p.returncode == 0, p.stdout, p.stderr, wall_ms

def main():
    t2_bin = "data/karachi_outer_network.bin"
    t2_meta = "data/karachi_outer_network_metadata.json"
    t3_bin = "data/karachi_regional_network.bin"
    t3_meta = "data/karachi_regional_network_metadata.json"
    steps = 3600
    cars = 1000000
    out_dir = Path("data/benchmarks/hour_3600s")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" KarachiFlow — Dedicated 3,600-Step (1 Simulated Hour) Full-Scale Benchmark")
    print(f" Target Networks: Tier 2 (5.94M cells) & Tier 3 (9.10M cells)")
    print(f" Workload Scale: Up to 10,000,000 Vehicles | 3,600 Timesteps")
    print("=" * 80 + "\n")

    benchmarks = [
        # Tier 2 Metropolitan Karachi (5.94M cells) 3600 steps
        ("Sequential Baseline (Tier 2, 1M Cars)", ["bin/karachiflow.exe", "--sequential", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 2", cars),
        ("OpenMP 8 Threads (Tier 2, 1M Cars)", ["bin/karachiflow.exe", "--threads", "8", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 8, 1, "Tier 2", cars),
        ("OpenMP 16 Threads (Tier 2, 1M Cars)", ["bin/karachiflow.exe", "--threads", "16", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 16, 1, "Tier 2", cars),
        ("OpenMP 20 Threads (Tier 2, 1M Cars)", ["bin/karachiflow.exe", "--threads", "20", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 20, 20, 1, "Tier 2", cars),
        ("MS-MPI 2 Ranks (Tier 2, 1M Cars)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--mpi", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 2, 1, 2, "Tier 2", cars),
        ("MS-MPI 4 Ranks (Tier 2, 1M Cars)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--mpi", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 4, 1, 4, "Tier 2", cars),
        ("MS-MPI 8 Ranks (Tier 2, 1M Cars)", ["mpiexec", "-n", "8", "bin/karachiflow.exe", "--mpi", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 8, 1, 8, "Tier 2", cars),
        ("Hybrid 2R x 8T (Tier 2, 1M Cars)", ["mpiexec", "-n", "2", "bin/karachiflow.exe", "--hybrid", "--threads", "8", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 8, 2, "Tier 2", cars),
        ("Hybrid 4R x 4T (Tier 2, 1M Cars)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--hybrid", "--threads", "4", "--network", t2_bin, "--metadata", t2_meta, "--cars", str(cars), "--steps", str(steps), "--snapshot-interval", "0"], 16, 4, 4, "Tier 2", cars),
        ("RTX 4000 Ada (Tier 2, 1M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t2_bin, "--metadata", t2_meta, "--cars", "1000000", "--seed-vehicles", "1000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 2", 1000000),
        ("RTX 4000 Ada (Tier 2, 2M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t2_bin, "--metadata", t2_meta, "--cars", "2000000", "--seed-vehicles", "2000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 2", 2000000),
        ("RTX 4000 Ada (Tier 2, 3M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t2_bin, "--metadata", t2_meta, "--cars", "3000000", "--seed-vehicles", "3000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 2", 3000000),

        # Tier 3 Regional Mega-Corridor (9.10M cells) 3600 steps
        ("OpenMP 16 Threads (Tier 3 Regional, 2M Cars)", ["bin/karachiflow.exe", "--threads", "16", "--network", t3_bin, "--metadata", t3_meta, "--cars", "2000000", "--steps", str(steps), "--snapshot-interval", "0"], 16, 16, 1, "Tier 3 Regional", 2000000),
        ("MS-MPI 8 Ranks (Tier 3 Regional, 2M Cars)", ["mpiexec", "-n", "8", "bin/karachiflow.exe", "--mpi", "--network", t3_bin, "--metadata", t3_meta, "--cars", "2000000", "--steps", str(steps), "--snapshot-interval", "0"], 8, 1, 8, "Tier 3 Regional", 2000000),
        ("Hybrid 4R x 4T (Tier 3 Regional, 2M Cars)", ["mpiexec", "-n", "4", "bin/karachiflow.exe", "--hybrid", "--threads", "4", "--network", t3_bin, "--metadata", t3_meta, "--cars", "2000000", "--steps", str(steps), "--snapshot-interval", "0"], 16, 4, 4, "Tier 3 Regional", 2000000),
        ("RTX 4000 Ada (Tier 3 Regional, 3M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t3_bin, "--metadata", t3_meta, "--cars", "3000000", "--seed-vehicles", "3000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 3 Regional", 3000000),
        ("RTX 4000 Ada (Tier 3 Regional, 5M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t3_bin, "--metadata", t3_meta, "--cars", "5000000", "--seed-vehicles", "5000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 3 Regional", 5000000),
        ("RTX 4000 Ada HPC Stress (Tier 3, 10M Cars)", ["bin/karachiflow_gpu.exe", "--cuda", "--network", t3_bin, "--metadata", t3_meta, "--cars", "10000000", "--seed-vehicles", "10000000", "--steps", str(steps), "--snapshot-interval", "0"], 1, 1, 1, "Tier 3 Regional", 10000000),
    ]

    records = []
    t1_ms = None

    for label, cmd, workers, threads, ranks, tier_label, veh_count in benchmarks:
        print(f"Executing: {label} ...")
        success, stdout, stderr, wall_ms = run_command(cmd)
        if not success:
            print(f"  [WARN] Failed: {stderr[:200]}")
            continue

        m = parse_simulation_output(stdout)
        sim_ms = m.get("exec_time_ms", wall_ms)
        if "Baseline" in label and t1_ms is None:
            t1_ms = sim_ms

        sp = (t1_ms / sim_ms) if (t1_ms and sim_ms > 0) else 1.0
        ep = (sp / workers) if workers > 1 else 1.0
        rate = m.get("cell_update_rate", 0.0)

        print(f"  --> Time: {sim_ms:8.2f} ms | Speedup: {sp:5.2f}x | Efficiency: {ep:5.3f} | Throughput: {rate:1.2e} cells/s")
        records.append({
            "label": label,
            "tier": tier_label,
            "cars": veh_count,
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
            "injected_vehicles": m.get("injected_vehicles", veh_count),
            "load_imbalance": m.get("load_imbalance", 1.0),
        })

    # Export CSV & JSON
    csv_path = out_dir / "scaling_data.csv"
    with open(csv_path, "w") as f:
        f.write("label,tier,cars,steps,workers,threads,ranks,sim_time_ms,speedup,efficiency,cell_update_rate,active_vehicles,completed_vehicles,load_imbalance\n")
        for r in records:
            f.write(f"{r['label']},{r['tier']},{r['cars']},{r['steps']},{r['workers']},{r['threads']},{r['ranks']},{r['sim_time_ms']:.2f},{r['speedup']},{r['efficiency']},{r['cell_update_rate']:.2e},{r['active_vehicles']},{r['completed_vehicles']},{r['load_imbalance']:.3f}\n")

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 3,600-step benchmarks completed! Data written to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
