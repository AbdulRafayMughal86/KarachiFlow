<div align="center">

# KarachiFlow

### City-Scale Microscopic Traffic & Congestion Simulator

KarachiFlow is an urban mobility and microscopic traffic simulator modeled on the road network of Karachi, Pakistan. Built using the Nagel–Schreckenberg cellular automaton model, it simulates vehicle dynamics and rush-hour bottlenecks across major city corridors using high-performance parallel computing.

[Overview](#overview) · [Architecture](#architecture) · [Quickstart](#quickstart) · [Interactive Dashboard](#interactive-dashboard) · [Benchmarking](#benchmarking) · [Verification](#verification) · [Tech Stack](#tech-stack)

</div>

---

## Overview

**KarachiFlow** simulates city-scale traffic flows by modeling individual vehicle dynamics, lane-level interactions, and arterial traffic patterns across Karachi's real road network (extracted via OpenStreetMap).

The simulator is designed to scale across multi-core processors, distributed nodes, and GPUs using a hybrid parallel architecture (OpenMP + MPI + CUDA), enabling fast simulation of millions of vehicle updates.

### Key Capabilities
- **Real Metropolitan Road Network:** Ingests 74,914 directed road segments, 27,006 intersections, and 1,074,814 spatial cells (7.5m cell resolution) serialized into zero-copy binary format.
- **Microscopic Physics:** Discrete 4-step Nagel–Schreckenberg (NaSch) cellular automaton with lookahead braking, stochastic slowing ($p_{\text{slow}} = 0.2$), and yellow-box intersection clearing.
- **Shared-Memory Parallelism (OpenMP):** Multi-threaded segment updates with cache line padding (`alignas(64)`), dynamic/static/guided loop scheduling, and bit-exact determinism (`splitmix64`).
- **Distributed-Memory Parallelism (MPI):** Spatial domain decomposition via recursive bisection, non-blocking halo exchanges (`MPI_Isend` / `MPI_Irecv`), and zero vehicle loss during boundary migrations.
- **Hybrid Orchestration (MPI + OpenMP):** Non-blocking communication-computation overlap with load imbalance ($\lambda$) and communication ratio ($R_{\text{comm}}$) telemetry.
- **GPU Acceleration (CUDA Extension):** Structure of Arrays (SoA) layout with deterministic 2-phase `atomicMin` collision arbitration, running resident in device memory with seamless host fallback for non-CUDA platforms.
- **Decoupled Geospatial Dashboard:** Interactive WebGL visualization built on Streamlit and PyDeck with time-scrubber playback and real-time congestion classification.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KarachiFlow Architecture                        │
└────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────┐                   ┌──────────────────────┐
│  Road Network Topology      │                   │ Microscopic Physics  │
│  - 74,914 Directed Segments │                   │ - 7.5m Cell Grid     │
│  - 1,074,814 Spatial Cells  │                   │ - Discrete NaSch CA  │
│  - 9 Origin/Destination Hubs│                   │ - Double Buffering   │
│  - Zero-Copy Mmap Binary    │                   │ - Yellow-Box Merges  │
└─────────────────────────────┘                   └──────────────────────┘
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Parallel Computing Frameworks                     │
├──────────────────────────┬─────────────────────────┬───────────────────┤
│ Shared-Memory (OpenMP)   │ Distributed (MS-MPI)    │ Hybrid Orchestration│
│ - 64-byte Cache Padding  │ - Recursive Bisection   │ - MPI_THREAD_     │
│ - Dynamic/Static Sched   │ - Non-Blocking Halos    │   FUNNELED        │
│ - Deterministic Hash PRNG│ - Communication Overlap │ - Inter-Node Halo │
│ - 100% Bit-Exact Match   │ - Exact Conservation    │ - Intra-Node Pool │
└──────────────────────────┴─────────────────────────┴───────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌─────────────────────────────┐                   ┌──────────────────────┐
│  Geospatial Visualization   │                   │ GPU Acceleration     │
│  - Streamlit + PyDeck WebGL │                   │ - Structure of Arrays│
│  - CartoDB Dark Basemap     │                   │ - 2-Phase Kernel /   │
│  - RGBA Congestion States   │                   │   Host CPU Fallback  │
│  - Plotly PDC Scaling Curves│                   │ - atomicMin Arbitrate│
└─────────────────────────────┘                   └──────────────────────┘
```

---

## Quickstart

### Prerequisites
- **C++ Compiler:** C++17 compatible compiler (MSVC 2022, MinGW-w64 GCC 14+, or Clang).
- **Build System:** CMake >= 3.20.
- **Python:** Python 3.10 to 3.13.
- *(Optional)* **MPI:** Microsoft MPI (MS-MPI v10+) or OpenMPI.
- *(Optional)* **GPU:** NVIDIA CUDA Toolkit 12+ (RTX 4000 / Ada Lovelace / Ampere).

### 1. Clone the Repository
```bash
git clone https://github.com/AbdulRafayMughal86/KarachiFlow.git
cd KarachiFlow
```

### 2. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Build C++ Targets
```bash
# Configure CMake build directory
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile all simulation binaries
cmake --build build --config Release
```

All compiled binaries are generated in `bin/`:
- `bin/karachiflow`: Unified production simulation CLI.
- `bin/test_network_reader`: Graph binary ingestion verification.
- `bin/test_nasch_validation`: Microscopic NaSch physics test.
- `bin/test_openmp_scaling`: OpenMP multi-thread scaling & determinism test.
- `bin/test_mpi_simulation`: Distributed-memory MPI domain decomposition test.
- `bin/test_hybrid_simulation`: Hybrid MPI + OpenMP orchestration test.
- `bin/test_cuda_simulation`: GPU Structure-of-Arrays & collision arbitration test.

---

## Running the Simulation

Execute `bin/karachiflow` with flexible engine and workload options:

```bash
# Fast developer mode (5,000 vehicles, 60s runtime < 1s)
./bin/karachiflow --dev

# OpenMP shared-memory execution across 8 threads
./bin/karachiflow --threads 8 --cars 50000 --steps 60

# Distributed-memory MPI execution across 4 processes
mpiexec -n 4 ./bin/karachiflow --mpi --cars 50000 --steps 60

# Hybrid orchestration (2 MPI ranks x 4 OpenMP threads = 8 workers)
mpiexec -n 2 ./bin/karachiflow --hybrid --threads 4 --cars 50000 --steps 60

# GPU Acceleration engine (runs on GPU if available, or CPU SoA fallback)
./bin/karachiflow --cuda --cars 50000 --steps 60

# Export periodic telemetry snapshots for dashboard playback (every 10s)
./bin/karachiflow --cuda --cars 50000 --steps 60 --snapshot-interval 10
```

### CLI Command Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--dev` | Fast developer mode (5k cars, 60s, <1s runtime) | Disabled |
| `--cars <N>` | Total target vehicle pool | 5,000 (dev) / 100,000 (full) |
| `--steps <S>` | Simulation timesteps (seconds) | 60 (dev) / 3,600 (full) |
| `--threads <T>` | OpenMP thread count | All available cores |
| `--schedule <sched> [chunk]` | OpenMP loop schedule (`static`, `dynamic`, `guided`) | `dynamic 64` |
| `--mpi` | Enable distributed-memory MPI engine | Disabled |
| `--hybrid` | Enable hybrid MPI + OpenMP orchestration | Disabled |
| `--cuda` | Enable GPU acceleration engine | Disabled |
| `--sequential` | Force single-threaded baseline reference | Disabled |
| `--snapshot-interval <sec>` | Telemetry snapshot frequency (0 to disable) | 10 |
| `--snapshots <dir>` | Directory for telemetry snapshot CSVs | `data/snapshots` |

---

## Interactive Dashboard

Launch the decoupled Streamlit + PyDeck geospatial visualization:

```bash
streamlit run scripts/dashboard.py
```
Open **`http://localhost:8501`** in your browser to inspect:
- **WebGL Map View:** 74,914 road segments rendered on a dark basemap.
- **Live Congestion States:** Color-coded arterial flow (Green: free-flow, Amber: moderate, Red: severe queue).
- **Time-Scrubber Replay:** Scrub through simulation timesteps ($t = 0 \to 120\text{s}$) to observe bottleneck shockwaves.
- **PDC Metrics Analytics:** Interactive Plotly charts for Speedup ($S_p$), Parallel Efficiency ($E_p$), Communication Overhead ($R_{\text{comm}}$), and Load Imbalance ($\lambda$).

---

## Benchmarking & Performance Evaluation

Run the automated benchmarking harness to perform strong scaling sweeps and generate publication-quality figures:

```bash
# Run on laptop / multi-core CPU (takes < 5 seconds)
python scripts/run_benchmarks.py --scale small --skip-cuda

# Run full evaluation across OpenMP, MPI, Hybrid, and GPU
python scripts/run_benchmarks.py --scale medium
```

### Generated Artifacts
- `data/benchmarks/scaling_data.csv`: Complete timing records across all configurations.
- `data/benchmarks/benchmark_summary.json`: Machine-readable performance summary.
- `data/benchmarks/scaling_curves.png`: Publication-grade Speedup ($S_p$) and Efficiency ($E_p$) comparison curves.

---

## Verification & Test Suite

KarachiFlow includes a comprehensive automated test suite verifying 100% mathematical and architectural correctness:

```bash
# 1. Road graph binary loader & zero-copy mmap
./bin/test_network_reader

# 2. NaSch fundamental diagram & microscopic physics
./bin/test_nasch_validation

# 3. OpenMP 100% bit-exact determinism across thread counts
./bin/test_openmp_scaling

# 4. Distributed MPI non-blocking halo exchanges & zero boundary loss
mpiexec -n 4 ./bin/test_mpi_simulation

# 5. Hybrid orchestration scaling & latency hiding
mpiexec -n 2 ./bin/test_hybrid_simulation

# 6. GPU collision arbitration & zero double-occupancy invariant (50k cars)
./bin/test_cuda_simulation 50000

# 7. Python automated unit test suite (18/18 tests)
python -m unittest discover tests
```

---

## Tech Stack

- **Core Engine:** C++17
- **Shared-Memory Parallelism:** OpenMP 4.5+
- **Distributed-Memory Computing:** MS-MPI / OpenMPI
- **GPU Acceleration:** NVIDIA CUDA (Compute Capability 8.9 `sm_89` / Ada Lovelace)
- **Data & Visuals:** Python (Streamlit, PyDeck, Plotly, OSMnx, GeoPandas, NetworkX)

---

## License

This project is licensed under the [MIT License](LICENSE).
