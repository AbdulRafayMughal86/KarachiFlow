# KarachiFlow: Screenshot & Visual Portfolio Catalog

> High-resolution visual portfolio captured from the live WebGL GIS Telemetry Dashboard and HPC test harnesses on the University Lab Workstation (Intel Core i7 14th Gen + NVIDIA RTX 4000 Ada 20GB).  
> All figures are formatted for direct inclusion into technical documentation, research reports, and presentations.

---

## 1. Tab 1: City Replay & Microscopic Congestion Dynamics

| Figure / Screenshot | Resolution | Primary Visual Subject | Description / Caption |
|---|:---:|---|---|
| **`dashboard_city_replay_full.png`** | 2400 × 1600 | Full UI canvas with dark-matter CartoDB basemap, Top Navigation bar, 6 KPI metric cards, dense traffic flows (100k vehicles), and unified Simulation Transport Deck. | **Figure 1:** Executive overview of the KarachiFlow GIS Telemetry Dashboard during morning peak commute on Karachi Urban Core. |
| **`dashboard_pydeck_map_canvas.png`** | 2280 × 612 | Pure PyDeck WebGL layer displaying 74,914 directed segments with RGBA congestion styling: Green (Free Flow $\ge 70\%$), Amber (Moderate Slowing $35\text{--}70\%$), Red (Arterial Bottleneck $5\text{--}35\%$), and Magenta (Gridlock Deadlocks $\le 5\%$). | **Figure 2:** Multi-lane microscopic traffic distribution and queue formation across Shahrah-e-Faisal and Saddar corridors. |
| **`dashboard_executive_kpi_ribbon.png`** | 2280 × 140 | 6 High-impact glassmorphic metric cards: Active Vehicles (98,133), Completed Trips (1,867), Average Network Speed (9.7 km/h), Jam Density (21.4%), Congestion Status, and Throughput. | **Figure 3:** Real-time macroscopic traffic telemetry metrics and network capacity utilization. |
| **`dashboard_transport_timeline_deck.png`** | 2280 × 80 | Unified playback transport dock featuring Play/Pause, Step Advance, continuous timeline slider ($t = 0\text{s}$ to $120\text{s}$), and color-coded status legend. | **Figure 4:** Interactive simulation transport deck and time-scrubber controls. |
| **`dashboard_bottleneck_heatmap_overlay.png`** | 2400 × 1600 | Heatmap layer overlay with origin/destination commuter zone boundaries. | **Figure 5:** Interactive Bottleneck Heatmap overlay and Origin-Destination residential/commercial zone bounding boxes. |

---

## 2. Tab 2: Parallel Computing Scalability & Benchmarks

| Figure / Screenshot | Resolution | Primary Visual Subject | Description / Caption |
|---|:---:|---|---|
| **`benchmark_scaling_efficiency_overview.png`** | 2400 × 1600 | Tab 2 upper view displaying the Active Architecture Focus KPI card, Strong Scaling Speedup curve ($S_p = T_1 / T_p$), and Parallel Efficiency curve ($E_p = S_p / p$) vs. ideal linear benchmarks. | **Figure 6:** Strong scaling speedup and parallel efficiency across OpenMP (1–20 threads), MS-MPI (1–8 ranks), and Hybrid orchestration. |
| **`benchmark_summary_matrix_table.png`** | 2400 × 850 | Complete publication-grade performance summary table containing Columns: Engine, Workers, Threads, Ranks, Schedule, Chunk, Vehicles, Steps, Runtime (ms), Speedup vs $T_1$, Efficiency, and Throughput. | **Figure 7:** Comprehensive multi-paradigm performance evaluation matrix across all hardware configurations. |

---

## 3. Tab 3: What-If Urban Mobility & HPC Physics

| Figure / Screenshot | Resolution | Primary Visual Subject | Description / Caption |
|---|:---:|---|---|
| **`what_if_incident_weak_scaling_overview.png`** | 2400 × 1600 | Tab 3 overview displaying the Shahrah-e-Faisal 2-lane bottleneck incident analysis, arterial transit velocity drop ($-24.8\%$), and weak scaling workload curves. | **Figure 8:** What-If urban mobility bottleneck analysis: upstream shockwave propagation and trip completion delay on Shahrah-e-Faisal. |

---

## 4. Hardware Verification & Console Test Harnesses

| Figure / Screenshot | Resolution | Primary Visual Subject | Description / Caption |
|---|:---:|---|---|
| **`terminal_hpc_verification_tests.png`** | 1600 × 1140 | High-resolution terminal capture of the 4-stage C++ verification suite: (1) NaSch Fundamental Diagram, (2) OpenMP Bit-Exact Determinism, (3) MS-MPI Domain Decomposition & Handover, and (4) CUDA GPU 100% Conservation & Zero Collisions. | **Figure 9:** Empirical test suite verification output validating zero collisions, bit-exact determinism, and 100% vehicle conservation. |

---

## 5. Direct High-Resolution Matplotlib Plot Artifacts (`data/benchmarks/`)

In addition to the UI dashboard screenshots, publication-grade 300 DPI vector charts are directly accessible in `data/benchmarks/`:

- **`data/benchmarks/multi_tier_3600s_comparison.png`**: Side-by-side runtime and speedup comparison across all three geographic tiers (Tier 1 500k, Tier 2 1M, Tier 3 2M) for 3,600 continuous simulation steps.
- **`data/benchmarks/scaling_curves.png`**: Multi-panel Speedup ($S_p$) and Efficiency ($E_p$) comparison with ideal linear reference lines.
- **`data/benchmarks/gpu_comparison.png`**: Multi-tier GPU speedup leap comparing Sequential CPU vs. Best Multi-Core CPU vs. RTX 4000 Ada GPU.
- **`data/benchmarks/what_if_comparison.png`**: Arterial velocity drop bar chart for Shahrah-e-Faisal incident scenario.
- **`data/benchmarks/tier1_500k_3600s/speedup_scaling.png`**: Dedicated 3,600-step Tier 1 speedup curve with RTX 4000 Ada marker.
- **`data/benchmarks/tier2_1m_3600s/speedup_scaling.png`**: Dedicated 3,600-step Tier 2 speedup curve with RTX 4000 Ada marker (43.41x).
- **`data/benchmarks/tier3_2m_3600s/speedup_scaling.png`**: Dedicated 3,600-step Tier 3 speedup curve with RTX 4000 Ada marker.
- **`data/benchmarks/hour_3600s/hour_gpu_throughput.png`**: Pre-seeded GPU throughput scaling up to 10,000,000 vehicles ($2.68 \times 10^{10}$ cell updates/sec).
