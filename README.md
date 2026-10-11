<div align="center">

# KarachiFlow

### City-Scale Traffic & Congestion Simulator

KarachiFlow is an urban mobility and microscopic traffic simulator modeled on the road network of Karachi, Pakistan. Built using the Nagel–Schreckenberg cellular automaton model, it simulates vehicle dynamics and rush-hour bottlenecks across major city corridors using high-performance parallel computing.

[Overview](#overview) · [Problem & Motivation](#problem--motivation) · [Core Objectives](#core-objectives) · [Geographic Scope](#geographic-scope) · [Tech Stack](#tech-stack)

</div>

---

## Overview

**KarachiFlow** simulates city-scale traffic flows by modeling individual vehicle dynamics, lane-level interactions, and arterial traffic patterns across Karachi's real road network (extracted via OpenStreetMap).

The simulator is designed to scale across multi-core processors, distributed nodes, and GPUs using a hybrid parallel architecture (OpenMP + MPI + CUDA), enabling fast simulation of millions of vehicle updates.

---

## Problem & Motivation

Karachi is one of the world's most populous megacities, where millions of daily commuters rely on critical arterial corridors such as **Shahrah-e-Faisal**, **University Road**, **M.A. Jinnah Road**, and **Korangi Road**. These arteries experience severe recurring gridlocks during peak morning and evening transit hours.

Understanding bottleneck formation and testing urban mitigation strategies requires granular, city-scale microscopic simulation that remains computationally feasible at scale.

---

## Core Objectives

- **Microscopic Traffic Modeling:** Simulate individual vehicle behaviors (acceleration, deceleration, stochastic slowing, lane changes) using the discrete Nagel–Schreckenberg cellular automaton model.
- **High-Performance Parallelization:** Accelerate city-wide simulation through multi-threading (OpenMP) and spatial domain decomposition (MPI) to scale across hybrid computing architectures.
- **Congestion & Bottleneck Analysis:** Dynamically identify queue formations, shockwave propagation, and critical choke points during simulated rush-hour flows.
- **Scenario Testing:** Model real-world urban disruptions, including road incidents, lane closures, and tidal commuter surges between residential hubs and business districts.

---

## Geographic Scope

The simulation is tailored to Karachi's primary commuter corridors:
- **Major Arterials:** Shahrah-e-Faisal, University Road, Rashid Minhas Road, Korangi Industrial Corridor.
- **Commuter Inflows:** Connecting residential sectors (Gulshan-e-Iqbal, Gulistan-e-Johar, North Nazimabad, Federal B Area) with key commercial zones (I.I. Chundrigar Road, Saddar, Clifton, SITE).

---

## Tech Stack

- **Core Engine:** C++17
- **Parallel Computing:** OpenMP, MPI (Hybrid Distributed-Shared Memory)
- **GPU Acceleration (Extension):** CUDA
- **Road Network & Analysis:** Python (OSMnx, NetworkX, NumPy, Matplotlib)

---

## License

This project is licensed under the [MIT License](LICENSE).
