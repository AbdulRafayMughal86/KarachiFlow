#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "simulation.hpp"
#include "openmp_engine.hpp"
#include "mpi_engine.hpp"
#include "mpi_partitioner.hpp"

#if defined(HAS_MPI) || defined(MPI_VERSION)
#include <mpi.h>
#endif

namespace {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --dev                     Fast developer mode (5,000 cars, 60s, <1s runtime)\n"
              << "  --cars <N>                Total target vehicles (default: 5000 in dev, 100000 in full)\n"
              << "  --steps <S>               Simulation timesteps (seconds) (default: 60 in dev, 3600 in full)\n"
              << "  --mpi                     Run distributed-memory MPI engine\n"
              << "  --threads <T>             OpenMP thread count (default: all available cores)\n"
              << "  --schedule <type> [chunk] OpenMP loop schedule: static, dynamic, guided (default: dynamic 64)\n"
              << "  --chunk-size <N>          OpenMP scheduling chunk size (default: 64)\n"
              << "  --sequential              Force single-threaded sequential baseline engine\n"
              << "  --benchmark               Run multi-thread scaling study across 1, 2, 4, 8 threads\n"
              << "  --snapshot-interval <sec> Telemetry snapshot interval (default: 10, 0 to disable)\n"
              << "  --snapshots <dir>         Directory for telemetry CSVs (default: data/snapshots)\n"
              << "  --network <path>          Binary road graph path (default: data/karachi_network.bin)\n"
              << "  --metadata <path>         Network metadata JSON path (default: data/karachi_network_metadata.json)\n"
              << "  --p-slow <float>          NaSch stochastic slowing probability (default: 0.2)\n"
              << "  --help                    Show this help message\n";
}

} // namespace

int main(int argc, char* argv[]) {
    karachiflow::SimulationConfig config;
    karachiflow::OpenMPConfig omp_config;

    bool cars_set = false;
    bool steps_set = false;
    bool interval_set = false;
    bool use_sequential = false;
    bool run_benchmark = false;
    bool use_mpi = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dev") {
            config.dev_mode = true;
        } else if (arg == "--mpi") {
            use_mpi = true;
        } else if (arg == "--sequential") {
            use_sequential = true;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            omp_config.thread_count = std::stoi(argv[++i]);
        } else if (arg == "--schedule" && i + 1 < argc) {
            std::string sched_str = argv[++i];
            std::transform(sched_str.begin(), sched_str.end(), sched_str.begin(), ::tolower);
            if (sched_str == "static") omp_config.schedule = karachiflow::OpenMPSchedule::Static;
            else if (sched_str == "dynamic") omp_config.schedule = karachiflow::OpenMPSchedule::Dynamic;
            else if (sched_str == "guided") omp_config.schedule = karachiflow::OpenMPSchedule::Guided;
            else if (sched_str == "auto") omp_config.schedule = karachiflow::OpenMPSchedule::Auto;

            // Optional chunk size as immediate next parameter if numeric
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                omp_config.chunk_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            omp_config.chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--cars" && i + 1 < argc) {
            config.target_vehicles = static_cast<uint32_t>(std::stoul(argv[++i]));
            cars_set = true;
        } else if (arg == "--steps" && i + 1 < argc) {
            config.total_steps = static_cast<uint32_t>(std::stoul(argv[++i]));
            steps_set = true;
        } else if (arg == "--snapshot-interval" && i + 1 < argc) {
            config.snapshot_interval = static_cast<uint32_t>(std::stoul(argv[++i]));
            interval_set = true;
        } else if (arg == "--snapshots" && i + 1 < argc) {
            config.snapshot_dir = argv[++i];
        } else if (arg == "--network" && i + 1 < argc) {
            config.network_path = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            config.metadata_path = argv[++i];
        } else if (arg == "--p-slow" && i + 1 < argc) {
            config.p_slow = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config.dev_mode) {
        if (!cars_set) config.target_vehicles = 5000;
        if (!steps_set) config.total_steps = 60;
        if (!interval_set) config.snapshot_interval = 10;
    } else {
        if (!cars_set) config.target_vehicles = 100000;
        if (!steps_set) config.total_steps = 3600;
        if (!interval_set) config.snapshot_interval = 30;
    }

    if (run_benchmark && !interval_set) {
        config.snapshot_interval = 0;
    }

    int mpi_rank = 0;
    int mpi_size = 1;
#if defined(HAS_MPI) || defined(MPI_VERSION)
    int mpi_init_status = 0;
    MPI_Initialized(&mpi_init_status);
    if (!mpi_init_status) {
        MPI_Init(&argc, &argv);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size > 1 && !use_sequential && !run_benchmark) {
        use_mpi = true;
    }

    if (mpi_size > 1 && (use_sequential || run_benchmark) && mpi_rank != 0) {
        MPI_Finalize();
        return 0;
    }
#endif

    if (mpi_rank == 0) {
        std::cout << "========================================================\n";
        std::cout << " KarachiFlow — Microscopic Traffic Simulation Engine   \n";
        std::cout << "========================================================\n";
        std::cout << "Configuration:\n";
        std::cout << "  - Execution Mode:      " << (config.dev_mode ? "Developer Fast-Test (--dev)" : "Full-Scale Production") << "\n";
        std::cout << "  - Target Vehicles:     " << config.target_vehicles << "\n";
        std::cout << "  - Simulation Steps:    " << config.total_steps << " timesteps (dt = 1.0s)\n";
        std::cout << "  - Snapshot Interval:   " << config.snapshot_interval << " s\n";
        std::cout << "  - Random Slowdown p:   " << config.p_slow << "\n";
        std::cout << "  - Network File:        " << config.network_path << "\n";
        std::cout << "  - Metadata File:       " << config.metadata_path << "\n";
    }

    if (use_mpi) {
        karachiflow::MPIEngine mpi_sim;
        std::string err_msg;

        const auto t_init_start = std::chrono::high_resolution_clock::now();
        if (!mpi_sim.initialize(config, true, &err_msg)) {
            if (mpi_rank == 0) {
                std::cerr << "[ERROR] Distributed MPI simulation initialization failed: " << err_msg << "\n";
            }
#if defined(HAS_MPI) || defined(MPI_VERSION)
            MPI_Finalize();
#endif
            return 1;
        }
        const auto t_init_end = std::chrono::high_resolution_clock::now();
        const double init_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t_init_end - t_init_start).count()) / 1000.0;

        if (mpi_rank == 0) {
            std::cout << "  - Parallel Engine:     Distributed-Memory MPI (" << mpi_size << " processes)\n";
            std::cout << "  - Boundary Mode:       Non-Blocking Halo Migration (MPI_Isend / MPI_Irecv)\n\n";
            std::cout << "[SUCCESS] Ingestion, spatial partitioning & routing complete in "
                      << std::fixed << std::setprecision(2) << init_ms << " ms\n";
            std::cout << "  - Total Cut Edges:     " << mpi_sim.partition().cut_edges.size() << "\n";
            std::cout << "  - Nodes:               " << mpi_sim.network().node_count() << "\n";
            std::cout << "  - Segments:            " << mpi_sim.network().segment_count() << "\n";
            std::cout << "  - Total 7.5m Cells:    " << mpi_sim.network().total_cells() << "\n\n";
            std::cout << "Beginning distributed simulation step loop...\n";
        }

        mpi_sim.run();

        uint64_t g_active = 0, g_completed = 0, g_injected = 0;
        const bool conserved = mpi_sim.verify_global_conservation(&g_active, &g_completed, &g_injected);

        if (mpi_rank == 0) {
            const auto& metrics = mpi_sim.metrics();
            std::cout << "\n========================================================\n";
            std::cout << " Simulation Execution Summary (Distributed MPI Engine)\n";
            std::cout << "========================================================\n";
            std::cout << "  - MPI Processes:         " << mpi_size << "\n";
            std::cout << "  - Execution Time:        " << std::fixed << std::setprecision(2)
                      << metrics.total_sim_time_ms << " ms ("
                      << (metrics.total_sim_time_ms / 1000.0) << " s)\n";
            std::cout << "  - Simulated Steps:       " << metrics.steps_executed << "\n";
            std::cout << "  - Injected Vehicles:     " << g_injected << "\n";
            std::cout << "  - Active Vehicles:       " << g_active << "\n";
            std::cout << "  - Completed Trips:       " << g_completed << "\n";
            std::cout << "  - Vehicle Conservation:  " << (conserved ? "VERIFIED (100% Conserved)" : "MISMATCH") << "\n";
            std::cout << "  - Max Rank Time:         " << metrics.max_rank_time_ms << " ms\n";
            std::cout << "  - Avg Rank Time:         " << metrics.avg_rank_time_ms << " ms\n";
            std::cout << "  - Load Imbalance:        " << std::setprecision(3)
                      << metrics.load_imbalance_ratio << " (t_max / t_avg)\n";
            std::cout << "  - MPI Waitall Latency:   " << std::fixed << std::setprecision(2)
                      << metrics.comm_wait_time_ms << " ms\n";
            std::cout << "  - Average Step Latency:  " << metrics.avg_step_time_us << " us/step\n";
            std::cout << "  - Cell Update Rate:      " << std::scientific << std::setprecision(3)
                      << metrics.updates_per_second << " cell updates/sec\n";
            std::cout << "  - Average Network Speed: " << std::fixed << std::setprecision(2)
                      << metrics.avg_network_speed << " cells/s ("
                      << (metrics.avg_network_speed * karachiflow::CELLS_PER_SEC_TO_KMH) << " km/h)\n";
            std::cout << "========================================================\n";
        }

#if defined(HAS_MPI) || defined(MPI_VERSION)
        MPI_Finalize();
#endif
        return 0;
    }

    if (run_benchmark) {
        std::cout << "\nRunning Multi-Thread PDC Scaling Benchmark...\n";
        const std::vector<int> thread_counts = {1, 2, 4, 8};
        double t1_baseline_ms = 0.0;

        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << " Threads | Schedule | Chunk | Time (ms) | Speedup Sp | Efficiency Ep | Mcells/s    \n";
        std::cout << "------------------------------------------------------------------------------------\n";

        for (int t : thread_counts) {
            karachiflow::OpenMPConfig bench_omp_cfg = omp_config;
            bench_omp_cfg.thread_count = t;

            karachiflow::OpenMPEngine bench_sim;
            std::string bench_err;
            if (!bench_sim.initialize(config, bench_omp_cfg, &bench_err)) {
                std::cerr << "Benchmark initialization failed for T=" << t << ": " << bench_err << "\n";
                return 1;
            }

            const auto t_start = std::chrono::high_resolution_clock::now();
            bench_sim.run();
            const auto t_end = std::chrono::high_resolution_clock::now();
            const double run_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()) / 1000.0;

            if (t == 1) {
                t1_baseline_ms = run_ms;
            }

            const double sp = (run_ms > 0.0 && t1_baseline_ms > 0.0) ? (t1_baseline_ms / run_ms) : 1.0;
            const double ep = sp / static_cast<double>(t);
            const double mcells_sec = (static_cast<double>(bench_sim.metrics().total_cell_updates) /
                                       (run_ms / 1000.0)) / 1e6;

            std::string sched_name = "Dynamic";
            if (bench_omp_cfg.schedule == karachiflow::OpenMPSchedule::Static) sched_name = "Static";
            else if (bench_omp_cfg.schedule == karachiflow::OpenMPSchedule::Guided) sched_name = "Guided";

            std::cout << " " << std::setw(7) << t << " | "
                      << std::setw(8) << sched_name << " | "
                      << std::setw(5) << bench_omp_cfg.chunk_size << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << run_ms << " | "
                      << std::setw(10) << sp << " | "
                      << std::setw(13) << ep << " | "
                      << std::setw(11) << mcells_sec << "\n";
        }
        std::cout << "------------------------------------------------------------------------------------\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
#endif
        return 0;
    }

    if (use_sequential) {
        std::cout << "  - Parallel Engine:     Single-Threaded Baseline\n\n";

        karachiflow::Simulation sim;
        std::string err_msg;

        const auto t_init_start = std::chrono::high_resolution_clock::now();
        if (!sim.initialize(config, &err_msg)) {
            std::cerr << "[ERROR] Simulation initialization failed: " << err_msg << "\n";
            return 1;
        }
        const auto t_init_end = std::chrono::high_resolution_clock::now();
        const double init_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t_init_end - t_init_start).count()) / 1000.0;

        std::cout << "[SUCCESS] Ingestion & routing precomputations complete in "
                  << std::fixed << std::setprecision(2) << init_ms << " ms\n";
        std::cout << "  - Nodes:               " << sim.network().node_count() << "\n";
        std::cout << "  - Segments:            " << sim.network().segment_count() << "\n";
        std::cout << "  - Total 7.5m Cells:    " << sim.network().total_cells() << "\n\n";

        std::cout << "Beginning simulation step loop (Single-Threaded Baseline)...\n";
        sim.run();

        const auto& metrics = sim.metrics();
        std::cout << "\n========================================================\n";
        std::cout << " Simulation Execution Summary (Single-Threaded Baseline)\n";
        std::cout << "========================================================\n";
        std::cout << "  - Execution Time:        " << std::fixed << std::setprecision(2)
                  << metrics.total_sim_time_ms << " ms ("
                  << (metrics.total_sim_time_ms / 1000.0) << " s)\n";
        std::cout << "  - Simulated Steps:       " << metrics.steps_executed << "\n";
        std::cout << "  - Injected Vehicles:     " << metrics.injected_vehicles << "\n";
        std::cout << "  - Active Vehicles:       " << metrics.active_vehicles << "\n";
        std::cout << "  - Completed Trips:       " << metrics.completed_vehicles << "\n";
        std::cout << "  - Total Cell Updates:    " << metrics.total_cell_updates << "\n";
        std::cout << "  - Average Step Latency:  " << std::fixed << std::setprecision(2)
                  << metrics.avg_step_time_us << " us/step\n";
        std::cout << "  - Cell Update Rate:      " << std::scientific << std::setprecision(3)
                  << metrics.updates_per_second << " cell updates/sec\n";
        std::cout << "  - Average Network Speed: " << std::fixed << std::setprecision(2)
                  << metrics.avg_network_speed << " cells/s ("
                  << (metrics.avg_network_speed * karachiflow::CELLS_PER_SEC_TO_KMH) << " km/h)\n";
        std::cout << "========================================================\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (!mpi_finalized) {
            MPI_Finalize();
        }
#endif
        return 0;
    }

    std::string sched_name = "dynamic";
    if (omp_config.schedule == karachiflow::OpenMPSchedule::Static) sched_name = "static";
    else if (omp_config.schedule == karachiflow::OpenMPSchedule::Guided) sched_name = "guided";

    karachiflow::OpenMPEngine omp_sim;
    std::string err_msg;

    const auto t_init_start = std::chrono::high_resolution_clock::now();
    if (!omp_sim.initialize(config, omp_config, &err_msg)) {
        std::cerr << "[ERROR] Simulation initialization failed: " << err_msg << "\n";
        return 1;
    }
    const auto t_init_end = std::chrono::high_resolution_clock::now();
    const double init_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_init_end - t_init_start).count()) / 1000.0;

    std::cout << "  - Parallel Threads:    " << omp_sim.effective_thread_count() << "\n";
    std::cout << "  - OpenMP Schedule:     " << sched_name << " (chunk: " << omp_config.chunk_size << ")\n\n";

    std::cout << "[SUCCESS] Ingestion & parallel routing precomputations complete in "
              << std::fixed << std::setprecision(2) << init_ms << " ms\n";
    std::cout << "  - Nodes:               " << omp_sim.network().node_count() << "\n";
    std::cout << "  - Segments:            " << omp_sim.network().segment_count() << "\n";
    std::cout << "  - Total 7.5m Cells:    " << omp_sim.network().total_cells() << "\n\n";

    std::cout << "Beginning simulation step loop (OpenMP Shared-Memory Parallel)...\n";
    omp_sim.run();

    const auto& metrics = omp_sim.metrics();
    std::cout << "\n========================================================\n";
    std::cout << " Simulation Execution Summary (OpenMP Parallel Engine)\n";
    std::cout << "========================================================\n";
    std::cout << "  - Threads Used:          " << omp_sim.effective_thread_count() << "\n";
    std::cout << "  - Execution Time:        " << std::fixed << std::setprecision(2)
              << metrics.total_sim_time_ms << " ms ("
              << (metrics.total_sim_time_ms / 1000.0) << " s)\n";
    std::cout << "  - Simulated Steps:       " << metrics.steps_executed << "\n";
    std::cout << "  - Injected Vehicles:     " << metrics.injected_vehicles << "\n";
    std::cout << "  - Active Vehicles:       " << metrics.active_vehicles << "\n";
    std::cout << "  - Completed Trips:       " << metrics.completed_vehicles << "\n";
    std::cout << "  - Total Cell Updates:    " << metrics.total_cell_updates << "\n";
    std::cout << "  - Average Step Latency:  " << std::fixed << std::setprecision(2)
              << metrics.avg_step_time_us << " us/step\n";
    std::cout << "  - Cell Update Rate:      " << std::scientific << std::setprecision(3)
              << metrics.updates_per_second << " cell updates/sec\n";
    std::cout << "  - Average Network Speed: " << std::fixed << std::setprecision(2)
              << metrics.avg_network_speed << " cells/s ("
              << (metrics.avg_network_speed * karachiflow::CELLS_PER_SEC_TO_KMH) << " km/h)\n";

    if (config.dev_mode) {
        if (metrics.total_sim_time_ms < 1000.0) {
            std::cout << "\n[PASS] Multi-threaded developer mode SLA satisfied: "
                      << metrics.total_sim_time_ms << " ms < 1000 ms (< 1.0s)\n";
        } else {
            std::cout << "\n[INFO] Developer mode execution runtime: "
                      << metrics.total_sim_time_ms << " ms\n";
        }
    }

    std::cout << "========================================================\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Finalize();
    }
#endif
    return 0;
}
