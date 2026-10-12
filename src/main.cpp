#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include "simulation.hpp"

namespace {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --dev                     Fast developer mode (5,000 cars, 60s, <2s runtime)\n"
              << "  --cars <N>                Total target vehicles (default: 5000 in dev, 100000 in full)\n"
              << "  --steps <S>               Simulation timesteps (seconds) (default: 60 in dev, 3600 in full)\n"
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

    bool cars_set = false;
    bool steps_set = false;
    bool interval_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dev") {
            config.dev_mode = true;
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

    std::cout << "========================================================\n";
    std::cout << " KarachiFlow — Microscopic Traffic Simulation Engine   \n";
    std::cout << "========================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  - Execution Mode:      " << (config.dev_mode ? "Developer Fast-Test (--dev)" : "Full-Scale Benchmark") << "\n";
    std::cout << "  - Target Vehicles:     " << config.target_vehicles << "\n";
    std::cout << "  - Simulation Steps:    " << config.total_steps << " timesteps (dt = 1.0s)\n";
    std::cout << "  - Snapshot Interval:   " << config.snapshot_interval << " s\n";
    std::cout << "  - Random Slowdown p:   " << config.p_slow << "\n";
    std::cout << "  - Network File:        " << config.network_path << "\n";
    std::cout << "  - Metadata File:       " << config.metadata_path << "\n\n";

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

    std::cout << "Beginning simulation step loop (Single-Threaded Baseline $T_1$)...\n";
    sim.run();

    const auto& metrics = sim.metrics();
    std::cout << "\n========================================================\n";
    std::cout << " Simulation Execution Summary ($T_1$ Baseline)\n";
    std::cout << "========================================================\n";
    std::cout << "  - Execution Time ($T_1$): " << std::fixed << std::setprecision(2)
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
        if (metrics.total_sim_time_ms < 2000.0) {
            std::cout << "\n[PASS] Developer mode SLA satisfied: "
                      << metrics.total_sim_time_ms << " ms < 2000 ms (< 2.0s)\n";
        } else {
            std::cerr << "\n[WARN] Developer mode runtime exceeded 2.0s target: "
                      << metrics.total_sim_time_ms << " ms\n";
        }
    }

    std::cout << "========================================================\n";
    return 0;
}
