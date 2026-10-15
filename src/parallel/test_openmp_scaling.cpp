#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include "simulation.hpp"
#include "openmp_engine.hpp"

namespace {

void check_condition(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "[TEST FAIL] " << msg << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << " KarachiFlow — Shared-Memory OpenMP Scaling Benchmark    \n";
    std::cout << "========================================================\n\n";

    karachiflow::SimulationConfig config;
    config.network_path = "data/karachi_network.bin";
    config.metadata_path = "data/karachi_network_metadata.json";
    config.snapshot_interval = 0; // Disable disk I/O to measure pure compute
    config.target_vehicles = 2000;
    config.total_steps = 30;
    config.p_slow = 0.2f;
    config.dev_mode = true;

    // -------------------------------------------------------------
    // Part 1: Deterministic Verification against Sequential Baseline
    // -------------------------------------------------------------
    std::cout << "1. Running single-threaded baseline simulation...\n";
    karachiflow::Simulation seq_sim;
    std::string err_msg;
    check_condition(seq_sim.initialize(config, &err_msg), "Baseline init error: " + err_msg);

    seq_sim.run();
    const double t1_baseline_ms = seq_sim.metrics().total_sim_time_ms;

    std::cout << "   Baseline execution time (T1): " << std::fixed << std::setprecision(2)
              << t1_baseline_ms << " ms\n";
    std::cout << "   Active vehicles: " << seq_sim.active_vehicle_count()
              << ", Completed: " << seq_sim.completed_vehicle_count()
              << ", Injected: " << seq_sim.injected_vehicle_count() << "\n\n";

    std::cout << "2. Verifying 100% bit-exact determinism across thread counts...\n";
    const std::vector<int> test_threads = {1, 2, 4, 8};

    for (int t : test_threads) {
        karachiflow::OpenMPConfig omp_cfg;
        omp_cfg.thread_count = t;
        omp_cfg.schedule = karachiflow::OpenMPSchedule::Dynamic;
        omp_cfg.chunk_size = 64;

        karachiflow::OpenMPEngine omp_sim;
        check_condition(omp_sim.initialize(config, omp_cfg, &err_msg),
                        "OpenMP engine init error (T=" + std::to_string(t) + "): " + err_msg);

        omp_sim.run();

        // 1. Scalar metric invariants
        check_condition(omp_sim.active_vehicle_count() == seq_sim.active_vehicle_count(),
                        "Active vehicle count mismatch at T=" + std::to_string(t) + ": " +
                        std::to_string(omp_sim.active_vehicle_count()) + " vs " +
                        std::to_string(seq_sim.active_vehicle_count()));
        check_condition(omp_sim.completed_vehicle_count() == seq_sim.completed_vehicle_count(),
                        "Completed trip count mismatch at T=" + std::to_string(t));
        check_condition(omp_sim.injected_vehicle_count() == seq_sim.injected_vehicle_count(),
                        "Injected vehicle count mismatch at T=" + std::to_string(t));

        // 2. Exact grid occupancy
        const auto& seq_cells = seq_sim.current_cells();
        const auto& omp_cells = omp_sim.current_cells();
        check_condition(seq_cells.size() == omp_cells.size(), "Grid cell size mismatch");

        size_t cell_mismatches = 0;
        for (size_t c = 0; c < seq_cells.size(); ++c) {
            if (seq_cells[c] != omp_cells[c]) {
                cell_mismatches++;
            }
        }
        check_condition(cell_mismatches == 0,
                        "Grid cell mismatches at T=" + std::to_string(t) + ": " +
                        std::to_string(cell_mismatches) + " cells differed");

        // 3. Exact per-vehicle state verification
        const auto& seq_vehs = seq_sim.vehicles();
        const auto& omp_vehs = omp_sim.vehicles();
        check_condition(seq_vehs.size() == omp_vehs.size(), "Vehicle pool size mismatch");

        size_t veh_mismatches = 0;
        for (size_t v = 0; v < seq_vehs.size(); ++v) {
            const auto& sv = seq_vehs[v];
            const auto& ov = omp_vehs[v];

            if (sv.active != ov.active ||
                sv.segment_id != ov.segment_id ||
                sv.lane != ov.lane ||
                sv.cell_idx != ov.cell_idx ||
                sv.velocity != ov.velocity ||
                sv.cells_traveled != ov.cells_traveled) {
                veh_mismatches++;
            }
        }
        check_condition(veh_mismatches == 0,
                        "Vehicle state mismatches at T=" + std::to_string(t) + ": " +
                        std::to_string(veh_mismatches) + " vehicles differed");

        std::cout << "   [PASS] T = " << t << " threads: 100% bit-exact match with baseline (0 cell/veh diffs)\n";
    }

    std::cout << "\n3. Conducting PDC Hybrid Scheduling Study (Static vs Dynamic vs Guided)...\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Threads | Schedule | Chunk | Time (ms) | Speedup Sp | Efficiency Ep | Mcells/s    \n";
    std::cout << "------------------------------------------------------------------------------------\n";

    struct ScheduleTest {
        std::string name;
        karachiflow::OpenMPSchedule sched;
        int chunk;
    };

    const std::vector<ScheduleTest> schedules = {
        {"Static",  karachiflow::OpenMPSchedule::Static,  64},
        {"Dynamic", karachiflow::OpenMPSchedule::Dynamic, 16},
        {"Dynamic", karachiflow::OpenMPSchedule::Dynamic, 32},
        {"Dynamic", karachiflow::OpenMPSchedule::Dynamic, 64},
        {"Dynamic", karachiflow::OpenMPSchedule::Dynamic, 128},
        {"Guided",  karachiflow::OpenMPSchedule::Guided,  64}
    };

    // Use higher workload for scheduling study to get high-fidelity timing
    karachiflow::SimulationConfig bench_config = config;
    bench_config.target_vehicles = 10000;
    bench_config.total_steps = 40;

    // Measure T1 with single thread for speedup base
    double bench_t1_ms = 0.0;
    {
        karachiflow::OpenMPConfig omp_cfg;
        omp_cfg.thread_count = 1;
        omp_cfg.schedule = karachiflow::OpenMPSchedule::Static;
        omp_cfg.chunk_size = 64;

        karachiflow::OpenMPEngine omp_sim;
        omp_sim.initialize(bench_config, omp_cfg);
        omp_sim.run();
        bench_t1_ms = omp_sim.metrics().total_sim_time_ms;
    }

    for (int t : test_threads) {
        for (const auto& s : schedules) {
            karachiflow::OpenMPConfig omp_cfg;
            omp_cfg.thread_count = t;
            omp_cfg.schedule = s.sched;
            omp_cfg.chunk_size = s.chunk;

            karachiflow::OpenMPEngine omp_sim;
            omp_sim.initialize(bench_config, omp_cfg);

            omp_sim.run();
            const double tp_ms = omp_sim.metrics().total_sim_time_ms;

            const double sp = (tp_ms > 0.0) ? (bench_t1_ms / tp_ms) : 1.0;
            const double ep = sp / static_cast<double>(t);
            const double mcells_sec = (static_cast<double>(omp_sim.metrics().total_cell_updates) /
                                       (tp_ms / 1000.0)) / 1e6;

            std::cout << " " << std::setw(7) << t << " | "
                      << std::setw(8) << s.name << " | "
                      << std::setw(5) << s.chunk << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << tp_ms << " | "
                      << std::setw(10) << sp << " | "
                      << std::setw(13) << ep << " | "
                      << std::setw(11) << mcells_sec << "\n";
        }
    }
    std::cout << "------------------------------------------------------------------------------------\n\n";

    // -------------------------------------------------------------
    // Part 3: Developer SLA Fast-Test Validation
    // -------------------------------------------------------------
    std::cout << "4. Verifying Developer Mode SLA (< 1.0s on multi-threading)...\n";
    karachiflow::SimulationConfig dev_config;
    dev_config.network_path = "data/karachi_network.bin";
    dev_config.metadata_path = "data/karachi_network_metadata.json";
    dev_config.target_vehicles = 5000;
    dev_config.total_steps = 60;
    dev_config.snapshot_interval = 0;
    dev_config.dev_mode = true;

    karachiflow::OpenMPConfig dev_omp_cfg;
    dev_omp_cfg.thread_count = 4;
    dev_omp_cfg.schedule = karachiflow::OpenMPSchedule::Dynamic;
    dev_omp_cfg.chunk_size = 64;

    karachiflow::OpenMPEngine dev_sim;
    check_condition(dev_sim.initialize(dev_config, dev_omp_cfg, &err_msg),
                    "Dev test initialization failed: " + err_msg);

    dev_sim.run();
    const double dev_runtime_ms = dev_sim.metrics().total_sim_time_ms;

    std::cout << "   Multi-threaded developer mode runtime (5,000 cars, 60s, 4 threads): "
              << std::fixed << std::setprecision(2) << dev_runtime_ms << " ms\n";

    check_condition(dev_runtime_ms < 1000.0,
                    "Developer mode SLA violated: runtime " + std::to_string(dev_runtime_ms) + " ms >= 1000 ms");

    std::cout << "   [PASS] Developer SLA satisfied (" << dev_runtime_ms << " ms < 1000.0 ms)\n\n";

    std::cout << "[PASS] All shared-memory OpenMP scaling and deterministic tests passed successfully!\n";
    std::cout << "========================================================\n";
    return 0;
}
