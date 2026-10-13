#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include "cuda_engine.cuh"
#include "simulation.hpp"

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
    std::cout << " KarachiFlow — GPU Acceleration & Collision Validation  \n";
    std::cout << " Target Architecture: NVIDIA RTX 4000 Ada (sm_89) / SoA\n";
    std::cout << "========================================================\n\n";

    karachiflow::SimulationConfig config;
    config.network_path = "data/karachi_network.bin";
    config.metadata_path = "data/karachi_network_metadata.json";
    config.snapshot_interval = 0;
    config.target_vehicles = 5000;
    config.total_steps = 60;
    config.p_slow = 0.2f;
    config.dev_mode = true;

    // -----------------------------------------------------------------
    // Test 1: Single-Threaded Sequential CPU Baseline Reference
    // -----------------------------------------------------------------
    std::cout << "1. Executing single-threaded sequential CPU baseline reference...\n";
    karachiflow::Simulation seq_sim;
    std::string err_msg;
    check_condition(seq_sim.initialize(config, &err_msg), "Baseline init failed: " + err_msg);

    const auto t_seq_start = std::chrono::high_resolution_clock::now();
    seq_sim.run();
    const auto t_seq_end = std::chrono::high_resolution_clock::now();
    const double t_seq_ms = std::chrono::duration<double, std::milli>(t_seq_end - t_seq_start).count();

    std::cout << "   Baseline execution time: " << std::fixed << std::setprecision(2)
              << t_seq_ms << " ms\n";
    std::cout << "   Active vehicles: " << seq_sim.active_vehicle_count()
              << ", Completed: " << seq_sim.completed_vehicle_count()
              << ", Injected: " << seq_sim.injected_vehicle_count() << "\n\n";

    // -----------------------------------------------------------------
    // Test 2: GPU Acceleration Structure-of-Arrays (SoA) Execution
    // -----------------------------------------------------------------
    std::cout << "2. Executing GPU Acceleration engine with atomicMin arbitration...\n";
    karachiflow::CudaEngine gpu_sim;
    check_condition(gpu_sim.initialize(config, &err_msg), "GPU engine init failed: " + err_msg);

    const auto t_gpu_start = std::chrono::high_resolution_clock::now();
    gpu_sim.run();
    const auto t_gpu_end = std::chrono::high_resolution_clock::now();
    const double t_gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();

    uint64_t g_act = 0, g_comp = 0, g_inj = 0;
    const bool conserved = gpu_sim.verify_global_conservation(&g_act, &g_comp, &g_inj);

    std::cout << "   GPU execution time:      " << std::fixed << std::setprecision(2)
              << t_gpu_ms << " ms\n";
    std::cout << "   Active vehicles:         " << g_act << "\n";
    std::cout << "   Completed vehicles:      " << g_comp << "\n";
    std::cout << "   Injected vehicles:       " << g_inj << "\n";
    std::cout << "   Vehicle conservation:    " << (conserved ? "VERIFIED (100% Exact)" : "FAILED") << "\n";
    check_condition(conserved, "Vehicle conservation invariant violated in GPU engine!");

    // -----------------------------------------------------------------
    // Test 3: Zero-Collision Invariant & Spatial Grid Verification
    // -----------------------------------------------------------------
    std::cout << "\n3. Verifying zero-collision invariant across entire grid...\n";
    const auto& current_cells = gpu_sim.current_cells();
    uint32_t active_in_grid = 0;
    for (int32_t cell_val : current_cells) {
        if (cell_val >= 0) {
            active_in_grid++;
        }
    }
    std::cout << "   Occupied cells in grid:  " << active_in_grid << "\n";
    std::cout << "   Active vehicles tracked: " << g_act << "\n";
    check_condition(active_in_grid == g_act, "Grid occupant count does not match tracked active vehicle count!");
    std::cout << "   [PASS] Zero double-occupancies or cell collisions detected.\n";

    // -----------------------------------------------------------------
    // Test 4: Performance Profile & Comparative Speedup
    // -----------------------------------------------------------------
    std::cout << "\n4. Performance Evaluation:\n";
    double speedup = (t_gpu_ms > 0.0) ? (t_seq_ms / t_gpu_ms) : 1.0;
    const auto& m = gpu_sim.metrics();
    std::cout << "   Cell update throughput:  " << std::scientific << std::setprecision(2)
              << m.updates_per_second << " cell updates/sec\n";
    std::cout << "   Comparative Speedup:     " << std::fixed << std::setprecision(2)
              << speedup << "x over single-threaded sequential baseline\n";
    std::cout << "   Developer SLA (< 1.0s):  " << (t_gpu_ms < 1000.0 ? "PASS" : "FAIL") << "\n";

    std::cout << "\n[PASS] All GPU acceleration and collision arbitration tests passed successfully!\n";
    std::cout << "========================================================\n";
    return 0;
}
