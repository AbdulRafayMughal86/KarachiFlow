#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
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
    std::cout << " KarachiFlow — Full Simulation Engine Test Suite        \n";
    std::cout << "========================================================\n";

    karachiflow::SimulationConfig config;
    config.network_path = "data/karachi_network.bin";
    config.metadata_path = "data/karachi_network_metadata.json";
    config.snapshot_dir = "data/snapshots";
    config.target_vehicles = 1000;
    config.total_steps = 30;
    config.snapshot_interval = 10;
    config.p_slow = 0.2f;
    config.dev_mode = true;

    karachiflow::Simulation sim;
    std::string err_msg;

    std::cout << "1. Initializing simulation...\n";
    check_condition(sim.initialize(config, &err_msg), "Initialization error: " + err_msg);
    std::cout << "   [PASS] Initialized successfully. Total cells: " << sim.network().total_cells() << "\n";

    std::cout << "2. Stepping through 30 timesteps and verifying invariants...\n";
    for (uint32_t step = 1; step <= config.total_steps; ++step) {
        sim.step();

        // Invariant 1: Number of occupied cells in current_cells buffer must equal active_vehicle_count
        const auto& cells = sim.current_cells();
        uint32_t occupied_cells = 0;
        for (int32_t c : cells) {
            if (c != karachiflow::EMPTY_CELL) {
                occupied_cells++;
            }
        }

        check_condition(occupied_cells == sim.active_vehicle_count(),
                        "Occupied cell count (" + std::to_string(occupied_cells) +
                        ") does not match active vehicles (" + std::to_string(sim.active_vehicle_count()) + ")");

        // Invariant 2: Vehicle state consistency with grid
        const auto& vehicles = sim.vehicles();
        const auto& segments = sim.segments();
        for (const auto& v : vehicles) {
            if (v.active) {
                check_condition(v.segment_id >= 0 && static_cast<uint32_t>(v.segment_id) < segments.size(),
                                "Vehicle segment ID out of bounds");
                const auto& seg = segments[v.segment_id];
                check_condition(v.lane >= 0 && v.lane < seg.lanes(), "Vehicle lane out of bounds");
                check_condition(v.cell_idx >= 0 && v.cell_idx < seg.cells_per_lane(),
                                "Vehicle cell out of bounds: id=" + std::to_string(v.id) +
                                ", cell_idx=" + std::to_string(v.cell_idx) +
                                ", cells_per_lane=" + std::to_string(seg.cells_per_lane()) +
                                ", seg_id=" + std::to_string(v.segment_id) +
                                ", step=" + std::to_string(step));

                const size_t g_idx = seg.local_to_global_cell(v.lane, v.cell_idx);
                check_condition(cells[g_idx] == v.id, "Cell grid does not contain vehicle ID");
            }
        }

        // Invariant 3: Vehicle conservation (injected == active + completed)
        const auto& m = sim.metrics();
        check_condition(sim.active_vehicle_count() + m.completed_vehicles == m.injected_vehicles,
                        "Vehicle count conservation violated");

        if (config.snapshot_interval > 0 && (step % config.snapshot_interval == 0)) {
            const std::string snap_path = config.snapshot_dir + "/snapshot_t" + std::to_string(step) + ".csv";
            check_condition(sim.save_snapshot(step, snap_path), "Failed to save snapshot at step " + std::to_string(step));
        }
    }
    check_condition(sim.injected_vehicle_count() == config.target_vehicles,
                    "Target vehicle injection SLA unmet: " + std::to_string(sim.injected_vehicle_count()) +
                    " vs " + std::to_string(config.target_vehicles));
    check_condition(sim.completed_vehicle_count() > 0,
                    "No completed trips observed (destination arrival failed)");
    std::cout << "   [PASS] Invariants verified across all 30 steps ("
              << sim.injected_vehicle_count() << " injected, "
              << sim.completed_vehicle_count() << " completed). Zero collisions detected.\n";

    std::cout << "3. Verifying telemetry snapshots...\n";
    const std::string test_snap = config.snapshot_dir + "/snapshot_t10.csv";
    std::ifstream snap_file(test_snap);
    check_condition(snap_file.is_open(), "Failed to open generated snapshot file: " + test_snap);

    std::string header_line;
    std::getline(snap_file, header_line);
    check_condition(header_line == "segment_id,vehicle_count,avg_speed,density_ratio",
                    "Invalid CSV header: " + header_line);

    uint32_t row_count = 0;
    std::string line;
    uint32_t total_snap_vehicles = 0;

    while (std::getline(snap_file, line)) {
        if (line.empty()) continue;
        row_count++;

        std::stringstream ss(line);
        std::string s_id_str, cnt_str, spd_str, dens_str;
        std::getline(ss, s_id_str, ',');
        std::getline(ss, cnt_str, ',');
        std::getline(ss, spd_str, ',');
        std::getline(ss, dens_str, ',');

        const int s_id = std::stoi(s_id_str);
        const int cnt = std::stoi(cnt_str);
        const float spd = std::stof(spd_str);
        const float dens = std::stof(dens_str);

        check_condition(s_id >= 0, "Negative segment ID in CSV");
        check_condition(cnt >= 0, "Negative vehicle count in CSV");
        check_condition(spd >= 0.0f && spd <= 4.0f, "Average speed out of range [0, 4]");
        check_condition(dens >= 0.0f && dens <= 1.0f, "Density ratio out of range [0, 1]");

        total_snap_vehicles += cnt;
    }

    check_condition(row_count == sim.network().segment_count(), "Snapshot segment row count mismatch");
    std::cout << "   [PASS] Snapshot CSV verified (" << row_count << " road segments, format strictly valid).\n";

    std::cout << "\n[PASS] All simulation engine tests passed successfully!\n";
    std::cout << "========================================================\n";
    return 0;
}
