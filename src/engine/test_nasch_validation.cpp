#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <cmath>
#include "cell.hpp"
#include "vehicle.hpp"

namespace karachiflow {

// Validates the fundamental diagram of traffic flow on a closed single-lane ring road.
struct FundamentalDiagramPoint {
    float density;
    float avg_speed;
    float flow;
};

std::vector<FundamentalDiagramPoint> run_fundamental_diagram_test(
    int road_length = 1000,
    uint8_t v_max = 3,
    float p_slow = 0.2f,
    int warmup_steps = 150,
    int measure_steps = 200
) {
    const std::vector<float> test_densities = {
        0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f
    };

    std::vector<FundamentalDiagramPoint> results;

    for (float rho : test_densities) {
        const int num_vehicles = std::max(1, static_cast<int>(std::round(rho * road_length)));
        const float actual_density = static_cast<float>(num_vehicles) / static_cast<float>(road_length);

        std::vector<int32_t> current_cells(road_length, EMPTY_CELL);
        std::vector<int32_t> next_cells(road_length, EMPTY_CELL);
        std::vector<int> positions(num_vehicles);
        std::vector<uint8_t> velocities(num_vehicles, 0);

        // Distribute vehicles evenly along the ring
        for (int i = 0; i < num_vehicles; ++i) {
            const int pos = (i * road_length) / num_vehicles;
            positions[i] = pos;
            current_cells[pos] = i;
        }

        const int total_steps = warmup_steps + measure_steps;
        double accumulated_speed_sum = 0.0;
        uint64_t accumulated_samples = 0;

        for (int step = 0; step < total_steps; ++step) {
            // Check for collision invariants in current buffer
            int occupied_count = 0;
            for (int c : current_cells) {
                if (c != EMPTY_CELL) occupied_count++;
            }
            assert(occupied_count == num_vehicles);

            for (int i = 0; i < num_vehicles; ++i) {
                const int pos = positions[i];
                uint8_t v = velocities[i];

                // 1. Acceleration
                v = std::min(static_cast<uint8_t>(v + 1), v_max);

                // 2. Deceleration: distance to lead car on ring
                int lead_dist = road_length;
                for (int d = 1; d <= v; ++d) {
                    const int next_pos = (pos + d) % road_length;
                    if (current_cells[next_pos] != EMPTY_CELL) {
                        lead_dist = d;
                        break;
                    }
                }

                if (lead_dist <= v) {
                    v = static_cast<uint8_t>(lead_dist - 1);
                }

                // 3. Stochastic slowing
                if (v > 0 && should_slow(static_cast<uint32_t>(i), static_cast<uint32_t>(step), p_slow)) {
                    v--;
                }

                // 4. Movement
                const int new_pos = (pos + v) % road_length;
                assert(next_cells[new_pos] == EMPTY_CELL); // Invariant: no collision
                next_cells[new_pos] = i;

                positions[i] = new_pos;
                velocities[i] = v;

                if (step >= warmup_steps) {
                    accumulated_speed_sum += v;
                    accumulated_samples++;
                }
            }

            std::swap(current_cells, next_cells);
            std::fill(next_cells.begin(), next_cells.end(), EMPTY_CELL);
        }

        const float mean_speed = static_cast<float>(accumulated_speed_sum / accumulated_samples);
        const float flow = actual_density * mean_speed;

        results.push_back({actual_density, mean_speed, flow});
    }

    return results;
}

} // namespace karachiflow

int main() {
    std::cout << "========================================================\n";
    std::cout << " KarachiFlow — NaSch Fundamental Diagram Validation    \n";
    std::cout << "========================================================\n";
    std::cout << "Executing single-road ring simulation (L=1000 cells, vmax=3, p=0.2)...\n\n";

    const auto results = karachiflow::run_fundamental_diagram_test();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Density (rho) | Avg Speed (v) | Traffic Flow (q) | Regime\n";
    std::cout << "---------------+---------------+------------------+--------------------\n";

    float max_flow = 0.0f;
    float peak_density = 0.0f;

    for (const auto& pt : results) {
        if (pt.flow > max_flow) {
            max_flow = pt.flow;
            peak_density = pt.density;
        }
    }

    for (const auto& pt : results) {
        std::string regime;
        if (pt.density <= peak_density * 0.75f) {
            regime = "Free Flow (Linear)";
        } else if (std::abs(pt.density - peak_density) < 0.06f) {
            regime = "Capacity Peak";
        } else {
            regime = "Congested / Queued";
        }

        std::cout << "   " << std::setw(6) << pt.density
                  << "      |    " << std::setw(6) << pt.avg_speed
                  << "     |      " << std::setw(6) << pt.flow
                  << "      | " << regime << "\n";
    }

    std::cout << "\nValidation Analysis:\n";
    std::cout << "  - Peak Flow Capacity:  " << max_flow << " veh/s at density " << peak_density << "\n";
    std::cout << "  - Free Flow at rho=0.05: " << results[0].flow << " (speed: " << results[0].avg_speed << " cells/s)\n";
    std::cout << "  - Congestion at rho=0.80: " << results.back().flow << " (speed: " << results.back().avg_speed << " cells/s)\n";

    // Criteria 1: Free flow speed close to vmax - p_slow (3 - 0.2 = 2.8)
    if (results[0].avg_speed < 2.5f) {
        std::cerr << "[FAIL] Free flow speed too low: " << results[0].avg_speed << "\n";
        return 1;
    }

    // Criteria 2: Flow increases from low density to peak
    if (max_flow <= results[0].flow) {
        std::cerr << "[FAIL] Peak flow did not exceed low density flow!\n";
        return 1;
    }

    // Criteria 3: Flow drops sharply under heavy congestion (rho = 0.80)
    if (results.back().flow >= max_flow * 0.5f) {
        std::cerr << "[FAIL] Congestion did not sharply drop flow: " << results.back().flow << " vs peak " << max_flow << "\n";
        return 1;
    }

    std::cout << "\n[PASS] Fundamental Diagram matches theoretical Nagel-Schreckenberg dynamics!\n";
    std::cout << "========================================================\n";
    return 0;
}
