#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "cell.hpp"
#include "vehicle.hpp"
#include "road_segment.hpp"
#include "road_network.hpp"
#include "demand_loader.hpp"

namespace karachiflow {

struct SimulationConfig {
    std::string network_path{"data/karachi_network.bin"};
    std::string metadata_path{"data/karachi_network_metadata.json"};
    std::string snapshot_dir{"data/snapshots"};
    uint32_t target_vehicles{5000};
    uint32_t total_steps{60};
    uint32_t snapshot_interval{10};
    float p_slow{0.2f};
    bool dev_mode{false};
};

struct SimulationMetrics {
    uint32_t steps_executed{0};
    uint64_t total_cell_updates{0};
    uint32_t active_vehicles{0};
    uint32_t completed_vehicles{0};
    uint32_t injected_vehicles{0};
    double total_sim_time_ms{0.0};
    double avg_step_time_us{0.0};
    double updates_per_second{0.0};
    float avg_network_speed{0.0f};
};

class Simulation {
public:
    Simulation() = default;

    bool initialize(const SimulationConfig& config, std::string* error_msg = nullptr);
    void step();
    void run();

    bool save_snapshot(uint32_t step, const std::string& filepath) const;

    const SimulationConfig& config() const noexcept { return m_config; }
    const SimulationMetrics& metrics() const noexcept { return m_metrics; }
    const RoadNetwork& network() const noexcept { return m_network; }
    const std::vector<RoadSegment>& segments() const noexcept { return m_segments; }
    const std::vector<Vehicle>& vehicles() const noexcept { return m_vehicles; }
    const std::vector<int32_t>& current_cells() const noexcept { return m_current_cells; }

    uint32_t current_step() const noexcept { return m_current_step; }
    uint32_t active_vehicle_count() const noexcept { return m_active_vehicle_count; }
    uint32_t completed_vehicle_count() const noexcept { return m_completed_vehicles; }
    uint32_t injected_vehicle_count() const noexcept { return m_injected_vehicles; }

private:
    void precompute_routing_tables();
    void prepare_demand_candidates();
    void inject_demand();
    void update_vehicles();
    void update_metrics();

    SimulationConfig m_config{};
    SimulationMetrics m_metrics{};
    RoadNetwork m_network;
    DemandMetadata m_demand;

    std::vector<RoadSegment> m_segments;
    std::vector<int32_t> m_current_cells;
    std::vector<int32_t> m_next_cells;
    std::vector<Vehicle> m_vehicles;

    std::vector<int32_t> m_routing_table;
    uint32_t m_dest_zone_count{0};

    std::vector<int32_t> m_origin_candidate_segments;
    std::vector<std::vector<int32_t>> m_active_dest_nodes;
    std::vector<uint8_t> m_is_dest_node;

    mutable std::vector<uint32_t> m_snapshot_counts;
    mutable std::vector<uint32_t> m_snapshot_speeds;

    uint32_t m_current_step{0};
    uint32_t m_active_vehicle_count{0};
    uint32_t m_completed_vehicles{0};
    uint32_t m_injected_vehicles{0};
};

} // namespace karachiflow
