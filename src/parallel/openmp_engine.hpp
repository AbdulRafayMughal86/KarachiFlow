#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "cell.hpp"
#include "vehicle.hpp"
#include "road_segment.hpp"
#include "road_network.hpp"
#include "demand_loader.hpp"
#include "simulation.hpp"

namespace karachiflow {

enum class OpenMPSchedule {
    Static,
    Dynamic,
    Guided,
    Auto
};

struct OpenMPConfig {
    int thread_count{0};
    OpenMPSchedule schedule{OpenMPSchedule::Dynamic};
    int chunk_size{64};
};

struct BoundaryCandidate {
    int32_t id{-1};
    int16_t speed{0};
    int16_t d_seg{0};
};

struct alignas(64) ThreadWorkspace {
    std::vector<BoundaryCandidate> boundary_candidates;
    uint64_t cell_updates{0};
    uint64_t total_velocity{0};
    char cache_padding[64];
};

class OpenMPEngine {
public:
    OpenMPEngine() = default;

    bool initialize(const SimulationConfig& config,
                    const OpenMPConfig& omp_config = {},
                    std::string* error_msg = nullptr);
    void step();
    void run();

    bool save_snapshot(uint32_t step, const std::string& filepath) const;

    const SimulationConfig& config() const noexcept { return m_config; }
    const OpenMPConfig& omp_config() const noexcept { return m_omp_config; }
    const SimulationMetrics& metrics() const noexcept { return m_metrics; }
    const RoadNetwork& network() const noexcept { return m_network; }
    const std::vector<RoadSegment>& segments() const noexcept { return m_segments; }
    const std::vector<Vehicle>& vehicles() const noexcept { return m_vehicles; }
    const std::vector<int32_t>& current_cells() const noexcept { return m_current_cells; }

    uint32_t current_step() const noexcept { return m_current_step; }
    uint32_t active_vehicle_count() const noexcept { return m_active_vehicle_count; }
    uint32_t completed_vehicle_count() const noexcept { return m_completed_vehicles; }
    uint32_t injected_vehicle_count() const noexcept { return m_injected_vehicles; }
    int effective_thread_count() const noexcept { return m_effective_threads; }

private:
    void precompute_routing_tables();
    void prepare_demand_candidates();
    void seed_interior_vehicles();
    void inject_demand();
    void update_vehicles_parallel();
    void update_metrics();
    void apply_omp_environment();

    SimulationConfig m_config{};
    OpenMPConfig m_omp_config{};
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

    std::vector<ThreadWorkspace> m_thread_workspaces;
    std::vector<BoundaryCandidate> m_merged_candidates;

    uint32_t m_current_step{0};
    uint32_t m_active_vehicle_count{0};
    uint32_t m_completed_vehicles{0};
    uint32_t m_injected_vehicles{0};
    int m_effective_threads{1};
};

} // namespace karachiflow
