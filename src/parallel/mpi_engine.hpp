#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include "cell.hpp"
#include "vehicle.hpp"
#include "road_segment.hpp"
#include "road_network.hpp"
#include "demand_loader.hpp"
#include "simulation.hpp"
#include "mpi_partitioner.hpp"

#if defined(HAS_MPI) || defined(MPI_VERSION)
#include <mpi.h>
#endif

namespace karachiflow {

struct MigratingVehicle {
    int32_t id{-1};
    int32_t target_segment_id{-1};
    int16_t target_lane{0};
    int16_t target_cell_idx{0};
    uint8_t velocity{0};
    uint8_t max_velocity{2};
    int32_t origin_node{-1};
    int32_t dest_node{-1};
    int32_t dest_zone{-1};
    uint32_t steps_traveled{0};
    uint32_t cells_traveled{0};
};

struct BoundaryHaloCell {
    int32_t segment_id{-1};
    int16_t lane{0};
    uint8_t occupied{0};
};

constexpr size_t MAX_MIGRATION_CAPACITY = 2048;
constexpr size_t MAX_HALO_CAPACITY = 2048;

struct alignas(64) BoundaryExchangePacket {
    uint32_t vehicle_count{0};
    uint32_t halo_count{0};
    BoundaryHaloCell halo_cells[MAX_HALO_CAPACITY];
    MigratingVehicle vehicles[MAX_MIGRATION_CAPACITY];
};

using MigrationPacket = BoundaryExchangePacket;

struct MPIEngineMetrics {
    uint32_t steps_executed{0};
    uint64_t total_cell_updates{0};
    uint32_t active_vehicles{0};
    uint32_t completed_vehicles{0};
    uint32_t injected_vehicles{0};
    uint32_t migrated_out{0};
    uint32_t migrated_in{0};
    double total_sim_time_ms{0.0};
    double avg_step_time_us{0.0};
    double updates_per_second{0.0};
    double comm_wait_time_ms{0.0};
    float avg_network_speed{0.0f};
    double max_rank_time_ms{0.0};
    double avg_rank_time_ms{0.0};
    double load_imbalance_ratio{1.0};
};

class MPIEngine {
public:
    MPIEngine() = default;

    bool initialize(const SimulationConfig& config,
                    bool enable_overlap = true,
                    std::string* error_msg = nullptr);

    void step();
    void run();

    bool verify_global_conservation(uint64_t* global_active = nullptr,
                                    uint64_t* global_completed = nullptr,
                                    uint64_t* global_injected = nullptr);

    bool save_snapshot(uint32_t step, const std::string& filepath) const;

    int rank() const noexcept { return m_rank; }
    int total_ranks() const noexcept { return m_total_ranks; }
    const PartitionResult& partition() const noexcept { return m_partition; }
    const SimulationConfig& config() const noexcept { return m_config; }
    const MPIEngineMetrics& metrics() const noexcept { return m_metrics; }
    const RoadNetwork& network() const noexcept { return m_network; }
    const std::vector<RoadSegment>& segments() const noexcept { return m_segments; }
    const std::vector<Vehicle>& vehicles() const noexcept { return m_vehicles; }

    uint32_t current_step() const noexcept { return m_current_step; }
    uint32_t active_vehicle_count() const noexcept { return m_active_vehicle_count; }
    uint32_t completed_vehicle_count() const noexcept { return m_completed_vehicles; }
    uint32_t injected_vehicle_count() const noexcept { return m_injected_vehicles; }

private:
    void precompute_routing_tables();
    void prepare_demand_candidates();
    void seed_interior_vehicles();
    void inject_demand();

    void update_boundary_segments();
    void initiate_boundary_exchange();
    void update_interior_segments();
    void finalize_boundary_exchange();
    void unpack_incoming_vehicles();

    void update_metrics();
    void update_vehicle_nasch(Vehicle& v, const RoadSegment& seg, bool is_boundary);

    SimulationConfig m_config{};
    MPIEngineMetrics m_metrics{};
    RoadNetwork m_network;
    DemandMetadata m_demand;
    PartitionResult m_partition;

    int m_rank{0};
    int m_total_ranks{1};
    bool m_enable_overlap{true};

    std::vector<RoadSegment> m_segments;
    std::vector<int32_t> m_current_cells;
    std::vector<int32_t> m_next_cells;
    std::vector<Vehicle> m_vehicles;

    std::vector<int32_t> m_routing_table;
    uint32_t m_dest_zone_count{0};

    std::vector<int32_t> m_origin_candidate_segments;
    std::vector<std::vector<int32_t>> m_active_dest_nodes;
    std::vector<uint8_t> m_is_dest_node;

    std::vector<BoundaryExchangePacket> m_send_packets;
    std::vector<BoundaryExchangePacket> m_recv_packets;

    std::vector<std::vector<CutEdge>> m_incoming_cut_edges_by_rank;
    std::vector<uint8_t> m_halo_cell_occupied;

    mutable std::vector<uint32_t> m_snapshot_counts;
    mutable std::vector<uint32_t> m_snapshot_speeds;
    mutable std::vector<uint32_t> m_global_snapshot_counts;
    mutable std::vector<uint32_t> m_global_snapshot_speeds;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    std::vector<MPI_Request> m_mpi_requests;
    std::vector<MPI_Status> m_mpi_statuses;
#endif

    std::vector<MigratingVehicle> m_arrival_queue;

    uint32_t m_current_step{0};
    uint32_t m_active_vehicle_count{0};
    uint32_t m_completed_vehicles{0};
    uint32_t m_injected_vehicles{0};
    uint32_t m_migrated_out{0};
    uint32_t m_migrated_in{0};

    uint64_t m_step_velocity_sum{0};
    uint32_t m_step_velocity_count{0};
    std::chrono::high_resolution_clock::time_point m_comm_start{};
};

} // namespace karachiflow
