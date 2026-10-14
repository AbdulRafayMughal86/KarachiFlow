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
#include "openmp_engine.hpp"
#include "mpi_partitioner.hpp"
#include "mpi_engine.hpp"

#if defined(HAS_MPI) || defined(MPI_VERSION)
#include <mpi.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace karachiflow {

struct alignas(64) HybridThreadWorkspace {
    std::vector<BoundaryCandidate> boundary_candidates;
    uint64_t cell_updates{0};
    uint64_t total_velocity{0};
    uint32_t velocity_count{0};
    char cache_padding[64];
};

struct HybridMetrics {
    uint32_t steps_executed{0};
    uint64_t total_cell_updates{0};
    uint32_t active_vehicles{0};
    uint32_t completed_vehicles{0};
    uint32_t injected_vehicles{0};
    uint32_t migrated_out{0};
    uint32_t migrated_in{0};
    double total_sim_time_ms{0.0};
    double compute_time_ms{0.0};
    double comm_wait_time_ms{0.0};
    double comm_to_compute_ratio{0.0};
    double avg_step_time_us{0.0};
    double updates_per_second{0.0};
    float avg_network_speed{0.0f};
    double max_rank_time_ms{0.0};
    double avg_rank_time_ms{0.0};
    double load_imbalance_ratio{1.0};
    int mpi_ranks{1};
    int omp_threads{1};
    int total_workers{1};
};

class HybridEngine {
public:
    HybridEngine() = default;

    bool initialize(const SimulationConfig& config,
                    const OpenMPConfig& omp_config = {},
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
    int effective_thread_count() const noexcept { return m_effective_threads; }
    int total_workers() const noexcept { return m_total_ranks * m_effective_threads; }

    const PartitionResult& partition() const noexcept { return m_partition; }
    const SimulationConfig& config() const noexcept { return m_config; }
    const OpenMPConfig& omp_config() const noexcept { return m_omp_config; }
    const HybridMetrics& metrics() const noexcept { return m_metrics; }
    const RoadNetwork& network() const noexcept { return m_network; }
    const std::vector<RoadSegment>& segments() const noexcept { return m_segments; }
    const std::vector<Vehicle>& vehicles() const noexcept { return m_vehicles; }

    uint32_t current_step() const noexcept { return m_current_step; }
    uint32_t active_vehicle_count() const noexcept { return m_active_vehicle_count; }
    uint32_t completed_vehicle_count() const noexcept { return m_completed_vehicles; }
    uint32_t injected_vehicle_count() const noexcept { return m_injected_vehicles; }

private:
    void apply_omp_environment();
    void precompute_routing_tables();
    void prepare_demand_candidates();
    void seed_interior_vehicles();
    void inject_demand();

    void update_boundary_segments();
    void initiate_boundary_exchange();
    void update_interior_segments_parallel();
    void arbitrate_interior_candidates();
    void finalize_boundary_exchange();
    void unpack_incoming_vehicles();

    void update_metrics();
    void update_vehicle_nasch_boundary(Vehicle& v, const RoadSegment& seg);

    SimulationConfig m_config{};
    OpenMPConfig m_omp_config{};
    HybridMetrics m_metrics{};
    RoadNetwork m_network;
    DemandMetadata m_demand;
    PartitionResult m_partition;

    int m_rank{0};
    int m_total_ranks{1};
    int m_effective_threads{1};
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
    std::vector<HybridThreadWorkspace> m_thread_workspaces;
    std::vector<BoundaryCandidate> m_merged_candidates;

    uint32_t m_current_step{0};
    uint32_t m_active_vehicle_count{0};
    uint32_t m_completed_vehicles{0};
    uint32_t m_injected_vehicles{0};
    uint32_t m_migrated_out{0};
    uint32_t m_migrated_in{0};

    uint64_t m_step_velocity_sum{0};
    uint32_t m_step_velocity_count{0};
    double m_accumulated_comm_wait_ms{0.0};
};

} // namespace karachiflow
