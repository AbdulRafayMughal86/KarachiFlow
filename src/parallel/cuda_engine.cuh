#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "simulation.hpp"
#include "road_network.hpp"
#include "road_segment.hpp"
#include "demand_loader.hpp"

namespace karachiflow {

// Structure of Arrays (SoA) layout for high-throughput memory coalescing
struct CudaDeviceArrays {
    // Vehicle SoA buffers
    int32_t* d_vehicle_segment_id{nullptr};
    int16_t* d_vehicle_lane{nullptr};
    int16_t* d_vehicle_cell_idx{nullptr};
    uint8_t* d_vehicle_velocity{nullptr};
    uint8_t* d_vehicle_max_velocity{nullptr};
    int32_t* d_vehicle_dest_zone{nullptr};
    uint8_t* d_vehicle_active{nullptr};
    uint32_t* d_vehicle_cells_traveled{nullptr};
    uint32_t* d_vehicle_steps_traveled{nullptr};

    // Intermediate intent and reservation claim buffers
    int32_t* d_target_g_idx{nullptr};
    int32_t* d_target_seg_id{nullptr};
    int16_t* d_target_lane{nullptr};
    int16_t* d_target_cell{nullptr};
    uint8_t* d_target_speed{nullptr};
    uint8_t* d_target_trip_done{nullptr};

    // Cell spatial grid buffers (double-buffered)
    int32_t* d_current_cells{nullptr};
    int32_t* d_next_cells{nullptr};
    int32_t* d_cell_claim{nullptr};

    // Static road segment topology buffers
    int16_t* d_seg_lanes{nullptr};
    int16_t* d_seg_cells_per_lane{nullptr};
    int16_t* d_seg_speed_limit{nullptr};
    size_t* d_seg_cell_offset{nullptr};
    int32_t* d_seg_to_node{nullptr};

    // Precomputed routing and destination field buffers
    int32_t* d_routing_table{nullptr};
    uint8_t* d_is_dest_node{nullptr};

    // Dynamic demand injection buffers
    int32_t* d_origin_candidates{nullptr};
    uint32_t* d_injected_count{nullptr};
};

struct CudaMetrics {
    uint32_t steps_executed{0};
    uint64_t total_cell_updates{0};
    uint32_t active_vehicles{0};
    uint32_t completed_vehicles{0};
    uint32_t injected_vehicles{0};
    double total_sim_time_ms{0.0};
    double kernel_compute_time_ms{0.0};
    double h2d_transfer_time_ms{0.0};
    double d2h_transfer_time_ms{0.0};
    double avg_step_time_us{0.0};
    double updates_per_second{0.0};
    float avg_network_speed{0.0f};
    int collision_arbitrations{0};
};

class CudaEngine {
public:
    CudaEngine();
    ~CudaEngine();

    bool initialize(const SimulationConfig& config, std::string* error_msg = nullptr);
    void step();
    void run();

    bool save_snapshot(uint32_t step, const std::string& filepath) const;
    void capture_buffered_snapshot(uint32_t step);
    void flush_buffered_snapshots();
    const std::vector<BufferedSnapshot>& buffered_snapshots() const noexcept { return m_buffered_snapshots; }
    void sync_device_to_host();

    const SimulationConfig& config() const noexcept { return m_config; }
    const CudaMetrics& metrics() const noexcept { return m_metrics; }
    const RoadNetwork& network() const noexcept { return m_network; }
    const std::vector<RoadSegment>& segments() const noexcept { return m_segments; }
    const std::vector<Vehicle>& vehicles() const noexcept { return m_vehicles; }
    const std::vector<int32_t>& current_cells() const noexcept { return m_h_current_cells; }

    uint32_t current_step() const noexcept { return m_current_step; }
    uint32_t active_vehicle_count() const noexcept { return m_active_vehicle_count; }
    uint32_t completed_vehicle_count() const noexcept { return m_completed_vehicles; }
    uint32_t injected_vehicle_count() const noexcept { return m_injected_vehicles; }

    bool verify_global_conservation(uint64_t* out_active, uint64_t* out_completed, uint64_t* out_injected) const;

private:
    void allocate_device_memory();
    void free_device_memory();
    void copy_static_topology_to_device();
    void copy_vehicles_to_device(uint32_t start_idx, uint32_t count);
    void precompute_routing_tables();
    void prepare_demand_candidates();
    void seed_interior_vehicles();
    void inject_demand();
    void update_metrics();

    SimulationConfig m_config{};
    CudaMetrics m_metrics{};
    RoadNetwork m_network;
    DemandMetadata m_demand;

    std::vector<RoadSegment> m_segments;
    std::vector<int32_t> m_h_current_cells;
    std::vector<int32_t> m_h_next_cells;
    std::vector<Vehicle> m_vehicles;

    std::vector<int32_t> m_routing_table;
    std::vector<uint8_t> m_is_dest_node;
    uint32_t m_dest_zone_count{0};

    std::vector<int32_t> m_origin_candidate_segments;
    std::vector<std::vector<int32_t>> m_active_dest_nodes;

    CudaDeviceArrays m_device_arrays{};

    // Pre-allocated host fallback arbitration buffers
    struct HostVehicleIntent {
        int32_t target_g{-1};
        int32_t target_seg{-1};
        int16_t target_lane{0};
        int16_t target_cell{0};
        uint8_t speed{0};
        bool trip_done{false};
    };
    std::vector<int32_t> m_host_claims;
    std::vector<HostVehicleIntent> m_host_intents;
    std::vector<BufferedSnapshot> m_buffered_snapshots;
    mutable std::vector<uint32_t> m_snapshot_counts;
    mutable std::vector<uint32_t> m_snapshot_speeds;

    uint32_t m_current_step{0};
    uint32_t m_injected_vehicles{0};
    uint32_t m_active_vehicle_count{0};
    uint32_t m_completed_vehicles{0};

    size_t m_total_cells{0};
    size_t m_max_vehicles{0};
    bool m_device_initialized{false};
};

} // namespace karachiflow
