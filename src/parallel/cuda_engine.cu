#include "cuda_engine.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <limits>
#include <filesystem>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define CUDA_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
    } \
} while (0)
#endif

namespace karachiflow {

namespace {

// Counter-based PRNG hash (splitmix-style) for lockless, reproducible stochastic deceleration
#if defined(__CUDACC__)
__host__ __device__
#endif
inline uint32_t gpu_hash_rng(uint32_t vehicle_id, uint32_t step) {
    uint32_t h = vehicle_id ^ (step * 0x9e3779b9u);
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    return h ^ (h >> 16);
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline bool gpu_should_slow(uint32_t vehicle_id, uint32_t step, float p_slow) {
    uint32_t h = gpu_hash_rng(vehicle_id, step);
    return (static_cast<float>(h & 0xFFFFu) / 65536.0f) < p_slow;
}

#if defined(__CUDACC__)
// CUDA Device Kernel: Stage 1 - Evaluate Acceleration, Gap, and Post Intent Reservation
__global__ void evaluate_intent_kernel(
    uint32_t injected_vehicles,
    uint32_t current_step,
    float p_slow,
    uint32_t dest_zone_count,
    uint32_t node_count,
    // Vehicle SoA buffers
    const int32_t* __restrict__ d_v_segment_id,
    const int16_t* __restrict__ d_v_lane,
    const int16_t* __restrict__ d_v_cell_idx,
    const uint8_t* __restrict__ d_v_velocity,
    const uint8_t* __restrict__ d_v_max_velocity,
    const int32_t* __restrict__ d_v_dest_zone,
    const uint8_t* __restrict__ d_v_active,
    // Intermediate target intent buffers
    int32_t* __restrict__ d_target_g_idx,
    int32_t* __restrict__ d_target_seg_id,
    int16_t* __restrict__ d_target_lane,
    int16_t* __restrict__ d_target_cell,
    uint8_t* __restrict__ d_target_speed,
    uint8_t* __restrict__ d_target_trip_done,
    // Spatial grid buffers
    const int32_t* __restrict__ d_current_cells,
    int32_t* __restrict__ d_cell_claim,
    // Static segment topology buffers
    const int16_t* __restrict__ d_seg_lanes,
    const int16_t* __restrict__ d_seg_cells_per_lane,
    const int16_t* __restrict__ d_seg_speed_limit,
    const size_t* __restrict__ d_seg_cell_offset,
    const int32_t* __restrict__ d_seg_to_node,
    // Precomputed routing buffers
    const int32_t* __restrict__ d_routing_table,
    const uint8_t* __restrict__ d_is_dest_node
) {
    uint32_t veh_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (veh_id >= injected_vehicles) return;
    if (d_v_active[veh_id] == 0) return;

    int32_t seg_id = d_v_segment_id[veh_id];
    int16_t lane = d_v_lane[veh_id];
    int16_t cell_idx = d_v_cell_idx[veh_id];
    uint8_t vel = d_v_velocity[veh_id];
    uint8_t max_vel = d_v_max_velocity[veh_id];
    int32_t dest_zone = d_v_dest_zone[veh_id];

    int16_t cpl = d_seg_cells_per_lane[seg_id];
    uint8_t seg_limit = static_cast<uint8_t>(d_seg_speed_limit[seg_id]);
    uint8_t effective_vmax = (max_vel < seg_limit) ? max_vel : seg_limit;
    size_t seg_offset = d_seg_cell_offset[seg_id];

    // 1. Acceleration
    int speed = (vel + 1 < effective_vmax) ? (vel + 1) : effective_vmax;

    // 2. Deceleration (Look-ahead on current lane)
    int lead_dist = 999999;
    int scan_limit = (cell_idx + speed < cpl - 1) ? (cell_idx + speed) : (cpl - 1);
    for (int c = cell_idx + 1; c <= scan_limit; ++c) {
        size_t g_idx = seg_offset + static_cast<size_t>(lane) * cpl + c;
        if (d_current_cells[g_idx] >= 0) {
            lead_dist = c - cell_idx;
            break;
        }
    }
    if (lead_dist <= speed) {
        speed = lead_dist - 1;
    }

    // Boundary transition calculation
    int d_seg = static_cast<int>(cpl - 1) - static_cast<int>(cell_idx);
    int32_t next_seg_id = -1;
    int16_t next_lane = 0;
    bool trip_completed = false;

    if (speed > d_seg) {
        int32_t to_node = d_seg_to_node[seg_id];
        if (to_node >= 0 && to_node < static_cast<int32_t>(node_count)) {
            uint32_t eff_zones = (dest_zone_count > 0) ? dest_zone_count : 1;
            uint32_t z = (dest_zone >= 0 && dest_zone < static_cast<int32_t>(dest_zone_count)) ? static_cast<uint32_t>(dest_zone) : 0;
            size_t dest_idx = static_cast<size_t>(to_node) * eff_zones + z;
            if (dest_zone_count > 0 && d_is_dest_node[dest_idx] != 0) {
                trip_completed = true;
            } else {
                next_seg_id = d_routing_table[dest_idx];
            }
        }

        if (!trip_completed && next_seg_id >= 0) {
            int16_t next_lanes = d_seg_lanes[next_seg_id];
            next_lane = (lane < next_lanes) ? lane : (next_lanes - 1);
            int16_t next_cpl = d_seg_cells_per_lane[next_seg_id];
            int16_t next_limit = d_seg_speed_limit[next_seg_id];

            int max_into_next = (next_cpl - 1 < next_limit - 1) ? (next_cpl - 1) : (next_limit - 1);
            if (speed - (d_seg + 1) > max_into_next) {
                speed = (d_seg + 1) + max_into_next;
            }

            size_t next_offset = d_seg_cell_offset[next_seg_id];
            int cells_into_next = speed - (d_seg + 1);
            for (int c_n = 0; c_n <= cells_into_next; ++c_n) {
                size_t next_g = next_offset + static_cast<size_t>(next_lane) * next_cpl + c_n;
                if (d_current_cells[next_g] >= 0) {
                    speed = (c_n == 0) ? d_seg : (d_seg + 1) + (c_n - 1);
                    break;
                }
            }
        } else if (!trip_completed && next_seg_id < 0) {
            trip_completed = true;
        }
    }

    // 3. Stochastic deceleration
    if (speed > 0 && gpu_should_slow(veh_id, current_step, p_slow)) {
        speed -= 1;
    }

    // Determine target global cell index
    int32_t target_g = -1;
    int32_t final_seg = seg_id;
    int16_t final_lane = lane;
    int16_t final_cell = cell_idx;

    if (trip_completed && speed > d_seg) {
        d_target_trip_done[veh_id] = 1;
        d_target_g_idx[veh_id] = -1;
        d_target_speed[veh_id] = static_cast<uint8_t>(speed);
        return;
    }

    d_target_trip_done[veh_id] = 0;
    if (speed > d_seg && next_seg_id >= 0) {
        final_seg = next_seg_id;
        final_lane = next_lane;
        final_cell = static_cast<int16_t>(speed - (d_seg + 1));
        size_t next_off = d_seg_cell_offset[next_seg_id];
        int16_t next_cpl = d_seg_cells_per_lane[next_seg_id];
        target_g = static_cast<int32_t>(next_off + static_cast<size_t>(final_lane) * next_cpl + final_cell);
    } else {
        int advance = (speed < d_seg) ? speed : d_seg;
        final_cell = static_cast<int16_t>(cell_idx + advance);
        target_g = static_cast<int32_t>(seg_offset + static_cast<size_t>(lane) * cpl + final_cell);
    }

    d_target_g_idx[veh_id] = target_g;
    d_target_seg_id[veh_id] = final_seg;
    d_target_lane[veh_id] = final_lane;
    d_target_cell[veh_id] = final_cell;
    d_target_speed[veh_id] = static_cast<uint8_t>(speed);

    // Deterministic collision arbitration: Lowest vehicle ID reserves target cell
    atomicMin(&d_cell_claim[target_g], static_cast<int32_t>(veh_id));
}

// CUDA Device Kernel: Stage 2 - Arbitrate Conflict Claims and Advance Movement
__global__ void arbitrate_and_advance_kernel(
    uint32_t injected_vehicles,
    // Vehicle SoA buffers
    int32_t* __restrict__ d_v_segment_id,
    int16_t* __restrict__ d_v_lane,
    int16_t* __restrict__ d_v_cell_idx,
    uint8_t* __restrict__ d_v_velocity,
    uint8_t* __restrict__ d_v_active,
    uint32_t* __restrict__ d_v_cells_traveled,
    uint32_t* __restrict__ d_v_steps_traveled,
    // Target intent buffers
    const int32_t* __restrict__ d_target_g_idx,
    const int32_t* __restrict__ d_target_seg_id,
    const int16_t* __restrict__ d_target_lane,
    const int16_t* __restrict__ d_target_cell,
    const uint8_t* __restrict__ d_target_speed,
    const uint8_t* __restrict__ d_target_trip_done,
    // Spatial grid buffers
    int32_t* __restrict__ d_next_cells,
    const int32_t* __restrict__ d_cell_claim,
    // Static segment topology buffers
    const int16_t* __restrict__ d_seg_cells_per_lane,
    const size_t* __restrict__ d_seg_cell_offset
) {
    uint32_t veh_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (veh_id >= injected_vehicles) return;
    if (d_v_active[veh_id] == 0) return;

    if (d_target_trip_done[veh_id] == 1) {
        d_v_active[veh_id] = 0;
        d_v_steps_traveled[veh_id]++;
        d_v_cells_traveled[veh_id] += d_target_speed[veh_id];
        return;
    }

    int32_t target_g = d_target_g_idx[veh_id];
    int32_t winner_id = d_cell_claim[target_g];

    if (winner_id == static_cast<int32_t>(veh_id)) {
        // Won reservation claim: Commit movement into next_cells buffer
        d_next_cells[target_g] = static_cast<int32_t>(veh_id);
        d_v_segment_id[veh_id] = d_target_seg_id[veh_id];
        d_v_lane[veh_id] = d_target_lane[veh_id];
        d_v_cell_idx[veh_id] = d_target_cell[veh_id];
        d_v_velocity[veh_id] = d_target_speed[veh_id];
        d_v_cells_traveled[veh_id] += d_target_speed[veh_id];
    } else {
        // Lost contention: Decelerate to 0 and remain in current cell
        int32_t cur_seg = d_v_segment_id[veh_id];
        int16_t cur_lane = d_v_lane[veh_id];
        int16_t cur_cell = d_v_cell_idx[veh_id];
        size_t cur_off = d_seg_cell_offset[cur_seg];
        int16_t cur_cpl = d_seg_cells_per_lane[cur_seg];
        size_t cur_g = cur_off + static_cast<size_t>(cur_lane) * cur_cpl + cur_cell;

        d_next_cells[cur_g] = static_cast<int32_t>(veh_id);
        d_v_velocity[veh_id] = 0;
    }
    d_v_steps_traveled[veh_id]++;
}
#endif

} // namespace

CudaEngine::CudaEngine() = default;

CudaEngine::~CudaEngine() {
    free_device_memory();
}

void CudaEngine::allocate_device_memory() {
#if defined(__CUDACC__)
    const size_t v_bytes = m_max_vehicles;
    const size_t c_bytes = m_total_cells;

    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_segment_id, v_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_lane, v_bytes * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_cell_idx, v_bytes * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_velocity, v_bytes * sizeof(uint8_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_max_velocity, v_bytes * sizeof(uint8_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_dest_zone, v_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_active, v_bytes * sizeof(uint8_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_cells_traveled, v_bytes * sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_vehicle_steps_traveled, v_bytes * sizeof(uint32_t)));

    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_g_idx, v_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_seg_id, v_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_lane, v_bytes * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_cell, v_bytes * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_speed, v_bytes * sizeof(uint8_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_target_trip_done, v_bytes * sizeof(uint8_t)));

    CUDA_CALL(cudaMalloc(&m_device_arrays.d_current_cells, c_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_next_cells, c_bytes * sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_cell_claim, c_bytes * sizeof(int32_t)));

    const size_t seg_count = m_segments.size();
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_seg_lanes, seg_count * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_seg_cells_per_lane, seg_count * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_seg_speed_limit, seg_count * sizeof(int16_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_seg_cell_offset, seg_count * sizeof(size_t)));
    CUDA_CALL(cudaMalloc(&m_device_arrays.d_seg_to_node, seg_count * sizeof(int32_t)));

    const size_t r_bytes = m_routing_table.size();
    if (r_bytes > 0) {
        CUDA_CALL(cudaMalloc(&m_device_arrays.d_routing_table, r_bytes * sizeof(int32_t)));
        CUDA_CALL(cudaMalloc(&m_device_arrays.d_is_dest_node, r_bytes * sizeof(uint8_t)));
    }
    m_device_initialized = true;
#else
    m_device_initialized = true;
#endif
}

void CudaEngine::free_device_memory() {
#if defined(__CUDACC__)
    if (!m_device_initialized) return;
    cudaFree(m_device_arrays.d_vehicle_segment_id);
    cudaFree(m_device_arrays.d_vehicle_lane);
    cudaFree(m_device_arrays.d_vehicle_cell_idx);
    cudaFree(m_device_arrays.d_vehicle_velocity);
    cudaFree(m_device_arrays.d_vehicle_max_velocity);
    cudaFree(m_device_arrays.d_vehicle_dest_zone);
    cudaFree(m_device_arrays.d_vehicle_active);
    cudaFree(m_device_arrays.d_vehicle_cells_traveled);
    cudaFree(m_device_arrays.d_vehicle_steps_traveled);

    cudaFree(m_device_arrays.d_target_g_idx);
    cudaFree(m_device_arrays.d_target_seg_id);
    cudaFree(m_device_arrays.d_target_lane);
    cudaFree(m_device_arrays.d_target_cell);
    cudaFree(m_device_arrays.d_target_speed);
    cudaFree(m_device_arrays.d_target_trip_done);

    cudaFree(m_device_arrays.d_current_cells);
    cudaFree(m_device_arrays.d_next_cells);
    cudaFree(m_device_arrays.d_cell_claim);

    cudaFree(m_device_arrays.d_seg_lanes);
    cudaFree(m_device_arrays.d_seg_cells_per_lane);
    cudaFree(m_device_arrays.d_seg_speed_limit);
    cudaFree(m_device_arrays.d_seg_cell_offset);
    cudaFree(m_device_arrays.d_seg_to_node);

    if (m_device_arrays.d_routing_table) cudaFree(m_device_arrays.d_routing_table);
    if (m_device_arrays.d_is_dest_node) cudaFree(m_device_arrays.d_is_dest_node);
#endif
    m_device_initialized = false;
}

void CudaEngine::copy_static_topology_to_device() {
#if defined(__CUDACC__)
    const size_t seg_count = m_segments.size();
    std::vector<int16_t> h_lanes(seg_count);
    std::vector<int16_t> h_cpl(seg_count);
    std::vector<int16_t> h_limit(seg_count);
    std::vector<size_t> h_offset(seg_count);
    std::vector<int32_t> h_to_node(seg_count);

    for (size_t i = 0; i < seg_count; ++i) {
        h_lanes[i] = m_segments[i].lanes();
        h_cpl[i] = m_segments[i].cells_per_lane();
        h_limit[i] = m_segments[i].speed_limit();
        h_offset[i] = m_segments[i].cell_offset();
        h_to_node[i] = m_segments[i].to_node();
    }

    CUDA_CALL(cudaMemcpy(m_device_arrays.d_seg_lanes, h_lanes.data(), seg_count * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_seg_cells_per_lane, h_cpl.data(), seg_count * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_seg_speed_limit, h_limit.data(), seg_count * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_seg_cell_offset, h_offset.data(), seg_count * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_seg_to_node, h_to_node.data(), seg_count * sizeof(int32_t), cudaMemcpyHostToDevice));

    if (!m_routing_table.empty()) {
        CUDA_CALL(cudaMemcpy(m_device_arrays.d_routing_table, m_routing_table.data(), m_routing_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(m_device_arrays.d_is_dest_node, m_is_dest_node.data(), m_is_dest_node.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
#endif
}

void CudaEngine::copy_vehicles_to_device(uint32_t start_idx, uint32_t count) {
#if defined(__CUDACC__)
    if (!m_device_initialized || count == 0) return;
    std::vector<int32_t> seg_ids(count);
    std::vector<int16_t> lanes(count);
    std::vector<int16_t> cells(count);
    std::vector<uint8_t> vels(count);
    std::vector<uint8_t> max_vels(count);
    std::vector<int32_t> dest_zones(count);
    std::vector<uint8_t> active(count);
    std::vector<uint32_t> cells_trav(count, 0);
    std::vector<uint32_t> steps_trav(count, 0);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& v = m_vehicles[start_idx + i];
        seg_ids[i] = v.segment_id;
        lanes[i] = v.lane;
        cells[i] = v.cell_idx;
        vels[i] = v.velocity;
        max_vels[i] = v.max_velocity;
        dest_zones[i] = v.dest_zone;
        active[i] = v.active ? 1 : 0;
        cells_trav[i] = v.cells_traveled;
        steps_trav[i] = v.steps_traveled;
    }

    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_segment_id + start_idx, seg_ids.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_lane + start_idx, lanes.data(), count * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_cell_idx + start_idx, cells.data(), count * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_velocity + start_idx, vels.data(), count * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_max_velocity + start_idx, max_vels.data(), count * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_dest_zone + start_idx, dest_zones.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_active + start_idx, active.data(), count * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_cells_traveled + start_idx, cells_trav.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_vehicle_steps_traveled + start_idx, steps_trav.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice));
#else
    (void)start_idx;
    (void)count;
#endif
}

void CudaEngine::precompute_routing_tables() {
    m_dest_zone_count = static_cast<uint32_t>(m_demand.destination_zones.size());
    const uint32_t n_nodes = m_network.node_count();
    const size_t table_size = static_cast<size_t>(n_nodes) * std::max(1u, m_dest_zone_count);

    m_routing_table.assign(table_size, -1);
    m_is_dest_node.assign(table_size, 0);
    m_active_dest_nodes.resize(m_dest_zone_count);

    for (uint32_t z = 0; z < m_dest_zone_count; ++z) {
        for (int32_t nid : m_demand.destination_zones[z].node_ids) {
            if (nid >= 0 && static_cast<uint32_t>(nid) < n_nodes) {
                m_is_dest_node[static_cast<size_t>(nid) * m_dest_zone_count + z] = 1;
                m_active_dest_nodes[z].push_back(nid);
            }
        }
    }

    std::vector<std::vector<std::pair<int32_t, int32_t>>> reverse_adj(n_nodes);
    for (size_t s_idx = 0; s_idx < m_segments.size(); ++s_idx) {
        const auto& seg = m_segments[s_idx];
        if (seg.from_node() >= 0 && seg.to_node() >= 0) {
            reverse_adj[seg.to_node()].emplace_back(seg.from_node(), static_cast<int32_t>(s_idx));
        }
    }

    for (uint32_t z = 0; z < m_dest_zone_count; ++z) {
        std::vector<float> dist(n_nodes, std::numeric_limits<float>::infinity());
        using QElem = std::pair<float, int32_t>;
        std::priority_queue<QElem, std::vector<QElem>, std::greater<QElem>> pq;

        for (int32_t dst_node : m_active_dest_nodes[z]) {
            dist[dst_node] = 0.0f;
            pq.emplace(0.0f, dst_node);
        }

        while (!pq.empty()) {
            auto [d_curr, u] = pq.top();
            pq.pop();
            if (d_curr > dist[u]) continue;

            for (const auto& [prev_node, seg_id] : reverse_adj[u]) {
                float weight = m_segments[seg_id].length_meters();
                if (dist[u] + weight < dist[prev_node]) {
                    dist[prev_node] = dist[u] + weight;
                    m_routing_table[static_cast<size_t>(prev_node) * m_dest_zone_count + z] = seg_id;
                    pq.emplace(dist[prev_node], prev_node);
                }
            }
        }
    }

    const uint32_t eff_zones = std::max(1u, m_dest_zone_count);
    for (uint32_t z = 0; z < eff_zones; ++z) {
        for (uint32_t u = 0; u < n_nodes; ++u) {
            const size_t idx = static_cast<size_t>(u) * eff_zones + z;
            if (m_is_dest_node[idx] == 0 && m_routing_table[idx] < 0) {
                auto [out_segs, out_count] = m_network.get_outgoing_segments(static_cast<int32_t>(u));
                if (out_count > 0) {
                    m_routing_table[idx] = out_segs[0];
                }
            }
        }
    }
}

void CudaEngine::prepare_demand_candidates() {
    m_origin_candidate_segments.clear();
    for (const auto& orig : m_demand.origin_zones) {
        for (int32_t nid : orig.node_ids) {
            if (nid >= 0 && static_cast<uint32_t>(nid) < m_network.node_count()) {
                auto [out_segs, out_count] = m_network.get_outgoing_segments(nid);
                for (uint32_t i = 0; i < out_count; ++i) {
                    m_origin_candidate_segments.push_back(out_segs[i]);
                }
            }
        }
    }
    if (m_origin_candidate_segments.empty() && !m_segments.empty()) {
        for (size_t i = 0; i < m_segments.size(); ++i) {
            m_origin_candidate_segments.push_back(static_cast<int32_t>(i));
        }
    }
}

void CudaEngine::seed_interior_vehicles() {
    uint32_t n_seed = 0;
    if (m_config.seed_vehicles > 0) {
        n_seed = std::min(m_config.seed_vehicles, static_cast<uint32_t>(m_max_vehicles));
    } else if (m_config.seed_density > 0.0f) {
        n_seed = std::min(static_cast<uint32_t>(m_total_cells * m_config.seed_density), static_cast<uint32_t>(m_max_vehicles));
    }
    if (n_seed == 0) return;

    double cell_accum = 0.0;
    const double cell_rate = static_cast<double>(n_seed) / static_cast<double>(m_total_cells);

    for (size_t s_idx = 0; s_idx < m_segments.size() && m_injected_vehicles < n_seed; ++s_idx) {
        const auto& seg = m_segments[s_idx];
        const int16_t lanes = seg.lanes();
        const int16_t cpl = seg.cells_per_lane();

        for (int16_t l = 0; l < lanes && m_injected_vehicles < n_seed; ++l) {
            for (int16_t c = 0; c < cpl && m_injected_vehicles < n_seed; ++c) {
                cell_accum += cell_rate;
                if (cell_accum >= 1.0) {
                    cell_accum -= 1.0;
                    size_t g_idx = seg.local_to_global_cell(l, c);
                    if (m_h_current_cells[g_idx] == EMPTY_CELL) {
                        uint32_t veh_id = m_injected_vehicles;
                        auto& v = m_vehicles[veh_id];
                        v.id = static_cast<int32_t>(veh_id);
                        v.segment_id = static_cast<int32_t>(s_idx);
                        v.lane = l;
                        v.cell_idx = c;
                        v.velocity = std::min(static_cast<uint8_t>(1), static_cast<uint8_t>(seg.speed_limit()));
                        v.max_velocity = static_cast<uint8_t>(seg.speed_limit());
                        v.active = true;
                        v.steps_traveled = 0;
                        v.cells_traveled = 0;
                        v.dest_zone = static_cast<int32_t>(veh_id % std::max(1u, m_dest_zone_count));

                        m_h_current_cells[g_idx] = static_cast<int32_t>(veh_id);
                        m_injected_vehicles++;
                        m_active_vehicle_count++;
                    }
                }
            }
        }
    }
}

bool CudaEngine::initialize(const SimulationConfig& config, std::string* error_msg) {
    m_config = config;
    if (!m_network.load_from_binary(config.network_path, error_msg)) return false;

    if (!load_demand_metadata(config.metadata_path, m_demand, error_msg)) {
        return false;
    }

    m_segments.clear();
    m_segments.reserve(m_network.segment_count());
    for (uint32_t i = 0; i < m_network.segment_count(); ++i) {
        m_segments.emplace_back(m_network.get_segment(i));
    }

    m_total_cells = m_network.total_cells();
    m_max_vehicles = config.target_vehicles;

    m_h_current_cells.assign(m_total_cells, EMPTY_CELL);
    m_h_next_cells.assign(m_total_cells, EMPTY_CELL);

    m_vehicles.resize(m_max_vehicles);
    for (uint32_t i = 0; i < m_max_vehicles; ++i) {
        m_vehicles[i].id = static_cast<int32_t>(i);
        m_vehicles[i].active = false;
        m_vehicles[i].velocity = 0;
        m_vehicles[i].max_velocity = 3;
        m_vehicles[i].dest_zone = static_cast<int32_t>(i % std::max(1u, static_cast<uint32_t>(m_demand.destination_zones.size())));
    }

    precompute_routing_tables();
    prepare_demand_candidates();

    m_current_step = 0;
    m_injected_vehicles = 0;
    m_active_vehicle_count = 0;
    m_completed_vehicles = 0;
    m_metrics = CudaMetrics{};

    m_snapshot_counts.assign(m_segments.size(), 0);
    m_snapshot_speeds.assign(m_segments.size(), 0);

    seed_interior_vehicles();

    allocate_device_memory();
    copy_static_topology_to_device();

#if defined(__CUDACC__)
    CUDA_CALL(cudaMemcpy(m_device_arrays.d_current_cells, m_h_current_cells.data(), m_total_cells * sizeof(int32_t), cudaMemcpyHostToDevice));
    if (m_injected_vehicles > 0) {
        copy_vehicles_to_device(0, m_injected_vehicles);
    }
#else
    m_host_claims.assign(m_total_cells, std::numeric_limits<int32_t>::max());
    m_host_intents.resize(m_max_vehicles);
#endif

    return true;
}

void CudaEngine::inject_demand() {
    if (m_injected_vehicles >= m_max_vehicles) return;

    const uint32_t remaining = m_max_vehicles - m_injected_vehicles;
    const uint32_t steps_left = std::max(1u, m_config.total_steps - m_current_step);
    const uint32_t to_inject = std::min(remaining, std::max(1u, (remaining + steps_left - 1) / steps_left));
    const size_t num_candidates = m_origin_candidate_segments.size();
    if (num_candidates == 0) return;

    const uint32_t old_injected = m_injected_vehicles;
    (void)old_injected;
    int failed_attempts = 0;

    for (uint32_t k = 0; k < to_inject && m_injected_vehicles < m_max_vehicles; ++k) {
        uint32_t veh_id = m_injected_vehicles;
        const size_t base_cand_idx = (static_cast<size_t>(veh_id) * 37u + m_current_step * 17u) % num_candidates;

        bool placed = false;
        for (size_t attempt = 0; attempt < 16 && !placed; ++attempt) {
            const size_t cand_idx = (base_cand_idx + attempt * 7919u) % num_candidates;
            const int32_t seg_id = m_origin_candidate_segments[cand_idx];
            const auto& seg = m_segments[seg_id];

            for (int16_t l = 0; l < seg.lanes(); ++l) {
                size_t g_idx = seg.local_to_global_cell(l, 0);
                if (m_h_current_cells[g_idx] == EMPTY_CELL) {
                    auto& v = m_vehicles[veh_id];
                    v.id = static_cast<int32_t>(veh_id);
                    v.segment_id = seg_id;
                    v.lane = l;
                    v.cell_idx = 0;
                    v.velocity = 0;
                    v.max_velocity = static_cast<uint8_t>(seg.speed_limit());
                    v.active = true;
                    v.cells_traveled = 0;
                    v.steps_traveled = 0;
                    v.dest_zone = static_cast<int32_t>(veh_id % std::max(1u, m_dest_zone_count));

                    m_h_current_cells[g_idx] = static_cast<int32_t>(veh_id);
                    m_injected_vehicles++;
                    m_active_vehicle_count++;
                    placed = true;
                    failed_attempts = 0;
                    break;
                }
            }
        }
        if (!placed) {
            failed_attempts++;
            if (failed_attempts >= 3) {
                break;
            }
        }
    }

#if defined(__CUDACC__)
    const uint32_t newly_injected = m_injected_vehicles - old_injected;
    if (newly_injected > 0) {
        copy_vehicles_to_device(old_injected, newly_injected);
        for (uint32_t i = old_injected; i < m_injected_vehicles; ++i) {
            const auto& v = m_vehicles[i];
            const auto& seg = m_segments[v.segment_id];
            size_t g_idx = seg.local_to_global_cell(v.lane, v.cell_idx);
            int32_t vid = v.id;
            CUDA_CALL(cudaMemcpy(m_device_arrays.d_current_cells + g_idx, &vid, sizeof(int32_t), cudaMemcpyHostToDevice));
        }
    }
#endif
}

void CudaEngine::step() {
    inject_demand();

#if defined(__CUDACC__)
    const uint32_t n_veh = m_injected_vehicles;
    if (n_veh > 0) {
        const int block_size = 256;
        const int grid_size = (n_veh + block_size - 1) / block_size;

        // Initialize claim buffer to INT32_MAX and next_cells to EMPTY_CELL (-1)
        CUDA_CALL(cudaMemsetAsync(m_device_arrays.d_cell_claim, 0x7F, m_total_cells * sizeof(int32_t)));
        CUDA_CALL(cudaMemsetAsync(m_device_arrays.d_next_cells, 0xFF, m_total_cells * sizeof(int32_t)));

        // Stage 1: Intent evaluation & reservation claim (GPU-resident)
        evaluate_intent_kernel<<<grid_size, block_size>>>(
            n_veh, m_current_step, m_config.p_slow, m_dest_zone_count, m_network.node_count(),
            m_device_arrays.d_vehicle_segment_id, m_device_arrays.d_vehicle_lane, m_device_arrays.d_vehicle_cell_idx,
            m_device_arrays.d_vehicle_velocity, m_device_arrays.d_vehicle_max_velocity, m_device_arrays.d_vehicle_dest_zone,
            m_device_arrays.d_vehicle_active, m_device_arrays.d_target_g_idx, m_device_arrays.d_target_seg_id,
            m_device_arrays.d_target_lane, m_device_arrays.d_target_cell, m_device_arrays.d_target_speed,
            m_device_arrays.d_target_trip_done, m_device_arrays.d_current_cells, m_device_arrays.d_cell_claim,
            m_device_arrays.d_seg_lanes, m_device_arrays.d_seg_cells_per_lane, m_device_arrays.d_seg_speed_limit,
            m_device_arrays.d_seg_cell_offset, m_device_arrays.d_seg_to_node,
            m_device_arrays.d_routing_table, m_device_arrays.d_is_dest_node
        );

        // Stage 2: Conflict resolution & movement advance (GPU-resident)
        arbitrate_and_advance_kernel<<<grid_size, block_size>>>(
            n_veh, m_device_arrays.d_vehicle_segment_id, m_device_arrays.d_vehicle_lane,
            m_device_arrays.d_vehicle_cell_idx, m_device_arrays.d_vehicle_velocity, m_device_arrays.d_vehicle_active,
            m_device_arrays.d_vehicle_cells_traveled, m_device_arrays.d_vehicle_steps_traveled,
            m_device_arrays.d_target_g_idx, m_device_arrays.d_target_seg_id, m_device_arrays.d_target_lane,
            m_device_arrays.d_target_cell, m_device_arrays.d_target_speed, m_device_arrays.d_target_trip_done,
            m_device_arrays.d_next_cells, m_device_arrays.d_cell_claim,
            m_device_arrays.d_seg_cells_per_lane, m_device_arrays.d_seg_cell_offset
        );

        // Swap ping-pong grid pointers on device
        std::swap(m_device_arrays.d_current_cells, m_device_arrays.d_next_cells);

        // Grid state remains 100% GPU-resident! No host copy per step!
    }
#else
    const uint32_t n_veh = m_injected_vehicles;
    std::fill(m_host_claims.begin(), m_host_claims.end(), std::numeric_limits<int32_t>::max());
    std::fill(m_h_next_cells.begin(), m_h_next_cells.end(), EMPTY_CELL);

    // Stage 1: Intent Evaluation
    for (uint32_t i = 0; i < n_veh; ++i) {
        auto& v = m_vehicles[i];
        if (!v.active) continue;

        const auto& seg = m_segments[v.segment_id];
        int16_t lane = v.lane;
        int16_t cell_idx = v.cell_idx;
        int16_t cpl = seg.cells_per_lane();
        uint8_t vmax = std::min(v.max_velocity, static_cast<uint8_t>(seg.speed_limit()));

        int speed = std::min(static_cast<int>(v.velocity) + 1, static_cast<int>(vmax));
        int lead_dist = 999999;
        int scan_limit = std::min(static_cast<int>(cell_idx + speed), static_cast<int>(cpl - 1));
        for (int c = cell_idx + 1; c <= scan_limit; ++c) {
            size_t g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(c));
            if (m_h_current_cells[g_idx] != EMPTY_CELL) {
                lead_dist = c - cell_idx;
                break;
            }
        }
        if (lead_dist <= speed) speed = lead_dist - 1;

        int d_seg = static_cast<int>(cpl - 1) - static_cast<int>(cell_idx);
        int32_t next_seg_id = -1;
        int16_t next_lane = 0;
        bool trip_completed = false;

        if (speed > d_seg) {
            int32_t to_node = seg.to_node();
            if (v.dest_zone >= 0 && static_cast<uint32_t>(v.dest_zone) < m_dest_zone_count &&
                to_node >= 0 && static_cast<uint32_t>(to_node) < m_network.node_count()) {
                size_t d_idx = static_cast<size_t>(to_node) * m_dest_zone_count + v.dest_zone;
                if (m_is_dest_node[d_idx] != 0) {
                    trip_completed = true;
                } else {
                    next_seg_id = m_routing_table[d_idx];
                }
            }

            if (!trip_completed && next_seg_id < 0 && to_node >= 0) {
                auto [out_segs, out_count] = m_network.get_outgoing_segments(to_node);
                if (out_count > 0) {
                    next_seg_id = out_segs[0];
                }
            }

            if (!trip_completed && next_seg_id >= 0) {
                const auto& next_seg = m_segments[next_seg_id];
                next_lane = std::min(lane, static_cast<int16_t>(next_seg.lanes() - 1));
                int max_into_next = std::min(static_cast<int>(next_seg.cells_per_lane() - 1),
                                             static_cast<int>(next_seg.speed_limit()) - 1);
                if (speed - (d_seg + 1) > max_into_next) {
                    speed = (d_seg + 1) + max_into_next;
                }
                const int cells_into_next = speed - (d_seg + 1);
                for (int c_n = 0; c_n <= cells_into_next; ++c_n) {
                    size_t next_g = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(c_n));
                    if (m_h_current_cells[next_g] != EMPTY_CELL) {
                        speed = (c_n == 0) ? d_seg : (d_seg + 1) + (c_n - 1);
                        break;
                    }
                }
            } else if (!trip_completed && next_seg_id < 0) {
                trip_completed = true;
            }
        }

        if (speed > 0 && gpu_should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
            speed -= 1;
        }

        auto& intent = m_host_intents[i];
        intent.speed = static_cast<uint8_t>(speed);

        if (trip_completed && speed > d_seg) {
            intent.trip_done = true;
            intent.target_g = -1;
        } else if (speed > d_seg && next_seg_id >= 0) {
            intent.target_seg = next_seg_id;
            intent.target_lane = next_lane;
            intent.target_cell = static_cast<int16_t>(speed - (d_seg + 1));
            intent.target_g = static_cast<int32_t>(m_segments[next_seg_id].local_to_global_cell(next_lane, intent.target_cell));
            m_host_claims[intent.target_g] = std::min(m_host_claims[intent.target_g], static_cast<int32_t>(v.id));
        } else {
            int advance = std::min(speed, d_seg);
            intent.target_seg = v.segment_id;
            intent.target_lane = lane;
            intent.target_cell = static_cast<int16_t>(cell_idx + advance);
            intent.target_g = static_cast<int32_t>(seg.local_to_global_cell(lane, intent.target_cell));
            m_host_claims[intent.target_g] = std::min(m_host_claims[intent.target_g], static_cast<int32_t>(v.id));
        }
    }

    // Stage 2: Arbitrate Conflict Claims & Advance Movement
    for (uint32_t i = 0; i < n_veh; ++i) {
        auto& v = m_vehicles[i];
        if (!v.active) continue;
        const auto& intent = m_host_intents[i];

        if (intent.trip_done) {
            v.active = false;
            m_completed_vehicles++;
            m_active_vehicle_count--;
            v.steps_traveled++;
            v.cells_traveled += intent.speed;
            continue;
        }

        int32_t winner = m_host_claims[intent.target_g];
        if (winner == v.id) {
            m_h_next_cells[intent.target_g] = v.id;
            v.segment_id = intent.target_seg;
            v.lane = intent.target_lane;
            v.cell_idx = intent.target_cell;
            v.velocity = intent.speed;
            v.cells_traveled += intent.speed;
        } else {
            // Decelerate and hold position
            size_t cur_g = m_segments[v.segment_id].local_to_global_cell(v.lane, v.cell_idx);
            m_h_next_cells[cur_g] = v.id;
            v.velocity = 0;
            m_metrics.collision_arbitrations++;
        }
        v.steps_traveled++;
    }

    std::swap(m_h_current_cells, m_h_next_cells);
#endif

    m_current_step++;
    m_metrics.total_cell_updates += m_total_cells;
    m_metrics.steps_executed = m_current_step;
    m_metrics.injected_vehicles = m_injected_vehicles;
#if !defined(__CUDACC__)
    update_metrics();
#endif
}

void CudaEngine::sync_device_to_host() {
#if defined(__CUDACC__)
    CUDA_CALL(cudaDeviceSynchronize());
    if (m_injected_vehicles > 0) {
        std::vector<int32_t> seg_ids(m_injected_vehicles);
        std::vector<int16_t> lanes(m_injected_vehicles);
        std::vector<int16_t> cells(m_injected_vehicles);
        std::vector<uint8_t> vels(m_injected_vehicles);
        std::vector<uint8_t> active(m_injected_vehicles);

        CUDA_CALL(cudaMemcpy(seg_ids.data(), m_device_arrays.d_vehicle_segment_id, m_injected_vehicles * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(lanes.data(), m_device_arrays.d_vehicle_lane, m_injected_vehicles * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(cells.data(), m_device_arrays.d_vehicle_cell_idx, m_injected_vehicles * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(vels.data(), m_device_arrays.d_vehicle_velocity, m_injected_vehicles * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(active.data(), m_device_arrays.d_vehicle_active, m_injected_vehicles * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        uint32_t act_count = 0;
        uint32_t comp_count = 0;
        for (uint32_t i = 0; i < m_injected_vehicles; ++i) {
            m_vehicles[i].segment_id = seg_ids[i];
            m_vehicles[i].lane = lanes[i];
            m_vehicles[i].cell_idx = cells[i];
            m_vehicles[i].velocity = vels[i];
            m_vehicles[i].active = (active[i] != 0);
            if (m_vehicles[i].active) act_count++;
            else comp_count++;
        }
        m_active_vehicle_count = act_count;
        m_completed_vehicles = comp_count;
    }
    // Synchronize global cell grid from device to host
    CUDA_CALL(cudaMemcpy(m_h_current_cells.data(), m_device_arrays.d_current_cells, m_total_cells * sizeof(int32_t), cudaMemcpyDeviceToHost));
    update_metrics();
#endif
}

void CudaEngine::capture_buffered_snapshot(uint32_t step) {
    const size_t seg_count = m_segments.size();
    std::fill(m_snapshot_counts.begin(), m_snapshot_counts.end(), 0);
    std::fill(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), 0);

    for (uint32_t i = 0; i < m_injected_vehicles; ++i) {
        const auto& v = m_vehicles[i];
        if (v.active && v.segment_id >= 0 && static_cast<size_t>(v.segment_id) < seg_count) {
            m_snapshot_counts[v.segment_id]++;
            m_snapshot_speeds[v.segment_id] += v.velocity;
        }
    }

    BufferedSnapshot snap;
    snap.step = step;
    snap.records.resize(seg_count);

    for (size_t s = 0; s < seg_count; ++s) {
        const auto& seg = m_segments[s];
        const uint32_t n_cars = m_snapshot_counts[s];
        const float avg_v = (n_cars > 0) ? (static_cast<float>(m_snapshot_speeds[s]) / n_cars) : 0.0f;
        const size_t total_c = seg.total_cells();
        const float density = (total_c > 0) ? (static_cast<float>(n_cars) / total_c) : 0.0f;

        snap.records[s] = {seg.id(), n_cars, avg_v, density};
    }

    m_buffered_snapshots.push_back(std::move(snap));
}

void CudaEngine::flush_buffered_snapshots() {
    if (m_buffered_snapshots.empty()) return;

    std::error_code ec;
    std::filesystem::create_directories(m_config.snapshot_dir, ec);

    char line_buf[128];
    for (const auto& snap : m_buffered_snapshots) {
        const std::string path = m_config.snapshot_dir + "/snapshot_t" + std::to_string(snap.step) + ".csv";
        FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) continue;

        std::vector<char> io_buffer(262144);
        std::setvbuf(fp, io_buffer.data(), _IOFBF, io_buffer.size());

        std::fputs("segment_id,vehicle_count,avg_speed,density_ratio\n", fp);
        for (const auto& rec : snap.records) {
            if (rec.vehicle_count == 0) {
                std::snprintf(line_buf, sizeof(line_buf), "%d,0,0.0000,0.0000\n", rec.segment_id);
            } else {
                std::snprintf(line_buf, sizeof(line_buf), "%d,%u,%.4f,%.4f\n",
                              rec.segment_id, rec.vehicle_count, rec.avg_speed, rec.density_ratio);
            }
            std::fputs(line_buf, fp);
        }
        std::fclose(fp);
    }
    m_buffered_snapshots.clear();
}

void CudaEngine::run() {
    m_buffered_snapshots.clear();
    if (m_config.snapshot_interval > 0) {
        capture_buffered_snapshot(0);
    }

    const auto t_start = std::chrono::high_resolution_clock::now();
    double snap_time_ms = 0.0;

    for (uint32_t s = 1; s <= m_config.total_steps; ++s) {
        step();
        if (m_config.snapshot_interval > 0 && (s % m_config.snapshot_interval == 0)) {
            const auto snap_t0 = std::chrono::high_resolution_clock::now();
            sync_device_to_host();
            capture_buffered_snapshot(s);
            const auto snap_t1 = std::chrono::high_resolution_clock::now();
            snap_time_ms += std::chrono::duration<double, std::milli>(snap_t1 - snap_t0).count();
        }
    }

#if defined(__CUDACC__)
    CUDA_CALL(cudaDeviceSynchronize());
#endif
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    m_metrics.total_sim_time_ms = std::max(0.0, elapsed_ms - snap_time_ms);
    m_metrics.steps_executed = m_current_step;
    if (m_metrics.total_sim_time_ms > 0.0) {
        m_metrics.updates_per_second = (static_cast<double>(m_metrics.total_cell_updates) / (m_metrics.total_sim_time_ms / 1000.0));
        m_metrics.avg_step_time_us = (m_metrics.total_sim_time_ms * 1000.0) / std::max(1u, m_current_step);
    }
    sync_device_to_host();
    update_metrics();

    flush_buffered_snapshots();
}

void CudaEngine::update_metrics() {
    m_metrics.steps_executed = m_current_step;
    m_metrics.active_vehicles = m_active_vehicle_count;
    m_metrics.completed_vehicles = m_completed_vehicles;
    m_metrics.injected_vehicles = m_injected_vehicles;

    double total_speed = 0.0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < m_injected_vehicles; ++i) {
        if (m_vehicles[i].active) {
            total_speed += m_vehicles[i].velocity;
            count++;
        }
    }
    m_metrics.avg_network_speed = (count > 0) ? static_cast<float>(total_speed / count) : 0.0f;
    if (m_metrics.total_sim_time_ms > 0.0) {
        m_metrics.avg_step_time_us = (m_metrics.total_sim_time_ms * 1000.0) / std::max(1u, m_current_step);
    }
}

bool CudaEngine::save_snapshot(uint32_t step, const std::string& filepath) const {
    (void)step;
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "segment_id,vehicle_count,avg_speed,density_ratio\n";
    const size_t seg_count = m_segments.size();

    std::fill(m_snapshot_counts.begin(), m_snapshot_counts.end(), 0);
    std::fill(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), 0);

    for (uint32_t i = 0; i < m_injected_vehicles; ++i) {
        const auto& v = m_vehicles[i];
        if (v.active && v.segment_id >= 0 && static_cast<size_t>(v.segment_id) < seg_count) {
            m_snapshot_counts[v.segment_id]++;
            m_snapshot_speeds[v.segment_id] += v.velocity;
        }
    }

    for (size_t s = 0; s < seg_count; ++s) {
        const auto& seg = m_segments[s];
        const uint32_t n_cars = m_snapshot_counts[s];
        const float avg_v = (n_cars > 0) ? (static_cast<float>(m_snapshot_speeds[s]) / n_cars) : 0.0f;
        const size_t total_c = seg.total_cells();
        const float density = (total_c > 0) ? (static_cast<float>(n_cars) / total_c) : 0.0f;

        out << seg.id() << "," << n_cars << ","
            << std::fixed << std::setprecision(2) << avg_v << ","
            << std::setprecision(4) << density << "\n";
    }
    return true;
}

bool CudaEngine::verify_global_conservation(uint64_t* out_active, uint64_t* out_completed, uint64_t* out_injected) const {
    if (out_active) *out_active = m_active_vehicle_count;
    if (out_completed) *out_completed = m_completed_vehicles;
    if (out_injected) *out_injected = m_injected_vehicles;
    return (m_active_vehicle_count + m_completed_vehicles == m_injected_vehicles);
}

} // namespace karachiflow
