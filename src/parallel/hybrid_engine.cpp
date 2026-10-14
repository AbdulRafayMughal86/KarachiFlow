#include "hybrid_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <cstdio>

namespace karachiflow {

void HybridEngine::apply_omp_environment() {
#if defined(_OPENMP)
    if (m_omp_config.thread_count > 0) {
        omp_set_num_threads(m_omp_config.thread_count);
    }
    m_effective_threads = (m_omp_config.thread_count > 0) ? m_omp_config.thread_count : omp_get_max_threads();

    std::string sched_str = "dynamic";
    switch (m_omp_config.schedule) {
        case OpenMPSchedule::Static: sched_str = "static"; break;
        case OpenMPSchedule::Dynamic: sched_str = "dynamic"; break;
        case OpenMPSchedule::Guided: sched_str = "guided"; break;
        case OpenMPSchedule::Auto: sched_str = "auto"; break;
    }
    if (m_omp_config.chunk_size > 0) {
        sched_str += "," + std::to_string(m_omp_config.chunk_size);
    }

#if defined(_WIN32)
    _putenv_s("OMP_SCHEDULE", sched_str.c_str());
#else
    setenv("OMP_SCHEDULE", sched_str.c_str(), 1);
#endif

#if _OPENMP >= 200805
    omp_sched_t sched = omp_sched_dynamic;
    switch (m_omp_config.schedule) {
        case OpenMPSchedule::Static: sched = omp_sched_static; break;
        case OpenMPSchedule::Dynamic: sched = omp_sched_dynamic; break;
        case OpenMPSchedule::Guided: sched = omp_sched_guided; break;
        case OpenMPSchedule::Auto: sched = omp_sched_auto; break;
    }
    omp_set_schedule(sched, m_omp_config.chunk_size > 0 ? m_omp_config.chunk_size : 64);
#endif
#else
    m_effective_threads = 1;
#endif

    m_thread_workspaces.resize(std::max(1, m_effective_threads));
    for (auto& ws : m_thread_workspaces) {
        ws.boundary_candidates.clear();
        ws.boundary_candidates.reserve(2048);
        ws.cell_updates = 0;
        ws.total_velocity = 0;
        ws.velocity_count = 0;
    }
}

bool HybridEngine::initialize(const SimulationConfig& config,
                              const OpenMPConfig& omp_config,
                              bool enable_overlap,
                              std::string* error_msg) {
    m_config = config;
    m_omp_config = omp_config;
    m_enable_overlap = enable_overlap;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        if (error_msg) *error_msg = "MPI has not been initialized. Call MPI_Init_thread before HybridEngine::initialize().";
        return false;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_total_ranks);
#else
    m_rank = 0;
    m_total_ranks = 1;
#endif

    apply_omp_environment();

    if (!m_network.load_from_binary(m_config.network_path, error_msg)) {
        return false;
    }

    if (!m_network.validate(error_msg)) {
        return false;
    }

    if (!load_demand_metadata(m_config.metadata_path, m_demand, error_msg)) {
        return false;
    }

    m_partition = SpatialPartitioner::partition(m_network, m_total_ranks);

    const uint32_t seg_count = m_network.segment_count();
    m_segments.resize(seg_count);
    for (uint32_t i = 0; i < seg_count; ++i) {
        m_segments[i] = RoadSegment(m_network.get_segment(i));
    }

    const uint32_t total_cells = m_network.total_cells();
    m_current_cells.assign(total_cells, EMPTY_CELL);
    m_next_cells.assign(total_cells, EMPTY_CELL);
    m_halo_cell_occupied.assign(total_cells, 0);

    m_vehicles.resize(m_config.target_vehicles);
    for (uint32_t i = 0; i < m_config.target_vehicles; ++i) {
        m_vehicles[i].id = static_cast<int32_t>(i);
        m_vehicles[i].active = false;
    }

    m_send_packets.resize(m_total_ranks);
    m_recv_packets.resize(m_total_ranks);
    for (int r = 0; r < m_total_ranks; ++r) {
        m_send_packets[r].vehicle_count = 0;
        m_send_packets[r].halo_count = 0;
        m_recv_packets[r].vehicle_count = 0;
        m_recv_packets[r].halo_count = 0;
    }

    m_incoming_cut_edges_by_rank.assign(m_total_ranks, {});
    for (const auto& cut : m_partition.cut_edges) {
        if (cut.dst_rank == m_rank && cut.src_rank >= 0 && cut.src_rank < m_total_ranks) {
            m_incoming_cut_edges_by_rank[cut.src_rank].push_back(cut);
        }
    }

    m_snapshot_counts.assign(seg_count, 0);
    m_snapshot_speeds.assign(seg_count, 0);
    m_global_snapshot_counts.assign(seg_count, 0);
    m_global_snapshot_speeds.assign(seg_count, 0);

    if (m_config.snapshot_interval > 0 && m_rank == 0) {
        std::error_code ec;
        std::filesystem::create_directories(m_config.snapshot_dir, ec);
    }

    precompute_routing_tables();
    prepare_demand_candidates();

    m_current_step = 0;
    m_active_vehicle_count = 0;
    m_completed_vehicles = 0;
    m_injected_vehicles = 0;
    m_migrated_out = 0;
    m_migrated_in = 0;
    m_accumulated_comm_wait_ms = 0.0;

    m_merged_candidates.reserve(4096);

    m_metrics = HybridMetrics{};
    m_metrics.mpi_ranks = m_total_ranks;
    m_metrics.omp_threads = m_effective_threads;
    m_metrics.total_workers = m_total_ranks * m_effective_threads;

    seed_interior_vehicles();

    return true;
}

void HybridEngine::precompute_routing_tables() {
    m_dest_zone_count = static_cast<uint32_t>(m_demand.destination_zones.size());
    const uint32_t node_count = m_network.node_count();
    const uint32_t seg_count = m_network.segment_count();

    m_active_dest_nodes.resize(m_dest_zone_count);
    for (uint32_t z = 0; z < m_dest_zone_count; ++z) {
        m_active_dest_nodes[z].clear();
        for (int32_t n_id : m_demand.destination_zones[z].node_ids) {
            if (n_id >= 0 && static_cast<uint32_t>(n_id) < node_count) {
                m_active_dest_nodes[z].push_back(n_id);
            }
        }
    }

    if (m_dest_zone_count == 0 || node_count == 0) {
        return;
    }

    m_routing_table.assign(static_cast<size_t>(node_count) * m_dest_zone_count, -1);
    m_is_dest_node.assign(static_cast<size_t>(node_count) * m_dest_zone_count, 0);

    for (uint32_t z = 0; z < m_dest_zone_count; ++z) {
        for (int32_t d_node : m_active_dest_nodes[z]) {
            if (d_node >= 0 && static_cast<uint32_t>(d_node) < node_count) {
                m_is_dest_node[static_cast<size_t>(d_node) * m_dest_zone_count + z] = 1;
            }
        }
    }

    struct RevEdge {
        int32_t from_node;
        int32_t segment_id;
        float length;
    };
    std::vector<std::vector<RevEdge>> rev_adj(node_count);
    for (uint32_t i = 0; i < seg_count; ++i) {
        const auto& seg = m_network.get_segment(i);
        if (seg.to_node >= 0 && static_cast<uint32_t>(seg.to_node) < node_count &&
            seg.from_node >= 0 && static_cast<uint32_t>(seg.from_node) < node_count) {
            rev_adj[seg.to_node].push_back({seg.from_node, seg.segment_id, seg.length_meters});
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < static_cast<int>(m_dest_zone_count); ++z) {
        std::vector<float> dist(node_count, std::numeric_limits<float>::infinity());
        std::priority_queue<std::pair<float, int32_t>,
                            std::vector<std::pair<float, int32_t>>,
                            std::greater<std::pair<float, int32_t>>> pq;

        for (int32_t d_node : m_active_dest_nodes[z]) {
            if (d_node >= 0 && static_cast<uint32_t>(d_node) < node_count) {
                dist[d_node] = 0.0f;
                pq.emplace(0.0f, d_node);
            }
        }

        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;

            for (const auto& edge : rev_adj[u]) {
                const float new_dist = dist[u] + edge.length;
                if (new_dist < dist[edge.from_node]) {
                    dist[edge.from_node] = new_dist;
                    m_routing_table[static_cast<size_t>(edge.from_node) * m_dest_zone_count + z] = edge.segment_id;
                    pq.emplace(new_dist, edge.from_node);
                }
            }
        }
    }
}

void HybridEngine::prepare_demand_candidates() {
    const uint32_t node_count = m_network.node_count();
    m_origin_candidate_segments.clear();

    for (const auto& oz : m_demand.origin_zones) {
        for (int32_t onode : oz.node_ids) {
            if (onode >= 0 && static_cast<uint32_t>(onode) < node_count) {
                auto [out_segs, out_count] = m_network.get_outgoing_segments(onode);
                for (size_t i = 0; i < out_count; ++i) {
                    m_origin_candidate_segments.push_back(out_segs[i]);
                }
            }
        }
    }

    std::sort(m_origin_candidate_segments.begin(), m_origin_candidate_segments.end());
    m_origin_candidate_segments.erase(
        std::unique(m_origin_candidate_segments.begin(), m_origin_candidate_segments.end()),
        m_origin_candidate_segments.end());

    if (m_origin_candidate_segments.empty()) {
        const uint32_t seg_count = m_network.segment_count();
        for (uint32_t i = 0; i < seg_count; ++i) {
            m_origin_candidate_segments.push_back(static_cast<int32_t>(i));
        }
    }
}

void HybridEngine::seed_interior_vehicles() {
    uint32_t total_seed = 0;
    if (m_config.seed_vehicles > 0) {
        total_seed = std::min(m_config.seed_vehicles, m_config.target_vehicles);
    } else if (m_config.seed_density > 0.0f) {
        total_seed = std::min(static_cast<uint32_t>(m_network.total_cells() * m_config.seed_density), m_config.target_vehicles);
    }
    if (total_seed == 0) return;

    const uint32_t base_seed = total_seed / m_total_ranks;
    const uint32_t rem_seed = total_seed % m_total_ranks;
    const uint32_t local_seed_target = base_seed + (static_cast<uint32_t>(m_rank) < rem_seed ? 1u : 0u);

    const uint32_t total_target = m_config.target_vehicles;
    const uint32_t base_share = total_target / m_total_ranks;
    const uint32_t remainder = total_target % m_total_ranks;
    const uint32_t local_id_start = static_cast<uint32_t>(m_rank) * base_share +
                                   std::min(static_cast<uint32_t>(m_rank), remainder);

    const auto& rank_segs = m_partition.rank_segments[m_rank];
    if (rank_segs.empty() || local_seed_target == 0) return;

    uint64_t rank_cells = 0;
    for (int32_t s_id : rank_segs) {
        rank_cells += m_segments[s_id].total_cells();
    }
    if (rank_cells == 0) return;

    double cell_accum = 0.0;
    const double cell_rate = static_cast<double>(local_seed_target) / static_cast<double>(rank_cells);

    for (int32_t s_id : rank_segs) {
        if (m_injected_vehicles >= local_seed_target) break;
        const auto& seg = m_segments[s_id];
        const int16_t lanes = seg.lanes();
        const int16_t cpl = seg.cells_per_lane();

        for (int16_t l = 0; l < lanes && m_injected_vehicles < local_seed_target; ++l) {
            for (int16_t c = 0; c < cpl && m_injected_vehicles < local_seed_target; ++c) {
                cell_accum += cell_rate;
                if (cell_accum >= 1.0) {
                    cell_accum -= 1.0;
                    size_t g_idx = seg.local_to_global_cell(l, c);
                    if (is_empty(m_current_cells[g_idx])) {
                        uint32_t veh_id = local_id_start + m_injected_vehicles;
                        if (veh_id < m_vehicles.size()) {
                            auto& v = m_vehicles[veh_id];
                            v.id = static_cast<int32_t>(veh_id);
                            v.segment_id = s_id;
                            v.lane = l;
                            v.cell_idx = c;
                            v.velocity = std::min(static_cast<uint8_t>(1), static_cast<uint8_t>(seg.speed_limit()));
                            v.max_velocity = static_cast<uint8_t>(seg.speed_limit());
                            v.origin_node = seg.from_node();
                            v.dest_zone = (m_dest_zone_count > 0) ? static_cast<int32_t>(veh_id % m_dest_zone_count) : 0;
                            if (m_dest_zone_count > 0 && !m_active_dest_nodes[v.dest_zone].empty()) {
                                const auto& dnodes = m_active_dest_nodes[v.dest_zone];
                                v.dest_node = dnodes[veh_id % dnodes.size()];
                            }
                            v.active = true;
                            v.steps_traveled = 0;
                            v.cells_traveled = 0;

                            m_current_cells[g_idx] = static_cast<int32_t>(veh_id);
                            m_injected_vehicles++;
                            m_active_vehicle_count++;
                        }
                    }
                }
            }
        }
    }
}

void HybridEngine::inject_demand() {
    const uint32_t total_target = m_config.target_vehicles;
    const uint32_t local_target = total_target / m_total_ranks +
                                  (m_rank < static_cast<int>(total_target % m_total_ranks) ? 1 : 0);

    const uint32_t local_id_start = (total_target / m_total_ranks) * m_rank +
                                    std::min(static_cast<uint32_t>(m_rank), total_target % m_total_ranks);
    const uint32_t local_id_end = local_id_start + local_target;

    if (m_injected_vehicles >= local_target) {
        return;
    }

    const uint32_t remaining_steps = (m_config.total_steps > m_current_step)
                                         ? (m_config.total_steps - m_current_step)
                                         : 1;
    const uint32_t remaining_vehicles = local_target - m_injected_vehicles;
    const uint32_t batch_size = std::max(1u, (remaining_vehicles + remaining_steps - 1) / remaining_steps);
    const uint32_t inject_limit = std::min(batch_size, remaining_vehicles);

    std::vector<int32_t> local_origin_segs;
    local_origin_segs.reserve(m_origin_candidate_segments.size());
    for (int32_t s_id : m_origin_candidate_segments) {
        if (m_partition.segment_to_rank[s_id] == m_rank) {
            local_origin_segs.push_back(s_id);
        }
    }

    if (local_origin_segs.empty()) {
        local_origin_segs = m_partition.rank_segments[m_rank];
    }
    if (local_origin_segs.empty()) {
        return;
    }

    const size_t num_candidates = local_origin_segs.size();

    for (uint32_t i = 0; i < inject_limit && m_injected_vehicles < local_target; ++i) {
        const uint32_t veh_id = local_id_start + m_injected_vehicles;
        if (veh_id >= local_id_end || veh_id >= total_target) break;

        const size_t base_cand_idx = (static_cast<size_t>(veh_id) * 37 + m_current_step * 17) % num_candidates;

        bool placed = false;
        for (size_t attempt = 0; attempt < 16 && !placed; ++attempt) {
            const size_t cand_idx = (base_cand_idx + attempt * 7919) % num_candidates;
            const int32_t seg_id = local_origin_segs[cand_idx];
            const auto& seg = m_segments[seg_id];

            for (int16_t lane = 0; lane < seg.lanes(); ++lane) {
                const size_t g_idx = seg.local_to_global_cell(lane, 0);
                if (is_empty(m_current_cells[g_idx])) {
                    auto& v = m_vehicles[veh_id];
                    v.id = static_cast<int32_t>(veh_id);
                    v.segment_id = seg_id;
                    v.lane = lane;
                    v.cell_idx = 0;
                    v.velocity = 0;
                    v.max_velocity = static_cast<uint8_t>(seg.speed_limit());
                    v.origin_node = seg.from_node();
                    v.dest_zone = (m_dest_zone_count > 0) ? static_cast<int32_t>(veh_id % m_dest_zone_count) : 0;
                    if (m_dest_zone_count > 0 && !m_active_dest_nodes[v.dest_zone].empty()) {
                        const auto& dnodes = m_active_dest_nodes[v.dest_zone];
                        v.dest_node = dnodes[veh_id % dnodes.size()];
                    }
                    v.active = true;
                    v.steps_traveled = 0;
                    v.cells_traveled = 0;

                    m_current_cells[g_idx] = static_cast<int32_t>(veh_id);
                    m_injected_vehicles++;
                    m_active_vehicle_count++;
                    placed = true;
                    break;
                }
            }
        }

        if (!placed) {
            break;
        }
    }
}

void HybridEngine::update_vehicle_nasch_boundary(Vehicle& v, const RoadSegment& seg) {
    const int16_t lane = v.lane;
    const int16_t cell_idx = v.cell_idx;
    const int16_t cpl = seg.cells_per_lane();
    const uint8_t v_max = std::min(v.max_velocity, static_cast<uint8_t>(seg.speed_limit()));

    // Rule 1: Acceleration
    int speed = std::min(static_cast<int>(v.velocity) + 1, static_cast<int>(v_max));

    // Rule 2: Deceleration (forward lookahead in read-only current_cells)
    int lead_dist = 999999;
    const int scan_limit = std::min(static_cast<int>(cell_idx + speed), static_cast<int>(cpl - 1));
    for (int c_scan = cell_idx + 1; c_scan <= scan_limit; ++c_scan) {
        const size_t g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(c_scan));
        if (is_occupied(m_current_cells[g_idx])) {
            lead_dist = c_scan - cell_idx;
            break;
        }
    }

    if (lead_dist <= speed) {
        speed = lead_dist - 1;
    }

    const int d_seg = static_cast<int>(cpl - 1) - static_cast<int>(cell_idx);

    if (speed <= d_seg) {
        if (speed > 0 && should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
            speed = speed - 1;
        }

        const int new_cell = cell_idx + speed;
        const size_t new_g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(new_cell));
        m_next_cells[new_g_idx] = v.id;

        v.velocity = static_cast<uint8_t>(speed);
        v.steps_traveled++;
        v.cells_traveled += static_cast<uint32_t>(speed);
        v.cell_idx = static_cast<int16_t>(new_cell);

        m_step_velocity_sum += static_cast<uint8_t>(speed);
        m_step_velocity_count++;
        return;
    }

    // Vehicle reaches end of segment (speed > d_seg)
    int32_t next_seg_id = -1;
    int16_t next_lane = 0;
    const int32_t to_node = seg.to_node();
    bool reached_destination = false;

    if (v.dest_zone >= 0 && static_cast<uint32_t>(v.dest_zone) < m_dest_zone_count &&
        to_node >= 0 && static_cast<uint32_t>(to_node) < m_network.node_count()) {
        reached_destination = (m_is_dest_node[static_cast<size_t>(to_node) * m_dest_zone_count + v.dest_zone] != 0);
        if (!reached_destination) {
            next_seg_id = m_routing_table[static_cast<size_t>(to_node) * m_dest_zone_count + v.dest_zone];
        }
    }

    if (!reached_destination && next_seg_id < 0 && to_node >= 0) {
        auto [out_segs, out_count] = m_network.get_outgoing_segments(to_node);
        if (out_count > 0) {
            next_seg_id = out_segs[0];
        }
    }

    if (next_seg_id >= 0) {
        const int32_t target_rank = m_partition.segment_to_rank[next_seg_id];
        const auto& next_seg = m_segments[next_seg_id];

        if (target_rank == m_rank) {
            int16_t chosen_lane = std::min(lane, static_cast<int16_t>(next_seg.lanes() - 1));
            size_t chosen_g0 = next_seg.local_to_global_cell(chosen_lane, 0);

            if (!is_empty(m_current_cells[chosen_g0]) || !is_empty(m_next_cells[chosen_g0])) {
                for (int16_t alt_l = 0; alt_l < next_seg.lanes(); ++alt_l) {
                    if (alt_l == chosen_lane) continue;
                    const size_t alt_g0 = next_seg.local_to_global_cell(alt_l, 0);
                    if (is_empty(m_current_cells[alt_g0]) && is_empty(m_next_cells[alt_g0])) {
                        chosen_lane = alt_l;
                        chosen_g0 = alt_g0;
                        break;
                    }
                }
            }
            next_lane = chosen_lane;

            const int max_into_next = std::min(static_cast<int>(next_seg.cells_per_lane() - 1),
                                               static_cast<int>(next_seg.speed_limit()) - 1);
            if (speed - (d_seg + 1) > max_into_next) {
                speed = (d_seg + 1) + max_into_next;
            }

            const size_t next_g0 = chosen_g0;
            if (!is_empty(m_current_cells[next_g0]) || !is_empty(m_next_cells[next_g0])) {
                speed = std::min(speed, d_seg);
            } else {
                const int cells_into_next = speed - (d_seg + 1);
                for (int c_n = 1; c_n <= cells_into_next; ++c_n) {
                    const size_t g_idx = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(c_n));
                    if (!is_empty(m_current_cells[g_idx]) || !is_empty(m_next_cells[g_idx])) {
                        speed = (d_seg + 1) + (c_n - 1);
                        break;
                    }
                }
            }
        } else {
            // Cut edge transition to neighbor rank
            int16_t chosen_lane = std::min(lane, static_cast<int16_t>(next_seg.lanes() - 1));
            size_t chosen_g0 = next_seg.local_to_global_cell(chosen_lane, 0);
            bool destination_vacant = (m_halo_cell_occupied[chosen_g0] == 0);

            if (!destination_vacant) {
                for (int16_t alt_l = 0; alt_l < next_seg.lanes(); ++alt_l) {
                    if (alt_l == chosen_lane) continue;
                    const size_t alt_g0 = next_seg.local_to_global_cell(alt_l, 0);
                    if (m_halo_cell_occupied[alt_g0] == 0) {
                        chosen_lane = alt_l;
                        chosen_g0 = alt_g0;
                        destination_vacant = true;
                        break;
                    }
                }
            }

            if (!destination_vacant) {
                speed = std::min(speed, d_seg);
            } else {
                if (speed > d_seg + 1) {
                    speed = d_seg + 1;
                }
                next_lane = chosen_lane;
                m_halo_cell_occupied[chosen_g0] = 1;
            }
        }
    }

    if (speed > 0 && should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
        speed = speed - 1;
    }

    v.velocity = static_cast<uint8_t>(speed);
    v.steps_traveled++;
    v.cells_traveled += static_cast<uint32_t>(speed);
    m_step_velocity_sum += static_cast<uint8_t>(speed);
    m_step_velocity_count++;

    if (speed > d_seg) {
        if (next_seg_id < 0) {
            v.active = false;
            m_completed_vehicles++;
            m_active_vehicle_count--;
        } else {
            const int32_t target_rank = m_partition.segment_to_rank[next_seg_id];

            if (target_rank == m_rank) {
                const auto& next_seg = m_segments[next_seg_id];
                const int new_cell = speed - (d_seg + 1);
                const size_t new_g_idx = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(new_cell));
                m_next_cells[new_g_idx] = v.id;

                v.segment_id = next_seg_id;
                v.lane = next_lane;
                v.cell_idx = static_cast<int16_t>(new_cell);
            } else {
                MigratingVehicle mv;
                mv.id = v.id;
                mv.target_segment_id = next_seg_id;
                mv.target_lane = next_lane;
                mv.target_cell_idx = 0;
                mv.velocity = static_cast<uint8_t>(speed);
                mv.max_velocity = v.max_velocity;
                mv.origin_node = v.origin_node;
                mv.dest_node = v.dest_node;
                mv.dest_zone = v.dest_zone;
                mv.steps_traveled = v.steps_traveled;
                mv.cells_traveled = v.cells_traveled;

                auto& pkt = m_send_packets[target_rank];
                if (pkt.vehicle_count < MAX_MIGRATION_CAPACITY) {
                    pkt.vehicles[pkt.vehicle_count++] = mv;
                }

                v.active = false;
                m_active_vehicle_count--;
                m_migrated_out++;
            }
        }
    } else {
        const int new_cell = cell_idx + speed;
        const size_t new_g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(new_cell));
        m_next_cells[new_g_idx] = v.id;
        v.cell_idx = static_cast<int16_t>(new_cell);
    }
}

void HybridEngine::update_boundary_segments() {
    for (int nbr : m_partition.rank_neighbors[m_rank]) {
        m_send_packets[nbr].vehicle_count = 0;
        m_send_packets[nbr].halo_count = 0;

        for (const auto& cut : m_incoming_cut_edges_by_rank[nbr]) {
            const auto& seg = m_segments[cut.dst_segment_id];
            for (int16_t l = 0; l < seg.lanes(); ++l) {
                if (m_send_packets[nbr].halo_count < MAX_HALO_CAPACITY) {
                    const size_t g0 = seg.local_to_global_cell(l, 0);
                    const uint8_t occ = is_occupied(m_current_cells[g0]) ? 1 : 0;
                    m_send_packets[nbr].halo_cells[m_send_packets[nbr].halo_count++] = {
                        cut.dst_segment_id, l, occ
                    };
                }
            }
        }
    }

    for (int32_t s_id : m_partition.boundary_segments[m_rank]) {
        const auto& seg = m_segments[s_id];
        for (int16_t l = 0; l < seg.lanes(); ++l) {
            for (int16_t c = 0; c < seg.cells_per_lane(); ++c) {
                const size_t g_idx = seg.local_to_global_cell(l, c);
                const int32_t v_id = m_current_cells[g_idx];
                if (is_occupied(v_id)) {
                    update_vehicle_nasch_boundary(m_vehicles[v_id], seg);
                }
            }
        }
    }
}

void HybridEngine::initiate_boundary_exchange() {
#if defined(HAS_MPI) || defined(MPI_VERSION)
    m_mpi_requests.clear();
    const auto& neighbors = m_partition.rank_neighbors[m_rank];

    for (int nbr : neighbors) {
        MPI_Request req;
        MPI_Irecv(&m_recv_packets[nbr],
                  sizeof(BoundaryExchangePacket),
                  MPI_BYTE,
                  nbr,
                  101,
                  MPI_COMM_WORLD,
                  &req);
        m_mpi_requests.push_back(req);
    }

    for (int nbr : neighbors) {
        MPI_Request req;
        MPI_Isend(&m_send_packets[nbr],
                  sizeof(BoundaryExchangePacket),
                  MPI_BYTE,
                  nbr,
                  101,
                  MPI_COMM_WORLD,
                  &req);
        m_mpi_requests.push_back(req);
    }
#endif
}

void HybridEngine::update_interior_segments_parallel() {
    const auto& interior_segs = m_partition.interior_segments[m_rank];
    const int n_int = static_cast<int>(interior_segs.size());
    if (n_int == 0) return;

    for (auto& ws : m_thread_workspaces) {
        ws.boundary_candidates.clear();
        ws.cell_updates = 0;
        ws.total_velocity = 0;
        ws.velocity_count = 0;
    }

    #pragma omp parallel num_threads(m_effective_threads)
    {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        if (tid >= static_cast<int>(m_thread_workspaces.size())) {
            tid = 0;
        }
        auto& ws = m_thread_workspaces[tid];

        #pragma omp for schedule(runtime)
        for (int i = 0; i < n_int; ++i) {
            const int32_t s_id = interior_segs[i];
            const auto& seg = m_segments[s_id];
            const int16_t lanes = seg.lanes();
            const int16_t cpl = seg.cells_per_lane();
            const uint8_t seg_limit = static_cast<uint8_t>(seg.speed_limit());

            for (int16_t l = 0; l < lanes; ++l) {
                for (int16_t c = 0; c < cpl; ++c) {
                    const size_t g_idx = seg.local_to_global_cell(l, c);
                    const int32_t v_id = m_current_cells[g_idx];
                    if (is_empty(v_id)) continue;

                    auto& v = m_vehicles[v_id];
                    const uint8_t v_max = std::min(v.max_velocity, seg_limit);

                    // Rule 1: Acceleration
                    int speed = std::min(static_cast<int>(v.velocity) + 1, static_cast<int>(v_max));

                    // Rule 2: Deceleration (forward lookahead in read-only current_cells)
                    int lead_dist = 999999;
                    const int scan_limit = std::min(static_cast<int>(c + speed), static_cast<int>(cpl - 1));
                    for (int c_scan = c + 1; c_scan <= scan_limit; ++c_scan) {
                        const size_t lead_g = seg.local_to_global_cell(l, static_cast<int16_t>(c_scan));
                        if (is_occupied(m_current_cells[lead_g])) {
                            lead_dist = c_scan - c;
                            break;
                        }
                    }
                    if (lead_dist <= speed) {
                        speed = lead_dist - 1;
                    }

                    const int d_seg = static_cast<int>(cpl - 1) - static_cast<int>(c);

                    if (speed <= d_seg) {
                        if (speed > 0 && should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
                            speed = speed - 1;
                        }

                        const int new_cell = c + speed;
                        const size_t new_g = seg.local_to_global_cell(l, static_cast<int16_t>(new_cell));
                        m_next_cells[new_g] = v.id;

                        v.velocity = static_cast<uint8_t>(speed);
                        v.steps_traveled++;
                        v.cells_traveled += static_cast<uint32_t>(speed);
                        v.cell_idx = static_cast<int16_t>(new_cell);

                        ws.total_velocity += static_cast<uint8_t>(speed);
                        ws.velocity_count++;
                    } else {
                        ws.boundary_candidates.push_back({v.id, static_cast<int16_t>(speed), static_cast<int16_t>(d_seg)});
                    }
                }
            }
        }
    }
}

void HybridEngine::arbitrate_interior_candidates() {
    m_merged_candidates.clear();
    for (const auto& ws : m_thread_workspaces) {
        m_merged_candidates.insert(m_merged_candidates.end(),
                                   ws.boundary_candidates.begin(),
                                   ws.boundary_candidates.end());
    }

    if (m_merged_candidates.empty()) return;

    std::sort(m_merged_candidates.begin(), m_merged_candidates.end(),
              [](const auto& a, const auto& b) noexcept { return a.id < b.id; });

    for (const auto& cand : m_merged_candidates) {
        auto& v = m_vehicles[cand.id];
        const auto& seg = m_segments[v.segment_id];
        const int16_t lane = v.lane;
        const int16_t cell_idx = v.cell_idx;
        int speed = cand.speed;
        const int d_seg = cand.d_seg;

        int32_t next_seg_id = -1;
        int16_t next_lane = 0;
        const int32_t to_node = seg.to_node();
        bool reached_destination = false;

        if (v.dest_zone >= 0 && static_cast<uint32_t>(v.dest_zone) < m_dest_zone_count &&
            to_node >= 0 && static_cast<uint32_t>(to_node) < m_network.node_count()) {
            reached_destination = (m_is_dest_node[static_cast<size_t>(to_node) * m_dest_zone_count + v.dest_zone] != 0);
            if (!reached_destination) {
                next_seg_id = m_routing_table[static_cast<size_t>(to_node) * m_dest_zone_count + v.dest_zone];
            }
        }

        if (!reached_destination && next_seg_id < 0 && to_node >= 0) {
            auto [out_segs, out_count] = m_network.get_outgoing_segments(to_node);
            if (out_count > 0) {
                next_seg_id = out_segs[0];
            }
        }

        if (next_seg_id >= 0) {
            const auto& next_seg = m_segments[next_seg_id];
            int16_t chosen_lane = std::min(lane, static_cast<int16_t>(next_seg.lanes() - 1));
            size_t chosen_g0 = next_seg.local_to_global_cell(chosen_lane, 0);

            if (!is_empty(m_current_cells[chosen_g0]) || !is_empty(m_next_cells[chosen_g0])) {
                for (int16_t alt_l = 0; alt_l < next_seg.lanes(); ++alt_l) {
                    if (alt_l == chosen_lane) continue;
                    const size_t alt_g0 = next_seg.local_to_global_cell(alt_l, 0);
                    if (is_empty(m_current_cells[alt_g0]) && is_empty(m_next_cells[alt_g0])) {
                        chosen_lane = alt_l;
                        chosen_g0 = alt_g0;
                        break;
                    }
                }
            }
            next_lane = chosen_lane;

            const int max_into_next = std::min(static_cast<int>(next_seg.cells_per_lane() - 1),
                                               static_cast<int>(next_seg.speed_limit()) - 1);
            if (speed - (d_seg + 1) > max_into_next) {
                speed = (d_seg + 1) + max_into_next;
            }

            const size_t next_g0 = chosen_g0;
            if (!is_empty(m_current_cells[next_g0]) || !is_empty(m_next_cells[next_g0])) {
                speed = std::min(speed, d_seg);
            } else {
                const int cells_into_next = speed - (d_seg + 1);
                for (int c_n = 1; c_n <= cells_into_next; ++c_n) {
                    const size_t g_idx = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(c_n));
                    if (!is_empty(m_current_cells[g_idx]) || !is_empty(m_next_cells[g_idx])) {
                        speed = (d_seg + 1) + (c_n - 1);
                        break;
                    }
                }
            }
        }

        if (speed > 0 && should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
            speed = speed - 1;
        }

        v.velocity = static_cast<uint8_t>(speed);
        v.steps_traveled++;
        v.cells_traveled += static_cast<uint32_t>(speed);
        m_step_velocity_sum += static_cast<uint8_t>(speed);
        m_step_velocity_count++;

        if (speed > d_seg) {
            if (next_seg_id < 0) {
                v.active = false;
                m_completed_vehicles++;
                m_active_vehicle_count--;
            } else {
                const auto& next_seg = m_segments[next_seg_id];
                const int new_cell = speed - (d_seg + 1);
                const size_t new_g_idx = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(new_cell));
                m_next_cells[new_g_idx] = v.id;

                v.segment_id = next_seg_id;
                v.lane = next_lane;
                v.cell_idx = static_cast<int16_t>(new_cell);
            }
        } else {
            const int new_cell = cell_idx + speed;
            const size_t new_g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(new_cell));
            m_next_cells[new_g_idx] = v.id;
            v.cell_idx = static_cast<int16_t>(new_cell);
        }
    }
}

void HybridEngine::finalize_boundary_exchange() {
#if defined(HAS_MPI) || defined(MPI_VERSION)
    if (!m_mpi_requests.empty()) {
        m_mpi_statuses.resize(m_mpi_requests.size());
        const auto t_start = std::chrono::high_resolution_clock::now();
        MPI_Waitall(static_cast<int>(m_mpi_requests.size()),
                    m_mpi_requests.data(),
                    m_mpi_statuses.data());
        const auto t_end = std::chrono::high_resolution_clock::now();
        m_accumulated_comm_wait_ms +=
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_mpi_requests.clear();
    }
#endif
}

void HybridEngine::unpack_incoming_vehicles() {
    for (int nbr : m_partition.rank_neighbors[m_rank]) {
        const auto& pkt = m_recv_packets[nbr];
        for (uint32_t h = 0; h < pkt.halo_count; ++h) {
            const auto& cell = pkt.halo_cells[h];
            const auto& seg = m_segments[cell.segment_id];
            const size_t g0 = seg.local_to_global_cell(cell.lane, 0);
            m_halo_cell_occupied[g0] = cell.occupied;
        }
    }

    if (!m_arrival_queue.empty()) {
        std::vector<MigratingVehicle> still_pending;
        for (const auto& mv : m_arrival_queue) {
            const auto& next_seg = m_segments[mv.target_segment_id];
            bool placed = false;
            for (int16_t l = 0; l < next_seg.lanes(); ++l) {
                const size_t g_idx = next_seg.local_to_global_cell(l, 0);
                if (is_empty(m_next_cells[g_idx])) {
                    auto& v = m_vehicles[mv.id];
                    v.id = mv.id;
                    v.segment_id = mv.target_segment_id;
                    v.lane = l;
                    v.cell_idx = 0;
                    v.velocity = mv.velocity;
                    v.max_velocity = mv.max_velocity;
                    v.origin_node = mv.origin_node;
                    v.dest_node = mv.dest_node;
                    v.dest_zone = mv.dest_zone;
                    v.steps_traveled = mv.steps_traveled;
                    v.cells_traveled = mv.cells_traveled;
                    v.active = true;

                    m_next_cells[g_idx] = mv.id;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                still_pending.push_back(mv);
            }
        }
        m_arrival_queue = std::move(still_pending);
    }

    for (int nbr : m_partition.rank_neighbors[m_rank]) {
        const auto& pkt = m_recv_packets[nbr];
        for (uint32_t i = 0; i < pkt.vehicle_count; ++i) {
            const auto& mv = pkt.vehicles[i];
            const auto& next_seg = m_segments[mv.target_segment_id];

            int16_t chosen_lane = mv.target_lane;
            int16_t chosen_cell = 0;
            size_t g_idx = next_seg.local_to_global_cell(chosen_lane, chosen_cell);

            if (!is_empty(m_next_cells[g_idx])) {
                bool alt_found = false;
                for (int16_t alt_l = 0; alt_l < next_seg.lanes(); ++alt_l) {
                    const size_t alt_g = next_seg.local_to_global_cell(alt_l, chosen_cell);
                    if (is_empty(m_next_cells[alt_g])) {
                        chosen_lane = alt_l;
                        g_idx = alt_g;
                        alt_found = true;
                        break;
                    }
                }

                if (!alt_found) {
                    m_arrival_queue.push_back(mv);
                    m_active_vehicle_count++;
                    m_migrated_in++;
                    continue;
                }
            }

            auto& v = m_vehicles[mv.id];
            v.id = mv.id;
            v.segment_id = mv.target_segment_id;
            v.lane = chosen_lane;
            v.cell_idx = chosen_cell;
            v.velocity = mv.velocity;
            v.max_velocity = mv.max_velocity;
            v.origin_node = mv.origin_node;
            v.dest_node = mv.dest_node;
            v.dest_zone = mv.dest_zone;
            v.steps_traveled = mv.steps_traveled;
            v.cells_traveled = mv.cells_traveled;
            v.active = true;

            m_next_cells[g_idx] = v.id;
            m_active_vehicle_count++;
            m_migrated_in++;
        }
    }
}

void HybridEngine::step() {
    m_step_velocity_sum = 0;
    m_step_velocity_count = 0;

    inject_demand();

    if (m_enable_overlap) {
        // Master thread executes boundary segment updates
        update_boundary_segments();

        // Master thread posts non-blocking MPI halo & migration transfers
        initiate_boundary_exchange();

        // OpenMP worker threads accelerate interior segment updates concurrently
        update_interior_segments_parallel();

        // Deterministic candidate arbitration
        arbitrate_interior_candidates();

        // Master thread waits for boundary exchange to finalize
        finalize_boundary_exchange();

        // Master thread unpacks received halo & migrating vehicles
        unpack_incoming_vehicles();
    } else {
        update_boundary_segments();
        initiate_boundary_exchange();
        finalize_boundary_exchange();
        unpack_incoming_vehicles();
        update_interior_segments_parallel();
        arbitrate_interior_candidates();
    }

    // Ping-pong double-buffer swap
    std::swap(m_current_cells, m_next_cells);
    std::memset(m_next_cells.data(), 0xFF, m_next_cells.size() * sizeof(int32_t));

    m_current_step++;

    for (const auto& ws : m_thread_workspaces) {
        m_step_velocity_sum += ws.total_velocity;
        m_step_velocity_count += ws.velocity_count;
    }

    uint64_t rank_cells = 0;
    for (int32_t s_id : m_partition.rank_segments[m_rank]) {
        rank_cells += m_segments[s_id].total_cells();
    }
    m_metrics.total_cell_updates += rank_cells;

    update_metrics();
}

void HybridEngine::update_metrics() {
    m_metrics.steps_executed = m_current_step;
    m_metrics.active_vehicles = m_active_vehicle_count;
    m_metrics.completed_vehicles = m_completed_vehicles;
    m_metrics.injected_vehicles = m_injected_vehicles;
    m_metrics.migrated_out = m_migrated_out;
    m_metrics.migrated_in = m_migrated_in;
    m_metrics.comm_wait_time_ms = m_accumulated_comm_wait_ms;
    m_metrics.compute_time_ms = std::max(0.0, m_metrics.total_sim_time_ms - m_metrics.comm_wait_time_ms);
    m_metrics.comm_to_compute_ratio = (m_metrics.total_sim_time_ms > 0.0)
        ? (m_metrics.comm_wait_time_ms / m_metrics.total_sim_time_ms)
        : 0.0;

    if (m_metrics.total_sim_time_ms > 0.0) {
        m_metrics.avg_step_time_us = (m_metrics.total_sim_time_ms * 1000.0) / std::max(1u, m_current_step);
        m_metrics.updates_per_second = (static_cast<double>(m_metrics.total_cell_updates) /
                                       (m_metrics.total_sim_time_ms / 1000.0));
    }

    m_metrics.avg_network_speed = (m_step_velocity_count > 0)
        ? (static_cast<float>(m_step_velocity_sum) / static_cast<float>(m_step_velocity_count))
        : 0.0f;
}

bool HybridEngine::verify_global_conservation(uint64_t* global_active,
                                              uint64_t* global_completed,
                                              uint64_t* global_injected) {
    uint64_t loc_active = m_active_vehicle_count;
    uint64_t loc_completed = m_completed_vehicles;
    uint64_t loc_injected = m_injected_vehicles;
    uint64_t loc_out = m_migrated_out;
    uint64_t loc_in = m_migrated_in;

    uint64_t tot_active = 0, tot_completed = 0, tot_injected = 0;
    uint64_t tot_out = 0, tot_in = 0;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    MPI_Allreduce(&loc_active, &tot_active, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc_completed, &tot_completed, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc_injected, &tot_injected, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc_out, &tot_out, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc_in, &tot_in, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#else
    tot_active = loc_active;
    tot_completed = loc_completed;
    tot_injected = loc_injected;
    tot_out = loc_out;
    tot_in = loc_in;
#endif

    if (global_active) *global_active = tot_active;
    if (global_completed) *global_completed = tot_completed;
    if (global_injected) *global_injected = tot_injected;

    const bool count_conserved = (tot_active + tot_completed == tot_injected);
    const bool migration_conserved = (tot_out == tot_in);

    return count_conserved && migration_conserved;
}

void HybridEngine::run() {
    if (m_config.snapshot_interval > 0) {
        save_snapshot(0, m_config.snapshot_dir + "/snapshot_t0.csv");
    }

    const auto t_start = std::chrono::high_resolution_clock::now();

    for (uint32_t s = 1; s <= m_config.total_steps; ++s) {
        step();
        if (m_config.snapshot_interval > 0 && (s % m_config.snapshot_interval == 0)) {
            const std::string snapshot_path = m_config.snapshot_dir + "/snapshot_t" + std::to_string(s) + ".csv";
            save_snapshot(s, snapshot_path);
        }
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    m_metrics.total_sim_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    m_metrics.comm_wait_time_ms = m_accumulated_comm_wait_ms;
    m_metrics.compute_time_ms = std::max(0.0, m_metrics.total_sim_time_ms - m_metrics.comm_wait_time_ms);
    m_metrics.comm_to_compute_ratio = (m_metrics.total_sim_time_ms > 0.0)
        ? (m_metrics.comm_wait_time_ms / m_metrics.total_sim_time_ms)
        : 0.0;
    update_metrics();

#if defined(HAS_MPI) || defined(MPI_VERSION)
    double loc_time = m_metrics.total_sim_time_ms;
    double max_time = 0.0, sum_time = 0.0;
    MPI_Reduce(&loc_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t loc_cell_updates = m_metrics.total_cell_updates;
    uint64_t tot_cell_updates = 0;
    MPI_Reduce(&loc_cell_updates, &tot_cell_updates, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t loc_v_sum = m_step_velocity_sum;
    uint32_t loc_v_cnt = m_step_velocity_count;
    uint64_t tot_v_sum = 0;
    uint32_t tot_v_cnt = 0;
    MPI_Reduce(&loc_v_sum, &tot_v_sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_v_cnt, &tot_v_cnt, 1, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

    double loc_comm_wait = m_metrics.comm_wait_time_ms;
    double max_comm_wait = 0.0;
    MPI_Reduce(&loc_comm_wait, &max_comm_wait, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (m_rank == 0) {
        m_metrics.max_rank_time_ms = max_time;
        m_metrics.avg_rank_time_ms = (m_total_ranks > 0) ? (sum_time / m_total_ranks) : max_time;
        m_metrics.load_imbalance_ratio = (m_metrics.avg_rank_time_ms > 0.0)
            ? (m_metrics.max_rank_time_ms / m_metrics.avg_rank_time_ms)
            : 1.0;
        m_metrics.total_cell_updates = tot_cell_updates;
        m_metrics.comm_wait_time_ms = max_comm_wait;
        m_metrics.compute_time_ms = std::max(0.0, max_time - max_comm_wait);
        m_metrics.comm_to_compute_ratio = (max_time > 0.0) ? (max_comm_wait / max_time) : 0.0;
        m_metrics.avg_network_speed = (tot_v_cnt > 0)
            ? (static_cast<float>(tot_v_sum) / static_cast<float>(tot_v_cnt))
            : 0.0f;

        if (max_time > 0.0) {
            m_metrics.avg_step_time_us = (max_time * 1000.0) / std::max(1u, m_config.total_steps);
            m_metrics.updates_per_second = (static_cast<double>(tot_cell_updates) / (max_time / 1000.0));
        }
    }
#else
    m_metrics.max_rank_time_ms = m_metrics.total_sim_time_ms;
    m_metrics.avg_rank_time_ms = m_metrics.total_sim_time_ms;
    m_metrics.load_imbalance_ratio = 1.0;
#endif
}

bool HybridEngine::save_snapshot(uint32_t step, const std::string& filepath) const {
    (void)step;
    const uint32_t seg_count = m_network.segment_count();
    std::fill(m_snapshot_counts.begin(), m_snapshot_counts.end(), 0);
    std::fill(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), 0);

    const int n_rank_segs = static_cast<int>(m_partition.rank_segments[m_rank].size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rank_segs; ++i) {
        const int32_t s_id = m_partition.rank_segments[m_rank][i];
        const auto& seg = m_segments[s_id];
        uint32_t v_cnt = 0;
        uint32_t v_spd = 0;

        for (int16_t l = 0; l < seg.lanes(); ++l) {
            for (int16_t c = 0; c < seg.cells_per_lane(); ++c) {
                const size_t g_idx = seg.local_to_global_cell(l, c);
                const int32_t v_id = m_current_cells[g_idx];
                if (is_occupied(v_id)) {
                    v_cnt++;
                    v_spd += m_vehicles[v_id].velocity;
                }
            }
        }
        m_snapshot_counts[s_id] = v_cnt;
        m_snapshot_speeds[s_id] = v_spd;
    }

#if defined(HAS_MPI) || defined(MPI_VERSION)
    MPI_Reduce(m_snapshot_counts.data(), m_global_snapshot_counts.data(),
               static_cast<int>(seg_count), MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(m_snapshot_speeds.data(), m_global_snapshot_speeds.data(),
               static_cast<int>(seg_count), MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    std::copy(m_snapshot_counts.begin(), m_snapshot_counts.end(), m_global_snapshot_counts.begin());
    std::copy(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), m_global_snapshot_speeds.begin());
#endif

    bool write_ok = true;
    if (m_rank == 0) {
        FILE* fp = std::fopen(filepath.c_str(), "w");
        if (!fp) {
            write_ok = false;
        } else {
            std::vector<char> io_buffer(262144);
            std::setvbuf(fp, io_buffer.data(), _IOFBF, io_buffer.size());

            std::fputs("segment_id,vehicle_count,avg_speed,density_ratio\n", fp);
            char line_buf[128];

            for (uint32_t s = 0; s < seg_count; ++s) {
                const uint32_t cnt = m_global_snapshot_counts[s];
                if (cnt == 0) {
                    std::snprintf(line_buf, sizeof(line_buf), "%u,0,0.0000,0.0000\n", s);
                } else {
                    const float avg_spd = static_cast<float>(m_global_snapshot_speeds[s]) / static_cast<float>(cnt);
                    const float density = static_cast<float>(cnt) / static_cast<float>(m_segments[s].total_cells());
                    std::snprintf(line_buf, sizeof(line_buf), "%u,%u,%.4f,%.4f\n", s, cnt, avg_spd, density);
                }
                std::fputs(line_buf, fp);
            }

            std::fclose(fp);
        }
    }

#if defined(HAS_MPI) || defined(MPI_VERSION)
    if (m_total_ranks > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    return write_ok;
}

} // namespace karachiflow
