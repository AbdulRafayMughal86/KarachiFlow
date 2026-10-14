#include "mpi_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <cstring>
#include <filesystem>

namespace karachiflow {

bool MPIEngine::initialize(const SimulationConfig& config,
                           bool enable_overlap,
                           std::string* error_msg) {
    m_config = config;
    m_enable_overlap = enable_overlap;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        if (error_msg) *error_msg = "MPI has not been initialized. Call MPI_Init before MPIEngine::initialize().";
        return false;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_total_ranks);
#else
    m_rank = 0;
    m_total_ranks = 1;
#endif

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

    m_metrics = MPIEngineMetrics{};

    seed_interior_vehicles();

    return true;
}

void MPIEngine::precompute_routing_tables() {
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
                    pq.emplace(new_dist, edge.from_node);
                }
            }
        }

        for (uint32_t u = 0; u < node_count; ++u) {
            if (m_is_dest_node[static_cast<size_t>(u) * m_dest_zone_count + z]) {
                m_routing_table[static_cast<size_t>(u) * m_dest_zone_count + z] = -1;
                continue;
            }

            auto [out_segs, out_count] = m_network.get_outgoing_segments(static_cast<int32_t>(u));
            int32_t best_seg = -1;
            float best_cost = std::numeric_limits<float>::infinity();

            for (size_t k = 0; k < out_count; ++k) {
                const int32_t s_id = out_segs[k];
                const auto& s_rec = m_network.get_segment(s_id);
                const int32_t v = s_rec.to_node;
                if (v >= 0 && static_cast<uint32_t>(v) < node_count && dist[v] < std::numeric_limits<float>::infinity()) {
                    const float cost = s_rec.length_meters + dist[v];
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_seg = s_id;
                    }
                }
            }

            m_routing_table[static_cast<size_t>(u) * m_dest_zone_count + z] = best_seg;
        }
    }
}

void MPIEngine::prepare_demand_candidates() {
    m_origin_candidate_segments.clear();
    const uint32_t node_count = m_network.node_count();

    for (const auto& oz : m_demand.origin_zones) {
        for (int32_t node_id : oz.node_ids) {
            if (node_id >= 0 && static_cast<uint32_t>(node_id) < node_count) {
                auto [out_segs, out_count] = m_network.get_outgoing_segments(node_id);
                for (size_t k = 0; k < out_count; ++k) {
                    m_origin_candidate_segments.push_back(out_segs[k]);
                }
            }
        }
    }

    if (m_origin_candidate_segments.empty()) {
        const uint32_t seg_count = m_network.segment_count();
        m_origin_candidate_segments.reserve(seg_count);
        for (uint32_t i = 0; i < seg_count; ++i) {
            m_origin_candidate_segments.push_back(static_cast<int32_t>(i));
        }
    }
}

void MPIEngine::seed_interior_vehicles() {
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

void MPIEngine::inject_demand() {
    const uint32_t total_target = m_config.target_vehicles;
    const uint32_t base_share = total_target / m_total_ranks;
    const uint32_t remainder = total_target % m_total_ranks;

    // Disjoint vehicle ID slice per rank guarantees zero vehicle ID collisions
    const uint32_t local_target = base_share + (static_cast<uint32_t>(m_rank) < remainder ? 1u : 0u);
    const uint32_t local_id_start = static_cast<uint32_t>(m_rank) * base_share +
                                   std::min(static_cast<uint32_t>(m_rank), remainder);
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

    // Filter candidate segments to those owned by this rank
    std::vector<int32_t> local_origin_segs;
    local_origin_segs.reserve(m_origin_candidate_segments.size());
    for (int32_t s_id : m_origin_candidate_segments) {
        if (m_partition.segment_to_rank[s_id] == m_rank) {
            local_origin_segs.push_back(s_id);
        }
    }

    // Fallback: if demand origin zones have no segments in this rank, use all local segments
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

void MPIEngine::update_vehicle_nasch(Vehicle& v, const RoadSegment& seg, bool is_boundary) {
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
        // Vehicle stays within the current segment
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
            // Local rank transition
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

            // Yellow Box Rule: Entry cell vacancy constraint
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
            // Boundary cut edge transition: check halo cell vacancy on target rank
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
                // Entry cell occupied on neighbor rank: enforce Yellow Box Rule and hold vehicle on source segment
                speed = std::min(speed, d_seg);
            } else {
                // Vacant: cap speed to enter at entry cell 0
                if (speed > d_seg + 1) {
                    speed = d_seg + 1;
                }
                next_lane = chosen_lane;
                // Pre-emptively reserve halo cell on local rank to prevent intra-rank double booking
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
            // Reached destination or dead end: exit simulation
            v.active = false;
            m_completed_vehicles++;
            m_active_vehicle_count--;
        } else {
            const int32_t target_rank = m_partition.segment_to_rank[next_seg_id];

            if (target_rank == m_rank) {
                // Next segment is within local rank
                const auto& next_seg = m_segments[next_seg_id];
                const int new_cell = speed - (d_seg + 1);
                const size_t new_g_idx = next_seg.local_to_global_cell(next_lane, static_cast<int16_t>(new_cell));
                m_next_cells[new_g_idx] = v.id;

                v.segment_id = next_seg_id;
                v.lane = next_lane;
                v.cell_idx = static_cast<int16_t>(new_cell);
            } else if (is_boundary) {
                // Next segment is owned by another rank: pack into migration buffer
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

void MPIEngine::update_boundary_segments() {
    // Reset send buffer counts and pack halo cell occupancies for all neighbor ranks
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

    // Update vehicles situated on boundary segments
    for (int32_t s_id : m_partition.boundary_segments[m_rank]) {
        const auto& seg = m_segments[s_id];
        for (int16_t l = 0; l < seg.lanes(); ++l) {
            for (int16_t c = 0; c < seg.cells_per_lane(); ++c) {
                const size_t g_idx = seg.local_to_global_cell(l, c);
                const int32_t v_id = m_current_cells[g_idx];
                if (is_occupied(v_id)) {
                    update_vehicle_nasch(m_vehicles[v_id], seg, true);
                }
            }
        }
    }
}

void MPIEngine::initiate_boundary_exchange() {
#if defined(HAS_MPI) || defined(MPI_VERSION)
    m_mpi_requests.clear();
    const auto& neighbors = m_partition.rank_neighbors[m_rank];

    // Post non-blocking receives for incoming boundary packets
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

    // Post non-blocking sends for outgoing boundary packets
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

void MPIEngine::update_interior_segments() {
    // Overlapping computation loop: update all interior segments while network packets are in flight
    for (int32_t s_id : m_partition.interior_segments[m_rank]) {
        const auto& seg = m_segments[s_id];
        for (int16_t l = 0; l < seg.lanes(); ++l) {
            for (int16_t c = 0; c < seg.cells_per_lane(); ++c) {
                const size_t g_idx = seg.local_to_global_cell(l, c);
                const int32_t v_id = m_current_cells[g_idx];
                if (is_occupied(v_id)) {
                    update_vehicle_nasch(m_vehicles[v_id], seg, false);
                }
            }
        }
    }
}

void MPIEngine::finalize_boundary_exchange() {
#if defined(HAS_MPI) || defined(MPI_VERSION)
    if (!m_mpi_requests.empty()) {
        m_mpi_statuses.resize(m_mpi_requests.size());
        const auto t_start = std::chrono::high_resolution_clock::now();
        MPI_Waitall(static_cast<int>(m_mpi_requests.size()),
                    m_mpi_requests.data(),
                    m_mpi_statuses.data());
        const auto t_end = std::chrono::high_resolution_clock::now();
        m_metrics.comm_wait_time_ms +=
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_mpi_requests.clear();
    }
#endif
}

void MPIEngine::unpack_incoming_vehicles() {
    // 1. Unpack incoming boundary halo cell occupancies from neighbors
    for (int nbr : m_partition.rank_neighbors[m_rank]) {
        const auto& pkt = m_recv_packets[nbr];
        for (uint32_t h = 0; h < pkt.halo_count; ++h) {
            const auto& cell = pkt.halo_cells[h];
            const auto& seg = m_segments[cell.segment_id];
            const size_t g0 = seg.local_to_global_cell(cell.lane, 0);
            m_halo_cell_occupied[g0] = cell.occupied;
        }
    }

    // 2. Try placing any pending backlogged vehicles from previous timestep into m_next_cells
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

    // 3. Unpack incoming vehicles into m_next_cells
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
                    // Cell full across all lanes: place into arrival backlog queue
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

void MPIEngine::step() {
    m_step_velocity_sum = 0;
    m_step_velocity_count = 0;

    inject_demand();

    if (m_enable_overlap) {
        // Communication-computation overlap:
        // 1. Compute boundary segments and pack outgoing vehicles
        update_boundary_segments();
        // 2. Post non-blocking MPI_Isend / MPI_Irecv
        initiate_boundary_exchange();
        // 3. Compute interior segments concurrently while network buffers travel
        update_interior_segments();
        // 4. Wait for all boundary exchanges to finish
        finalize_boundary_exchange();
        // 5. Unpack newly arrived vehicles into local cells
        unpack_incoming_vehicles();
    } else {
        // Serialized mode (no overlap)
        update_boundary_segments();
        initiate_boundary_exchange();
        finalize_boundary_exchange();
        unpack_incoming_vehicles();
        update_interior_segments();
    }

    // Ping-pong double buffer swap
    std::swap(m_current_cells, m_next_cells);

    // Reset next_cells for next timestep in O(1) page clear
    std::memset(m_next_cells.data(), 0xFF, m_next_cells.size() * sizeof(int32_t));

    m_current_step++;

    // Tally cell updates on this rank's assigned segments
    uint64_t rank_cells = 0;
    for (int32_t s_id : m_partition.rank_segments[m_rank]) {
        rank_cells += m_segments[s_id].total_cells();
    }
    m_metrics.total_cell_updates += rank_cells;

    update_metrics();
}

void MPIEngine::update_metrics() {
    m_metrics.steps_executed = m_current_step;
    m_metrics.active_vehicles = m_active_vehicle_count;
    m_metrics.completed_vehicles = m_completed_vehicles;
    m_metrics.injected_vehicles = m_injected_vehicles;
    m_metrics.migrated_out = m_migrated_out;
    m_metrics.migrated_in = m_migrated_in;

    if (m_metrics.total_sim_time_ms > 0.0) {
        m_metrics.avg_step_time_us = (m_metrics.total_sim_time_ms * 1000.0) / std::max(1u, m_current_step);
        m_metrics.updates_per_second = (static_cast<double>(m_metrics.total_cell_updates) /
                                       (m_metrics.total_sim_time_ms / 1000.0));
    }

    m_metrics.avg_network_speed = (m_step_velocity_count > 0)
        ? (static_cast<float>(m_step_velocity_sum) / static_cast<float>(m_step_velocity_count))
        : 0.0f;
}

bool MPIEngine::verify_global_conservation(uint64_t* global_active,
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

void MPIEngine::run() {
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

    if (m_rank == 0) {
        m_metrics.max_rank_time_ms = max_time;
        m_metrics.avg_rank_time_ms = sum_time / std::max(1, m_total_ranks);
        m_metrics.load_imbalance_ratio = max_time / std::max(0.001, m_metrics.avg_rank_time_ms);
        m_metrics.total_cell_updates = tot_cell_updates;
        if (m_metrics.total_sim_time_ms > 0.0) {
            m_metrics.updates_per_second = (static_cast<double>(tot_cell_updates) /
                                           (m_metrics.total_sim_time_ms / 1000.0));
        }
        if (tot_v_cnt > 0) {
            m_metrics.avg_network_speed = static_cast<float>(tot_v_sum) / static_cast<float>(tot_v_cnt);
        }
    }
#endif
}

bool MPIEngine::save_snapshot(uint32_t /*step*/, const std::string& filepath) const {
    const uint32_t seg_count = m_network.segment_count();
    std::fill(m_snapshot_counts.begin(), m_snapshot_counts.end(), 0);
    std::fill(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), 0);

    for (int32_t s_id : m_partition.rank_segments[m_rank]) {
        const auto& seg = m_segments[s_id];
        for (int16_t l = 0; l < seg.lanes(); ++l) {
            for (int16_t c = 0; c < seg.cells_per_lane(); ++c) {
                const size_t g_idx = seg.local_to_global_cell(l, c);
                const int32_t v_id = m_current_cells[g_idx];
                if (is_occupied(v_id)) {
                    m_snapshot_counts[s_id]++;
                    m_snapshot_speeds[s_id] += m_vehicles[v_id].velocity;
                }
            }
        }
    }

#if defined(HAS_MPI) || defined(MPI_VERSION)
    if (m_total_ranks > 1) {
        MPI_Reduce(m_snapshot_counts.data(),
                   m_global_snapshot_counts.data(),
                   static_cast<int>(seg_count),
                   MPI_UINT32_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
        MPI_Reduce(m_snapshot_speeds.data(),
                   m_global_snapshot_speeds.data(),
                   static_cast<int>(seg_count),
                   MPI_UINT32_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
    } else {
        std::copy(m_snapshot_counts.begin(), m_snapshot_counts.end(), m_global_snapshot_counts.begin());
        std::copy(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), m_global_snapshot_speeds.begin());
    }
#else
    std::copy(m_snapshot_counts.begin(), m_snapshot_counts.end(), m_global_snapshot_counts.begin());
    std::copy(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), m_global_snapshot_speeds.begin());
#endif

    if (m_rank != 0) {
        return true;
    }

    FILE* fp = std::fopen(filepath.c_str(), "w");
    if (!fp) {
        return false;
    }

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
    return true;
}

} // namespace karachiflow
