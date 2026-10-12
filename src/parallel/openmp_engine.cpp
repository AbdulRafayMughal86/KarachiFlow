#include "openmp_engine.hpp"
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace karachiflow {

void OpenMPEngine::apply_omp_environment() {
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
    omp_set_schedule(sched, m_omp_config.chunk_size);
    #endif
#else
    m_effective_threads = 1;
#endif

    m_thread_workspaces.resize(std::max(1, m_effective_threads));
    for (auto& ws : m_thread_workspaces) {
        ws.boundary_candidates.clear();
        ws.boundary_candidates.reserve(1024);
        ws.total_velocity = 0;
    }
}

bool OpenMPEngine::initialize(const SimulationConfig& config,
                              const OpenMPConfig& omp_config,
                              std::string* error_msg) {
    m_config = config;
    m_omp_config = omp_config;
    apply_omp_environment();

    // 1. Ingest binary road network
    if (!m_network.load_from_binary(m_config.network_path, error_msg)) {
        return false;
    }
    if (!m_network.validate(error_msg)) {
        return false;
    }

    // 2. Wrap segments
    m_segments.clear();
    m_segments.reserve(m_network.segment_count());
    for (uint32_t i = 0; i < m_network.segment_count(); ++i) {
        m_segments.emplace_back(m_network.get_segment(i));
    }

    // 3. Allocate ping-pong cell buffers
    const size_t total_cells = m_network.total_cells();
    m_current_cells.assign(total_cells, EMPTY_CELL);
    m_next_cells.assign(total_cells, EMPTY_CELL);

    // 4. Ingest OD demand metadata
    if (!load_demand_metadata(m_config.metadata_path, m_demand, error_msg)) {
        return false;
    }

    // 5. Precompute Reverse-Dijkstra routing tables using OpenMP
    precompute_routing_tables();

    // 6. Prepare origin candidate segments for demand injection
    prepare_demand_candidates();

    // 7. Allocate vehicle pool
    m_vehicles.clear();
    m_vehicles.resize(m_config.target_vehicles);
    for (uint32_t i = 0; i < m_config.target_vehicles; ++i) {
        m_vehicles[i].id = static_cast<int32_t>(i);
        m_vehicles[i].active = false;
    }

    // 8. Pre-allocate snapshot buffers
    m_snapshot_counts.assign(m_network.segment_count(), 0);
    m_snapshot_speeds.assign(m_network.segment_count(), 0);

    // 9. Ensure snapshot directory exists if snapshots are enabled
    if (m_config.snapshot_interval > 0) {
        std::error_code ec;
        std::filesystem::create_directories(m_config.snapshot_dir, ec);
        if (ec && error_msg) {
            *error_msg = "Failed to create snapshot directory: " + ec.message();
            return false;
        }
    }

    m_current_step = 0;
    m_active_vehicle_count = 0;
    m_completed_vehicles = 0;
    m_injected_vehicles = 0;
    m_metrics = SimulationMetrics{};

    return true;
}

void OpenMPEngine::precompute_routing_tables() {
    const uint32_t node_count = m_network.node_count();
    const uint32_t seg_count = m_network.segment_count();

    m_active_dest_nodes.clear();
    for (const auto& dz : m_demand.destination_zones) {
        if (!dz.node_ids.empty()) {
            m_active_dest_nodes.push_back(dz.node_ids);
        }
    }

    m_dest_zone_count = static_cast<uint32_t>(m_active_dest_nodes.size());
    if (m_dest_zone_count == 0) {
        m_routing_table.assign(static_cast<size_t>(node_count), -1);
        m_is_dest_node.assign(static_cast<size_t>(node_count), 0);
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

    // Parallelize shortest path tree calculation across destination zones
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

void OpenMPEngine::prepare_demand_candidates() {
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

void OpenMPEngine::inject_demand() {
    if (m_injected_vehicles >= m_config.target_vehicles) {
        return;
    }

    const uint32_t remaining_steps = (m_config.total_steps > m_current_step)
                                         ? (m_config.total_steps - m_current_step)
                                         : 1;
    const uint32_t remaining_vehicles = m_config.target_vehicles - m_injected_vehicles;
    const uint32_t batch_size = std::max(1u, (remaining_vehicles + remaining_steps - 1) / remaining_steps);
    const uint32_t inject_limit = std::min(batch_size, remaining_vehicles);

    const size_t num_candidates = m_origin_candidate_segments.size();
    if (num_candidates == 0) return;

    for (uint32_t i = 0; i < inject_limit && m_injected_vehicles < m_config.target_vehicles; ++i) {
        const uint32_t veh_id = m_injected_vehicles;
        const size_t base_cand_idx = (static_cast<size_t>(veh_id) * 37 + m_current_step * 17) % num_candidates;

        bool placed = false;
        for (size_t attempt = 0; attempt < 16 && !placed; ++attempt) {
            const size_t cand_idx = (base_cand_idx + attempt * 7919) % num_candidates;
            const int32_t seg_id = m_origin_candidate_segments[cand_idx];
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

void OpenMPEngine::update_vehicles_parallel() {
    const uint32_t num_veh = m_injected_vehicles;
    if (num_veh == 0) return;

    for (auto& ws : m_thread_workspaces) {
        ws.boundary_candidates.clear();
        ws.total_velocity = 0;
    }

    #pragma omp parallel
    {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        auto& ws = m_thread_workspaces[tid];

        #pragma omp for schedule(runtime)
        for (int i = 0; i < static_cast<int>(num_veh); ++i) {
            auto& v = m_vehicles[i];
            if (!v.active) continue;

            const auto& seg = m_segments[v.segment_id];
            const int16_t lane = v.lane;
            const int16_t cell_idx = v.cell_idx;
            const int16_t cpl = seg.cells_per_lane();
            const uint8_t v_max = std::min(v.max_velocity, static_cast<uint8_t>(seg.speed_limit()));

            // Step 1: Acceleration
            int speed = std::min(static_cast<int>(v.velocity) + 1, static_cast<int>(v_max));

            // Step 2: Deceleration (forward lookahead in read-only current_cells)
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
            if (speed > d_seg) {
                ws.boundary_candidates.push_back({v.id, static_cast<int16_t>(speed), static_cast<int16_t>(d_seg)});
            } else {
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

                ws.total_velocity += static_cast<uint8_t>(speed);
            }
        }
    }

    // Step C: Deterministic intersection and boundary conflict arbitration
    m_merged_candidates.clear();
    for (const auto& ws : m_thread_workspaces) {
        m_merged_candidates.insert(m_merged_candidates.end(),
                                   ws.boundary_candidates.begin(),
                                   ws.boundary_candidates.end());
    }

    std::sort(m_merged_candidates.begin(), m_merged_candidates.end(),
              [](const auto& a, const auto& b) noexcept { return a.id < b.id; });

    uint64_t boundary_v = 0;

    for (const auto& cand : m_merged_candidates) {
        auto& v = m_vehicles[cand.id];
        const auto& seg = m_segments[v.segment_id];
        const int16_t lane = v.lane;
        const int16_t cell_idx = v.cell_idx;
        int speed = cand.speed;
        const int d_seg = cand.d_seg;

        int32_t next_seg_id = -1;
        int16_t next_lane = 0;

        if (speed > d_seg) {
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
            }
        }

        if (speed > 0 && should_slow(static_cast<uint32_t>(v.id), m_current_step, m_config.p_slow)) {
            speed = speed - 1;
        }

        v.velocity = static_cast<uint8_t>(speed);
        v.steps_traveled++;
        v.cells_traveled += static_cast<uint32_t>(speed);

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
                boundary_v += speed;
            }
        } else {
            const int new_cell = cell_idx + speed;
            const size_t new_g_idx = seg.local_to_global_cell(lane, static_cast<int16_t>(new_cell));
            m_next_cells[new_g_idx] = v.id;

            v.cell_idx = static_cast<int16_t>(new_cell);
            boundary_v += speed;
        }
    }

    uint64_t total_v = boundary_v;
    for (const auto& ws : m_thread_workspaces) {
        total_v += ws.total_velocity;
    }
    m_metrics.avg_network_speed = (m_active_vehicle_count > 0)
        ? (static_cast<float>(total_v) / static_cast<float>(m_active_vehicle_count))
        : 0.0f;
}

void OpenMPEngine::step() {
    inject_demand();
    update_vehicles_parallel();

    std::swap(m_current_cells, m_next_cells);
    std::memset(m_next_cells.data(), 0xFF, m_next_cells.size() * sizeof(int32_t));

    m_current_step++;
    m_metrics.total_cell_updates += m_network.total_cells();
    update_metrics();
}

void OpenMPEngine::update_metrics() {
    m_metrics.steps_executed = m_current_step;
    m_metrics.active_vehicles = m_active_vehicle_count;
    m_metrics.completed_vehicles = m_completed_vehicles;
    m_metrics.injected_vehicles = m_injected_vehicles;

    if (m_metrics.total_sim_time_ms > 0.0) {
        m_metrics.avg_step_time_us = (m_metrics.total_sim_time_ms * 1000.0) / std::max(1u, m_current_step);
        m_metrics.updates_per_second = (static_cast<double>(m_metrics.total_cell_updates) /
                                       (m_metrics.total_sim_time_ms / 1000.0));
    }
}

void OpenMPEngine::run() {
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
    m_metrics.total_sim_time_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()) / 1000.0;

    update_metrics();
}

bool OpenMPEngine::save_snapshot(uint32_t /*step*/, const std::string& filepath) const {
    const uint32_t seg_count = m_network.segment_count();
    std::fill(m_snapshot_counts.begin(), m_snapshot_counts.end(), 0);
    std::fill(m_snapshot_speeds.begin(), m_snapshot_speeds.end(), 0);

    for (uint32_t i = 0; i < m_injected_vehicles; ++i) {
        const auto& v = m_vehicles[i];
        if (v.active && v.segment_id >= 0 && static_cast<uint32_t>(v.segment_id) < seg_count) {
            m_snapshot_counts[v.segment_id]++;
            m_snapshot_speeds[v.segment_id] += v.velocity;
        }
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
        const uint32_t cnt = m_snapshot_counts[s];
        if (cnt == 0) {
            std::snprintf(line_buf, sizeof(line_buf), "%u,0,0.0000,0.0000\n", s);
        } else {
            const float avg_spd = static_cast<float>(m_snapshot_speeds[s]) / static_cast<float>(cnt);
            const float density = static_cast<float>(cnt) / static_cast<float>(m_segments[s].total_cells());
            std::snprintf(line_buf, sizeof(line_buf), "%u,%u,%.4f,%.4f\n", s, cnt, avg_spd, density);
        }
        std::fputs(line_buf, fp);
    }

    std::fclose(fp);
    return true;
}

} // namespace karachiflow
