#include "mpi_partitioner.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace karachiflow {

namespace {

struct SegmentCentroid {
    int32_t id{0};
    float lat{0.0f};
    float lon{0.0f};
};

// Karachi latitude is ~24.86 degrees N; cos(24.86 deg) relates lon delta to lat delta
constexpr float KARACHI_LON_SCALE = 0.90731f;

void bisect_recursive(std::vector<SegmentCentroid>& items,
                      size_t start,
                      size_t end,
                      int rank_start,
                      int num_ranks,
                      std::vector<int32_t>& seg_to_rank) {
    if (num_ranks <= 1 || start >= end) {
        for (size_t i = start; i < end; ++i) {
            seg_to_rank[items[i].id] = rank_start;
        }
        return;
    }

    float min_lat = items[start].lat, max_lat = items[start].lat;
    float min_lon = items[start].lon, max_lon = items[start].lon;
    for (size_t i = start + 1; i < end; ++i) {
        min_lat = std::min(min_lat, items[i].lat);
        max_lat = std::max(max_lat, items[i].lat);
        min_lon = std::min(min_lon, items[i].lon);
        max_lon = std::max(max_lon, items[i].lon);
    }

    const float span_lat = max_lat - min_lat;
    const float span_lon = (max_lon - min_lon) * KARACHI_LON_SCALE;

    const int left_ranks = num_ranks / 2;
    const int right_ranks = num_ranks - left_ranks;
    const size_t count = end - start;
    const size_t mid = start + (count * left_ranks) / num_ranks;

    if (span_lat >= span_lon) {
        std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end,
            [](const SegmentCentroid& a, const SegmentCentroid& b) noexcept {
                if (a.lat != b.lat) return a.lat < b.lat;
                return a.id < b.id;
            });
    } else {
        std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end,
            [](const SegmentCentroid& a, const SegmentCentroid& b) noexcept {
                if (a.lon != b.lon) return a.lon < b.lon;
                return a.id < b.id;
            });
    }

    bisect_recursive(items, start, mid, rank_start, left_ranks, seg_to_rank);
    bisect_recursive(items, mid, end, rank_start + left_ranks, right_ranks, seg_to_rank);
}

} // namespace

PartitionResult SpatialPartitioner::partition(const RoadNetwork& network, int k_ranks) {
    PartitionResult res;
    res.total_ranks = std::max(1, k_ranks);

    const uint32_t seg_count = network.segment_count();
    const uint32_t node_count = network.node_count();

    res.segment_to_rank.assign(seg_count, 0);
    res.node_to_rank.assign(node_count, 0);
    res.rank_segments.resize(res.total_ranks);
    res.interior_segments.resize(res.total_ranks);
    res.boundary_segments.resize(res.total_ranks);
    res.rank_neighbors.resize(res.total_ranks);

    if (seg_count == 0) {
        return res;
    }

    if (res.total_ranks == 1) {
        for (uint32_t i = 0; i < seg_count; ++i) {
            res.rank_segments[0].push_back(static_cast<int32_t>(i));
            res.interior_segments[0].push_back(static_cast<int32_t>(i));
        }
        return res;
    }

    std::vector<SegmentCentroid> items(seg_count);
    for (uint32_t i = 0; i < seg_count; ++i) {
        const auto& seg = network.get_segment(i);
        const auto& n_from = network.get_node(seg.from_node);
        const auto& n_to = network.get_node(seg.to_node);
        items[i] = {
            static_cast<int32_t>(i),
            0.5f * (n_from.lat + n_to.lat),
            0.5f * (n_from.lon + n_to.lon)
        };
    }

    bisect_recursive(items, 0, seg_count, 0, res.total_ranks, res.segment_to_rank);

    for (uint32_t i = 0; i < seg_count; ++i) {
        const int32_t r = res.segment_to_rank[i];
        if (r >= 0 && r < res.total_ranks) {
            res.rank_segments[r].push_back(static_cast<int32_t>(i));
        }
    }

    // Assign node to the rank owning the majority of its adjacent segments
    std::vector<std::vector<int32_t>> node_rank_votes(node_count, std::vector<int32_t>(res.total_ranks, 0));
    for (uint32_t i = 0; i < seg_count; ++i) {
        const auto& seg = network.get_segment(i);
        const int32_t r = res.segment_to_rank[i];
        if (seg.from_node >= 0 && static_cast<uint32_t>(seg.from_node) < node_count) {
            node_rank_votes[seg.from_node][r]++;
        }
        if (seg.to_node >= 0 && static_cast<uint32_t>(seg.to_node) < node_count) {
            node_rank_votes[seg.to_node][r]++;
        }
    }

    for (uint32_t u = 0; u < node_count; ++u) {
        int best_r = 0;
        int32_t max_votes = -1;
        for (int r = 0; r < res.total_ranks; ++r) {
            if (node_rank_votes[u][r] > max_votes) {
                max_votes = node_rank_votes[u][r];
                best_r = r;
            }
        }
        res.node_to_rank[u] = best_r;
    }

    // Classify interior vs boundary segments and identify cut edges
    std::vector<uint8_t> is_boundary(seg_count, 0);
    for (int r = 0; r < res.total_ranks; ++r) {
        for (int32_t s_id : res.rank_segments[r]) {
            const auto& seg = network.get_segment(s_id);
            const int32_t v = seg.to_node;
            auto [out_segs, out_count] = network.get_outgoing_segments(v);

            for (size_t k = 0; k < out_count; ++k) {
                const int32_t out_id = out_segs[k];
                const int32_t out_rank = res.segment_to_rank[out_id];
                if (out_rank != r) {
                    is_boundary[s_id] = 1;
                    is_boundary[out_id] = 1;
                    res.cut_edges.push_back({ s_id, out_id, r, out_rank, v });
                }
            }
        }
    }

    for (int r = 0; r < res.total_ranks; ++r) {
        for (int32_t s_id : res.rank_segments[r]) {
            if (is_boundary[s_id]) {
                res.boundary_segments[r].push_back(s_id);
            } else {
                res.interior_segments[r].push_back(s_id);
            }
        }
    }

    // Build unique neighbor rank list per rank
    for (const auto& cut : res.cut_edges) {
        res.rank_neighbors[cut.src_rank].push_back(cut.dst_rank);
        res.rank_neighbors[cut.dst_rank].push_back(cut.src_rank);
    }

    for (int r = 0; r < res.total_ranks; ++r) {
        auto& nbrs = res.rank_neighbors[r];
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }

    return res;
}

} // namespace karachiflow
