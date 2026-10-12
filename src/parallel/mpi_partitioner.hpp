#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "road_network.hpp"

namespace karachiflow {

struct CutEdge {
    int32_t src_segment_id{-1};
    int32_t dst_segment_id{-1};
    int32_t src_rank{-1};
    int32_t dst_rank{-1};
    int32_t boundary_node{-1};
};

struct PartitionResult {
    int total_ranks{1};
    std::vector<int32_t> segment_to_rank;
    std::vector<int32_t> node_to_rank;
    std::vector<std::vector<int32_t>> rank_segments;
    std::vector<std::vector<int32_t>> interior_segments;
    std::vector<std::vector<int32_t>> boundary_segments;
    std::vector<CutEdge> cut_edges;
    std::vector<std::vector<int>> rank_neighbors;
};

class SpatialPartitioner {
public:
    SpatialPartitioner() = default;

    // Partitions the road network into k geographic spatial zones using coordinate bisection.
    static PartitionResult partition(const RoadNetwork& network, int k_ranks);
};

} // namespace karachiflow
