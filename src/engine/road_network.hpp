#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace karachiflow {

#pragma pack(push, 1)

struct RoadNetworkHeader {
    char magic[4];          // Must be "KFLW"
    uint32_t version;       // Format version (currently 1)
    uint32_t node_count;    // Total nodes (N)
    uint32_t segment_count; // Total directed road segments (M)
    uint32_t total_cells;   // Sum of (lanes * cells_per_lane) across all segments
    uint32_t total_adj;     // Total outgoing adjacency links in CSR table
    uint32_t reserved[2];   // Alignment padding / future extensions
};
static_assert(sizeof(RoadNetworkHeader) == 32, "RoadNetworkHeader must be exactly 32 bytes");

struct NodeRecord {
    float lat;              // Latitude (WGS84 EPSG:4326)
    float lon;              // Longitude (WGS84 EPSG:4326)
    int64_t osm_id;         // Original OpenStreetMap 64-bit node identifier
};
static_assert(sizeof(NodeRecord) == 16, "NodeRecord must be exactly 16 bytes");

struct SegmentRecord {
    int32_t segment_id;     // 0-indexed contiguous segment identifier
    int32_t from_node;      // 0-indexed origin node
    int32_t to_node;        // 0-indexed destination node
    int16_t lanes;          // Number of parallel lanes (1..8)
    int16_t cells_per_lane; // Discrete 7.5m cell count per lane (>= 1)
    int16_t speed_limit;    // Speed limit in discrete cells/sec (1..4)
    int16_t flags;          // Bitmask: bit 0 = oneway, bit 1 = bridge, etc.
    int32_t cell_offset;    // Starting index in global flat simulation cell array
    float length_meters;    // Physical length in meters
};
static_assert(sizeof(SegmentRecord) == 28, "SegmentRecord must be exactly 28 bytes");

#pragma pack(pop)

/**
 * RoadNetwork
 * High-performance, contiguous C++17 representation of the road network
 * loaded directly from the flat binary format.
 */
class RoadNetwork {
public:
    RoadNetwork() = default;

    /**
     * Loads the road network from a binary file in under 10ms.
     */
    bool load_from_binary(const std::string& filepath, std::string* error_msg = nullptr) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            if (error_msg) *error_msg = "Failed to open file: " + filepath;
            return false;
        }

        const std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size < static_cast<std::streamsize>(sizeof(RoadNetworkHeader))) {
            if (error_msg) *error_msg = "File too small to contain valid header";
            return false;
        }

        // 1. Read and validate Header
        file.read(reinterpret_cast<char*>(&m_header), sizeof(RoadNetworkHeader));
        if (std::memcmp(m_header.magic, "KFLW", 4) != 0) {
            if (error_msg) *error_msg = "Invalid magic bytes in file header (expected 'KFLW')";
            return false;
        }
        if (m_header.version != 1) {
            if (error_msg) *error_msg = "Unsupported format version: " + std::to_string(m_header.version);
            return false;
        }

        // Compute expected file size to guard against truncation or corrupted writes
        const std::streamsize expected_size =
            sizeof(RoadNetworkHeader) +
            static_cast<std::streamsize>(m_header.node_count) * sizeof(NodeRecord) +
            static_cast<std::streamsize>(m_header.segment_count) * sizeof(SegmentRecord) +
            static_cast<std::streamsize>(m_header.node_count + 1) * sizeof(int32_t) +
            static_cast<std::streamsize>(m_header.total_adj) * sizeof(int32_t);

        if (file_size != expected_size) {
            if (error_msg) {
                *error_msg = "File size mismatch: actual " + std::to_string(file_size) +
                             " bytes, expected " + std::to_string(expected_size) + " bytes";
            }
            return false;
        }

        // 2. Read Node Table
        m_nodes.resize(m_header.node_count);
        if (m_header.node_count > 0) {
            file.read(reinterpret_cast<char*>(m_nodes.data()),
                      m_header.node_count * sizeof(NodeRecord));
        }

        // 3. Read Segment Table
        m_segments.resize(m_header.segment_count);
        if (m_header.segment_count > 0) {
            file.read(reinterpret_cast<char*>(m_segments.data()),
                      m_header.segment_count * sizeof(SegmentRecord));
        }

        // 4. Read Topology CSR Offsets
        m_node_offsets.resize(m_header.node_count + 1);
        file.read(reinterpret_cast<char*>(m_node_offsets.data()),
                  (m_header.node_count + 1) * sizeof(int32_t));

        // 5. Read Topology CSR Outgoing Segments
        m_outgoing_segments.resize(m_header.total_adj);
        if (m_header.total_adj > 0) {
            file.read(reinterpret_cast<char*>(m_outgoing_segments.data()),
                      m_header.total_adj * sizeof(int32_t));
        }

        return true;
    }

    /**
     * Verifies structural invariants and topological consistency.
     */
    bool validate(std::string* error_msg = nullptr) const {
        if (std::memcmp(m_header.magic, "KFLW", 4) != 0) {
            if (error_msg) *error_msg = "Invalid magic bytes";
            return false;
        }

        if (m_nodes.size() != m_header.node_count) {
            if (error_msg) *error_msg = "Node vector size mismatch";
            return false;
        }

        if (m_segments.size() != m_header.segment_count) {
            if (error_msg) *error_msg = "Segment vector size mismatch";
            return false;
        }

        if (m_node_offsets.size() != m_header.node_count + 1) {
            if (error_msg) *error_msg = "Node offsets vector size mismatch";
            return false;
        }

        if (m_node_offsets[0] != 0 || m_node_offsets[m_header.node_count] != static_cast<int32_t>(m_header.total_adj)) {
            if (error_msg) *error_msg = "CSR offset boundaries invalid";
            return false;
        }

        // Validate segments and cell offsets
        int32_t expected_cell_offset = 0;
        for (uint32_t i = 0; i < m_header.segment_count; ++i) {
            const auto& seg = m_segments[i];

            if (seg.segment_id != static_cast<int32_t>(i)) {
                if (error_msg) *error_msg = "Segment ID non-contiguous at index " + std::to_string(i);
                return false;
            }

            if (seg.from_node < 0 || static_cast<uint32_t>(seg.from_node) >= m_header.node_count) {
                if (error_msg) *error_msg = "Invalid from_node out of range at segment " + std::to_string(i);
                return false;
            }

            if (seg.to_node < 0 || static_cast<uint32_t>(seg.to_node) >= m_header.node_count) {
                if (error_msg) *error_msg = "Invalid to_node out of range at segment " + std::to_string(i);
                return false;
            }

            if (seg.lanes < 1 || seg.lanes > 8) {
                if (error_msg) *error_msg = "Invalid lane count at segment " + std::to_string(i);
                return false;
            }

            if (seg.cells_per_lane < 1) {
                if (error_msg) *error_msg = "Degenerate cells_per_lane < 1 at segment " + std::to_string(i);
                return false;
            }

            if (seg.speed_limit < 1 || seg.speed_limit > 4) {
                if (error_msg) *error_msg = "Invalid speed limit out of [1..4] at segment " + std::to_string(i);
                return false;
            }

            if (seg.cell_offset != expected_cell_offset) {
                if (error_msg) *error_msg = "Cell offset mismatch at segment " + std::to_string(i);
                return false;
            }

            expected_cell_offset += seg.lanes * seg.cells_per_lane;
        }

        if (expected_cell_offset != static_cast<int32_t>(m_header.total_cells)) {
            if (error_msg) *error_msg = "Total cells mismatch with segment sum";
            return false;
        }

        // Validate CSR monotonicity and edge origins
        for (uint32_t u = 0; u < m_header.node_count; ++u) {
            const int32_t start = m_node_offsets[u];
            const int32_t end = m_node_offsets[u + 1];

            if (start > end) {
                if (error_msg) *error_msg = "Non-monotonic CSR offset at node " + std::to_string(u);
                return false;
            }

            for (int32_t idx = start; idx < end; ++idx) {
                const int32_t seg_id = m_outgoing_segments[idx];
                if (seg_id < 0 || static_cast<uint32_t>(seg_id) >= m_header.segment_count) {
                    if (error_msg) *error_msg = "CSR outgoing segment out of range at node " + std::to_string(u);
                    return false;
                }
                if (m_segments[seg_id].from_node != static_cast<int32_t>(u)) {
                    if (error_msg) *error_msg = "CSR outgoing segment origin does not match node " + std::to_string(u);
                    return false;
                }
            }
        }

        return true;
    }

    // Accessors
    const RoadNetworkHeader& header() const { return m_header; }
    uint32_t node_count() const { return m_header.node_count; }
    uint32_t segment_count() const { return m_header.segment_count; }
    uint32_t total_cells() const { return m_header.total_cells; }
    uint32_t total_adj() const { return m_header.total_adj; }

    const std::vector<NodeRecord>& nodes() const { return m_nodes; }
    const std::vector<SegmentRecord>& segments() const { return m_segments; }
    const std::vector<int32_t>& node_offsets() const { return m_node_offsets; }
    const std::vector<int32_t>& outgoing_segments() const { return m_outgoing_segments; }

    const NodeRecord& get_node(uint32_t idx) const { return m_nodes[idx]; }
    const SegmentRecord& get_segment(uint32_t idx) const { return m_segments[idx]; }

    /**
     * O(1) retrieval of outgoing segment IDs for a given node.
     * Returns pointer to first outgoing segment ID and count.
     */
    std::pair<const int32_t*, size_t> get_outgoing_segments(int32_t node_id) const {
        if (node_id < 0 || static_cast<uint32_t>(node_id) >= m_header.node_count) {
            return {nullptr, 0};
        }
        const int32_t start = m_node_offsets[node_id];
        const int32_t end = m_node_offsets[node_id + 1];
        return {&m_outgoing_segments[start], static_cast<size_t>(end - start)};
    }

private:
    RoadNetworkHeader m_header{};
    std::vector<NodeRecord> m_nodes;
    std::vector<SegmentRecord> m_segments;
    std::vector<int32_t> m_node_offsets;
    std::vector<int32_t> m_outgoing_segments;
};

} // namespace karachiflow
