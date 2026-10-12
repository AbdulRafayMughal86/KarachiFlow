#pragma once

#include <cstdint>
#include <vector>
#include "cell.hpp"
#include "road_network.hpp"

namespace karachiflow {

struct SegmentMetrics {
    int32_t segment_id{0};
    uint32_t vehicle_count{0};
    float avg_speed{0.0f};
    float density_ratio{0.0f};
};

class RoadSegment {
public:
    RoadSegment() = default;
    explicit RoadSegment(const SegmentRecord& record) : m_record(record) {}

    const SegmentRecord& record() const noexcept { return m_record; }
    int32_t id() const noexcept { return m_record.segment_id; }
    int32_t from_node() const noexcept { return m_record.from_node; }
    int32_t to_node() const noexcept { return m_record.to_node; }
    int16_t lanes() const noexcept { return m_record.lanes; }
    int16_t cells_per_lane() const noexcept { return m_record.cells_per_lane; }
    int16_t speed_limit() const noexcept { return m_record.speed_limit; }
    int32_t cell_offset() const noexcept { return m_record.cell_offset; }
    float length_meters() const noexcept { return m_record.length_meters; }

    uint32_t total_cells() const noexcept {
        return static_cast<uint32_t>(m_record.lanes) * static_cast<uint32_t>(m_record.cells_per_lane);
    }

    size_t local_to_global_cell(int16_t lane, int16_t cell_idx) const noexcept {
        return static_cast<size_t>(m_record.cell_offset) +
               static_cast<size_t>(lane) * static_cast<size_t>(m_record.cells_per_lane) +
               static_cast<size_t>(cell_idx);
    }

private:
    SegmentRecord m_record{};
};

} // namespace karachiflow
