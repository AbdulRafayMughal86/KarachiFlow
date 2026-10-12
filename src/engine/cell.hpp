#pragma once

#include <cstdint>

namespace karachiflow {

// Represents a vacant 7.5-meter cell on a road lane.
constexpr int32_t EMPTY_CELL = -1;

// Global physical standard cell length (Nagel-Schreckenberg standard).
constexpr float CELL_LENGTH_METERS = 7.5f;

// 1 cell/s = 7.5 m/s = 27.0 km/h.
constexpr float CELLS_PER_SEC_TO_KMH = 27.0f;

inline bool is_empty(int32_t cell_value) noexcept {
    return cell_value == EMPTY_CELL;
}

inline bool is_occupied(int32_t cell_value) noexcept {
    return cell_value >= 0;
}

} // namespace karachiflow
