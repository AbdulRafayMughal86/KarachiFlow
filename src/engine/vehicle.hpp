#pragma once

#include <cstdint>
#include "cell.hpp"

namespace karachiflow {

struct Vehicle {
    int32_t id{-1};
    int32_t segment_id{-1};
    int16_t lane{0};
    int16_t cell_idx{0};
    uint8_t velocity{0};
    uint8_t max_velocity{2};
    int32_t origin_node{-1};
    int32_t dest_node{-1};
    int32_t dest_zone{-1};
    bool active{false};
    uint32_t steps_traveled{0};
    uint32_t cells_traveled{0};
};

inline uint32_t hash_rng(uint32_t vehicle_id, uint32_t step) noexcept {
    uint32_t h = vehicle_id ^ (step * 0x9e3779b9u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    return h ^ (h >> 16);
}

inline bool should_slow(uint32_t vehicle_id, uint32_t step, float p_slow) noexcept {
    if (p_slow <= 0.0f) return false;
    if (p_slow >= 1.0f) return true;
    const uint32_t threshold = static_cast<uint32_t>(p_slow * 4294967295.0);
    return hash_rng(vehicle_id, step) < threshold;
}

} // namespace karachiflow
