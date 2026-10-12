#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace karachiflow {

struct Zone {
    std::string name;
    std::vector<int32_t> node_ids;
};

struct DemandMetadata {
    std::vector<Zone> origin_zones;
    std::vector<Zone> destination_zones;

    size_t total_origin_nodes() const noexcept {
        size_t count = 0;
        for (const auto& z : origin_zones) {
            count += z.node_ids.size();
        }
        return count;
    }

    size_t total_destination_nodes() const noexcept {
        size_t count = 0;
        for (const auto& z : destination_zones) {
            count += z.node_ids.size();
        }
        return count;
    }
};

bool load_demand_metadata(const std::string& filepath, DemandMetadata& out_metadata, std::string* error_msg = nullptr);

} // namespace karachiflow
