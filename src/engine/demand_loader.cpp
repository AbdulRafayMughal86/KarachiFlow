#include "demand_loader.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdlib>

namespace karachiflow {

namespace {

bool parse_zones_section(const std::string& content, size_t section_start, std::vector<Zone>& out_zones) {
    size_t pos = content.find('{', section_start);
    if (pos == std::string::npos) return false;
    pos++;

    while (pos < content.size()) {
        while (pos < content.size() && (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
            pos++;
        }
        if (pos >= content.size() || content[pos] == '}') {
            break;
        }

        if (content[pos] != '"') {
            pos++;
            continue;
        }
        pos++;
        const size_t name_end = content.find('"', pos);
        if (name_end == std::string::npos) break;
        std::string zone_name = content.substr(pos, name_end - pos);
        pos = name_end + 1;

        const size_t obj_open = content.find('{', pos);
        if (obj_open == std::string::npos) break;

        size_t obj_close = obj_open + 1;
        int depth = 1;
        while (obj_close < content.size() && depth > 0) {
            if (content[obj_close] == '{') depth++;
            else if (content[obj_close] == '}') depth--;
            obj_close++;
        }

        Zone zone;
        zone.name = std::move(zone_name);
        const std::string zone_content = content.substr(obj_open, obj_close - obj_open);
        const size_t node_ids_pos = zone_content.find("\"node_ids\"");
        if (node_ids_pos != std::string::npos) {
            const size_t arr_open = zone_content.find('[', node_ids_pos);
            if (arr_open != std::string::npos) {
                const size_t arr_close = zone_content.find(']', arr_open);
                if (arr_close != std::string::npos) {
                    size_t p = arr_open + 1;
                    while (p < arr_close) {
                        while (p < arr_close && !std::isdigit(static_cast<unsigned char>(zone_content[p])) && zone_content[p] != '-') {
                            p++;
                        }
                        if (p < arr_close) {
                            const char* start_ptr = &zone_content[p];
                            char* end_ptr = nullptr;
                            const long val = std::strtol(start_ptr, &end_ptr, 10);
                            if (end_ptr > start_ptr) {
                                zone.node_ids.push_back(static_cast<int32_t>(val));
                                p += static_cast<size_t>(end_ptr - start_ptr);
                            } else {
                                p++;
                            }
                        }
                    }
                }
            }
        }

        out_zones.push_back(std::move(zone));
        pos = obj_close;
    }
    return true;
}

} // namespace

bool load_demand_metadata(const std::string& filepath, DemandMetadata& out_metadata, std::string* error_msg) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        if (error_msg) *error_msg = "Failed to open metadata file: " + filepath;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    out_metadata.origin_zones.clear();
    out_metadata.destination_zones.clear();

    const size_t orig_pos = content.find("\"origin_zones\"");
    if (orig_pos != std::string::npos) {
        parse_zones_section(content, orig_pos, out_metadata.origin_zones);
    } else {
        if (error_msg) *error_msg = "Key 'origin_zones' not found in metadata JSON";
        return false;
    }

    const size_t dest_pos = content.find("\"destination_zones\"");
    if (dest_pos != std::string::npos) {
        parse_zones_section(content, dest_pos, out_metadata.destination_zones);
    } else {
        if (error_msg) *error_msg = "Key 'destination_zones' not found in metadata JSON";
        return false;
    }

    return true;
}

} // namespace karachiflow
