#include "road_network.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::string bin_path = "data/karachi_network.bin";
    if (argc > 1) {
        bin_path = argv[1];
    }

    std::cout << "========================================================\n";
    std::cout << " KarachiFlow — Road Network Binary Ingestion Benchmark  \n";
    std::cout << "========================================================\n";
    std::cout << "Loading binary: " << bin_path << "\n";

    karachiflow::RoadNetwork network;
    std::string error_msg;

    const auto t_start = std::chrono::high_resolution_clock::now();
    bool loaded = network.load_from_binary(bin_path, &error_msg);
    const auto t_end = std::chrono::high_resolution_clock::now();

    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    const double duration_ms = static_cast<double>(duration_us) / 1000.0;

    if (!loaded) {
        std::cerr << "[ERROR] Failed to load binary: " << error_msg << "\n";
        return 1;
    }

    std::cout << "[SUCCESS] Loaded binary in " << std::fixed << std::setprecision(3)
              << duration_ms << " ms (" << duration_us << " us)\n";

    std::cout << "Validating topological invariants and CSR offsets...\n";
    const auto t_val_start = std::chrono::high_resolution_clock::now();
    bool valid = network.validate(&error_msg);
    const auto t_val_end = std::chrono::high_resolution_clock::now();
    const double val_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_val_end - t_val_start).count()) / 1000.0;

    if (!valid) {
        std::cerr << "[FAIL] Validation error: " << error_msg << "\n";
        return 1;
    }
    std::cout << "[SUCCESS] Invariants verified in " << std::fixed << std::setprecision(3) << val_ms << " ms\n";

    std::cout << "\nNetwork Summary Statistics:\n";
    std::cout << "  - Nodes:            " << network.node_count() << "\n";
    std::cout << "  - Road Segments:    " << network.segment_count() << "\n";
    std::cout << "  - Total 7.5m Cells: " << network.total_cells() << "\n";
    std::cout << "  - Adjacency Links:  " << network.total_adj() << "\n";

    if (network.node_count() > 0) {
        const auto& n0 = network.get_node(0);
        std::cout << "  - Sample Node 0:    OSM " << n0.osm_id << " (lat: " << n0.lat << ", lon: " << n0.lon << ")\n";
    }

    if (network.segment_count() > 0) {
        const auto& s0 = network.get_segment(0);
        std::cout << "  - Sample Segment 0: [" << s0.from_node << " -> " << s0.to_node
                  << "], Lanes: " << s0.lanes << ", Cells/Lane: " << s0.cells_per_lane
                  << ", Speed: " << s0.speed_limit << " cells/s (" << (s0.speed_limit * 27) << " km/h)"
                  << ", Offset: " << s0.cell_offset << "\n";
    }

    if (duration_ms < 1000.0) {
        std::cout << "\n[PASS] Ingestion latency within target threshold (< 1.0 second): "
                  << duration_ms << " ms < 1000 ms\n";
    } else {
        std::cerr << "\n[WARN] Exceeded 1.0 second threshold: " << duration_ms << " ms\n";
    }

    std::cout << "========================================================\n";
    return 0;
}
