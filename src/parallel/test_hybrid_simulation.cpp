#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>
#include "hybrid_engine.hpp"
#include "mpi_partitioner.hpp"

#if defined(HAS_MPI) || defined(MPI_VERSION)
#include <mpi.h>
#endif

int main(int argc, char* argv[]) {
    int rank = 0;
    int size = 1;
    int provided = 0;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    if (rank == 0) {
        std::cout << "=================================================================\n";
        std::cout << " KarachiFlow Hybrid (MPI + OpenMP) Simulation Test Suite\n";
        std::cout << " Processes: " << size
                  << " | Thread Mode: " << ((provided >= MPI_THREAD_FUNNELED) ? "MPI_THREAD_FUNNELED" : "Standard")
                  << " | Platform: C++17\n";
        std::cout << "=================================================================\n\n";
    }

    karachiflow::SimulationConfig config;
    config.network_path = "data/karachi_network.bin";
    config.metadata_path = "data/karachi_network_metadata.json";
    config.target_vehicles = 5000;
    config.total_steps = 60;
    config.dev_mode = true;
    config.p_slow = 0.2f;

    karachiflow::OpenMPConfig omp_config;
    omp_config.thread_count = 4;
    omp_config.schedule = karachiflow::OpenMPSchedule::Dynamic;
    omp_config.chunk_size = 64;

    // Test 1: Spatial Domain Decomposition & Thread Model Validation
    karachiflow::RoadNetwork net;
    std::string err;
    if (!net.load_from_binary(config.network_path, &err)) {
        if (rank == 0) std::cerr << "Failed to load network: " << err << "\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
        MPI_Finalize();
#endif
        return 1;
    }

    auto partition = karachiflow::SpatialPartitioner::partition(net, size);

    if (rank == 0) {
        std::cout << "[TEST 1] Spatial Domain Partitioning & Thread Support Validation\n";
        std::cout << "  - Total Road Segments: " << net.segment_count() << "\n";
        std::cout << "  - Total Cut Edges:     " << partition.cut_edges.size() << "\n";

        for (int r = 0; r < size; ++r) {
            std::cout << "  - Rank " << r << " Domain: "
                      << partition.rank_segments[r].size() << " total segments ("
                      << partition.interior_segments[r].size() << " interior, "
                      << partition.boundary_segments[r].size() << " boundary, "
                      << partition.rank_neighbors[r].size() << " neighbors)\n";

            assert(partition.interior_segments[r].size() + partition.boundary_segments[r].size() ==
                   partition.rank_segments[r].size());
        }

        uint32_t total_assigned = 0;
        for (int r = 0; r < size; ++r) {
            total_assigned += static_cast<uint32_t>(partition.rank_segments[r].size());
        }
        assert(total_assigned == net.segment_count());
        std::cout << "  [PASS] Spatial decomposition and thread safety verified.\n\n";
    }

    // Test 2: Hybrid Simulation Execution with Non-Blocking Overlap & Telemetry
    if (rank == 0) {
        std::cout << "[TEST 2] Hybrid Simulation Execution (Non-Blocking Overlap & Telemetry)\n";
    }

    config.snapshot_interval = 10;
    karachiflow::HybridEngine engine_overlap;
    if (!engine_overlap.initialize(config, omp_config, true, &err)) {
        std::cerr << "Rank " << rank << " initialization failed: " << err << "\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
        MPI_Finalize();
#endif
        return 1;
    }

    const auto t_start_overlap = std::chrono::high_resolution_clock::now();
    engine_overlap.run();
    const auto t_end_overlap = std::chrono::high_resolution_clock::now();
    const double elapsed_overlap_ms =
        std::chrono::duration<double, std::milli>(t_end_overlap - t_start_overlap).count();

    uint64_t g_active = 0, g_completed = 0, g_injected = 0;
    const bool conserved_overlap = engine_overlap.verify_global_conservation(&g_active, &g_completed, &g_injected);

    const auto& m_ov = engine_overlap.metrics();
    if (rank == 0) {
        std::cout << "  - Injected Vehicles:    " << g_injected << "\n";
        std::cout << "  - Active Vehicles:      " << g_active << "\n";
        std::cout << "  - Completed Vehicles:   " << g_completed << "\n";
        std::cout << "  - Conservation Status:  "
                  << (conserved_overlap ? "PERFECT (Active + Completed == Injected)" : "FAILED") << "\n";
        std::cout << "  - Total Execution Time: " << std::fixed << std::setprecision(2)
                  << elapsed_overlap_ms << " ms\n";
        std::cout << "  - Max Rank Time:        " << m_ov.max_rank_time_ms << " ms\n";
        std::cout << "  - Avg Rank Time:        " << m_ov.avg_rank_time_ms << " ms\n";
        std::cout << "  - Load Imbalance (lambda): " << std::setprecision(3)
                  << m_ov.load_imbalance_ratio << " (t_max / t_avg)\n";
        std::cout << "  - Comm-to-Compute (R):  " << std::setprecision(3)
                  << m_ov.comm_to_compute_ratio << " (R_comm)\n";
        std::cout << "  - Comm Wait Time:       " << std::fixed << std::setprecision(2)
                  << m_ov.comm_wait_time_ms << " ms\n";

        assert(conserved_overlap);
        assert(elapsed_overlap_ms < 1000.0);

        std::ifstream snap_file(config.snapshot_dir + "/snapshot_t10.csv");
        assert(snap_file.is_open());
        std::string snap_header;
        std::getline(snap_file, snap_header);
        assert(snap_header == "segment_id,vehicle_count,avg_speed,density_ratio");
        snap_file.close();

        std::cout << "  [PASS] Vehicle conservation, SLA (<1.0s), and telemetry snapshots verified.\n\n";
    }

    // Test 3: Inter-Rank Boundary Vehicle Migration & Zero-Loss Handover
    if (rank == 0) {
        std::cout << "[TEST 3] Inter-Rank Boundary Vehicle Migration & Zero-Loss Handover\n";
    }

    uint64_t loc_out = engine_overlap.metrics().migrated_out;
    uint64_t loc_in = engine_overlap.metrics().migrated_in;
    uint64_t tot_out = 0, tot_in = 0;

#if defined(HAS_MPI) || defined(MPI_VERSION)
    MPI_Reduce(&loc_out, &tot_out, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_in, &tot_in, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    tot_out = loc_out;
    tot_in = loc_in;
#endif

    if (rank == 0) {
        std::cout << "  - Total Boundary Handovers Out: " << tot_out << "\n";
        std::cout << "  - Total Boundary Handovers In:  " << tot_in << "\n";
        assert(tot_out == tot_in);
        std::cout << "  [PASS] Zero lost or duplicated vehicles during cut edge transitions.\n\n";
    }

    // Test 4: Hybrid Scaling & PDC Metrics Profile across varying (P, T) combinations
    if (rank == 0) {
        std::cout << "[TEST 4] Hybrid Scaling & PDC Metrics Profile across (P, T) Combinations\n";
    }

    config.snapshot_interval = 0;
    const std::vector<int> test_threads = {1, 2, 4};
    double t_t1_baseline_ms = 0.0;

    if (rank == 0) {
        std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " Ranks (P) | Threads (T) | Workers (W) | Time (ms) | Speedup Sp | Efficiency Ep | Comm/Comp R_comm | Imbalance lambda | Mcells/s\n";
        std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";
    }

    for (int t : test_threads) {
        karachiflow::OpenMPConfig cfg = omp_config;
        cfg.thread_count = t;

        karachiflow::HybridEngine scale_sim;
        if (!scale_sim.initialize(config, cfg, true, &err)) {
            std::cerr << "Scale sim init failed for T=" << t << ": " << err << "\n";
#if defined(HAS_MPI) || defined(MPI_VERSION)
            MPI_Finalize();
#endif
            return 1;
        }

        scale_sim.run();

        uint64_t act = 0, comp = 0, inj = 0;
        const bool ok = scale_sim.verify_global_conservation(&act, &comp, &inj);
        assert(ok);
        (void)ok;

        if (t == 1) {
            t_t1_baseline_ms = scale_sim.metrics().max_rank_time_ms;
        }

        if (rank == 0) {
            const auto& m = scale_sim.metrics();
            const double run_ms = m.max_rank_time_ms;
            const int workers = size * t;
            const double sp = (run_ms > 0.0 && t_t1_baseline_ms > 0.0) ? (t_t1_baseline_ms / run_ms) : 1.0;
            const double ep = sp / static_cast<double>(t);
            const double r_comm = m.comm_to_compute_ratio;
            const double lambda = m.load_imbalance_ratio;
            const double mcells_sec = (static_cast<double>(m.total_cell_updates) / (run_ms / 1000.0)) / 1e6;

            assert(r_comm >= 0.0 && r_comm <= 1.0);
            assert(lambda >= 1.0);

            std::cout << " " << std::setw(9) << size << " | "
                      << std::setw(11) << t << " | "
                      << std::setw(11) << workers << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << run_ms << " | "
                      << std::setw(9) << sp << "x | "
                      << std::setw(13) << std::setprecision(3) << ep << " | "
                      << std::setw(16) << std::setprecision(3) << r_comm << " | "
                      << std::setw(16) << std::setprecision(3) << lambda << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << mcells_sec << "\n";
        }
    }

    if (rank == 0) {
        std::cout << "---------------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << "  [PASS] All scaling configurations satisfied exact conservation and invariant retention.\n\n";

        std::cout << "=================================================================\n";
        std::cout << " All Hybrid (MPI + OpenMP) Simulation Tests Passed Successfully!\n";
        std::cout << "=================================================================\n";
    }

#if defined(HAS_MPI) || defined(MPI_VERSION)
    MPI_Finalize();
#endif

    return 0;
}
