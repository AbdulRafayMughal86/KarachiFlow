"""
KarachiFlow — Geospatial Dashboard & Visualization Test Suite
==============================================================
Validates binary network ingestion, dynamic telemetry merging,
congestion classification physics, PyDeck layers, and Streamlit execution.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import visualizer
import metrics_plotter


class TestGeospatialVisualizer(unittest.TestCase):
    """Unit tests for road network ingestion and PyDeck layer builders."""

    @classmethod
    def setUpClass(cls):
        cls.bin_path = "data/karachi_network.bin"
        cls.metadata_path = "data/karachi_network_metadata.json"
        cls.snapshot_path = "data/snapshots/snapshot_t60.csv"

        if not os.path.exists(cls.bin_path):
            cls.bin_path = "data/karachi_synthetic.bin"
            cls.metadata_path = "data/karachi_synthetic_metadata.json"

    def test_network_geometry_loading(self):
        """Verifies binary road network decoding and coordinate computation."""
        df, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)

        self.assertGreater(len(df), 0)
        self.assertIn("segment_id", df.columns)
        self.assertIn("start_lat", df.columns)
        self.assertIn("start_lon", df.columns)
        self.assertIn("end_lat", df.columns)
        self.assertIn("end_lon", df.columns)
        self.assertIn("lanes", df.columns)
        self.assertIn("speed_limit", df.columns)

        # Ensure geographic coordinates are within Karachi bounds
        self.assertTrue(df["start_lat"].between(24.0, 26.0).all())
        self.assertTrue(df["start_lon"].between(66.0, 68.0).all())
        self.assertTrue((df["lanes"] >= 1).all())

    def test_missing_binary_raises_error(self):
        """Ensures non-existent binary path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            visualizer.load_network_geometry("data/non_existent.bin", self.metadata_path)

    def test_snapshot_loading(self):
        """Verifies CSV telemetry ingestion."""
        if os.path.exists(self.snapshot_path):
            df = visualizer.load_snapshot(self.snapshot_path)
            self.assertIn("segment_id", df.columns)
            self.assertIn("vehicle_count", df.columns)
            self.assertIn("avg_speed", df.columns)
            self.assertIn("density_ratio", df.columns)
            self.assertGreaterEqual(df["vehicle_count"].sum(), 0)

    def test_partial_empty_snapshot_resilience(self):
        """Verifies graceful handling of empty or partially flushed CSV snapshots without exceptions."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            df = visualizer.load_snapshot(tmp_path)
            self.assertIn("segment_id", df.columns)
            self.assertEqual(len(df), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_telemetry_merging_and_congestion_physics(self):
        """Verifies correct congestion state assignment based on speed ratio and density."""
        net_df, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)
        if os.path.exists(self.snapshot_path):
            snap_df = visualizer.load_snapshot(self.snapshot_path)
            merged = visualizer.merge_telemetry(net_df, snap_df)

            self.assertEqual(len(merged), len(net_df))
            self.assertIn("congestion_status", merged.columns)
            self.assertIn("color", merged.columns)
            self.assertIn("line_width", merged.columns)

            # Ensure valid categories
            valid_statuses = {
                "Empty Road",
                "Free Flow (Normal)",
                "Moderate Flow (Slowing)",
                "Severe Bottleneck (Queue)",
            }
            self.assertTrue(set(merged["congestion_status"].unique()).issubset(valid_statuses))

            # Active segments must have vehicles
            active_mask = merged["vehicle_count"] > 0
            empty_mask = merged["vehicle_count"] == 0
            self.assertTrue((merged.loc[empty_mask, "congestion_status"] == "Empty Road").all())
            self.assertFalse((merged.loc[active_mask, "congestion_status"] == "Empty Road").any())

    def test_zone_polygon_construction(self):
        """Verifies origin and destination zone polygons are properly extracted."""
        _, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)
        zones = visualizer.build_zone_polygons(meta)

        self.assertGreater(len(zones), 0)
        for z in zones:
            self.assertIn("zone_name", z)
            self.assertIn("polygon", z)
            self.assertIn("color", z)
            # Each polygon is a closed 5-point bounding box
            self.assertEqual(len(z["polygon"]), 5)
            self.assertEqual(z["polygon"][0], z["polygon"][-1])

    def test_pydeck_deck_builder(self):
        """Verifies complete PyDeck Deck object assembly."""
        net_df, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)
        if os.path.exists(self.snapshot_path):
            snap_df = visualizer.load_snapshot(self.snapshot_path)
            merged = visualizer.merge_telemetry(net_df, snap_df)
        else:
            dummy_snap = pd.DataFrame({
                "segment_id": net_df["segment_id"],
                "vehicle_count": [0] * len(net_df),
                "avg_speed": [0.0] * len(net_df),
                "density_ratio": [0.0] * len(net_df),
            })
            merged = visualizer.merge_telemetry(net_df, dummy_snap)

        presets = visualizer.get_viewport_presets(meta)
        view_state = presets["Metropolitan Overview"]

        # Test active traffic mode
        deck = visualizer.build_traffic_deck(
            merged_df=merged,
            metadata=meta,
            view_state=view_state,
            display_mode="Active Traffic Only",
            show_zones=True,
            show_heatmap=True,
        )
        self.assertIsNotNone(deck)
        self.assertGreaterEqual(len(deck.layers), 1)

    def test_zero_traffic_telemetry_merging(self):
        """Verifies graceful handling of empty network with 0 vehicles (e.g. at t=0)."""
        net_df, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)
        snap0_path = "data/snapshots/snapshot_t0.csv"
        if os.path.exists(snap0_path):
            snap_df = visualizer.load_snapshot(snap0_path)
            merged = visualizer.merge_telemetry(net_df, snap_df)

            self.assertEqual(merged["vehicle_count"].sum(), 0)
            self.assertEqual(merged["congestion_status"].value_counts().to_dict(), {"Empty Road": len(net_df)})
            self.assertEqual(len(merged[merged["congestion_status"] == "Severe Bottleneck (Queue)"]), 0)

    def test_all_display_modes_and_deck_serialization(self):
        """Verifies Deck creation and JSON serialization across all display modes."""
        net_df, meta = visualizer.load_network_geometry(self.bin_path, self.metadata_path)
        snap_df = visualizer.load_snapshot(self.snapshot_path)
        merged = visualizer.merge_telemetry(net_df, snap_df)
        presets = visualizer.get_viewport_presets(meta)

        display_modes = [
            "Active Traffic Only",
            "Congested Corridors Only",
            "Severe Bottlenecks Only",
            "Major Arteries + Active Traffic",
        ]

        for mode in display_modes:
            deck = visualizer.build_traffic_deck(
                merged_df=merged,
                metadata=meta,
                view_state=presets["Shahrah-e-Faisal Expressway"],
                display_mode=mode,
                show_zones=True,
                show_heatmap=True,
            )
            # Verify Deck to_json serialization
            deck_json = deck.to_json()
            self.assertIsInstance(deck_json, str)
            self.assertGreater(len(deck_json), 100)


class TestMetricsPlotter(unittest.TestCase):
    """Unit tests for PDC benchmark curves and figure generators."""

    def setUp(self):
        self.benchmarks = metrics_plotter.get_empirical_benchmarks()

    def test_benchmark_data_structure(self):
        """Verifies required benchmark measurements are present and positive."""
        self.assertIn("sequential", self.benchmarks)
        self.assertIn("openmp", self.benchmarks)
        self.assertIn("mpi", self.benchmarks)
        self.assertIn("hybrid", self.benchmarks)

        self.assertGreater(self.benchmarks["sequential"]["runtime_ms"], 0)
        self.assertEqual(len(self.benchmarks["openmp"]["threads"]), 4)
        self.assertEqual(len(self.benchmarks["mpi"]["ranks"]), 4)
        self.assertEqual(len(self.benchmarks["hybrid"]["total_workers"]), 4)

    def test_speedup_figure_generation(self):
        """Verifies self-relative and vs-sequential speedup figure plots."""
        fig1 = metrics_plotter.plot_speedup_curves(self.benchmarks, "Self-Relative (T1/Tp)")
        self.assertIsNotNone(fig1)
        self.assertGreater(len(fig1.data), 0)

        fig2 = metrics_plotter.plot_speedup_curves(self.benchmarks, "Speedup vs Sequential Baseline")
        self.assertIsNotNone(fig2)
        self.assertGreater(len(fig2.data), 0)

    def test_efficiency_and_runtime_figures(self):
        """Verifies parallel efficiency and runtime bar chart generators."""
        fig_eff = metrics_plotter.plot_efficiency_curves(self.benchmarks)
        self.assertIsNotNone(fig_eff)
        self.assertGreater(len(fig_eff.data), 0)

        fig_rt = metrics_plotter.plot_runtime_comparison(self.benchmarks)
        self.assertIsNotNone(fig_rt)
        self.assertGreater(len(fig_rt.data), 0)

    def test_mpi_communication_overhead_figure(self):
        """Verifies MPI communication overhead donut chart."""
        fig_comm = metrics_plotter.plot_mpi_communication_overhead(self.benchmarks)
        self.assertIsNotNone(fig_comm)
        self.assertEqual(len(fig_comm.data), 1)

    def test_load_imbalance_figure(self):
        """Verifies MPI load imbalance bar chart generator."""
        fig_imb = metrics_plotter.plot_load_imbalance(self.benchmarks)
        self.assertIsNotNone(fig_imb)
        self.assertEqual(len(fig_imb.data), 1)

    def test_summary_table_generation(self):
        """Verifies executive summary table formatting."""
        table = metrics_plotter.create_scaling_summary_table(self.benchmarks)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(len(table), 6)
        self.assertIn("Architecture", table.columns)
        self.assertIn("Speedup vs Seq", table.columns)


class TestDashboardStreamlitApp(unittest.TestCase):
    """End-to-end headless execution test of the Streamlit dashboard."""

    def test_streamlit_app_execution(self):
        """Executes the dashboard script through Streamlit's AppTest harness."""
        from streamlit.testing.v1 import AppTest

        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "dashboard.py"))
        at = AppTest.from_file(dashboard_path)
        at.run(timeout=30)

        # Assert no unhandled exceptions occurred
        self.assertEqual(len(at.exception), 0, f"App raised unhandled exceptions: {at.exception}")

        # Assert KPI metrics are populated across sidebar and main cards
        self.assertGreaterEqual(len(at.metric), 6)
        metric_labels = [m.label for m in at.metric]
        self.assertTrue(any("Active Vehicles" in l for l in metric_labels))
        self.assertTrue(any("Completed Trips" in l for l in metric_labels))
        self.assertTrue(any("Average" in l for l in metric_labels))

        # Assert main tabs are present
        self.assertEqual(len(at.tabs), 3)

        # Assert data tables are populated
        self.assertGreaterEqual(len(at.dataframe), 2)

    def test_streamlit_timeline_scrubbing(self):
        """Verifies time slider navigation to beginning (t=0), midway (t=30), and end (t=60)."""
        from streamlit.testing.v1 import AppTest

        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "dashboard.py"))
        at = AppTest.from_file(dashboard_path)
        at.run(timeout=30)

        # Find select_slider for time step
        self.assertGreater(len(at.select_slider), 0)
        time_slider = at.select_slider[0]

        # Set to t=0 (beginning)
        time_slider.set_value(0)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        veh_metric = [m for m in at.metric if "Active Vehicles" in m.label][0]
        comp_metric = [m for m in at.metric if "Completed Trips" in m.label][0]
        self.assertEqual(veh_metric.value, "0")
        self.assertEqual(comp_metric.value, "0")

        # Set to t=30 (midway)
        at.select_slider[0].set_value(30)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        veh_metric = [m for m in at.metric if "Active Vehicles" in m.label][0]
        self.assertEqual(veh_metric.value, "2,510")

        # Set to t=60 (final snapshot)
        at.select_slider[0].set_value(60)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        veh_metric = [m for m in at.metric if "Active Vehicles" in m.label][0]
        comp_metric = [m for m in at.metric if "Completed Trips" in m.label][0]
        self.assertEqual(veh_metric.value, "4,964")
        self.assertEqual(comp_metric.value, "36")

        # Test jump buttons (First, Next, Prev, Latest) execute without widget instantiation errors
        btn_dict = {btn.label: btn for btn in at.button}
        if "⏮ First" in btn_dict:
            btn_dict["⏮ First"].click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state.active_step, 0)

        btn_dict = {btn.label: btn for btn in at.button}
        if "Next ▶" in btn_dict:
            btn_dict["Next ▶"].click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state.active_step, 10)

        btn_dict = {btn.label: btn for btn in at.button}
        if "◀ Prev" in btn_dict:
            btn_dict["◀ Prev"].click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state.active_step, 0)

        btn_dict = {btn.label: btn for btn in at.button}
        if "Latest ⏭" in btn_dict:
            btn_dict["Latest ⏭"].click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.session_state.active_step, 60)

    def test_streamlit_live_monitor_mode(self):
        """Verifies execution when switching to Live Simulation Monitor mode."""
        from streamlit.testing.v1 import AppTest

        dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "dashboard.py"))
        at = AppTest.from_file(dashboard_path)
        at.run(timeout=30)

        mode_radios = [r for r in at.radio if "Operation Mode" in r.label]
        self.assertGreater(len(mode_radios), 0)
        mode_radios[0].set_value("Live Simulation Monitor")
        at.run(timeout=30)

        self.assertEqual(len(at.exception), 0)
        self.assertGreaterEqual(len(at.metric), 6)



if __name__ == "__main__":
    unittest.main()
