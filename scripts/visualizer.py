"""
KarachiFlow — Geospatial Traffic Visualizer & PyDeck Layer Helpers
===================================================================
Constructs high-performance geospatial layers for rendering microscopic
traffic flow, congestion shockwaves, and road network topology in Karachi.
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import pydeck as pdk

# Magic bytes and header size for the flat binary road network format
MAGIC_BYTES = b"KFLW"
HEADER_SIZE_BYTES = 32
NODE_STRUCT_BYTES = 16    # float32 lat, float32 lon, int64 osm_id
SEGMENT_STRUCT_BYTES = 28 # int32 id, int32 from, int32 to, int16 lanes, int16 cells, int16 speed, int16 flags, int32 offset, float32 len

# Discrete speed mapping: 1 cell/s = 7.5 m/s = 27.0 km/h
KMH_PER_CELL_PER_SEC = 27.0

# Congestion color palette (RGBA) — High-contrast neon tones on dark basemap
COLOR_FREE_FLOW = [0, 230, 118, 245]       # Vibrant Neon Green: v >= 0.75 vmax, low density
COLOR_MODERATE = [255, 214, 0, 245]        # Bright Amber Yellow: 0.4 vmax <= v < 0.75 vmax
COLOR_BOTTLENECK = [255, 23, 68, 255]      # Vivid Glowing Crimson Red: v < 0.4 vmax or density >= 0.5
COLOR_INACTIVE = [50, 60, 80, 50]          # Faint subtle slate: No active vehicles

# Zone boundary colors (RGBA) — Thin crisp outlines without blocking fills
COLOR_RESIDENTIAL_ZONE = [0, 210, 255, 180]  # Crisp Neon Cyan outline
COLOR_COMMERCIAL_ZONE = [255, 150, 0, 180]   # Warm Amber outline


def load_network_geometry(bin_path: str, metadata_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses the binary road network and metadata into a vectorized DataFrame of segments.
    Computes geographic coordinates (start_lon, start_lat, end_lon, end_lat) for every road.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Road network binary not found at '{bin_path}'")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Network metadata not found at '{metadata_path}'")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(bin_path, "rb") as f:
        header_data = f.read(HEADER_SIZE_BYTES)
        magic, version, node_count, segment_count, total_cells, total_adj, _, _ = struct.unpack(
            "<4sIIIIIII", header_data
        )

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic bytes in road network binary: {magic}")

        # Vectorized node loading
        nodes_dtype = np.dtype([("lat", "<f4"), ("lon", "<f4"), ("osm_id", "<i8")])
        nodes = np.fromfile(f, dtype=nodes_dtype, count=node_count)

        # Vectorized segment loading
        segs_dtype = np.dtype([
            ("segment_id", "<i4"),
            ("from_node", "<i4"),
            ("to_node", "<i4"),
            ("lanes", "<i2"),
            ("cells_per_lane", "<i2"),
            ("speed_limit", "<i2"),
            ("flags", "<i2"),
            ("cell_offset", "<i4"),
            ("length_meters", "<f4"),
        ])
        segments = np.fromfile(f, dtype=segs_dtype, count=segment_count)

    from_nodes = segments["from_node"]
    to_nodes = segments["to_node"]

    start_lats = nodes["lat"][from_nodes]
    start_lons = nodes["lon"][from_nodes]
    end_lats = nodes["lat"][to_nodes]
    end_lons = nodes["lon"][to_nodes]

    df = pd.DataFrame({
        "segment_id": segments["segment_id"].astype(np.int32),
        "from_node": from_nodes.astype(np.int32),
        "to_node": to_nodes.astype(np.int32),
        "lanes": segments["lanes"].astype(np.int16),
        "cells_per_lane": segments["cells_per_lane"].astype(np.int16),
        "speed_limit": segments["speed_limit"].astype(np.int16),
        "speed_limit_kmh": (segments["speed_limit"] * KMH_PER_CELL_PER_SEC).astype(np.float32),
        "length_meters": segments["length_meters"].astype(np.float32),
        "total_cells": (segments["lanes"] * segments["cells_per_lane"]).astype(np.int32),
        "start_lon": start_lons.astype(np.float64),
        "start_lat": start_lats.astype(np.float64),
        "end_lon": end_lons.astype(np.float64),
        "end_lat": end_lats.astype(np.float64),
        "mid_lon": ((start_lons + end_lons) * 0.5).astype(np.float64),
        "mid_lat": ((start_lats + end_lats) * 0.5).astype(np.float64),
    })

    return df, metadata


def load_snapshot(snapshot_path: str) -> pd.DataFrame:
    """Loads a microscopic simulation snapshot CSV file."""
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot file not found at '{snapshot_path}'")
    
    try:
        df = pd.read_csv(snapshot_path)
        required_cols = {"segment_id", "vehicle_count", "avg_speed", "density_ratio"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Snapshot CSV missing required columns: {required_cols - set(df.columns)}")
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=["segment_id", "vehicle_count", "avg_speed", "density_ratio"])


def merge_telemetry(network_df: pd.DataFrame, snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges segment geometry with dynamic simulation telemetry and evaluates
    real-time congestion states and rendering visual properties.
    """
    merged = network_df.merge(snapshot_df, on="segment_id", how="left")
    merged["vehicle_count"] = merged["vehicle_count"].fillna(0).astype(int)
    merged["avg_speed"] = merged["avg_speed"].fillna(0.0).astype(float)
    merged["density_ratio"] = merged["density_ratio"].fillna(0.0).astype(float)

    # Derived physics indicators
    merged["avg_speed_kmh"] = (merged["avg_speed"] * KMH_PER_CELL_PER_SEC).round(1)
    merged["density_pct"] = (merged["density_ratio"] * 100.0).round(1)

    v_max = np.maximum(merged["speed_limit"].values, 1)
    speed_ratio = np.clip(merged["avg_speed"].values / v_max, 0.0, 1.0)
    density = merged["density_ratio"].values
    cars = merged["vehicle_count"].values

    # Congestion status assignment
    # Red: severe bottleneck (queue formation, density >= 0.4 with cars, or speed < 0.4 vmax)
    is_bottleneck = (cars > 0) & ((density >= 0.4) | (speed_ratio < 0.4))
    # Yellow: moderate slowing (speed < 0.75 vmax or density >= 0.2)
    is_moderate = (cars > 0) & (~is_bottleneck) & ((density >= 0.2) | (speed_ratio < 0.75))
    # Green: free flow
    is_free_flow = (cars > 0) & (~is_bottleneck) & (~is_moderate)

    status_list = np.full(len(merged), "Empty Road", dtype=object)
    status_list[is_free_flow] = "Free Flow (Normal)"
    status_list[is_moderate] = "Moderate Flow (Slowing)"
    status_list[is_bottleneck] = "Severe Bottleneck (Queue)"
    merged["congestion_status"] = status_list

    # Assign RGBA color tuples as Python lists (critical for PyDeck JSON serialization)
    colors = np.empty((len(merged), 4), dtype=np.int32)
    colors[:] = COLOR_INACTIVE
    colors[is_free_flow] = COLOR_FREE_FLOW
    colors[is_moderate] = COLOR_MODERATE
    colors[is_bottleneck] = COLOR_BOTTLENECK
    merged["color"] = colors.tolist()

    # Sleek line widths: sharp and crisp to prevent black blobs on high density
    lanes = merged["lanes"].values
    widths = np.full(len(merged), 1.0, dtype=np.float32)
    active_mask = cars > 0
    widths[active_mask] = np.clip(1.5 + lanes[active_mask] * 0.4 + density[active_mask] * 1.5, 1.8, 3.8)
    merged["line_width"] = widths

    return merged


def get_viewport_presets(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, pdk.ViewState]:
    """Provides curated geographic camera viewpoints across Karachi."""
    bbox = metadata.get("bounding_box", {}) if metadata else {}
    min_lat = bbox.get("min_lat", 24.840)
    max_lat = bbox.get("max_lat", 24.935)
    min_lon = bbox.get("min_lon", 67.010)
    max_lon = bbox.get("max_lon", 67.165)

    center_lat = (min_lat + max_lat) * 0.5
    center_lon = (min_lon + max_lon) * 0.5

    return {
        "Metropolitan Overview": pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11.8,
            pitch=0,
            bearing=0,
        ),
        "Shahrah-e-Faisal Expressway": pdk.ViewState(
            latitude=24.868,
            longitude=67.082,
            zoom=13.5,
            pitch=15,
            bearing=-25,
        ),
        "University Road Arterial": pdk.ViewState(
            latitude=24.925,
            longitude=67.112,
            zoom=13.5,
            pitch=15,
            bearing=20,
        ),
        "Saddar & I.I. Chundrigar Downtown": pdk.ViewState(
            latitude=24.854,
            longitude=67.016,
            zoom=14.0,
            pitch=15,
            bearing=10,
        ),
        "Clifton & Marine Drive Corridor": pdk.ViewState(
            latitude=24.818,
            longitude=67.032,
            zoom=13.8,
            pitch=15,
            bearing=-10,
        ),
    }


def build_zone_polygons(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Builds polygon boundaries for Origin and Destination zones."""
    zones = []
    
    # Residential origin zones
    origin_zones = metadata.get("origin_zones", {})
    for name, data in origin_zones.items():
        bbox = data.get("bbox", [])
        if len(bbox) == 4:
            min_lat, max_lat, min_lon, max_lon = bbox
            coords = [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
            zones.append({
                "zone_name": f"Origin: {name}",
                "zone_type": "Origin (Residential)",
                "polygon": coords,
                "center_lat": (min_lat + max_lat) * 0.5,
                "center_lon": (min_lon + max_lon) * 0.5,
                "color": COLOR_RESIDENTIAL_ZONE,
            })

    # Commercial destination zones
    dest_zones = metadata.get("destination_zones", {})
    for name, data in dest_zones.items():
        bbox = data.get("bbox", [])
        if len(bbox) == 4:
            min_lat, max_lat, min_lon, max_lon = bbox
            coords = [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
            zones.append({
                "zone_name": f"Destination: {name}",
                "zone_type": "Destination (Commercial / Employment Hub)",
                "polygon": coords,
                "center_lat": (min_lat + max_lat) * 0.5,
                "center_lon": (min_lon + max_lon) * 0.5,
                "color": COLOR_COMMERCIAL_ZONE,
            })

    return zones


def build_traffic_deck(
    merged_df: pd.DataFrame,
    metadata: Dict[str, Any],
    view_state: pdk.ViewState,
    display_mode: str = "Active Traffic Only",
    show_zones: bool = False,
    show_heatmap: bool = False,
) -> pdk.Deck:
    """
    Constructs a PyDeck visual Deck containing layered road segments,
    congestion overlays, OD zone bounds, and bottleneck heatmaps.
    """
    layers: List[pdk.Layer] = []

    # Filter data based on chosen display mode to maintain fluid WebGL interaction
    if display_mode == "Active Traffic Only":
        render_df = merged_df[merged_df["vehicle_count"] > 0].copy()
    elif display_mode == "Congested Corridors Only":
        render_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"].isin(["Moderate Flow (Slowing)", "Severe Bottleneck (Queue)"]))
        ].copy()
    elif display_mode == "Severe Bottlenecks Only":
        render_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"] == "Severe Bottleneck (Queue)")
        ].copy()
    elif display_mode == "Major Arteries + Active Traffic":
        # Major arteries (lanes >= 2 or speed limit >= 2) plus active roads
        major_mask = (merged_df["lanes"] >= 2) | (merged_df["speed_limit"] >= 2)
        active_mask = merged_df["vehicle_count"] > 0
        render_df = merged_df[major_mask | active_mask].copy()
    else:  # Complete Road Network (74k segments)
        render_df = merged_df.copy()

    # 1. Base layer: Origin / Destination zone bounding polygon layer
    if show_zones:
        zone_polygons = build_zone_polygons(metadata)
        if zone_polygons:
            zones_df = pd.DataFrame(zone_polygons)
            zone_layer = pdk.Layer(
                "PolygonLayer",
                data=zones_df,
                id="od-zones-polygons",
                get_polygon="polygon",
                get_fill_color=[0, 0, 0, 0],
                get_line_color="color",
                get_line_width=1.5,
                line_width_units="pixels",
                stroked=True,
                filled=False,
                extruded=False,
                pickable=False,
            )
            layers.append(zone_layer)

    # 2. Main layer: Active traffic line layer
    if not render_df.empty:
        traffic_layer = pdk.Layer(
            "LineLayer",
            data=render_df,
            id="traffic-flow-lines",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_color="color",
            get_width="line_width",
            width_units="pixels",
            width_min_pixels=1.0,
            width_max_pixels=4.0,
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 180],
        )
        layers.append(traffic_layer)

    # 3. Top layer: Optional bottleneck density heatmap layer
    if show_heatmap:
        bottleneck_df = merged_df[merged_df["congestion_status"] == "Severe Bottleneck (Queue)"]
        if not bottleneck_df.empty:
            heatmap_data = pd.DataFrame({
                "lon": bottleneck_df["mid_lon"],
                "lat": bottleneck_df["mid_lat"],
                "weight": bottleneck_df["vehicle_count"] * (bottleneck_df["density_ratio"] + 0.1),
            })
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=heatmap_data,
                id="bottleneck-heatmap",
                get_position=["lon", "lat"],
                get_weight="weight",
                radius_pixels=40,
                intensity=1.8,
                threshold=0.08,
            )
            layers.append(heatmap_layer)

    # Tooltip configuration for interactive segment inspection
    tooltip = {
        "html": """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; line-height: 1.5; color: #f9fafb; background: rgba(15, 23, 42, 0.95); padding: 10px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
            <div style="font-weight: 700; font-size: 14px; margin-bottom: 4px; color: #38bdf8;">
                Road Segment #{segment_id}
            </div>
            <div><b>Status:</b> {congestion_status}</div>
            <div><b>Active Cars:</b> <span style="color: #4ade80; font-weight: 600;">{vehicle_count}</span></div>
            <div><b>Average Speed:</b> {avg_speed_kmh} km/h <span style="color: #94a3b8;">({avg_speed} cells/s)</span></div>
            <div><b>Density Ratio:</b> {density_pct}%</div>
            <hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);" />
            <div style="color: #cbd5e1; font-size: 11px;">
                Lanes: {lanes} | Length: {length_meters} m | Limit: {speed_limit_kmh} km/h
            </div>
        </div>
        """,
        "style": {
            "backgroundColor": "transparent",
            "border": "none",
        }
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip=tooltip,
    )
    return deck
