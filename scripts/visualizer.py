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
import pydeck.bindings.json_tools as jt

# Magic bytes and header size for the flat binary road network format
MAGIC_BYTES = b"KFLW"
HEADER_SIZE_BYTES = 32
NODE_STRUCT_BYTES = 16    # float32 lat, float32 lon, int64 osm_id
SEGMENT_STRUCT_BYTES = 28 # int32 id, int32 from, int32 to, int16 lanes, int16 cells, int16 speed, int16 flags, int32 offset, float32 len

# Discrete speed mapping: 1 cell/s = 7.5 m/s = 27.0 km/h
KMH_PER_CELL_PER_SEC = 27.0

# Congestion color palette (RGBA) — High-contrast neon tones on dark basemap
COLOR_FREE_FLOW = [0, 230, 118, 245]       # Vibrant Neon Green: v >= 0.70 vmax, density < 0.25
COLOR_MODERATE = [255, 214, 0, 245]        # Bright Amber Yellow: 0.35 vmax <= v < 0.70 vmax or 0.25 <= density < 0.45
COLOR_BOTTLENECK = [255, 23, 68, 255]      # Vivid Glowing Crimson Red: 0.05 vmax < v < 0.35 vmax or 0.45 <= density < 0.65
COLOR_DEADLOCK = [255, 0, 128, 255]        # Vivid Electric Magenta/Ruby: Standstill Gridlock / Deadlock (v <= 0.05, n >= 2, rho >= 0.65)
COLOR_INACTIVE = [65, 88, 120, 130]        # Crisp steel-slate blue: Open, connected road network grid

# Zone boundary colors (RGBA) — Thin crisp outlines without blocking fills
COLOR_RESIDENTIAL_ZONE = [0, 210, 255, 180]  # Crisp Neon Cyan outline
COLOR_COMMERCIAL_ZONE = [255, 150, 0, 180]   # Warm Amber outline

# Columns required by the PyDeck LineLayer and tooltip
LINE_LAYER_COLUMNS = [
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
    "color",
    "line_width",
    "segment_id",
    "lanes",
    "congestion_status",
    "vehicle_count",
    "avg_speed_kmh",
    "density_pct",
    "length_meters",
    "speed_limit_kmh",
]


class FastDeck(pdk.Deck):
    """
    High-performance PyDeck Deck subclass that pre-serializes and caches its compact JSON representation in RAM.
    Eliminates redundant multi-second serialization during timeline scrubbing and state updates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_json: Optional[str] = None
        self._cached_json = self._serialize_compact()

    def _serialize_compact(self) -> str:
        return json.dumps(self, default=jt.default_serialize, separators=(",", ":"))

    def to_json(self) -> str:
        if self._cached_json is None:
            self._cached_json = self._serialize_compact()
        return self._cached_json


# Module-level in-memory cache for PyDeck Deck instances across timesteps
_DECK_CACHE: Dict[Tuple, FastDeck] = {}
MAX_DECK_CACHE_ENTRIES = 32


def clear_deck_cache() -> None:
    """Clears the in-memory PyDeck Deck frame cache."""
    _DECK_CACHE.clear()


def get_cached_traffic_deck(
    merged_df: pd.DataFrame,
    metadata: Dict[str, Any],
    view_state: pdk.ViewState,
    display_mode: str = "Major Arteries + Active Traffic",
    show_zones: bool = False,
    show_heatmap: bool = False,
    highlight_deadlocks: bool = True,
    scale_widths: bool = True,
    cache_key: Optional[Tuple] = None,
) -> FastDeck:
    """
    Returns a cached FastDeck from RAM if available, or constructs, pre-serializes,
    and caches a new FastDeck for instantaneous (< 1 ms) playback.
    """
    if cache_key is not None and cache_key in _DECK_CACHE:
        return _DECK_CACHE[cache_key]

    deck = build_traffic_deck(
        merged_df=merged_df,
        metadata=metadata,
        view_state=view_state,
        display_mode=display_mode,
        show_zones=show_zones,
        show_heatmap=show_heatmap,
        highlight_deadlocks=highlight_deadlocks,
        scale_widths=scale_widths,
    )

    if cache_key is not None:
        if len(_DECK_CACHE) >= MAX_DECK_CACHE_ENTRIES:
            _DECK_CACHE.pop(next(iter(_DECK_CACHE)))
        _DECK_CACHE[cache_key] = deck
    return deck


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
        "speed_limit_kmh": np.round(segments["speed_limit"] * KMH_PER_CELL_PER_SEC, 1).astype(np.float64),
        "length_meters": np.round(segments["length_meters"].astype(np.float64), 1).astype(np.float64),
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

    # Derived physics indicators sanitized to 64-bit precision rounded to 1 decimal place
    merged["avg_speed_kmh"] = np.round((merged["avg_speed"] * KMH_PER_CELL_PER_SEC).astype(np.float64), 1)
    merged["density_pct"] = np.round((merged["density_ratio"] * 100.0).astype(np.float64), 1)
    merged["length_meters"] = np.round(merged["length_meters"].astype(np.float64), 1)
    merged["speed_limit_kmh"] = np.round(merged["speed_limit_kmh"].astype(np.float64), 1)

    v_max = np.maximum(merged["speed_limit"].values, 1)
    speed_ratio = np.clip(merged["avg_speed"].values / v_max, 0.0, 1.0)
    density = merged["density_ratio"].values
    cars = merged["vehicle_count"].values

    # 4-Tier Traffic Physics & Deadlock Detection:
    # 1. STANDSTILL GRIDLOCK / DEADLOCK (Magenta):
    #    Multiple vehicles (cars >= 2) jammed at high density (density >= 0.65) and near zero speed (speed <= 0.05 vmax)
    is_gridlock = (cars >= 2) & (density >= 0.65) & (speed_ratio <= 0.05)

    # 2. Severe Bottleneck / Queue (Crimson Red):
    #    Requires multiple vehicles (cars >= 2) AND elevated density (density >= 0.35) along with low speed (speed_ratio < 0.35).
    #    A single vehicle can NEVER form a bottleneck.
    is_bottleneck = (cars >= 2) & (~is_gridlock) & (
        (speed_ratio < 0.35) & (density >= 0.35)
    )

    # 3. Moderate Flow / Slowing (Bright Amber):
    #    - Single vehicle stopped or yielding at intersection (v <= 0.05 vmax)
    #    - Multiple vehicles with moderate speed (0.35 <= speed_ratio < 0.70) or elevated density (density >= 0.25)
    #    - Multiple vehicles with low speed but low density (non-queue slowing, density < 0.35)
    single_stopped = (cars == 1) & (speed_ratio <= 0.05)
    multi_moderate = (cars >= 2) & (~is_gridlock) & (~is_bottleneck) & (
        ((speed_ratio >= 0.35) & (speed_ratio < 0.70)) |
        (density >= 0.25) |
        (speed_ratio < 0.35)
    )
    is_moderate = single_stopped | multi_moderate

    # 4. Free Flow (Neon Green):
    #    - Single moving vehicle (cars == 1 and speed_ratio > 0.05)
    #    - Multiple vehicles at high speed (speed_ratio >= 0.70) and low density (density < 0.25)
    #    - Any remaining active vehicles not classified above
    is_free_flow = (cars > 0) & (~is_gridlock) & (~is_bottleneck) & (~is_moderate)

    status_list = np.full(len(merged), "Empty Road", dtype=object)
    status_list[is_free_flow] = "Free Flow (Normal)"
    status_list[is_moderate] = "Moderate Flow (Slowing)"
    status_list[is_bottleneck] = "Severe Bottleneck (Queue)"
    status_list[is_gridlock] = "Standstill Gridlock (Deadlock)"
    merged["congestion_status"] = status_list

    # Assign RGBA color tuples as Python lists (strictly serialized via .tolist() for PyDeck JSON)
    colors = np.empty((len(merged), 4), dtype=np.int32)
    colors[:] = COLOR_INACTIVE
    colors[is_free_flow] = COLOR_FREE_FLOW
    colors[is_moderate] = COLOR_MODERATE
    colors[is_bottleneck] = COLOR_BOTTLENECK
    colors[is_gridlock] = COLOR_DEADLOCK
    merged["color"] = colors.tolist()

    # Sleek line widths: sharp and crisp to prevent black blobs on high density
    lanes = merged["lanes"].values
    widths = np.full(len(merged), 1.0, dtype=np.float32)
    active_mask = cars > 0
    widths[active_mask] = np.clip(1.5 + lanes[active_mask] * 0.35 + density[active_mask] * 2.0, 1.8, 4.0)
    widths[is_gridlock] = np.clip(widths[is_gridlock] + 1.2, 2.8, 5.0)
    merged["line_width"] = widths

    return merged


def get_viewport_presets(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, pdk.ViewState]:
    """Provides curated geographic camera viewpoints across Karachi key corridors."""
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
            zoom=13.6,
            pitch=25,
            bearing=-25,
        ),
        "Saddar Commercial Core": pdk.ViewState(
            latitude=24.854,
            longitude=67.016,
            zoom=14.2,
            pitch=20,
            bearing=10,
        ),
        "University Road Arterial": pdk.ViewState(
            latitude=24.925,
            longitude=67.112,
            zoom=13.6,
            pitch=20,
            bearing=20,
        ),
        "Clifton Marine Drive": pdk.ViewState(
            latitude=24.818,
            longitude=67.032,
            zoom=14.0,
            pitch=20,
            bearing=-10,
        ),
        "Superhighway M-9": pdk.ViewState(
            latitude=24.945,
            longitude=67.125,
            zoom=13.2,
            pitch=25,
            bearing=30,
        ),
        "Port Qasim Freight Corridor": pdk.ViewState(
            latitude=24.845,
            longitude=67.160,
            zoom=13.2,
            pitch=20,
            bearing=-20,
        ),
        # Backward-compatibility aliases
        "Saddar & I.I. Chundrigar Downtown": pdk.ViewState(
            latitude=24.854,
            longitude=67.016,
            zoom=14.2,
            pitch=20,
            bearing=10,
        ),
        "Clifton & Marine Drive Corridor": pdk.ViewState(
            latitude=24.818,
            longitude=67.032,
            zoom=14.0,
            pitch=20,
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
    display_mode: str = "Major Arteries + Active Traffic",
    show_zones: bool = False,
    show_heatmap: bool = False,
    highlight_deadlocks: bool = True,
    scale_widths: bool = True,
) -> pdk.Deck:
    """
    Constructs a PyDeck visual Deck containing layered road segments,
    congestion overlays, OD zone bounds, deadlock highlighters, and bottleneck heatmaps.
    """
    layers: List[pdk.Layer] = []

    # Filter data based on chosen display mode to maintain fluid WebGL interaction
    if display_mode == "Active Traffic Only":
        render_df = merged_df[merged_df["vehicle_count"] > 0].copy()
    elif display_mode == "Congested Corridors Only":
        render_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"].isin([
                "Moderate Flow (Slowing)",
                "Severe Bottleneck (Queue)",
                "Standstill Gridlock (Deadlock)",
            ]))
        ].copy()
    elif display_mode == "Severe Bottlenecks Only":
        render_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"].isin([
                "Severe Bottleneck (Queue)",
                "Standstill Gridlock (Deadlock)",
            ]))
        ].copy()
    elif display_mode == "Standstill Deadlocks Only":
        render_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"] == "Standstill Gridlock (Deadlock)")
        ].copy()
    elif display_mode == "Major Arteries + Active Traffic":
        # Major arteries (lanes >= 2 or speed limit >= 2) plus active roads
        major_mask = (merged_df["lanes"] >= 2) | (merged_df["speed_limit"] >= 2)
        active_mask = merged_df["vehicle_count"] > 0
        render_df = merged_df[major_mask | active_mask].copy()
    else:  # Complete Road Network (74k segments)
        render_df = merged_df.copy()

    # Ensure inactive open roads render first and active traffic renders on top
    if not render_df.empty and "vehicle_count" in render_df.columns:
        render_df = render_df.sort_values(by="vehicle_count", ascending=True)

    # Optional uniform width override
    if not scale_widths and not render_df.empty:
        render_df["line_width"] = 1.8

    # Column pruning: Drop all unused attributes before PyDeck serialization
    if not render_df.empty:
        active_cols = [c for c in LINE_LAYER_COLUMNS if c in render_df.columns]
        render_df = render_df[active_cols].copy()

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
            width_max_pixels=5.0,
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 180],
        )
        layers.append(traffic_layer)

    # 3. Deadlock Highlighter layer (Standstill Gridlock in Vivid Magenta)
    if highlight_deadlocks:
        deadlock_df = merged_df[
            (merged_df["vehicle_count"] > 0) &
            (merged_df["congestion_status"] == "Standstill Gridlock (Deadlock)")
        ]
        if not deadlock_df.empty:
            deadlock_cols = [c for c in LINE_LAYER_COLUMNS if c in deadlock_df.columns]
            deadlock_df = deadlock_df[deadlock_cols].copy()
            deadlock_layer = pdk.Layer(
                "LineLayer",
                data=deadlock_df,
                id="deadlock-highlighter-lines",
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_color=COLOR_DEADLOCK,
                get_width=5.5,
                width_units="pixels",
                width_min_pixels=3.0,
                width_max_pixels=8.0,
                pickable=True,
                auto_highlight=True,
                highlight_color=[255, 255, 255, 230],
            )
            layers.append(deadlock_layer)

    # 4. Top layer: Optional bottleneck and deadlock density heatmap layer
    if show_heatmap:
        bottleneck_df = merged_df[
            merged_df["congestion_status"].isin(["Severe Bottleneck (Queue)", "Standstill Gridlock (Deadlock)"])
        ]
        if not bottleneck_df.empty:
            heatmap_data = pd.DataFrame({
                "lon": bottleneck_df["mid_lon"],
                "lat": bottleneck_df["mid_lat"],
                "weight": bottleneck_df["vehicle_count"] * (bottleneck_df["density_ratio"] + 0.15),
            })
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=heatmap_data,
                id="bottleneck-heatmap",
                get_position=["lon", "lat"],
                get_weight="weight",
                radius_pixels=42,
                intensity=2.0,
                threshold=0.08,
            )
            layers.append(heatmap_layer)

    # Tooltip configuration for interactive segment inspection (Compact HUD)
    tooltip = {
        "html": """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; font-size: 11px; line-height: 1.4; color: #f8fafc; background: rgba(16, 23, 38, 0.94); padding: 7px 11px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 16px rgba(0,0,0,0.5); backdrop-filter: blur(8px); max-width: 230px; box-sizing: border-box; overflow: hidden; contain: layout style; word-break: break-word; pointer-events: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 3px;">
                <span style="font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 700; font-size: 11px; color: #38bdf8;">SEG #{segment_id}</span>
                <span style="font-size: 10px; padding: 1px 4px; border-radius: 3px; background: rgba(255,255,255,0.08); color: #cbd5e1;">Lanes: {lanes}</span>
            </div>
            <div style="margin-bottom: 2px;"><b>Status:</b> {congestion_status}</div>
            <div style="margin-bottom: 2px;"><b>Traffic:</b> <span style="font-family: 'JetBrains Mono', Consolas, monospace; color: #4ade80;">{vehicle_count} veh</span> • <span style="font-family: 'JetBrains Mono', Consolas, monospace;">{avg_speed_kmh} km/h</span></div>
            <div style="margin-bottom: 2px;"><b>Density:</b> <span style="font-family: 'JetBrains Mono', Consolas, monospace;">{density_pct}%</span></div>
            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 3px; color: #94a3b8; font-size: 10px; display: flex; justify-content: space-between;">
                <span>Len: {length_meters}m</span>
                <span>Limit: {speed_limit_kmh} km/h</span>
            </div>
        </div>
        """,
        "style": {
            "backgroundColor": "transparent",
            "border": "none",
            "transform": "translate(12px, 12px)",
            "zIndex": "99999",
            "maxWidth": "240px",
            "overflow": "hidden",
            "pointerEvents": "none",
        }
    }

    deck = FastDeck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip=tooltip,
    )
    return deck


# Backward compatibility alias
build_deck = build_traffic_deck
