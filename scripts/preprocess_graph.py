#!/usr/bin/env python3
"""
KarachiFlow — Road Network Preprocessor
========================================
Extracts the Karachi drivable road network via OSMnx, discretizes lanes
into uniform 7.5m cells, tags origin/destination zones, and serializes
into a compact flat binary format for C++ ingestion.
"""

import os
import sys
import argparse
import json
import struct
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("KarachiFlow.DataPrep")

CELL_LENGTH_METERS = 7.5
MAGIC_BYTES = b"KFLW"
BINARY_VERSION = 1

# Bounding boxes (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)
BBOX_DEV = (67.010, 24.840, 67.165, 24.935)   # Saddar <-> Shahrah-e-Faisal <-> Airport / Gulshan corridor
BBOX_FULL = (66.850, 24.750, 67.350, 25.100)  # Complete metropolitan Karachi

# Origin (Residential) and Destination (Commercial / Hub) Zones: [min_lat, max_lat, min_lon, max_lon]
ZONES_RESIDENTIAL = {
    "Gulshan-e-Iqbal":    [24.900, 24.945, 67.070, 67.125],
    "Gulistan-e-Johar":   [24.905, 24.950, 67.125, 67.175],
    "North Nazimabad":    [24.920, 24.965, 67.020, 67.065],
    "Korangi":            [24.815, 24.860, 67.100, 67.170],
}

ZONES_COMMERCIAL = {
    "I.I. Chundrigar":    [24.845, 24.860, 66.995, 67.015],
    "Saddar":             [24.850, 24.870, 67.010, 67.035],
    "Clifton":            [24.805, 24.840, 67.015, 67.050],
    "SITE":               [24.890, 24.930, 66.980, 67.020],
    "Airport":            [24.895, 24.920, 67.150, 67.180],
}

# Highway defaults: (lanes, speed_limit in cells/s [1 cell/s = 27 km/h])
HIGHWAY_DEFAULTS = {
    "motorway":        (3, 4),
    "motorway_link":   (2, 3),
    "trunk":           (3, 3),
    "trunk_link":      (2, 2),
    "primary":         (3, 2),
    "primary_link":    (2, 2),
    "secondary":       (2, 2),
    "secondary_link":  (1, 1),
    "tertiary":        (2, 1),
    "tertiary_link":   (1, 1),
    "residential":     (1, 1),
    "living_street":   (1, 1),
    "unclassified":    (1, 1),
    "service":         (1, 1),
}


def extract_primary_tag(value: Any) -> str:
    """Extract string tag if OSMnx stored it as a list."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def parse_lanes(raw_lanes: Any, highway_type: str) -> int:
    """Robustly parse lanes attribute with realistic taxonomy fallback."""
    highway_tag = extract_primary_tag(highway_type).lower()
    default_lanes, _ = HIGHWAY_DEFAULTS.get(highway_tag, (1, 1))

    if raw_lanes is None:
        return default_lanes

    if isinstance(raw_lanes, (int, np.integer)):
        return max(1, min(8, int(raw_lanes)))

    if isinstance(raw_lanes, list):
        if not raw_lanes:
            return default_lanes
        return parse_lanes(raw_lanes[0], highway_type)

    if isinstance(raw_lanes, str):
        raw_str = raw_lanes.split(";")[0].split(",")[0].strip()
        try:
            return max(1, min(8, int(float(raw_str))))
        except ValueError:
            return default_lanes

    return default_lanes


def parse_speed_limit(raw_speed: Any, highway_type: str) -> int:
    """
    Parse speed limit into discrete cells/second (1..4).
    1 cell/s = 27 km/h, 2 = 54 km/h, 3 = 81 km/h, 4 = 108 km/h.
    """
    highway_tag = extract_primary_tag(highway_type).lower()
    _, default_speed = HIGHWAY_DEFAULTS.get(highway_tag, (1, 1))

    if raw_speed is None:
        return default_speed

    if isinstance(raw_speed, list):
        if not raw_speed:
            return default_speed
        return parse_speed_limit(raw_speed[0], highway_type)

    speed_kmh: Optional[float] = None
    if isinstance(raw_speed, (int, float, np.number)):
        speed_kmh = float(raw_speed)
    elif isinstance(raw_speed, str):
        s = raw_speed.lower().replace("km/h", "").replace("kph", "").replace("mph", "").strip()
        try:
            val = float(s)
            if "mph" in raw_speed.lower():
                val *= 1.60934
            speed_kmh = val
        except ValueError:
            pass

    if speed_kmh is not None and speed_kmh > 0:
        discrete_speed = int(round(speed_kmh / 27.0))
        return max(1, min(4, discrete_speed))

    return default_speed


def discretize_cells(length_meters: float) -> int:
    """
    Discretize road length into 7.5m cells.
    Strictly clamps N_cells >= 1 to prevent zero-cell degeneracies.
    """
    if length_meters <= 0.0:
        return 1
    cells = int(round(length_meters / CELL_LENGTH_METERS))
    return max(1, cells)


def fetch_osm_graph(bbox: Tuple[float, float, float, float], cache_file: str, use_cache: bool = True) -> nx.MultiDiGraph:
    """Fetch drivable road network via OSMnx with local GraphML disk caching."""
    import osmnx as ox

    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading cached graph from: {cache_file}")
        t0 = time.time()
        G = ox.load_graphml(cache_file)
        logger.info(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in {time.time() - t0:.2f}s")
        return G

    logger.info(f"Querying Overpass API for bbox {bbox} (network_type='drive')...")
    t0 = time.time()

    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(os.path.dirname(os.path.abspath(cache_file)), "osmnx")
    ox.settings.timeout = 180

    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )
    logger.info(f"Fetched graph in {time.time() - t0:.2f}s ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    if use_cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        logger.info(f"Saving graph cache to: {cache_file}")
        ox.save_graphml(G, cache_file)

    return G


def generate_synthetic_karachi_graph() -> nx.MultiDiGraph:
    """Generates a deterministic synthetic graph of Karachi's major arterial corridors."""
    logger.info("Generating deterministic synthetic Karachi arterial network...")
    G = nx.MultiDiGraph()

    corridors = [
        ("Shahrah-e-Faisal", (24.855, 67.025), (24.905, 24.905, 67.165), "trunk", 4, 3, 40),
        ("University Road", (24.865, 67.035), (24.935, 24.935, 67.140), "primary", 3, 2, 30),
        ("M.A. Jinnah Road", (24.848, 67.000), (24.885, 24.885, 67.035), "primary", 3, 2, 20),
        ("Rashid Minhas Road", (24.890, 67.125), (24.940, 24.940, 67.095), "primary", 3, 2, 25),
        ("Korangi Road", (24.855, 67.050), (24.830, 24.830, 67.140), "secondary", 2, 2, 25),
    ]

    node_counter = 1000000

    for name, start_pt, end_pt, hwy, lanes, speed, num_segs in corridors:
        start_lat, start_lon = start_pt[0], start_pt[1]
        end_lat, end_lon = end_pt[0], end_pt[2] if len(end_pt) == 3 else end_pt[1]

        lats = np.linspace(start_lat, end_lat, num_segs + 1)
        lons = np.linspace(start_lon, end_lon, num_segs + 1)

        corridor_nodes = []
        for i in range(num_segs + 1):
            nid = node_counter
            node_counter += 1
            G.add_node(nid, y=float(lats[i]), x=float(lons[i]))
            corridor_nodes.append(nid)

        for i in range(num_segs):
            u, v = corridor_nodes[i], corridor_nodes[i + 1]
            dist_m = float(np.hypot((lats[i+1] - lats[i]) * 111000, (lons[i+1] - lons[i]) * 102000))
            dist_m = max(15.0, dist_m)

            G.add_edge(u, v, key=0, length=dist_m, lanes=lanes, maxspeed=speed * 27, highway=hwy, name=name, oneway=True)
            G.add_edge(v, u, key=0, length=dist_m, lanes=lanes, maxspeed=speed * 27, highway=hwy, name=name, oneway=True)

    nodes_list = list(G.nodes())
    for i in range(0, len(nodes_list) - 10, 8):
        u = nodes_list[i]
        v = nodes_list[i + 8]
        lat_u, lon_u = G.nodes[u]['y'], G.nodes[u]['x']
        lat_v, lon_v = G.nodes[v]['y'], G.nodes[v]['x']
        dist_m = float(np.hypot((lat_v - lat_u) * 111000, (lon_v - lon_u) * 102000))
        dist_m = max(20.0, dist_m)
        G.add_edge(u, v, key=0, length=dist_m, lanes=2, maxspeed=54, highway="secondary", oneway=False)
        G.add_edge(v, u, key=0, length=dist_m, lanes=2, maxspeed=54, highway="secondary", oneway=False)

    logger.info(f"Generated synthetic network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def normalize_and_discretize_network(G: nx.MultiDiGraph) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transforms the MultiDiGraph into flat contiguous arrays:
    - Re-indexes nodes to [0, N-1]
    - Discretizes edges into 7.5m cells
    - Computes cell offsets and CSR topology
    - Tags residential origin and commercial destination zones
    """
    logger.info("Normalizing network topology and discretizing road segments...")

    raw_nodes = sorted(G.nodes(data=True), key=lambda x: int(x[0]))
    node_count = len(raw_nodes)

    osm_to_idx: Dict[int, int] = {}
    nodes_table: List[Dict[str, Any]] = []

    for idx, (osm_id, data) in enumerate(raw_nodes):
        osm_to_idx[int(osm_id)] = idx
        lat = float(data.get("y", 0.0))
        lon = float(data.get("x", 0.0))
        nodes_table.append({
            "node_idx": idx,
            "osm_id": int(osm_id),
            "lat": lat,
            "lon": lon,
        })

    raw_segments: List[Dict[str, Any]] = []

    for u, v, k, data in G.edges(keys=True, data=True):
        u_int, v_int = int(u), int(v)
        if u_int not in osm_to_idx or v_int not in osm_to_idx:
            continue

        from_node = osm_to_idx[u_int]
        to_node = osm_to_idx[v_int]

        length_m = float(data.get("length", 7.5))
        highway_type = data.get("highway", "unclassified")
        lanes = parse_lanes(data.get("lanes"), highway_type)
        speed_limit = parse_speed_limit(data.get("maxspeed"), highway_type)
        cells_per_lane = discretize_cells(length_m)

        oneway = bool(data.get("oneway", False))
        flags = 1 if oneway else 0

        raw_segments.append({
            "from_node": from_node,
            "to_node": to_node,
            "lanes": lanes,
            "cells_per_lane": cells_per_lane,
            "speed_limit": speed_limit,
            "flags": flags,
            "length_meters": length_m,
            "osm_u": u_int,
            "osm_v": v_int,
        })

    raw_segments.sort(key=lambda s: (s["from_node"], s["to_node"]))
    segment_count = len(raw_segments)

    segments_table: List[Dict[str, Any]] = []
    current_cell_offset = 0

    for seg_id, seg in enumerate(raw_segments):
        seg["segment_id"] = seg_id
        seg["cell_offset"] = current_cell_offset
        total_seg_cells = seg["lanes"] * seg["cells_per_lane"]
        current_cell_offset += total_seg_cells
        segments_table.append(seg)

    total_cells = current_cell_offset

    logger.info("Constructing Compressed Sparse Row (CSR) adjacency topology...")
    node_offsets = [0] * (node_count + 1)
    outgoing_segments: List[int] = []

    for seg in segments_table:
        node_offsets[seg["from_node"] + 1] += 1

    for i in range(node_count):
        node_offsets[i + 1] += node_offsets[i]

    for seg in segments_table:
        outgoing_segments.append(seg["segment_id"])

    for u in range(node_count):
        start = node_offsets[u]
        end = node_offsets[u + 1]
        for idx in range(start, end):
            assert segments_table[outgoing_segments[idx]]["from_node"] == u, "CSR invariant violation"

    logger.info("Tagging Origin & Destination zones based on geographic coordinates...")
    origin_zones: Dict[str, Any] = {}
    for zone_name, (min_lat, max_lat, min_lon, max_lon) in ZONES_RESIDENTIAL.items():
        matching_nodes = [
            n["node_idx"] for n in nodes_table
            if min_lat <= n["lat"] <= max_lat and min_lon <= n["lon"] <= max_lon
        ]
        origin_zones[zone_name] = {
            "bbox": [min_lat, max_lat, min_lon, max_lon],
            "node_count": len(matching_nodes),
            "node_ids": matching_nodes,
        }
        logger.info(f"  Residential Zone '{zone_name}': {len(matching_nodes)} nodes")

    destination_zones: Dict[str, Any] = {}
    for zone_name, (min_lat, max_lat, min_lon, max_lon) in ZONES_COMMERCIAL.items():
        matching_nodes = [
            n["node_idx"] for n in nodes_table
            if min_lat <= n["lat"] <= max_lat and min_lon <= n["lon"] <= max_lon
        ]
        destination_zones[zone_name] = {
            "bbox": [min_lat, max_lat, min_lon, max_lon],
            "node_count": len(matching_nodes),
            "node_ids": matching_nodes,
        }
        logger.info(f"  Commercial Zone '{zone_name}': {len(matching_nodes)} nodes")

    lats = [n["lat"] for n in nodes_table]
    lons = [n["lon"] for n in nodes_table]
    metadata = {
        "format": "KarachiFlow Flat Binary",
        "version": BINARY_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell_length_meters": CELL_LENGTH_METERS,
        "counts": {
            "node_count": node_count,
            "segment_count": segment_count,
            "total_cells": total_cells,
            "total_outgoing_adj": len(outgoing_segments),
        },
        "bounding_box": {
            "min_lat": min(lats) if lats else 0.0,
            "max_lat": max(lats) if lats else 0.0,
            "min_lon": min(lons) if lons else 0.0,
            "max_lon": max(lons) if lons else 0.0,
        },
        "lane_distribution": {int(k): int(v) for k, v in zip(*np.unique([s["lanes"] for s in segments_table], return_counts=True))},
        "speed_distribution": {int(k): int(v) for k, v in zip(*np.unique([s["speed_limit"] for s in segments_table], return_counts=True))},
        "origin_zones": origin_zones,
        "destination_zones": destination_zones,
    }

    topology = {
        "node_offsets": node_offsets,
        "outgoing_segments": outgoing_segments,
    }

    return metadata, nodes_table, segments_table, topology


def export_flat_binary(
    bin_path: str,
    metadata: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    topology: Dict[str, Any]
) -> int:
    """Serializes the network into the high-performance v1 flat binary layout."""
    logger.info(f"Serializing network into flat binary: {bin_path}")
    t0 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)

    node_count = len(nodes)
    segment_count = len(segments)
    total_cells = metadata["counts"]["total_cells"]
    total_adj = len(topology["outgoing_segments"])

    with open(bin_path, "wb") as f:
        header_data = struct.pack(
            "<4sIIIIIII",
            MAGIC_BYTES,
            BINARY_VERSION,
            node_count,
            segment_count,
            total_cells,
            total_adj,
            0,
            0
        )
        f.write(header_data)

        for n in nodes:
            node_bytes = struct.pack("<ffq", n["lat"], n["lon"], n["osm_id"])
            f.write(node_bytes)

        for s in segments:
            seg_bytes = struct.pack(
                "<iiihhhhif",
                s["segment_id"],
                s["from_node"],
                s["to_node"],
                s["lanes"],
                s["cells_per_lane"],
                s["speed_limit"],
                s["flags"],
                s["cell_offset"],
                s["length_meters"]
            )
            f.write(seg_bytes)

        offsets = topology["node_offsets"]
        f.write(struct.pack(f"<{len(offsets)}i", *offsets))

        adj = topology["outgoing_segments"]
        f.write(struct.pack(f"<{len(adj)}i", *adj))

    file_size_bytes = os.path.getsize(bin_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info(f"Binary export completed in {time.time() - t0:.3f}s. Size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    return file_size_bytes


def main():
    parser = argparse.ArgumentParser(description="KarachiFlow OSMnx Road Network Preprocessor")
    parser.add_argument(
        "--mode",
        choices=["dev", "full", "synthetic"],
        default="dev",
        help="Preprocessing mode: dev (corridor sub-bbox), full (metro Karachi), or synthetic (offline grid)"
    )
    parser.add_argument(
        "--output-bin",
        default="data/karachi_network.bin",
        help="Output binary file path (default: data/karachi_network.bin)"
    )
    parser.add_argument(
        "--output-meta",
        default="data/karachi_network_metadata.json",
        help="Output metadata JSON path (default: data/karachi_network_metadata.json)"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory to store raw GraphML caches (default: data/cache)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass GraphML disk caching and force re-query"
    )

    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    logger.info(f"=== KarachiFlow Network Preprocessor (Mode: {args.mode}) ===")
    t_start = time.time()

    if args.mode == "synthetic":
        G = generate_synthetic_karachi_graph()
    elif args.mode == "dev":
        cache_file = os.path.join(args.cache_dir, "karachi_dev_raw.graphml")
        G = fetch_osm_graph(BBOX_DEV, cache_file, use_cache=not args.no_cache)
    elif args.mode == "full":
        cache_file = os.path.join(args.cache_dir, "karachi_full_raw.graphml")
        G = fetch_osm_graph(BBOX_FULL, cache_file, use_cache=not args.no_cache)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    metadata, nodes, segments, topology = normalize_and_discretize_network(G)
    metadata["mode"] = args.mode

    bin_size = export_flat_binary(args.output_bin, metadata, nodes, segments, topology)

    logger.info(f"Writing metadata JSON to: {args.output_meta}")
    with open(args.output_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t_start
    logger.info("==================================================================")
    logger.info(f"Preprocessing finished successfully in {total_time:.2f}s!")
    logger.info(f"  Nodes:        {metadata['counts']['node_count']:,}")
    logger.info(f"  Segments:     {metadata['counts']['segment_count']:,}")
    logger.info(f"  Total Cells:  {metadata['counts']['total_cells']:,}")
    logger.info(f"  Binary Size:  {bin_size / (1024 * 1024):.2f} MB")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
