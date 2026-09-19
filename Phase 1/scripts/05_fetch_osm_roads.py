"""
Step 3.4 - Roads: OSM road network for the AOI, via OSMnx / Overpass API.

Downloads the drivable ("drive") road network for the AOI bounding box and
saves the edges (road segments) and nodes (intersections) as GeoPackage
layers -- used later for the distance-to-road feature (Step 7.3 of the plan).
"""
import os
import sys
import json

import osmnx as ox

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)
os.makedirs("data/raw", exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]  # lon_min, lat_min, lon_max, lat_max
# osmnx graph_from_bbox expects (left, bottom, right, top) = (west, south, east, north)
OX_BBOX = (BBOX[0], BBOX[1], BBOX[2], BBOX[3])

print("=" * 80)
print("STEP 3.4: FETCHING OSM ROAD NETWORK FOR AOI (via Overpass API)")
print("=" * 80)

ox.settings.timeout = 180
ox.settings.log_console = False

print(f"[+] Requesting 'drive' network for bbox {OX_BBOX} ...")
G = ox.graph_from_bbox(OX_BBOX, network_type="drive", simplify=True, retain_all=False)
print(f"[OK] Graph downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

out_gpkg = "data/raw/osm_roads_aoi.gpkg"
edges_gdf.reset_index().to_file(out_gpkg, layer="edges", driver="GPKG")
nodes_gdf.reset_index().to_file(out_gpkg, layer="nodes", driver="GPKG")
print(f"[OK] Saved edges + nodes -> {out_gpkg}")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print("\n[+] Verifying output opens correctly...")
import geopandas as gpd

edges_check = gpd.read_file(out_gpkg, layer="edges")
nodes_check = gpd.read_file(out_gpkg, layer="nodes")

edges_utm = edges_check.to_crs(epsg=32643)
total_length_km = edges_utm.geometry.length.sum() / 1000.0

print(f"    edges layer: {len(edges_check)} features, crs={edges_check.crs}")
print(f"    nodes layer: {len(nodes_check)} features, crs={nodes_check.crs}")
print(f"    total road network length (UTM 43N): {total_length_km:.1f} km")
print(f"    bounds: {list(edges_check.total_bounds)}")

if len(edges_check) == 0:
    print("[WARN] 0 road edges downloaded -- check Overpass connectivity/bbox.")
else:
    print("[OK] Road network non-empty and geometry is valid.")

highway_counts = edges_check["highway"].astype(str).value_counts().to_dict() if "highway" in edges_check.columns else {}

summary = {
    "dataset": "OpenStreetMap road network (via OSMnx / Overpass API)",
    "network_type": "drive",
    "bbox_wgs84": BBOX,
    "n_nodes": int(len(nodes_check)),
    "n_edges": int(len(edges_check)),
    "total_length_km": float(total_length_km),
    "highway_type_counts": {str(k): int(v) for k, v in highway_counts.items()},
    "output_file": out_gpkg,
}
with open("data/raw/osm_roads_metadata.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Metadata written -> data/raw/osm_roads_metadata.json")
print("=" * 80)
