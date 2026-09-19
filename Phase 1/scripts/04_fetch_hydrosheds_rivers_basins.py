"""
Step 3.3 - Rivers/drainage: HydroSHEDS river network (HydroRIVERS) and
drainage basins (HydroBASINS) for the Periyar / AOI area.

Downloads the continental ("as" = Asia) HydroRIVERS v10 and HydroBASINS
level-06 shapefiles directly from data.hydrosheds.org, then clips them to
the AOI bounding box (with a small buffer so up/downstream river segments
aren't cut off exactly at the AOI edge).
"""
import os
import sys
import json
import zipfile

import requests
import geopandas as gpd
from shapely.geometry import box

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

RAW_DIR = "data/raw/hydrosheds"
os.makedirs(RAW_DIR, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
BUFFER_DEG = 0.10  # buffer so rivers/basins aren't hard-cut at the AOI edge
CLIP_BBOX = [BBOX[0] - BUFFER_DEG, BBOX[1] - BUFFER_DEG, BBOX[2] + BUFFER_DEG, BBOX[3] + BUFFER_DEG]

SOURCES = {
    "rivers": {
        "url": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip",
        "zip": os.path.join(RAW_DIR, "HydroRIVERS_v10_as_shp.zip"),
        "extract_dir": os.path.join(RAW_DIR, "HydroRIVERS_v10_as_shp"),
    },
    "basins": {
        "url": "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_as_lev06_v1c.zip",
        "zip": os.path.join(RAW_DIR, "hybas_as_lev06_v1c.zip"),
        "extract_dir": os.path.join(RAW_DIR, "hybas_as_lev06_v1c"),
    },
}

print("=" * 80)
print("STEP 3.3: FETCHING HYDROSHEDS RIVER NETWORK + DRAINAGE BASINS")
print("=" * 80)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"    [skip] already downloaded -> {dest}")
        return
    print(f"    downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
        print(f"    [OK] {written / 1e6:.1f} MB (expected ~{total / 1e6:.1f} MB) -> {dest}")


def find_shp(extract_dir):
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".shp"):
                return os.path.join(root, f)
    raise RuntimeError(f"No .shp found under {extract_dir}")


results = {}
for name, cfg in SOURCES.items():
    print(f"\n[+] {name.upper()}")
    download(cfg["url"], cfg["zip"])
    if not os.path.isdir(cfg["extract_dir"]) or not os.listdir(cfg["extract_dir"]):
        print(f"    extracting -> {cfg['extract_dir']}")
        os.makedirs(cfg["extract_dir"], exist_ok=True)
        with zipfile.ZipFile(cfg["zip"]) as zf:
            zf.extractall(cfg["extract_dir"])
    else:
        print(f"    [skip] already extracted -> {cfg['extract_dir']}")
    results[name] = find_shp(cfg["extract_dir"])
    print(f"    shapefile: {results[name]}")

clip_geom = box(*CLIP_BBOX)
clip_gdf = gpd.GeoDataFrame(geometry=[clip_geom], crs="EPSG:4326")

outputs = {}

# --- Rivers: clip by bbox using a spatial filter (fast, avoids loading whole continent) ---
print("\n[+] Clipping HydroRIVERS to AOI (bbox mask read)...")
rivers_gdf = gpd.read_file(results["rivers"], bbox=tuple(CLIP_BBOX))
rivers_clip = gpd.clip(rivers_gdf, clip_gdf)
rivers_out = "data/raw/hydrosheds/periyar_rivers_clip.gpkg"
rivers_clip.to_file(rivers_out, driver="GPKG")
print(f"    [OK] {len(rivers_clip)} river segments -> {rivers_out}")
outputs["rivers"] = rivers_out

# --- Basins: small file (~13 MB) -- the shipped .sbn spatial index is
# corrupt, so read the full file (bbox prefilter fails) and clip in-memory.
print("\n[+] Clipping HydroBASINS (level 6) to AOI (full read, bbox index unusable)...")
basins_gdf = gpd.read_file(results["basins"])
basins_clip = gpd.clip(basins_gdf, clip_gdf)
basins_out = "data/raw/hydrosheds/periyar_basins_lev06_clip.gpkg"
basins_clip.to_file(basins_out, driver="GPKG")
print(f"    [OK] {len(basins_clip)} basin polygons -> {basins_out}")
outputs["basins"] = basins_out

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print("\n[+] Verifying outputs open correctly...")
summary = {
    "dataset": "HydroSHEDS v1 (HydroRIVERS v10 + HydroBASINS level-06 v1c), continent=Asia",
    "source": "https://data.hydrosheds.org (direct download)",
    "clip_bbox_wgs84": CLIP_BBOX,
    "outputs": {},
}
for name, path in outputs.items():
    gdf = gpd.read_file(path)
    print(f"    {path}: {len(gdf)} features, crs={gdf.crs}, cols={list(gdf.columns)[:6]}...")
    if len(gdf) == 0:
        print(f"    [WARN] {name} clip produced 0 features -- check bbox/CRS!")
    summary["outputs"][name] = {
        "path": path,
        "n_features": int(len(gdf)),
        "crs": str(gdf.crs),
        "bounds": [float(x) for x in gdf.total_bounds] if len(gdf) else None,
    }

if "rivers" in outputs:
    rivers_gdf_check = gpd.read_file(outputs["rivers"])
    if "ORD_STRA" in rivers_gdf_check.columns:
        major = rivers_gdf_check[rivers_gdf_check["ORD_STRA"] >= 4]
        print(f"    Major river segments (Strahler order >= 4, likely Periyar mainstem): {len(major)}")
        summary["outputs"]["rivers"]["n_major_strahler_ge4"] = int(len(major))

with open("data/raw/hydrosheds/hydrosheds_metadata.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Metadata written -> data/raw/hydrosheds/hydrosheds_metadata.json")

# ---------------------------------------------------------------------------
# Cleanup: the continental source zips/shapefiles (~500 MB) are only needed
# to produce the small AOI-clipped outputs above -- remove them afterward.
# ---------------------------------------------------------------------------
print("\n[+] Cleaning up continental source downloads (only clipped AOI outputs are needed)...")
import shutil
for cfg in SOURCES.values():
    if os.path.exists(cfg["zip"]):
        os.remove(cfg["zip"])
    if os.path.isdir(cfg["extract_dir"]):
        shutil.rmtree(cfg["extract_dir"])
print("[OK] Removed continental source files; kept AOI-clipped GeoPackages.")
print("=" * 80)
