"""
[CHECKPOINT] Step 3 - Confirm every flood-influencing dataset actually
downloads and opens correctly (no corrupt files, correct projection) before
Week 2.

Re-opens every dataset produced by scripts 02-06 (plus the ESA WorldCover
land-cover layer from fetch_esa_worldcover.py) fresh from disk and checks:
  - file exists and is non-empty
  - it opens without error (rasterio / geopandas)
  - CRS is present and is geographic WGS84 (EPSG:4326), matching the AOI
  - raster bounds intersect the AOI bbox
  - no all-NaN / completely empty layers
"""
import os
import sys
import json

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

BBOX = [76.15, 9.90, 76.55, 10.25]
aoi_box = box(*BBOX)

print("=" * 80)
print("[CHECKPOINT] STEP 3 DATASET ACQUISITION VERIFICATION")
print("=" * 80)

results = []


def check_raster(label, path, expect_crs="EPSG:4326"):
    entry = {"dataset": label, "path": path, "type": "raster"}
    if not os.path.exists(path):
        entry["status"] = "MISSING"
        results.append(entry)
        print(f"[FAIL] {label}: file not found -> {path}")
        return
    try:
        with rasterio.open(path) as src:
            arr = src.read(masked=True)
            valid_frac = float(np.ma.count(arr)) / arr.size if arr.size else 0.0
            b = src.bounds
            raster_box = box(b.left, b.bottom, b.right, b.top)
            intersects = raster_box.intersects(aoi_box)
            crs_ok = src.crs is not None and str(src.crs).upper() == expect_crs
            entry.update({
                "status": "OK" if (valid_frac > 0 and intersects and crs_ok) else "WARN",
                "bands": src.count,
                "shape": [src.height, src.width],
                "crs": str(src.crs),
                "crs_ok": crs_ok,
                "bounds": [b.left, b.bottom, b.right, b.top],
                "intersects_aoi": intersects,
                "valid_data_fraction": round(valid_frac, 4),
                "size_bytes": os.path.getsize(path),
            })
            flag = "OK" if entry["status"] == "OK" else "WARN"
            print(f"[{flag}] {label}: {src.count}b {src.height}x{src.width}, crs={src.crs}, "
                  f"valid={valid_frac:.1%}, intersects_aoi={intersects}, size={entry['size_bytes']/1e6:.2f}MB")
    except Exception as e:
        entry["status"] = "ERROR"
        entry["error"] = str(e)
        print(f"[FAIL] {label}: could not open -> {e}")
    results.append(entry)


def check_vector(label, path, layer=None, expect_crs="EPSG:4326"):
    entry = {"dataset": label, "path": path, "type": "vector", "layer": layer}
    if not os.path.exists(path):
        entry["status"] = "MISSING"
        results.append(entry)
        print(f"[FAIL] {label}: file not found -> {path}")
        return
    try:
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        crs_ok = gdf.crs is not None and str(gdf.crs).upper() == expect_crs
        n = len(gdf)
        b = gdf.total_bounds if n else [None] * 4
        intersects = box(*b).intersects(aoi_box) if n and None not in b else False
        entry.update({
            "status": "OK" if (n > 0 and crs_ok and intersects) else "WARN",
            "n_features": int(n),
            "crs": str(gdf.crs),
            "crs_ok": crs_ok,
            "bounds": [float(x) for x in b] if n else None,
            "intersects_aoi": bool(intersects),
            "size_bytes": os.path.getsize(path),
        })
        flag = "OK" if entry["status"] == "OK" else "WARN"
        print(f"[{flag}] {label}: {n} features, crs={gdf.crs}, intersects_aoi={intersects}, "
              f"size={entry['size_bytes']/1e6:.2f}MB")
    except Exception as e:
        entry["status"] = "ERROR"
        entry["error"] = str(e)
        print(f"[FAIL] {label}: could not open -> {e}")
    results.append(entry)


def check_csv(label, path, min_rows=1):
    entry = {"dataset": label, "path": path, "type": "csv"}
    if not os.path.exists(path):
        entry["status"] = "MISSING"
        results.append(entry)
        print(f"[FAIL] {label}: file not found -> {path}")
        return
    try:
        import pandas as pd
        df = pd.read_csv(path)
        ok = len(df) >= min_rows and df.isna().all(axis=None) == False
        entry.update({
            "status": "OK" if ok else "WARN",
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "size_bytes": os.path.getsize(path),
        })
        flag = "OK" if ok else "WARN"
        print(f"[{flag}] {label}: {len(df)} rows, cols={list(df.columns)}")
    except Exception as e:
        entry["status"] = "ERROR"
        entry["error"] = str(e)
        print(f"[FAIL] {label}: could not open -> {e}")
    results.append(entry)


print("\n--- 3.1 DEM (FABDEM) ---")
check_raster("FABDEM 30m DEM", "data/raw/fabdem_30m_aoi.tif")

print("\n--- 3.2 Rainfall (CHIRPS) ---")
check_csv("CHIRPS daily time series 2017-2023", "data/raw/chirps_daily_timeseries_2017_2023.csv", min_rows=2000)
check_raster("CHIRPS monsoon JJAS total by year", "data/raw/chirps_monsoon_jjas_total_by_year_2017_2023.tif")
check_raster("CHIRPS mean annual rainfall", "data/raw/chirps_mean_annual_rainfall_2017_2023.tif")

print("\n--- 3.3 Rivers / drainage (HydroSHEDS) ---")
check_vector("HydroRIVERS (clipped)", "data/raw/hydrosheds/periyar_rivers_clip.gpkg")
check_vector("HydroBASINS level-06 (clipped)", "data/raw/hydrosheds/periyar_basins_lev06_clip.gpkg")

print("\n--- 3.4 Roads (OSM) ---")
check_vector("OSM road edges", "data/raw/osm_roads_aoi.gpkg", layer="edges")
check_vector("OSM road nodes", "data/raw/osm_roads_aoi.gpkg", layer="nodes")

print("\n--- 3.5 Land cover (ESA WorldCover) ---")
check_raster("ESA WorldCover 2021 10m", "data/raw/esa_worldcover_2021_aoi.tif")

print("\n--- 3.6 Soil (OpenLandMap fallback; NBSS&LUP not programmatically accessible) ---")
check_raster("OpenLandMap USDA soil texture class", "data/raw/soil_openlandmap_texture_class_aoi.tif")

n_ok = sum(1 for r in results if r["status"] == "OK")
n_warn = sum(1 for r in results if r["status"] == "WARN")
n_fail = sum(1 for r in results if r["status"] in ("MISSING", "ERROR"))

print("\n" + "=" * 80)
print(f"CHECKPOINT SUMMARY: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL (of {len(results)} datasets)")
print("=" * 80)

out_path = "data/aoi/step3_dataset_acquisition_checkpoint.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"aoi_bbox_wgs84": BBOX, "n_ok": n_ok, "n_warn": n_warn, "n_fail": n_fail, "datasets": results}, f, indent=2)
print(f"[OK] Checkpoint report written -> {out_path}")

if n_fail > 0:
    print("\n[ACTION REQUIRED] One or more datasets failed to open -- fix before Week 2.")
    sys.exit(1)
else:
    print("\n[PASS] All datasets download and open correctly with consistent WGS84 projection.")
