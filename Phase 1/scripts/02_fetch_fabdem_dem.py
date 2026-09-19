"""
Step 3.1 - DEM: FABDEM (30m, forest/building-removed) for the AOI.

FABDEM (Hawker et al., 2022) corrects SRTM/Copernicus DEM for forest- and
building-height bias -- this is the fix noted for the SRTM canopy-bias
limitation flagged in Kendagannaswamy et al. Served via the GEE community
("sat-io") catalog as 1x1-degree tiles, so we mosaic the tiles covering the
AOI and clip.
"""
import os
import sys
import json

import ee
import requests
import rasterio
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)
os.makedirs("data/raw", exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]  # lon_min, lat_min, lon_max, lat_max

print("=" * 80)
print("STEP 3.1: FETCHING FABDEM 30m DEM (FOREST/BUILDING-REMOVED) FOR AOI")
print("=" * 80)

ee.Initialize(project="btp-flood")
geom = ee.Geometry.Rectangle(BBOX)

fabdem_ic = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
tiles = fabdem_ic.filterBounds(geom)
n_tiles = tiles.size().getInfo()
print(f"[+] FABDEM tiles intersecting AOI: {n_tiles}")
if n_tiles == 0:
    raise RuntimeError("No FABDEM tiles found for this AOI.")

dem = tiles.mosaic().clip(geom).rename("elevation_m")

print("[+] Requesting FABDEM GeoTIFF via getDownloadURL (30m, EPSG:4326)...")
url = dem.getDownloadURL({
    "region": geom,
    "scale": 30,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})

out_tif = "data/raw/fabdem_30m_aoi.tif"
resp = requests.get(url, timeout=300)
resp.raise_for_status()
with open(out_tif, "wb") as f:
    f.write(resp.content)
print(f"[OK] Saved -> {out_tif} ({len(resp.content) / 1e6:.2f} MB)")

# Open & sanity-check
with rasterio.open(out_tif) as src:
    arr = src.read(1, masked=True)
    meta_summary = {
        "driver": src.driver,
        "width": src.width,
        "height": src.height,
        "crs": str(src.crs),
        "bounds": list(src.bounds),
        "dtype": str(src.dtypes[0]),
        "nodata": src.nodata,
    }
    valid = arr.compressed()
    stats = {
        "min_m": float(valid.min()) if valid.size else None,
        "max_m": float(valid.max()) if valid.size else None,
        "mean_m": float(valid.mean()) if valid.size else None,
        "valid_pixels": int(valid.size),
        "total_pixels": int(arr.size),
    }

print("\n[+] Raster metadata:")
for k, v in meta_summary.items():
    print(f"    {k}: {v}")
print("\n[+] Elevation stats:")
for k, v in stats.items():
    print(f"    {k}: {v}")

if stats["min_m"] is None or stats["min_m"] < -50 or stats["max_m"] > 3000:
    print("[WARN] Elevation range looks suspicious for Kerala lowlands -- inspect visually.")
else:
    print("[OK] Elevation range plausible for Periyar lower basin / Western Ghats foothills.")

summary = {
    "dataset": "FABDEM v1-2 (Hawker et al., 2022)",
    "source": "GEE community catalog: projects/sat-io/open-datasets/FABDEM",
    "resolution_m": 30,
    "crs": meta_summary["crs"],
    "n_source_tiles": n_tiles,
    "raster_shape": [meta_summary["height"], meta_summary["width"]],
    "elevation_stats_m": stats,
    "output_file": out_tif,
}
with open("data/raw/fabdem_30m_aoi_metadata.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Metadata written -> data/raw/fabdem_30m_aoi_metadata.json")
print("=" * 80)
