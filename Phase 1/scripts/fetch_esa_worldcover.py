import os
import sys
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import numpy as np

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

os.makedirs("data/raw", exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]

print("=" * 80)
print("FETCHING & PROCESSING ESA WORLDCOVER 2021 10m LULC FOR AOI")
print("=" * 80)

# 1. Query STAC API for ESA WorldCover Tile
print("[+] Querying Planetary Computer STAC for ESA WorldCover 2021 tile...")
stac_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
body = {'collections': ['esa-worldcover'], 'bbox': BBOX}
stac_res = requests.post(stac_url, json=body).json()

if not stac_res.get('features'):
    raise RuntimeError("No ESA WorldCover tile found for the given BBOX.")

item = stac_res['features'][0]
tile_id = item['id']
raw_href = item['assets']['map']['href']
print(f"    • STAC Tile ID: {tile_id}")

# 2. Get SAS Auth Token
token_url = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-worldcover'
token_res = requests.get(token_url).json()
token = token_res['token']
signed_href = f"{raw_href}?{token}"

# Target SAR dimensions from dry dB array
dry_db_path = "data/raw/sentinel1_2018_dry_db.npy"
if os.path.exists(dry_db_path):
    dry_db = np.load(dry_db_path)
    target_shape = dry_db.shape
else:
    target_shape = (2900, 3800) # lat x lon (~10m)

print(f"[+] Resampling ESA WorldCover grid to match SAR resolution: {target_shape[1]} x {target_shape[0]}...")

# 3. Read and Clip Window
with rasterio.open(signed_href) as src:
    window = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], src.transform)
    wc_data = src.read(1, window=window, out_shape=target_shape, resampling=Resampling.nearest)
    
    # Also save raw clipped GeoTIFF
    win_transform = src.window_transform(window)
    out_tif_path = "data/raw/esa_worldcover_2021_aoi.tif"
    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": target_shape[0],
        "width": target_shape[1],
        "transform": win_transform,
        "crs": src.crs
    })
    with rasterio.open(out_tif_path, "w", **meta) as dst:
        dst.write(wc_data, 1)

print(f"[OK] Saved ESA WorldCover 10m GeoTIFF -> {out_tif_path}")

# 4. Save NPY Arrays
wc_npy_path = "data/raw/esa_worldcover_2021_resampled.npy"
builtup_mask_path = "data/raw/builtup_mask_10m.npy"

np.save(wc_npy_path, wc_data)
builtup_mask = (wc_data == 50)
np.save(builtup_mask_path, builtup_mask)

total_pixels = wc_data.size
builtup_pixels = np.sum(builtup_mask)
builtup_pct = (builtup_pixels / total_pixels) * 100.0
builtup_sqkm = (builtup_pct / 100.0) * 1696.64

print("\n" + "=" * 80)
print("              ESA WORLDCOVER 2021 10m LULC BREAKDOWN (AOI)")
print("=" * 80)
classes_info = {
    10: "Tree Cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up (Settlement / Urban)",
    60: "Bare / Sparse Vegetation",
    80: "Permanent Water Bodies",
    90: "Herbaceous Wetland",
    95: "Mangroves"
}

for c_val, c_name in classes_info.items():
    c_count = np.sum(wc_data == c_val)
    if c_count > 0:
        c_pct = (c_count / total_pixels) * 100.0
        c_sqkm = (c_pct / 100.0) * 1696.64
        print(f"  • Class {c_val:2d} ({c_name:30s}): {c_sqkm:7.2f} sq km ({c_pct:5.2f}%)")

print("=" * 80)
print(f"[SUMMARY] Total Built-Up Land (Class 50): {builtup_sqkm:.2f} sq km ({builtup_pct:.2f}% of AOI)")
print(f"[OK] Exported built-up mask array -> {builtup_mask_path}")
print("=" * 80)
