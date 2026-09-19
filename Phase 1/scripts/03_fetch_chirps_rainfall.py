"""
Step 3.2 - Rainfall: CHIRPS daily rainfall (via GEE) for the AOI, 2017-2023.

Produces:
  1. A daily AOI-mean rainfall time series CSV (2017-01-01 .. 2023-12-31) --
     needed later for antecedent-precipitation-index features around flood
     dates (e.g. Aug 2018).
  2. Per-year total monsoon-season (Jun-Sep) rainfall raster, multi-band
     GeoTIFF, one band per year 2017-2023 -- CHIRPS native ~5.5km grid,
     to be resampled onto the DEM grid in the feature-engineering step.
  3. A multi-year mean annual total rainfall raster (single band) as a
     static rainfall-climatology feature.
"""
import os
import sys
import json

import ee
import requests
import rasterio
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)
os.makedirs("data/raw", exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
YEARS = list(range(2017, 2024))  # 2017-2023 inclusive
CHIRPS_SCALE_M = 5566  # native ~0.05 deg

print("=" * 80)
print("STEP 3.2: FETCHING CHIRPS DAILY RAINFALL (2017-2023) FOR AOI")
print("=" * 80)

ee.Initialize(project="btp-flood")
geom = ee.Geometry.Rectangle(BBOX)

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
    .filterDate("2017-01-01", "2024-01-01") \
    .filterBounds(geom)
n_images = chirps.size().getInfo()
print(f"[+] CHIRPS daily images 2017-01-01..2023-12-31: {n_images}")

# ---------------------------------------------------------------------------
# 1. Daily AOI-mean time series
# ---------------------------------------------------------------------------
print("\n[+] Reducing daily images to AOI-mean precipitation (server-side map)...")


def reduce_mean(img):
    stat = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=CHIRPS_SCALE_M, maxPixels=1e9)
    return img.set({
        "date_str": img.date().format("YYYY-MM-dd"),
        "precip_mm": stat.get("precipitation"),
    })


reduced = chirps.map(reduce_mean)
dates = reduced.aggregate_array("date_str").getInfo()
precip_vals = reduced.aggregate_array("precip_mm").getInfo()

df = pd.DataFrame({"date": dates, "aoi_mean_precip_mm": precip_vals})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

csv_path = "data/raw/chirps_daily_timeseries_2017_2023.csv"
df.to_csv(csv_path, index=False)
print(f"[OK] Saved daily time series ({len(df)} rows) -> {csv_path}")
print(f"    date range: {df['date'].min().date()} .. {df['date'].max().date()}")
print(f"    missing values: {df['aoi_mean_precip_mm'].isna().sum()}")
print(f"    mean daily rainfall: {df['aoi_mean_precip_mm'].mean():.2f} mm")

aug2018 = df[(df["date"] >= "2018-08-01") & (df["date"] <= "2018-08-31")]
print(f"    Aug 2018 total rainfall (flood month sanity check): {aug2018['aoi_mean_precip_mm'].sum():.1f} mm")

# ---------------------------------------------------------------------------
# 2. Per-year monsoon-season (JJAS) total rainfall raster, multi-band
# ---------------------------------------------------------------------------
print("\n[+] Building per-year JJAS (Jun-Sep) total rainfall rasters...")

yearly_bands = []
for yr in YEARS:
    yr_coll = chirps.filter(ee.Filter.calendarRange(yr, yr, "year")) \
                     .filter(ee.Filter.calendarRange(6, 9, "month"))
    yr_total = yr_coll.sum().rename(f"monsoon_total_{yr}").clip(geom)
    yearly_bands.append(yr_total)

monsoon_stack = ee.Image.cat(yearly_bands)

url_monsoon = monsoon_stack.getDownloadURL({
    "region": geom,
    "scale": CHIRPS_SCALE_M,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})
out_monsoon = "data/raw/chirps_monsoon_jjas_total_by_year_2017_2023.tif"
resp = requests.get(url_monsoon, timeout=300)
resp.raise_for_status()
with open(out_monsoon, "wb") as f:
    f.write(resp.content)
print(f"[OK] Saved -> {out_monsoon} ({len(resp.content) / 1e6:.2f} MB)")

# ---------------------------------------------------------------------------
# 3. Multi-year mean annual total rainfall (climatology, single band)
# ---------------------------------------------------------------------------
print("\n[+] Building multi-year mean annual total rainfall raster...")

annual_totals = []
for yr in YEARS:
    yr_total = chirps.filter(ee.Filter.calendarRange(yr, yr, "year")).sum()
    annual_totals.append(yr_total)
mean_annual = ee.ImageCollection(annual_totals).mean().rename("mean_annual_rainfall_mm").clip(geom)

url_annual = mean_annual.getDownloadURL({
    "region": geom,
    "scale": CHIRPS_SCALE_M,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})
out_annual = "data/raw/chirps_mean_annual_rainfall_2017_2023.tif"
resp = requests.get(url_annual, timeout=300)
resp.raise_for_status()
with open(out_annual, "wb") as f:
    f.write(resp.content)
print(f"[OK] Saved -> {out_annual} ({len(resp.content) / 1e6:.2f} MB)")

# ---------------------------------------------------------------------------
# Sanity checks: open both rasters
# ---------------------------------------------------------------------------
print("\n[+] Verifying rasters open correctly...")
for path in [out_monsoon, out_annual]:
    with rasterio.open(path) as src:
        arr = src.read(masked=True)
        print(f"    {path}")
        print(f"      shape: {src.count} bands x {src.height} x {src.width}, crs: {src.crs}")
        print(f"      value range: {float(arr.min()):.1f} .. {float(arr.max()):.1f} mm")

summary = {
    "dataset": "CHIRPS Daily v2.0 (UCSB Climate Hazards Group)",
    "source": "GEE: UCSB-CHG/CHIRPS/DAILY",
    "native_resolution_m": CHIRPS_SCALE_M,
    "date_range": ["2017-01-01", "2023-12-31"],
    "n_daily_images": n_images,
    "outputs": {
        "daily_timeseries_csv": csv_path,
        "monsoon_jjas_total_by_year_tif": out_monsoon,
        "mean_annual_rainfall_tif": out_annual,
    },
    "daily_timeseries_stats": {
        "n_rows": int(len(df)),
        "missing_values": int(df["aoi_mean_precip_mm"].isna().sum()),
        "mean_daily_mm": float(df["aoi_mean_precip_mm"].mean()),
        "aug_2018_total_mm": float(aug2018["aoi_mean_precip_mm"].sum()),
    },
}
with open("data/raw/chirps_rainfall_metadata.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Metadata written -> data/raw/chirps_rainfall_metadata.json")
print("=" * 80)
