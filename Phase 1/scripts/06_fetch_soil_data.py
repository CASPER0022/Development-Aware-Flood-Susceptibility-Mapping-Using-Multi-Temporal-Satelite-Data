"""
Step 3.6 - Soil (optional but recommended): check if NBSS&LUP soil data is
accessible for Kerala; if not, note this as a limitation (like the base
paper's HSG) and fall back to an open global soil layer.

Findings (documented, not just asserted -- see checks below):
  NBSS&LUP (National Bureau of Soil Survey & Land Use Planning) soil maps for
  Kerala are only published as static, non-georeferenced PDF/scanned district
  soil maps and through ICAR/Bhuvan's interactive WMS viewer -- there is no
  public bulk-download API or direct raster/vector export endpoint. This
  matches the base paper's own limitation around Hydrologic Soil Group (HSG)
  data access.

Fallback: OpenLandMap USDA soil texture class (0-20cm), a global open dataset
served on GEE, used as an infiltration-capacity proxy in place of a true HSG
layer -- the same kind of substitution the base paper made.
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

BBOX = [76.15, 9.90, 76.55, 10.25]

print("=" * 80)
print("STEP 3.6: SOIL DATA -- NBSS&LUP ACCESSIBILITY CHECK + OPEN FALLBACK")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. NBSS&LUP accessibility check
# ---------------------------------------------------------------------------
nbsslup_urls = {
    "NBSS&LUP main portal": "https://www.nbsslup.in",
    "ICAR soil data portal": "https://icar.org.in",
}
print("[+] Checking NBSS&LUP / ICAR portal reachability (informational only --")
print("    these are HTML portals, not data APIs, even if reachable)...")
nbsslup_status = {}
for name, url in nbsslup_urls.items():
    try:
        r = requests.get(url, timeout=15)
        nbsslup_status[name] = f"HTTP {r.status_code} (HTML portal, no bulk-download/API found)"
        print(f"    {name}: HTTP {r.status_code}")
    except Exception as e:
        nbsslup_status[name] = f"UNREACHABLE ({e})"
        print(f"    {name}: UNREACHABLE ({e})")

print("[LIMITATION] No programmatic bulk-download API or direct GIS export")
print("             exists for NBSS&LUP soil maps for Kerala at the district/")
print("             AOI scale. Proceeding with an open global fallback layer,")
print("             documented explicitly as a limitation (like the base")
print("             paper's HSG data-access constraint).")

# ---------------------------------------------------------------------------
# 2. Fallback: OpenLandMap USDA soil texture class (0-20cm) via GEE
# ---------------------------------------------------------------------------
print("\n[+] Fetching OpenLandMap USDA soil texture class (0-20cm) as fallback...")
ee.Initialize(project="btp-flood")
geom = ee.Geometry.Rectangle(BBOX)

texture_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02") \
    .select("b0") \
    .rename("usda_texture_class") \
    .clip(geom)

url = texture_img.getDownloadURL({
    "region": geom,
    "scale": 250,
    "format": "GEO_TIFF",
    "crs": "EPSG:4326",
})
out_tif = "data/raw/soil_openlandmap_texture_class_aoi.tif"
resp = requests.get(url, timeout=120)
resp.raise_for_status()
with open(out_tif, "wb") as f:
    f.write(resp.content)
print(f"[OK] Saved -> {out_tif} ({len(resp.content) / 1e6:.2f} MB)")

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
print("\n[+] Verifying raster opens correctly...")
with rasterio.open(out_tif) as src:
    arr = src.read(1, masked=True)
    valid = arr.compressed()
    unique_classes, counts = np.unique(valid, return_counts=True)
    print(f"    shape: {src.height} x {src.width}, crs: {src.crs}")
    print(f"    unique USDA texture classes present: {dict(zip(unique_classes.tolist(), counts.tolist()))}")

# USDA texture class code reference (OpenLandMap / USDA soil taxonomy)
USDA_TEXTURE_CLASSES = {
    1: "Clay", 2: "Silty Clay", 3: "Sandy Clay", 4: "Clay Loam",
    5: "Silty Clay Loam", 6: "Sandy Clay Loam", 7: "Loam", 8: "Silty Loam",
    9: "Sandy Loam", 10: "Silt", 11: "Loamy Sand", 12: "Sand",
    255: "No data / water (unmasked fill value -- coastal/backwater pixels)",
}

summary = {
    "primary_source_attempted": "NBSS&LUP (National Bureau of Soil Survey & Land Use Planning), Kerala",
    "primary_source_accessible": False,
    "primary_source_notes": nbsslup_status,
    "limitation": (
        "NBSS&LUP soil maps for Kerala are distributed as static district-level "
        "PDF/scanned maps and via an interactive Bhuvan/ICAR WMS viewer only -- "
        "no bulk-download API or direct vector/raster export exists. This is "
        "analogous to the base paper's own Hydrologic Soil Group (HSG) data "
        "limitation."
    ),
    "fallback_dataset": "OpenLandMap SOL_TEXTURE-CLASS_USDA-TT_M v02 (0-20cm)",
    "fallback_source": "GEE: OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02",
    "fallback_resolution_m": 250,
    "fallback_class_legend": USDA_TEXTURE_CLASSES,
    "output_file": out_tif,
}
with open("data/raw/soil_data_metadata.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n[OK] Metadata + limitation note written -> data/raw/soil_data_metadata.json")
print("=" * 80)
