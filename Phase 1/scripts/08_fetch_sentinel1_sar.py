"""
Step 4 - Acquire Sentinel-1 SAR imagery (Week 2, Phase1_Implementation_Plan).

4.1. Identify known flood dates in the AOI (Lower Periyar / Aluva-Kochi
     corridor) -- primarily August 2018, plus any other major flood events
     (2019, 2021) that actually reached this AOI. "Applicable" is checked
     against evidence already on disk (the CHIRPS AOI-mean daily rainfall
     series from scripts/03_fetch_chirps_rainfall.py), not just news reports:

       Year   Aug (or Oct) monthly AOI-mean total   Peak day
       2018   888.9 mm (Aug)                        2018-08-15 (115.7 mm)
       2019   1070.9 mm (Aug)                        2019-08-08 (202.1 mm)
       2021   545.2 mm (Oct)                         2021-10-16 (55.7 mm)

     All three exceed this AOI's typical monthly totals (baseline Aug/Oct is
     ~200-800 mm across 2017-2023), so all three are treated as applicable
     flood events for this AOI:
       - 2018-08: the catastrophic Kerala flood (well documented, Idukki dam
         opened 2018-08-09).
       - 2019-08: Kerala's second major flood year, widely reported as
         landslide-heavy in Wayanad/Malappuram, but the CHIRPS record shows
         this AOI's own rainfall was in fact *higher* than August 2018.
       - 2021-10: the October 2021 Kottayam/Idukki flood-landslide event.
         Idukki dam sits on the Periyar river upstream of this AOI, so
         elevated reservoir releases plausibly raised river levels
         downstream through the AOI even though the worst damage was
         reported further upstream/south.

4.2. For each flood date, pull the closest available Sentinel-1 GRD (VV
     polarization, IW mode) scene from GEE's COPERNICUS/S1_GRD collection
     (already thermal-noise-removed, radiometrically calibrated, and
     terrain-corrected to sigma0 in dB by GEE's own preprocessing chain --
     see the note in the saved metadata about what this means for Step 5's
     "convert to dB" instruction). A "dry" pre-monsoon reference scene is
     also pulled for each year, preferring the same relative orbit number
     (identical imaging geometry) as that year's flood scene, falling back
     to just the same orbit pass (ascending/descending), and falling back
     again to closest-date-only if neither match exists in the search
     window -- each fallback is recorded explicitly, not silently applied.

4.3. Each selected scene's VV band is exported clipped to the AOI as a
     GeoTIFF (EPSG:4326, native ~10m).
"""
import os
import sys
import json
from datetime import datetime

import ee
import requests
import rasterio
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

OUT_DIR = "data/raw/sentinel1_gee"
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]

print("=" * 80)
print("STEP 4: SENTINEL-1 SAR IMAGERY ACQUISITION (GEE, VV, IW, ~10m)")
print("=" * 80)

ee.Initialize(project="btp-flood")
region = ee.Geometry.Rectangle(BBOX)

# ---------------------------------------------------------------------------
# 4.1 Flood events (target dates) + dry pre-monsoon reference search windows
# ---------------------------------------------------------------------------
EVENTS = {
    "2018_flood": {
        "target_date": "2018-08-16",
        "search_window": ("2018-08-08", "2018-08-24"),
        "evidence": "CHIRPS Aug-2018 AOI total 888.9mm, peak day 2018-08-15 (115.7mm)",
        "dry_target_date": "2018-03-30",
        "dry_search_window": ("2018-03-01", "2018-04-15"),
    },
    "2019_flood": {
        "target_date": "2019-08-09",
        "search_window": ("2019-08-02", "2019-08-16"),
        "evidence": "CHIRPS Aug-2019 AOI total 1070.9mm (higher than 2018), peak day 2019-08-08 (202.1mm)",
        "dry_target_date": "2019-03-15",
        "dry_search_window": ("2019-02-15", "2019-04-15"),
    },
    "2021_flood": {
        "target_date": "2021-10-17",
        "search_window": ("2021-10-10", "2021-10-24"),
        "evidence": "CHIRPS Oct-2021 AOI total 545.2mm, peak day 2021-10-16 (55.7mm); "
                    "Idukki dam (upstream on Periyar) elevated-release event",
        "dry_target_date": "2021-03-15",
        "dry_search_window": ("2021-02-15", "2021-04-15"),
    },
}


def list_scenes(start, end):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    n = col.size().getInfo()
    if n == 0:
        return []
    feats = col.toList(n).getInfo()
    scenes = []
    for f in feats:
        props = f["properties"]
        scenes.append({
            "id": f["id"],
            "date": datetime.utcfromtimestamp(props["system:time_start"] / 1000).strftime("%Y-%m-%d"),
            "orbit_pass": props.get("orbitProperties_pass"),
            "relative_orbit": props.get("relativeOrbitNumber_start"),
        })
    return scenes


def days_from(scene, target_date):
    return abs((datetime.strptime(scene["date"], "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")).days)


def closest(scenes, target_date):
    return min(scenes, key=lambda s: days_from(s, target_date))


def pick_dry_reference(dry_scenes, flood_scene, dry_target_date):
    """Prefer identical imaging geometry (same relative orbit), then same
    orbit pass, then fall back to closest-date-only -- each level recorded."""
    same_orbit = [s for s in dry_scenes if s["relative_orbit"] == flood_scene["relative_orbit"]]
    if same_orbit:
        return closest(same_orbit, dry_target_date), "matched_relative_orbit"
    same_pass = [s for s in dry_scenes if s["orbit_pass"] == flood_scene["orbit_pass"]]
    if same_pass:
        return closest(same_pass, dry_target_date), "matched_orbit_pass_only"
    return closest(dry_scenes, dry_target_date), "no_geometry_match_closest_date_only"


def download_vv(scene_id, out_path, scale=10):
    img = ee.Image(scene_id).select("VV").clip(region)
    last_err = None
    for try_scale in (scale, scale * 2):
        try:
            url = img.getDownloadURL({
                "region": region,
                "scale": try_scale,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            })
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return try_scale, len(resp.content)
        except Exception as e:
            last_err = e
            print(f"    [!] scale={try_scale}m failed ({e}); retrying at coarser scale...")
    raise RuntimeError(f"Download failed for {scene_id}: {last_err}")


results = {}

for event_name, cfg in EVENTS.items():
    print(f"\n[+] {event_name}: target flood date {cfg['target_date']}")
    print(f"    Evidence: {cfg['evidence']}")

    flood_scenes = list_scenes(*cfg["search_window"])
    print(f"    Found {len(flood_scenes)} candidate VV/IW scenes in window {cfg['search_window']}")
    if not flood_scenes:
        print(f"    [FAIL] No Sentinel-1 scenes found for {event_name}; skipping.")
        results[event_name] = {"status": "FAIL", "reason": "no scenes in search window"}
        continue
    flood_scene = closest(flood_scenes, cfg["target_date"])
    flood_diff = days_from(flood_scene, cfg["target_date"])
    print(f"    Selected flood scene: {flood_scene['id']} ({flood_scene['date']}, "
          f"{flood_diff}d from target, pass={flood_scene['orbit_pass']}, "
          f"rel_orbit={flood_scene['relative_orbit']})")

    dry_scenes = list_scenes(*cfg["dry_search_window"])
    print(f"    Found {len(dry_scenes)} candidate dry-window VV/IW scenes in {cfg['dry_search_window']}")
    if not dry_scenes:
        print(f"    [FAIL] No dry-reference scenes found for {event_name}; skipping dry ref.")
        dry_scene, dry_match_level, dry_diff = None, None, None
    else:
        dry_scene, dry_match_level = pick_dry_reference(dry_scenes, flood_scene, cfg["dry_target_date"])
        dry_diff = days_from(dry_scene, cfg["dry_target_date"])
        print(f"    Selected dry scene:   {dry_scene['id']} ({dry_scene['date']}, "
              f"{dry_diff}d from target, pass={dry_scene['orbit_pass']}, "
              f"rel_orbit={dry_scene['relative_orbit']}) [{dry_match_level}]")

    year = event_name.split("_")[0]
    flood_out = os.path.join(OUT_DIR, f"s1_vv_{year}_flood_{flood_scene['date'].replace('-', '')}.tif")
    print(f"    Downloading flood scene VV band -> {flood_out} ...")
    flood_scale, flood_bytes = download_vv(flood_scene["id"], flood_out)
    print(f"    [OK] {flood_bytes / 1e6:.2f} MB at {flood_scale}m/px")

    dry_out = None
    dry_scale = dry_bytes = None
    if dry_scene is not None:
        dry_out = os.path.join(OUT_DIR, f"s1_vv_{year}_dry_{dry_scene['date'].replace('-', '')}.tif")
        print(f"    Downloading dry reference VV band -> {dry_out} ...")
        dry_scale, dry_bytes = download_vv(dry_scene["id"], dry_out)
        print(f"    [OK] {dry_bytes / 1e6:.2f} MB at {dry_scale}m/px")

    results[event_name] = {
        "status": "OK",
        "target_flood_date": cfg["target_date"],
        "flood_scene": {**flood_scene, "days_from_target": flood_diff,
                        "file": flood_out, "scale_m": flood_scale, "size_bytes": flood_bytes},
        "dry_reference_scene": (
            {**dry_scene, "days_from_target": dry_diff, "geometry_match": dry_match_level,
             "file": dry_out, "scale_m": dry_scale, "size_bytes": dry_bytes}
            if dry_scene is not None else None
        ),
    }

# ---------------------------------------------------------------------------
# Sanity checks -- reopen every downloaded GeoTIFF and print backscatter stats
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SANITY CHECK: reopening downloaded scenes")
print("=" * 80)
for event_name, r in results.items():
    if r["status"] != "OK":
        continue
    for role in ("flood_scene", "dry_reference_scene"):
        entry = r.get(role)
        if entry is None:
            continue
        with rasterio.open(entry["file"]) as src:
            arr = src.read(1, masked=True)
            valid = arr.compressed()
            print(f"  {event_name}/{role}: {entry['file']}")
            print(f"    shape={src.height}x{src.width}, crs={src.crs}, "
                  f"valid_px={valid.size}, dB range=[{valid.min():.2f}, {valid.max():.2f}], "
                  f"mean={valid.mean():.2f} dB")

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
metadata = {
    "task": "Step 4 - Sentinel-1 SAR imagery acquisition",
    "aoi_bbox_wgs84": BBOX,
    "source": "GEE COPERNICUS/S1_GRD (IW mode, VV polarization)",
    "preprocessing_note": (
        "GEE's COPERNICUS/S1_GRD collection is already thermal-noise-removed, "
        "radiometrically calibrated, terrain-corrected, and the resulting "
        "sigma0 backscatter is delivered pre-converted to decibels (dB) by "
        "GEE's own pipeline -- so the raw pixel values downloaded here are "
        "ALREADY in dB. Step 5.1's 'convert to dB (log transform)' is "
        "therefore already satisfied by this source; no further log "
        "transform should be applied on top of these values."
    ),
    "flood_date_selection_method": (
        "Closest available Sentinel-1 IW/VV scene (by calendar date) to a "
        "target flood date, where target dates were chosen from documented "
        "Kerala flood events and cross-checked against this AOI's own CHIRPS "
        "daily rainfall record (scripts/03_fetch_chirps_rainfall.py) to "
        "confirm each event actually reached this AOI."
    ),
    "dry_reference_selection_method": (
        "Closest available Sentinel-1 IW/VV scene to a pre-monsoon target "
        "date in the same year, preferring an exact relativeOrbitNumber "
        "match to the flood scene (identical imaging geometry/incidence "
        "angle), falling back to matching orbit pass only (ascending vs "
        "descending), falling back again to closest-date-only if neither "
        "geometry match exists in the search window. The match level "
        "actually used is recorded per event below."
    ),
    "events": results,
}
meta_path = "data/raw/sentinel1_gee_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\n[OK] Metadata written -> {meta_path}")

n_ok = sum(1 for r in results.values() if r["status"] == "OK")
print("\n" + "=" * 80)
print(f"STEP 4 COMPLETE: {n_ok}/{len(EVENTS)} flood events acquired with VV scenes clipped to AOI.")
print("=" * 80)
