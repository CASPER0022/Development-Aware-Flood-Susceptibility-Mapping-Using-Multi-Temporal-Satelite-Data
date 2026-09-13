import os
import sys
import json
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from scipy.ndimage import median_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_dir = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_dir)

print("=" * 80)
print("DUAL-CRITERION URBAN SAR FLOOD DETECTOR (SPECULAR + DOUBLE-BOUNCE)")
print("Settlement-Masked (ESA WorldCover) + Speckle-Filtered (3x3 Median)")
print("=" * 80)

BBOX = [76.15, 9.90, 76.55, 10.25]
# Native ~10m resolution over 0.4deg x 0.35deg AOI (~44 km x ~38.7 km)
GRID_W, GRID_H = 3800, 2900

dst_crs = 'EPSG:4326'
transform = rasterio.transform.from_bounds(*BBOX, GRID_W, GRID_H)

dry_url = "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2018/3/30/IW/DV/S1A_IW_GRDH_1SDV_20180330T004036_20180330T004101_021237_024863_DC48/measurement/iw-vv.tiff"
flood_url = "https://sentinel-s1-l1c.s3.amazonaws.com/GRD/2018/8/21/IW/DV/S1A_IW_GRDH_1SDV_20180821T004044_20180821T004109_023337_0289D5_D07A/measurement/iw-vv.tiff"

# Unified Calibration Function: 20 * log10(DN) - 55.0 dB (approximate; see metadata note)
def dn_to_db(dn_array):
    dn_clean = np.maximum(dn_array.astype(np.float32), 1.0)
    return 20.0 * np.log10(dn_clean) - 55.0

def load_and_warp_raw_dn(url):
    """Warp raw linear DN intensity to the AOI grid. dB conversion happens
    AFTER speckle filtering, since SAR speckle is multiplicative noise on the
    linear intensity — filtering post-log-transform is the wrong order."""
    with rasterio.open(url) as src:
        gcps, gcp_crs = src.gcps
        raw_dn = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=raw_dn,
            src_gcps=gcps,
            src_crs=gcp_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    return raw_dn

print("[+] Loading & Warping Pre-Monsoon Dry Scene (March 30, 2018) at 10m resolution...")
dry_dn_raw = load_and_warp_raw_dn(dry_url)

print("[+] Loading & Warping Peak Flood Scene (August 21, 2018) at 10m resolution...")
flood_dn_raw = load_and_warp_raw_dn(flood_url)

# -------------------------------------------------------------------------
# Speckle Filtering: 3x3 median filter on raw linear DN, BEFORE dB conversion
# -------------------------------------------------------------------------
print("[+] Applying 3x3 median speckle filter to raw DN intensity (both scenes)...")
dry_dn_filt = median_filter(dry_dn_raw, size=3)
flood_dn_filt = median_filter(flood_dn_raw, size=3)

# -------------------------------------------------------------------------
# Cross-Image Co-Registration
# -------------------------------------------------------------------------
# dry_dn_raw and flood_dn_raw were each warped independently using their own
# scene's GCPs (see load_and_warp_raw_dn above) -- nothing so far cross-checks
# alignment between the two resulting rasters. Sentinel-1 GCP geolocation is
# good but not pixel-perfect, so a residual sub-pixel offset between the two
# dates is expected. Over dense urban texture (building edges, roads) that
# offset creates large, spurious backscatter swings that masquerade as
# double-bounce flood surge -- concentrated exactly where local contrast is
# highest, i.e. across the built-up area rather than at genuine flood
# hotspots. Estimate that offset via phase correlation and shift the flood
# scene onto the dry scene's grid before differencing.
print("[+] Estimating sub-pixel misregistration between dry and flood scenes...")
shift_yx, corr_error, _diffphase = phase_cross_correlation(
    dry_dn_filt, flood_dn_filt, upsample_factor=20, normalization=None
)
shift_mag_px = float(np.hypot(*shift_yx))
print(f"    Detected shift (row, col): ({shift_yx[0]:+.3f}, {shift_yx[1]:+.3f}) px "
      f"(~{shift_mag_px * 10:.1f} m at 10m/pixel), correlation error={corr_error:.4f}")

# Keep the pre-correction flood raster so the registration's actual effect on
# delta_db can be measured below instead of asserted.
flood_dn_filt_unreg = flood_dn_filt.copy()

if shift_mag_px > 0.03:
    flood_dn_filt = ndi_shift(flood_dn_filt, shift=shift_yx, order=3, mode='nearest')
    flood_dn_raw_reg = ndi_shift(flood_dn_raw, shift=shift_yx, order=3, mode='nearest')
    print(f"[OK] Applied sub-pixel registration correction to flood scene ({shift_mag_px:.3f} px).")
else:
    flood_dn_raw_reg = flood_dn_raw
    print("[OK] Misregistration negligible (<0.03 px); no correction applied.")

# Save the UNFILTERED, registered linear intensities too. A local ratio-based
# statistical change test (see local_ccd_flood_detection.py) needs raw
# multiplicative-speckle statistics -- its own windowed averaging is the
# speckle suppression step, so pre-smoothing with the 3x3 median filter above
# would distort the Gamma-distributed noise assumption that test relies on.
os.makedirs("data/raw", exist_ok=True)
np.save("data/raw/sentinel1_2018_dry_dn_raw.npy", dry_dn_raw)
np.save("data/raw/sentinel1_2018_flood_dn_raw_registered.npy", flood_dn_raw_reg)

dry_db = np.where(dry_dn_filt > 10.0, dn_to_db(dry_dn_filt), np.nan)
flood_db = np.where(flood_dn_filt > 10.0, dn_to_db(flood_dn_filt), np.nan)
flood_db_unreg = np.where(flood_dn_filt_unreg > 10.0, dn_to_db(flood_dn_filt_unreg), np.nan)

valid_mask = ~np.isnan(dry_db) & ~np.isnan(flood_db)
total_valid = np.sum(valid_mask)

# Delta dB = Flood_dB - Dry_dB
delta_db = flood_db - dry_db
delta_db_unregistered = flood_db_unreg - dry_db

# Save the filtered SAR arrays now (before loading the built-up mask), so
# fetch_esa_worldcover.py can read dry_db.shape and resample WorldCover onto
# a grid that is guaranteed to match this run.
os.makedirs("data/raw", exist_ok=True)
np.save("data/raw/sentinel1_2018_dry_db.npy", dry_db)
np.save("data/raw/sentinel1_2018_flood_db.npy", flood_db)
np.save("data/raw/sentinel1_2018_delta_db.npy", delta_db)
print("[OK] Saved speckle-filtered dry/flood/delta dB arrays -> data/raw/")

# -------------------------------------------------------------------------
# Load ESA WorldCover Built-up Mask (generated by fetch_esa_worldcover.py)
# -------------------------------------------------------------------------
builtup_mask_path = "data/raw/builtup_mask_10m.npy"

def builtup_mask_is_valid():
    if not os.path.exists(builtup_mask_path):
        return False
    try:
        return np.load(builtup_mask_path).shape == dry_db.shape
    except Exception:
        return False

if not builtup_mask_is_valid():
    print("[!] Built-up mask missing or shape-mismatched. Running fetch_esa_worldcover.py...")
    import fetch_esa_worldcover  # noqa: F401  (regenerates builtup_mask_10m.npy against dry_db.shape)

builtup_mask = np.load(builtup_mask_path)
if builtup_mask.shape != dry_db.shape:
    raise RuntimeError(
        f"Built-up mask shape {builtup_mask.shape} still does not match SAR grid "
        f"shape {dry_db.shape} after regeneration. Check fetch_esa_worldcover.py."
    )
print(f"[OK] Loaded ESA WorldCover built-up mask -> {builtup_mask_path} ({builtup_mask.shape})")

# -------------------------------------------------------------------------
# Per-class delta_db diagnostic (measured, not assumed)
# -------------------------------------------------------------------------
# Every number below is computed from this run's own arrays and printed here
# -- nothing in this file states a dB figure that wasn't just calculated.
wc_path = "data/raw/esa_worldcover_2021_resampled.npy"
class_refs = {}
worldcover = None
if os.path.exists(wc_path):
    worldcover = np.load(wc_path)
    if worldcover.shape == dry_db.shape:
        class_refs = {
            "Tree Cover (10)": (worldcover == 10) & valid_mask,
            "Built-up (50)": (worldcover == 50) & valid_mask,
            "Permanent Water (80)": (worldcover == 80) & valid_mask,
            "Cropland (40)": (worldcover == 40) & valid_mask,
        }
    else:
        print("[!] WorldCover raster shape mismatch; skipping per-class diagnostics.")
else:
    print("[!] esa_worldcover_2021_resampled.npy not found; skipping per-class diagnostics.")

print("\n" + "-" * 78)
print("  PER-CLASS DELTA_DB DIAGNOSTIC -- registration effect (measured)")
print("-" * 78)
class_stats_unreg = {}
class_stats_reg = {}
for name, cls_mask in class_refs.items():
    n = int(np.sum(cls_mask))
    if n < 500:
        print(f"  • {name:22s}: n={n} pixels, too few to report")
        continue
    med_unreg = float(np.nanmedian(delta_db_unregistered[cls_mask]))
    med_reg = float(np.nanmedian(delta_db[cls_mask]))
    class_stats_unreg[name] = med_unreg
    class_stats_reg[name] = med_reg
    print(f"  • {name:22s}: unregistered={med_unreg:+.3f} dB  registered={med_reg:+.3f} dB  "
          f"shift_effect={med_reg - med_unreg:+.3f} dB  (n={n})")
print("-" * 78)
print("  Interpretation: if 'shift_effect' is small relative to the registered")
print("  median itself, the residual positive delta_db is NOT explained by the")
print("  sub-pixel registration correction -- consistent with a radiometric")
print("  (calibration) offset rather than a geometric one, but this is only")
print("  evidence for that claim, not proof of it.")
print("-" * 78)

# -------------------------------------------------------------------------
# Relative Radiometric Normalization (Pseudo-Invariant Feature bias removal)
# -------------------------------------------------------------------------
# dry_db/flood_db are converted from raw DN via a flat, uncalibrated linear
# approximation (see dn_to_db), not each product's real per-scene calibration
# LUT, which can leave a residual scene-to-scene radiometric offset. The
# per-class table printed above is the actual evidence for whether that is
# happening in this run -- read it before trusting the correction below.
#
# If a bias is present, tree cover (WorldCover class 10) is used as a
# Pseudo-Invariant Feature (PIF) reference: canopy backscatter isn't expected
# to change meaningfully between the dry and flood dates except where forest
# is itself inundated, so its median delta_db is a candidate proxy for a pure
# scene-to-scene calibration offset. CAVEAT: Kerala's canopy was heavily
# rain-saturated by 2018-08-21 (peak monsoon), so wet-canopy backscatter
# change is a real, competing explanation for a nonzero tree-cover delta --
# this PIF choice is not validated against that alternative, and the
# resulting pif_bias_db should be treated as a rough correction, not a
# calibrated one.
pif_bias_db = 0.0
tree_cover_ref = class_refs.get("Tree Cover (10)")
if tree_cover_ref is not None and np.sum(tree_cover_ref) > 1000:
    pif_bias_db = float(np.nanmedian(delta_db[tree_cover_ref]))
    print(f"\n[+] PIF normalization: tree-cover median delta = {pif_bias_db:+.3f} dB "
          f"(n={np.sum(tree_cover_ref)}) -- subtracting AOI-wide.")
    delta_db = delta_db - pif_bias_db
    np.save("data/raw/sentinel1_2018_delta_db.npy", delta_db)

    print("  Post-correction per-class median delta_db (built-up/water/cropland should")
    print("  move toward 0 too if the bias really is uniform across land-cover types;")
    print("  if built-up stays elevated while water/cropland collapse to ~0, that argues")
    print("  the built-up signal is NOT calibration drift and this correction is masking")
    print("  it rather than removing noise):")
    for name, cls_mask in class_refs.items():
        if name == "Tree Cover (10)" or name not in class_stats_reg:
            continue
        post_med = float(np.nanmedian(delta_db[cls_mask]))
        print(f"    • {name:22s}: pre-PIF={class_stats_reg[name]:+.3f} dB  "
              f"post-PIF={post_med:+.3f} dB")
else:
    print("[!] Too few tree-cover pixels to estimate PIF bias; skipping normalization.")

# 1. Permanent Pre-existing Water Bodies (Dry March 2018 backscatter <= -15.5 dB)
perm_water = (dry_db <= -15.5) & valid_mask

# 2. Open-Land / Rural Specular Attenuation Inundation:
# Backscatter drops significantly (Delta dB <= -3.0 dB) AND Flood dB <= -15.5 dB
open_flood = (delta_db <= -3.0) & (flood_db <= -15.5) & (~perm_water) & valid_mask

# 3. Urban / Built-up Double-Bounce Backscatter Surge:
# Restricted to genuine ESA WorldCover settlement pixels (class 50) instead of
# a backscatter-value heuristic, so rough terrain / forest / speckle can no
# longer be misclassified as "urban."
urban_zone = builtup_mask & valid_mask
urban_double_bounce_flood = urban_zone & (delta_db >= +2.2) & valid_mask

# Combined Total Event Flood Inundation (Open Specular + Urban Double Bounce)
total_event_flood = open_flood | urban_double_bounce_flood

calc_area_sqkm = 1696.64
perm_sqkm = round((np.sum(perm_water) / total_valid) * calc_area_sqkm, 2)
open_flood_sqkm = round((np.sum(open_flood) / total_valid) * calc_area_sqkm, 2)
urban_flood_sqkm = round((np.sum(urban_double_bounce_flood) / total_valid) * calc_area_sqkm, 2)
total_event_sqkm = round((np.sum(total_event_flood) / total_valid) * calc_area_sqkm, 2)
builtup_total_sqkm = round((np.sum(urban_zone) / total_valid) * calc_area_sqkm, 2)
urban_flood_pct_of_builtup = round((urban_flood_sqkm / builtup_total_sqkm) * 100.0, 2) if builtup_total_sqkm > 0 else 0.0

print("\n" + "=" * 80)
print("     SETTLEMENT-MASKED, SPECKLE-FILTERED SAR FLOOD DETECTION RESULTS")
print("=" * 80)
print(f"  • Master AOI Spatial Resolution                      : {GRID_W} x {GRID_H} pixels (~10m/pixel)")
print(f"  • Total ESA WorldCover Built-up Land in AOI           : {builtup_total_sqkm} sq km")
print(f"  • Permanent Pre-existing Water (Dry March 2018)       : {perm_sqkm} sq km ({(perm_sqkm/calc_area_sqkm)*100:.2f}%)")
print(f"  • Open-Land Specular Flood Inundation (Delta <= -3dB) : {open_flood_sqkm} sq km ({(open_flood_sqkm/calc_area_sqkm)*100:.2f}%)")
print(f"  • URBAN DOUBLE-BOUNCE FLOOD SURGE (Delta >= +2.2dB)   : {urban_flood_sqkm} sq km ({(urban_flood_sqkm/calc_area_sqkm)*100:.2f}%)")
print(f"    -> as % of total built-up land in AOI               : {urban_flood_pct_of_builtup:.2f}%")
print(f"  • TOTAL 2018 EVENT FLOOD INUNDATION FOOTPRINT         : {total_event_sqkm} sq km ({(total_event_sqkm/calc_area_sqkm)*100:.2f}%)")
print("=" * 80)
print("  • Sanity check: urban flood area should be a modest fraction of total")
print("    built-up area and should visually cluster around Aluva / COK /")
print("    Kalamassery — not spread uniformly across forest or farmland.")
print("=" * 80)

# Check specifically around Cochin Airport (COK: 10.152 N, 76.402 E) and Aluva (10.108 N, 76.352 E)
def check_location(name, lat, lon, radius_deg=0.015):
    lat_axis = np.linspace(10.25, 9.90, GRID_H)
    lon_axis = np.linspace(76.15, 76.55, GRID_W)
    r_mask = (np.abs(lat_axis[:, None] - lat) < radius_deg) & \
             (np.abs(lon_axis[None, :] - lon) < radius_deg)
    loc_builtup = np.sum(urban_zone & r_mask)
    loc_urban_f = np.sum(urban_double_bounce_flood & r_mask)
    loc_open_f = np.sum(open_flood & r_mask)
    loc_total_f = np.sum(total_event_flood & r_mask)
    loc_pct = round((loc_urban_f / loc_builtup) * 100.0, 2) if loc_builtup > 0 else 0.0
    print(f"  • Location Check [{name}]:")
    print(f"    - Built-up cells in zone: {loc_builtup}")
    print(f"    - Open Flood Cells: {loc_open_f}, Urban Double-Bounce Flood Cells: {loc_urban_f} ({loc_pct}% of local built-up)")
    print(f"    - Total Event Flood Capture: {loc_total_f} pixels inside hotspot zone")

check_location("Cochin Int'l Airport (COK)", 10.1520, 76.4019)
check_location("Aluva Town Center", 10.1076, 76.3516)

# Save remaining arrays
np.save("data/raw/perm_water_mask_2018.npy", perm_water)
np.save("data/raw/open_flood_mask_2018.npy", open_flood)
np.save("data/raw/urban_double_bounce_mask_2018.npy", urban_double_bounce_flood)
np.save("data/raw/new_flood_inundation_mask_2018.npy", total_event_flood)

# Persist co-registration diagnostics so downstream metadata (Task 1 summary)
# can report the applied correction instead of just this run's stdout.
registration_info = {
    "method": "skimage.registration.phase_cross_correlation (upsample_factor=20) "
              "on 3x3-median-filtered DN intensity, both scenes independently "
              "GCP-warped onto the common AOI grid beforehand",
    "detected_shift_row_col_px": [round(float(shift_yx[0]), 4), round(float(shift_yx[1]), 4)],
    "shift_magnitude_px": round(shift_mag_px, 4),
    "shift_magnitude_m_approx": round(shift_mag_px * 10.0, 2),
    "correlation_error": round(float(corr_error), 6),
    "correction_applied": bool(shift_mag_px > 0.03),
    "correction_method": "scipy.ndimage.shift(order=3, mode='nearest') applied to flood scene DN",
    "per_class_delta_db_diagnostic": {
        "note": "Measured medians, computed and printed by this run -- not asserted. "
                "shift_effect_db is registered-minus-unregistered per class; a small "
                "shift_effect relative to the registered median is evidence (not proof) "
                "that the registration correction is not what's driving the residual "
                "positive delta_db.",
        "unregistered_median_db": {k: round(v, 4) for k, v in class_stats_unreg.items()},
        "registered_median_db": {k: round(v, 4) for k, v in class_stats_reg.items()},
    },
    "radiometric_normalization": {
        "method": "Pseudo-Invariant Feature (PIF) bias removal using WorldCover "
                  "class 10 (Tree Cover) as the stable reference target",
        "pif_median_bias_db": round(pif_bias_db, 4),
        "caveat": "Tree-cover backscatter between a dry-season date and peak-monsoon "
                  "date can itself shift due to canopy water content, not just "
                  "calibration drift -- this PIF choice is not validated against that "
                  "alternative. Treat pif_bias_db as a rough correction.",
        "note": "Subtracted from delta_db AOI-wide before thresholding. See "
                "per_class_delta_db_diagnostic for the measured evidence this is based on."
    }
}
with open("data/raw/registration_diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(registration_info, f, indent=2)

urban_flood_pct_of_builtup_final = round((np.sum(urban_double_bounce_flood) / np.sum(urban_zone)) * 100.0, 2) if np.sum(urban_zone) > 0 else 0.0
print(f"  • Urban double-bounce flood as % of ALL built-up land in AOI: {urban_flood_pct_of_builtup_final}%")
print("\n[OK] Saved settlement-masked, speckle-filtered flood detection arrays -> data/raw/")