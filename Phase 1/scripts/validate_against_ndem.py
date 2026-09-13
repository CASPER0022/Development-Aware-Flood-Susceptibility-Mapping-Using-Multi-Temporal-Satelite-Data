import os
import sys
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_dir = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_dir)

print("=" * 80)
print("VALIDATION AGAINST NDEM (NRSC/ISRO) OFFICIAL 2018-08-21 FLOOD GROUND TRUTH")
print("=" * 80)
print("See data/validation/SOURCE.md for provenance and the temporal-offset caveat")
print("(NDEM passes are ~18:00-19:30 IST, ~12-13h after our ~06:10 IST Sentinel-1")
print("scene, same calendar day).")
print("=" * 80)

BBOX = [76.15, 9.90, 76.55, 10.25]
lon_min, lat_min, lon_max, lat_max = BBOX
GRID_W, GRID_H = 3800, 2900
transform = rasterio.transform.from_bounds(*BBOX, GRID_W, GRID_H)

# -------------------------------------------------------------------------
# Load & rasterize ground truth
# -------------------------------------------------------------------------
gdf = gpd.read_parquet("data/validation/NDEM_KL_Floods_Inundation.parquet")
aug21 = gdf[gdf['from_time'].astype(str).str.startswith('21-08-2018')].copy()
aoi_poly = box(*BBOX)
aug21 = aug21[aug21.intersects(aoi_poly)]
print(f"[+] Loaded {len(aug21)} NDEM flood polygons for 21-08-2018 intersecting the AOI "
      f"(across {aug21['from_time'].nunique()} pass times).")

gt_mask = rasterize(
    [(geom, 1) for geom in aug21.geometry],
    out_shape=(GRID_H, GRID_W),
    transform=transform,
    fill=0,
    dtype=np.uint8
).astype(bool)
print(f"[+] Rasterized ground truth onto the {GRID_W}x{GRID_H} SAR grid: "
      f"{np.sum(gt_mask)} flooded pixels ({np.sum(gt_mask)/(GRID_W*GRID_H)*100:.2f}% of grid).")

# -------------------------------------------------------------------------
# Load detections: legacy fixed-threshold vs. class-adaptive local CCD
# -------------------------------------------------------------------------
legacy_mask = np.load("data/raw/new_flood_inundation_mask_2018.npy")
ccd_mask = np.load("data/raw/new_flood_inundation_mask_2018_ccd.npy")
builtup_mask = np.load("data/raw/builtup_mask_10m.npy")
dry_db = np.load("data/raw/sentinel1_2018_dry_db.npy")
flood_db = np.load("data/raw/sentinel1_2018_flood_db.npy")
valid_mask = ~np.isnan(dry_db) & ~np.isnan(flood_db)

def confusion(pred, truth, domain):
    hits = np.sum(pred & truth & domain)
    misses = np.sum((~pred) & truth & domain)
    false_alarms = np.sum(pred & (~truth) & domain)
    correct_neg = np.sum((~pred) & (~truth) & domain)
    n = np.sum(domain)
    pod = hits / (hits + misses) if (hits + misses) > 0 else float('nan')
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float('nan')
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float('nan')
    acc = (hits + correct_neg) / n if n > 0 else float('nan')
    bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else float('nan')
    return {
        "n_pixels": int(n), "hits": int(hits), "misses": int(misses),
        "false_alarms": int(false_alarms), "correct_negatives": int(correct_neg),
        "POD": round(float(pod), 4), "FAR": round(float(far), 4),
        "CSI": round(float(csi), 4), "accuracy": round(float(acc), 4),
        "frequency_bias": round(float(bias), 4),
    }

domains = {
    "all_valid_AOI": valid_mask,
    "built_up_only": valid_mask & builtup_mask,
    "non_builtup_only": valid_mask & (~builtup_mask),
}

results = {}
for method_name, mask in [("legacy_fixed_threshold", legacy_mask), ("local_ccd_class_adaptive", ccd_mask)]:
    results[method_name] = {}
    print(f"\n{'-'*78}\n  {method_name}\n{'-'*78}")
    for dom_name, dom in domains.items():
        stats = confusion(mask, gt_mask, dom)
        results[method_name][dom_name] = stats
        print(f"  [{dom_name:16s}] POD={stats['POD']:.4f}  FAR={stats['FAR']:.4f}  "
              f"CSI={stats['CSI']:.4f}  Acc={stats['accuracy']:.4f}  Bias={stats['frequency_bias']:.4f}  "
              f"(hits={stats['hits']}, misses={stats['misses']}, false_alarms={stats['false_alarms']})")

print("\n" + "=" * 80)
print("  READING THE NUMBERS")
print("=" * 80)
print("  POD  = fraction of NDEM-flooded pixels we also flagged (higher is better)")
print("  FAR  = fraction of our flagged pixels NDEM does NOT call flooded (lower is better)")
print("  CSI  = hits / (hits+misses+false_alarms), the standard single-number flood-")
print("         detection skill score that penalizes both misses and false alarms")
print("  Bias = (hits+false_alarms)/(hits+misses); >1 means we over-predict area,")
print("         <1 means we under-predict, independent of spatial agreement (CSI)")
print("=" * 80)

with open("data/raw/ndem_validation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("[OK] Saved -> data/raw/ndem_validation_results.json")
