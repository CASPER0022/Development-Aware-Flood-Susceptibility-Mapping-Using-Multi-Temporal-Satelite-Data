"""
Step 8 - Build the training / testing dataset
(Week 3, Phase1_Implementation_Plan).

Event: Aug 2018 (Sentinel-1 21-08-2018). 2019 is weaker (GTI 37 %) and 2021 is
unusable (GTI 15 %), per Step 6, so they are not used for labels here.

Label grid. Samples live on the Step 7 feature grid (native FABDEM 1", ~30 m).
The Step 5 inventory is on the S1 grid (~20 m). The two grids are exactly
commensurate: the pixel size ratio is 1.5 and the 30 m grid starts half a 20 m
pixel up-left of the 20 m grid (both asserted below). Each 20 m pixel is split
into 2 x 2 sub-pixels of 10 m; a 30 m cell is then exactly a 3 x 3 block of
sub-pixels. So every 30 m cell gets the *exact* area fraction of each 20 m
class -- no resampling approximation.

8.1 Label pools (all inside the Step 7 land domain)
    FLOOD (1)      >= 50 % of the cell is Otsu flood that is
                     - in a patch >= 10 px (small isolated patches are the least
                       reliable part of the inventory, Step 6), and
                     - confirmed by the NRSC/ISRO NDEM 2018 event envelope
                       within 40 m (Step 6 GTI_40m rule).
    NON-FLOOD (0)  cell and its 60 m surroundings have no Otsu flood at all,
                   the cell is outside the NDEM envelope (+40 m), and it is
                   more than 60 m from reference (dry-season) water.
                   Why the envelope: our scene is 3-4 days post-peak; land that
                   NDEM mapped flooded on 17-18 Aug but was dry by the 21st
                   would otherwise enter the dataset as "non-flood".
                   Why 60 m from reference water: Step 6 found the river /
                   backwater fringe is where the inventory is least reliable.
    Pixels in neither pool (unconfirmed Otsu flood, mixed cells, the water
    fringe) are not sampled.
    Sampling: all flood cells, and an equal number of non-flood cells drawn
    uniformly at random (class-stratified, seed 42) -- the base paper's
    balanced design.

8.2 Features: all 11 Step 7 stack bands, read at each sampled cell. Metadata
    columns (not features): ids, row/col, lon/lat, UTM x/y, label fractions,
    and block_id = 2 km UTM block, kept for Phase 2 spatial cross-validation.

8.3 Quality pass: missing / non-finite values, negative distances, ranges vs
    the Step 7 checkpoint, class balance, duplicate cells, per-class feature
    summaries and single-feature AUC. No feature is derived from Sentinel-1,
    so none can encode the label directly; a single-feature AUC > 0.95 is
    still flagged as suspicious.

Outputs
    data/processed/training/training_samples_2018.csv
    data/processed/training/label_pools_2018.tif   uint8 0 = non-flood pool,
                                                   1 = flood pool, 255 = excluded
    outputs/metrics/step8_training_dataset_summary.json
    outputs/maps/step8_sample_locations.png
    outputs/maps/step8_feature_distributions.png
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

CLASSES_2018 = "data/processed/flood_inventory/water_classes_2018_20180821.tif"
NDEM = "data/validation/NDEM_KL_Floods_Inundation.parquet"
STACK = "data/processed/features/feature_stack.tif"
DOMAIN = "data/processed/features/analysis_domain.tif"
STEP7_JSON = "outputs/metrics/step7_feature_stack_checkpoint.json"
OUT_DIR = "data/processed/training"
MAP_DIR = "outputs/maps"
METRICS_DIR = "outputs/metrics"
for d in (OUT_DIR, MAP_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
UTM = "EPSG:32643"
NDEM_ENVELOPE_2018 = ["17-08-2018", "18-08-2018", "21-08-2018", "24-08-2018", "27-08-2018", "28-08-2018"]
MIN_PATCH_PX = 10          # 20 m px, Step 6 GTI_patch rule
TOL_PX20 = 2               # 40 m NDEM confirmation tolerance (Step 6)
WATER_BELT_PX20 = 3        # 60 m fringe around reference water (Step 6)
FLOOD_FRAC_MIN = 0.5       # majority rule for a flood cell
NEG_FLOOD_BUFFER_PX30 = 2  # non-flood cells: no Otsu flood within 60 m
BLOCK_M = 2000             # spatial block size for Phase 2 CV
SEED = 42
AUC_SUSPICIOUS = 0.95
C_FLOOD = "#2a78d6"
C_DRY = "#eb6834"
C_ENV = "#52514e"

print("=" * 80)
print("STEP 8: TRAINING / TESTING DATASET (Aug 2018 flood inventory + Step 7 features)")
print("=" * 80)
t_start = time.time()
results = {"step": 8, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "event": "2018-08-21"}

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
print("\n[1/6] Inputs")
with rasterio.open(STACK) as src:
    stack = src.read()
    names = list(src.descriptions)
    T30 = src.transform
    CRS = src.crs
    H30, W30 = src.shape
with rasterio.open(DOMAIN) as src:
    land = src.read(1).astype(bool)
with rasterio.open(CLASSES_2018) as src:
    cls = src.read(1)
    T20 = src.transform
    prof20 = src.profile
    assert src.crs == CRS
H20, W20 = cls.shape
print(f"    feature grid {H30} x {W30} ({len(names)} bands), inventory grid {H20} x {W20}")

# exact commensurability of the two grids
ratio = T30.a / T20.a
off_x = (T20.c - T30.c) / T20.a
off_y = (T30.f - T20.f) / abs(T20.e)
assert abs(ratio - 1.5) < 1e-9 and abs(T30.e / T20.e - 1.5) < 1e-9, ratio
assert abs(off_x - 0.5) < 1e-6 and abs(off_y - 0.5) < 1e-6, (off_x, off_y)
print(f"    pixel ratio {ratio:.9f}, 20 m grid offset by ({off_x:.6f}, {off_y:.6f}) px -> exact 3x3 sub-pixel aggregation")

ndem = gpd.read_parquet(NDEM)
if ndem.crs is None:
    ndem = ndem.set_crs("EPSG:4326", allow_override=True)
ndem = ndem[ndem.intersects(box(*BBOX))].copy()
ndem["date"] = ndem["from_time"].astype(str).str[:10]
env_polys = ndem[ndem["date"].isin(NDEM_ENVELOPE_2018)]
env20 = rasterize([(g, 1) for g in env_polys.geometry], out_shape=(H20, W20), transform=T20,
                  fill=0, dtype="uint8").astype(bool)
print(f"    NDEM 2018 envelope: {len(env_polys)} polygons over {len(NDEM_ENVELOPE_2018)} dates")


# ---------------------------------------------------------------------------
# 20 m masks -> exact 30 m fractions
# ---------------------------------------------------------------------------
def dilate(mask, px):
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=px)


def to_30m_fraction(mask20):
    """Exact area fraction of a 20 m mask in every 30 m cell (NaN-free).
    20 m px -> 2x2 sub-pixels of 10 m; 30 m cell = 3x3 sub-pixels, with the
    30 m grid starting one sub-pixel up-left of the 20 m grid."""
    sub = np.repeat(np.repeat(mask20.astype(np.float32), 2, axis=0), 2, axis=1)
    need_h, need_w = 3 * H30, 3 * W30
    canvas = np.zeros((need_h, need_w), np.float32)
    h = min(sub.shape[0], need_h - 1)
    w = min(sub.shape[1], need_w - 1)
    canvas[1:1 + h, 1:1 + w] = sub[:h, :w]
    return canvas.reshape(H30, 3, W30, 3).mean(axis=(1, 3))


print("\n[2/6] 8.1 Label pools")
valid20 = cls != 255
flood20 = cls == 1
refw20 = cls == 2
lab, n = ndimage.label(flood20, structure=np.ones((3, 3)))
sizes = np.bincount(lab.ravel())
big = sizes >= MIN_PATCH_PX
big[0] = False
flood_big20 = big[lab]
env_tol20 = dilate(env20, TOL_PX20)
confirmed20 = flood_big20 & env_tol20
belt20 = dilate(refw20, WATER_BELT_PX20)

f_valid = to_30m_fraction(valid20)
f_flood = to_30m_fraction(flood20)
f_conf = to_30m_fraction(confirmed20)
f_env = to_30m_fraction(env_tol20)
f_belt = to_30m_fraction(belt20)
f_refw = to_30m_fraction(refw20)

# sanity: the whole 20 m grid fits inside the 30 m grid, so area is conserved exactly
# (every 20 m px = 4 sub-pixels, every 30 m cell = 9 sub-pixels)
assert abs(f_flood.sum() * 9 - flood20.sum() * 4) < 1e-2, "label aggregation lost area"

any_flood30 = f_flood > 0
near_flood30 = ndimage.binary_dilation(any_flood30, structure=np.ones((3, 3)), iterations=NEG_FLOOD_BUFFER_PX30)
full_valid = f_valid >= 1 - 1e-6

pos_pool = land & full_valid & (f_conf >= FLOOD_FRAC_MIN)
neg_pool = land & full_valid & ~near_flood30 & (f_env == 0) & (f_belt == 0) & (f_refw == 0)
assert not (pos_pool & neg_pool).any()
unconf_flood = land & (f_flood >= FLOOD_FRAC_MIN) & ~pos_pool

lat_rows = T30.f + (np.arange(H30) + 0.5) * T30.e
row_km2 = (T30.a * 111.320 * np.cos(np.radians(lat_rows))) * (abs(T30.e) * 110.574)
km2 = lambda m: float((m.sum(axis=1) * row_km2).sum())
pools = {
    "otsu_flood_cells_ge50pct": int((land & (f_flood >= FLOOD_FRAC_MIN)).sum()),
    "flood_pool_cells": int(pos_pool.sum()), "flood_pool_km2": round(km2(pos_pool), 2),
    "unconfirmed_or_small_patch_flood_cells_excluded": int(unconf_flood.sum()),
    "nonflood_pool_cells": int(neg_pool.sum()), "nonflood_pool_km2": round(km2(neg_pool), 2),
    "land_cells": int(land.sum()),
    "land_excluded_from_both_pools_km2": round(km2(land & ~pos_pool & ~neg_pool), 2),
    "excluded_by_ndem_envelope_km2": round(km2(land & full_valid & ~near_flood30 & (f_env > 0)), 2),
    "excluded_by_water_fringe_km2": round(km2(land & full_valid & ~near_flood30 & (f_env == 0) & (f_belt > 0)), 2),
}
print(f"    Otsu flood cells (>= 50 %): {pools['otsu_flood_cells_ge50pct']}; "
      f"flood pool (patch >= {MIN_PATCH_PX} px AND NDEM-confirmed): {pools['flood_pool_cells']} cells, "
      f"{pools['flood_pool_km2']} km2")
print(f"    non-flood pool: {pools['nonflood_pool_cells']} cells, {pools['nonflood_pool_km2']} km2 "
      f"(NDEM envelope removed {pools['excluded_by_ndem_envelope_km2']} km2, water fringe "
      f"{pools['excluded_by_water_fringe_km2']} km2)")

pool_raster = np.full((H30, W30), 255, np.uint8)
pool_raster[neg_pool] = 0
pool_raster[pos_pool] = 1
p30 = {"driver": "GTiff", "height": H30, "width": W30, "count": 1, "dtype": "uint8", "crs": CRS,
       "transform": T30, "nodata": 255, "compress": "deflate"}
with rasterio.open(f"{OUT_DIR}/label_pools_2018.tif", "w", **p30) as dst:
    dst.write(pool_raster, 1)
    dst.set_band_description(1, "0 = non-flood pool, 1 = flood pool, 255 = not sampled")

# ---------------------------------------------------------------------------
# Balanced class-stratified sampling
# ---------------------------------------------------------------------------
print("\n[3/6] Balanced sampling")
rng = np.random.default_rng(SEED)
pos_idx = np.flatnonzero(pos_pool.ravel())
neg_all = np.flatnonzero(neg_pool.ravel())
n_per_class = len(pos_idx)
assert len(neg_all) >= n_per_class
neg_idx = rng.choice(neg_all, size=n_per_class, replace=False)
idx = np.r_[pos_idx, neg_idx]
label = np.r_[np.ones(n_per_class, np.int8), np.zeros(n_per_class, np.int8)]
order = rng.permutation(len(idx))
idx, label = idx[order], label[order]
print(f"    {n_per_class} flood + {n_per_class} non-flood = {len(idx)} samples "
      f"(non-flood drawn from {len(neg_all)} eligible cells, seed {SEED})")

# ---------------------------------------------------------------------------
# 8.2 Feature extraction
# ---------------------------------------------------------------------------
print("\n[4/6] 8.2 Feature extraction")
rows, cols = np.divmod(idx, W30)
lon = T30.c + (cols + 0.5) * T30.a
lat = T30.f + (rows + 0.5) * T30.e
x_utm, y_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(lon, lat)
df = pd.DataFrame({
    "sample_id": np.arange(len(idx)),
    "label": label,
    "row": rows, "col": cols,
    "lon": np.round(lon, 7), "lat": np.round(lat, 7),
    "x_utm": np.round(x_utm, 2), "y_utm": np.round(y_utm, 2),
    "block_id": [f"{int(x // BLOCK_M)}_{int(y // BLOCK_M)}" for x, y in zip(x_utm, y_utm)],
    "otsu_flood_frac": np.round(f_flood.ravel()[idx], 4),
    "confirmed_flood_frac": np.round(f_conf.ravel()[idx], 4),
})
flat = stack.reshape(len(names), -1)
for b, name in enumerate(names):
    df[name] = flat[b, idx].astype(np.float64)
csv_path = f"{OUT_DIR}/training_samples_2018.csv"
df.to_csv(csv_path, index=False, float_format="%.6g")
print(f"    {csv_path}: {len(df)} rows x {df.shape[1]} columns ({len(names)} features)")

# ---------------------------------------------------------------------------
# 8.3 Data-quality pass (re-read the CSV from disk)
# ---------------------------------------------------------------------------
print("\n[5/6] 8.3 Data-quality pass (on the CSV as written)")
d = pd.read_csv(csv_path)
step7 = json.load(open(STEP7_JSON, encoding="utf-8"))["checkpoint_7_5"]["layers"]
checks = {}
na = d[names].isna().sum()
checks["no_missing_values"] = bool(na.sum() == 0)
checks["all_finite"] = bool(np.isfinite(d[names].to_numpy()).all())
dist_cols = [c for c in names if c.startswith("dist_")]
checks["no_negative_distances"] = bool((d[dist_cols] >= 0).all().all())
checks["no_negative_slope_twi_density"] = bool((d[["slope_deg", "drainage_density_km_km2"]] >= 0).all().all())
checks["aspect_in_range"] = bool(d["aspect_deg"].between(-1, 360).all())
out_of_range = {c: int((~d[c].between(step7[c]["min"] - 1e-3, step7[c]["max"] + 1e-3)).sum()) for c in names}
checks["within_step7_layer_ranges"] = bool(sum(out_of_range.values()) == 0)
counts = d["label"].value_counts().to_dict()
checks["classes_balanced"] = bool(counts.get(0, 0) == counts.get(1, 0))
checks["no_duplicate_cells"] = bool(not d.duplicated(["row", "col"]).any())
checks["all_samples_on_land"] = bool(land[d["row"], d["col"]].all())
checks["labels_match_pools"] = bool((pool_raster[d["row"], d["col"]] == d["label"]).all())
all_ok = all(checks.values())
for k, v in checks.items():
    print(f"    [{'PASS' if v else 'FAIL'}] {k}")

per_class = {}
auc = {}
for c in names:
    g = d.groupby("label")[c]
    per_class[c] = {f"label_{k}": {"median": round(float(v.median()), 3), "p10": round(float(v.quantile(0.1)), 3),
                                   "p90": round(float(v.quantile(0.9)), 3)} for k, v in g}
    a = roc_auc_score(d["label"], d[c])
    auc[c] = {"auc": round(float(a), 4), "separability": round(float(max(a, 1 - a)), 4),
              "higher_in": "flood" if a >= 0.5 else "non-flood"}
suspicious = [c for c in names if auc[c]["separability"] > AUC_SUSPICIOUS]
print("    single-feature AUC (separability, direction):")
for c in sorted(names, key=lambda c: -auc[c]["separability"]):
    print(f"      {c:<26} {auc[c]['separability']:.3f}  higher in {auc[c]['higher_in']}")
print(f"    features above {AUC_SUSPICIOUS} (leakage suspicion): {suspicious if suspicious else 'none'}")

# spatial clustering diagnostic: nearest same-class neighbour distance
from scipy.spatial import cKDTree
nn = {}
for k in (0, 1):
    pts = d.loc[d["label"] == k, ["x_utm", "y_utm"]].to_numpy()
    dd, _ = cKDTree(pts).query(pts, k=2)
    nn[f"label_{k}"] = {"median_m": round(float(np.median(dd[:, 1])), 1),
                        "share_with_neighbour_within_45m": round(float((dd[:, 1] <= 45).mean()), 4)}
n_blocks = d["block_id"].nunique()
blocks_pos = d.loc[d["label"] == 1, "block_id"].nunique()
print(f"    nearest same-class sample: flood median {nn['label_1']['median_m']} m "
      f"({100 * nn['label_1']['share_with_neighbour_within_45m']:.0f}% have an adjacent flood sample), "
      f"non-flood median {nn['label_0']['median_m']} m")
print(f"    2 km blocks: {n_blocks} total, flood samples in {blocks_pos}")

results.update({
    "inputs": {"inventory": CLASSES_2018, "ndem": NDEM, "ndem_envelope_dates": NDEM_ENVELOPE_2018,
               "features": STACK, "domain": DOMAIN},
    "rules": {
        "grid": "Step 7 feature grid (FABDEM 1 arcsec); labels aggregated exactly from the 20 m inventory",
        "flood": f">= {FLOOD_FRAC_MIN:.0%} of cell = Otsu flood in patches >= {MIN_PATCH_PX} px, "
                 f"NDEM-envelope confirmed within {TOL_PX20 * 20} m",
        "nonflood": f"no Otsu flood within {NEG_FLOOD_BUFFER_PX30 * 30} m, outside NDEM envelope (+{TOL_PX20 * 20} m), "
                    f"> {WATER_BELT_PX20 * 20} m from reference water",
        "sampling": f"all flood-pool cells + equal number of non-flood cells, uniform random, seed {SEED}",
        "block_id": f"{BLOCK_M} m UTM blocks ({UTM}) for Phase 2 spatial CV",
    },
    "pools": pools,
    "dataset": {"path": csv_path, "n_rows": int(len(d)), "class_counts": {str(k): int(v) for k, v in counts.items()},
                "feature_columns": names,
                "metadata_columns": [c for c in d.columns if c not in names],
                "n_blocks": int(n_blocks), "n_blocks_with_flood": int(blocks_pos)},
    "quality": {"checks": checks, "all_pass": bool(all_ok), "missing_per_feature": {c: int(v) for c, v in na.items()},
                "out_of_step7_range": out_of_range, "per_class_summary": per_class,
                "single_feature_auc": auc, "suspicious_features": suspicious,
                "nearest_same_class_neighbour": nn},
})
with open(f"{METRICS_DIR}/step8_training_dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[6/6] Figures")
elev = stack[names.index("elevation_m")].astype(np.float64)
e = np.nan_to_num(elev, nan=0.0)
gy, gx = np.gradient(e, 29.8, 29.5)
slope_r = np.arctan(np.hypot(gx, gy))
asp_r = np.arctan2(-gx, gy)
hs = np.clip(np.cos(np.radians(45)) * np.cos(slope_r)
             + np.sin(np.radians(45)) * np.sin(slope_r) * np.cos(np.radians(315) - asp_r), 0, 1)
extent = (T30.c, T30.c + W30 * T30.a, T30.f + H30 * T30.e, T30.f)
env30 = f_env > 0

fig, axes = plt.subplots(1, 2, figsize=(17, 8.4), dpi=110)
views = [(axes[0], extent, "Whole AOI"), (axes[1], (76.22, 76.42, 10.08, 10.22), "Zoom: Periyar floodplain (Aluva – Paravur)")]
for ax, lims, ttl in views:
    ax.set_facecolor("#d6dde3")
    ax.imshow(np.where(land, hs, np.nan), cmap="Greys_r", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.where(env30 & land, 1.0, np.nan), cmap=ListedColormap([C_ENV]), alpha=0.25, extent=extent,
              interpolation="nearest")
    ms = 1.2 if ax is axes[0] else 4
    for k, col in ((0, C_DRY), (1, C_FLOOD)):
        s = d[d["label"] == k]
        ax.scatter(s["lon"], s["lat"], s=ms, c=col, lw=0, alpha=0.9)
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_title(ttl, loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_FLOOD, label=f"Flood (label 1), n = {counts[1]:,}"),
                    Line2D([], [], marker="o", ls="", color=C_DRY, label=f"Non-flood (label 0), n = {counts[0]:,}"),
                    Patch(color=C_ENV, alpha=0.25, label="NDEM 2018 envelope (+40 m): no non-flood samples")],
           loc="lower center", ncol=3, fontsize=11, frameon=False)
fig.suptitle("Step 8  Balanced training samples — Aug 2018 Otsu flood (NDEM-confirmed) vs confident non-flood",
             fontsize=13.5, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
p_map = f"{MAP_DIR}/step8_sample_locations.png"
fig.savefig(p_map, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_map}")

pretty = {"elevation_m": "Elevation (m)", "slope_deg": "Slope (°)", "aspect_deg": "Aspect (°, −1 flat)",
          "curvature_plan": "Plan curvature (1/100 m)", "curvature_profile": "Profile curvature (1/100 m)",
          "twi": "TWI", "dist_to_river_m": "Distance to river (m)",
          "dist_to_major_river_m": "Distance to major river (m)", "dist_to_road_m": "Distance to road (m)",
          "drainage_density_km_km2": "Drainage density (km/km²)", "rainfall_mean_annual_mm": "Rainfall (mm/yr)"}
fig, axes = plt.subplots(3, 4, figsize=(18, 11.5), dpi=110)
for ax, c in zip(axes.ravel(), names):
    lo, hi = d[c].quantile([0.005, 0.995])
    bins = np.linspace(lo, hi, 50)
    for k, col, lbl in ((0, C_DRY, "Non-flood"), (1, C_FLOOD, "Flood")):
        ax.hist(d.loc[d["label"] == k, c].clip(lo, hi), bins=bins, density=True, histtype="step",
                lw=2, color=col, label=lbl)
    ax.set_title(f"{pretty[c]}\nsingle-feature AUC {auc[c]['separability']:.2f}", loc="left",
                 fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)
ax = axes.ravel()[-1]
ax.axis("off")
ax.legend(handles=[Line2D([], [], color=C_FLOOD, lw=2, label="Flood (label 1)"),
                   Line2D([], [], color=C_DRY, lw=2, label="Non-flood (label 0)")],
          loc="center", fontsize=12, frameon=False)
fig.suptitle(f"Step 8  Feature distributions by class ({len(d):,} samples; 0.5–99.5 % range shown)",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
p_hist = f"{MAP_DIR}/step8_feature_distributions.png"
fig.savefig(p_hist, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_hist}")

print("\n" + "=" * 80)
print("STEP 8 SUMMARY")
print("=" * 80)
print(f"  Dataset: {csv_path}  ({counts[1]} flood / {counts[0]} non-flood, {len(names)} features)")
print(f"  Quality pass: {'PASS' if all_ok else 'FAIL'}; suspicious single features: {suspicious if suspicious else 'none'}")
print(f"  Runtime {time.time() - t_start:.0f} s")
print("=" * 80)
if not all_ok:
    raise SystemExit("Step 8 quality pass failed - see above.")
