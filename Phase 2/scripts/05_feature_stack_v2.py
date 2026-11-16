"""
Phase 2 Step 5 - Upgrade the feature stack (feature stack v2).

5.1 HAND (height above nearest drainage) from the Step 4 full-basin routing:
    D8 steepest-descent paths on the filled DEM, height above the first
    drainage cell reached (FABDEM elevation). Drainage = cells with upslope
    area >= HAND_CHANNEL_KM2 (fixed before looking at labels) plus open water.
    Sensitivity to the threshold is reported, not used to choose it.
5.2 Event rainfall from CHIRPS daily (0.05 deg): 5-day total ending on the
    Sentinel-1 date of each kept event, and 30-day antecedent total before
    that. Native CHIRPS cells are downloaded, ocean cells filled from the
    nearest land cell, then bilinear to the 30 m grid (Phase 1 Step 7 rule).
5.3 TWI with full-basin accumulation (fixes the Phase 1 truncation); local
    impervious fraction (Step 2) and Dynamic World land-cover class (annual
    mode, 30 m majority) for every event year.
5.4 Soil texture (optional): not added - the 250 m OpenLandMap layer has a
    handful of classes over the AOI (Phase 1 Step 3 note).
5.5 Feature stack v2 per event year (2018, 2019, 2020, 2024) with the Phase 1
    Step 7.5 checks: same grid, no gaps on land, physical ranges, VIF.
    Plus the Step 4 checkpoint question: is UICA physically right (high below
    Kochi's built-up areas, ~0 in forested catchments)?

Outputs
    data/processed/features_v2/{hand_m.tif, twi_full.tif, stack_v2_<event>.tif, features_v2_metadata.json}
    outputs/metrics/step5_feature_stack_v2.json
    outputs/maps/step5_twi_hand.png, step5_event_rainfall.png, step5_stack_v2_overview.png,
                 step5_correlation_vif.png, step5_uica_sanity.png
"""
import os
import json
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

import common as C

BASIN = C.p2("data", "processed", "basin")
UO = C.p2("data", "processed", "uica")
LC = C.p2("data", "processed", "landcover")
OUT = C.p2("data", "processed", "features_v2")
os.makedirs(OUT, exist_ok=True)

HAND_CHANNEL_KM2 = 1.0
HAND_SENSITIVITY_KM2 = [0.25, 1.0, 5.0, 25.0]
EVENT_DAYS, ANTECEDENT_DAYS = 5, 30
YEARS = list(range(2017, 2026))
STEP3_JSON = f"{C.METRICS_DIR}/step3_flood_inventory_events.json"
SAMPLE_N = 200_000
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare",
              "snow_and_ice"]
DW_COLORS = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635", "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]

C.banner("PHASE 2 STEP 5: FEATURE STACK v2")
t_start = time.time()
res = {"step": 5, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "hand_channel_km2": HAND_CHANNEL_KM2}

with rasterio.open(C.P1_STACK) as src:
    p1 = {n: src.read(i + 1) for i, n in enumerate(src.descriptions)}
    T30 = src.transform
    H30, W30 = src.shape
with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
g = json.load(open(f"{BASIN}/ext_grid.json"))
TX = Affine(*g["transform"])
HX, WX, R0, C0 = g["height"], g["width"], g["aoi_row0"], g["aoi_col0"]
AOI = (slice(R0, R0 + H30), slice(C0, C0 + W30))
events = [e for e in json.load(open(STEP3_JSON, encoding="utf-8"))["events"] if e["status"] == "kept"]
EV = [(e["event"], e["selected"]["date"]) for e in events]
print(f"    events: {EV}")

# ---------------------------------------------------------------------------
# 5.1 / 5.3 HAND and full-basin TWI on the extended grid
# ---------------------------------------------------------------------------
print("\n[1/7] 5.1 HAND and 5.3 full-basin TWI")
with rasterio.open(f"{BASIN}/fabdem_ext.tif") as src:
    demx = src.read(1)
zf = np.load(f"{BASIN}/zf_ext.npy")
acc = np.load(f"{BASIN}/acc_ext.npy")
water_x, land_x = C.open_water_and_land(demx)
dx, dy, _ = C.row_cell_sizes_m(TX, HX)
t = time.time()
rec = C.d8_receiver(zf, land_x, dx, dy)
flat = np.flatnonzero(land_x.ravel())
order_asc = flat[np.argsort(zf.ravel()[flat], kind="stable")]
acc_km2 = acc / 1e6
hands = {}
for thr in HAND_SENSITIVITY_KM2:
    drain = land_x & (acc_km2 >= thr)
    h, _ = C.hand_from_receivers(demx.astype(np.float64), land_x, drain, rec, order_asc)
    hands[thr] = h[AOI].astype(np.float32)
print(f"    HAND for {len(HAND_SENSITIVITY_KM2)} thresholds in {time.time() - t:.0f} s")
p, q = C.horn_gradient(demx, dx, dy)
twi_full = C.twi_from_acc(acc, p, q, dx, dy)[AOI].astype(np.float32)
del zf, acc, p, q, rec, order_asc, demx
hand = hands[HAND_CHANNEL_KM2]
nan_hand = int((land & ~np.isfinite(hand)).sum())
print(f"    HAND land cells without a drainage cell: {nan_hand}")
hand = np.where(land & ~np.isfinite(hand), 0.0, hand).astype(np.float32)

twi_p1 = p1["twi"]
d_twi = np.where(land, twi_full - twi_p1, np.nan)
res["twi_full"] = {"cells_changed_share": round(float((np.abs(d_twi[land]) > 1e-4).mean()), 4),
                   "cells_increase_gt_1_share": round(float((d_twi[land] > 1).mean()), 4),
                   "max_increase": round(float(np.nanmax(d_twi)), 3),
                   "median_land": round(float(np.median(twi_full[land])), 3),
                   "phase1_median_land": round(float(np.median(twi_p1[land])), 3)}
print(f"    TWI full vs Phase 1: {res['twi_full']}")

# single-feature AUC on the Phase 1 2018 training samples (diagnostic only)
df = pd.read_csv(C.P1_TRAINING_2018)
rr, cc, yl = df["row"].to_numpy(), df["col"].to_numpy(), df["label"].to_numpy()
auc = {"elevation_m": roc_auc_score(yl, -p1["elevation_m"][rr, cc]),
       "twi_phase1": roc_auc_score(yl, p1["twi"][rr, cc]), "twi_full": roc_auc_score(yl, twi_full[rr, cc])}
for thr, h in hands.items():
    hv = np.nan_to_num(h[rr, cc], nan=0.0)
    auc[f"hand_{thr:g}km2"] = roc_auc_score(yl, -hv)
e95 = float(np.percentile(df.loc[df["label"] == 1, "elevation_m"], 95))
hard = ((df["elevation_m"] <= e95) & (df["slope_deg"] <= 2)).to_numpy()
auc_hard = {"elevation_m": roc_auc_score(yl[hard], -p1["elevation_m"][rr, cc][hard]),
            f"hand_{HAND_CHANNEL_KM2:g}km2": roc_auc_score(yl[hard], -hand[rr, cc][hard]),
            "twi_full": roc_auc_score(yl[hard], twi_full[rr, cc][hard])}
res["single_feature_auc_2018_samples"] = {k: round(float(v), 4) for k, v in auc.items()}
res["single_feature_auc_2018_floodplain_subset"] = {k: round(float(v), 4) for k, v in auc_hard.items()}
res["hand_land_cells_without_drainage_set_to_0"] = nan_hand
print(f"    single-feature AUC (2018 samples): {res['single_feature_auc_2018_samples']}")
print(f"    ... floodplain-only subset: {res['single_feature_auc_2018_floodplain_subset']}")

# ---------------------------------------------------------------------------
# 5.2 Event rainfall (CHIRPS daily)
# ---------------------------------------------------------------------------
print("\n[2/7] 5.2 Event and antecedent rainfall (CHIRPS daily)")
ee = C.ee_init()
CH = 0.05
cx0 = int(np.floor((C.BBOX[0] - 0.2 + 180) / CH)); cx1 = int(np.ceil((C.BBOX[2] + 0.2 + 180) / CH))
cy0 = int(np.floor((50 - (C.BBOX[3] + 0.2)) / CH)); cy1 = int(np.ceil((50 - (C.BBOX[1] - 0.2)) / CH))
TCH = Affine(CH, 0, -180 + cx0 * CH, 0, -CH, 50 - cy0 * CH)
WCH, HCH = cx1 - cx0, cy1 - cy0
lon_c = TCH.c + (np.arange(WCH) + 0.5) * CH
lat_c = TCH.f - (np.arange(HCH) + 0.5) * CH
lon30 = T30.c + (np.arange(W30) + 0.5) * T30.a
lat30 = T30.f + (np.arange(H30) + 0.5) * T30.e
LL = np.stack(np.meshgrid(lat30, lon30, indexing="ij"), -1)
rain = {}
from datetime import datetime, timedelta
for ev, d in EV:
    d1 = datetime.strptime(d, "%Y-%m-%d")
    win = {"event": (d1 - timedelta(days=EVENT_DAYS - 1), d1),
           "antecedent": (d1 - timedelta(days=EVENT_DAYS + ANTECEDENT_DAYS - 1), d1 - timedelta(days=EVENT_DAYS))}
    for kind, (a, b) in win.items():
        path = f"{C.RAW_DIR}/chirps_{kind}_{d.replace('-', '')}.tif"
        if not os.path.exists(path):
            img = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(a.strftime("%Y-%m-%d"),
                                                                         (b + timedelta(days=1)).strftime("%Y-%m-%d"))
                   .sum().unmask(-1).toFloat().rename("mm"))
            C.ee_download_grid(img, TCH, WCH, HCH, path)
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
        sea = arr < 0
        if sea.any():                                   # nearest land value into ocean cells
            idx = ndimage.distance_transform_edt(sea, return_distances=False, return_indices=True)
            arr = arr[tuple(idx)]
        f = RegularGridInterpolator((lat_c[::-1], lon_c), arr[::-1], method="linear")
        rain[(ev, kind)] = f(LL).astype(np.float32)
    ev_mm, an_mm = rain[(ev, "event")], rain[(ev, "antecedent")]
    res.setdefault("rainfall", {})[ev] = {
        "s1_date": d, "event_window": [f"{win['event'][0]:%Y-%m-%d}", d],
        "antecedent_window": [f"{win['antecedent'][0]:%Y-%m-%d}", f"{win['antecedent'][1]:%Y-%m-%d}"],
        "event_mm_aoi_mean": round(float(ev_mm[land].mean()), 1), "event_mm_range": [round(float(ev_mm[land].min()), 1), round(float(ev_mm[land].max()), 1)],
        "antecedent_mm_aoi_mean": round(float(an_mm[land].mean()), 1),
        "antecedent_mm_range": [round(float(an_mm[land].min()), 1), round(float(an_mm[land].max()), 1)]}
    print(f"    {ev}: {EVENT_DAYS}-day {res['rainfall'][ev]['event_mm_aoi_mean']} mm "
          f"(range {res['rainfall'][ev]['event_mm_range']}), {ANTECEDENT_DAYS}-day antecedent "
          f"{res['rainfall'][ev]['antecedent_mm_aoi_mean']} mm")

# ---------------------------------------------------------------------------
# 5.3 Land-cover class per event year (Dynamic World annual mode, 30 m majority)
# ---------------------------------------------------------------------------
print("\n[3/7] 5.3 Land-cover class per event year (Dynamic World)")
region = ee.Geometry.Rectangle(C.BBOX)
utm10 = ee.Projection("EPSG:32643").atScale(10)
lcls = {}
for ev, d in EV:
    y = int(d[:4])
    path = f"{LC}/dw_class_30m_{y}.tif"
    if not os.path.exists(path):
        img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
               .filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("label").mode().reproject(utm10)
               .reduceResolution(ee.Reducer.mode(), maxPixels=256).unmask(255).toByte().rename("dw_class"))
        C.ee_download_grid(img, T30, W30, H30, path, tiles=(2, 2), workers=4)
    with rasterio.open(path) as src:
        lcls[y] = src.read(1)
    shares = {DW_CLASSES[k]: round(float((lcls[y][land] == k).mean()), 4) for k in range(9)}
    res.setdefault("landcover_share_on_land", {})[y] = shares
    print(f"    {y}: trees {shares['trees']:.2f}, built {shares['built']:.2f}, crops {shares['crops']:.2f}, "
          f"water {shares['water']:.2f}, nodata {float((lcls[y][land] == 255).mean()):.4f}")

# ---------------------------------------------------------------------------
# 5.5 Feature stack v2 per event
# ---------------------------------------------------------------------------
print("\n[4/7] 5.5 Feature stack v2 per event year")
with rasterio.open(f"{LC}/impervious_fraction_30m.tif") as src:
    imp = {y: src.read(i + 1) for i, y in enumerate(YEARS)}
with rasterio.open(f"{UO}/uica_km2_aoi.tif") as src:
    uica = {y: src.read(i + 1) for i, y in enumerate(YEARS)}
with rasterio.open(f"{UO}/uica_ratio_aoi.tif") as src:
    uratio = {y: src.read(i + 1) for i, y in enumerate(YEARS)}
with rasterio.open(f"{UO}/upslope_area_full_km2_aoi.tif") as src:
    upa = src.read(1)

STATIC = ["elevation_m", "slope_deg", "aspect_deg", "curvature_plan", "curvature_profile", "dist_to_river_m",
          "dist_to_major_river_m", "dist_to_road_m", "drainage_density_km_km2"]
FEATURES = STATIC + ["twi_full", "hand_m", "rain_event_5d_mm", "rain_antecedent_30d_mm",
                     "impervious_frac", "uica_km2", "uica_ratio", "duica_since_2017_km2", "landcover_class"]
GROUPS = {"terrain/hydrology (Phase 1)": STATIC, "terrain/hydrology (new)": ["twi_full", "hand_m"],
          "event rainfall (new)": ["rain_event_5d_mm", "rain_antecedent_30d_mm"],
          "development (new)": ["impervious_frac", "uica_km2", "uica_ratio", "duica_since_2017_km2", "landcover_class"]}
RANGES = {"elevation_m": (-20, 2700), "slope_deg": (0, 90), "aspect_deg": (-1, 360), "curvature_plan": (-1e3, 1e3),
          "curvature_profile": (-1e3, 1e3), "dist_to_river_m": (0, 2e4), "dist_to_major_river_m": (0, 5e4),
          "dist_to_road_m": (0, 2e4), "drainage_density_km_km2": (0, 20), "twi_full": (-5, 40), "hand_m": (0, 2700),
          "rain_event_5d_mm": (0, 2000), "rain_antecedent_30d_mm": (0, 5000), "impervious_frac": (0, 1),
          "uica_km2": (0, 1e4), "uica_ratio": (0, 1), "duica_since_2017_km2": (-1e3, 1e3), "landcover_class": (0, 8)}
checks, stacks = {}, {}
prof = {"driver": "GTiff", "height": H30, "width": W30, "count": len(FEATURES), "dtype": "float32",
        "crs": "EPSG:4326", "transform": T30, "nodata": np.nan, "compress": "deflate", "predictor": 3,
        "tiled": True, "blockxsize": 256, "blockysize": 256}
for ev, d in EV:
    y = int(d[:4])
    lc = lcls[y].astype(np.float32)
    lc[lc == 255] = np.nan
    if np.isnan(lc[land]).any():                       # nearest valid class for rare gaps
        idx = ndimage.distance_transform_edt(np.isnan(lc), return_distances=False, return_indices=True)
        lc = lc[tuple(idx)]
    layers = {**{n: p1[n] for n in STATIC}, "twi_full": twi_full, "hand_m": hand,
              "rain_event_5d_mm": rain[(ev, "event")], "rain_antecedent_30d_mm": rain[(ev, "antecedent")],
              "impervious_frac": imp[y], "uica_km2": uica[y], "uica_ratio": uratio[y],
              "duica_since_2017_km2": uica[y] - uica[2017], "landcover_class": lc}
    path = f"{OUT}/stack_v2_{ev}.tif"
    with rasterio.open(path, "w", **prof) as dst:
        for i, n in enumerate(FEATURES, 1):
            dst.write(np.where(land, layers[n], np.nan).astype(np.float32), i)
            dst.set_band_description(i, n)
    gaps = {n: int((land & ~np.isfinite(layers[n])).sum()) for n in FEATURES}
    out_rng = {n: int((land & np.isfinite(layers[n]) & ((layers[n] < RANGES[n][0]) | (layers[n] > RANGES[n][1]))).sum())
               for n in FEATURES}
    with rasterio.open(path) as src:
        same_grid = src.transform == T30 and src.shape == (H30, W30)
    checks[ev] = {"same_grid_as_phase1": bool(same_grid), "land_gaps": {k: v for k, v in gaps.items() if v},
                  "out_of_range": {k: v for k, v in out_rng.items() if v},
                  "pass": bool(same_grid and not any(gaps.values()) and not any(out_rng.values()))}
    stacks[ev] = layers
    print(f"    {ev}: {path.split(os.sep)[-1]}  grid ok {same_grid}, gaps {checks[ev]['land_gaps'] or 0}, "
          f"out of range {checks[ev]['out_of_range'] or 0}")
res["stack_v2"] = {"features": FEATURES, "groups": GROUPS, "events": [e for e, _ in EV], "checks": checks,
                   "replaced": {"twi": "twi_full", "rainfall_mean_annual_mm": "rain_event_5d_mm + rain_antecedent_30d_mm"},
                   "categorical": ["landcover_class"], "landcover_codes": dict(enumerate(DW_CLASSES))}

# correlation and VIF (numeric features, pooled sample over events)
print("\n[5/7] Correlation and VIF")
rng = np.random.default_rng(C.SEED)
num = [f for f in FEATURES if f != "landcover_class"]
li = np.flatnonzero(land.ravel())
Xs = []
for ev, _ in EV:
    idx = rng.choice(li, SAMPLE_N // len(EV), replace=False)
    Xs.append(np.column_stack([stacks[ev][n].ravel()[idx] for n in num]))
X = np.vstack(Xs).astype(np.float64)
Xt = X.copy()
for j, n in enumerate(num):                            # log for heavy-tailed areas (as used by trees, rank-equivalent)
    if n in ("uica_km2",):
        Xt[:, j] = np.log10(np.maximum(X[:, j], 1e-6))
corr = np.corrcoef(Xt, rowvar=False)
vif = np.diag(np.linalg.inv(corr))
res["vif"] = {n: round(float(v), 2) for n, v in zip(num, vif)}
res["max_vif"] = round(float(vif.max()), 2)
pairs = [(num[i], num[j], round(float(corr[i, j]), 3)) for i in range(len(num)) for j in range(i + 1, len(num))
         if abs(corr[i, j]) >= 0.7]
res["high_correlation_pairs_abs_ge_0.7"] = pairs
print(f"    max VIF {res['max_vif']} ({num[int(vif.argmax())]}); |r| >= 0.7 pairs: {pairs}")

# UICA physical sanity (the Step 4 checkpoint question)
print("\n[6/7] UICA physical sanity")
y = int(EV[-1][1][:4])
boxes = {"Kochi city core": (76.265, 9.955, 76.32, 10.02), "Eloor-Kalamassery (below Aluva)": (76.28, 10.04, 76.33, 10.09),
         "Eastern forested hills": (76.47, 10.15, 76.55, 10.25)}
san = {}
for name, (x0, y0, x1, y1) in boxes.items():
    c0, r0 = ~T30 * (x0, y1); c1, r1 = ~T30 * (x1, y0)
    w = (slice(int(r0), int(r1)), slice(int(c0), int(c1)))
    m = land[w]
    san[name] = {"uica_ratio_median": round(float(np.median(uratio[y][w][m])), 4),
                 "uica_km2_p90": round(float(np.percentile(uica[y][w][m], 90)), 3),
                 "impervious_frac_mean": round(float(imp[y][w][m].mean()), 4),
                 "trees_share": round(float((lcls[y][w][m] == 1).mean()), 3)}
by_class = {DW_CLASSES[k]: round(float(np.median(uratio[y][land & (lcls[y] == k)])), 4)
            for k in (1, 4, 6) if (land & (lcls[y] == k)).sum() > 100}
res["uica_sanity"] = {"year": y, "boxes": san, "uica_ratio_median_by_landcover": by_class}
for k, v in san.items():
    print(f"    {k:34s} {v}")
print(f"    UICA ratio median by land cover: {by_class}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[7/7] Figures")
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])


def show(ax, arr, title, cmap, norm=None, vmin=None, vmax=None, lab=""):
    im = ax.imshow(np.where(land, arr, np.nan), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, extent=ext,
                   interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, label=lab)


fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=105)
show(axes[0, 0], twi_p1, "TWI, Phase 1 (upstream cut at AOI edge)", "Blues", vmin=2, vmax=20, lab="TWI")
show(axes[0, 1], twi_full, "TWI, full basin (Step 4 routing)", "Blues", vmin=2, vmax=20, lab="TWI")
show(axes[0, 2], d_twi, "TWI change (full − Phase 1)", "RdBu_r", vmin=-4, vmax=4, lab="Δ TWI")
show(axes[1, 0], p1["elevation_m"], "Elevation (FABDEM)", "terrain", vmin=-5, vmax=150, lab="m")
show(axes[1, 1], hand, f"HAND (drainage ≥ {HAND_CHANNEL_KM2:g} km²)", "YlGnBu_r", vmin=0, vmax=40, lab="m above drainage")
ax = axes[1, 2]
keys = ["elevation_m", "twi_phase1", "twi_full"] + [f"hand_{t:g}km2" for t in HAND_SENSITIVITY_KM2]
vals = [auc[k] for k in keys]
cols = [C.C_GREY, "#9aa4ad", C.C_BLUE] + [C.C_ORANGE if t == HAND_CHANNEL_KM2 else "#f3b08e" for t in HAND_SENSITIVITY_KM2]
ax.barh(range(len(keys))[::-1], vals, color=cols)
ax.set_yticks(range(len(keys))[::-1], [k.replace("_", " ") for k in keys], fontsize=9)
for i, v in enumerate(vals):
    ax.text(v + 0.003, len(keys) - 1 - i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("single-feature AUC on 2018 samples (diagnostic)")
ax.set_title("Which terrain feature separates flood / non-flood alone?", loc="left", fontweight="bold", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Step 5.1 / 5.3 — HAND and full-basin TWI", fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{C.MAP_DIR}/step5_twi_hand.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, len(EV), figsize=(4.8 * len(EV), 9), dpi=105)
vmax_e = max(float(rain[(e, 'event')][land].max()) for e, _ in EV)
vmax_a = max(float(rain[(e, 'antecedent')][land].max()) for e, _ in EV)
for j, (ev, d) in enumerate(EV):
    show(axes[0, j], rain[(ev, "event")], f"{ev}: {EVENT_DAYS}-day total to {d}", "YlGnBu", vmin=0, vmax=vmax_e, lab="mm")
    show(axes[1, j], rain[(ev, "antecedent")], f"{ev}: {ANTECEDENT_DAYS}-day antecedent", "YlGnBu", vmin=0,
         vmax=vmax_a, lab="mm")
fig.suptitle("Step 5.2 — event and antecedent rainfall per flood event (CHIRPS daily)", fontsize=14,
             fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{C.MAP_DIR}/step5_event_rainfall.png", bbox_inches="tight")
plt.close(fig)

ev_last = EV[-1][0]
L = stacks[ev_last]
fig, axes = plt.subplots(3, 6, figsize=(24, 12.5), dpi=95)
for ax, n in zip(axes.ravel(), FEATURES):
    a = L[n]
    if n == "landcover_class":
        im = ax.imshow(np.where(land, a, np.nan), cmap=ListedColormap(DW_COLORS),
                       norm=BoundaryNorm(np.arange(-0.5, 9.5), 9), extent=ext, interpolation="nearest")
    elif n == "uica_km2":
        im = ax.imshow(np.where(land, np.maximum(a, 1e-4), np.nan), cmap="magma_r", norm=LogNorm(1e-4, 100),
                       extent=ext, interpolation="nearest")
    else:
        lo, hi = np.nanpercentile(a[land], [2, 98])
        cm = "RdBu_r" if n.startswith(("curvature", "duica")) else "viridis"
        if cm == "RdBu_r":
            v = max(abs(lo), abs(hi)); lo, hi = -v, v
        im = ax.imshow(np.where(land, a, np.nan), cmap=cm, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    new = n not in STATIC
    ax.set_title(("★ " if new else "") + n, loc="left", fontsize=10, fontweight="bold",
                 color=C.C_ORANGE if new else "black")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045)
fig.suptitle(f"Step 5.5 — feature stack v2 for the {ev_last} event ({len(FEATURES)} features, ★ = new in Phase 2)",
             fontsize=15, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{C.MAP_DIR}/step5_stack_v2_overview.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(19, 8.2), dpi=105, gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num)), num, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(num)), num, fontsize=8)
for i in range(len(num)):
    for j in range(len(num)):
        if i != j and abs(corr[i, j]) >= 0.5:
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr[i, j]) > 0.75 else "black")
plt.colorbar(im, ax=ax, fraction=0.04, label="Pearson r (log10 UICA)")
ax.set_title("Correlation between v2 features (|r| ≥ 0.5 labelled)", loc="left", fontweight="bold")
ax = axes[1]
o = np.argsort(vif)
ax.barh(range(len(num)), vif[o], color=[C.C_ORANGE if num[i] not in STATIC else C.C_BLUE for i in o])
ax.set_yticks(range(len(num)), [num[i] for i in o], fontsize=8)
ax.axvline(5, color="k", ls=":", lw=1)
ax.axvline(10, color="k", ls="--", lw=1)
for i, k in enumerate(o):
    ax.text(vif[k] + 0.1, i, f"{vif[k]:.1f}", va="center", fontsize=8)
ax.set_xlabel("VIF (5 = moderate, 10 = serious collinearity)")
ax.set_title(f"Variance inflation factor (max {vif.max():.1f}); orange = new in v2", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{C.MAP_DIR}/step5_correlation_vif.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=105)
show(axes[0], np.maximum(uica[y], 1e-4), f"UICA {y} (log)", "magma_r", norm=LogNorm(1e-4, 100), lab="km²")
show(axes[1], uratio[y], f"UICA ratio {y}", "viridis_r", vmin=0, vmax=0.5, lab="paved share of upstream area")
for ax in axes[:2]:
    for name, (x0, y0, x1, y1) in boxes.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=C.C_GREEN, lw=1.6))
        ax.text(x0, y1 + 0.004, name, color=C.C_GREEN, fontsize=7.5, fontweight="bold")
ax = axes[2]
names = list(san)
ax.bar(range(len(names)), [san[n]["uica_ratio_median"] for n in names], color=[C.C_ORANGE, C.C_ORANGE, C.C_GREEN])
ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
for i, n in enumerate(names):
    ax.text(i, san[n]["uica_ratio_median"] + 0.005, f"{san[n]['uica_ratio_median']:.3f}", ha="center", fontsize=9)
ax.set_ylabel("median UICA ratio")
ax.set_title("Is UICA physically right? built-up vs forest", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Step 4 checkpoint — UICA sanity in {y}", fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{C.MAP_DIR}/step5_uica_sanity.png", bbox_inches="tight")
plt.close(fig)

meta = {"features": FEATURES, "groups": GROUPS, "events": dict(EV), "hand_channel_km2": HAND_CHANNEL_KM2,
        "rain_windows_days": {"event": EVENT_DAYS, "antecedent": ANTECEDENT_DAYS},
        "landcover_codes": dict(enumerate(DW_CLASSES)), "files": {ev: f"stack_v2_{ev}.tif" for ev, _ in EV}}
C.save_json(meta, f"{OUT}/features_v2_metadata.json")
res["all_checks_pass"] = bool(all(c["pass"] for c in checks.values()))
res["runtime_s"] = round(time.time() - t_start, 1)
C.save_json(res, f"{C.METRICS_DIR}/step5_feature_stack_v2.json")
print(f"    done in {res['runtime_s']} s; all stack checks pass: {res['all_checks_pass']}")
