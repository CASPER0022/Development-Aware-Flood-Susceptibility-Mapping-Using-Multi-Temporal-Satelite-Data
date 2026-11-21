"""
Phase 2 Step 9.0c - Physics effect size of development: extra runoff (label-free).

Step 9.0b showed the ML model cannot supply the size of the development effect
(its constrained response to paving is flat). This step computes it from
physics, with no flood labels:

1. SCS curve number (USDA TR-55) runoff for one storm, cell by cell:
       CN  = f_imp * 98 + (1 - f_imp) * CN_pervious(soil group)
       S   = 25400 / CN - 254                         (mm)
       Q   = (P - 0.2 S)^2 / (P + 0.8 S) if P > 0.2 S else 0   (mm)
   Monsoon: antecedent moisture condition III (wet soil),
       CN_III = 23 CN / (10 + 0.13 CN).
   Pervious CN: TR-55 "woods-grass combination, fair" (B 65, C 76, D 82).
   Soil group: no usable basin-wide soil map (Phase 1: NBSS&LUP not
   downloadable, OpenLandMap 250 m has only a handful of classes over the AOI),
   so one group for the whole basin. Central case C (laterite / clay loam),
   B and D as sensitivity.
2. Paved fraction for 2017 and 2025 on the full upstream grid (Step 4).
3. Extra runoff depth dQ = Q_2025 - Q_2017 per cell. Runoff volumes are routed
   with the Step 4 FD8 routing (same as UICA): for every cell, the volume of
   storm runoff arriving from upstream in 2017 and 2025, and the extra volume.
   Main number: % increase in upstream storm-runoff volume, 2017 -> 2025.
4. Sensitivity (no routing): basin and Kochi-core extra runoff for soil group
   B / C / D, storm 100 / 200 / 400 mm, wet (III) / normal (II) soil.

Two routed cases, soil group C:
  monsoon   P = 200 mm, wet soil (AMC III): the big riverine floods
            (CHIRPS 5-day event totals were 155-517 mm)
  frequent  P = 50 mm, normal soil (AMC II): the frequent storms behind
            urban waterlogging, where paving matters most

Outputs: data/processed/runoff/extra_runoff_aoi.tif (gitignored)
         outputs/metrics/step9c_extra_runoff.json
         outputs/maps/step9c_extra_runoff.png
"""
import gc
import json
import os
import time

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

P_MM = 200.0
HSG = "C"
CASES = {"monsoon": (200.0, True), "frequent": (50.0, False)}   # storm mm, wet soil (AMC III)
CASE_NAME = {"monsoon": "Monsoon flood storm: 200 mm on wet soil", "frequent": "Frequent storm: 50 mm on normal soil"}
CN_PERV = {"B": 65.0, "C": 76.0, "D": 82.0}
CN_IMP = 98.0
YEARS = (2017, 2025)
BASIN = f"{C.PROC_DIR}/basin"
OUT = f"{C.PROC_DIR}/runoff"
os.makedirs(OUT, exist_ok=True)
C.banner("PHASE 2 STEP 9.0c: EXTRA STORM RUNOFF FROM DEVELOPMENT (SCS-CN + FD8)")
t0 = time.time()
res = {"step": "9.0c", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "soil_group": HSG,
       "cases": {k: {"storm_mm": v[0], "amc": "III" if v[1] else "II"} for k, v in CASES.items()},
       "cn_pervious": CN_PERV, "cn_impervious": CN_IMP}


def amc3(cn):
    return 23.0 * cn / (10.0 + 0.13 * cn)


def scs_q(p, cn):
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    return np.where(p > ia, (p - ia) ** 2 / (p + 0.8 * s), 0.0).astype(np.float32)


def runoff_mm(imp, p=P_MM, hsg=HSG, wet=True):
    cn = imp * CN_IMP + (1.0 - imp) * CN_PERV[hsg]
    return scs_q(p, amc3(cn) if wet else cn)


# ---------------------------------------------------------------------------
# Inputs on the extended (full-basin) grid
# ---------------------------------------------------------------------------
print("\n[1/4] Inputs")
g = json.load(open(f"{BASIN}/ext_grid.json", encoding="utf-8"))
HX, WX = g["height"], g["width"]
with rasterio.open(f"{BASIN}/fabdem_ext.tif") as src:
    TX = src.transform
    dem = src.read(1)
_, land = C.open_water_and_land(dem)
with rasterio.open(C.P1_DOMAIN) as src:
    land_aoi = src.read(1).astype(bool)
    T30, H30, W30 = src.transform, src.height, src.width
R0, C0 = g["aoi_row0"], g["aoi_col0"]
AOI = (slice(R0, R0 + H30), slice(C0, C0 + W30))
del dem
dx, dy, cell_area = C.row_cell_sizes_m(TX, HX)
with rasterio.open(f"{C.PROC_DIR}/landcover/impervious_fraction_ext_30m.tif") as src:
    names = list(src.descriptions)
    imp = {y: src.read(names.index(f"impervious_{y}_x200") + 1).astype(np.float32) / 200.0 for y in YEARS}
for y in YEARS:
    imp[y][~land] = 0.0
area_m2 = np.broadcast_to(cell_area[:, None].astype(np.float32), (HX, WX))
import geopandas as gpd
from rasterio.features import rasterize
cat = gpd.read_file(f"{BASIN}/upstream_catchment.geojson").to_crs("EPSG:4326")
catch = rasterize([(gg, 1) for gg in cat.geometry], out_shape=(HX, WX), transform=TX, fill=0, dtype="uint8").astype(bool)
catch |= np.zeros_like(catch)
catch[AOI] |= land_aoi
catch &= land
res["catchment_land_km2"] = round(float((catch * area_m2).sum() / 1e6), 1)
res["paved_km2"] = {y: round(float((imp[y] * catch * area_m2).sum() / 1e6), 1) for y in YEARS}
print(f"    catchment + AOI land {res['catchment_land_km2']} km2; paved km2 {res['paved_km2']}")

# ---------------------------------------------------------------------------
# Runoff, catchment totals and routing for each case
# ---------------------------------------------------------------------------
print("\n[2/4] SCS-CN runoff and FD8 routing")
zf = np.load(f"{BASIN}/zf_ext.npy")
flat = np.flatnonzero(land.ravel())
order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
del flat
dq, pct, totals = {}, {}, {}
for case, (p_mm, wet) in CASES.items():
    q = {y: runoff_mm(imp[y], p_mm, HSG, wet) for y in YEARS}
    dq[case] = (q[2025] - q[2017])[AOI].copy()
    vol = {y: float((q[y] * catch * area_m2).sum() / 1e3) for y in YEARS}
    totals[case] = {"runoff_Mm3_2017": round(vol[2017] / 1e6, 2), "runoff_Mm3_2025": round(vol[2025] / 1e6, 2),
                    "extra_Mm3": round((vol[2025] - vol[2017]) / 1e6, 3),
                    "extra_pct": round(100 * (vol[2025] / vol[2017] - 1), 3),
                    "runoff_coefficient_2017": round(float(q[2017][catch].mean() / p_mm), 3),
                    "runoff_coefficient_unpaved": round(float(runoff_mm(np.float32(0), p_mm, HSG, wet) / p_mm), 3),
                    "runoff_coefficient_fully_paved": round(float(runoff_mm(np.float32(1), p_mm, HSG, wet) / p_mm), 3)}
    up = {}
    for y in YEARS:
        w = (q[y] * area_m2 / 1e3).astype(np.float64)
        acc, *_ = C.quinn_fd8_weighted(zf, land, order, w, dx, dy)
        up[y] = acc[AOI].astype(np.float32)
        del acc, w
        gc.collect()
    del q
    dV = up[2025] - up[2017]
    pc = np.where(up[2017] > 0, 100 * dV / np.maximum(up[2017], 1e-6), 0.0).astype(np.float32)
    pc[~land_aoi] = np.nan
    pct[case] = pc
    with rasterio.open(f"{OUT}/extra_runoff_{case}_aoi.tif", "w", driver="GTiff", height=H30, width=W30, count=4,
                       dtype="float32", crs="EPSG:4326", transform=T30, nodata=np.nan, compress="deflate") as dst:
        for k, (a, n) in enumerate(((up[2017], "upstream_runoff_m3_2017"), (up[2025], "upstream_runoff_m3_2025"),
                                    (dV, "extra_upstream_runoff_m3"), (pc, "extra_upstream_runoff_pct")), 1):
            dst.write(np.where(land_aoi, a, np.nan).astype(np.float32), k)
            dst.set_band_description(k, n)
    print(f"    {case}: {totals[case]}")
del zf, order
gc.collect()
res["catchment_totals"] = totals

# summaries in the AOI
with rasterio.open(f"{C.PROC_DIR}/uica/uica_ratio_aoi.tif") as src:
    ur = src.read(src.count)
rows, cols = np.nonzero(land_aoi)
lon = T30.c + (cols + 0.5) * T30.a
lat = T30.f + (rows + 0.5) * T30.e
boxes = {"Kochi city core": (76.265, 9.955, 76.32, 10.02),
         "Eloor-Kalamassery (below Aluva)": (76.28, 10.04, 76.33, 10.09),
         "Eastern forested hills": (76.47, 10.15, 76.55, 10.25)}
acc_up = np.load(f"{BASIN}/acc_ext.npy", mmap_mode="r")[AOI]
ri = np.unravel_index(np.nanargmax(np.where(land_aoi, acc_up, -1)), acc_up.shape)
urb = ur[rows, cols] >= 0.10
with rasterio.open(f"{C.PROC_DIR}/uica/duica_km2_aoi.tif") as src:
    du = src.read(src.count)[rows, cols]
grow = du >= np.nanpercentile(du, 99)
res["aoi"] = {}
for case in CASES:
    pv = pct[case][rows, cols]
    dl = dq[case][rows, cols]
    places = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        places[name] = {"median_pct": round(float(np.nanmedian(pv[m])), 2),
                        "p90_pct": round(float(np.nanpercentile(pv[m], 90)), 2),
                        "local_extra_mm_p90": round(float(np.percentile(dl[m], 90)), 2)}
    places["Periyar main stem (largest catchment in AOI)"] = {
        "upslope_km2": round(float(acc_up[ri] / 1e6), 0), "pct": round(float(pct[case][ri]), 3)}
    res["aoi"][case] = {
        "median_pct_all_land": round(float(np.nanmedian(pv)), 3),
        "p99_pct": round(float(np.nanpercentile(pv, 99)), 2), "max_pct": round(float(np.nanmax(pv)), 2),
        "median_pct_uica_ratio_ge_0.10": round(float(np.nanmedian(pv[urb])), 3),
        "median_pct_top1pct_duica_cells": round(float(np.nanmedian(pv[grow])), 2),
        "share_land_pct_ge_1": round(float(np.nanmean(pv >= 1)), 4),
        "share_land_pct_ge_5": round(float(np.nanmean(pv >= 5)), 4),
        "local_extra_mm_max": round(float(dl.max()), 1), "local_extra_mm_p99": round(float(np.percentile(dl, 99)), 2),
        "places": places}
    print(f"    {case}: {res['aoi'][case]}")

# ---------------------------------------------------------------------------
# Sensitivity (local, no routing)
# ---------------------------------------------------------------------------
print("\n[4/4] Sensitivity: soil group x storm x soil wetness")
km = (lon >= boxes["Kochi city core"][0]) & (lon <= boxes["Kochi city core"][2]) & \
     (lat >= boxes["Kochi city core"][1]) & (lat <= boxes["Kochi city core"][3])
dimp = (imp[2025] - imp[2017])[AOI][rows, cols]
gm = dimp >= np.percentile(dimp, 99)                       # cells paved the most themselves, 2017 -> 2025
res["local_growth_cells"] = {"n": int(gm.sum()), "paved_2017_mean": round(float(imp[2017][AOI][rows[gm], cols[gm]].mean()), 3),
                             "paved_2025_mean": round(float(imp[2025][AOI][rows[gm], cols[gm]].mean()), 3)}
ia17, ia25 = imp[2017][AOI][rows[gm], cols[gm]], imp[2025][AOI][rows[gm], cols[gm]]
sens = []
landf = catch.astype(np.float32) * area_m2
for hsg in ("B", "C", "D"):
    for p in (50.0, 100.0, 200.0, 400.0):
        for wet in (True, False):
            v17 = float((runoff_mm(imp[2017], p, hsg, wet) * landf).sum())
            v25 = float((runoff_mm(imp[2025], p, hsg, wet) * landf).sum())
            k17 = runoff_mm(ia17, p, hsg, wet); k25 = runoff_mm(ia25, p, hsg, wet)
            sens.append({"soil_group": hsg, "storm_mm": p, "amc": "III" if wet else "II",
                         "basin_extra_pct": round(100 * (v25 / v17 - 1), 3) if v17 > 0 else None,
                         "growth_cells_runoff_coef_2017": round(float(k17.mean() / p), 3),
                         "growth_cells_local_extra_pct": round(100 * float(k25.sum() / max(k17.sum(), 1e-9) - 1), 2)})
for s in sens:
    print(f"    {s}")
res["sensitivity"] = sens

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
fig = plt.figure(figsize=(17, 5.2), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
vmax = max(1.0, float(np.nanpercentile(pct["frequent"][rows, cols], 99.5)))
for k, case in enumerate(CASES):
    ax = fig.add_subplot(gs[k])
    im = ax.imshow(pct[case], cmap="magma_r", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
    for name, (x0, y0, x1, y1) in boxes.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=C.C_BLUE, lw=1))
    t = res["catchment_totals"][case]
    ax.set_title(f"{CASE_NAME[case]}\nextra runoff arriving from upstream, 2017 → 2025 (%)\n"
                 f"whole catchment +{t['extra_pct']:.2f} %; runoff coefficient unpaved {t['runoff_coefficient_unpaved']:.2f} "
                 f"vs paved {t['runoff_coefficient_fully_paved']:.2f}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.72, label="% more runoff than in 2017")
ax = fig.add_subplot(gs[2])
labels, vals = [], []
for p in (50.0, 100.0, 200.0, 400.0):
    for wet in (False, True):
        sx = next(x for x in sens if x["soil_group"] == "C" and x["storm_mm"] == p and x["amc"] == ("III" if wet else "II"))
        labels.append(f"{p:.0f} mm, {'wet' if wet else 'normal'} soil"); vals.append(sx["growth_cells_local_extra_pct"])
ax.barh(labels[::-1], vals[::-1], color=[C.C_ORANGE if "wet" not in l else C.C_BLUE for l in labels[::-1]])
for i_, v in enumerate(vals[::-1]):
    ax.text(v, i_, f" {v:.0f} %", va="center", fontsize=8)
ax.set_xlabel("extra runoff 2017 → 2025 from the 1 % of cells paved the most (%)")
ax.set_title("Paving matters for small storms on normal soil,\nlittle for big storms on soaked soil (soil group C)")
fig.suptitle("Step 9.0c  Physics effect size of development (SCS curve number, routed with FD8); no flood labels used",
             fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step9c_extra_runoff.png", dpi=140)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step9c_extra_runoff.json")
print(f"\nDone in {res['runtime_s']} s")
