"""
Phase 2 Step 9 - Current vs post-development: the effect of development on flooding.

Steps 9.0a-c showed: the ML model locates riverine floods but cannot size the
development effect (flat constrained response); physics (SCS-CN) shows
development adds little water in extreme monsoon storms but a lot in frequent
storms (urban waterlogging). Step 9 combines both, per scenario.

Scenarios (paved fraction on the full upstream grid):
  2017            baseline (Step 2 / Step 4 layer)
  2025            current, observed
  2030 trend      every cell keeps its own 2017 -> 2025 paving rate for 5 more years
                  (growth only): imp_2025 + 5/8 * max(imp_2025 - imp_2017, 0)
  2030 GHSL       imp_2025 + max(GHSL_2030 - GHSL_2025, 0)  (JRC GHS-BUILT-S P2023A
                  projection, independent of our own trend)
For each scenario: UICA, UICA ratio, dUICA (FD8, as Step 4) and storm runoff
(SCS-CN, as Step 9.0c), routed through the full basin.

Layer A  riverine flood susceptibility: Step 9.0b scenario model (D - land cover,
         monotone in paving / UICA), static and rainfall features of the 2018
         reference flood, development features swapped per scenario. Classes use
         FIXED breaks (quintiles of the 2025 map), so change is visible.
Layer B  waterlogging pressure: % more frequent-storm (50 mm, normal soil) runoff
         arriving from upstream than in 2017; split into local paving vs upstream.
Combined development-impact classes:
         hotspot   = flood-prone (High / Very High in A) AND >= 5 % more water (B)
         emerging  = >= 5 % more water, not flood-prone
         existing  = flood-prone, < 5 % more water
Outputs: outputs/metrics/step9_dual_prediction.json, step9_hotspot_blocks.csv
         outputs/maps/step9_development_effect.png (headline), step9_susceptibility_scenarios.png
         data/processed/scenarios/*.tif (gitignored)
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import common as C

STORM = {"frequent": (50.0, False), "monsoon": (200.0, True)}
HSG_CN_PERV, CN_IMP = 76.0, 98.0               # soil group C (Step 9.0c central case)
MORE_WATER_PCT = 5.0
BASIN = f"{C.PROC_DIR}/basin"
LC = f"{C.PROC_DIR}/landcover"
OUT = f"{C.PROC_DIR}/scenarios"
os.makedirs(OUT, exist_ok=True)
SCEN = ["2017", "2025", "2030 trend", "2030 GHSL"]
TOWNS = {"Aluva": (10.108, 76.351), "Kalamassery": (10.051, 76.316), "Eloor": (10.075, 76.300),
         "North Paravur": (10.143, 76.228), "Angamaly": (10.196, 76.386), "Perumbavoor": (10.115, 76.477),
         "Kakkanad": (10.016, 76.341), "Tripunithura": (9.945, 76.348), "Edappally": (10.025, 76.308),
         "Ernakulam (Kochi)": (9.982, 76.282), "Kolenchery": (9.980, 76.470), "Nedumbassery": (10.155, 76.390),
         "Varapuzha": (10.070, 76.270), "Kadungalloor": (10.090, 76.330), "Vypin": (10.050, 76.220),
         "Kalady": (10.166, 76.438), "Pallikkara": (10.040, 76.400)}
C.banner("PHASE 2 STEP 9: CURRENT VS POST-DEVELOPMENT, EFFECT OF DEVELOPMENT ON FLOODING")
t0 = time.time()
res = {"step": 9, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "scenarios": SCEN,
       "more_water_threshold_pct": MORE_WATER_PCT, "storms": {k: {"mm": v[0], "amc": "III" if v[1] else "II"} for k, v in STORM.items()}}


def runoff_mm(imp, p, wet):
    cn = imp * CN_IMP + (1.0 - imp) * HSG_CN_PERV
    if wet:
        cn = 23.0 * cn / (10.0 + 0.13 * cn)
    s = 25400.0 / cn - 254.0
    return np.where(p > 0.2 * s, (p - 0.2 * s) ** 2 / (p + 0.8 * s), 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Paving scenarios on the full-basin grid
# ---------------------------------------------------------------------------
print("\n[1/5] Paving scenarios")
g = json.load(open(f"{BASIN}/ext_grid.json", encoding="utf-8"))
HX, WX = g["height"], g["width"]
with rasterio.open(f"{BASIN}/fabdem_ext.tif") as src:
    TX = src.transform
    dem = src.read(1)
_, land = C.open_water_and_land(dem)
del dem
with rasterio.open(C.P1_DOMAIN) as src:
    land_aoi = src.read(1).astype(bool)
    T30, H30, W30 = src.transform, src.height, src.width
AOI = (slice(g["aoi_row0"], g["aoi_row0"] + H30), slice(g["aoi_col0"], g["aoi_col0"] + W30))
dx, dy, cell_area = C.row_cell_sizes_m(TX, HX)
area_m2 = np.broadcast_to(cell_area[:, None].astype(np.float32), (HX, WX))
with rasterio.open(f"{LC}/impervious_fraction_ext_30m.tif") as src:
    nm = list(src.descriptions)
    imp = {s: src.read(nm.index(f"impervious_{s}_x200") + 1).astype(np.float32) / 200.0 for s in ("2017", "2025")}
imp["2030 trend"] = np.clip(imp["2025"] + 5 / 8 * np.maximum(imp["2025"] - imp["2017"], 0), 0, 1)
ghsl = {}
for yr in (2025, 2030):
    pth = f"{LC}/ghsl{yr}_ext_30m.tif"
    if not os.path.exists(pth):
        import ee
        C.ee_init()
        im = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filter(ee.Filter.eq("system:index", str(yr)))
              .first().select("built_surface").divide(10000).clamp(0, 1).toFloat().rename("ghsl"))
        C.ee_download_grid(im, TX, WX, HX, pth, tiles=(2, 2), workers=4)
    with rasterio.open(pth) as src:
        ghsl[yr] = np.nan_to_num(src.read(1).astype(np.float32), nan=0.0)
imp["2030 GHSL"] = np.clip(imp["2025"] + np.maximum(ghsl[2030] - ghsl[2025], 0), 0, 1)
del ghsl
for s in SCEN:
    imp[s][~land] = 0.0
cat_km2 = lambda a: round(float((a * area_m2 * land).sum() / 1e6), 1)
res["paved_km2_full_grid"] = {s: cat_km2(imp[s]) for s in SCEN}
res["paved_km2_aoi"] = {s: round(float((imp[s][AOI] * area_m2[AOI] * land_aoi).sum() / 1e6), 1) for s in SCEN}
print(f"    paved km2, full grid {res['paved_km2_full_grid']}; AOI {res['paved_km2_aoi']}")

# ---------------------------------------------------------------------------
# 2. Routing: UICA and frequent-storm runoff per scenario
# ---------------------------------------------------------------------------
print("\n[2/5] Routing UICA and storm runoff per scenario")
zf = np.load(f"{BASIN}/zf_ext.npy")
flat = np.flatnonzero(land.ravel())
order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
del flat
acc_aoi_km2 = np.load(f"{BASIN}/acc_ext.npy", mmap_mode="r")[AOI] / 1e6


def route(w):
    a, *_ = C.quinn_fd8_weighted(zf, land, order, w.astype(np.float64), dx, dy)
    out = a[AOI].astype(np.float32)
    del a
    gc.collect()
    return out


uica, up_run, loc_run, totals = {}, {}, {}, {}
for s in SCEN:
    t1 = time.time()
    uica[s] = route(imp[s] * area_m2) / 1e6                                 # km2
    q = runoff_mm(imp[s], *STORM["frequent"])
    up_run[s] = route(q * area_m2 / 1e3)                                    # m3
    loc_run[s] = (q * area_m2 / 1e3)[AOI].copy()
    totals[s] = {k: round(float((runoff_mm(imp[s], *v) * area_m2 * land).sum() / 1e9), 4) for k, v in STORM.items()}
    del q
    print(f"    {s}: routed in {time.time() - t1:.0f} s; total storm runoff (Mm3) {totals[s]}")
del zf, order
gc.collect()
with rasterio.open(f"{C.PROC_DIR}/uica/uica_km2_aoi.tif") as src:
    u25 = src.read(list(src.descriptions).index("uica_km2_2025") + 1)
res["check_uica_2025_vs_step4_max_abs_km2"] = round(float(np.nanmax(np.abs(np.where(land_aoi, uica["2025"] - u25, 0)))), 6)
print(f"    check: UICA 2025 vs Step 4 max |diff| = {res['check_uica_2025_vs_step4_max_abs_km2']} km2")
res["catchment_storm_runoff_change_pct_vs_2017"] = {
    k: {s: round(100 * (totals[s][k] / totals["2017"][k] - 1), 3) for s in SCEN[1:]} for k in STORM}

# ---------------------------------------------------------------------------
# 3. Layer A: riverine susceptibility per scenario (constrained scenario model)
# ---------------------------------------------------------------------------
print("\n[3/5] Layer A: riverine flood susceptibility (scenario model, 2018 reference flood)")
model = lgb.Booster(model_file=f"{C.MODEL_DIR}/model_v2_scenario.txt")
feats = model.feature_name()
lm, rr, cc, _ = C.land_matrix("2018-08_ndem")
acc_c = np.maximum(acc_aoi_km2[rr, cc], 1e-6)
prob = {}
for s in SCEN:
    x = lm.copy()
    x["impervious_frac"] = imp[s][AOI][rr, cc]
    x["uica_km2"] = uica[s][rr, cc]
    x["uica_ratio"] = np.clip(uica[s][rr, cc] / acc_c, 0, 1)
    x["duica_since_2017_km2"] = np.maximum(uica[s][rr, cc] - uica["2017"][rr, cc], 0)
    prob[s] = model.predict(x[feats]).astype(np.float32)
breaks = np.quantile(prob["2025"], [0.2, 0.4, 0.6, 0.8])
cell_km2 = (C.row_cell_sizes_m(T30, H30)[2] / 1e6)[rr]
res["aoi_land_km2"] = round(float(cell_km2.sum()), 1)
cls ={s: (np.searchsorted(breaks, prob[s], side="right") + 1).astype(np.int8) for s in SCEN}
susc = {}
for s in SCEN[1:]:
    d = cls[s].astype(int) - cls["2017"]
    susc[s] = {"share_land_up_one_class_or_more_vs_2017": round(float((d >= 1).mean()), 5),
               "share_land_down_vs_2017": round(float((d <= -1).mean()), 5),
               "high_vhigh_km2": round(float(cell_km2[cls[s] >= 4].sum()), 1),
               "mean_prob_change_vs_2017": round(float((prob[s] - prob["2017"]).mean()), 5),
               "max_prob_change_vs_2017": round(float((prob[s] - prob["2017"]).max()), 4)}
    print(f"    {s}: {susc[s]}")
res["layer_A_riverine_susceptibility"] = {"reference_flood": "2018-08 (static + rainfall features)",
                                          "classes": "fixed quintile breaks of the 2025 map", "change": susc}

# ---------------------------------------------------------------------------
# 4. Layer B and combined classes
# ---------------------------------------------------------------------------
print("\n[4/5] Layer B: waterlogging pressure, and development-impact classes")
more, impact, layerB = {}, {}, {}
for s in SCEN[1:]:
    base = np.maximum(up_run["2017"][rr, cc], 1e-6)
    extra = up_run[s][rr, cc] - up_run["2017"][rr, cc]
    pct = 100 * extra / base
    local_extra = loc_run[s][rr, cc] - loc_run["2017"][rr, cc]
    more[s] = pct
    fp = cls[s] >= 4
    mw = pct >= MORE_WATER_PCT
    k = np.where(fp & mw, 3, np.where(mw, 2, np.where(fp, 1, 0)))
    impact[s] = k
    upstream_share = np.where(extra > 0, 1 - np.clip(local_extra / np.maximum(extra, 1e-9), 0, 1), np.nan)
    layerB[s] = {"aoi_km2_more_water": round(float(cell_km2[mw].sum()), 1),
                 "aoi_share_more_water": round(float(mw.mean()), 4),
                 "median_pct_more_water_cells": round(float(np.median(pct[mw])), 1) if mw.any() else None,
                 "upstream_share_of_extra_water_median_in_more_water_cells": round(float(np.nanmedian(upstream_share[mw])), 3) if mw.any() else None,
                 "km2_hotspot": round(float(cell_km2[k == 3].sum()), 1),
                 "km2_emerging": round(float(cell_km2[k == 2].sum()), 1),
                 "km2_existing_flood_prone": round(float(cell_km2[k == 1].sum()), 1)}
    # absolute extra water: extra runoff depth over the cell's upslope area (mm of the 50 mm storm)
    extra_mm = 1000 * extra / np.maximum(acc_aoi_km2[rr, cc] * 1e6, 1.0)
    layerB[s]["extra_mm_over_upslope_median_more_water_cells"] = round(float(np.median(extra_mm[mw])), 2) if mw.any() else None
    layerB[s]["extra_mm_over_upslope_p90_more_water_cells"] = round(float(np.percentile(extra_mm[mw], 90)), 2) if mw.any() else None
    layerB[s]["extra_mm_over_upslope_median_hotspots"] = round(float(np.median(extra_mm[k == 3])), 2) if (k == 3).any() else None
    layerB[s]["threshold_sensitivity"] = {
        f"{t}%": {"km2_more_water": round(float(cell_km2[pct >= t].sum()), 1),
                  "km2_hotspot": round(float(cell_km2[fp & (pct >= t)].sum()), 1)} for t in (2, 5, 10)}
    print(f"    {s}: {layerB[s]}")
res["layer_B_waterlogging_pressure"] = layerB

# hotspot blocks (2 km) ranked by 2030-trend hotspot area, labelled by nearest town (<= 5 km)
lon = T30.c + (cc + 0.5) * T30.a
lat = T30.f + (rr + 0.5) * T30.e
_, _, xu, yu = C.cell_xy_utm(rr, cc, T30)
blk = (xu // 2000).astype(int).astype(str).astype(object) + "_" + (yu // 2000).astype(int).astype(str).astype(object)
bdf = pd.DataFrame({"block": blk, "lon": lon, "lat": lat, "km2": cell_km2,
                    "hot25": (impact["2025"] == 3) * cell_km2, "hot30": (impact["2030 trend"] == 3) * cell_km2,
                    "hot30g": (impact["2030 GHSL"] == 3) * cell_km2,
                    "pct30": more["2030 trend"], "paved17": imp["2017"][AOI][rr, cc], "paved30": imp["2030 trend"][AOI][rr, cc]})
agg = bdf.groupby("block").agg(lon=("lon", "mean"), lat=("lat", "mean"), land_km2=("km2", "sum"),
                               hotspot_km2_2025=("hot25", "sum"), hotspot_km2_2030_trend=("hot30", "sum"),
                               hotspot_km2_2030_ghsl=("hot30g", "sum"), median_more_water_pct_2030=("pct30", "median"),
                               paved_2017=("paved17", "mean"), paved_2030_trend=("paved30", "mean"))
agg = agg[agg.land_km2 >= 1.0].sort_values("hotspot_km2_2030_trend", ascending=False)


def nearest_town(la, lo):
    best, dist = None, 1e9
    for t, (tla, tlo) in TOWNS.items():
        d_ = np.hypot((la - tla) * 110.6, (lo - tlo) * 109.6)
        if d_ < dist:
            best, dist = t, d_
    return f"near {best} ({dist:.1f} km)" if dist <= 5 else f"{la:.3f} N, {lo:.3f} E"


top = agg.head(12).copy()
top.insert(0, "place", [nearest_town(a, b) for a, b in zip(top.lat, top.lon)])
top.round(3).to_csv(f"{C.METRICS_DIR}/step9_hotspot_blocks.csv")
res["top_hotspot_blocks"] = top.round(3).reset_index().to_dict(orient="records")
print(top[["place", "hotspot_km2_2025", "hotspot_km2_2030_trend", "hotspot_km2_2030_ghsl", "median_more_water_pct_2030",
           "paved_2017", "paved_2030_trend"]].round(2).to_string())


def to_map(v, dtype=np.float32, fill=np.nan):
    m = np.full((H30, W30), fill, dtype)
    m[rr, cc] = v
    return m


for s in SCEN[1:]:
    tag = s.replace(" ", "_")
    with rasterio.open(f"{OUT}/scenario_{tag}.tif", "w", driver="GTiff", height=H30, width=W30, count=4, dtype="float32",
                       crs="EPSG:4326", transform=T30, nodata=np.nan, compress="deflate") as dst:
        for i_, (a, n) in enumerate(((prob[s], "riverine_susceptibility"), (cls[s], "riverine_class_fixed_breaks"),
                                     (more[s], "more_frequent_storm_water_vs_2017_pct"), (impact[s], "impact_class")), 1):
            dst.write(to_map(a.astype(np.float32)), i_)
            dst.set_band_description(i_, n)

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
print("\n[5/5] Figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
fig = plt.figure(figsize=(17, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(to_map(imp["2030 trend"][AOI][rr, cc] - imp["2017"][AOI][rr, cc]), cmap="Greys", vmin=0, vmax=0.5,
               extent=ext, interpolation="nearest")
fig.colorbar(im, ax=ax, shrink=0.7, label="added paved fraction")
ax.set_title("(a) Development: new paving 2017 → 2030 (trend)")
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(to_map(np.clip(more["2030 trend"], 0, None)), cmap="magma_r", vmin=0, vmax=30, extent=ext, interpolation="nearest")
fig.colorbar(im, ax=ax, shrink=0.7, label="% more water than in 2017")
ax.set_title("(b) Effect on water: extra runoff arriving in a\nfrequent 50 mm storm, 2017 → 2030 (trend)")
ax = fig.add_subplot(gs[0, 2])
icm = ListedColormap(["#f2f2f2", "#9bbbe0", "#f5b48f", "#c0262d"])
ax.imshow(to_map(impact["2030 trend"], np.float32), cmap=icm, vmin=-0.5, vmax=3.5, extent=ext, interpolation="nearest")
for name, (la, lo) in TOWNS.items():
    if C.BBOX[0] < lo < C.BBOX[2] and C.BBOX[1] < la < C.BBOX[3]:
        ax.plot(lo, la, "k.", ms=3); ax.text(lo + 0.004, la, name.split(" (")[0], fontsize=6.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=icm(3), label="Hotspot: flood-prone + ≥5 % more water"),
                   Patch(color=icm(2), label="Emerging: ≥5 % more water"),
                   Patch(color=icm(1), label="Existing flood-prone (riverine)"),
                   Patch(color=icm(0), label="Neither")], fontsize=7, loc="lower left")
ax.set_title("(c) Development-impact classes, 2030 (trend)")
for a_ in fig.axes:
    a_.set_xticks([]); a_.set_yticks([])
ax = fig.add_subplot(gs[1, 0])
xs = np.arange(3)
lab = ["2025\n(observed)", "2030\n(trend)", "2030\n(GHSL)"]
w = 0.38
ax.bar(xs - w / 2, [layerB[s]["aoi_km2_more_water"] for s in SCEN[1:]], w, color=C.C_ORANGE, label="land with ≥5 % more water")
ax.bar(xs + w / 2, [layerB[s]["km2_hotspot"] for s in SCEN[1:]], w, color="#c0262d", label="hotspot (also flood-prone)")
for i_, s in enumerate(SCEN[1:]):
    ax.text(i_ - w / 2, layerB[s]["aoi_km2_more_water"], f"{layerB[s]['aoi_km2_more_water']:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(i_ + w / 2, layerB[s]["km2_hotspot"], f"{layerB[s]['km2_hotspot']:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylabel(f"km² of the study area ({res['aoi_land_km2']:,.0f} km² land)")
ax.set_title("(d) Area affected by development, vs 2017"); ax.legend(fontsize=8); ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[1, 1])
ch = res["catchment_storm_runoff_change_pct_vs_2017"]
ax.bar(xs - w / 2, [ch["frequent"][s] for s in SCEN[1:]], w, color=C.C_ORANGE, label="frequent storm (50 mm, normal soil)")
ax.bar(xs + w / 2, [ch["monsoon"][s] for s in SCEN[1:]], w, color=C.C_BLUE, label="monsoon flood storm (200 mm, wet soil)")
for i_, s in enumerate(SCEN[1:]):
    ax.text(i_ - w / 2, ch["frequent"][s], f"+{ch['frequent'][s]:.1f} %", ha="center", va="bottom", fontsize=8)
    ax.text(i_ + w / 2, ch["monsoon"][s], f"+{ch['monsoon'][s]:.2f} %", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(lab); ax.set_ylabel("% more storm runoff than 2017 (whole grid)")
ax.set_title("(e) Development matters for frequent storms,\nlittle for extreme monsoon floods"); ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, max(ch["frequent"].values()) * 1.3)
ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
rows_ = [[r["place"], f"{r['hotspot_km2_2025']:.2f}", f"{r['hotspot_km2_2030_trend']:.2f}", f"{r['median_more_water_pct_2030']:.0f} %",
          f"{100 * r['paved_2017']:.0f} → {100 * r['paved_2030_trend']:.0f} %"] for r in res["top_hotspot_blocks"][:10]]
tb = ax.table(cellText=rows_, colLabels=["2 km block", "hotspot\nkm² 2025", "hotspot\nkm² 2030", "more water\n2030", "paved\n2017 → 2030"],
              loc="center", cellLoc="left", colWidths=[0.36, 0.14, 0.14, 0.15, 0.21])
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.45)
ax.set_title("(f) Top hotspot blocks (2030 trend)")
fig.suptitle("Step 9  Effect of development on flooding, Lower Periyar: development adds water where it is already flood-prone",
             fontsize=12)
fig.savefig(f"{C.MAP_DIR}/step9_development_effect.png", dpi=140)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
cmap5 = plt.get_cmap("RdYlBu_r", 5)
for ax, s in zip(axes[:2], ("2025", "2030 trend")):
    im = ax.imshow(to_map(cls[s].astype(np.float32)), cmap=cmap5, vmin=0.5, vmax=5.5, extent=ext, interpolation="nearest")
    ax.set_title(f"Riverine flood susceptibility, {s}\n(scenario model, fixed class breaks)")
cb = fig.colorbar(im, ax=axes[:2], shrink=0.6, ticks=range(1, 6))
cb.ax.set_yticklabels(["Very Low", "Low", "Moderate", "High", "Very High"])
dp = to_map(prob["2030 trend"] - prob["2017"])
im2 = axes[2].imshow(dp, cmap="Reds", vmin=0, vmax=max(1e-3, float(np.nanpercentile(dp, 99.9))), extent=ext, interpolation="nearest")
axes[2].set_title(f"Δ susceptibility 2017 → 2030 (trend)\n{100 * susc['2030 trend']['share_land_up_one_class_or_more_vs_2017']:.2f} % of land "
                  f"moves up a class")
fig.colorbar(im2, ax=axes[2], shrink=0.6, label="Δ probability vs 2017")
for a_ in axes:
    a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("Step 9  Layer A: riverine flood susceptibility under development (2018 reference flood)", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step9_susceptibility_scenarios.png", dpi=130)
plt.close(fig)

res["headline"] = {
    "frequent_storm_runoff_change_pct": ch["frequent"], "monsoon_storm_runoff_change_pct": ch["monsoon"],
    "aoi_km2_with_5pct_more_water": {s: layerB[s]["aoi_km2_more_water"] for s in SCEN[1:]},
    "hotspot_km2": {s: layerB[s]["km2_hotspot"] for s in SCEN[1:]},
    "riverine_share_land_up_a_class": {s: susc[s]["share_land_up_one_class_or_more_vs_2017"] for s in SCEN[1:]}}
res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step9_dual_prediction.json")
print(f"\nheadline: {res['headline']}\nDone in {res['runtime_s']} s")
