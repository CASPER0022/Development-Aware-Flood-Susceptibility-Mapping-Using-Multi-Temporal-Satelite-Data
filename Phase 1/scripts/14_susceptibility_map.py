"""
Step 10 - Full-AOI flood susceptibility map
(Week 4, Phase1_Implementation_Plan).

10.1 The Step 9 LightGBM baseline (all 11 features, trained on the 70 % random
     split) is applied to every land pixel of the Step 7 grid (~30 m) ->
     continuous flood susceptibility (predicted flood probability, 0-1).
     Open water (Step 7 domain mask) stays NoData.
     Alongside (agreed at Step 9): the same model without rainfall, trained
     identically (same split, parameters and number of trees). Rainfall acts
     as a location proxy (Step 9: AUC 0.58 on unseen regions), and 5.5 km
     CHIRPS cells would otherwise imprint blocks on the map.
10.2 Five classes by quantile breaks (20/40/60/80th percentiles of the land
     pixels), Very Low -> Very High, the convention of the reference papers.
     By construction each class covers ~20 % of the land.
10.3 Publication map: classified susceptibility, legend, scale bar, north
     arrow, major rivers, open water, drawn at true scale (lon/lat axes with
     aspect 1/cos(latitude), otherwise the map is stretched ~1.5 % E-W).

Validation against an event the model never saw. Labels came only from the
Aug 2018 flood. The Aug 2019 flood is an independent test:
    - NDEM (NRSC/ISRO) 2019 flood polygons (10 + 12 Aug 2019), and
    - our Step 5 Otsu inventory of 10 Aug 2019 (>= 50 % of a 30 m cell),
both on the land domain minus 2019 dry-season reference water. Reported per
class: share of land, share of 2019 flood, frequency ratio (flood share /
area share; > 1 = over-represented), plus the AUC of the continuous map for
2019 flood pixels, also restricted to 2019 flood *outside* the 2018 NDEM
envelope (places that did not flood in 2018 at all). The 2018 training flood
pool is shown per class as well, but that is a success rate, not a test.

Outputs
    data/processed/susceptibility/susceptibility_prob_{all,no_rainfall}.tif   float32, NaN = open water
    data/processed/susceptibility/susceptibility_class_{all,no_rainfall}.tif  uint8 1-5, 0 = open water
    outputs/models/step10_lightgbm_no_rainfall.txt
    outputs/metrics/step10_susceptibility_summary.json
    outputs/maps/step10_susceptibility_map.png               headline (all 11 features)
    outputs/maps/step10_susceptibility_map_no_rainfall.png
    outputs/maps/step10_susceptibility_comparison.png
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

STACK = "data/processed/features/feature_stack.tif"
DOMAIN = "data/processed/features/analysis_domain.tif"
POOLS_2018 = "data/processed/training/label_pools_2018.tif"
DATA = "data/processed/training/training_samples_2018.csv"
MODEL_ALL = "outputs/models/step9_lightgbm_baseline.txt"
STEP9_JSON = "outputs/metrics/step9_baseline_metrics.json"
CLASSES_2019 = "data/processed/flood_inventory/water_classes_2019_20190810.tif"
NDEM = "data/validation/NDEM_KL_Floods_Inundation.parquet"
RIVERS = "data/raw/hydrosheds/periyar_rivers_clip.gpkg"
OUT_DIR = "data/processed/susceptibility"
MODEL_DIR = "outputs/models"
MAP_DIR = "outputs/maps"
METRICS_DIR = "outputs/metrics"
for d in (OUT_DIR, MODEL_DIR, MAP_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
NDEM_2019 = ["10-08-2019", "12-08-2019"]
NDEM_2018_ENVELOPE = ["17-08-2018", "18-08-2018", "21-08-2018", "24-08-2018", "27-08-2018", "28-08-2018"]
QUANTILES = [0.2, 0.4, 0.6, 0.8]
CLASS_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]
CLASS_COLORS = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]   # sequential, light -> dark
WATER = "#dfe5ea"
C_RIVER = "#2a78d6"
C_ALL = "#2a78d6"
C_NORAIN = "#eb6834"
C_BASE = "#52514e"
MID_LAT = (BBOX[1] + BBOX[3]) / 2
SEED = 42

print("=" * 80)
print("STEP 10: FULL-AOI FLOOD SUSCEPTIBILITY MAP")
print("=" * 80)
t_start = time.time()

# ---------------------------------------------------------------------------
# Inputs and models
# ---------------------------------------------------------------------------
print("\n[1/6] Inputs and models")
with rasterio.open(STACK) as src:
    stack = src.read()
    FEATURES = list(src.descriptions)
    T = src.transform
    CRS = src.crs
    H, W = src.shape
with rasterio.open(DOMAIN) as src:
    land = src.read(1).astype(bool)
booster_all = lgb.Booster(model_file=MODEL_ALL)
assert booster_all.feature_name() == FEATURES, "Step 9 model and feature stack disagree on feature order"
s9 = json.load(open(STEP9_JSON, encoding="utf-8"))
PARAMS, ROUNDS = s9["model"]["params"], s9["model"]["num_boost_round"]

# no-rainfall variant: same split, parameters and number of trees as Step 9
df = pd.read_csv(DATA)
y = df["label"].to_numpy()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=SEED)
FEAT_NR = [f for f in FEATURES if f != "rainfall_mean_annual_mm"]
booster_nr = lgb.train(PARAMS, lgb.Dataset(df.iloc[idx_tr][FEAT_NR], y[idx_tr]), num_boost_round=ROUNDS)
booster_nr.save_model(f"{MODEL_DIR}/step10_lightgbm_no_rainfall.txt")
p_chk = booster_all.predict(df.iloc[idx_te][FEATURES])
assert abs(roc_auc_score(y[idx_te], p_chk) - s9["headline_test_metrics"]["auc"]) < 1e-4, "Step 9 model not reproduced"
p_nr = booster_nr.predict(df.iloc[idx_te][FEAT_NR])
nr_test = {"auc": round(float(roc_auc_score(y[idx_te], p_nr)), 4),
           "accuracy": round(float(accuracy_score(y[idx_te], p_nr >= 0.5)), 4),
           "f1": round(float(f1_score(y[idx_te], p_nr >= 0.5)), 4)}
print(f"    Step 9 model reloaded (test AUC reproduced: {roc_auc_score(y[idx_te], p_chk):.4f})")
print(f"    no-rainfall model: {len(FEAT_NR)} features, {ROUNDS} trees, random-split test AUC {nr_test['auc']:.4f}, "
      f"accuracy {100 * nr_test['accuracy']:.2f}")

# ---------------------------------------------------------------------------
# 10.1 Predict every land pixel
# ---------------------------------------------------------------------------
print("\n[2/6] 10.1 Predicting every land pixel")
X_land = pd.DataFrame(stack[:, land].T.astype(np.float64), columns=FEATURES)
assert np.isfinite(X_land.to_numpy()).all()


def predict_map(booster, feats):
    p = np.full((H, W), np.nan, np.float32)
    out = np.empty(len(X_land))
    for s in range(0, len(X_land), 250_000):
        out[s:s + 250_000] = booster.predict(X_land.iloc[s:s + 250_000][feats])
    p[land] = out
    return p


prob = {"all": predict_map(booster_all, FEATURES), "no_rainfall": predict_map(booster_nr, FEAT_NR)}
for k, p in prob.items():
    v = p[land]
    print(f"    {k:<12} {land.sum():,} pixels, probability min {v.min():.4f} median {np.median(v):.4f} "
          f"max {v.max():.4f}; > 0.5: {100 * (v > 0.5).mean():.1f}% of land")

# ---------------------------------------------------------------------------
# 10.2 Quantile classes
# ---------------------------------------------------------------------------
print("\n[3/6] 10.2 Five quantile classes")
lat_rows = T.f + (np.arange(H) + 0.5) * T.e
row_km2 = (T.a * 111.320 * np.cos(np.radians(lat_rows))) * (abs(T.e) * 110.574)
cell_km2 = np.broadcast_to(row_km2[:, None], (H, W))
land_km2 = float(cell_km2[land].sum())
breaks, classes, class_stats = {}, {}, {}
for k, p in prob.items():
    b = np.quantile(p[land], QUANTILES)
    assert np.all(np.diff(b) > 0), f"{k}: quantile breaks not strictly increasing {b}"
    c = np.zeros((H, W), np.uint8)
    c[land] = np.digitize(p[land], b) + 1
    breaks[k], classes[k] = b, c
    class_stats[k] = []
    lo = [0.0] + list(b)
    hi = list(b) + [1.0]
    for i, name in enumerate(CLASS_NAMES, start=1):
        a = float(cell_km2[c == i].sum())
        class_stats[k].append({"class": i, "name": name, "p_range": [float(f"{lo[i - 1]:.4g}"), float(f"{hi[i - 1]:.4g}")],
                               "area_km2": round(a, 2), "share_of_land": round(a / land_km2, 4)})
    print(f"    {k:<12} breaks " + ", ".join(f"{x:.3g}" for x in b) + "  |  shares "
          + " ".join(f"{100 * s['share_of_land']:.1f}%" for s in class_stats[k]))

prof = {"driver": "GTiff", "height": H, "width": W, "count": 1, "crs": CRS, "transform": T, "compress": "deflate"}
for k in prob:
    with rasterio.open(f"{OUT_DIR}/susceptibility_prob_{k}.tif", "w", **prof, dtype="float32", nodata=np.nan,
                       predictor=3) as dst:
        dst.write(prob[k], 1)
        dst.set_band_description(1, f"flood susceptibility (LightGBM probability), model: {k}")
    with rasterio.open(f"{OUT_DIR}/susceptibility_class_{k}.tif", "w", **prof, dtype="uint8", nodata=0) as dst:
        dst.write(classes[k], 1)
        dst.set_band_description(1, "1 Very Low, 2 Low, 3 Moderate, 4 High, 5 Very High (quantile breaks); 0 open water")
agree_same = float((classes["all"][land] == classes["no_rainfall"][land]).mean())
agree_pm1 = float((np.abs(classes["all"][land].astype(int) - classes["no_rainfall"][land]) <= 1).mean())
rank_corr = float(pd.Series(prob["all"][land]).corr(pd.Series(prob["no_rainfall"][land]), method="spearman"))
print(f"    models agree on the class for {100 * agree_same:.1f}% of land (within one class: {100 * agree_pm1:.1f}%), "
      f"Spearman r {rank_corr:.3f}")

# ---------------------------------------------------------------------------
# Validation: Aug 2019 (never used for training)
# ---------------------------------------------------------------------------
print("\n[4/6] Validation against the Aug 2019 flood (independent event)")
ndem = gpd.read_parquet(NDEM)
if ndem.crs is None:
    ndem = ndem.set_crs("EPSG:4326", allow_override=True)
ndem = ndem[ndem.intersects(box(*BBOX))].copy()
ndem["date"] = ndem["from_time"].astype(str).str[:10]


def ndem_30m(dates):
    g = ndem[ndem["date"].isin(dates)].geometry
    return rasterize([(x, 1) for x in g], out_shape=(H, W), transform=T, fill=0, dtype="uint8").astype(bool)


with rasterio.open(CLASSES_2019) as src:
    cls19 = src.read(1)
    T20 = src.transform
assert abs(T.a / T20.a - 1.5) < 1e-9 and abs((T20.c - T.c) / T20.a - 0.5) < 1e-6 and abs((T.f - T20.f) / abs(T20.e) - 0.5) < 1e-6


def to_30m_fraction(mask20):
    """Exact 20 m -> 30 m area fraction (same method as Step 8)."""
    sub = np.repeat(np.repeat(mask20.astype(np.float32), 2, axis=0), 2, axis=1)
    canvas = np.zeros((3 * H, 3 * W), np.float32)
    h = min(sub.shape[0], 3 * H - 1)
    w = min(sub.shape[1], 3 * W - 1)
    canvas[1:1 + h, 1:1 + w] = sub[:h, :w]
    return canvas.reshape(H, 3, W, 3).mean(axis=(1, 3))


otsu19 = to_30m_fraction(cls19 == 1) >= 0.5
refw19 = to_30m_fraction(cls19 == 2) >= 0.5
ndem19 = ndem_30m(NDEM_2019)
env18 = ndem_30m(NDEM_2018_ENVELOPE)
with rasterio.open(POOLS_2018) as src:
    train18 = src.read(1) == 1
dom = land & ~refw19
refs = {
    "ndem_2019": ndem19 & dom,
    "otsu_2019": otsu19 & dom,
    "ndem_2019_outside_2018_envelope": ndem19 & dom & ~env18,
    "training_flood_pool_2018 (success rate, not a test)": train18 & dom,
}
validation = {"domain": "land minus 2019 dry-season reference water",
              "domain_km2": round(float(cell_km2[dom].sum()), 2), "models": {}}
for k in prob:
    vm = {}
    c = classes[k]
    area_share = np.array([cell_km2[dom & (c == i)].sum() for i in range(1, 6)]) / cell_km2[dom].sum()
    for rname, rmask in refs.items():
        fl_km2 = float(cell_km2[rmask].sum())
        fshare = np.array([cell_km2[rmask & (c == i)].sum() for i in range(1, 6)]) / max(fl_km2, 1e-12)
        auc = roc_auc_score(rmask[dom], prob[k][dom]) if rmask.any() else float("nan")
        vm[rname] = {
            "flood_km2": round(fl_km2, 2), "auc": round(float(auc), 4),
            "per_class": [{"class": CLASS_NAMES[i], "area_share": round(float(area_share[i]), 4),
                           "flood_share": round(float(fshare[i]), 4),
                           "frequency_ratio": round(float(fshare[i] / area_share[i]), 3)} for i in range(5)],
            "share_in_high_or_very_high": round(float(fshare[3] + fshare[4]), 4),
        }
    validation["models"][k] = vm
    for rname in ("ndem_2019", "otsu_2019", "ndem_2019_outside_2018_envelope"):
        r = vm[rname]
        print(f"    {k:<12} {rname:<32} {r['flood_km2']:6.1f} km2 | AUC {r['auc']:.3f} | High+Very High "
              f"{100 * r['share_in_high_or_very_high']:.1f}% | FR by class "
              + " ".join(f"{x['frequency_ratio']:.2f}" for x in r["per_class"]))

# Physical plausibility: how much high / steep ground lands in High + Very High
elev = stack[FEATURES.index("elevation_m")]
slope = stack[FEATURES.index("slope_deg")]
plausibility = {}
for k in prob:
    hv = classes[k] >= 4
    plausibility[k] = {}
    for nm, m in (("elevation_gt_50m", land & (elev > 50)), ("slope_gt_10deg", land & (slope > 10))):
        plausibility[k][nm] = {"share_of_such_land_in_high_or_very_high": round(float((hv & m).sum() / m.sum()), 4),
                               "share_of_high_or_very_high_land": round(float((hv & m).sum() / hv.sum()), 4)}
    print(f"    plausibility {k:<12} land > 50 m rated High/Very High: "
          f"{100 * plausibility[k]['elevation_gt_50m']['share_of_such_land_in_high_or_very_high']:.1f}%, "
          f"slopes > 10 deg: {100 * plausibility[k]['slope_gt_10deg']['share_of_such_land_in_high_or_very_high']:.1f}%")
share_lt_001 = {k: round(float((p[land] < 0.01).mean()), 4) for k, p in prob.items()}

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
results = {
    "step": 10, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models": {"all": {"path": MODEL_ALL, "features": FEATURES, "note": "Step 9 baseline (headline)"},
               "no_rainfall": {"path": f"{MODEL_DIR}/step10_lightgbm_no_rainfall.txt", "features": FEAT_NR,
                               "trees": ROUNDS, "random_split_test": nr_test,
                               "note": "same split/params/trees as Step 9, rainfall removed"}},
    "prediction": {"land_pixels": int(land.sum()), "land_km2": round(land_km2, 2),
                   "probability_stats": {k: {"min": round(float(np.nanmin(p)), 5),
                                             "median": round(float(np.nanmedian(p)), 5),
                                             "max": round(float(np.nanmax(p)), 5),
                                             "share_above_0.5": round(float((p[land] > 0.5).mean()), 4)}
                                         for k, p in prob.items()}},
    "classification": {"method": "quantile breaks at 20/40/60/80 % of land pixels",
                       "breaks": {k: [float(f"{x:.4g}") for x in b] for k, b in breaks.items()},
                       "classes": class_stats},
    "model_agreement": {"same_class_share": round(agree_same, 4), "within_one_class_share": round(agree_pm1, 4),
                        "spearman_probability": round(rank_corr, 4)},
    "validation_2019": validation,
    "plausibility_high_ground": plausibility,
    "note_on_probabilities": "trained on balanced 50/50 samples and overconfident (perfect train fit), so the "
                             "scores are a relative susceptibility index, not real-world flood probabilities",
    "share_of_land_with_score_below_0.01": share_lt_001,
    "outputs": {k: {"probability": f"{OUT_DIR}/susceptibility_prob_{k}.tif",
                    "classes": f"{OUT_DIR}/susceptibility_class_{k}.tif"} for k in prob},
}
with open(f"{METRICS_DIR}/step10_susceptibility_summary.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

# ---------------------------------------------------------------------------
# 10.3 Maps
# ---------------------------------------------------------------------------
print("\n[5/6] 10.3 Publication map")
extent = (T.c, T.c + W * T.a, T.f + H * T.e, T.f)
rivers = gpd.read_file(RIVERS)
major = gpd.clip(rivers[rivers["ORD_STRA"] >= 4], box(*BBOX))
cmap = ListedColormap(CLASS_COLORS)
norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)
KM_PER_DEG_LON = 111.320 * np.cos(np.radians(MID_LAT))


def draw_classes(ax, c, title=None):
    ax.set_facecolor(WATER)
    ax.imshow(np.where(land, c, np.nan), cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    major.plot(ax=ax, color=C_RIVER, lw=0.9)
    ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_aspect(1 / np.cos(np.radians(MID_LAT)))
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}°N"))
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


def scale_bar(ax, km=10, x0=76.165, y0=9.915):
    seg = km / 2 / KM_PER_DEG_LON
    for i, col in enumerate(("#0b0b0b", "white")):
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, 0.0035, facecolor=col, edgecolor="#0b0b0b", lw=0.8, zorder=5))
    for i, lbl in enumerate(("0", f"{km // 2}", f"{km} km")):
        ax.text(x0 + i * seg, y0 + 0.0055, lbl, ha="center", va="bottom", fontsize=8, zorder=5)


def north_arrow(ax, x=76.525, y=10.215):
    ax.annotate("", xy=(x, y + 0.022), xytext=(x, y), zorder=5,
                arrowprops=dict(facecolor="#0b0b0b", edgecolor="#0b0b0b", width=4, headwidth=12, headlength=10))
    ax.text(x, y + 0.026, "N", ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=5)


def fmt_p(v):
    if v == 0:
        return "0"
    if v < 0.01:
        m, e = f"{v:.1e}".split("e")
        return f"{m}×10$^{{{int(e)}}}$"
    return f"{v:.2f}"


def legend_handles(stats):
    h = [Patch(facecolor=col, edgecolor="#8c8c8c", lw=0.5,
               label=f"{s['name']}  ({fmt_p(s['p_range'][0])} – {fmt_p(s['p_range'][1])})")
         for col, s in zip(CLASS_COLORS, stats)]
    h += [Patch(facecolor=WATER, edgecolor="#8c8c8c", lw=0.5, label="Open water (not modelled)"),
          Line2D([], [], color=C_RIVER, lw=1.5, label="Major rivers (Strahler ≥ 4)")]
    return h


MODEL_TEXT = {"all": ("all 11 features (plan baseline)", "11 terrain/hydrology/rainfall features"),
              "no_rainfall": ("without rainfall", "10 terrain/hydrology features, rainfall removed")}
for k in prob:
    fig, ax = plt.subplots(figsize=(10.5, 10.2), dpi=150)
    draw_classes(ax, classes[k])
    scale_bar(ax)
    north_arrow(ax)
    ax.legend(handles=legend_handles(class_stats[k]), title="Flood susceptibility (model score range)",
              loc="upper left", fontsize=8.5, title_fontsize=9, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    v19 = validation["models"][k]["ndem_2019"]
    ax.set_title(f"Flood susceptibility — Lower Periyar & Aluva–Kochi corridor\nLightGBM, {MODEL_TEXT[k][0]}",
                 loc="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.015,
             f"LightGBM ({MODEL_TEXT[k][1]}), trained on the Aug 2018 Sentinel-1 flood inventory. "
             f"Classes: quantile breaks (20 % of land each); scores are a relative index, not flood probabilities.\n"
             f"Independent check, Aug 2019 NDEM flood (not used in training): "
             f"{100 * v19['share_in_high_or_very_high']:.0f}% falls in High/Very High; Very High holds "
             f"{v19['per_class'][4]['frequency_ratio']:.1f}× its area share; AUC {v19['auc']:.2f}. "
             f"Grid: FABDEM 1″ (~30 m), WGS84. Rivers: HydroRIVERS v10.",
             fontsize=7.6, color="#3a3a3a", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    p_main = f"{MAP_DIR}/step10_susceptibility_map.png" if k == "all" else f"{MAP_DIR}/step10_susceptibility_map_{k}.png"
    fig.savefig(p_main, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] {p_main}")

print("\n[6/6] Comparison figure")
fig = plt.figure(figsize=(20, 7.6), dpi=120)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
ax = fig.add_subplot(gs[0])
ax.set_facecolor(WATER)
im = ax.imshow(np.log10(np.clip(prob["all"], 1e-6, 1)), cmap="YlOrRd", vmin=-6, vmax=0, extent=extent,
               interpolation="nearest")
major.plot(ax=ax, color=C_RIVER, lw=0.8)
ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
ax.set_aspect(1 / np.cos(np.radians(MID_LAT)))
ax.tick_params(labelsize=7)
ax.set_title("(a) Continuous score, all 11 features (log scale)", loc="left", fontsize=11, fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=range(-6, 1))
cb.ax.set_yticklabels(["≤10$^{-6}$"] + [f"10$^{{{e}}}$" for e in range(-5, 0)] + ["1"])
cb.set_label(f"Model score, log scale ({100 * share_lt_001['all']:.0f}% of land < 0.01)", fontsize=8)
cb.outline.set_visible(False)
ax = fig.add_subplot(gs[1])
draw_classes(ax, classes["no_rainfall"],
             f"(b) Classes without rainfall — same class as (a) on {100 * agree_same:.0f}% of land")
ax.tick_params(labelsize=7)
ax.legend(handles=legend_handles(class_stats["no_rainfall"])[:5], loc="upper left", fontsize=7, framealpha=0.9)

ax = fig.add_subplot(gs[2])
xx = np.arange(5)
w = 0.38
for i, (k, col, lbl) in enumerate((("all", C_ALL, "All 11 features"), ("no_rainfall", C_NORAIN, "Without rainfall"))):
    r = validation["models"][k]["ndem_2019"]
    fr = [x["frequency_ratio"] for x in r["per_class"]]
    bars = ax.bar(xx + (i - 0.5) * w, fr, width=w - 0.03, color=col, zorder=2,
                  label=f"{lbl} (AUC {r['auc']:.2f})")
    for b_, v in zip(bars, fr):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.axhline(1, color=C_BASE, ls="--", lw=1, zorder=1)
ax.text(4.65, 1.05, "= no better than chance", ha="right", va="bottom", fontsize=8, color=C_BASE)
ax.set_xticks(xx, CLASS_NAMES)
ax.set_ylabel("Frequency ratio  (share of 2019 flood ÷ share of land)")
ax.set_title("(c) Independent check: Aug 2019 NDEM flood by class", loc="left", fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=0)
fig.suptitle("Step 10  Susceptibility: continuous map, no-rainfall variant, and validation on an unseen flood event",
             fontsize=13.5, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
p_cmp = f"{MAP_DIR}/step10_susceptibility_comparison.png"
fig.savefig(p_cmp, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_cmp}")

print("\n" + "=" * 80)
print("STEP 10 SUMMARY")
print("=" * 80)
for k in prob:
    r = validation["models"][k]["ndem_2019"]
    print(f"  {k:<12} 2019 NDEM flood: {100 * r['share_in_high_or_very_high']:.1f}% in High/Very High "
          f"(40% of land), Very High FR {r['per_class'][4]['frequency_ratio']:.2f}, AUC {r['auc']:.3f}")
print(f"  Runtime {time.time() - t_start:.0f} s")
print("=" * 80)
