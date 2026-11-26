"""
Phase 2 Step 11 - Explanations (SHAP).

What SHAP can and cannot say: it explains the MODEL, which was trained on
big riverine flood labels (radar, radar + NDEM). It therefore shows how much the
model relies on each input to place those floods. It says nothing about
waterlogging (no labels); that is tested separately (Step 10b).

Models explained (TreeSHAP via LightGBM pred_contrib, log-odds units):
  M1 chosen riverine model: PU bagging on feature set C, radar labels, trained on
     2018 + 2019 (contributions averaged over the 25 bags)
  M2 unconstrained D (with land cover), radar + NDEM labels, standard design
  M3 scenario model (Step 9.0b): D - land cover, monotone in paving / UICA

11.1 Global importance: mean |SHAP| share per feature and per group; development
     share overall, in Step 9 hotspot blocks (2030 trend) and in the Kochi core.
     Checks: built = dry in M2 (land cover 'built' and paving push risk down),
     removed in M3; how much the mean-annual-rainfall location proxy is used.
11.2 Local explanations for named places (Aluva, North Paravur, Ernakulam core,
     Kakkanad, a Kolenchery hotspot): top contributions.
     Development share of the 2017 -> 2030 (trend) change at hotspots in M3:
     local paving vs upstream paving (UICA, UICA ratio, dUICA).
Outputs: outputs/metrics/step11_shap.json, outputs/maps/step11_shap_global.png,
         outputs/maps/step11_shap_local.png
"""
import gc
import json
import time

import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

N_SAMPLE = 30000
GROUPS = {"terrain & hydrology": ["elevation_m", "slope_deg", "aspect_deg", "curvature_plan", "curvature_profile", "twi",
                                  "twi_full", "hand_m", "dist_to_river_m", "dist_to_major_river_m", "drainage_density_km_km2"],
          "roads": ["dist_to_road_m"],
          "event rainfall": ["rain_event_5d_mm", "rain_antecedent_30d_mm"],
          "annual rainfall (location proxy)": ["rainfall_mean_annual_mm"],
          "land cover": ["landcover_class"],
          "development (paving, UICA)": ["impervious_frac", "uica_km2", "uica_ratio", "duica_since_2017_km2"]}
DEV = GROUPS["development (paving, UICA)"]
PLACES = {"Aluva": (10.108, 76.351), "North Paravur": (10.143, 76.228), "Ernakulam core (Kochi)": (9.982, 76.282),
          "Kakkanad": (10.016, 76.341), "Kolenchery hotspot": (9.995, 76.475)}
C.banner("PHASE 2 STEP 11: SHAP EXPLANATIONS")
t0 = time.time()
res = {"step": 11, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
       "note": "SHAP explains the riverine-flood models; it is not evidence about waterlogging"}
s8 = json.load(open(f"{C.METRICS_DIR}/step8_model_v2_ablation.json", encoding="utf-8"))
b = s8["tuning"]["best"]
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": b["num_leaves"], "min_data_in_leaf": b["min_data_in_leaf"],
         "lambda_l2": b["lambda_l2"], "feature_fraction": 0.8}
ROUNDS = s8["tuning"]["rounds"]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
print("\n[1/4] Models")
df = pd.read_parquet(C.DATASET_V2)
tr = df[df.event.isin(C.TRAIN_EVENTS)]
pos, unl = tr[tr.set == "pos"], tr[tr.set == "rnd"]
FC = C.FEATURES_C
rng = np.random.default_rng(C.SEED)
bags = []
for _ in range(C.PU_BAGS):
    u = unl.iloc[rng.choice(len(unl), size=min(len(pos), len(unl)), replace=False)]
    d = pd.concat([pos, u])
    bags.append(C.lgb_fit(d, FC, np.r_[np.ones(len(pos)), np.zeros(len(u))], TUNED, max(ROUNDS // 2, 100)))
dn = C.ndem_label_dataset()
dtr = C.std_rows(dn[dn.event.isin(C.TRAIN_EVENTS)])
m2 = C.lgb_fit(dtr, C.FEATURES_D, C.y_of(dtr), TUNED, ROUNDS)
m3 = lgb.Booster(model_file=f"{C.MODEL_DIR}/model_v2_scenario.txt")
F3 = m3.feature_name()
print(f"    M1 PU bagging C ({len(bags)} bags), M2 unconstrained D, M3 scenario model ({len(F3)} features)")


def shap(model_or_bags, X, feats):
    if isinstance(model_or_bags, list):
        return np.mean([m.predict(X[feats], pred_contrib=True) for m in model_or_bags], axis=0)
    return model_or_bags.predict(X[feats], pred_contrib=True)


# ---------------------------------------------------------------------------
# Sample (2018 reference flood features) + places + hotspot flags
# ---------------------------------------------------------------------------
print("\n[2/4] Sample and SHAP values")
lm, rr, cc, land = C.land_matrix("2018-08_ndem")
with rasterio.open(C.P1_DOMAIN) as src:
    T30 = src.transform
lon_all = T30.c + (cc + 0.5) * T30.a
lat_all = T30.f + (rr + 0.5) * T30.e
with rasterio.open(f"{C.PROC_DIR}/scenarios/scenario_2030_trend.tif") as src:
    impact = src.read(list(src.descriptions).index("impact_class") + 1)[rr, cc]
idx = rng.choice(len(lm), size=N_SAMPLE, replace=False)
kb = C.KOCHI_BOX
in_k = (lon_all >= kb[0]) & (lon_all <= kb[2]) & (lat_all >= kb[1]) & (lat_all <= kb[3])
hot_idx = rng.choice(np.flatnonzero(impact == 3), size=5000, replace=False)
k_idx = rng.choice(np.flatnonzero(in_k), size=3000, replace=False)
place_idx = {}
for p, (la, lo) in PLACES.items():
    d2 = (lat_all - la) ** 2 + (lon_all - lo) ** 2
    place_idx[p] = np.argsort(d2)[:300]                      # ~300 nearest land cells (~0.3 km2)
all_idx = np.unique(np.concatenate([idx, hot_idx, k_idx] + list(place_idx.values())))
X = lm.iloc[all_idx].reset_index(drop=True)
pos_of = {v: i for i, v in enumerate(all_idx)}
sel = lambda ids: np.array([pos_of[i] for i in ids])
S = {"M1": (shap(bags, X, FC), FC), "M2": (shap(m2, X, C.FEATURES_D), C.FEATURES_D), "M3": (shap(m3, X, F3), F3)}
print(f"    explained {len(X):,} cells x 3 models")


def shares(sv, feats, rows):
    a = np.abs(sv[rows, :-1]).mean(axis=0)
    a = a / a.sum()
    per = {f: float(v) for f, v in zip(feats, a)}
    grp = {g: float(sum(per.get(f, 0) for f in fs)) for g, fs in GROUPS.items()}
    return per, grp


glob = {}
for k, (sv, feats) in S.items():
    rs = sel(idx)
    per, grp = shares(sv, feats, rs)
    glob[k] = {"feature_share": {f: round(v, 4) for f, v in sorted(per.items(), key=lambda x: -x[1])},
               "group_share": {g: round(v, 4) for g, v in grp.items()},
               "development_share_hotspots": round(shares(sv, feats, sel(hot_idx))[1]["development (paving, UICA)"], 4),
               "development_share_kochi_core": round(shares(sv, feats, sel(k_idx))[1]["development (paving, UICA)"], 4)}
    print(f"    {k}: groups {glob[k]['group_share']}; development share hotspots {glob[k]['development_share_hotspots']}, "
          f"Kochi {glob[k]['development_share_kochi_core']}")
res["global"] = glob

# built = dry check: mean SHAP of paving / built cells
built = (X["landcover_class"].to_numpy() == 6)
paved = X["impervious_frac"].to_numpy() >= 0.3
chk = {}
for k, (sv, feats) in S.items():
    f_i = {f: j for j, f in enumerate(feats)}
    row = {}
    if "landcover_class" in f_i:
        row["landcover_SHAP_mean_on_built_cells"] = round(float(sv[built, f_i["landcover_class"]].mean()), 4)
    if "impervious_frac" in f_i:
        row["impervious_SHAP_mean_on_paved_cells"] = round(float(sv[paved, f_i["impervious_frac"]].mean()), 4)
        row["impervious_SHAP_min"] = round(float(sv[:, f_i["impervious_frac"]].min()), 4)
    from scipy.stats import spearmanr
    for f in ("impervious_frac", "uica_ratio"):
        if f in f_i and np.std(sv[:, f_i[f]]) > 0:
            # direction of the learned effect: + = more paving raises risk, - = lowers it ("built = dry")
            row[f"spearman_value_vs_SHAP_{f}"] = round(float(spearmanr(X[f], sv[:, f_i[f]]).correlation), 3)
        elif f in f_i:
            row[f"spearman_value_vs_SHAP_{f}"] = "feature not used (SHAP constant 0)"
    chk[k] = row
    print(f"    built=dry check {k}: {row}")
res["built_equals_dry_check"] = chk

# ---------------------------------------------------------------------------
# Local places and the 2017 -> 2030 change at hotspots (M3)
# ---------------------------------------------------------------------------
print("\n[3/4] Local explanations and development share of the change")
loc = {}
for p, ids in place_idx.items():
    out = {}
    for k in ("M1", "M3"):
        sv, feats = S[k]
        m = sv[sel(ids)].mean(axis=0)
        contrib = sorted(zip(feats, m[:-1]), key=lambda x: -abs(x[1]))
        prob = float(1 / (1 + np.exp(-m.sum())))
        out[k] = {"mean_logodds": round(float(m.sum()), 3), "approx_prob": round(prob, 4),
                  "top": [(f, round(float(v), 3)) for f, v in contrib[:6]],
                  "development_logodds": round(float(sum(v for f, v in zip(feats, m[:-1]) if f in DEV)), 3)}
    loc[p] = out
    print(f"    {p}: M1 top {out['M1']['top'][:3]}; M3 development {out['M3']['development_logodds']}")
res["local"] = loc

# development features of the hotspot cells under 2017 and 2030 trend (recomputed as in Step 9)
g = json.load(open(f"{C.PROC_DIR}/basin/ext_grid.json", encoding="utf-8"))
HX, WX = g["height"], g["width"]
with rasterio.open(f"{C.PROC_DIR}/basin/fabdem_ext.tif") as src:
    TX = src.transform
    dem = src.read(1)
_, landx = C.open_water_and_land(dem)
del dem
H30, W30 = land.shape
AOI = (slice(g["aoi_row0"], g["aoi_row0"] + H30), slice(g["aoi_col0"], g["aoi_col0"] + W30))
dx, dy, cell_area = C.row_cell_sizes_m(TX, HX)
area_m2 = np.broadcast_to(cell_area[:, None].astype(np.float32), (HX, WX))
with rasterio.open(f"{C.PROC_DIR}/landcover/impervious_fraction_ext_30m.tif") as src:
    nm = list(src.descriptions)
    i17 = src.read(nm.index("impervious_2017_x200") + 1).astype(np.float32) / 200
    i25 = src.read(nm.index("impervious_2025_x200") + 1).astype(np.float32) / 200
i30 = np.clip(i25 + 5 / 8 * np.maximum(i25 - i17, 0), 0, 1)
i17[~landx] = 0; i30[~landx] = 0
zf = np.load(f"{C.PROC_DIR}/basin/zf_ext.npy")
flat = np.flatnonzero(landx.ravel())
order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
del flat
u = {}
for nm_, im in (("2017", i17), ("2030", i30)):
    a, *_ = C.quinn_fd8_weighted(zf, landx, order, (im * area_m2).astype(np.float64), dx, dy)
    u[nm_] = (a[AOI] / 1e6).astype(np.float32)
    del a
    gc.collect()
acc = np.load(f"{C.PROC_DIR}/basin/acc_ext.npy", mmap_mode="r")[AOI] / 1e6
hr, hc = rr[hot_idx], cc[hot_idx]
Xh = lm.iloc[hot_idx].reset_index(drop=True)
scen = {}
for nm_, im in (("2017", i17), ("2030", i30)):
    x = Xh.copy()
    x["impervious_frac"] = im[AOI][hr, hc]
    x["uica_km2"] = u[nm_][hr, hc]
    x["uica_ratio"] = np.clip(u[nm_][hr, hc] / np.maximum(acc[hr, hc], 1e-6), 0, 1)
    x["duica_since_2017_km2"] = np.maximum(u[nm_][hr, hc] - u["2017"][hr, hc], 0)
    scen[nm_] = m3.predict(x[F3], pred_contrib=True)
dlt = scen["2030"] - scen["2017"]
fi = {f: j for j, f in enumerate(F3)}
local_ch = dlt[:, fi["impervious_frac"]]
up_ch = dlt[:, [fi["uica_km2"], fi["uica_ratio"], fi["duica_since_2017_km2"]]].sum(axis=1)
tot = dlt[:, :-1].sum(axis=1)
p17 = 1 / (1 + np.exp(-scen["2017"].sum(axis=1)))
p30 = 1 / (1 + np.exp(-scen["2030"].sum(axis=1)))
res["hotspot_change_2017_2030_M3"] = {
    "n_cells": int(len(hot_idx)), "mean_logodds_change": round(float(tot.mean()), 4),
    "mean_prob_2017": round(float(p17.mean()), 4), "mean_prob_2030": round(float(p30.mean()), 4),
    "share_of_change_from_local_paving": round(float(local_ch.sum() / max(tot.sum(), 1e-9)), 3),
    "share_of_change_from_upstream_paving": round(float(up_ch.sum() / max(tot.sum(), 1e-9)), 3),
    "share_cells_risk_up": round(float((tot > 1e-6).mean()), 3)}
print(f"    hotspots 2017 -> 2030 (M3): {res['hotspot_change_2017_2030_M3']}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[4/4] Figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
gcol = {"terrain & hydrology": C.C_BLUE, "roads": C.C_GREY, "event rainfall": C.C_GREEN,
        "annual rainfall (location proxy)": C.C_YELLOW, "land cover": "#8a5cc2", "development (paving, UICA)": C.C_ORANGE}
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
labels = {"M1": "M1 chosen model\n(radar labels, set C)", "M2": "M2 unconstrained D\n(radar + NDEM)", "M3": "M3 scenario model\n(monotone)"}
left = np.zeros(3)
for gname, col in gcol.items():
    vals = np.array([glob[k]["group_share"][gname] for k in ("M1", "M2", "M3")]) * 100
    ax.barh([labels[k] for k in ("M1", "M2", "M3")][::-1], vals[::-1], left=left[::-1], color=col, label=gname)
    left += vals
ax.set_xlabel("share of model decisions (mean |SHAP|, %)"); ax.set_xlim(0, 100)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
ax.set_title("11.1 What the models rely on")
ax = fig.add_subplot(gs[0, 1])
top = list(glob["M2"]["feature_share"].items())[:12][::-1]
ax.barh([t[0] for t in top], [100 * t[1] for t in top],
        color=[gcol[next(g for g, fs in GROUPS.items() if t[0] in fs)] for t in top])
ax.set_xlabel("share (%)"); ax.set_title("Top features, M2 (unconstrained D)")
ax = fig.add_subplot(gs[0, 2])
xs = np.arange(3)
w = 0.27
for j, (lab, key) in enumerate((("all land", None), ("hotspots (2030)", "development_share_hotspots"),
                                ("Kochi core", "development_share_kochi_core"))):
    vals = [100 * (glob[k]["group_share"]["development (paving, UICA)"] if key is None else glob[k][key]) for k in ("M1", "M2", "M3")]
    ax.bar(xs + (j - 1) * w, vals, w, label=lab, color=[C.C_GREY, C.C_ORANGE, C.C_BLUE][j])
    for xi, v in zip(xs, vals):
        ax.text(xi + (j - 1) * w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["M1", "M2", "M3"]); ax.set_ylabel("development share of decisions (%)")
ax.set_title("Development's share (riverine-flood models)"); ax.legend(fontsize=8)
for j, (k, title) in enumerate((("M2", "M2 unconstrained: paving pushes risk DOWN"), ("M3", "M3 scenario model: never down"))):
    ax = fig.add_subplot(gs[1, j])
    sv, feats = S[k]
    f_i = {f: i for i, f in enumerate(feats)}
    ax.scatter(X["impervious_frac"], sv[:, f_i["impervious_frac"]], s=2, alpha=0.3, color=C.C_ORANGE, label="paved fraction")
    ax.scatter(X["uica_ratio"], sv[:, f_i["uica_ratio"]], s=2, alpha=0.3, color=C.C_BLUE, label="UICA ratio")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("feature value"); ax.set_ylabel("SHAP (log-odds)"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=4)
ax = fig.add_subplot(gs[1, 2])
names = list(PLACES)
dev_vals = [loc[p]["M3"]["development_logodds"] for p in names]
tot_vals = [loc[p]["M3"]["mean_logodds"] for p in names]
ax.barh(names[::-1], tot_vals[::-1], color=C.C_GREY, label="total (log-odds)")
ax.barh(names[::-1], dev_vals[::-1], color=C.C_ORANGE, label="development part")
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Named places (M3): total score vs development part"); ax.legend(fontsize=8)
fig.suptitle("Step 11  SHAP: how the riverine-flood models use their inputs (development is a small part of big-flood prediction)",
             fontsize=11)
fig.savefig(f"{C.MAP_DIR}/step11_shap_global.png", dpi=140)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step11_shap.json")
print(f"\nDone in {res['runtime_s']} s")
