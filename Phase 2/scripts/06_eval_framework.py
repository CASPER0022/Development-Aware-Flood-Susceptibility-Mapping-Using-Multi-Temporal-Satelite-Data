"""
Phase 2 Step 6 - Evaluation framework (Week 3).

6.1 Spatial block CV as the standard: one fixed 5-fold StratifiedGroupKFold
    over 10 km UTM blocks, saved to disk (block -> fold) so every later model
    is tested on exactly the same folds.
6.2 Temporal hold-out: train on 2018 + 2019, test on 2020 and 2024 (Step 3).
6.3 Metrics: accuracy, precision, recall, F1, AUC (base-paper set) plus the
    2019-style map checks on held-out events: share of flood in the High +
    Very High classes (top 40 % of land) and frequency ratio per class.
6.4 Re-score the Phase 1 baseline (11 features, Phase 1 settings, 648 rounds)
    under this framework. That number is the bar for every Phase 2 model.

Dataset v2 (also used by Steps 7 and 8). For each kept event, from that
event's Sentinel-1 class raster and NDEM envelope (Phase 1 Step 8 rules):
    pos   every confirmed flood cell                      (label 1)
    neg   confident non-flood cells, as many as pos       (label 0, standard design)
    rnd   uniform random land cells (60,000)              (PU 'unlabelled' + land percentiles)
    ndem  NDEM-only flood cells: inside the NDEM map but no radar flood
          (peak water we missed; evaluation only, up to 20,000)
Features: the 18 stack-v2 bands of that event + Phase 1 'twi' and
'rainfall_mean_annual_mm', so models A-D read one table.

Outputs
    data/processed/training_v2/samples_v2.parquet      (gitignored, regenerable)
    data/processed/training_v2/label_pools_<event>.tif (0 neg, 1 pos, 2 ndem-only, 255 other)
    outputs/metrics/step6_fold_assignment.json
    outputs/metrics/step6_eval_framework.json
    outputs/maps/step6_folds_and_labels.png, step6_baseline_rescored.png
"""
import json
import time

import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import common as C

N_RND = 60000
N_NDEM_MAX = 20000
P1_ROUNDS = 648

C.banner("PHASE 2 STEP 6: EVALUATION FRAMEWORK + DATASET V2 + PHASE 1 BASELINE RE-SCORED")
t0 = time.time()
rng = np.random.default_rng(C.SEED)
res = {"step": 6, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
ev_json = json.load(open(f"{C.METRICS_DIR}/step3_flood_inventory_events.json", encoding="utf-8"))
ev_info = {e["event"]: e for e in ev_json["events"] if e["status"] == "kept"}
meta = json.load(open(f"{C.PROC_DIR}/features_v2/features_v2_metadata.json", encoding="utf-8"))
OUT = f"{C.PROC_DIR}/training_v2"
import os
os.makedirs(OUT, exist_ok=True)

with rasterio.open(C.P1_STACK) as src:
    p1 = {n: src.read(i + 1) for i, n in enumerate(src.descriptions)}
    T30, H30, W30 = src.transform, src.height, src.width
    prof30 = src.profile
with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
ndem = C.load_ndem()

# ---------------------------------------------------------------------------
# Dataset v2
# ---------------------------------------------------------------------------
print("\n[1/4] Label pools and dataset v2")
frames, pool_stats = [], {}
for ev in C.EVENTS_V2:
    info = ev_info[ev]
    with rasterio.open(os.path.join(C.PHASE2_ROOT, info["selected"]["water_classes"])) as src:
        cls = src.read(1)
        T20 = src.transform
    C.check_20_30_commensurate(T20, T30)
    env, n_poly = C.ndem_mask(ndem, info["ndem_dates"], cls.shape, T20)
    pos, neg, ndo = C.label_pools(cls, env, land, H30, W30)
    pr = np.full((H30, W30), 255, np.uint8)
    pr[neg] = 0; pr[ndo] = 2; pr[pos] = 1
    with rasterio.open(f"{OUT}/label_pools_{ev}.tif", "w", **{**prof30, "count": 1, "dtype": "uint8",
                                                            "nodata": 255, "compress": "deflate"}) as dst:
        dst.write(pr, 1)

    pos_i = np.flatnonzero(pos.ravel())
    neg_i = rng.choice(np.flatnonzero(neg.ravel()), size=len(pos_i), replace=False)
    rnd_i = rng.choice(np.flatnonzero(land.ravel()), size=N_RND, replace=False)
    ndo_all = np.flatnonzero(ndo.ravel())
    ndo_i = rng.choice(ndo_all, size=min(N_NDEM_MAX, len(ndo_all)), replace=False)
    with rasterio.open(f"{C.PROC_DIR}/features_v2/{meta['files'][ev]}") as src:
        names = list(src.descriptions)
        stack = src.read()
    parts = []
    for tag, idx in (("pos", pos_i), ("neg", neg_i), ("rnd", rnd_i), ("ndem", ndo_i)):
        r, c = np.unravel_index(idx, (H30, W30))
        d = pd.DataFrame({n: stack[k][r, c] for k, n in enumerate(names)})
        d["twi"] = p1["twi"][r, c]
        d["rainfall_mean_annual_mm"] = p1["rainfall_mean_annual_mm"][r, c]
        d.insert(0, "set", tag); d.insert(1, "event", ev); d.insert(2, "row", r); d.insert(3, "col", c)
        d["is_pos"] = pos[r, c]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lon, lat, x, y = C.cell_xy_utm(d["row"].to_numpy(), d["col"].to_numpy(), T30)
    d["lon"], d["lat"], d["x_utm"], d["y_utm"] = lon, lat, x, y
    d["block10"] = C.block10(x, y)
    kb = C.KOCHI_BOX
    d["in_kochi"] = (lon >= kb[0]) & (lon <= kb[2]) & (lat >= kb[1]) & (lat <= kb[3])
    frames.append(d)
    row_km2 = C.pixel_area_km2(T30, H30)
    pool_stats[ev] = {"date": info["selected"]["date"], "ndem_polygons": n_poly,
                      "pos_cells": int(pos.sum()), "pos_km2": round(C.area_km2(pos, row_km2), 2),
                      "neg_pool_cells": int(neg.sum()), "ndem_only_cells": int(ndo.sum()),
                      "ndem_only_km2": round(C.area_km2(ndo, row_km2), 2),
                      "pos_built_share": round(float((stack[names.index("landcover_class")][pos] == 6).mean()), 3),
                      "rnd_pos_share": round(float(d.loc[d.set == "rnd", "is_pos"].mean()), 4)}
    print(f"    {ev}: pos {pos.sum():,} cells ({pool_stats[ev]['pos_km2']} km2), neg pool {neg.sum():,}, "
          f"NDEM-only {ndo.sum():,}, built share of pos {pool_stats[ev]['pos_built_share']:.2f}")

df = pd.concat(frames, ignore_index=True)
feat_cols = sorted(set(C.FEATURES_D))
assert df[feat_cols].notna().all().all() and np.isfinite(df[feat_cols].to_numpy(float)).all(), "gaps in features"
df["landcover_class"] = df["landcover_class"].astype(int)
res["pools"] = pool_stats

# ---------------------------------------------------------------------------
# 6.1 Fixed folds over 10 km blocks (stratified on the pooled standard rows)
# ---------------------------------------------------------------------------
print("\n[2/4] 6.1 Fixed 10 km block folds")
std = df[df.set.isin(["pos", "neg"])]
gk = StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
b2f = {}
for k, (_, te) in enumerate(gk.split(std, std.set.eq("pos"), std.block10)):
    for b in std.block10.iloc[te].unique():
        b2f[b] = k
for b in df.block10.unique():                      # blocks with no standard rows (rnd only)
    if b not in b2f:
        bx, by = map(int, b.split("_"))
        b2f[b] = (bx + by) % C.N_FOLDS
df["fold"] = df.block10.map(b2f).astype(int)
fold_tab = (std.assign(fold=std.block10.map(b2f)).groupby(["fold", "event"]).set.apply(lambda s: int((s == "pos").sum()))
            .unstack(fill_value=0))
print(fold_tab.to_string())
C.save_json({"block_m": C.FOLD_BLOCK_M, "n_folds": C.N_FOLDS, "seed": C.SEED, "block_to_fold": b2f,
             "positives_per_fold_event": {int(k): {e: int(v) for e, v in r.items()} for k, r in fold_tab.iterrows()}},
            C.FOLDS_JSON)
df.to_parquet(C.DATASET_V2, index=False)
res["dataset_v2"] = {"path": os.path.relpath(C.DATASET_V2, C.PHASE2_ROOT), "rows": len(df),
                     "rows_by_set_event": {f"{s}|{e}": int(n) for (s, e), n in df.groupby(["set", "event"]).size().items()},
                     "n_blocks10": int(df.block10.nunique()),
                     "positives_per_fold_event": {int(k): {e: int(v) for e, v in r.items()} for k, r in fold_tab.iterrows()}}

# ---------------------------------------------------------------------------
# 6.4 Phase 1 baseline under the framework
# ---------------------------------------------------------------------------
print("\n[3/4] 6.4 Phase 1 baseline (11 features, Phase 1 settings) re-scored")
FA = C.FEATURES_A


def fit(d, feats, params=C.LGB_PARAMS, rounds=P1_ROUNDS):
    return lgb.train(params, lgb.Dataset(d[feats], d.set.eq("pos").astype(int)), num_boost_round=rounds)


def std_rows(events, folds=None):
    m = df.set.isin(["pos", "neg"]) & df.event.isin(events)
    if folds is not None:
        m &= df.fold.isin(folds)
    return df[m]


def mets(d, p):
    return C.metrics(d.set.eq("pos").astype(int).to_numpy(), p)


base = {}
# (a) Phase 1 protocol for reference: random 70/30 split on 2018
d18 = std_rows(["2018-08_ndem"])
tr, te = train_test_split(d18, test_size=0.3, stratify=d18.set, random_state=C.SEED)
base["random_split_2018"] = C.rounded(mets(te, fit(tr, FA).predict(te[FA])))
# (b) spatial CV on 2018 only (the Phase 1 training event)
fm = []
for k in range(C.N_FOLDS):
    trd = std_rows(["2018-08_ndem"], [f for f in range(C.N_FOLDS) if f != k]); ted = std_rows(["2018-08_ndem"], [k])
    fm.append(mets(ted, fit(trd, FA).predict(ted[FA])))
base["spatial_cv_2018"] = C.cv_summary(fm)


def p1_fit_predict(tr, te):
    return fit(C.std_rows(tr), FA).predict(te[FA])


# (c) the three Phase 2 protocols (pooled spatial CV, temporal, strict new-year-and-new-place)
base.update(C.run_protocols(df, p1_fit_predict))
base["headline"] = C.headline(base)
print(f"    random split 2018      AUC {base['random_split_2018']['auc']:.4f}")
print(f"    10 km CV, 2018 only    AUC {base['spatial_cv_2018']['auc']['mean']:.4f}")
print(f"    10 km CV, 4 events     AUC {base['spatial']['auc']['mean']:.4f}  F1 {base['spatial']['f1']['mean']:.4f}")
for p_ in ("temporal", "strict"):
    print(f"    {p_:9s}: " + " | ".join(
        f"{e}: AUC {m['auc']:.3f} (vs all land {m['auc_pos_vs_all_land']:.3f}), F1 {m['f1']:.3f}, "
        f"H+VH {m['pos_in_high_vhigh']:.2f}, NDEM-only H+VH {m.get('ndem_only_in_high_vhigh', float('nan')):.2f}"
        for e, m in base[p_].items()))
res["phase1_baseline"] = base
res["framework"] = {"spatial": f"5-fold StratifiedGroupKFold over {C.FOLD_BLOCK_M // 1000} km UTM blocks, fixed in "
                               "step6_fold_assignment.json, same folds for every model",
                    "temporal": {"train": C.TRAIN_EVENTS, "test": C.TEST_EVENTS},
                    "metrics": ["accuracy", "precision", "recall", "f1", "auc", "specificity",
                                "share of held-out flood in High+Very High (top 40 % of land)",
                                "frequency ratio per quintile class"],
                    "strict": "train 2018+2019 outside fold k, test 2020/2024 inside fold k (new year and new place)",
                    "bar": base["headline"]}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[4/4] Figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
fig, axes = plt.subplots(1, 5, figsize=(17, 4.3), constrained_layout=True)
fr = np.full((H30, W30), np.nan)
rr, cc = np.nonzero(land)
step = 7
rs, cs = rr[::step], cc[::step]
_, _, xs, ys = C.cell_xy_utm(rs, cs, T30)
fr[rs, cs] = [b2f.get(b, np.nan) for b in C.block10(xs, ys)]
from scipy import ndimage
idx = ndimage.distance_transform_edt(np.isnan(fr), return_distances=False, return_indices=True)
fr = np.where(land, fr[idx[0], idx[1]], np.nan)
fold_cm = ListedColormap(["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc2"])
axes[0].imshow(fr, cmap=fold_cm, vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
axes[0].set_title("Fixed folds (10 km blocks)")
for ax, ev in zip(axes[1:], C.EVENTS_V2):
    with rasterio.open(f"{OUT}/label_pools_{ev}.tif") as src:
        pr = src.read(1)
    img = np.full(pr.shape + (3,), 1.0)
    img[~land] = (0.80, 0.86, 0.93)
    img[pr == 0] = (0.93, 0.93, 0.93)
    img[pr == 2] = (0.96, 0.62, 0.20)
    img[pr == 1] = (0.10, 0.35, 0.80)
    ax.imshow(img, extent=ext, interpolation="nearest")
    s = pool_stats[ev]
    ax.set_title(f"{C.EVENT_LABEL[ev]} ({'train' if ev in C.TRAIN_EVENTS else 'temporal test'})\n"
                 f"{s['pos_km2']} km² flood · {s['ndem_only_km2']} km² NDEM-only")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Step 6  Fixed spatial folds and per-event labels. Blue: confirmed flood (positives); grey: confident "
             "non-flood pool; orange: NDEM-only flood (evaluation only)", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step6_folds_and_labels.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.0), constrained_layout=True)
labels = ["Random split\n(2018,\nPhase 1 way)", "10 km blocks\n(2018)", "10 km blocks\n(4 events)",
          "New year\n2020", "New year\n2024", "New year +\nnew place\n2020", "New year +\nnew place\n2024"]
vals = [base["random_split_2018"]["auc"], base["spatial_cv_2018"]["auc"]["mean"], base["spatial"]["auc"]["mean"],
        base["temporal"]["2020"]["auc_pos_vs_all_land"], base["temporal"]["2024"]["auc_pos_vs_all_land"],
        base["strict"]["2020"]["auc_pos_vs_all_land"], base["strict"]["2024"]["auc_pos_vs_all_land"]]
bars = ax.bar(labels, vals, color=[C.C_GREY, C.C_BLUE, C.C_BLUE, C.C_ORANGE, C.C_ORANGE, C.C_GREEN, C.C_GREEN])
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.6, 1.02); ax.set_ylabel("AUC")
ax.text(0.99, 0.97, "Orange / green: flood vs all other land (map-level, like the Phase 1 2019 check)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C.C_GREY)
ax.set_title("Step 6  Phase 1 baseline (11 features) under the new evaluation framework")
ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
fig.savefig(f"{C.MAP_DIR}/step6_baseline_rescored.png", dpi=150)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step6_eval_framework.json")
print(f"\nDone in {res['runtime_s']} s")
