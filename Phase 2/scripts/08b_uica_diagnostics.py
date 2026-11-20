"""
Phase 2 Step 8b - Why does UICA not add skill? (Step 8.3 follow-up, plan: "investigate").

D (C + UICA / dUICA) equals C on every held-out test. Three checks:
  1. Do the labels even reach UICA-relevant land? UICA ratio and dUICA of the
     positives vs random land and vs the Kochi core.
  2. How much does model D use the UICA features? (gain share, tuned LightGBM D)
  3. D vs C restricted to urban-influenced land (UICA ratio >= 0.10), strict test.

Output: outputs/metrics/step8b_uica_diagnostics.json, outputs/maps/step8b_uica_diagnostics.png
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import common as C

C.banner("PHASE 2 STEP 8b: UICA DIAGNOSTICS")
t0 = time.time()
df = pd.read_parquet(C.DATASET_V2)
s8 = json.load(open(f"{C.METRICS_DIR}/step8_model_v2_ablation.json", encoding="utf-8"))
b = s8["tuning"]["best"]
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": b["num_leaves"], "min_data_in_leaf": b["min_data_in_leaf"],
         "lambda_l2": b["lambda_l2"], "feature_fraction": 0.8}
ROUNDS = s8["tuning"]["rounds"]
UICA_COLS = ["uica_km2", "uica_ratio", "duica_since_2017_km2"]
URBAN = 0.10
res = {"step": "8b", "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

# 1. where the labels are
groups = {"flood (pos)": df.set == "pos", "random land": df.set == "rnd",
          "Kochi core (random land)": (df.set == "rnd") & df.in_kochi}
dist = {}
for g, m in groups.items():
    d = df[m]
    dist[g] = {"median_uica_ratio": round(float(d.uica_ratio.median()), 4),
               "share_uica_ratio_ge_0.10": round(float((d.uica_ratio >= URBAN).mean()), 4),
               "share_duica_gt_0.01km2": round(float((d.duica_since_2017_km2 > 0.01).mean()), 4),
               "median_impervious": round(float(d.impervious_frac.median()), 4)}
    print(f"    {g:26s} {dist[g]}")
res["label_reach"] = dist
n_pos_urban = int(((df.set == "pos") & (df.uica_ratio >= URBAN)).sum())
res["positives_on_urban_influenced_land"] = n_pos_urban

# 2. gain share in model D
m = lgb.Booster(model_file=f"{C.MODEL_DIR}/model_v2_D.txt")
gain = pd.Series(m.feature_importance("gain"), index=m.feature_name())
gain = (gain / gain.sum()).sort_values(ascending=False)
res["gain_share_D"] = {k: round(float(v), 4) for k, v in gain.items()}
res["gain_share_uica_total"] = round(float(gain[UICA_COLS].sum()), 4)
print(f"    UICA features: {res['gain_share_uica_total']:.1%} of model D gain")

# 3. strict test restricted to urban-influenced land
sub = {}
for v in ("C", "D"):
    fp = C.make_fit_predict("standard", C.FEATURE_SETS[v], TUNED, ROUNDS)
    te_all = df[df.event.isin(C.TEST_EVENTS)]
    sc = pd.Series(np.nan, index=te_all.index)
    for k in range(C.N_FOLDS):
        tr = df[df.event.isin(C.TRAIN_EVENTS) & (df.fold != k)]
        te = te_all[te_all.fold == k]
        sc.loc[te.index] = fp(tr, te)
    u = te_all[(te_all.uica_ratio >= URBAN) & (te_all.set.isin(["pos", "rnd"]))]
    y = u.is_pos.astype(int).to_numpy()
    sub[v] = {"n_cells": int(len(u)), "n_flood": int(y.sum()),
              "auc_flood_vs_land": round(float(roc_auc_score(y, sc.loc[u.index])), 4) if 0 < y.sum() < len(y) else None}
    print(f"    {v}: urban-influenced test cells {sub[v]}")
res["strict_urban_influenced"] = sub

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
edges = [0, 0.01, 0.05, 0.10, 0.30, 1.01]
names = ["< 0.01", "0.01-0.05", "0.05-0.10", "0.10-0.30", "> 0.30"]
w = 0.27
for gi, (g, col) in enumerate(zip(groups, (C.C_BLUE, C.C_GREY, C.C_ORANGE))):
    sh = np.histogram(df.loc[groups[g], "uica_ratio"], bins=edges)[0] / groups[g].sum()
    axes[0].bar(np.arange(5) + (gi - 1) * w, sh, w, color=col, label=g)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(names)
axes[0].set_xlabel("UICA ratio (share of the catchment that is paved)"); axes[0].set_ylabel("share of cells")
axes[0].set_title("Flood labels sit where UICA is low; Kochi is high"); axes[0].legend(fontsize=8)
axes[0].grid(axis="y", color=C.C_GRID); axes[0].set_axisbelow(True)
top = gain.head(12)[::-1]
axes[1].barh(top.index, top.values, color=[C.C_ORANGE if i in UICA_COLS else C.C_BLUE for i in top.index])
axes[1].set_title(f"Model D gain share (UICA total {res['gain_share_uica_total']:.1%})")
axes[1].set_xlabel("share of total gain")
fig.suptitle("Step 8b  Why UICA adds no measurable skill with radar-only labels", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step8b_uica_diagnostics.png", dpi=150)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step8b_uica_diagnostics.json")
print(f"Done in {res['runtime_s']} s")
