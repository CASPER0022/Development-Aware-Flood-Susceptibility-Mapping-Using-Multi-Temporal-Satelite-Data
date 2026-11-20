"""
Phase 2 Step 9.0b - Physically constrained scenario model.

The radar labels taught the model "built = dry", so a paving scenario could
lower predicted risk. Physics says paving cannot make water soak in better, so:
  - monotone constraints: risk is non-decreasing in impervious_frac, uica_km2,
    uica_ratio and duica_since_2017_km2 (LightGBM monotone_constraints,
    method 'advanced');
  - land-cover class removed as a free signal (its 'built' class carries the bias).

Base: Step 9.0a choice = feature set D, radar + NDEM labels, standard design,
tuned Step 8 settings, fixed 10 km folds.

Variants:   D            unconstrained (9.0a choice)
            D-LC         land cover removed
            D+mono       monotone, land cover kept
            D-LC+mono    monotone, land cover removed   (scenario model candidate)

Checks:
  1. Cost: spatial CV and strict (2020, 2024) AUC vs radar + NDEM truth,
     on all land and urban subsets.
  2. Stress test on 100,000 random 2024 land cells that are < 20 % paved:
     'develop' them (paved -> max(paved, 0.6), upstream paved area +50 %,
     UICA ratio +0.2, dUICA + 50 % of UICA, land cover -> built) and count cells
     whose predicted risk FALLS. Physics allows none.
  3. Response curves: mean predicted risk as paved fraction / UICA ratio is
     swept 0 -> 1 on 2,000 cells (ICE averages).

Outputs: outputs/metrics/step9b_constrained_model.json,
         outputs/models/model_v2_scenario.txt,
         outputs/maps/step9b_constraint_checks.png
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import common as C

C.banner("PHASE 2 STEP 9.0b: MONOTONE-CONSTRAINED SCENARIO MODEL")
t0 = time.time()
df = C.ndem_label_dataset()
s8 = json.load(open(f"{C.METRICS_DIR}/step8_model_v2_ablation.json", encoding="utf-8"))
b = s8["tuning"]["best"]
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": b["num_leaves"], "min_data_in_leaf": b["min_data_in_leaf"],
         "lambda_l2": b["lambda_l2"], "feature_fraction": 0.8}
ROUNDS = s8["tuning"]["rounds"]
DEV = ["impervious_frac", "uica_km2", "uica_ratio", "duica_since_2017_km2"]
FD = C.FEATURES_D
FD_noLC = [f for f in FD if f != "landcover_class"]
VARIANTS = {"D": (FD, False), "D-LC": (FD_noLC, False), "D+mono": (FD, True), "D-LC+mono": (FD_noLC, True)}
SUBSETS = {"all land": lambda d: np.ones(len(d), bool),
           "DW built": lambda d: (d.landcover_class == 6).to_numpy(),
           "paved >= 20 %": lambda d: (d.impervious_frac >= 0.2).to_numpy(),
           "UICA ratio >= 0.10": lambda d: (d.uica_ratio >= 0.10).to_numpy()}
res = {"step": "9.0b", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "monotone_features": DEV,
       "labels": "radar + NDEM (Step 9.0a)", "design": "standard", "rounds": ROUNDS}


def params_for(feats, mono):
    p = dict(TUNED)
    if mono:
        p["monotone_constraints"] = [1 if f in DEV else 0 for f in feats]
        p["monotone_constraints_method"] = "advanced"
    return p


def fit(tr, feats, mono):
    d = C.std_rows(tr)
    return C.lgb_fit(d, feats, C.y_of(d), params_for(feats, mono), ROUNDS)


def score_eval(te, sc):
    ev_rows = te[(te.set == "pos") | ((te.set == "rnd") & ~te.is_pos2)]
    y = (ev_rows.set == "pos").to_numpy().astype(int)
    s = sc.loc[ev_rows.index].to_numpy()
    out = {}
    for name, f in SUBSETS.items():
        m = f(ev_rows)
        out[name] = round(float(roc_auc_score(y[m], s[m])), 4) if y[m].sum() >= 30 else None
    return out


# ---------------------------------------------------------------------------
# 1. Cost of the constraints
# ---------------------------------------------------------------------------
print("\n[1/3] Accuracy cost (AUC vs radar + NDEM truth)")
cost = {}
for v, (feats, mono) in VARIANTS.items():
    sc = pd.Series(np.nan, index=df.index)
    for k in range(C.N_FOLDS):
        te = df[df.fold == k]
        sc.loc[te.index] = fit(df[df.fold != k], feats, mono).predict(te[feats])
    out = {"spatial": score_eval(df, sc)}
    te_all = df[df.event.isin(C.TEST_EVENTS)]
    sc = pd.Series(np.nan, index=te_all.index)
    for k in range(C.N_FOLDS):
        te = te_all[te_all.fold == k]
        sc.loc[te.index] = fit(df[df.event.isin(C.TRAIN_EVENTS) & (df.fold != k)], feats, mono).predict(te[feats])
    for e in C.TEST_EVENTS:
        out[f"strict_{C.EVENT_LABEL[e]}"] = score_eval(te_all[te_all.event == e], sc)
    cost[v] = out
    print(f"    {v:10s} spatial {out['spatial']['all land']:.4f} | strict 2020 " +
          " ".join(f"{n}: {out['strict_2020'][n]}" for n in SUBSETS) + f" | strict 2024 all {out['strict_2024']['all land']}")
res["cost"] = cost

# ---------------------------------------------------------------------------
# 2. Stress test: develop open land, count cells that get SAFER
# ---------------------------------------------------------------------------
print("\n[2/3] Development stress test on 2024 land")
lm, rr, cc, land = C.land_matrix("2024-08_ndem")
rng = np.random.default_rng(C.SEED)
open_idx = np.flatnonzero(lm["impervious_frac"].to_numpy() < 0.2)
samp = lm.iloc[rng.choice(open_idx, size=min(100000, len(open_idx)), replace=False)].reset_index(drop=True)
dev = samp.copy()
dev["impervious_frac"] = np.maximum(dev["impervious_frac"], 0.6)
dev["uica_km2"] = dev["uica_km2"] * 1.5
dev["uica_ratio"] = np.minimum(dev["uica_ratio"] + 0.2, 1.0)
dev["duica_since_2017_km2"] = dev["duica_since_2017_km2"] + 0.5 * samp["uica_km2"]
dev["landcover_class"] = 6
tr = df[df.event.isin(C.TRAIN_EVENTS)]
models, stress = {}, {}
for v, (feats, mono) in VARIANTS.items():
    m = fit(tr, feats, mono)
    models[v] = m
    p0, p1 = m.predict(samp[feats]), m.predict(dev[feats])
    d = p1 - p0
    stress[v] = {"share_cells_safer": round(float((d < -1e-9).mean()), 4),
                 "share_cells_riskier": round(float((d > 1e-9).mean()), 4),
                 "mean_change": round(float(d.mean()), 4), "median_change": round(float(np.median(d)), 4)}
    print(f"    {v:10s} {stress[v]}")
res["stress_test"] = {"n_cells": len(samp), "scenario": "paved -> max(paved, 0.6); UICA x1.5; UICA ratio +0.2; "
                      "dUICA + 0.5 UICA; land cover -> built", "results": stress}

# ---------------------------------------------------------------------------
# 3. Response curves
# ---------------------------------------------------------------------------
print("\n[3/3] Response curves")
ice = samp.sample(2000, random_state=C.SEED)
grid = np.linspace(0, 1, 21)
curves = {}
for feat in ("impervious_frac", "uica_ratio"):
    curves[feat] = {}
    for v, (feats, _) in VARIANTS.items():
        vals = []
        for g in grid:
            x = ice.copy()
            x[feat] = g
            vals.append(float(models[v].predict(x[feats]).mean()))
        curves[feat][v] = [round(a, 5) for a in vals]
        steps = np.diff(vals)
        curves[feat][f"{v}_drops"] = int((steps < -1e-6).sum())
    print(f"    {feat}: drops along the curve " + ", ".join(f"{v} {curves[feat][f'{v}_drops']}" for v in VARIANTS))
res["response_curves"] = {"grid": grid.round(2).tolist(), **curves}

# ---------------------------------------------------------------------------
# Choice
# ---------------------------------------------------------------------------
ref = cost["D"]["strict_2020"]["all land"]
cand = cost["D-LC+mono"]["strict_2020"]["all land"]
ok = stress["D-LC+mono"]["share_cells_safer"] == 0 and (ref - cand) < 0.01
res["choice"] = {"scenario_model": "D-LC+mono" if ok else "review",
                 "strict2020_cost_vs_D": round(ref - cand, 4),
                 "rule": "no cell may get safer in the stress test, and strict-2020 cost < 0.01 AUC"}
print(f"\n  Scenario model: {res['choice']}")
models["D-LC+mono"].save_model(f"{C.MODEL_DIR}/model_v2_scenario.txt")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
cols = {"D": C.C_GREY, "D-LC": C.C_YELLOW, "D+mono": C.C_GREEN, "D-LC+mono": C.C_BLUE}
fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
for ax, feat, t in ((axes[0], "impervious_frac", "Response to local paved fraction"),
                    (axes[1], "uica_ratio", "Response to upstream paved share (UICA ratio)")):
    for v in VARIANTS:
        ax.plot(grid, curves[feat][v], color=cols[v], lw=2, label=v)
    ax.set_xlabel(feat); ax.set_ylabel("mean predicted risk"); ax.set_title(t); ax.grid(color=C.C_GRID)
axes[0].legend(fontsize=8)
v_ = list(VARIANTS)
axes[2].bar(range(4), [100 * stress[v]["share_cells_safer"] for v in v_], color=[cols[v] for v in v_])
for i, v in enumerate(v_):
    axes[2].text(i, 100 * stress[v]["share_cells_safer"] + 0.5, f"{100 * stress[v]['share_cells_safer']:.1f} %", ha="center")
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(v_)
axes[2].set_title("Stress test: cells that get SAFER\nwhen paved (physics allows 0 %)"); axes[2].set_ylabel("% of 100,000 cells")
names = list(SUBSETS)
w = 0.2
for j, v in enumerate(v_):
    axes[3].bar(np.arange(4) + (j - 1.5) * w, [cost[v]["strict_2020"][n] for n in names], w, color=cols[v], label=v)
axes[3].set_xticks(range(4)); axes[3].set_xticklabels(["all land", "DW built", "paved\n>= 20 %", "UICA ratio\n>= 0.10"])
lo = min(cost[v]["strict_2020"][n] for v in v_ for n in names)
axes[3].set_ylim(lo - 0.03, max(cost[v]["strict_2020"][n] for v in v_ for n in names) + 0.02)
axes[3].set_title("Cost: strict 2020 AUC"); axes[3].grid(axis="y", color=C.C_GRID); axes[3].set_axisbelow(True)
fig.suptitle("Step 9.0b  Physically constrained model (D, radar + NDEM labels). Mono = risk may not fall as paving rises; "
             "-LC = land-cover class removed", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step9b_constraint_checks.png", dpi=150)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step9b_constrained_model.json")
print(f"\nDone in {res['runtime_s']} s")
