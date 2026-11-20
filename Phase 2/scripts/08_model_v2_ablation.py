"""
Phase 2 Step 8 - Model v2 and the development ablation (Week 3).

8.1 Tune LightGBM under spatial CV. Small grid (num_leaves, min_data_in_leaf,
    lambda_l2), learning rate 0.05, rounds by early stopping. Tuning sees ONLY
    the training events (2018 + 2019, 10 km folds), so the temporal and strict
    tests on 2020 / 2024 stay untouched. Train-test gap reported for the Phase 1
    settings and for the tuned settings.
8.2 Comparison model: XGBoost (hist), same features, same folds.
8.3 Ablation A -> B -> C -> D (+ D without the annual-rainfall location
    proxy), each under spatial CV, temporal and strict (new year + new place),
    with the standard design and with PU bagging (Step 7's best).
    Location-proxy check: random-split AUC vs spatial AUC per model.
8.4 Calibration of the chosen model: prior correction for the balanced
    sampling, then Platt and isotonic scaling fitted on out-of-fold scores of
    random land cells of the training events; checked on 2020 / 2024 random
    land cells (reliability curve, Brier score, ECE). The probability is
    "P(cell is in the confirmed flood extent of an event like this one)".

Outputs
    outputs/metrics/step8_model_v2_ablation.json
    outputs/models/model_v2_<variant>.txt (LightGBM, trained on 2018 + 2019)
    outputs/maps/step8_ablation.png, step8_tuning_gap.png, step8_calibration.png,
                 step8_maps_A_vs_chosen.png, step8b (see 08b script)
"""
import itertools
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

import common as C

C.banner("PHASE 2 STEP 8: MODEL V2, ABLATION A-D, CALIBRATION")
t0 = time.time()
df = pd.read_parquet(C.DATASET_V2)
res = {"step": 8, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
tr_ev = df[df.event.isin(C.TRAIN_EVENTS)]

# ---------------------------------------------------------------------------
# 8.1 Tuning under spatial CV (training events only)
# ---------------------------------------------------------------------------
print("\n[1/5] 8.1 LightGBM tuning, 10 km CV on 2018 + 2019 only")
GRID = {"num_leaves": [15, 31, 63], "min_data_in_leaf": [20, 200, 1000], "lambda_l2": [1.0, 10.0]}
FT = C.FEATURES_D


def cv_tune(params, feats=FT):
    aucs, its, tr_aucs = [], [], []
    for k in range(C.N_FOLDS):
        a, b = C.std_rows(tr_ev[tr_ev.fold != k]), C.std_rows(tr_ev[tr_ev.fold == k])
        if not len(b):
            continue
        cat = [f for f in feats if f in C.CATEGORICAL]
        dtr = lgb.Dataset(a[feats], C.y_of(a), categorical_feature=cat or "auto", free_raw_data=False)
        dva = lgb.Dataset(b[feats], C.y_of(b), reference=dtr)
        m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        aucs.append(roc_auc_score(C.y_of(b), m.predict(b[feats], num_iteration=m.best_iteration)))
        tr_aucs.append(roc_auc_score(C.y_of(a), m.predict(a[feats], num_iteration=m.best_iteration)))
        its.append(m.best_iteration)
    return float(np.mean(aucs)), float(np.std(aucs)), int(np.median(its)), float(np.mean(tr_aucs))


tuning = []
for nl, md, l2 in itertools.product(*GRID.values()):
    p = {**C.LGB_PARAMS_V2, "num_leaves": nl, "min_data_in_leaf": md, "lambda_l2": l2, "feature_fraction": 0.8}
    auc, sd, it, tra = cv_tune(p)
    tuning.append({"num_leaves": nl, "min_data_in_leaf": md, "lambda_l2": l2, "cv_auc": round(auc, 4),
                   "cv_auc_sd": round(sd, 4), "rounds": it, "train_auc": round(tra, 4)})
    print(f"    leaves {nl:3d} min_leaf {md:5d} l2 {l2:4.1f}: CV AUC {auc:.4f} +- {sd:.4f}, rounds {it}, train {tra:.4f}")
best = max(tuning, key=lambda r: (r["cv_auc"], -r["num_leaves"]))
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": best["num_leaves"], "min_data_in_leaf": best["min_data_in_leaf"],
         "lambda_l2": best["lambda_l2"], "feature_fraction": 0.8}
ROUNDS = max(best["rounds"], 50)


def gap(params, rounds):
    """Train AUC vs 10 km test AUC on the training events with fixed rounds."""
    tra, tea = [], []
    for k in range(C.N_FOLDS):
        a, b = C.std_rows(tr_ev[tr_ev.fold != k]), C.std_rows(tr_ev[tr_ev.fold == k])
        if not len(b):
            continue
        m = C.lgb_fit(a, FT, C.y_of(a), params, rounds)
        tra.append(roc_auc_score(C.y_of(a), m.predict(a[FT]))); tea.append(roc_auc_score(C.y_of(b), m.predict(b[FT])))
    return {"train_auc": round(float(np.mean(tra)), 4), "test_auc": round(float(np.mean(tea)), 4),
            "gap": round(float(np.mean(tra) - np.mean(tea)), 4)}


g_p1 = gap({**C.LGB_PARAMS_V2}, 648)
g_tu = gap(TUNED, ROUNDS)
res["tuning"] = {"grid": GRID, "results": tuning, "best": best, "rounds": ROUNDS,
                 "gap_phase1_settings_648_rounds": g_p1, "gap_tuned": g_tu}
print(f"    best: {best}; gap Phase 1 settings {g_p1}, tuned {g_tu}")

# ---------------------------------------------------------------------------
# 8.3 Ablation (+ 8.2 XGBoost)
# ---------------------------------------------------------------------------
print("\n[2/5] 8.3 Ablation A-D (standard and PU bagging) + 8.2 XGBoost")
VARIANTS = {**C.FEATURE_SETS, "D-noAnnualRain": [f for f in C.FEATURES_D if f != "rainfall_mean_annual_mm"]}


def random_split_auc(feats):
    s = C.std_rows(df)
    a, b = train_test_split(s, test_size=0.3, stratify=s.set, random_state=C.SEED)
    return round(float(roc_auc_score(C.y_of(b), C.lgb_fit(a, feats, C.y_of(a), TUNED, ROUNDS).predict(b[feats]))), 4)


def xgb_fp(feats):
    import xgboost as xgb

    def fp(tr, te):
        d = C.std_rows(tr)
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", random_state=C.SEED, n_jobs=8)
        m.fit(d[feats].to_numpy(float), C.y_of(d))
        return m.predict_proba(te[feats].to_numpy(float))[:, 1]
    return fp


abl = {}
for design in ("standard", "pu_bagging"):
    for v, feats in VARIANTS.items():
        t1 = time.time()
        r = C.run_protocols(df, C.make_fit_predict(design, feats, TUNED, ROUNDS))
        r["headline"] = C.headline(r)
        if design == "standard":
            r["headline"]["random_split_auc"] = random_split_auc(feats)
        abl[f"{design}|{v}"] = r
        h = r["headline"]
        print(f"    {design:10s} {v:15s} spatial {h['spatial_auc']:.4f} | temporal land {h['temporal_2020_auc_land']:.4f} "
              f"{h['temporal_2024_auc_land']:.4f} | strict land {h['strict_2020_auc_land']:.4f} "
              f"{h['strict_2024_auc_land']:.4f}  ({time.time() - t1:.0f} s)")
r = C.run_protocols(df, xgb_fp(C.FEATURES_D))
r["headline"] = C.headline(r)
abl["xgboost|D"] = r
h = r["headline"]
print(f"    xgboost    D               spatial {h['spatial_auc']:.4f} | temporal land {h['temporal_2020_auc_land']:.4f} "
      f"{h['temporal_2024_auc_land']:.4f} | strict land {h['strict_2020_auc_land']:.4f} {h['strict_2024_auc_land']:.4f}")
res["ablation"] = {k: v for k, v in abl.items()}
res["ablation_table"] = {k: v["headline"] for k, v in abl.items()}


def strict_mean(k):
    h = abl[k]["headline"]
    return (h["strict_2020_auc_land"] + h["strict_2024_auc_land"]) / 2


res["ablation_summary"] = {k: {"strict_auc_land_mean": round(strict_mean(k), 4),
                               "spatial_auc": abl[k]["headline"]["spatial_auc"]} for k in abl}
for design in ("standard", "pu_bagging"):
    dd = strict_mean(f"{design}|D") - strict_mean(f"{design}|C")
    res["ablation_summary"][f"{design}: D minus C (strict)"] = round(dd, 4)
    print(f"    {design}: D - C strict AUC (vs land) = {dd:+.4f}")

# ---------------------------------------------------------------------------
# Chosen model
# ---------------------------------------------------------------------------
cands = [k for k in abl if not k.startswith("xgboost")]
chosen = max(cands, key=strict_mean)
c_design, c_var = chosen.split("|")
c_feats = VARIANTS[c_var]
res["chosen"] = {"key": chosen, "by": "mean strict AUC (flood vs all land) on 2020 and 2024",
                 "strict_auc_land_mean": round(strict_mean(chosen), 4)}
print(f"\n    chosen model: {chosen} (strict AUC vs land {strict_mean(chosen):.4f})")
for v in ("A", "D", c_var):
    d = C.std_rows(tr_ev)
    C.lgb_fit(d, VARIANTS[v], C.y_of(d), TUNED, ROUNDS).save_model(f"{C.MODEL_DIR}/model_v2_{v}.txt")

# ---------------------------------------------------------------------------
# 8.4 Calibration
# ---------------------------------------------------------------------------
print("\n[3/5] 8.4 Calibration")
fp = C.make_fit_predict(c_design, c_feats, TUNED, ROUNDS)
rnd_tr = tr_ev[tr_ev.set == "rnd"]
oof = pd.Series(np.nan, index=rnd_tr.index)
for k in range(C.N_FOLDS):
    te = rnd_tr[rnd_tr.fold == k]
    if len(te):
        oof.loc[te.index] = fp(tr_ev[tr_ev.fold != k], te)
te_all = df[df.event.isin(C.TEST_EVENTS) & (df.set == "rnd")]
s_te = fp(tr_ev, te_all)
y_tr, y_te = rnd_tr.is_pos.to_numpy().astype(int), te_all.is_pos.to_numpy().astype(int)
pi = float(y_tr.mean())


def prior_correct(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p * pi / (p * pi + (1 - p) * (1 - pi))          # trained at 50 % prevalence


logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(C=1e6).fit(logit(oof.to_numpy())[:, None], y_tr)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.to_numpy(), y_tr)
cal = {"raw (balanced score)": s_te, "prior-corrected": prior_correct(s_te),
       "Platt": platt.predict_proba(logit(s_te)[:, None])[:, 1], "isotonic": iso.predict(s_te)}


def ece(p, y, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(nb) if (b == i).any()))


calib = {"training_prevalence_pi": round(pi, 5), "test_prevalence": round(float(y_te.mean()), 5),
         "n_train_rnd": int(len(y_tr)), "n_test_rnd": int(len(y_te)), "methods": {}}
for k, p in cal.items():
    calib["methods"][k] = {"brier": round(float(brier_score_loss(y_te, p)), 5), "ece": round(ece(p, y_te), 5),
                           "mean_pred": round(float(np.mean(p)), 5)}
    print(f"    {k:22s} Brier {calib['methods'][k]['brier']:.5f}  ECE {calib['methods'][k]['ece']:.5f}  "
          f"mean p {calib['methods'][k]['mean_pred']:.4f} (observed {y_te.mean():.4f})")
for ev in C.TEST_EVENTS:
    m = (te_all.event == ev).to_numpy()
    calib[f"isotonic_{C.EVENT_LABEL[ev]}"] = {"mean_pred": round(float(cal["isotonic"][m].mean()), 5),
                                               "observed": round(float(y_te[m].mean()), 5)}
res["calibration"] = calib

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[4/5] Figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
order = ["A", "B", "C", "D", "D-noAnnualRain"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
for ax, (title, key) in zip(axes, [("10 km spatial CV (4 events)", "spatial_auc"),
                                   ("Strict: new year + new place, 2020", "strict_2020_auc_land"),
                                   ("Strict: new year + new place, 2024", "strict_2024_auc_land")]):
    x = np.arange(len(order))
    for j, (design, col) in enumerate((("standard", C.C_BLUE), ("pu_bagging", C.C_ORANGE))):
        v = [abl[f"{design}|{o}"]["headline"][key] for o in order]
        ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=col, label="Standard" if j == 0 else "PU bagging")
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 0.5) * 0.38, vi + 0.002, f"{vi:.3f}", ha="center", fontsize=7, rotation=90)
    xv = abl["xgboost|D"]["headline"][key]
    ax.axhline(xv, color=C.C_GREY, ls="--", lw=1)
    ax.text(-0.45, xv + 0.0008, f"XGBoost D {xv:.3f}", ha="left", fontsize=7, color=C.C_GREY)
    ax.set_xticks(x); ax.set_xticklabels(["A\nPhase 1", "B\n+hydrology", "C\n+land cover", "D\n+UICA", "D without\nannual rain"])
    lo = min(abl[f"{d}|{o}"]["headline"][key] for d in ("standard", "pu_bagging") for o in order)
    ax.set_ylim(lo - 0.02, 1.0); ax.set_title(title); ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("AUC (strict panels: flood vs all other land)")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Step 8  Development ablation, tuned LightGBM, same fixed folds", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step8_ablation.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
tb = pd.DataFrame(tuning)
for nl, col in zip(GRID["num_leaves"], (C.C_BLUE, C.C_ORANGE, C.C_GREEN)):
    s = tb[tb.num_leaves == nl]
    axes[0].scatter(s.train_auc - s.cv_auc, s.cv_auc, color=col, label=f"{nl} leaves", s=28)
axes[0].set_xlabel("train AUC - 10 km CV AUC (overfit gap)"); axes[0].set_ylabel("10 km CV AUC (2018 + 2019)")
axes[0].legend(fontsize=8); axes[0].grid(color=C.C_GRID); axes[0].set_title("8.1 Tuning grid (18 settings)")
lab = ["Phase 1 settings\n(648 rounds)", f"Tuned\n({ROUNDS} rounds)"]
w = 0.35
axes[1].bar(np.arange(2) - w / 2, [g_p1["train_auc"], g_tu["train_auc"]], w, color=C.C_GREY, label="train")
axes[1].bar(np.arange(2) + w / 2, [g_p1["test_auc"], g_tu["test_auc"]], w, color=C.C_BLUE, label="10 km test")
for i, g in enumerate((g_p1, g_tu)):
    axes[1].text(i, max(g["train_auc"], g["test_auc"]) + 0.004, f"gap {g['gap']:.3f}", ha="center")
axes[1].set_xticks(range(2)); axes[1].set_xticklabels(lab); axes[1].set_ylim(0.9, 1.01); axes[1].legend(fontsize=8)
axes[1].set_title("Train-test gap, feature set D"); axes[1].grid(axis="y", color=C.C_GRID); axes[1].set_axisbelow(True)
fig.savefig(f"{C.MAP_DIR}/step8_tuning_gap.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for (k, p), col in zip(cal.items(), (C.C_GREY, C.C_YELLOW, C.C_ORANGE, C.C_BLUE)):
    q = np.unique(np.quantile(p, np.linspace(0, 1, 16)))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    mp = [p[b == i].mean() for i in range(len(q) - 1) if (b == i).sum() > 30]
    mo = [y_te[b == i].mean() for i in range(len(q) - 1) if (b == i).sum() > 30]
    axes[0].plot(mp, np.maximum(mo, 1e-5), "o-", color=col, ms=3, label=f"{k} (ECE {calib['methods'][k]['ece']:.4f})")
axes[0].plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8)
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlim(1e-5, 1); axes[0].set_ylim(1e-5, 1)
axes[0].set_xlabel("predicted probability"); axes[0].set_ylabel("observed flood share")
axes[0].legend(fontsize=7); axes[0].grid(color=C.C_GRID); axes[0].set_title("Reliability on 2020 + 2024 random land cells")
best_cal = min(cal, key=lambda k: calib["methods"][k]["ece"])
calib["best_method"] = best_cal
axes[1].hist(cal[best_cal], bins=np.logspace(-5, 0, 40), color=C.C_BLUE)
axes[1].set_xscale("log"); axes[1].set_title(f"Calibrated probabilities ({best_cal}), test cells")
axes[1].set_xlabel("P(confirmed flood in an event like these)"); axes[1].set_ylabel("cells")
fig.suptitle(f"Step 8  Calibration of the chosen model ({chosen})", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step8_calibration.png", dpi=150)
plt.close(fig)

# A vs chosen model, 2024 map
lm, rr, cc, land = C.land_matrix("2024-08_ndem")
pa = C.percentile_map(C.make_fit_predict("standard", C.FEATURES_A, TUNED, ROUNDS)(tr_ev, lm), rr, cc, land)
pd_ = C.percentile_map(fp(tr_ev, lm), rr, cc, land)
import rasterio
with rasterio.open(f"{C.PROC_DIR}/training_v2/label_pools_2024-08_ndem.tif") as src:
    pools24 = src.read(1)
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
cmap = plt.get_cmap("RdYlBu_r", 5)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for ax, m, t in ((axes[0], pa, "A: Phase 1 features (tuned)"), (axes[1], pd_, f"Chosen: {c_var} features, {c_design.replace('_', ' ')}")):
    im = ax.imshow(np.floor(m / 20).clip(0, 4), cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
    fy, fx = np.nonzero(pools24 == 1)
    ax.set_title(t)
dd = (np.floor(pd_ / 20) - np.floor(pa / 20))
im2 = axes[2].imshow(dd, cmap="PuOr_r", vmin=-3, vmax=3, extent=ext, interpolation="nearest")
axes[2].set_title("Class change (chosen - A)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes[:2], shrink=0.6, ticks=range(5))
cb.ax.set_yticklabels(["Very Low", "Low", "Moderate", "High", "Very High"])
fig.colorbar(im2, ax=axes[2], shrink=0.6, label="classes")
fig.suptitle("Step 8  2024 susceptibility, models trained on 2018 + 2019", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step8_maps_A_vs_chosen.png", dpi=130)
plt.close(fig)
built = lm["landcover_class"].to_numpy() == 6
res["map_2024"] = {"built_in_high_vhigh_A": round(float((pa[rr, cc][built] >= 60).mean()), 4),
                   "built_in_high_vhigh_chosen": round(float((pd_[rr, cc][built] >= 60).mean()), 4),
                   "share_of_land_changing_class": round(float((dd[rr, cc] != 0).mean()), 4)}
print(f"    2024 map: {res['map_2024']}")

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step8_model_v2_ablation.json")
print(f"\n[5/5] Done in {res['runtime_s']} s")
