"""
Step 9 - Baseline LightGBM flood-susceptibility model
(Week 3, Phase1_Implementation_Plan).

9.1 Split: simple random 70/30, stratified by class, seed 42 -- deliberately
    the "naive" evaluation used in the reference papers, so the numbers are
    directly comparable to them.
9.2 Model: LightGBM binary classifier on the 11 Step 7/8 features. Fixed,
    conventional hyper-parameters (no tuning on the test set). The number of
    boosting rounds is chosen by 5-fold stratified CV *inside the 70 % training
    part only* (early stopping on AUC), then the model is refit on all of it.
9.3 Metrics on the 30 % test part: Accuracy, Precision, Recall, F1, AUC-ROC
    (the base paper's set), plus specificity, the confusion matrix and the
    train-test gap. Stability: the same model under random 5-fold CV.
9.4 Importance: LightGBM gain (built-in) and permutation importance on the
    test set (drop in AUC when a feature is shuffled).
    Base-paper comparison. The base paper's Table 4 is not an importance
    ranking; it is an input ablation (the U-Net retrained on 8 combinations of
    slope, HSG, imperviousness and rainfall). The same kind of ablation is run
    here on our feature groups, so the two can be compared like for like.

Checkpoint diagnostics (the plan warns that ~0.999 AUC usually means
leakage, not a good model):
    - leakage by construction: no feature is derived from Sentinel-1, no
      metadata column (coordinates, ids, label fractions) is a feature;
    - spatial proximity of the random split: share of test samples with a
      training sample in the adjacent 30 m cell (Step 8: flood samples are
      clustered), and
    - supplementary spatial check: StratifiedGroupKFold over square UTM
      blocks of 2, 5 and 10 km, i.e. the model is tested on whole areas it
      never saw; 10 km is quoted. This is *not* the headline (the plan keeps
      the random split for comparability; spatial CV is the Phase 2 method)
      -- it shows how much of the random-split score is spatial
      autocorrelation; and
    - the hard question: the same out-of-block predictions scored only on
      low, flat floodplain samples, where elevation alone is weak.

Outputs
    outputs/models/step9_lightgbm_baseline.txt          LightGBM booster
    outputs/metrics/step9_baseline_metrics.json
    outputs/maps/step9_feature_importance.png
    outputs/maps/step9_model_evaluation.png
    data/processed/training/step9_test_predictions.csv
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

DATA = "data/processed/training/training_samples_2018.csv"
STEP8_JSON = "outputs/metrics/step8_training_dataset_summary.json"
MODEL_DIR = "outputs/models"
MAP_DIR = "outputs/maps"
METRICS_DIR = "outputs/metrics"
PRED_DIR = "data/processed/training"
for d in (MODEL_DIR, MAP_DIR, METRICS_DIR, PRED_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 42
TEST_SIZE = 0.30
THRESHOLD = 0.5
N_FOLDS = 5
PERM_REPEATS = 5
ADJACENT_M = 45.0          # a sample in the neighbouring 30 m cell (incl. diagonal)
BLOCK_SIZES_M = [2000, 5000, 10000]   # spatial hold-out block sizes (supplementary)
SPATIAL_REPORT_M = 10000   # block size quoted as the spatial number
LEAKAGE_AUC = 0.999
PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 20, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "force_row_wise": True,
    "num_threads": 4, "verbose": -1,
}
MAX_ROUNDS = 3000
EARLY_STOP = 100

# Base paper, Mohamadiazar et al. (2024) J. Hydrol. 639:131508 -- U-Net, Miami-Dade
BASE_PAPER = {
    "table3_testing": {"precision": 94.07, "recall": 86.00, "f1": 89.85, "accuracy": 93.73},
    "auc": 0.93,
    "table4_all_inputs": "slope + HSG + imperviousness + rainfall: P 94.07, R 86.00, F1 89.85, OA 93.73",
    "table4_rainfall_only": "rainfall: P 90.03, R 83.27, F1 86.52, OA 91.57",
    "table4_ranges": {"precision": [90.03, 94.07], "recall": [83.27, 86.00], "f1": [86.52, 89.85],
                      "accuracy": [91.57, 93.73]},
    "ground_truth_index": 84.05,
}

C_MAIN = "#2a78d6"
C_SPATIAL = "#eb6834"
C_BASE = "#52514e"
C_GRID = "#e5e5e5"

print("=" * 80)
print("STEP 9: BASELINE LIGHTGBM MODEL (random 70/30 split)")
print("=" * 80)
t_start = time.time()

df = pd.read_csv(DATA)
FEATURES = json.load(open(STEP8_JSON, encoding="utf-8"))["dataset"]["feature_columns"]
META = [c for c in df.columns if c not in FEATURES]
assert "label" in META and not set(FEATURES) & {"lon", "lat", "x_utm", "y_utm", "row", "col", "block_id",
                                                "otsu_flood_frac", "confirmed_flood_frac", "sample_id"}
X = df[FEATURES]
y = df["label"].to_numpy()
print(f"\n    {len(df)} samples, {len(FEATURES)} features, class counts {np.bincount(y).tolist()}")


def metrics(y_true, p):
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "auc": roc_auc_score(y_true, p), "specificity": tn / (tn + fp),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def rounded(m, nd=4):
    return {k: (round(float(v), nd) if not isinstance(v, dict) else v) for k, v in m.items()}


def fit(Xa, ya, rounds, feats=None):
    feats = feats or FEATURES
    return lgb.train(PARAMS, lgb.Dataset(Xa[feats], ya, free_raw_data=False), num_boost_round=rounds)


def cv_summary(fold_metrics):
    keys = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    return {k: {"mean": round(float(np.mean([m[k] for m in fold_metrics])), 4),
                "sd": round(float(np.std([m[k] for m in fold_metrics])), 4)} for k in keys}


# ---------------------------------------------------------------------------
# 9.1 Random stratified 70/30 split
# ---------------------------------------------------------------------------
print("\n[1/6] 9.1 Random stratified 70/30 split")
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
print(f"    train {len(idx_tr)} ({np.bincount(y_tr).tolist()}), test {len(idx_te)} ({np.bincount(y_te).tolist()})")

# spatial proximity of the random split
xy = df[["x_utm", "y_utm"]].to_numpy()
d_nn, _ = cKDTree(xy[idx_tr]).query(xy[idx_te])
adj = d_nn <= ADJACENT_M
proximity = {
    "test_with_train_sample_in_adjacent_cell_share": round(float(adj.mean()), 4),
    "flood_test_share": round(float(adj[y_te == 1].mean()), 4),
    "nonflood_test_share": round(float(adj[y_te == 0].mean()), 4),
    "median_distance_test_to_nearest_train_m": {"flood": round(float(np.median(d_nn[y_te == 1])), 1),
                                                "nonflood": round(float(np.median(d_nn[y_te == 0])), 1)},
}
print(f"    test samples with a training sample in the adjacent 30 m cell: flood "
      f"{100 * proximity['flood_test_share']:.1f}%, non-flood {100 * proximity['nonflood_test_share']:.1f}%")

# ---------------------------------------------------------------------------
# 9.2 Train (rounds chosen by CV inside the training part only)
# ---------------------------------------------------------------------------
print("\n[2/6] 9.2 LightGBM training")
cv = lgb.cv(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=MAX_ROUNDS, nfold=N_FOLDS, stratified=True,
            seed=SEED, callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
best_rounds = len(cv["valid auc-mean"])
print(f"    5-fold CV on the training part: best {best_rounds} rounds, "
      f"CV AUC {cv['valid auc-mean'][-1]:.4f} ± {cv['valid auc-stdv'][-1]:.4f}")
booster = fit(X_tr, y_tr, best_rounds)
model_path = f"{MODEL_DIR}/step9_lightgbm_baseline.txt"
booster.save_model(model_path)
print(f"    [OK] {model_path}")

# ---------------------------------------------------------------------------
# 9.3 Evaluation
# ---------------------------------------------------------------------------
print("\n[3/6] 9.3 Evaluation")
p_tr = booster.predict(X_tr)
p_te = booster.predict(X_te)
m_tr = metrics(y_tr, p_tr)
m_te = metrics(y_te, p_te)
print(f"    {'':<10}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}{'Spec':>8}")
for name, m in (("train", m_tr), ("test", m_te)):
    print(f"    {name:<10}" + "".join(f"{100 * m[k]:>8.2f}" for k in ("accuracy", "precision", "recall", "f1"))
          + f"{m['auc']:>8.4f}{100 * m['specificity']:>8.2f}")
print(f"    test confusion: {m_te['confusion']}")
pd.DataFrame({"sample_id": df["sample_id"].iloc[idx_te].to_numpy(), "label": y_te,
              "p_flood": np.round(p_te, 6), "pred": (p_te >= THRESHOLD).astype(int)}).to_csv(
    f"{PRED_DIR}/step9_test_predictions.csv", index=False)

# stability: random 5-fold CV of the same model on all samples
rk = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
rand_folds = [metrics(y[te], fit(X.iloc[tr], y[tr], best_rounds).predict(X.iloc[te])) for tr, te in rk.split(X, y)]
rand_cv = cv_summary(rand_folds)
print(f"    random 5-fold CV (all data): AUC {rand_cv['auc']['mean']:.4f} ± {rand_cv['auc']['sd']:.4f}, "
      f"F1 {100 * rand_cv['f1']['mean']:.2f} ± {100 * rand_cv['f1']['sd']:.2f}")

# supplementary spatial check: whole blocks held out (2, 5, 10 km), and the
# hard floodplain-only subset. The largest block size is the reported number.
gk = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)


def block_groups(block_m):
    return ((df["x_utm"] // block_m).astype(int).astype(str) + "_"
            + (df["y_utm"] // block_m).astype(int).astype(str)).to_numpy()


def spatial_cv(block_m, feats=None):
    feats = feats or FEATURES
    groups = block_groups(block_m)
    oof = np.full(len(df), np.nan)
    folds = []
    for tr, te in gk.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        p = fit(X.iloc[tr], y[tr], best_rounds, feats).predict(X.iloc[te][feats])
        oof[te] = p
        folds.append(metrics(y[te], p))
    return folds, oof, len(set(groups))


print("    supplementary spatial check (whole blocks held out, StratifiedGroupKFold):")
spatial = {}
for bm in BLOCK_SIZES_M:
    folds, oof_b, n_groups = spatial_cv(bm)
    spatial[f"{bm // 1000}km"] = {"n_blocks": n_groups, "per_fold_mean_sd": cv_summary(folds),
                                  "pooled_out_of_fold": rounded(metrics(y, oof_b))}
    sc = spatial[f"{bm // 1000}km"]["per_fold_mean_sd"]
    print(f"      {bm // 1000:>2} km blocks ({n_groups:>3}): AUC {sc['auc']['mean']:.4f} ± {sc['auc']['sd']:.4f}, "
          f"accuracy {100 * sc['accuracy']['mean']:.2f} ± {100 * sc['accuracy']['sd']:.2f}, "
          f"F1 {100 * sc['f1']['mean']:.2f} ± {100 * sc['f1']['sd']:.2f}")
    if bm == SPATIAL_REPORT_M:
        oof, sp_cv = oof_b, sc
sp_pooled = metrics(y, oof)

# the hard question: only low, flat floodplain samples (elevation <= flood p95, slope <= 2 deg)
e95 = float(np.percentile(df.loc[df["label"] == 1, "elevation_m"], 95))
hard = ((df["elevation_m"] <= e95) & (df["slope_deg"] <= 2)).to_numpy()
floodplain = {"rule": f"elevation <= {e95:.2f} m (flood p95) and slope <= 2 deg",
              "n": int(hard.sum()), "flood_share": round(float(y[hard].mean()), 4),
              f"spatial_{SPATIAL_REPORT_M // 1000}km_out_of_fold": rounded(metrics(y[hard], oof[hard])),
              "elevation_only_auc": round(float(roc_auc_score(y[hard], -df["elevation_m"].to_numpy()[hard])), 4)}
fp = floodplain[f"spatial_{SPATIAL_REPORT_M // 1000}km_out_of_fold"]
print(f"    floodplain-only subset ({floodplain['n']} samples, {floodplain['rule']}): "
      f"AUC {fp['auc']:.4f}, accuracy {100 * fp['accuracy']:.2f} (elevation alone AUC {floodplain['elevation_only_auc']:.3f})")

flags = {
    "test_auc_suspiciously_perfect": bool(m_te["auc"] >= LEAKAGE_AUC),
    "train_test_auc_gap": round(float(m_tr["auc"] - m_te["auc"]), 4),
    "random_minus_spatial_auc": round(float(m_te["auc"] - sp_cv["auc"]["mean"]), 4),
}
print(f"    flags: test AUC >= {LEAKAGE_AUC}: {flags['test_auc_suspiciously_perfect']}; "
      f"train-test AUC gap {flags['train_test_auc_gap']:.4f}; random - spatial AUC {flags['random_minus_spatial_auc']:.4f}")

# ---------------------------------------------------------------------------
# 9.4 Importance
# ---------------------------------------------------------------------------
print("\n[4/6] 9.4 Feature importance")
gain = booster.feature_importance("gain")
split = booster.feature_importance("split")
gain_pct = 100 * gain / gain.sum()
rng = np.random.default_rng(SEED)
base_auc = m_te["auc"]
perm = {}
for f in FEATURES:
    drops = []
    for _ in range(PERM_REPEATS):
        Xp = X_te.copy()
        Xp[f] = rng.permutation(Xp[f].to_numpy())
        drops.append(base_auc - roc_auc_score(y_te, booster.predict(Xp)))
    perm[f] = (float(np.mean(drops)), float(np.std(drops)))
imp = pd.DataFrame({"feature": FEATURES, "gain_pct": gain_pct, "split_count": split,
                    "perm_auc_drop": [perm[f][0] for f in FEATURES],
                    "perm_auc_drop_sd": [perm[f][1] for f in FEATURES]}).sort_values("gain_pct", ascending=False)
for r in imp.itertuples():
    print(f"    {r.feature:<26} gain {r.gain_pct:5.1f}%   permutation ΔAUC {r.perm_auc_drop:.4f} ± {r.perm_auc_drop_sd:.4f}")

# ---------------------------------------------------------------------------
# Table 4 analogue: input ablation, random split + spatial blocks
# ---------------------------------------------------------------------------
print("\n[5/6] Input ablation (analogue of base-paper Table 4)")
TERRAIN = ["elevation_m", "slope_deg", "aspect_deg", "curvature_plan", "curvature_profile"]
HYDRO = ["twi", "dist_to_river_m", "dist_to_major_river_m", "drainage_density_km_km2"]
ROAD = ["dist_to_road_m"]
RAIN = ["rainfall_mean_annual_mm"]
PLAN_MIN = ["elevation_m", "slope_deg", "twi", "dist_to_river_m", "dist_to_road_m", "rainfall_mean_annual_mm"]
ABLATION = [
    ("All 11 features", FEATURES),
    ("Plan minimum set (elev, slope, TWI, d-river, d-road, rain)", PLAN_MIN),
    ("Without terrain (elev, slope, aspect, curvatures)", [f for f in FEATURES if f not in TERRAIN]),
    ("Without hydrology (TWI, river distances, drainage dens.)", [f for f in FEATURES if f not in HYDRO]),
    ("Without distance to road", [f for f in FEATURES if f not in ROAD]),
    ("Without rainfall", [f for f in FEATURES if f not in RAIN]),
    ("Slope + rainfall (base-paper row analogue)", ["slope_deg", "rainfall_mean_annual_mm"]),
    ("Rainfall only (base-paper last row)", RAIN),
]
ablation = []
for name, feats in ABLATION:
    b = fit(X_tr, y_tr, best_rounds, feats)
    m = metrics(y_te, b.predict(X_te[feats]))
    sp_folds_ab, _, _ = spatial_cv(SPATIAL_REPORT_M, feats)
    sp = [m_["auc"] for m_ in sp_folds_ab]
    sp_acc = [m_["accuracy"] for m_ in sp_folds_ab]
    row = {"inputs": name, "features": feats, **{k: round(float(v), 4) for k, v in m.items() if k != "confusion"},
           f"spatial_{SPATIAL_REPORT_M // 1000}km_auc_mean": round(float(np.mean(sp)), 4),
           f"spatial_{SPATIAL_REPORT_M // 1000}km_auc_sd": round(float(np.std(sp)), 4),
           f"spatial_{SPATIAL_REPORT_M // 1000}km_accuracy_mean": round(float(np.mean(sp_acc)), 4)}
    ablation.append(row)
    print(f"    {name:<58} P {100 * row['precision']:5.2f} R {100 * row['recall']:5.2f} F1 {100 * row['f1']:5.2f} "
          f"OA {100 * row['accuracy']:5.2f} AUC {row['auc']:.3f} | {SPATIAL_REPORT_M // 1000} km blocks: AUC "
          f"{row[f'spatial_{SPATIAL_REPORT_M // 1000}km_auc_mean']:.3f} OA "
          f"{100 * row[f'spatial_{SPATIAL_REPORT_M // 1000}km_accuracy_mean']:5.2f}")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
results = {
    "step": 9, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "data": {"path": DATA, "n": int(len(df)), "features": FEATURES, "n_train": int(len(idx_tr)),
             "n_test": int(len(idx_te))},
    "split": {"method": "random stratified 70/30 (plan 9.1, naive on purpose)", "seed": SEED},
    "model": {"type": "LightGBM gbdt", "lightgbm_version": lgb.__version__, "params": PARAMS,
              "num_boost_round": best_rounds, "rounds_selection": "5-fold stratified CV on the training part, "
              f"early stopping {EARLY_STOP} on AUC", "threshold": THRESHOLD, "path": model_path,
              "cv_auc_on_train_part": round(float(cv["valid auc-mean"][-1]), 4)},
    "headline_test_metrics": rounded(m_te),
    "train_metrics": rounded(m_tr),
    "random_5fold_cv": rand_cv,
    "supplementary_spatial_block_cv": {"method": "StratifiedGroupKFold(5) over square UTM blocks; whole blocks held out",
                                       "reported_block_size_m": SPATIAL_REPORT_M, "by_block_size": spatial,
                                       "floodplain_only_subset": floodplain},
    "diagnostics": {"spatial_proximity_of_random_split": proximity, "flags": flags,
                    "leakage_by_construction": "no Sentinel-1-derived feature; coordinates/ids/label fractions excluded"},
    "importance": imp.round(5).to_dict(orient="records"),
    "ablation_table4_analogue": ablation,
    "base_paper": BASE_PAPER,
}
with open(f"{METRICS_DIR}/step9_baseline_metrics.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[6/6] Figures")
pretty = {"elevation_m": "Elevation", "slope_deg": "Slope", "aspect_deg": "Aspect", "curvature_plan": "Plan curvature",
          "curvature_profile": "Profile curvature", "twi": "TWI", "dist_to_river_m": "Distance to river",
          "dist_to_major_river_m": "Distance to major river", "dist_to_road_m": "Distance to road",
          "drainage_density_km_km2": "Drainage density", "rainfall_mean_annual_mm": "Rainfall (mean annual)"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=130, sharey=False)
for ax, col, err, title, xlabel, fmt in [
        (axes[0], "gain_pct", None, "LightGBM gain importance (built-in)", "Share of total gain, %", "{:.1f}%"),
        (axes[1], "perm_auc_drop", "perm_auc_drop_sd", "Permutation importance on the test set",
         "Drop in test AUC when the feature is shuffled", "{:.3f}")]:
    s = imp.sort_values(col)
    yy = np.arange(len(s))
    ax.barh(yy, s[col], height=0.6, color=C_MAIN, zorder=2,
            xerr=s[err] if err else None, error_kw={"ecolor": C_BASE, "lw": 1})
    for yi, v in zip(yy, s[col]):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8.5, color="#0b0b0b")
    ax.set_yticks(yy, [pretty[f] for f in s["feature"]], fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_xlim(0, s[col].max() * 1.22)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
fig.suptitle(f"Step 9  Feature importance — LightGBM baseline (test AUC {m_te['auc']:.3f}, random 70/30 split)",
             fontsize=13, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
p_imp = f"{MAP_DIR}/step9_feature_importance.png"
fig.savefig(p_imp, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_imp}")

fig = plt.figure(figsize=(18, 6.2), dpi=120)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1.35])
ax = fig.add_subplot(gs[0])
fpr, tpr, _ = roc_curve(y_te, p_te)
ax.plot(fpr, tpr, color=C_MAIN, lw=2, label=f"Random 70/30 test (AUC {m_te['auc']:.3f})")
fpr, tpr, _ = roc_curve(y, oof)
ax.plot(fpr, tpr, color=C_SPATIAL, lw=2, label=f"Unseen {SPATIAL_REPORT_M // 1000} km blocks, out-of-fold (AUC {sp_pooled['auc']:.3f})")
ax.plot([0, 1], [0, 1], color=C_BASE, lw=1, ls="--", label="Chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[1])
cm = np.array([[m_te["confusion"]["TN"], m_te["confusion"]["FP"]], [m_te["confusion"]["FN"], m_te["confusion"]["TP"]]])
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / cm[i].sum():.1f}% of row)", ha="center", va="center",
                fontsize=10, color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b")
ax.set_xticks([0, 1], ["Predicted non-flood", "Predicted flood"], fontsize=9)
ax.set_yticks([0, 1], ["Actual non-flood", "Actual flood"], fontsize=9)
ax.set_title(f"Confusion matrix (test, n = {len(y_te):,})", loc="left", fontweight="bold")
for s_ in ax.spines.values():
    s_.set_visible(False)

ax = fig.add_subplot(gs[2])
keys = ["accuracy", "precision", "recall", "f1"]
labels = ["Accuracy", "Precision", "Recall", "F1"]
series = [("Ours — random 70/30 test", [100 * m_te[k] for k in keys], C_MAIN),
          (f"Ours — unseen {SPATIAL_REPORT_M // 1000} km blocks (mean of 5 folds)", [100 * sp_cv[k]["mean"] for k in keys], C_SPATIAL),
          ("Base paper — U-Net test (Table 3)", [BASE_PAPER["table3_testing"][k] for k in keys], C_BASE)]
w = 0.26
xx = np.arange(len(keys))
for i, (nm, vals, col) in enumerate(series):
    bars = ax.bar(xx + (i - 1) * w, vals, width=w - 0.02, color=col, label=nm, zorder=2)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
lo = min(min(v) for _, v, _ in series)
ax.set_ylim(max(0, np.floor(lo / 10) * 10 - 5), 101)
ax.set_xticks(xx, labels)
ax.set_ylabel("%")
ax.set_title("Metrics vs base paper", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
fig.suptitle("Step 9  LightGBM baseline — evaluation", fontsize=13, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
p_eval = f"{MAP_DIR}/step9_model_evaluation.png"
fig.savefig(p_eval, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_eval}")

print("\n" + "=" * 80)
print("STEP 9 SUMMARY")
print("=" * 80)
print(f"  Test (random 70/30): Acc {100 * m_te['accuracy']:.2f}  P {100 * m_te['precision']:.2f}  "
      f"R {100 * m_te['recall']:.2f}  F1 {100 * m_te['f1']:.2f}  AUC {m_te['auc']:.4f}")
print(f"  Unseen {SPATIAL_REPORT_M // 1000} km blocks: AUC {sp_cv['auc']['mean']:.4f} ± {sp_cv['auc']['sd']:.4f}  "
      f"F1 {100 * sp_cv['f1']['mean']:.2f} ± {100 * sp_cv['f1']['sd']:.2f}")
print(f"  Base paper (U-Net, test): Acc 93.73  P 94.07  R 86.00  F1 89.85  AUC 0.93")
print(f"  Runtime {time.time() - t_start:.0f} s")
print("=" * 80)
