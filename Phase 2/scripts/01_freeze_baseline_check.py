"""
Phase 2 Step 1 - Freeze the Phase 1 baseline and verify the shared module.

[CHECKPOINT] Every reusable piece of Phase 1 now lives in common.py. This
script recomputes Phase 1 results through common.py only and compares them to
the numbers / rasters Phase 1 saved. If refactoring changed a number,
something broke.

    A. Step 7 routing   Priority-Flood + FD8 on FABDEM -> flow accumulation
                        and TWI vs Phase 1 rasters (bit-level tolerance)
    B. Step 8 labels    exact 20 m -> 30 m aggregation of the 2018 Otsu flood
                        vs the otsu_flood_frac column of the training CSV
    C. Step 6 GTI       2018 flood vs NDEM event envelope -> GTI 40 m = 58.44 %
    D. Step 9 model     random 70/30 split, CV-chosen rounds, LightGBM
                        -> test AUC 0.9993 (all test metrics, and every test
                        prediction vs step9_test_predictions.csv), plus the
                        10 km spatial-block number (AUC 0.979, OA 88.5 %)

Output: outputs/metrics/step1_baseline_reproduction.json
"""
import json
import time

import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

import common as C

C.banner("PHASE 2 STEP 1: BASELINE REPRODUCTION THROUGH common.py")
t0 = time.time()
step9 = json.load(open(C.P1_STEP9_JSON, encoding="utf-8"))
out = {"step": 1, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
ok_all = True


def record(name, passed, **info):
    global ok_all
    ok_all &= bool(passed)
    out["checks"][name] = {"pass": bool(passed), **info}
    print(f"    [{'PASS' if passed else 'FAIL'}] {name}: {info}")


# ---------------------------------------------------------------------------
# A. Step 7 routing
# ---------------------------------------------------------------------------
print("\n[A] Step 7 flow routing and TWI")
with rasterio.open(C.P1_FABDEM) as src:
    dem = src.read(1)
    T30 = src.transform
H30, W30 = dem.shape
dx, dy, cell_area = C.row_cell_sizes_m(T30, H30)
water, land = C.open_water_and_land(dem)
with rasterio.open(C.P1_DOMAIN) as src:
    land_p1 = src.read(1).astype(bool)
record("A1_land_domain_identical", np.array_equal(land, land_p1), land_cells=int(land.sum()))

zf, acc, flow = C.route_flow(dem, land, dx, dy, cell_area)
with rasterio.open(C.P1_FLOWACC) as src:
    acc_p1 = src.read(1).astype(np.float64)
acc_km2 = (acc / 1e6).astype(np.float32).astype(np.float64)   # Phase 1 stored float32
rel = np.abs(acc_km2[land] - acc_p1[land]) / np.maximum(acc_p1[land], 1e-12)
record("A2_flow_accumulation_matches", float(rel.max()) < 1e-6, max_rel_diff=float(rel.max()),
       interior_sinks=flow["n_interior_sinks"])

p, q = C.horn_gradient(dem, dx, dy)
twi = C.twi_from_acc(acc, p, q, dx, dy).astype(np.float32)
with rasterio.open(C.p1("data", "processed", "features", "twi.tif")) as src:
    twi_p1 = src.read(1)
d = np.abs(twi[land] - twi_p1[land])
record("A3_twi_matches", float(d.max()) < 1e-5, max_abs_diff=float(d.max()))

# ---------------------------------------------------------------------------
# B. Step 8 label aggregation
# ---------------------------------------------------------------------------
print("\n[B] Step 8 exact 20 m -> 30 m label aggregation")
with rasterio.open(f"{C.P1_INV_DIR}/water_classes_2018_20180821.tif") as src:
    cls = src.read(1)
    T20 = src.transform
C.check_20_30_commensurate(T20, T30)
flood20 = cls == 1
f_flood = C.to_30m_fraction(flood20, H30, W30)
area_err = abs(f_flood.sum() * 9 - flood20.sum() * 4)
df = pd.read_csv(C.P1_TRAINING_2018)
d = np.abs(f_flood[df["row"], df["col"]] - df["otsu_flood_frac"].to_numpy())
record("B1_area_conserved", area_err < 1e-2, abs_err_subpixels=float(area_err))
record("B2_sample_fractions_match_csv", float(d.max()) < 1e-4, n_samples=len(df), max_abs_diff=float(d.max()))

# ---------------------------------------------------------------------------
# C. Step 6 GTI
# ---------------------------------------------------------------------------
print("\n[C] Step 6 GTI vs NDEM 2018 event envelope")
ndem = C.load_ndem()
env, n_env = C.ndem_mask(ndem, ["17-08-2018", "18-08-2018", "21-08-2018", "24-08-2018", "27-08-2018",
                                "28-08-2018"], cls.shape, T20)
row_area = C.pixel_area_km2(T20, cls.shape[0])
g = C.agreement(cls == 1, env, (cls != 255) & (cls != 2), row_area)
record("C1_gti_40m_2018", abs(g["GTI_px_40m"] - 0.5844) < 1e-4, GTI_px_40m=g["GTI_px_40m"],
       lift=g["lift_px_40m"], n_polygons=n_env)

# ---------------------------------------------------------------------------
# D. Step 9 model
# ---------------------------------------------------------------------------
print("\n[D] Step 9 LightGBM baseline")
FEATURES = json.load(open(C.P1_STEP8_JSON, encoding="utf-8"))["dataset"]["feature_columns"]
assert not set(FEATURES) & C.META_COLUMNS
X, y = df[FEATURES], df["label"].to_numpy()
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=C.SEED)
X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
cv = lgb.cv(C.LGB_PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=3000, nfold=5, stratified=True,
            seed=C.SEED, callbacks=[lgb.early_stopping(100, verbose=False)])
rounds = len(cv["valid auc-mean"])
record("D1_boosting_rounds", rounds == step9["model"]["num_boost_round"], rounds=rounds,
       phase1=step9["model"]["num_boost_round"])

booster = lgb.train(C.LGB_PARAMS, lgb.Dataset(X_tr, y_tr, free_raw_data=False), num_boost_round=rounds)
p_te = booster.predict(X_te)
m_te = C.rounded(C.metrics(y_te, p_te))
ref = step9["headline_test_metrics"]
same = all(abs(m_te[k] - ref[k]) < 1e-4 for k in ("accuracy", "precision", "recall", "f1", "auc", "specificity"))
record("D2_test_metrics_identical", same and m_te["confusion"] == ref["confusion"],
       auc=m_te["auc"], accuracy=m_te["accuracy"], f1=m_te["f1"], confusion=m_te["confusion"])

pred_p1 = pd.read_csv(C.p1("data", "processed", "training", "step9_test_predictions.csv"))
same_ids = np.array_equal(pred_p1["sample_id"].to_numpy(), df["sample_id"].iloc[idx_te].to_numpy())
dp = np.abs(pred_p1["p_flood"].to_numpy() - np.round(p_te, 6))
record("D3_every_test_prediction_identical", same_ids and float(dp.max()) < 2e-6,
       same_test_ids=bool(same_ids), max_abs_diff=float(dp.max()))

gk = StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED)
groups = C.block_groups(df, 10000)
folds = []
for tr, te in gk.split(X, y, groups):
    b = lgb.train(C.LGB_PARAMS, lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=False), num_boost_round=rounds)
    folds.append(C.metrics(y[te], b.predict(X.iloc[te])))
sp = C.cv_summary(folds)
sp_ref = step9["supplementary_spatial_block_cv"]["by_block_size"]["10km"]["per_fold_mean_sd"]
record("D4_spatial_10km_identical",
       abs(sp["auc"]["mean"] - sp_ref["auc"]["mean"]) < 1e-4
       and abs(sp["accuracy"]["mean"] - sp_ref["accuracy"]["mean"]) < 1e-4,
       auc_mean=sp["auc"]["mean"], accuracy_mean=sp["accuracy"]["mean"])

out["all_pass"] = bool(ok_all)
out["runtime_s"] = round(time.time() - t0, 1)
out["baseline_reference"] = {"git_tag": "phase1-final", "test_auc": m_te["auc"],
                             "spatial_10km_auc": sp["auc"]["mean"], "spatial_10km_accuracy": sp["accuracy"]["mean"],
                             "gti_40m_2018": g["GTI_px_40m"]}
C.save_json(out, f"{C.METRICS_DIR}/step1_baseline_reproduction.json")
print(f"\n{'CHECKPOINT PASS' if ok_all else 'CHECKPOINT FAIL'}: "
      f"{sum(v['pass'] for v in out['checks'].values())}/{len(out['checks'])} checks, {out['runtime_s']} s")
