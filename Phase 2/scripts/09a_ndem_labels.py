"""
Phase 2 Step 9.0a - Radar + NDEM labels: give UICA a fair test on urban land.

Week 3 found radar flood labels are 0.1 % built-up, so UICA could not earn skill
(D = C) and the model learned "built = dry". NDEM (NRSC) maps also contain the
flooding our single radar pass missed, which is far more urban.

1. Label quality. NDEM polygons are either traced at pixel detail or coarse
   (generalised). Rule fixed before modelling: a polygon is DETAILED if it has
   >= 50 vertices per km of outline; only detailed polygons are used. Reported:
   per event, the share of detailed polygons, and the paved / UICA / HAND
   profile of NDEM flood vs radar flood vs random land.
2. Labels "radar + NDEM": positives = confirmed radar flood OR >= 50 % of the
   cell inside a detailed NDEM polygon (land, fully observed, away from
   reference water, Phase 1 rule). Negatives = the unchanged confident
   non-flood pool (it already excludes all NDEM + 40 m). Random land cells are
   the Step 6 'rnd' rows, re-flagged.
3. Models: feature sets C and D x {standard, PU bagging} x {radar labels
   (Week 3), radar + NDEM labels}. Same tuned LightGBM settings and fixed
   10 km folds. Tests:
     spatial  4 events pooled, fold k held out
     strict   train 2018 + 2019 outside fold k, test 2020 / 2024 inside fold k
   All are scored against radar + NDEM truth (flood vs all other land), on all
   land and on urban subsets: DW built, paved >= 20 %, UICA ratio >= 0.10.
4. 2024 map: built land in High + Very High, and Kochi core percentile.

Outputs: outputs/metrics/step9a_ndem_labels.json,
         outputs/maps/step9a_label_quality.png, step9a_urban_tests.png
"""
import json
import os
import time

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.metrics import roc_auc_score

import common as C

DETAIL_V_PER_KM = 50.0
N_BAGS = 15            # 25 in Week 3; 15 keeps memory / time in check with the larger positive set
MIN_POS_SUBSET = 30
C.banner("PHASE 2 STEP 9.0a: RADAR + NDEM LABELS, C vs D ON URBAN LAND")
t0 = time.time()
rng = np.random.default_rng(C.SEED)
res = {"step": "9.0a", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
       "detail_rule_vertices_per_km": DETAIL_V_PER_KM, "pu_bags": N_BAGS}
s8 = json.load(open(f"{C.METRICS_DIR}/step8_model_v2_ablation.json", encoding="utf-8"))
b = s8["tuning"]["best"]
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": b["num_leaves"], "min_data_in_leaf": b["min_data_in_leaf"],
         "lambda_l2": b["lambda_l2"], "feature_fraction": 0.8}
ROUNDS = s8["tuning"]["rounds"]
ev_json = json.load(open(f"{C.METRICS_DIR}/step3_flood_inventory_events.json", encoding="utf-8"))
ev_info = {e["event"]: e for e in ev_json["events"] if e["status"] == "kept"}
meta = json.load(open(f"{C.PROC_DIR}/features_v2/features_v2_metadata.json", encoding="utf-8"))
old = pd.read_parquet(C.DATASET_V2)
b2f = C.load_folds()

with rasterio.open(C.P1_STACK) as src:
    p1 = {n: src.read(i + 1) for i, n in enumerate(src.descriptions)}
    T30, H30, W30 = src.transform, src.height, src.width
with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
nd = C.load_ndem()
ndu = nd.to_crs(C.UTM)
nd["v_per_km"] = [sum(len(p.exterior.coords) for p in getattr(g, "geoms", [g])) / max(l / 1000, 1e-6)
                  for g, l in zip(ndu.geometry, ndu.length)]
nd["detailed"] = nd["v_per_km"] >= DETAIL_V_PER_KM

# ---------------------------------------------------------------------------
# 1-2. Label quality and radar + NDEM labels
# ---------------------------------------------------------------------------
print("\n[1/4] NDEM label quality and radar + NDEM positives")
frames, quality = [], {}
for ev in C.EVENTS_V2:
    info = ev_info[ev]
    with rasterio.open(os.path.join(C.PHASE2_ROOT, info["selected"]["water_classes"])) as src:
        cls = src.read(1)
        T20 = src.transform
    sel = nd[nd["date"].isin(info["ndem_dates"])]
    env_all, _ = C.ndem_mask(nd, info["ndem_dates"], cls.shape, T20)
    det = sel[sel["detailed"]]
    env_det = (rasterize([(g, 1) for g in det.geometry], out_shape=cls.shape, transform=T20, fill=0,
                         dtype="uint8").astype(bool) if len(det) else np.zeros(cls.shape, bool))
    pos_r, neg, _ = C.label_pools(cls, env_all, land, H30, W30)
    f_det = C.to_30m_fraction(env_det, H30, W30)
    f_valid = C.to_30m_fraction(cls != 255, H30, W30)
    f_belt = C.to_30m_fraction(C.dilate(cls == 2, 3), H30, W30)
    ndem_pos = land & (f_valid >= 1 - 1e-6) & (f_det >= 0.5) & (f_belt == 0) & ~pos_r
    pos2 = pos_r | ndem_pos
    assert not (pos2 & neg).any()
    row_km2 = C.pixel_area_km2(T30, H30)
    quality[ev] = {"polygons": int(len(sel)), "detailed_share": round(float(sel["detailed"].mean()), 3),
                   "median_vertices_per_km": round(float(sel["v_per_km"].median()), 1),
                   "radar_pos_km2": round(C.area_km2(pos_r, row_km2), 2),
                   "ndem_added_km2": round(C.area_km2(ndem_pos, row_km2), 2)}

    with rasterio.open(f"{C.PROC_DIR}/features_v2/{meta['files'][ev]}") as src:
        names = list(src.descriptions)
        stack = src.read()
    pos_i = np.flatnonzero(pos2.ravel())
    neg_i = rng.choice(np.flatnonzero(neg.ravel()), size=len(pos_i), replace=False)
    parts = []
    for tag, idx in (("pos", pos_i), ("neg", neg_i)):
        r, c = np.unravel_index(idx, (H30, W30))
        d = pd.DataFrame({n: stack[k][r, c] for k, n in enumerate(names)})
        d["twi"] = p1["twi"][r, c]
        d["rainfall_mean_annual_mm"] = p1["rainfall_mean_annual_mm"][r, c]
        d.insert(0, "set", tag); d.insert(1, "event", ev); d.insert(2, "row", r); d.insert(3, "col", c)
        d["src"] = np.where(pos_r[r, c], "radar", np.where(ndem_pos[r, c], "ndem", "neg"))
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    lon, lat, x, y = C.cell_xy_utm(d["row"].to_numpy(), d["col"].to_numpy(), T30)
    d["lon"], d["lat"], d["x_utm"], d["y_utm"] = lon, lat, x, y
    d["block10"] = C.block10(x, y)
    kb = C.KOCHI_BOX
    d["in_kochi"] = (lon >= kb[0]) & (lon <= kb[2]) & (lat >= kb[1]) & (lat <= kb[3])
    rnd = old[(old.event == ev) & (old.set == "rnd")].copy()
    rnd["src"] = "rnd"
    rnd["is_pos2"] = pos2[rnd["row"], rnd["col"]]
    d["is_pos2"] = d["set"] == "pos"
    frames += [d, rnd.drop(columns=[c for c in ("is_pos", "fold") if c in rnd])]

    prof = {}
    for g, m in (("radar flood", d.src == "radar"), ("NDEM-added flood", d.src == "ndem"), ("random land", None)):
        x_ = rnd if m is None else d[m]
        prof[g] = {"n": int(len(x_)), "dw_built": round(float((x_.landcover_class == 6).mean()), 3),
                   "paved_ge_0.2": round(float((x_.impervious_frac >= 0.2).mean()), 3),
                   "uica_ratio_ge_0.1": round(float((x_.uica_ratio >= 0.1).mean()), 3),
                   "hand_median_m": round(float(x_.hand_m.median()), 2)}
    quality[ev]["profile"] = prof
    print(f"    {ev}: {len(sel)} polygons, {quality[ev]['detailed_share']:.0%} detailed (median "
          f"{quality[ev]['median_vertices_per_km']} vert/km) -> radar {quality[ev]['radar_pos_km2']} km2 + NDEM "
          f"{quality[ev]['ndem_added_km2']} km2")
    for g, v in prof.items():
        print(f"        {g:17s} {v}")
res["label_quality"] = quality

df = pd.concat(frames, ignore_index=True)
df["landcover_class"] = df["landcover_class"].astype(int)
df["fold"] = df["block10"].map(b2f).astype(int)
assert df[C.FEATURES_D].notna().all().all()
res["dataset"] = {"rows_by_set_src": {f"{s}|{r}": int(n) for (s, r), n in df.groupby(["set", "src"]).size().items()}}

# ---------------------------------------------------------------------------
# 3. Models and urban tests
# ---------------------------------------------------------------------------
print("\n[2/4] C vs D, radar vs radar + NDEM labels, spatial and strict tests")
SUBSETS = {"all land": lambda d: np.ones(len(d), bool),
           "DW built": lambda d: (d.landcover_class == 6).to_numpy(),
           "paved >= 20 %": lambda d: (d.impervious_frac >= 0.2).to_numpy(),
           "UICA ratio >= 0.10": lambda d: (d.uica_ratio >= 0.10).to_numpy()}


def radar_view(tr):
    """Training rows as Week 3 saw them: radar positives only, rnd unlabelled."""
    return tr[(tr.src != "ndem")]


def score_eval(te, sc):
    """Flood (radar + NDEM truth) vs all other land, on each subset."""
    ev_rows = te[(te.set == "pos") | ((te.set == "rnd") & ~te.is_pos2)]
    y = (ev_rows.set == "pos").to_numpy().astype(int)
    s = sc.loc[ev_rows.index].to_numpy()
    out = {}
    for name, f in SUBSETS.items():
        m = f(ev_rows)
        n1 = int(y[m].sum())
        out[name] = {"n_flood": n1, "n_land": int((m & (y == 0)).sum()),
                     "auc": round(float(roc_auc_score(y[m], s[m])), 4) if n1 >= MIN_POS_SUBSET and (y[m] == 0).sum() >= 30 else None}
    nd_m = (ev_rows.src == "ndem").to_numpy()
    if nd_m.any():
        ref = sc.loc[te[te.set == "rnd"].index].to_numpy()
        out["ndem_added_in_high_vhigh"] = round(float((C.land_percentile(s[nd_m], ref) >= C.HV_PERCENTILE).mean()), 4)
    return out


def run(design, feats, labels):
    fp = C.make_fit_predict(design, feats, TUNED, ROUNDS, n_bags=N_BAGS)
    view = (lambda t: t) if labels == "radar+ndem" else radar_view
    out = {}
    sc = pd.Series(np.nan, index=df.index)
    for k in range(C.N_FOLDS):
        te = df[df.fold == k]
        sc.loc[te.index] = fp(view(df[df.fold != k]), te)
    out["spatial"] = score_eval(df, sc)
    te_all = df[df.event.isin(C.TEST_EVENTS)]
    sc = pd.Series(np.nan, index=te_all.index)
    for k in range(C.N_FOLDS):
        tr = df[df.event.isin(C.TRAIN_EVENTS) & (df.fold != k)]
        te = te_all[te_all.fold == k]
        sc.loc[te.index] = fp(view(tr), te)
    out["strict"] = {C.EVENT_LABEL[e]: score_eval(te_all[te_all.event == e], sc) for e in C.TEST_EVENTS}
    STRICT_SCORES[(labels, design, feats is C.FEATURE_SETS["D"])] = sc
    return out


STRICT_SCORES = {}


runs = {}
for labels in ("radar", "radar+ndem"):
    for design in ("standard", "pu_bagging"):
        for v in ("C", "D"):
            t1 = time.time()
            key = f"{labels}|{design}|{v}"
            runs[key] = run(design, C.FEATURE_SETS[v], labels)
            sp = runs[key]["spatial"]
            st = runs[key]["strict"]["2020"]
            fmt = lambda r: " | ".join(f"{n}: {r[n]['auc']}" for n in SUBSETS)
            print(f"    {key:28s} spatial  {fmt(sp)}  ({time.time() - t1:.0f} s)")
            print(f"    {'':28s} strict20 {fmt(st)}")
res["runs"] = runs


def gain(labels, design, protocol, subset, ev=None):
    get = lambda v: (runs[f"{labels}|{design}|{v}"][protocol] if ev is None
                     else runs[f"{labels}|{design}|{v}"][protocol][ev])[subset]["auc"]
    c, d_ = get("C"), get("D")
    return None if c is None or d_ is None else round(d_ - c, 4)


summary = {}
for labels in ("radar", "radar+ndem"):
    for design in ("standard", "pu_bagging"):
        summary[f"{labels}|{design}"] = {
            f"spatial|{s}": gain(labels, design, "spatial", s) for s in SUBSETS} | {
            f"strict2020|{s}": gain(labels, design, "strict", s, "2020") for s in SUBSETS}
res["D_minus_C"] = summary

# Paired block bootstrap (2 km blocks) of the strict-2020 D - C difference, radar + NDEM labels
print("\n  Bootstrap of strict 2020 D - C (2 km blocks, 1000 reps)")
te20 = df[(df.event == "2020-08_ndem")]
te20 = te20[(te20.set == "pos") | ((te20.set == "rnd") & ~te20.is_pos2)]
blk = ((te20.x_utm // 2000).astype(int).astype(str) + "_" + (te20.y_utm // 2000).astype(int).astype(str)).to_numpy()
ub = np.unique(blk)
idx_by_blk = {u: np.flatnonzero(blk == u) for u in ub}
yb = (te20.set == "pos").to_numpy().astype(int)
boot = {}
rb = np.random.default_rng(C.SEED)
draws = [rb.choice(ub, size=len(ub), replace=True) for _ in range(1000)]
for labels in ("radar", "radar+ndem"):
    for design in ("standard", "pu_bagging"):
        sC = STRICT_SCORES[(labels, design, False)].loc[te20.index].to_numpy()
        sD = STRICT_SCORES[(labels, design, True)].loc[te20.index].to_numpy()
        for name, f in SUBSETS.items():
            m = f(te20)
            diffs = []
            for dr in draws:
                ii = np.concatenate([idx_by_blk[u] for u in dr])
                ii = ii[m[ii]]
                if yb[ii].sum() < 10 or (yb[ii] == 0).sum() < 10:
                    continue
                diffs.append(roc_auc_score(yb[ii], sD[ii]) - roc_auc_score(yb[ii], sC[ii]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            boot[f"{labels}|{design}|{name}"] = {"ci95": [round(float(lo), 4), round(float(hi), 4)],
                                                 "p_D_better": round(float(np.mean(np.array(diffs) > 0)), 3),
                                                 "n_valid": len(diffs)}
            print(f"    {labels:10s} {design:10s} {name:18s} 95% CI [{lo:+.4f}, {hi:+.4f}], P(D>C) {np.mean(np.array(diffs) > 0):.2f}")
res["bootstrap_strict2020_D_minus_C"] = boot
print("\n  D - C (AUC):")
for k, v in summary.items():
    print(f"    {k:22s} " + "  ".join(f"{a}: {b:+.4f}" if b is not None else f"{a}: n/a" for a, b in v.items()))

# ---------------------------------------------------------------------------
# 4. 2024 map: built land and Kochi
# ---------------------------------------------------------------------------
print("\n[3/4] 2024 map (trained on 2018 + 2019)")
lm, rr, cc, land_ = C.land_matrix("2024-08_ndem")
built = lm["landcover_class"].to_numpy() == 6
kb = C.KOCHI_BOX
lon = T30.c + (cc + 0.5) * T30.a
lat = T30.f + (rr + 0.5) * T30.e
koc = (lon >= kb[0]) & (lon <= kb[2]) & (lat >= kb[1]) & (lat <= kb[3])
trn = df[df.event.isin(C.TRAIN_EVENTS)]
maps = {}
for key in ("radar|pu_bagging|C", "radar+ndem|pu_bagging|C", "radar+ndem|pu_bagging|D"):
    labels, design, v = key.split("|")
    fp = C.make_fit_predict(design, C.FEATURE_SETS[v], TUNED, ROUNDS, n_bags=N_BAGS)
    pm = C.percentile_map(fp(trn if labels == "radar+ndem" else radar_view(trn), lm), rr, cc, land_)
    vals = pm[rr, cc]
    maps[key] = {"built_in_high_vhigh": round(float((vals[built] >= C.HV_PERCENTILE).mean()), 4),
                 "kochi_median_pct": round(float(np.median(vals[koc])), 1),
                 "kochi_in_high_vhigh": round(float((vals[koc] >= C.HV_PERCENTILE).mean()), 4)}
    print(f"    {key:26s} {maps[key]}")
res["map_2024"] = maps

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[4/4] Figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
evs = C.EVENTS_V2
x = np.arange(len(evs))
axes[0].bar(x, [quality[e]["median_vertices_per_km"] for e in evs],
            color=[C.C_BLUE if quality[e]["detailed_share"] > 0.5 else C.C_GREY for e in evs])
axes[0].axhline(DETAIL_V_PER_KM, color="k", ls=":", lw=1)
axes[0].set_xticks(x); axes[0].set_xticklabels([C.EVENT_LABEL[e] for e in evs])
axes[0].set_ylabel("median vertices per km of outline"); axes[0].set_title("NDEM polygon detail (blue = used)")
for ax, key, title in ((axes[1], "paved_ge_0.2", "Share of cells >= 20 % paved"),
                       (axes[2], "uica_ratio_ge_0.1", "Share of cells with UICA ratio >= 0.10")):
    w = 0.27
    for j, (g, col) in enumerate(zip(("radar flood", "NDEM-added flood", "random land"),
                                     (C.C_BLUE, C.C_ORANGE, C.C_GREY))):
        ax.bar(x + (j - 1) * w, [quality[e]["profile"][g][key] for e in evs], w, color=col, label=g)
    ax.set_xticks(x); ax.set_xticklabels([C.EVENT_LABEL[e] for e in evs]); ax.set_title(title)
    ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
axes[2].legend(fontsize=8)
fig.suptitle("Step 9.0a  NDEM label quality: detailed polygons only; NDEM-added flood reaches paved, high-UICA land", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step9a_label_quality.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
names = list(SUBSETS)
for ax, (proto, ev, title) in zip(axes, (("spatial", None, "10 km spatial CV, 4 events"),
                                         ("strict", "2020", "Strict: new year + new place, 2020"))):
    w = 0.2
    combos = [("radar", "C", "#9bbbe0"), ("radar", "D", C.C_BLUE), ("radar+ndem", "C", "#f5b48f"), ("radar+ndem", "D", C.C_ORANGE)]
    for j, (lab, v, col) in enumerate(combos):
        r = runs[f"{lab}|pu_bagging|{v}"][proto]
        r = r if ev is None else r[ev]
        vals = [r[n]["auc"] or np.nan for n in names]
        ax.bar(np.arange(4) + (j - 1.5) * w, vals, w, color=col, label=f"{'radar' if lab == 'radar' else 'radar + NDEM'} labels, {v}")
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    lo = np.nanmin([((runs[f"{l}|pu_bagging|{v}"][proto] if ev is None
                      else runs[f"{l}|pu_bagging|{v}"][proto][ev])[n]["auc"] or np.nan)
                    for l, v, _ in combos for n in names])
    ax.set_ylim(max(0.5, lo - 0.05), 1.0); ax.set_title(title); ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
    ax.set_ylabel("AUC vs radar + NDEM truth")
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Step 9.0a  C vs D (PU bagging) on all land and urban subsets", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step9a_urban_tests.png", dpi=150)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step9a_ndem_labels.json")
print(f"\nDone in {res['runtime_s']} s")
