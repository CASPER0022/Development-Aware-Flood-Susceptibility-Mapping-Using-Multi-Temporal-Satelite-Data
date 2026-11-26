"""
Phase 2 Step 10 - Natural experiment: did places where development grew
actually flood more? (difference-in-differences, DiD)

Step 9 showed (by physics) that development sends more water to flood-prone
places. This step asks the observed flood record: between the earlier floods
(2018, 2019) and the later ones (2020, 2024), did flooding rise more where
upstream paving grew than in matched places where it did not?

DESIGN, fixed before running (no tuning on the result):
  Universe   floodplain land cells: HAND <= 5 m, valid in every radar scene, away
             from reference water. (HAND <= 10 m reported as a sensitivity.)
  Outcomes   R_e = radar flood (>= 50 % of the 30 m cell) in event e, all 4 events
             N_e = R_e OR >= 50 % inside a detailed NDEM polygon (2018, 2020 only;
                   2019 / 2024 NDEM polygons are coarse, Step 9.0a)
  Designs    D1 radar, 4 events: before = mean(R_2018, R_2019), after = mean(R_2020, R_2024)
             D2 radar + NDEM:    before = N_2018, after = N_2020   (plan: urban flooding visible)
             P  placebo:         before = R_2018, after = R_2019   (both before most growth;
                                 the 'effect' of FUTURE growth must be ~0)
  Treatments (growth between the periods; treated = top 10 % of the universe,
             control = bottom 25 % (near-zero growth))
             T1 upstream paving:   change in UICA ratio over the design's years   (plan 10.0)
             T2 local paving:      change in local impervious fraction
             T3 modelled water:    Step 9 extra frequent-storm runoff, 2025 vs 2017 (%)
  Matching   coarsened exact matching: strata = HAND bin x TWI quintile x distance-to-
             river quintile x event-rain tercile; DiD computed within strata and
             averaged with treated-cell weights.
  Effect     DiD = (after - before)_treated - (after - before)_control, in flood-rate
             percentage points; also the ratio (after/before)_T / (after/before)_C.
  Inference  2 km block bootstrap (500 reps): 95 % CI, and the minimum detectable
             effect (MDE = 2.8 x bootstrap SE, 80 % power at 5 %).
10.3 Temporal hold-out summary: Phase 1 vs Phase 2 models on held-out flood years.

Outputs: outputs/metrics/step10_natural_experiment.json
         outputs/maps/step10_natural_experiment.png
"""
import json
import os
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

HAND_MAX = 5.0
TOP_Q, CTRL_Q = 0.90, 0.25
N_BOOT = 500
C.banner("PHASE 2 STEP 10: NATURAL EXPERIMENT (DIFFERENCE-IN-DIFFERENCES)")
t0 = time.time()
res = {"step": 10, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
       "design": {"universe": f"HAND <= {HAND_MAX} m floodplain", "treated": f"top {100 - TOP_Q * 100:.0f} % growth",
                  "control": f"bottom {CTRL_Q * 100:.0f} % growth", "bootstrap": f"2 km blocks, {N_BOOT} reps"}}

# ---------------------------------------------------------------------------
# 1. Outcomes per event (30 m)
# ---------------------------------------------------------------------------
print("\n[1/4] Flood outcomes per event")
with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
    T30, H30, W30 = src.transform, src.height, src.width
ev_json = json.load(open(f"{C.METRICS_DIR}/step3_flood_inventory_events.json", encoding="utf-8"))
ev_info = {e["event"]: e for e in ev_json["events"] if e["status"] == "kept"}
nd = C.load_ndem()
ndu = nd.to_crs(C.UTM)
nd["v_per_km"] = [sum(len(p.exterior.coords) for p in getattr(g, "geoms", [g])) / max(l / 1000, 1e-6)
                  for g, l in zip(ndu.geometry, ndu.length)]
R, N, valid_all, belt_any = {}, {}, land.copy(), np.zeros_like(land)
for ev in C.EVENTS_V2:
    info = ev_info[ev]
    with rasterio.open(os.path.join(C.PHASE2_ROOT, info["selected"]["water_classes"])) as src:
        cls = src.read(1)
        T20 = src.transform
    valid_all &= C.to_30m_fraction(cls != 255, H30, W30) >= 1 - 1e-6
    belt_any |= C.to_30m_fraction(C.dilate(cls == 2, 3), H30, W30) > 0
    y = C.EVENT_LABEL[ev]
    R[y] = C.to_30m_fraction(cls == 1, H30, W30) >= 0.5
    det = nd[nd["date"].isin(info["ndem_dates"]) & (nd["v_per_km"] >= C.NDEM_DETAIL_V_PER_KM)]
    if len(det):
        m = rasterize([(g, 1) for g in det.geometry], out_shape=cls.shape, transform=T20, fill=0, dtype="uint8").astype(bool)
        N[y] = R[y] | (C.to_30m_fraction(m, H30, W30) >= 0.5)
print(f"    detailed NDEM available for: {sorted(N)}")

# ---------------------------------------------------------------------------
# 2. Covariates, treatments, universe
# ---------------------------------------------------------------------------
print("\n[2/4] Covariates and treatments")
meta = json.load(open(f"{C.PROC_DIR}/features_v2/features_v2_metadata.json", encoding="utf-8"))
with rasterio.open(f"{C.PROC_DIR}/features_v2/{meta['files']['2018-08_ndem']}") as src:
    nm = list(src.descriptions)
    hand = src.read(nm.index("hand_m") + 1)
    twi = src.read(nm.index("twi_full") + 1)
    drv = src.read(nm.index("dist_to_river_m") + 1)
    rain = src.read(nm.index("rain_event_5d_mm") + 1)
with rasterio.open(f"{C.PROC_DIR}/uica/uica_ratio_aoi.tif") as src:
    ur = {d[-4:]: src.read(i + 1) for i, d in enumerate(src.descriptions)}
with rasterio.open(f"{C.PROC_DIR}/landcover/impervious_fraction_30m.tif") as src:
    ip = {d[-4:]: src.read(i + 1) for i, d in enumerate(src.descriptions)}
with rasterio.open(f"{C.PROC_DIR}/scenarios/scenario_2025.tif") as src:
    more25 = src.read(list(src.descriptions).index("more_frequent_storm_water_vs_2017_pct") + 1)

base_u = land & valid_all & ~belt_any & np.isfinite(hand) & np.isfinite(more25)
rows, cols = np.nonzero(base_u)
_, _, xu, yu = C.cell_xy_utm(rows, cols, T30)
blk_ids, blk = np.unique((xu // 2000).astype(np.int64) * 100000 + (yu // 2000).astype(np.int64), return_inverse=True)
v = lambda a: a[rows, cols]
H, TW, DR, RA = v(hand), v(twi), v(drv), v(rain)


def qbin(x, q):
    e = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]))
    return np.searchsorted(e, x, side="right")


hand_bin = np.searchsorted([0.5, 1.0, 2.0, 3.0, 5.0, 7.5], H, side="right")
strata_all = (hand_bin * 1000 + qbin(TW, 5) * 100 + qbin(DR, 5) * 10 + qbin(RA, 3)).astype(np.int64)

DESIGNS = {
    "D1 radar, 4 events": {"before": lambda: (v(R["2018"]).astype(float) + v(R["2019"])) / 2,
                           "after": lambda: (v(R["2020"]).astype(float) + v(R["2024"])) / 2, "years": ("2018", "2024")},
    "D2 radar + NDEM, 2018 vs 2020": {"before": lambda: v(N["2018"]).astype(float),
                                      "after": lambda: v(N["2020"]).astype(float), "years": ("2018", "2020")},
    "P placebo, 2018 vs 2019": {"before": lambda: v(R["2018"]).astype(float),
                                "after": lambda: v(R["2019"]).astype(float), "years": ("2019", "2024")},
}
TREAT = {"T1 upstream paving (UICA ratio)": lambda y0, y1: v(ur[y1]) - v(ur[y0]),
         "T2 local paving": lambda y0, y1: v(ip[y1]) - v(ip[y0]),
         "T3 modelled extra water (Step 9)": lambda y0, y1: v(more25)}


def did_prepare(t_mask, c_mask, strata):
    ks, sidx = np.unique(strata, return_inverse=True)
    return {"T": t_mask, "C": c_mask, "sidx": sidx, "k": len(ks)}


def did(P, before, after, w=None):
    """Stratified DiD (treated-weighted) and ratio; w = cell weights (bootstrap)."""
    if w is None:
        w = np.ones(len(before))
    out = {}
    for name in ("T", "C"):
        m = P[name]
        out[name] = tuple(np.bincount(P["sidx"][m], weights=x[m], minlength=P["k"])
                          for x in (w, w * before, w * after))
    nT, bT, aT = out["T"]
    nC, bC, aC = out["C"]
    keep = (nT > 0) & (nC > 0)
    if not keep.any():
        return np.nan, np.nan, {}
    wt = nT[keep] / nT[keep].sum()
    rbT, raT = bT[keep] / nT[keep], aT[keep] / nT[keep]
    rbC, raC = bC[keep] / nC[keep], aC[keep] / nC[keep]
    d = float(np.sum(wt * ((raT - rbT) - (raC - rbC))))
    BT, AT, BC, AC = (float(np.sum(wt * x)) for x in (rbT, raT, rbC, raC))
    ratio = (AT / BT) / (AC / BC) if min(BT, BC, AC) > 0 else np.nan
    return d, ratio, {"before_T": BT, "after_T": AT, "before_C": BC, "after_C": AC,
                      "treated_cells_matched": int(round(nT[keep].sum())), "control_cells_matched": int(round(nC[keep].sum())),
                      "strata_used": int(keep.sum())}


# ---------------------------------------------------------------------------
# 3. Estimates with block bootstrap
# ---------------------------------------------------------------------------
print("\n[3/4] Difference-in-differences with 2 km block bootstrap")
rng = np.random.default_rng(C.SEED)
nb = len(blk_ids)
boots = [np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float) for _ in range(N_BOOT)]
est, BOOT = {}, {}
FLOOR_PP = 0.1          # treated 'before' flood rate below this: radar cannot see flooding there (floor effect)
for hmax in (HAND_MAX, 10.0):
    univ = H <= hmax
    for dname, D in DESIGNS.items():
        before, after = D["before"](), D["after"]()
        for tname, Tf in TREAT.items():
            g = Tf(*D["years"])
            gu = g[univ]
            hi, lo = np.quantile(gu, TOP_Q), np.quantile(gu, CTRL_Q)
            P = did_prepare(univ & (g >= hi) & (g > lo), univ & (g <= lo), strata_all)
            d, ratio, info = did(P, before, after)
            bs = np.array([did(P, before, after, w=bw[blk])[0] for bw in boots])
            key = f"HAND<={hmax:g} | {dname} | {tname}"
            BOOT[key] = bs
            se = float(np.nanstd(bs))
            est[key] = {"did_pp": round(100 * d, 3), "ci95_pp": [round(100 * float(np.nanpercentile(bs, 2.5)), 3),
                                                                round(100 * float(np.nanpercentile(bs, 97.5)), 3)],
                        "mde_pp": round(100 * 2.8 * se, 3), "ratio_of_ratios": round(float(ratio), 3) if np.isfinite(ratio) else None,
                        "growth_threshold_treated": round(float(hi), 5), "growth_threshold_control": round(float(lo), 5),
                        **{k: (round(100 * x, 3) if isinstance(x, float) else x) for k, x in info.items()}}
            e = est[key]
            e["floor_effect"] = bool(e["before_T"] < FLOOR_PP)
            rr_ = e["ratio_of_ratios"]
            e["ratio_disagrees"] = bool(rr_ is not None and ((rr_ < 1 and e["did_pp"] > 0) or (rr_ > 1 and e["did_pp"] < 0)))
            e["invalid"] = e["floor_effect"] or e["ratio_disagrees"]
            print(f"    {key:80s} DiD {e['did_pp']:+.2f} pp [{e['ci95_pp'][0]:+.2f}, {e['ci95_pp'][1]:+.2f}], "
                  f"MDE {e['mde_pp']:.2f}, T {e['before_T']:.2f}->{e['after_T']:.2f} %, "
                  f"C {e['before_C']:.2f}->{e['after_C']:.2f} %, ratio {rr_}"
                  f"{'  FLOOR EFFECT' if e['floor_effect'] else ''}{'  RATIO DISAGREES' if e['ratio_disagrees'] else ''}")

# placebo-corrected (triple difference): D1 effect minus placebo effect, paired bootstrap
corr = {}
for hmax in (HAND_MAX, 10.0):
    for tname in TREAT:
        k1 = f"HAND<={hmax:g} | D1 radar, 4 events | {tname}"
        kp = f"HAND<={hmax:g} | P placebo, 2018 vs 2019 | {tname}"
        diff = BOOT[k1] - BOOT[kp]
        corr[f"HAND<={hmax:g} | {tname}"] = {
            "did_minus_placebo_pp": round(est[k1]["did_pp"] - est[kp]["did_pp"], 3),
            "ci95_pp": [round(100 * float(np.nanpercentile(diff, 2.5)), 3), round(100 * float(np.nanpercentile(diff, 97.5)), 3)],
            "placebo_significant": bool(est[kp]["ci95_pp"][0] > 0 or est[kp]["ci95_pp"][1] < 0),
            "floor_effect": est[k1]["floor_effect"], "invalid": est[k1]["invalid"] or est[kp]["invalid"]}
        c_ = corr[f"HAND<={hmax:g} | {tname}"]
        print(f"    placebo-corrected HAND<={hmax:g} {tname:36s} {c_['did_minus_placebo_pp']:+.2f} pp "
              f"[{c_['ci95_pp'][0]:+.2f}, {c_['ci95_pp'][1]:+.2f}]  placebo significant: {c_['placebo_significant']}")
res["placebo_corrected"] = corr
res["estimates"] = est
res["universe_cells"] = {f"HAND<={h:g}": int((H <= h).sum()) for h in (HAND_MAX, 10.0)}
res["n_blocks"] = int(nb)

# ---------------------------------------------------------------------------
# 10.3 Temporal hold-out summary
# ---------------------------------------------------------------------------
s6 = json.load(open(f"{C.METRICS_DIR}/step6_eval_framework.json", encoding="utf-8"))["phase1_baseline"]
s8 = json.load(open(f"{C.METRICS_DIR}/step8_model_v2_ablation.json", encoding="utf-8"))["ablation_table"]
s9a = json.load(open(f"{C.METRICS_DIR}/step9a_ndem_labels.json", encoding="utf-8"))["runs"]
res["temporal_holdout_summary"] = {
    "phase1_2019_check (Phase 1 Step 10, Otsu flood vs all land)": 0.83,
    "phase1_model_temporal_2020_2024 (flood vs all land)": [s6["temporal"]["2020"]["auc_pos_vs_all_land"], s6["temporal"]["2024"]["auc_pos_vs_all_land"]],
    "phase1_model_strict_2020_2024": [s6["strict"]["2020"]["auc_pos_vs_all_land"], s6["strict"]["2024"]["auc_pos_vs_all_land"]],
    "phase2_chosen_pu_bagging_C_temporal_2020_2024": [s8["pu_bagging|C"]["temporal_2020_auc_land"], s8["pu_bagging|C"]["temporal_2024_auc_land"]],
    "phase2_chosen_pu_bagging_C_strict_2020_2024": [s8["pu_bagging|C"]["strict_2020_auc_land"], s8["pu_bagging|C"]["strict_2024_auc_land"]],
    "phase2_scenario_model_strict_2020_vs_radar_plus_ndem": s9a["radar+ndem|standard|D"]["strict"]["2020"]["all land"]["auc"]}
print(f"\n  10.3 temporal hold-out: {res['temporal_holdout_summary']}")

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
print("\n[4/4] Figure")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1])
ax = fig.add_subplot(gs[:, 0])
items = []
for k, e in est.items():
    if k.startswith(f"HAND<={HAND_MAX:g}"):
        tag = "  [floor effect]" if e["floor_effect"] else ("  [ratio disagrees]" if e["ratio_disagrees"] else "")
        items.append((k.split(" | ", 1)[1].replace(" | ", "\n") + tag, e["did_pp"], e["ci95_pp"],
                      C.C_YELLOW if e["invalid"] else
                      (C.C_GREY if "placebo" in k else (C.C_ORANGE if "D2" in k else C.C_BLUE))))
for k, e in corr.items():
    if k.startswith(f"HAND<={HAND_MAX:g}"):
        items.append(("D1 minus placebo\n" + k.split(" | ")[1] + ("  [invalid]" if e["invalid"] else ""),
                      e["did_minus_placebo_pp"], e["ci95_pp"], C.C_YELLOW if e["invalid"] else C.C_GREEN))
lab = [it[0] for it in items]
y = np.arange(len(items))[::-1]
for yi, (_, dv, ci, col) in zip(y, items):
    ax.errorbar(dv, yi, xerr=[[dv - ci[0]], [ci[1] - dv]], fmt="o", color=col, capsize=3)
    ax.text(ci[1], yi, f"  {dv:+.2f}", va="center", fontsize=7.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
ax.set_xlabel("difference-in-differences: extra change in flood rate where development grew (percentage points)")
ax.set_title("Did flooding rise more where development grew?\n(95 % CI, 2 km block bootstrap; floodplain HAND ≤ 5 m)\n"
             "blue/orange: main designs, grey: placebo, green: placebo-corrected, yellow: invalid (floor effect or ratio disagrees)", fontsize=9)
ax.grid(axis="x", color=C.C_GRID)
for j, dname in enumerate(("D1 radar, 4 events", "D2 radar + NDEM, 2018 vs 2020")):
    ax2 = fig.add_subplot(gs[j, 1])
    k = f"HAND<={HAND_MAX:g} | {dname} | T1 upstream paving (UICA ratio)"
    e = est[k]
    ax2.plot([0, 1], [e["before_T"], e["after_T"]], "o-", color=C.C_ORANGE, lw=2, label="treated: upstream paving grew most")
    ax2.plot([0, 1], [e["before_C"], e["after_C"]], "o-", color=C.C_GREY, lw=2, label="matched control: no growth")
    ax2.plot([0, 1], [e["before_T"], e["before_T"] + e["after_C"] - e["before_C"]], ":", color=C.C_ORANGE,
             label="treated if it had followed control")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["before", "after"]); ax2.set_ylabel("flooded share of cells (%)")
    ax2.set_title(f"{dname}\nT1 upstream paving: DiD {e['did_pp']:+.2f} pp", fontsize=9); ax2.legend(fontsize=7)
    ax2.grid(color=C.C_GRID)
ax3 = fig.add_subplot(gs[:, 2])
k = f"HAND<={HAND_MAX:g} | D1 radar, 4 events | T1 upstream paving (UICA ratio)"
g = TREAT["T1 upstream paving (UICA ratio)"]("2018", "2024")
univ = H <= HAND_MAX
tm = univ & (g >= est[k]["growth_threshold_treated"]) & (g > est[k]["growth_threshold_control"])
cm = univ & (g <= est[k]["growth_threshold_control"])
img = np.full((H30, W30, 3), 1.0)
img[~land] = (0.80, 0.86, 0.93)
img[rows[univ], cols[univ]] = (0.88, 0.88, 0.88)
img[rows[cm], cols[cm]] = (0.55, 0.55, 0.55)
img[rows[tm], cols[tm]] = (0.92, 0.41, 0.20)
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
ax3.imshow(img, extent=ext, interpolation="nearest")
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_title("Floodplain (HAND ≤ 5 m)\norange: treated (top 10 %\nupstream-paving growth)\ndark grey: control", fontsize=9)
fig.suptitle("Step 10  Natural experiment: observed flooding before (2018/2019) vs after (2020/2024) development", fontsize=11)
fig.savefig(f"{C.MAP_DIR}/step10_natural_experiment.png", dpi=140)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step10_natural_experiment.json")
print(f"\nDone in {res['runtime_s']} s")
