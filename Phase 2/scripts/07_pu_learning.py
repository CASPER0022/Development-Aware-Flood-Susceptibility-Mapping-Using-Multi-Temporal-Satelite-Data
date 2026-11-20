"""
Phase 2 Step 7 - PU-learning (Week 3).

7.1 Dataset v2 (Step 6): positives = confirmed flood cells of all kept events;
    unlabelled = random land cells ('rnd', no longer called "non-flood"),
    with block10, fold and event columns.
7.2 Four models on the same features (stack v2, feature set D) and the same
    Step 6 folds:
      standard    confirmed flood vs confident non-flood (Phase 1 design)
      pu_naive    flood vs unlabelled, no correction
      pu_elkan    Elkan & Noto (2008), c = P(labelled | flood) from held-out positives
      pu_bagging  Mordelet & Vert (2014), 25 bags
7.3 Evaluated on what PU cannot fake:
      - recall on held-out events: share of 2020 / 2024 flood in the top 40 % of land
      - NDEM-only flood (peak water the radar pass missed): share in the top 40 %
      - Kochi city core: median land percentile (the Phase 1 blind spot)
      - built-up land: share in the top 40 % (radar positives are 0.1 % built)
    AUC (flood vs confident non-flood, and flood vs all land) is reported for
    comparability. Elkan-Noto rescaling is monotone, so its AUC equals pu_naive's
    ranking; it changes the probability scale (c and the implied flood prior).

Outputs
    outputs/metrics/step7_pu_learning.json
    outputs/maps/step7_pu_comparison.png, step7_pu_kochi_maps.png
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

C.banner("PHASE 2 STEP 7: PU-LEARNING VS STANDARD")
t0 = time.time()
df = pd.read_parquet(C.DATASET_V2)
FEATS = C.FEATURES_D
KINDS = ["standard", "pu_naive", "pu_elkan", "pu_bagging"]
NICE = {"standard": "Standard (flood vs dry)", "pu_naive": "PU naive", "pu_elkan": "PU Elkan-Noto",
        "pu_bagging": "PU bagging"}
res = {"step": 7, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "features": FEATS,
       "rounds": 400, "pu_bags": C.PU_BAGS, "models": {}}

for kind in KINDS:
    t1 = time.time()
    fp = C.make_fit_predict(kind, FEATS)
    r = C.run_protocols(df, fp)
    r["headline"] = C.headline(r)
    if kind == "pu_elkan":
        r["c_last_fit"] = round(fp.last_c, 4)
    res["models"][kind] = r
    h = r["headline"]
    print(f"\n  {NICE[kind]} ({time.time() - t1:.0f} s)")
    print(f"    spatial AUC {h['spatial_auc']:.4f}")
    for p in ("temporal", "strict"):
        for e in ("2020", "2024"):
            m = r[p][e]
            print(f"    {p:8s} {e}: AUC {m['auc']:.3f} | vs land {m['auc_pos_vs_all_land']:.3f} | flood in H+VH "
                  f"{m['pos_in_high_vhigh']:.3f} | NDEM-only H+VH {m.get('ndem_only_in_high_vhigh', float('nan')):.3f}"
                  f" | Kochi median pct {m.get('kochi_median_pct', float('nan')):.1f}")

# ---------------------------------------------------------------------------
# Full-map check for 2024 (train 2018 + 2019): built-up land and Kochi
# ---------------------------------------------------------------------------
print("\n  Full 2024 map, trained on 2018 + 2019")
lm, rr, cc, land = C.land_matrix("2024-08_ndem")
tr = df[df.event.isin(C.TRAIN_EVENTS)]
maps, full = {}, {}
built = lm["landcover_class"].to_numpy() == 6
for kind in KINDS:
    sc = C.make_fit_predict(kind, FEATS)(tr, lm)
    pm = C.percentile_map(sc, rr, cc, land)
    maps[kind] = pm
    v = pm[rr, cc]
    full[kind] = {"built_in_high_vhigh": round(float((v[built] >= C.HV_PERCENTILE).mean()), 4),
                  "built_share_of_high_vhigh": round(float(built[v >= C.HV_PERCENTILE].mean()), 4),
                  "elkan_mean_score_land": round(float(np.mean(sc)), 4) if kind == "pu_elkan" else None}
    if kind == "pu_elkan":
        full[kind]["implied_flood_prior"] = round(float(np.mean(sc)), 4)
    print(f"    {NICE[kind]:24s} built land in H+VH {full[kind]['built_in_high_vhigh']:.3f}; "
          f"built share of H+VH {full[kind]['built_share_of_high_vhigh']:.3f}")
res["full_map_2024"] = full
res["land_built_share"] = round(float(built.mean()), 4)

# ---------------------------------------------------------------------------
# Choice for Step 8
# ---------------------------------------------------------------------------
def score(k):
    m = res["models"][k]["strict"]
    return np.mean([m[e]["auc_pos_vs_all_land"] for e in ("2020", "2024")])


best = max(KINDS, key=score)
res["choice"] = {"by": "mean strict (new year + new place) AUC flood vs all land, 2020 and 2024",
                 "scores": {k: round(float(score(k)), 4) for k in KINDS}, "best": best}
print(f"\n  Strict AUC vs all land: " + ", ".join(f"{k} {v}" for k, v in res["choice"]["scores"].items())
      + f"  -> best: {best}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
cols = [C.C_GREY, C.C_BLUE, C.C_GREEN, C.C_ORANGE]
panels = [("Strict AUC\nflood vs all land", lambda k: score(k)),
          ("Held-out flood in\nHigh + Very High", lambda k: np.mean([res["models"][k]["strict"][e]["pos_in_high_vhigh"] for e in ("2020", "2024")])),
          ("NDEM-only flood in\nHigh + Very High", lambda k: np.mean([res["models"][k]["strict"][e]["ndem_only_in_high_vhigh"] for e in ("2020", "2024")])),
          ("Kochi core: median\nland percentile", lambda k: np.mean([res["models"][k]["strict"][e]["kochi_median_pct"] for e in ("2020", "2024")]) / 100),
          ("Built-up land in\nHigh + Very High (2024 map)", lambda k: full[k]["built_in_high_vhigh"])]
fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.6), constrained_layout=True)
for ax, (t, f) in zip(axes, panels):
    v = [f(k) for k in KINDS]
    b = ax.bar(range(4), v, color=cols)
    for i, x in enumerate(v):
        ax.text(i, x + 0.01, f"{x:.3f}" if t.startswith("Strict") else f"{x:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Std", "PU\nnaive", "PU\nElkan", "PU\nbag"])
    ax.set_ylim((0.9, 1.0) if t.startswith("Strict") else (0, 1.1)); ax.set_title(t, fontsize=9); ax.grid(axis="y", color=C.C_GRID); ax.set_axisbelow(True)
fig.suptitle("Step 7  Standard vs PU-learning (feature set D, mean of 2020 and 2024 under the strict test)", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step7_pu_comparison.png", dpi=150)
plt.close(fig)

kb = C.KOCHI_BOX
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), constrained_layout=True)
cmap = plt.get_cmap("RdYlBu_r", 5)
for j, kind in enumerate(KINDS):
    for i, zoom in enumerate((False, True)):
        ax = axes[i, j]
        im = ax.imshow(np.floor(maps[kind] / 20).clip(0, 4), cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext,
                       interpolation="nearest")
        if zoom:
            ax.set_xlim(kb[0] - 0.03, kb[2] + 0.03); ax.set_ylim(kb[1] - 0.03, kb[3] + 0.03)
            ax.add_patch(plt.Rectangle((kb[0], kb[1]), kb[2] - kb[0], kb[3] - kb[1], fill=False, lw=1.2, ec="k"))
        else:
            ax.add_patch(plt.Rectangle((kb[0], kb[1]), kb[2] - kb[0], kb[3] - kb[1], fill=False, lw=1, ec="k"))
            ax.set_title(NICE[kind])
        ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, shrink=0.6, ticks=range(5))
cb.ax.set_yticklabels(["Very Low", "Low", "Moderate", "High", "Very High"])
fig.suptitle("Step 7  2024 susceptibility classes (trained on 2018 + 2019). Bottom: Kochi city core (box)", fontsize=10)
fig.savefig(f"{C.MAP_DIR}/step7_pu_kochi_maps.png", dpi=130)
plt.close(fig)

res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step7_pu_learning.json")
print(f"\nDone in {res['runtime_s']} s")
