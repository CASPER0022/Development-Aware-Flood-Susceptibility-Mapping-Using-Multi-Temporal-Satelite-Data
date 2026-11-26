"""
Phase 2 Step 12 - Final maps, headline tables and architecture status.

12.1 Report maps (legend, scale bar, north arrow, lon/lat ticks):
     a. Phase 1 vs Phase 2 susceptibility (Phase 1 map vs the chosen Phase 2 model:
        PU bagging, set C, trained on 2018 + 2019, applied to the 2024 stack)
     b. Riverine susceptibility under development: 2025, 2030 trend, class change
        (Step 9 scenario model)
     c. Development effect (headline): frequent-storm extra water 2017 -> 2030 and
        development-impact classes with named towns
12.2 Headline tables: Phase 1 vs Phase 2 under the same tests, and the effect of
     development (ML / physics / natural experiment / waterlogging check).
12.3 Architecture status diagram: each block done in Phase 1, Phase 2, or planned
     for Phase 3.
Outputs: outputs/maps/step12_*.png, outputs/metrics/step12_headline_tables.json,
         data/processed/scenarios/phase2_current_susceptibility.tif (gitignored)
"""
import json
import time

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch

import common as C

C.banner("PHASE 2 STEP 12: FINAL MAPS, HEADLINE TABLES, ARCHITECTURE STATUS")
t0 = time.time()
M = C.METRICS_DIR
J = lambda f: json.load(open(f"{M}/{f}", encoding="utf-8"))
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
CLS5 = ListedColormap(["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"])
CLS_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
TOWNS = {"Aluva": (10.108, 76.351), "Kalamassery": (10.051, 76.316), "North Paravur": (10.143, 76.228),
         "Angamaly": (10.196, 76.386), "Perumbavoor": (10.115, 76.477), "Kakkanad": (10.016, 76.341),
         "Tripunithura": (9.944, 76.349), "Edappally": (10.024, 76.308), "Ernakulam": (9.982, 76.282),
         "Kolenchery": (9.980, 76.470), "Nedumbassery": (10.155, 76.390), "Kalady": (10.166, 76.438)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def decorate(ax, towns=False, scale_km=10):
    """Lon/lat ticks, scale bar (km at ~10 N), north arrow, optional towns."""
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([76.2, 76.3, 76.4, 76.5]); ax.set_yticks([9.95, 10.05, 10.15, 10.25])
    ax.set_xticklabels([f"{x:.1f}°E" for x in ax.get_xticks()], fontsize=7)
    ax.set_yticklabels([f"{y:.2f}°N" for y in ax.get_yticks()], fontsize=7)
    deg = scale_km / (111.32 * np.cos(np.radians(10.07)))
    x0, y0 = ext[0] + 0.015, ext[2] + 0.015
    ax.add_patch(plt.Rectangle((x0, y0), deg / 2, 0.006, fc="k", ec="k", zorder=6))
    ax.add_patch(plt.Rectangle((x0 + deg / 2, y0), deg / 2, 0.006, fc="w", ec="k", zorder=6))
    ax.text(x0, y0 + 0.009, "0", fontsize=7, ha="center", zorder=6)
    ax.text(x0 + deg, y0 + 0.009, f"{scale_km} km", fontsize=7, ha="center", zorder=6)
    ax.annotate("N", xy=(ext[1] - 0.025, ext[3] - 0.02), xytext=(ext[1] - 0.025, ext[3] - 0.065),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5), ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)
    if towns:
        for n, (la, lo) in TOWNS.items():
            ax.plot(lo, la, "o", ms=3, mfc="k", mec="w", zorder=7)
            ax.text(lo + 0.004, la + 0.003, n, fontsize=6.5, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
    T30, prof = src.transform, src.profile
H, W = land.shape

# ---------------------------------------------------------------------------
# 12.1a Phase 2 current susceptibility (chosen model) and Phase 1 vs Phase 2
# ---------------------------------------------------------------------------
print("\n[1/5] Phase 2 current susceptibility map (chosen model)")
s8 = J("step8_model_v2_ablation.json")
b = s8["tuning"]["best"]
TUNED = {**C.LGB_PARAMS_V2, "num_leaves": b["num_leaves"], "min_data_in_leaf": b["min_data_in_leaf"],
         "lambda_l2": b["lambda_l2"], "feature_fraction": 0.8}
df = pd.read_parquet(C.DATASET_V2)
tr = df[df.event.isin(C.TRAIN_EVENTS)]
lm, rr, cc, _ = C.land_matrix("2024-08_ndem")
score = C.make_fit_predict("pu_bagging", C.FEATURES_C, TUNED, s8["tuning"]["rounds"])(tr, lm)
pct2 = C.percentile_map(score, rr, cc, land)
cls2 = np.where(land, np.clip(np.floor(pct2 / 20), 0, 4) + 1, 0).astype(np.uint8)
with rasterio.open(f"{C.PROC_DIR}/scenarios/phase2_current_susceptibility.tif", "w",
                   **{**prof, "count": 2, "dtype": "float32", "nodata": np.nan, "compress": "deflate"}) as dst:
    dst.write(pct2.astype(np.float32), 1); dst.set_band_description(1, "land_percentile")
    dst.write(cls2.astype(np.float32), 2); dst.set_band_description(2, "class_1VL_5VH_quintiles")
with rasterio.open(C.p1("data", "processed", "susceptibility", "susceptibility_class_all.tif")) as src:
    cls1 = src.read(1)
agree = float((cls1[land] == cls2[land]).mean())
within1 = float((np.abs(cls1[land].astype(int) - cls2[land].astype(int)) <= 1).mean())
print(f"    Phase 1 vs Phase 2 classes: same {agree:.1%}, within one class {within1:.1%}")
fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
norm = BoundaryNorm(np.arange(0.5, 6.5), 5)
for ax, cl, t in ((axes[0], cls1, "Phase 1: 11 features, 2018 flood, random design"),
                  (axes[1], cls2, "Phase 2: HAND, event rain, land cover; PU bagging;\ntrained 2018 + 2019, shown for 2024")):
    ax.imshow(np.where(cl > 0, cl, np.nan), cmap=CLS5, norm=norm, extent=ext, interpolation="nearest")
    decorate(ax, towns=True); ax.set_title(t, fontsize=10)
d = np.where(land, cls2.astype(int) - cls1.astype(int), np.nan)
im = axes[2].imshow(d, cmap="PuOr_r", vmin=-4, vmax=4, extent=ext, interpolation="nearest")
decorate(axes[2]); axes[2].set_title(f"Class change (Phase 2 − Phase 1)\nsame class {agree:.0%}, within one class {within1:.0%}",
                                     fontsize=10)
fig.colorbar(im, ax=axes[2], shrink=0.7, label="classes")
fig.legend(handles=[Patch(color=CLS5(i), label=l) for i, l in enumerate(CLS_LABELS)], loc="lower center", ncol=5,
           bbox_to_anchor=(0.36, -0.04), fontsize=9, title="Flood susceptibility (quintiles of land)")
fig.suptitle("Phase 1 vs Phase 2 flood susceptibility, Lower Periyar (Aluva–Kochi)", fontsize=12)
fig.savefig(f"{C.MAP_DIR}/step12_map_phase1_vs_phase2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 12.1b riverine susceptibility under development (Step 9 scenario rasters)
# ---------------------------------------------------------------------------
print("\n[2/5] Current vs post-development maps")
sc = {}
for s in ("2025", "2030_trend"):
    with rasterio.open(f"{C.PROC_DIR}/scenarios/scenario_{s}.tif") as src:
        dsc = list(src.descriptions)
        sc[s] = {n: src.read(i + 1) for i, n in enumerate(dsc)}
fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
for ax, s, t in ((axes[0], "2025", "Current (2025 paving)"), (axes[1], "2030_trend", "Post-development (2030, current trend)")):
    ax.imshow(sc[s]["riverine_class_fixed_breaks"], cmap=CLS5, norm=norm, extent=ext, interpolation="nearest")
    decorate(ax, towns=True); ax.set_title(f"Riverine flood susceptibility: {t}", fontsize=10)
dcl = sc["2030_trend"]["riverine_class_fixed_breaks"] - sc["2025"]["riverine_class_fixed_breaks"]
up = float(np.nanmean(dcl[land] >= 1))
axes[2].imshow(np.where(dcl >= 1, 1, np.where(land, 0, np.nan)), cmap=ListedColormap(["#eeeeee", "#d7191c"]),
               vmin=0, vmax=1, extent=ext, interpolation="nearest")
decorate(axes[2]); axes[2].set_title(f"Cells moving up a class 2025 → 2030: {100 * up:.2f} % of land\n"
                                     "(big riverine floods barely change; see the water map)", fontsize=10)
fig.legend(handles=[Patch(color=CLS5(i), label=l) for i, l in enumerate(CLS_LABELS)], loc="lower center", ncol=5,
           bbox_to_anchor=(0.36, -0.04), fontsize=9, title="Riverine susceptibility (fixed class breaks of 2025)")
fig.suptitle("Current vs post-development riverine flood susceptibility (physically constrained scenario model)", fontsize=12)
fig.savefig(f"{C.MAP_DIR}/step12_map_current_vs_post_development.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 12.1c development effect (headline map)
# ---------------------------------------------------------------------------
print("\n[3/5] Development-effect map")
s9 = J("step9_dual_prediction.json")
lb = s9["layer_B_waterlogging_pressure"]
fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
im = axes[0].imshow(np.clip(sc["2030_trend"]["more_frequent_storm_water_vs_2017_pct"], 0, 30), cmap="magma_r", vmin=0,
                    vmax=30, extent=ext, interpolation="nearest")
decorate(axes[0], towns=True)
fig.colorbar(im, ax=axes[0], shrink=0.65, label="% more storm water arriving than in 2017")
axes[0].set_title(f"Extra water from development in a frequent 50 mm storm, 2017 → 2030\n"
                  f"land with ≥ 5 % more water: {lb['2025']['aoi_km2_more_water']:.0f} km² (2025) → "
                  f"{lb['2030 trend']['aoi_km2_more_water']:.0f} km² (2030)", fontsize=10)
icm = ListedColormap(["#f2f2f2", "#9bbbe0", "#f5b48f", "#c0262d"])
axes[1].imshow(sc["2030_trend"]["impact_class"], cmap=icm, vmin=-0.5, vmax=3.5, extent=ext, interpolation="nearest")
decorate(axes[1], towns=True)
axes[1].legend(handles=[Patch(color=icm(3), label=f"Hotspot: flood-prone + ≥5 % more water ({lb['2030 trend']['km2_hotspot']:.0f} km²)"),
                        Patch(color=icm(2), label=f"Emerging: ≥5 % more water ({lb['2030 trend']['km2_emerging']:.0f} km²)"),
                        Patch(color=icm(1), label="Flood-prone (riverine) only"), Patch(color=icm(0), label="Neither")],
               loc="lower right", fontsize=8, framealpha=0.95)
axes[1].set_title(f"Development-impact classes, 2030 (trend)\nhotspots {lb['2025']['km2_hotspot']:.0f} km² (2025) → "
                  f"{lb['2030 trend']['km2_hotspot']:.0f} km² (2030); "
                  f"{100 * lb['2025']['upstream_share_of_extra_water_median_in_more_water_cells']:.0f} % of the extra water "
                  f"comes from upstream paving", fontsize=10)
fig.suptitle("Effect of development on flooding: development adds water where land is already flood-prone", fontsize=12)
fig.savefig(f"{C.MAP_DIR}/step12_map_development_effect.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 12.2 headline tables
# ---------------------------------------------------------------------------
print("\n[4/5] Headline tables")
s1 = J("step1_baseline_reproduction.json")
s6 = J("step6_eval_framework.json")["phase1_baseline"]
abl = s8["ablation_table"]
s9a = J("step9a_ndem_labels.json")
s9b = J("step9b_constrained_model.json")
s9c = J("step9c_extra_runoff.json")
s10 = J("step10_natural_experiment.json")
s10b = J("step10b_waterlogging_check.json")
s11 = J("step11_shap.json")
pb = abl["pu_bagging|C"]
t1 = [["Test", "Phase 1 model", "Phase 2 model"],
      ["Random 70/30 split (Phase 1 protocol)", f"{s6['random_split_2018']['auc']:.3f}", f"{abl['standard|C']['random_split_auc']:.3f}"],
      ["Unseen 10 km regions (4 events)", f"{s6['spatial']['auc']['mean']:.3f}", f"{pb['spatial_auc']:.3f}"],
      ["New year (2020 / 2024), flood vs all land",
       f"{s6['temporal']['2020']['auc_pos_vs_all_land']:.3f} / {s6['temporal']['2024']['auc_pos_vs_all_land']:.3f}",
       f"{pb['temporal_2020_auc_land']:.3f} / {pb['temporal_2024_auc_land']:.3f}"],
      ["New year + new place (strict), flood vs all land",
       f"{s6['strict']['2020']['auc_pos_vs_all_land']:.3f} / {s6['strict']['2024']['auc_pos_vs_all_land']:.3f}",
       f"{pb['strict_2020_auc_land']:.3f} / {pb['strict_2024_auc_land']:.3f}"]]
nat = s10["placebo_corrected"]["HAND<=5 | T1 upstream paving (UICA ratio)"]
wl = s10b["summary"]
t2 = [["Evidence", "Big riverine floods", "Frequent storms / waterlogging"],
      ["ML, radar labels (Step 8)", f"UICA adds {s8['ablation_summary']['pu_bagging: D minus C (strict)']:+.4f} AUC", "not observable"],
      ["ML, radar + NDEM labels (9.0a)", "paved land: +0.037 AUC (CI > 0, one test year)", "not observable"],
      ["SHAP share of development (11)",
       f"{100 * s11['global']['M1']['group_share']['development (paving, UICA)']:.1f} % (chosen model)", "–"],
      ["Physics, SCS-CN (9.0c, 9)",
       f"+{s9c['catchment_totals']['monsoon']['extra_pct']:.2f} % runoff (monsoon storm)",
       f"+{s9c['catchment_totals']['frequent']['extra_pct']:.1f} % runoff; hotspots "
       f"{lb['2025']['km2_hotspot']:.0f} → {lb['2030 trend']['km2_hotspot']:.0f} km²"],
      ["Natural experiment (10)", f"no effect: {nat['did_minus_placebo_pp']:+.2f} pp [{nat['ci95_pp'][0]:+.2f}, {nat['ci95_pp'][1]:+.2f}]",
       "not testable"],
      ["Reported waterlogging spots (10b)", "–",
       f"UICA ratio {wl['UICA ratio 2025 (upstream paving)']['mean_percentile']:.0f}th pct of urban land "
       f"({wl['UICA ratio 2025 (upstream paving)']['spots_above_50']}/15 above; not significant)"]]
C.save_json({"phase1_vs_phase2": t1, "effect_of_development": t2,
             "map_agreement_phase1_phase2": {"same_class": round(agree, 4), "within_one_class": round(within1, 4)},
             "riverine_up_a_class_2025_to_2030": round(up, 5)}, f"{M}/step12_headline_tables.json")
for t in (t1, t2):
    for r in t:
        print("    | " + " | ".join(r) + " |")

fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), constrained_layout=True)
for ax, t, ttl in ((axes[0], t1, "Phase 1 vs Phase 2 (AUC, same tests)"), (axes[1], t2, "Effect of development on flooding")):
    ax.axis("off")
    tb = ax.table(cellText=t[1:], colLabels=t[0], loc="center", cellLoc="left", colWidths=[0.34, 0.33, 0.33])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
    for (r_, c_), cell in tb.get_celld().items():
        if r_ == 0:
            cell.set_facecolor("#dae3ee"); cell.set_text_props(weight="bold")
    ax.set_title(ttl, fontsize=11)
fig.savefig(f"{C.MAP_DIR}/step12_headline_tables.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 12.3 architecture status
# ---------------------------------------------------------------------------
print("\n[5/5] Architecture status diagram")
P1, P2, P3 = "#dbe8f5", "#fde5cf", "#eeeeee"
blocks = [  # (x, y, w, h, text, colour, dashed)
    (0.2, 4.2, 2.2, 0.9, "Sentinel-1 SAR\n2017–2025", P1, False), (0.2, 3.1, 2.2, 0.9, "FABDEM, HydroSHEDS,\nCHIRPS, OSM", P1, False),
    (0.2, 2.0, 2.2, 0.9, "Dynamic World, GHSL\n(paving 2017–2030)", P2, False), (0.2, 0.9, 2.2, 0.9, "NDEM maps +\nwaterlogging reports", P2, False),
    (3.0, 4.2, 2.6, 0.9, "Flood inventory\n4 validated events", P2, False), (3.0, 3.1, 2.6, 0.9, "Terrain & hydrology\nfull-basin FD8, HAND", P2, False),
    (3.0, 2.0, 2.6, 0.9, "UICA / ΔUICA\n(upstream paving)", P2, False), (3.0, 0.9, 2.6, 0.9, "Physics: SCS-CN\nextra runoff", P2, False),
    (6.2, 3.6, 2.8, 1.0, "LightGBM + PU bagging\nstrict test 0.985", P2, False), (6.2, 2.3, 2.8, 1.0, "Constrained scenario\nmodel (monotone)", P2, False),
    (6.2, 0.9, 2.8, 1.0, "Natural experiment +\nwaterlogging check", P2, False),
    (9.6, 3.6, 2.8, 1.0, "Susceptibility map\n(Phase 1 baseline → Phase 2)", P2, False), (9.6, 2.3, 2.8, 1.0, "Current vs 2030 maps,\ndevelopment hotspots, SHAP", P2, False),
    (9.6, 0.9, 2.8, 1.0, "Observed proof of the\ndevelopment effect (WP1)", P3, True),
    (12.9, 2.3, 2.4, 1.0, "Planner dashboard,\nreport", P3, True)]
fig, ax = plt.subplots(figsize=(15.5, 5.6))
for x, y, w, h, txt, col, dash in blocks:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=col, ec="#555555",
                                lw=1.2, ls="--" if dash else "-"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
for (x1, y1), (x2, y2) in [((2.4, 4.65), (3.0, 4.65)), ((2.4, 3.55), (3.0, 3.55)), ((2.4, 2.45), (3.0, 2.45)), ((2.4, 1.35), (3.0, 1.35)),
                           ((5.6, 4.65), (6.2, 4.1)), ((5.6, 3.55), (6.2, 4.0)), ((5.6, 2.45), (6.2, 2.8)), ((5.6, 1.35), (6.2, 1.4)),
                           ((9.0, 4.1), (9.6, 4.1)), ((9.0, 2.8), (9.6, 2.8)), ((9.0, 1.4), (9.6, 1.4)), ((12.4, 2.8), (12.9, 2.8))]:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#555555"))
ax.legend(handles=[Patch(fc=P1, ec="#555555", label="Done in Phase 1"), Patch(fc=P2, ec="#555555", label="Done in Phase 2"),
                   Patch(fc=P3, ec="#555555", ls="--", label="Planned for Phase 3")], loc="upper right", fontsize=9)
ax.set_xlim(0, 15.5); ax.set_ylim(0.6, 5.4); ax.axis("off")
ax.set_title("System architecture: status at the end of Phase 2", fontsize=12)
fig.savefig(f"{C.MAP_DIR}/step12_architecture_status.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nDone in {time.time() - t0:.0f} s")
