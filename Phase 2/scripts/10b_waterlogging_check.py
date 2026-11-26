"""
Phase 2 Step 10b - Waterlogging check: first OBSERVED test of the development layers.

Satellite and NDEM maps record only the big riverine floods. Urban waterlogging,
where Step 9 says development matters, is recorded only in news reports. This
step uses reported waterlogging spots in Kochi as observed points.

Data: data/raw/waterlogging/kochi_waterlogging_reports.csv (15 spots, each with a
news / official source; events 21 Oct 2019 (~20 cm in 4 h) and 28 May 2024
cloudburst, plus recurring reports), geocoded with OpenStreetMap Nominatim
(kochi_waterlogging_spots_geocoded.csv; 6 doubtful / missing ones re-queried and
fixed by hand below, with the OSM feature used).

Test (fixed before running): for each spot take the land cells within 300 m.
Compare the spot's median value with CONTROL cells = urban land (paved fraction
>= 0.2) within 15 km of MG Road, as a land percentile (50 = no different).
  paved fraction 2025          local development level
  UICA ratio 2025              upstream development level (share of catchment paved)
  more water 2017 -> 2025 (%)  Step 9 frequent-storm extra water (development GROWTH)
  HAND                         low-lying terrain (expected: spots low)
  riverine susceptibility      Step 9 scenario model (expected: does NOT flag them)
Statistics: mean percentile over the 15 spots with a bootstrap CI over spots, and
a one-sided sign / Wilcoxon test against 50.
Outputs: outputs/metrics/step10b_waterlogging_check.json, outputs/maps/step10b_waterlogging_check.png
"""
import csv
import json
import time

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import common as C

RADIUS_M = 300
FIX = {"Ernakulam South railway station": (9.96900, 76.29095, "station: Ernakulam Junction"),
       "KSRTC bus stand Ernakulam": (9.97755, 76.28898, "bus_station: KSRTC Bus Station Ernakulam"),
       "Marottichuvadu": (10.02337, 76.31287, "neighbourhood: Marottichodu, Edappally"),
       "Thrikkakara": (10.02285, 76.33355, "Thrikkakara municipality (NPOL area)"),
       "Tripunithura": (9.94403, 76.34899, "town: Thrippunithura"),
       "Edappally": (10.02421, 76.30849, "Edappally Junction")}
MG = (9.9827, 76.2826)
C.banner("PHASE 2 STEP 10b: REPORTED WATERLOGGING SPOTS VS DEVELOPMENT LAYERS")
t0 = time.time()
rows = list(csv.DictReader(open(f"{C.RAW_DIR}/waterlogging/kochi_waterlogging_spots_geocoded.csv", encoding="utf-8")))
for r in rows:
    if r["spot"] in FIX:
        la, lo, nm = FIX[r["spot"]]
        r["lat"], r["lon"], r["osm_name"] = la, lo, nm
    r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])

with rasterio.open(C.P1_DOMAIN) as src:
    land = src.read(1).astype(bool)
    T = src.transform
H, W = land.shape
rr, cc = np.nonzero(land)
lon = T.c + (cc + 0.5) * T.a
lat = T.f + (rr + 0.5) * T.e
with rasterio.open(f"{C.PROC_DIR}/landcover/impervious_fraction_30m.tif") as src:
    paved = src.read(src.count)[rr, cc]
with rasterio.open(f"{C.PROC_DIR}/uica/uica_ratio_aoi.tif") as src:
    uica_r = src.read(src.count)[rr, cc]
with rasterio.open(f"{C.PROC_DIR}/features_v2/stack_v2_2024-08_ndem.tif") as src:
    hand = src.read(list(src.descriptions).index("hand_m") + 1)[rr, cc]
with rasterio.open(f"{C.PROC_DIR}/scenarios/scenario_2025.tif") as src:
    d = list(src.descriptions)
    more = src.read(d.index("more_frequent_storm_water_vs_2017_pct") + 1)[rr, cc]
    rsus = src.read(d.index("riverine_susceptibility") + 1)[rr, cc]
LAYERS = {"paved fraction 2025": paved, "UICA ratio 2025 (upstream paving)": uica_r,
          "more water 2017-2025 (%) (Step 9)": more, "HAND (m)": hand, "riverine susceptibility (Step 9)": rsus}
km = lambda la, lo: np.hypot((lat - la) * 110.6, (lon - lo) * 109.6)
dist_mg = km(*MG)
ctrl = (paved >= 0.2) & (dist_mg <= 15)
res = {"step": "10b", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_spots": len(rows), "radius_m": RADIUS_M,
       "control": "urban land (paved >= 0.2) within 15 km of MG Road", "n_control_cells": int(ctrl.sum()),
       "spots": [], "summary": {}}
pct = {k: [] for k in LAYERS}
for r in rows:
    m = km(r["lat"], r["lon"]) <= RADIUS_M / 1000
    s = {"spot": r["spot"], "lat": r["lat"], "lon": r["lon"], "event": r["event"], "source": r["source"], "n_cells": int(m.sum())}
    for k, v in LAYERS.items():
        val = float(np.nanmedian(v[m])) if m.any() else np.nan
        ref = v[ctrl]
        ref = ref[np.isfinite(ref)]
        p = float(100 * (np.mean(ref < val) + 0.5 * np.mean(ref == val)))
        s[k] = round(val, 4)
        s[k + " | percentile vs urban"] = round(p, 1)
        pct[k].append(p)
    res["spots"].append(s)
    print(f"    {r['spot']:32s} cells {s['n_cells']:3d} | " + " | ".join(
        f"{k.split(' (')[0][:16]} {s[k + ' | percentile vs urban']:5.1f}" for k in LAYERS))
rng = np.random.default_rng(C.SEED)
for k, ps in pct.items():
    ps = np.array(ps)
    boot = [rng.choice(ps, len(ps)).mean() for _ in range(5000)]
    try:
        pg = float(wilcoxon(ps - 50, alternative="greater").pvalue)
        pl = float(wilcoxon(ps - 50, alternative="less").pvalue)
    except ValueError:
        pg = pl = None
    res["summary"][k] = {"mean_percentile": round(float(ps.mean()), 1),
                         "ci95": [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)],
                         "spots_above_50": int((ps > 50).sum()),
                         "wilcoxon_p_higher": round(pg, 4) if pg is not None else None,
                         "wilcoxon_p_lower": round(pl, 4) if pl is not None else None}
    print(f"  {k:40s} mean pct {ps.mean():5.1f} [{res['summary'][k]['ci95'][0]}, {res['summary'][k]['ci95'][1]}], "
          f"{(ps > 50).sum()}/{len(ps)} above 50, p(higher) {res['summary'][k]['wilcoxon_p_higher']}, "
          f"p(lower) {res['summary'][k]['wilcoxon_p_lower']}")

# sensitivity: without the airport (its 2019 closure was river flooding, not urban waterlogging)
keep = np.array(["Airport" not in r["spot"] for r in rows])
res["summary_without_airport"] = {}
for k, ps in pct.items():
    ps = np.array(ps)[keep]
    res["summary_without_airport"][k] = {"mean_percentile": round(float(ps.mean()), 1), "spots_above_50": int((ps > 50).sum()),
                                         "n": int(len(ps)),
                                         "wilcoxon_p_higher": round(float(wilcoxon(ps - 50, alternative="greater").pvalue), 4),
                                         "wilcoxon_p_lower": round(float(wilcoxon(ps - 50, alternative="less").pvalue), 4)}
print("  without airport:", {k: (v["mean_percentile"], v["wilcoxon_p_higher"]) for k, v in res["summary_without_airport"].items()})

# figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
ax = fig.add_subplot(gs[0])
uimg = np.full((H, W), np.nan, np.float32)
uimg[rr, cc] = uica_r
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
im = ax.imshow(uimg, cmap="Greys", vmin=0, vmax=0.6, extent=ext, interpolation="nearest")
ax.scatter([r["lon"] for r in rows], [r["lat"] for r in rows], s=40, c=C.C_ORANGE, edgecolors="k", zorder=3)
for r in rows:
    ax.text(r["lon"] + 0.003, r["lat"], r["spot"].split(" (")[0], fontsize=6.5)
ax.set_xlim(76.24, 76.42); ax.set_ylim(9.90, 10.18)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.7, label="UICA ratio 2025 (share of upstream catchment paved)")
ax.set_title("Reported waterlogging spots (news / official, 2018-2024)\nover upstream paving (UICA ratio)")
ax = fig.add_subplot(gs[1])
keys = list(LAYERS)
for i, k in enumerate(keys):
    ps = np.array(pct[k])
    ax.scatter(ps, np.full(len(ps), i) + rng.uniform(-0.15, 0.15, len(ps)), s=18, color=C.C_GREY, alpha=0.7)
    smy = res["summary"][k]
    ax.errorbar(smy["mean_percentile"], i, xerr=[[smy["mean_percentile"] - smy["ci95"][0]], [smy["ci95"][1] - smy["mean_percentile"]]],
                fmt="D", color=C.C_ORANGE, capsize=4, ms=7)
    ax.text(101, i, f"{smy['mean_percentile']:.0f}", va="center", fontsize=8)
ax.axvline(50, color="k", ls="--", lw=0.8)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
ax.set_xlim(0, 105); ax.set_xlabel("percentile of the spot among other urban land (50 = no different)")
ax.set_title("Are waterlogging spots different from other urban land?\n(grey: each spot; orange: mean with 95 % CI)")
ax.grid(axis="x", color=C.C_GRID)
fig.suptitle("Step 10b  Reported waterlogging spots in Kochi vs the development layers", fontsize=11)
fig.savefig(f"{C.MAP_DIR}/step10b_waterlogging_check.png", dpi=140)
plt.close(fig)
res["runtime_s"] = round(time.time() - t0, 1)
C.save_json(res, f"{C.METRICS_DIR}/step10b_waterlogging_check.json")
print(f"Done in {res['runtime_s']} s")
