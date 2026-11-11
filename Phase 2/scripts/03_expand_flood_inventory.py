"""
Phase 2 Step 3 - Expand the flood inventory (more events, both orbits).

3.0 Regression test first: the Step 5 Otsu chain, now in common.py, is rerun
    on the Phase 1 2018 scene pair; the class raster must equal Phase 1's
    water_classes_2018_20180821.tif pixel for pixel.
3.1 Candidate events 2017-2025 from two independent lists, merged:
      - every NDEM (NRSC/ISRO) inundation-map date over the AOI, grouped into
        events (dates <= 14 days apart);
      - CHIRPS daily AOI-mean rainfall: the largest 3-day totals, declustered
        (>= 15 days apart), top N_RAIN_PEAKS, plus the wettest months.
3.2 Sentinel-1 VV/IW acquisitions within +-3 days of the event (NDEM events:
    of any NDEM date; rainfall events: of the 3-day peak), in BOTH pass
    directions. One acquisition = all frames of one date and relative orbit,
    mosaicked. Each needs a dry reference of the same relative orbit
    (Jan 15 - Apr 30 of the same year, closest to Mar 20). Sentinel-1B ended
    in Dec 2021, so 2022-2024 events have about half the passes.
    All scenes are sampled onto the Phase 1 20 m inventory grid exactly.
3.3 Every acquisition goes through the Phase 1 Step 5 chain (Lee + Eq. 1,
    per-image Otsu, change vs dry ref, slope > 5 deg and < 5 px removed) and
    the Step 6 NDEM cross-check (event envelope, 40 m tolerance).
    KEEP rule (fixed before running, calibrated on Phase 1: 2018 kept,
    2019 kept, 2021 rejected):
      a. AOI coverage >= 80 % (flood and dry scene)
      b. both Otsu thresholds in the open-water range -24 .. -12 dB
      c. bimodal flood-scene histogram: Otsu separability eta >= ETA_MIN
      d. flood area >= 2 km2
      e. NDEM exists for the event and GTI_40m >= 0.30 with lift >= 3
    An event without NDEM cannot pass (e): it is recorded as "unverified".
    The event inventory is the passing acquisition with the highest GTI_40m.
3.4 (Optional urban double-bounce positives) not attempted in this step.

Outputs
    data/raw/sentinel1/*.tif                                  cached scenes (gitignored)
    data/processed/flood_inventory/water_classes_<event>.tif  0 dry, 1 flood, 2 ref water, 255 nodata
    outputs/metrics/step3_flood_inventory_events.json, step3_event_table.csv
    outputs/maps/step3_event_table.png, step3_kept_events.png, step3_chirps_candidates.png
"""
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import common as C

RAW = C.p2("data", "raw", "sentinel1")
OUT = C.p2("data", "processed", "flood_inventory")
for d in (RAW, OUT):
    os.makedirs(d, exist_ok=True)

YEAR0, YEAR1 = 2017, 2025
WINDOW_DAYS = 3
NDEM_GROUP_DAYS = 14
N_RAIN_PEAKS = 10
DECLUSTER_DAYS = 15
DRY_WINDOW = ("01-15", "04-30")
DRY_TARGET = "03-20"
MIN_COVERAGE = 0.80
ETA_MIN = 0.75        # sanity floor: Phase 1 scenes 0.86-0.89 (sea/backwater make every scene bimodal)
MIN_FLOOD_KM2 = 2.0
GTI_MIN, LIFT_MIN = 0.30, 3.0
P1_PAIR_2018 = ("data/raw/sentinel1_gee/s1_vv_2018_flood_20180821.tif", "data/raw/sentinel1_gee/s1_vv_2018_dry_")

C.banner("PHASE 2 STEP 3: EXPANDED FLOOD INVENTORY (2017-2025, both orbits)")
t_start = time.time()
res = {"step": 3, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
       "keep_rule": {"min_coverage": MIN_COVERAGE, "otsu_range_db": C.PLAUSIBLE_WATER_THRESHOLD_DB,
                     "eta_min": ETA_MIN, "min_flood_km2": MIN_FLOOD_KM2, "gti_40m_min": GTI_MIN,
                     "lift_min": LIFT_MIN, "window_days": WINDOW_DAYS}}

# Phase 1 20 m inventory grid
with rasterio.open(f"{C.P1_INV_DIR}/water_classes_2018_20180821.tif") as src:
    P1_CLS_2018 = src.read(1)
    T20 = src.transform
    H20, W20 = src.shape
ROW_AREA = C.pixel_area_km2(T20, H20)
PROFILE20 = {"driver": "GTiff", "height": H20, "width": W20, "count": 1, "crs": "EPSG:4326", "transform": T20}


def slope_on_grid():
    """FABDEM resampled (bilinear) to the 20 m grid, slope in degrees (Step 5)."""
    import math
    dem = np.full((H20, W20), np.nan, dtype="float32")
    with rasterio.open(C.P1_FABDEM) as src:
        reproject(rasterio.band(src, 1), dem, dst_transform=T20, dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear, dst_nodata=np.nan)
    mid_lat = C.BBOX[1] + (C.BBOX[3] - C.BBOX[1]) / 2
    dx_m = abs(T20.a) * 111320 * math.cos(math.radians(mid_lat))
    dy_m = abs(T20.e) * 110574
    gy, gx = np.gradient(dem.astype("float64"), dy_m, dx_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


SLOPE20 = slope_on_grid()


def flood_map(flood_raw, fvalid, dry_raw, dvalid):
    """Phase 1 Step 5 chain. Returns class raster and diagnostics."""
    valid = fvalid & dvalid
    flood_db, _ = C.preprocess(flood_raw, valid, "flood", verbose=False)
    dry_db, _ = C.preprocess(dry_raw, valid, "dry", verbose=False)
    t_f, ok_f, eta_f, _ = C.otsu_threshold(flood_db, valid)
    t_d, ok_d, eta_d, _ = C.otsu_threshold(dry_db, valid)
    wf = valid & (flood_db < t_f)
    wd = valid & (dry_db < t_d)
    raw_new = wf & ~wd
    after_slope = raw_new & ~(SLOPE20 > C.SLOPE_MAX_DEG)
    flood, _ = C.remove_small_patches(after_slope, C.MIN_PATCH_PX)
    cls = np.where(~valid, 255, np.where(wd, 2, np.where(flood, 1, 0))).astype(np.uint8)
    return cls, {"otsu_flood_db": round(t_f, 3), "otsu_dry_db": round(t_d, 3), "otsu_ok": bool(ok_f and ok_d),
                 "eta_flood": round(eta_f, 4), "eta_dry": round(eta_d, 4),
                 "flood_km2": round(C.area_km2(flood, ROW_AREA), 2),
                 "ref_water_km2": round(C.area_km2(wd, ROW_AREA), 2)}, flood_db


# ---------------------------------------------------------------------------
# 3.0 Regression test on the Phase 1 2018 pair
# ---------------------------------------------------------------------------
print("\n[1/7] 3.0 Regression test: Phase 1 2018 scene pair through the shared chain")
import glob
f18 = C.p1(P1_PAIR_2018[0])
d18 = glob.glob(C.p1(P1_PAIR_2018[1] + "*.tif"))[0]
fr, fv, fp = C.load_s1(f18)
dr, dv, _ = C.load_s1(d18)
assert fp["transform"] == T20
cls18, diag18, _ = flood_map(fr, fv, dr, dv)
same = np.array_equal(cls18, P1_CLS_2018)
print(f"    class raster identical to Phase 1: {same}; flood {diag18['flood_km2']} km2 (Phase 1: 23.54), "
      f"Otsu {diag18['otsu_flood_db']} dB, eta {diag18['eta_flood']}")
assert same, "shared Step 5 chain does not reproduce Phase 1 2018"
res["regression_2018"] = {"identical_to_phase1": bool(same), **diag18}

# Phase 1 eta values, used to report where ETA_MIN sits
p1_eta = {}
for tag, ff in (("2019", "2019_flood_20190810"), ("2021", "2021_flood_20211016")):
    a = glob.glob(C.p1(f"data/raw/sentinel1_gee/s1_vv_{ff}.tif"))
    b = glob.glob(C.p1(f"data/raw/sentinel1_gee/s1_vv_{tag}_dry_*.tif"))
    if a and b:
        r1, v1, _ = C.load_s1(a[0]); r2, v2, _ = C.load_s1(b[0])
        p1_eta[tag] = flood_map(r1, v1, r2, v2)[1]["eta_flood"]
p1_eta["2018"] = diag18["eta_flood"]
res["phase1_eta_flood"] = p1_eta
print(f"    Phase 1 flood-scene eta: {p1_eta} (ETA_MIN = {ETA_MIN})")

# ---------------------------------------------------------------------------
# 3.1 Candidate events
# ---------------------------------------------------------------------------
print("\n[2/7] 3.1 Candidate events (NDEM dates + CHIRPS extremes)")
ee = C.ee_init()
region = ee.Geometry.Rectangle(C.BBOX)

chirps_csv = f"{C.RAW_DIR}/chirps_daily_aoi_mean_{YEAR0}_{YEAR1}.csv"
if not os.path.exists(chirps_csv):
    parts = []
    for y in range(YEAR0, YEAR1 + 1):
        col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(f"{y}-01-01", f"{y + 1}-01-01")
        fc = col.map(lambda im: ee.Feature(None, {
            "date": im.date().format("YYYY-MM-dd"),
            "mm": im.reduceRegion(ee.Reducer.mean(), region, 5000).get("precipitation")}))
        d = fc.aggregate_array("date").getInfo()
        v = fc.aggregate_array("mm").getInfo()
        parts.append(pd.DataFrame({"date": d, "mm": v}))
    pd.concat(parts).to_csv(chirps_csv, index=False)
rain = pd.read_csv(chirps_csv, parse_dates=["date"]).set_index("date")["mm"].asfreq("D").fillna(0)
r3 = rain.rolling(3).sum()
peaks, taken = [], r3.copy()
while len(peaks) < N_RAIN_PEAKS:
    d = taken.idxmax()
    peaks.append((d, float(taken[d])))
    taken[(taken.index > d - timedelta(days=DECLUSTER_DAYS)) & (taken.index < d + timedelta(days=DECLUSTER_DAYS))] = -1
monthly = rain.resample("MS").sum().sort_values(ascending=False)
print(f"    CHIRPS {rain.index[0].date()} - {rain.index[-1].date()}: top 3-day totals " +
      ", ".join(f"{d.date()} {v:.0f} mm" for d, v in peaks))

ndem = C.load_ndem()
nd_dates = sorted(set(pd.to_datetime(ndem["date"], format="%d-%m-%Y")))
nd_dates = [d for d in nd_dates if YEAR0 <= d.year <= YEAR1]
groups = []
for d in nd_dates:
    if groups and (d - groups[-1][-1]).days <= NDEM_GROUP_DAYS:
        groups[-1].append(d)
    else:
        groups.append([d])

events = []
for g in groups:
    events.append({"event": f"{g[0]:%Y-%m}_ndem", "kind": "NDEM", "ndem_dates": [f"{d:%d-%m-%Y}" for d in g],
                   "start": g[0] - timedelta(days=WINDOW_DAYS), "end": g[-1] + timedelta(days=WINDOW_DAYS),
                   "n_ndem_polygons": int(ndem["date"].isin([f"{d:%d-%m-%Y}" for d in g]).sum())})
for d, v in peaks:      # 3-day total ending on d -> peak window around the 3-day block
    s, e = d - timedelta(days=2 + WINDOW_DAYS), d + timedelta(days=WINDOW_DAYS)
    hit = next((ev for ev in events if ev["start"] <= d <= ev["end"] + timedelta(days=WINDOW_DAYS)), None)
    if hit:
        hit.setdefault("rain_3day_mm", round(v, 1))
        hit.setdefault("rain_peak_end", f"{d.date()}")
        continue
    events.append({"event": f"{d:%Y-%m-%d}_rain", "kind": "CHIRPS", "ndem_dates": [], "start": s, "end": e,
                   "rain_3day_mm": round(v, 1), "rain_peak_end": f"{d.date()}", "n_ndem_polygons": 0})
for ev in events:
    win = (rain.index >= ev["start"] - timedelta(days=5)) & (rain.index <= ev["end"])
    ev.setdefault("rain_3day_mm", round(float(r3[win].max()), 1))
    ev["month_total_mm"] = round(float(rain[f"{ev['start']:%Y-%m}"].sum()), 1)
events.sort(key=lambda e: e["start"])
for ev in events:
    print(f"    {ev['event']:14s} {ev['kind']:6s} window {ev['start'].date()} - {ev['end'].date()}  "
          f"3-day max {ev['rain_3day_mm']:6.1f} mm  NDEM polygons {ev['n_ndem_polygons']}")

# ---------------------------------------------------------------------------
# 3.2 Sentinel-1 acquisitions (both passes) and dry references
# ---------------------------------------------------------------------------
print("\n[3/7] 3.2 Sentinel-1 acquisitions (+-3 days, both passes, same-orbit dry reference)")


def acquisitions(start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    info = col.toList(500).map(lambda im: ee.Dictionary({
        "id": ee.Image(im).id(), "t": ee.Image(im).get("system:time_start"),
        "pass": ee.Image(im).get("orbitProperties_pass"), "orbit": ee.Image(im).get("relativeOrbitNumber_start"),
        "platform": ee.Image(im).get("platform_number")})).getInfo()
    acq = {}
    for i in info:
        date = datetime.fromtimestamp(i["t"] / 1000, timezone.utc).strftime("%Y-%m-%d")
        key = (date, int(i["orbit"]))
        a = acq.setdefault(key, {"date": date, "orbit": int(i["orbit"]), "pass": i["pass"],
                                 "platform": i["platform"], "ids": []})
        a["ids"].append(i["id"])
    return list(acq.values())


def download(acq, path):
    if not os.path.exists(path):
        imgs = ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{i}") for i in acq["ids"]])
        img = imgs.select("VV").mosaic().toFloat().unmask(-9999).rename("VV")
        C.ee_download_grid(img, T20, W20, H20, path, tiles=(2, 1))
        with rasterio.open(path, "r+") as dst:
            dst.nodata = -9999
    return C.load_s1(path)


scene_rows = []
for ev in events:
    s = ev["start"].strftime("%Y-%m-%d")
    e = (ev["end"] + timedelta(days=1)).strftime("%Y-%m-%d")
    acqs = acquisitions(s, e)
    ev["n_acquisitions"] = len(acqs)
    if not acqs:
        print(f"    {ev['event']}: no Sentinel-1 acquisition in window")
    y = ev["start"].year
    dry_all = acquisitions(f"{y}-{DRY_WINDOW[0]}", f"{y}-{DRY_WINDOW[1]}")
    for a in sorted(acqs, key=lambda a: (a["date"], a["orbit"])):
        same = [d for d in dry_all if d["orbit"] == a["orbit"]]
        row = {"event": ev["event"], "kind": ev["kind"], "date": a["date"], "pass": a["pass"],
               "orbit": a["orbit"], "platform": a["platform"], "n_frames": len(a["ids"])}
        if not same:
            row.update(status="rejected", reason="no same-orbit dry reference")
            scene_rows.append(row)
            print(f"    {ev['event']} {a['date']} {a['pass'][:4]} orbit {a['orbit']}: no same-orbit dry reference")
            continue
        tgt = datetime.strptime(f"{y}-{DRY_TARGET}", "%Y-%m-%d")
        dry = min(same, key=lambda d: abs((datetime.strptime(d["date"], "%Y-%m-%d") - tgt).days))
        row["dry_date"] = dry["date"]
        row["_files"] = (f"{RAW}/s1_vv_{a['date'].replace('-', '')}_o{a['orbit']}.tif",
                         f"{RAW}/s1_vv_{dry['date'].replace('-', '')}_o{dry['orbit']}.tif")
        row["_acq"] = (a, dry)
        scene_rows.append(row)
        print(f"    {ev['event']} {a['date']} {a['pass'][:4]} orbit {a['orbit']:3d} S1{a['platform']} "
              f"({len(a['ids'])} frame) dry {dry['date']}")

# ---------------------------------------------------------------------------
# 3.3 Flood mapping + NDEM cross-check + keep rule
# ---------------------------------------------------------------------------
print("\n[4/7] 3.3 Otsu inventory + NDEM cross-check per acquisition")
ev_by = {ev["event"]: ev for ev in events}
cls_cache = {}
for row in scene_rows:
    if "_files" not in row:
        continue
    a, dry = row.pop("_acq")
    ff, df_ = row.pop("_files")
    t = time.time()
    fr, fv, _ = download(a, ff)
    dr, dv, _ = download(dry, df_)
    cov_f, cov_d = float(fv.mean()), float(dv.mean())
    row.update(coverage_flood=round(cov_f, 3), coverage_dry=round(cov_d, 3))
    cls, diag, _ = flood_map(fr, fv, dr, dv)
    row.update(diag)
    ev = ev_by[row["event"]]
    gti = None
    if ev["ndem_dates"]:
        env, n_env = C.ndem_mask(ndem, ev["ndem_dates"], (H20, W20), T20)
        domain = (cls != 255) & (cls != 2)
        gti = C.agreement(cls == 1, env, domain, ROW_AREA)
        row.update(ndem_km2=gti["reference_km2"], GTI_px=gti["GTI_px"], GTI_px_40m=gti["GTI_px_40m"],
                   random_40m=gti["random_baseline_40m"], lift_40m=gti["lift_px_40m"], POD=gti["POD"],
                   CSI=gti["CSI"])
    fails = []
    if min(cov_f, cov_d) < MIN_COVERAGE:
        fails.append(f"coverage {min(cov_f, cov_d):.2f} < {MIN_COVERAGE}")
    if not diag["otsu_ok"]:
        fails.append(f"Otsu {diag['otsu_flood_db']}/{diag['otsu_dry_db']} dB outside water range")
    if diag["eta_flood"] < ETA_MIN:
        fails.append(f"eta {diag['eta_flood']:.2f} < {ETA_MIN} (weak bimodality)")
    if diag["flood_km2"] < MIN_FLOOD_KM2:
        fails.append(f"flood {diag['flood_km2']} km2 < {MIN_FLOOD_KM2}")
    if gti is None:
        status = "unverified" if not fails else "rejected"
        fails.append("no NDEM map for this event (GTI not computable)")
    else:
        if gti["GTI_px_40m"] < GTI_MIN or (gti["lift_px_40m"] or 0) < LIFT_MIN:
            fails.append(f"GTI_40m {gti['GTI_px_40m']:.2f} (lift {gti['lift_px_40m']}) below {GTI_MIN}/{LIFT_MIN}")
        status = "kept" if not fails else "rejected"
    row.update(status=status, reason="; ".join(fails) if fails else "passes all checks")
    tag = f"{row['date'].replace('-', '')}_o{row['orbit']}"
    path = f"{OUT}/water_classes_{tag}.tif"
    with rasterio.open(path, "w", **PROFILE20, dtype="uint8", nodata=255, compress="lzw") as dst:
        dst.write(cls, 1)
    row["water_classes"] = os.path.relpath(path, C.PHASE2_ROOT)
    cls_cache[(row["event"], row["date"], row["orbit"])] = cls
    g = f"GTI40 {row.get('GTI_px_40m', float('nan')):.3f} lift {row.get('lift_40m') or float('nan'):.1f}" \
        if gti else "no NDEM"
    print(f"    {row['event']:14s} {row['date']} {row['pass'][:4]} o{row['orbit']:<3d} flood {diag['flood_km2']:6.2f} km2 "
          f"Otsu {diag['otsu_flood_db']:6.2f} eta {diag['eta_flood']:.2f} | {g} -> {status.upper()} "
          f"({time.time() - t:.0f} s)")

table = pd.DataFrame(scene_rows)
table.to_csv(f"{C.METRICS_DIR}/step3_event_table.csv", index=False)

# event-level selection
event_summary = []
for ev in events:
    rows = table[table["event"] == ev["event"]]
    kept = rows[rows["status"] == "kept"]
    best = kept.sort_values("GTI_px_40m", ascending=False).iloc[0] if len(kept) else None
    st = "kept" if best is not None else ("unverified" if (rows["status"] == "unverified").any() else "rejected")
    if not len(rows):
        st = "rejected"
    event_summary.append({
        "event": ev["event"], "kind": ev["kind"], "window": [f"{ev['start'].date()}", f"{ev['end'].date()}"],
        "ndem_dates": ev["ndem_dates"], "n_ndem_polygons": ev["n_ndem_polygons"],
        "rain_3day_mm": ev["rain_3day_mm"], "month_total_mm": ev["month_total_mm"],
        "n_acquisitions": ev.get("n_acquisitions", 0), "status": st,
        "selected": None if best is None else {
            "date": best["date"], "pass": best["pass"], "orbit": int(best["orbit"]), "dry_date": best["dry_date"],
            "flood_km2": float(best["flood_km2"]), "GTI_px_40m": float(best["GTI_px_40m"]),
            "lift_40m": float(best["lift_40m"]), "water_classes": best["water_classes"]}})
res["events"] = event_summary
res["acquisitions"] = table.replace({np.nan: None}).to_dict("records")
kept_ev = [e for e in event_summary if e["status"] == "kept"]
res["summary"] = {"n_candidate_events": len(events), "n_acquisitions": int(len(table)),
                  "n_acquisitions_kept": int((table["status"] == "kept").sum()),
                  "n_events_kept": len(kept_ev), "kept_events": [e["event"] for e in kept_ev],
                  "temporal_holdout_possible": len(kept_ev) >= 3}
print(f"\n    events kept: {len(kept_ev)} / {len(events)} -> {[e['event'] for e in kept_ev]}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[5/7] Figures")
# CHIRPS candidates
fig, ax = plt.subplots(figsize=(16, 4.8), dpi=110)
ax.plot(rain.index, rain.values, color="#9aa4ad", lw=0.6, label="CHIRPS daily AOI mean")
ax.plot(r3.index, r3.values, color=C.C_BLUE, lw=0.8, label="3-day total")
for d, v in peaks:
    ax.plot(d, v, "v", color=C.C_ORANGE, ms=7)
for g in groups:
    ax.axvspan(g[0] - timedelta(days=2), g[-1] + timedelta(days=2), color=C.C_YELLOW, alpha=0.35, lw=0)
ax.legend(handles=[Line2D([0], [0], color="#9aa4ad", label="daily"), Line2D([0], [0], color=C.C_BLUE, label="3-day total"),
                   Line2D([0], [0], marker="v", color=C.C_ORANGE, ls="", label=f"top {N_RAIN_PEAKS} 3-day peaks"),
                   Patch(color=C.C_YELLOW, alpha=0.5, label="NDEM map dates")], frameon=False, ncol=4, loc="upper left")
ax.set_ylabel("mm")
ax.set_title("3.1  Candidate flood events 2017–2025: CHIRPS extremes and NDEM map dates", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=C.C_GRID, lw=0.6)
fig.tight_layout()
fig.savefig(f"{C.MAP_DIR}/step3_chirps_candidates.png", bbox_inches="tight")
plt.close(fig)

# event table chart
tt = table[table["status"].isin(["kept", "rejected", "unverified"]) & table["flood_km2"].notna()].copy()
tt["label"] = tt["date"] + " " + tt["pass"].str[:4] + " o" + tt["orbit"].astype(str)
colors = {"kept": C.C_GREEN, "rejected": C.C_ORANGE, "unverified": C.C_GREY}
fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.33 * len(tt) + 1.5)), dpi=110, sharey=True)
yy = np.arange(len(tt))[::-1]
axes[0].barh(yy, tt["flood_km2"], color=[colors[s] for s in tt["status"]])
axes[0].set_yticks(yy, tt["label"], fontsize=8)
axes[0].axvline(MIN_FLOOD_KM2, color="k", ls=":", lw=1)
axes[0].set_xlabel("Otsu flood area (km²)")
axes[0].set_title("Flood area per acquisition", loc="left", fontweight="bold")
g40 = tt["GTI_px_40m"].fillna(0) if "GTI_px_40m" in tt else np.zeros(len(tt))
axes[1].barh(yy, 100 * g40, color=[colors[s] for s in tt["status"]])
if "random_40m" in tt:
    axes[1].plot(100 * tt["random_40m"].fillna(0), yy, "k|", ms=10, label="random baseline")
axes[1].axvline(100 * GTI_MIN, color="k", ls=":", lw=1)
axes[1].set_xlabel("GTI vs NDEM envelope, 40 m tolerance (%)")
axes[1].set_title("Agreement with official NDEM maps", loc="left", fontweight="bold")
axes[1].legend(handles=[Patch(color=v, label=k) for k, v in colors.items()]
               + [Line2D([0], [0], marker="|", color="k", ls="", ms=10, label="random baseline")],
               frameon=False, loc="lower right")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="x", color=C.C_GRID, lw=0.6)
fig.suptitle("Step 3.3 — every Sentinel-1 acquisition tested (green = kept)", fontsize=14, fontweight="bold",
             x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{C.MAP_DIR}/step3_event_table.png", bbox_inches="tight")
plt.close(fig)

# kept events over imagery
if kept_ev:
    base = C.basemap(C.BBOX, 12, 1200)
    ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
    n = len(kept_ev)
    ncol = 2 if n > 2 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 7.0 * nrow), dpi=110, squeeze=False)
    for a in axes.ravel()[n:]:
        a.axis("off")
    cm = ListedColormap([(0, 0, 0, 0), C.C_ORANGE, C.C_BLUE])
    for ax, e in zip(axes.ravel(), kept_ev):
        s = e["selected"]
        cls = cls_cache[(e["event"], s["date"], s["orbit"])]
        ax.imshow(base, extent=ext)
        ax.imshow(np.where(cls == 255, 0, cls), cmap=cm, vmin=0, vmax=2, extent=ext, interpolation="nearest")
        env, _ = C.ndem_mask(ndem, e["ndem_dates"], (H20, W20), T20)
        ax.contour(np.flipud(env), levels=[0.5], colors=["white"], linewidths=0.35, extent=ext, origin="lower")
        ax.set_title(f"{e['event']}: S1 {s['date']} {s['pass'][:4]}\nflood {s['flood_km2']:.1f} km², "
                     f"GTI(40 m) {100 * s['GTI_px_40m']:.1f}%, lift {s['lift_40m']:.1f}", loc="left",
                     fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=6)
    fig.legend(handles=[Patch(color=C.C_ORANGE, label="Otsu flood"), Patch(color=C.C_BLUE, label="reference water"),
                        Line2D([0], [0], color="#9aa4ad", label="NDEM event envelope (white outline)")],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Step 3 — kept flood events (Imagery: Esri, Maxar, Earthstar Geographics)", fontsize=14,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(f"{C.MAP_DIR}/step3_kept_events.png", bbox_inches="tight")
    plt.close(fig)

print("\n[6/7] Metrics")
res["rain_peaks"] = [{"end_date": f"{d.date()}", "mm_3day": round(v, 1)} for d, v in peaks]
res["wettest_months"] = {f"{k:%Y-%m}": round(float(v), 1) for k, v in monthly.head(8).items()}
res["urban_double_bounce_3_4"] = "not attempted in this step (optional, flagged for Step 7 PU-learning)"
res["runtime_s"] = round(time.time() - t_start, 1)
C.save_json(res, f"{C.METRICS_DIR}/step3_flood_inventory_events.json")
print(f"[7/7] done in {res['runtime_s']} s")
