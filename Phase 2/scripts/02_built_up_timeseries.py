"""
Phase 2 Step 2 - Multi-year land-cover / imperviousness record.

2.1 Source comparison on the AOI (all through Google Earth Engine):
      DW    Dynamic World V1, annual mode of the per-image label, class 6 = built
      ESRI  Esri/Impact Observatory 10 m annual LULC (sat-io community catalogue),
            class 7 = built area
      WC    ESA WorldCover v100 (2020) / v200 (2021), class 50 = built-up, fetched
            for the FULL AOI (the Phase 1 tile stopped at 10.008 N)
      GHSL  JRC GHS-BUILT-S P2023A, built surface m2 per 100 m cell, epochs
            2015 / 2020 / 2025 / 2030 (2025, 2030 are projections) -> kept for the
            Step 9 post-development scenario, not for the yearly series
    Criteria (decided before looking at the result):
      a. agreement with WorldCover 2021 (highest published accuracy) at 30 m:
         Pearson r of the built fraction, F1 of the >= 0.5 built cell
      b. temporal stability: raw year-to-year LOSS of built 10 m pixels as a
         share of the previous year's built area (built-up almost never disappears)
      c. yearly coverage 2017 - latest full year
    Score = mean rank over a, b; ties broken by c.

2.2 Yearly built-up fraction (0-1) on the Phase 1 30 m grid. Every source is
    sampled (nearest) onto a 1/3" grid (~10 m) that nests exactly 3 x 3 inside
    the FABDEM 1" grid, so the 30 m fraction is an exact area-weighted mean of
    nine sub-pixels (same idea as the Step 8 label aggregation).

2.3 Cleaning of the chosen source, per 10 m pixel over the years:
      - years with no observation (DW: cloud all year) take the nearest valid year;
      - single-year flickers are removed with a temporal majority filter of
        width 3 (built in t only, or missing in t only, are both reverted);
        the first and last year are only compared with their one neighbour, so
        they are left unchanged and reported as less certain.
    Residual loss after cleaning is reported, not forced to zero.

2.4 Validation: first-year / last-year built-up over Esri World Imagery at
    four known growth sites (Kakkanad-Infopark, Cochin airport, Kalamassery,
    Aluva fringe), and the built-up area per year with the growth rate.

Outputs
    data/raw/landcover/*.tif                                cached 10 m binaries (gitignored)
    data/processed/landcover/builtup_fraction_30m_<SRC>.tif  raw yearly fractions
    data/processed/landcover/builtup_fraction_30m_clean.tif  cleaned, chosen source (bands = years)
    data/processed/landcover/ghsl_builtup_fraction_30m.tif   GHSL 2015/2020/2025/2030
    outputs/metrics/step2_builtup_timeseries.json
    outputs/maps/step2_source_comparison.png, step2_builtup_first_last.png, step2_growth_sites.png
"""
import os
import time

import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import common as C

RAW = C.p2("data", "raw", "landcover")
OUT = C.p2("data", "processed", "landcover")
for d in (RAW, OUT):
    os.makedirs(d, exist_ok=True)

YEARS = list(range(2017, 2026))       # 2025 = latest full year
WC_YEARS = [2020, 2021]
GHSL_EPOCHS = [2015, 2020, 2025, 2030]
REF_YEAR = 2021                       # WorldCover v200 comparison year
NODATA = 255
DW_BUILT, ESRI_BUILT, WC_BUILT = 6, 7, 50
TILES = (2, 2)
SITES = [  # (name, lat, lon, half-size deg, why it is a test)
    ("Kakkanad - Infopark", 10.012, 76.362, 0.022, "IT park + housing growth since 2017"),
    ("Cochin Airport", 10.152, 76.392, 0.022, "airport expansion and surroundings"),
    ("Kalamassery", 10.052, 76.318, 0.022, "Aluva-Kalamassery corridor, KINFRA / HMT"),
    ("Aluva fringe", 10.112, 76.368, 0.022, "peri-urban growth east of Aluva"),
]

C.banner("PHASE 2 STEP 2: MULTI-YEAR BUILT-UP / IMPERVIOUS RECORD")
t_start = time.time()
res = {"step": 2, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "years": YEARS}

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
with rasterio.open(C.P1_FABDEM) as src:
    T30 = src.transform
    H30, W30 = src.shape
with rasterio.open(C.P1_DOMAIN) as src:
    land30 = src.read(1).astype(bool)
T10 = Affine(T30.a / 3, 0, T30.c, 0, T30.e / 3, T30.f)
H10, W10 = 3 * H30, 3 * W30
_, _, cell30 = C.row_cell_sizes_m(T30, H30)          # m2 per 30 m row
land_km2 = float((land30.sum(axis=1) * cell30).sum() / 1e6)
print(f"    30 m grid {H30} x {W30}, 10 m sub-grid {H10} x {W10} (exact 3 x 3 nesting); land {land_km2:.1f} km2")
res["grid"] = {"h30": H30, "w30": W30, "h10": H10, "w10": W10, "land_km2": round(land_km2, 2)}


def to30(b10):
    """Exact 30 m fraction from a 10 m binary (NODATA-aware: fraction of valid sub-pixels)."""
    v = (b10 != NODATA).reshape(H30, 3, W30, 3).sum(axis=(1, 3))
    s = (b10 == 1).reshape(H30, 3, W30, 3).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v > 0, s / np.maximum(v, 1), np.nan).astype(np.float32)


def built_km2(f30):
    return float((np.where(land30, np.nan_to_num(f30), 0).sum(axis=1) * cell30).sum() / 1e6)


# ---------------------------------------------------------------------------
# 2.1 / 2.2 Download every source onto the 10 m sub-grid (cached)
# ---------------------------------------------------------------------------
print("\n[1/6] Fetching sources from Earth Engine (cached)")
ee = C.ee_init()
region = ee.Geometry.Rectangle(C.BBOX)


def ee_binary(source, year):
    if source == "DW":
        col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("label"))
        return col.mode().eq(DW_BUILT), col.size()
    if source == "ESRI":
        col = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
               .filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
        return col.mosaic().select(0).eq(ESRI_BUILT), col.size()
    if source == "WC":
        asset = {2020: "ESA/WorldCover/v100", 2021: "ESA/WorldCover/v200"}[year]
        col = ee.ImageCollection(asset).filterBounds(region)
        return col.mosaic().select("Map").eq(WC_BUILT), col.size()
    raise ValueError(source)


def fetch(source, year):
    path = f"{RAW}/{source.lower()}_built_10m_{year}.tif"
    if not os.path.exists(path):
        img, n = ee_binary(source, year)
        img = img.unmask(NODATA).toByte().rename("built")
        t = time.time()
        C.ee_download_grid(img, T10, W10, H10, path, tiles=TILES)
        print(f"    {source} {year}: {n.getInfo()} source images -> {os.path.basename(path)} ({time.time() - t:.0f} s)")
    with rasterio.open(path) as src:
        return src.read(1)


series10 = {"DW": {}, "ESRI": {}}
frac30 = {"DW": {}, "ESRI": {}, "WC": {}}
for src_name in ("DW", "ESRI"):
    for y in YEARS:
        b = fetch(src_name, y)
        series10[src_name][y] = b
        frac30[src_name][y] = to30(b)
for y in WC_YEARS:
    frac30["WC"][y] = to30(fetch("WC", y))

ghsl_path = f"{OUT}/ghsl_builtup_fraction_30m.tif"
if not os.path.exists(ghsl_path):
    g = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
    img = ee.Image.cat([g.filter(ee.Filter.eq("system:index", str(e))).first().select("built_surface")
                        .divide(10000).clamp(0, 1).rename(f"ghsl_{e}") for e in GHSL_EPOCHS]).toFloat()
    C.ee_download_grid(img, T30, W30, H30, ghsl_path)
with rasterio.open(ghsl_path) as src:
    ghsl = {e: src.read(i + 1) for i, e in enumerate(GHSL_EPOCHS)}
print("    all sources on disk")

# ---------------------------------------------------------------------------
# 2.1 Source comparison
# ---------------------------------------------------------------------------
print("\n[2/6] 2.1 Source comparison")


def corr_f1(a, b):
    m = land30 & np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    pa, pb = a[m] >= 0.5, b[m] >= 0.5
    tp = float((pa & pb).sum())
    f1 = 2 * tp / max(pa.sum() + pb.sum(), 1)
    mae = float(np.abs(a[m] - b[m]).mean())
    return round(r, 4), round(f1, 4), round(mae, 4)


def loss_share(series):
    """Raw year-to-year loss of built 10 m pixels / previous year's built pixels."""
    out = {}
    for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
        a, b = series[y0], series[y1]
        v = (a != NODATA) & (b != NODATA)
        prev = (a == 1) & v
        out[f"{y0}-{y1}"] = round(float(((b == 0) & prev).sum() / max(prev.sum(), 1)), 4)
    return out


comp = {}
for s in ("DW", "ESRI"):
    r, f1, mae = corr_f1(frac30[s][REF_YEAR], frac30["WC"][REF_YEAR])
    losses = loss_share(series10[s])
    nod = {y: round(float((series10[s][y] == NODATA).mean()), 4) for y in YEARS}
    comp[s] = {"vs_worldcover_2021": {"pearson_r": r, "f1_at_0.5": f1, "mae": mae},
               "built_km2_by_year": {y: round(built_km2(frac30[s][y]), 2) for y in YEARS},
               "raw_loss_share_by_pair": losses, "mean_raw_loss_share": round(float(np.mean(list(losses.values()))), 4),
               "nodata_share_by_year": nod, "years": [YEARS[0], YEARS[-1]]}
    print(f"    {s:5s} vs WC2021: r {r:.3f}, F1 {f1:.3f}, MAE {mae:.3f} | mean yearly built loss "
          f"{100 * comp[s]['mean_raw_loss_share']:.1f}% | built {comp[s]['built_km2_by_year'][YEARS[0]]:.1f} -> "
          f"{comp[s]['built_km2_by_year'][YEARS[-1]]:.1f} km2")
comp["WC"] = {"built_km2_by_year": {y: round(built_km2(frac30["WC"][y]), 2) for y in WC_YEARS}}
comp["GHSL"] = {"built_km2_by_epoch": {e: round(built_km2(ghsl[e]), 2) for e in GHSL_EPOCHS},
                "vs_worldcover_2021_using_2020_epoch": dict(zip(("pearson_r", "f1_at_0.5", "mae"),
                                                                corr_f1(ghsl[2020], frac30["WC"][2021])))}
print(f"    WC    built 2020 {comp['WC']['built_km2_by_year'][2020]:.1f}, 2021 {comp['WC']['built_km2_by_year'][2021]:.1f} km2")
print(f"    GHSL  built {comp['GHSL']['built_km2_by_epoch']} km2; 2020 vs WC2021 {comp['GHSL']['vs_worldcover_2021_using_2020_epoch']}")

# ranking: a = agreement (r and F1, higher better), b = stability (loss, lower better)
cands = ["DW", "ESRI"]
rank = {s: 0.0 for s in cands}
for key, better_high in ((lambda s: comp[s]["vs_worldcover_2021"]["pearson_r"], True),
                         (lambda s: comp[s]["vs_worldcover_2021"]["f1_at_0.5"], True),
                         (lambda s: comp[s]["mean_raw_loss_share"], False)):
    order = sorted(cands, key=key, reverse=better_high)
    for i, s in enumerate(order):
        rank[s] += i / 3
CHOSEN = min(cands, key=lambda s: rank[s])
res["source_comparison"] = comp
res["choice"] = {"chosen": CHOSEN, "mean_rank": {s: round(v, 3) for s, v in rank.items()},
                 "rule": "mean rank over (r vs WC2021, F1 vs WC2021, mean raw yearly loss); both cover 2017-2025",
                 "ghsl_role": "post-development scenario source (Step 9): 2025/2030 projections, 100 m"}
print(f"    -> chosen yearly source: {CHOSEN} (mean rank {rank})")

for s in ("DW", "ESRI"):
    p = f"{OUT}/builtup_fraction_30m_{s}.tif"
    prof = {"driver": "GTiff", "height": H30, "width": W30, "count": len(YEARS), "dtype": "float32",
            "crs": "EPSG:4326", "transform": T30, "nodata": np.nan, "compress": "deflate", "predictor": 3}
    with rasterio.open(p, "w", **prof) as dst:
        for i, y in enumerate(YEARS, 1):
            dst.write(np.where(land30, frac30[s][y], np.nan).astype(np.float32), i)
            dst.set_band_description(i, f"built_{y}")

# ---------------------------------------------------------------------------
# 2.3 Cleaning
# ---------------------------------------------------------------------------
print(f"\n[3/6] 2.3 Cleaning the {CHOSEN} series (gap fill + width-3 temporal majority)")
S = np.stack([series10[CHOSEN][y] for y in YEARS]).astype(np.uint8)      # (n, H10, W10)
n = len(YEARS)
gap = S == NODATA
filled = S.copy()
for i in range(n):                         # nearest valid year: look back, then forward, alternating
    for k in range(1, n):
        for j in (i - k, i + k):
            if 0 <= j < n:
                m = (filled[i] == NODATA) & (S[j] != NODATA)
                filled[i][m] = S[j][m]
still = filled == NODATA
filled[still] = 0
clean = filled.copy()
for i in range(1, n - 1):                  # width-3 majority on 0/1
    clean[i] = ((filled[i - 1].astype(np.int8) + filled[i] + filled[i + 1]) >= 2).astype(np.uint8)
flick_add = [(int(((filled[i] == 1) & (clean[i] == 0)).sum())) for i in range(n)]
flick_fill = [(int(((filled[i] == 0) & (clean[i] == 1)).sum())) for i in range(n)]
px10_km2 = float(np.mean(cell30)) / 9 / 1e6
loss_after = {}
for i in range(1, n):
    prev = clean[i - 1] == 1
    loss_after[f"{YEARS[i - 1]}-{YEARS[i]}"] = round(float(((clean[i] == 0) & prev).sum() / max(prev.sum(), 1)), 4)
clean30 = {y: to30(clean[i]) for i, y in enumerate(YEARS)}
clean_km2 = {y: round(built_km2(clean30[y]), 2) for y in YEARS}
growth = {f"{YEARS[i - 1]}-{YEARS[i]}": round(100 * (clean_km2[YEARS[i]] / clean_km2[YEARS[i - 1]] - 1), 2)
          for i in range(1, n)}
span = YEARS[-1] - YEARS[0]
cagr = 100 * ((clean_km2[YEARS[-1]] / clean_km2[YEARS[0]]) ** (1 / span) - 1)
new_built = (clean[-1] == 1) & (clean[0] == 0)
res["cleaning"] = {
    "source": CHOSEN,
    "gap_px_share_by_year": {y: round(float(gap[i].mean()), 4) for i, y in enumerate(YEARS)},
    "never_observed_px_set_to_not_built": int(still[0].sum()),
    "single_year_built_flickers_removed_km2": {y: round(v * px10_km2, 2) for y, v in zip(YEARS, flick_add)},
    "single_year_gaps_filled_km2": {y: round(v * px10_km2, 2) for y, v in zip(YEARS, flick_fill)},
    "loss_share_after_cleaning": loss_after,
    "mean_loss_share_raw": comp[CHOSEN]["mean_raw_loss_share"],
    "mean_loss_share_after_cleaning": round(float(np.mean(list(loss_after.values()))), 4),
    "endpoint_note": f"{YEARS[0]} and {YEARS[-1]} have one neighbour only and are not filtered",
}
res["growth"] = {"built_km2_by_year": clean_km2, "built_share_of_land_by_year":
                 {y: round(v / land_km2, 4) for y, v in clean_km2.items()},
                 "yearly_growth_pct": growth, "cagr_pct": round(cagr, 2),
                 "net_change_km2": round(clean_km2[YEARS[-1]] - clean_km2[YEARS[0]], 2),
                 "new_built_10m_px_km2": round(float(new_built.sum()) * px10_km2, 2)}
print(f"    gaps (no observation) mean {100 * np.mean(list(res['cleaning']['gap_px_share_by_year'].values())):.2f}% of px; "
      f"flickers removed {sum(flick_add) * px10_km2:.1f} km2-years, gaps filled {sum(flick_fill) * px10_km2:.1f} km2-years")
print(f"    mean yearly built loss: raw {100 * res['cleaning']['mean_loss_share_raw']:.2f}% -> "
      f"cleaned {100 * res['cleaning']['mean_loss_share_after_cleaning']:.2f}%")
print("    built km2: " + ", ".join(f"{y} {v:.1f}" for y, v in clean_km2.items()))
print(f"    growth {YEARS[0]}-{YEARS[-1]}: {res['growth']['net_change_km2']:+.1f} km2, CAGR {cagr:.2f}%/yr")

clean_path = f"{OUT}/builtup_fraction_30m_clean.tif"
prof = {"driver": "GTiff", "height": H30, "width": W30, "count": n, "dtype": "float32", "crs": "EPSG:4326",
        "transform": T30, "nodata": np.nan, "compress": "deflate", "predictor": 3}
with rasterio.open(clean_path, "w", **prof) as dst:
    for i, y in enumerate(YEARS, 1):
        dst.write(np.where(land30, clean30[y], np.nan).astype(np.float32), i)
        dst.set_band_description(i, f"built_{y}")
    dst.update_tags(source=CHOSEN, cleaning="nearest-year gap fill + width-3 temporal majority (10 m)")
print(f"    [OK] {clean_path}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[4/6] Figures")
ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])

# source comparison: area per year + 2021 maps
fig = plt.figure(figsize=(17, 10), dpi=110)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25])
ax = fig.add_subplot(gs[0, :2])
cols = {"DW": C.C_BLUE, "ESRI": C.C_ORANGE}
for s in ("DW", "ESRI"):
    ax.plot(YEARS, [comp[s]["built_km2_by_year"][y] for y in YEARS], "-o", color=cols[s], lw=2, ms=4,
            label=f"{s} raw")
ax.plot(YEARS, [clean_km2[y] for y in YEARS], "--", color=cols[CHOSEN], lw=2, label=f"{CHOSEN} cleaned (chosen)")
ax.plot(WC_YEARS, [comp["WC"]["built_km2_by_year"][y] for y in WC_YEARS], "s", color=C.C_GREEN, ms=8,
        label="WorldCover")
ax.plot(GHSL_EPOCHS, [comp["GHSL"]["built_km2_by_epoch"][e] for e in GHSL_EPOCHS], "^:", color=C.C_GREY, ms=7,
        label="GHSL (2025, 2030 projected)")
ax.set_ylabel("Built-up area on land (km²)")
ax.set_title("2.1  Built-up area per year by source", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=9, ncol=2)
ax.grid(color=C.C_GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 2:])
x = np.arange(len(YEARS) - 1)
for k, s in enumerate(("DW", "ESRI")):
    ax.bar(x + (k - 0.5) * 0.38, [100 * v for v in comp[s]["raw_loss_share_by_pair"].values()], 0.38,
           color=cols[s], label=f"{s} raw")
ax.plot(x, [100 * v for v in loss_after.values()], "k_", ms=14, mew=2, label=f"{CHOSEN} after cleaning")
ax.set_xticks(x, [k.replace("20", "'", 1).replace("-20", "-'") for k in comp["DW"]["raw_loss_share_by_pair"]],
              fontsize=8)
ax.set_ylabel("Built pixels lost next year (%)")
ax.set_title("2.1b  Temporal noise: built-up that 'disappears'", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color=C.C_GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)

panels = [("DW", frac30["DW"][REF_YEAR]), ("ESRI", frac30["ESRI"][REF_YEAR]), ("WorldCover", frac30["WC"][REF_YEAR]),
          ("GHSL 2020", ghsl[2020])]
for k, (name, f) in enumerate(panels):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(np.where(land30, f, np.nan), cmap="magma_r", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    sub = ""
    if name in ("DW", "ESRI"):
        v = comp[name]["vs_worldcover_2021"]
        sub = f"\nvs WC: r {v['pearson_r']:.2f}, F1 {v['f1_at_0.5']:.2f}"
    ax.set_title(f"{name} {'' if 'GHSL' in name else REF_YEAR} built fraction{sub}", loc="left", fontsize=10,
                 fontweight="bold")
    ax.tick_params(labelsize=7)
fig.suptitle("Step 2.1 — built-up source comparison on the AOI (30 m fraction)", fontsize=14, fontweight="bold",
             x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{C.MAP_DIR}/step2_source_comparison.png", bbox_inches="tight")
plt.close(fig)

# first vs last year, new built-up
fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=110)
for ax, y in zip(axes[:2], (YEARS[0], YEARS[-1])):
    im = ax.imshow(np.where(land30, clean30[y], np.nan), cmap="magma_r", vmin=0, vmax=1, extent=ext,
                   interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    ax.set_title(f"{CHOSEN} cleaned built fraction {y}  ({clean_km2[y]:.1f} km²)", loc="left", fontweight="bold")
fig.colorbar(im, ax=axes[:2], fraction=0.025, label="built-up fraction of 30 m cell")
dlt = np.where(land30, clean30[YEARS[-1]] - clean30[YEARS[0]], np.nan)
im = axes[2].imshow(dlt, cmap="RdBu_r", vmin=-0.6, vmax=0.6, extent=ext, interpolation="nearest")
axes[2].set_facecolor("#d6dde3")
axes[2].set_title(f"Change {YEARS[0]} → {YEARS[-1]}  (net {res['growth']['net_change_km2']:+.1f} km², "
                  f"{res['growth']['cagr_pct']:.2f} %/yr)", loc="left", fontweight="bold")
fig.colorbar(im, ax=axes[2], fraction=0.04, label="Δ built-up fraction")
for ax in axes:
    for name, lat, lon, hs, _ in SITES:
        ax.add_patch(plt.Rectangle((lon - hs, lat - hs), 2 * hs, 2 * hs, fill=False, ec=C.C_GREEN, lw=1.1))
    ax.tick_params(labelsize=7)
fig.suptitle("Step 2.2–2.3 — yearly built-up record (green boxes = validation sites)", fontsize=14,
             fontweight="bold", x=0.01, ha="left")
fig.savefig(f"{C.MAP_DIR}/step2_builtup_first_last.png", bbox_inches="tight")
plt.close(fig)

# 2.4 validation sites over Esri imagery (10 m cleaned pixels)
print("\n[5/6] 2.4 Validation sites")
fig, axes = plt.subplots(2, 4, figsize=(20, 10.8), dpi=110)
site_stats = []
ov = ListedColormap([(0, 0, 0, 0), (42 / 255, 120 / 255, 214 / 255, 0.55), (235 / 255, 104 / 255, 52 / 255, 0.9)])
for k, (name, lat, lon, hs, why) in enumerate(SITES):
    b = (lon - hs, lat - hs, lon + hs, lat + hs)
    img = C.basemap(b, 16, 900)
    c0, r0 = ~T10 * (b[0], b[3])
    c1, r1 = ~T10 * (b[2], b[1])
    c0, r0, c1, r1 = int(c0), int(r0), int(np.ceil(c1)), int(np.ceil(r1))
    e0, e1 = clean[0][r0:r1, c0:c1], clean[-1][r0:r1, c0:c1]
    cls = np.where(e0 == 1, 1, np.where(e1 == 1, 2, 0))
    sub_ext = (T10.c + c0 * T10.a, T10.c + c1 * T10.a, T10.f + r1 * T10.e, T10.f + r0 * T10.e)
    ax = axes[0, k]
    ax.imshow(img, extent=(b[0], b[2], b[1], b[3]))
    ax.set_title(f"{name}\nEsri World Imagery (recent)", loc="left", fontsize=10, fontweight="bold")
    ax = axes[1, k]
    ax.imshow(img, extent=(b[0], b[2], b[1], b[3]))
    ax.imshow(cls, cmap=ov, vmin=0, vmax=2, extent=sub_ext, interpolation="nearest")
    s0, s1 = float(e0.mean()), float(e1.mean())
    site_stats.append({"site": name, "why": why, f"built_share_{YEARS[0]}": round(s0, 4),
                       f"built_share_{YEARS[-1]}": round(s1, 4), "change_pts": round(100 * (s1 - s0), 2)})
    ax.set_title(f"built {100 * s0:.1f}% ({YEARS[0]}) → {100 * s1:.1f}% ({YEARS[-1]})", loc="left", fontsize=10,
                 fontweight="bold")
    for a in (axes[0, k], axes[1, k]):
        a.set_xlim(b[0], b[2]); a.set_ylim(b[1], b[3]); a.tick_params(labelsize=6)
    print(f"    {name:22s} built {100 * s0:5.1f}% -> {100 * s1:5.1f}%")
fig.legend(handles=[Patch(color=C.C_BLUE, alpha=0.6, label=f"built in {YEARS[0]}"),
                    Patch(color=C.C_ORANGE, label=f"new built by {YEARS[-1]}")],
           loc="lower center", ncol=2, fontsize=11, frameon=False)
fig.suptitle(f"Step 2.4 — {CHOSEN} cleaned built-up at known growth sites (Imagery: Esri, Maxar, Earthstar Geographics)",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(f"{C.MAP_DIR}/step2_growth_sites.png", bbox_inches="tight")
plt.close(fig)
res["validation_sites"] = site_stats

# ---------------------------------------------------------------------------
# 2.5 Checkpoint decision (Option A): hybrid imperviousness
#     level  = GHSL 2020 built-surface fraction (physical roof + paving area)
#     timing = DW cleaned yearly built fraction, scaled by k = sum(GHSL2020)/sum(DW2020)
#     imp_y  = clip(GHSL2020 + k * (DW_y - DW_2020), 0, 1)
# DW / ESRI "built" is settlement extent (houses under tree canopy, 40-67 % of
# land), GHSL / WorldCover measure impervious surface (10-13 %); runoff, and so
# UICA, needs the impervious area.
# ---------------------------------------------------------------------------
print("\n[5b/6] 2.5 Hybrid imperviousness (Option A: GHSL level + DW yearly change)")
ANCHOR = 2020
g20 = np.where(land30, np.nan_to_num(ghsl[ANCHOR]), 0.0)
dw = {y: np.where(land30, np.nan_to_num(clean30[y]), 0.0) for y in YEARS}
k = float(g20.sum() / dw[ANCHOR].sum())
imp = {y: np.clip(g20 + k * (dw[y] - dw[ANCHOR]), 0, 1) for y in YEARS}
imp_km2 = {y: round(float((imp[y].sum(axis=1) * cell30).sum() / 1e6), 2) for y in YEARS}
clipped = {y: round(float(((g20 + k * (dw[y] - dw[ANCHOR])) < 0)[land30].mean()), 4) for y in YEARS}
r_wc, f1_wc, mae_wc = corr_f1(np.where(land30, imp[2021], np.nan), frac30["WC"][2021])
imp_cagr = 100 * ((imp_km2[YEARS[-1]] / imp_km2[YEARS[0]]) ** (1 / span) - 1)
d_imp = imp[YEARS[-1]] - imp[YEARS[0]]
res["choice"].update({
    "decision": "Option A (hybrid) chosen at the Step 2 checkpoint",
    "reason": "DW/ESRI built = settlement extent (40-67 % of land); GHSL/WorldCover = impervious surface (10-13 %)",
    "final_layer": "impervious_fraction_30m.tif"})
res["hybrid_impervious"] = {
    "formula": f"imp_y = clip(GHSL{ANCHOR} + k * (DW_y - DW_{ANCHOR}), 0, 1)", "k": round(k, 4),
    "impervious_km2_by_year": imp_km2,
    "impervious_share_of_land_by_year": {y: round(v / land_km2, 4) for y, v in imp_km2.items()},
    "net_change_km2": round(imp_km2[YEARS[-1]] - imp_km2[YEARS[0]], 2), "cagr_pct": round(imp_cagr, 2),
    "cells_clipped_at_zero_share": clipped,
    "vs_worldcover_2021": {"pearson_r": r_wc, "f1_at_0.5": f1_wc, "mae": mae_wc},
    "ghsl_epochs_km2_for_reference": comp["GHSL"]["built_km2_by_epoch"],
    "cells_with_increase_share": round(float((d_imp > 1e-6)[land30].mean()), 4),
    "cells_with_decrease_share": round(float((d_imp < -1e-6)[land30].mean()), 4)}
print(f"    k = {k:.3f}; impervious km2: " + ", ".join(f"{y} {v:.1f}" for y, v in imp_km2.items()))
print(f"    {YEARS[0]}-{YEARS[-1]}: {imp_km2[YEARS[-1]] - imp_km2[YEARS[0]]:+.1f} km2 ({imp_cagr:.2f} %/yr); "
      f"2021 vs WorldCover r {r_wc:.3f}, MAE {mae_wc:.3f} (DW alone r {comp['DW']['vs_worldcover_2021']['pearson_r']:.3f}, "
      f"MAE {comp['DW']['vs_worldcover_2021']['mae']:.3f})")
imp_path = f"{OUT}/impervious_fraction_30m.tif"
with rasterio.open(imp_path, "w", **prof) as dst:
    for i, y in enumerate(YEARS, 1):
        dst.write(np.where(land30, imp[y], np.nan).astype(np.float32), i)
        dst.set_band_description(i, f"impervious_{y}")
    dst.update_tags(method=res["hybrid_impervious"]["formula"], k=str(round(k, 4)))
print(f"    [OK] {imp_path}")

fig = plt.figure(figsize=(18, 11), dpi=110)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, :])
ax.plot(YEARS, [clean_km2[y] for y in YEARS], "--", color=C.C_BLUE, lw=1.8, label="DW cleaned 'built' (settlement extent)")
ax.plot(YEARS, [imp_km2[y] for y in YEARS], "-o", color=C.C_ORANGE, lw=2.4, ms=5,
        label="Hybrid impervious area (final layer)")
ax.plot(GHSL_EPOCHS, [comp["GHSL"]["built_km2_by_epoch"][e] for e in GHSL_EPOCHS], "^:", color=C.C_GREY, ms=8,
        label="GHSL built surface (2025, 2030 projected)")
ax.plot(WC_YEARS, [comp["WC"]["built_km2_by_year"][y] for y in WC_YEARS], "s", color=C.C_GREEN, ms=9,
        label="WorldCover built-up")
for y in (YEARS[0], ANCHOR, YEARS[-1]):
    ax.annotate(f"{imp_km2[y]:.0f} km²", (y, imp_km2[y]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color=C.C_ORANGE, fontweight="bold")
ax.set_ylabel("Area on land (km²)")
ax.set_yscale("log")
ax.set_title(f"2.5  Option A: impervious level from GHSL {ANCHOR}, yearly change from DW (k = {k:.3f})",
             loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.grid(color=C.C_GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
for j, y in enumerate((YEARS[0], YEARS[-1])):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(np.where(land30, imp[y], np.nan), cmap="magma_r", vmin=0, vmax=0.8, extent=ext,
                   interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    ax.set_title(f"Impervious fraction {y}  ({imp_km2[y]:.0f} km²)", loc="left", fontweight="bold")
    ax.tick_params(labelsize=7)
fig.colorbar(im, ax=fig.axes[1:3], orientation="horizontal", fraction=0.04, pad=0.08, shrink=0.6,
             label="impervious fraction of 30 m cell")
ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(np.where(land30, d_imp, np.nan), cmap="RdBu_r", vmin=-0.15, vmax=0.15, extent=ext,
               interpolation="nearest")
ax.set_facecolor("#d6dde3")
ax.set_title(f"Change {YEARS[0]} → {YEARS[-1]}  ({imp_km2[YEARS[-1]] - imp_km2[YEARS[0]]:+.0f} km²)", loc="left",
             fontweight="bold")
ax.tick_params(labelsize=7)
fig.colorbar(im, ax=ax, fraction=0.04, label="Δ impervious fraction")
fig.suptitle("Step 2.5 — final yearly imperviousness layer (input to UICA / ΔUICA)", fontsize=14,
             fontweight="bold", x=0.01, ha="left")
fig.savefig(f"{C.MAP_DIR}/step2_impervious_hybrid.png", bbox_inches="tight")
plt.close(fig)

print("\n[6/6] Metrics")
res["outputs"] = {"clean": clean_path, "ghsl": ghsl_path, "impervious_final": imp_path,
                  "raw": {s: f"{OUT}/builtup_fraction_30m_{s}.tif" for s in ("DW", "ESRI")},
                  "maps": ["step2_source_comparison.png", "step2_builtup_first_last.png", "step2_growth_sites.png", "step2_impervious_hybrid.png"]}
res["runtime_s"] = round(time.time() - t_start, 1)
C.save_json(res, f"{C.METRICS_DIR}/step2_builtup_timeseries.json")
print(f"    [OK] step2_builtup_timeseries.json ({res['runtime_s']} s)")
