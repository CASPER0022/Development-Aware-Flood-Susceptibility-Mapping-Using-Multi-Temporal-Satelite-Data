"""
Step 5 - Build the flood inventory using Otsu thresholding (Week 2,
Phase1_Implementation_Plan).

Inputs are the GEE Sentinel-1 VV scenes from scripts/08_fetch_sentinel1_sar.py
(one flood scene + one same-relative-orbit dry pre-monsoon reference per year).

5.1. dB conversion. The base paper (Mohamadiazar et al., 2024, Eq. 1) converts
     calibrated backscatter to decibels as  sigma0_dB = 10 * log10(sigma0).
     GEE's COPERNICUS/S1_GRD already delivers sigma0 in dB, so the input is
     *checked* to be in dB (not blindly log-transformed a second time). The
     base paper's SNAP chain also speckle-filters before thresholding, and a
     speckle filter must run on linear intensity, not dB -- so each scene is
     taken dB -> linear, Lee-filtered (5x5), then brought back to dB with Eq. 1.
     That is the one place Eq. 1 is actually applied here.

5.2. Otsu thresholding on the (filtered) VV dB histogram of each scene, per
     the base paper's core flood-detection step. The threshold is computed
     per image (Otsu is image-adaptive by design) over the [-30, +5] dB range
     so the handful of extreme urban corner-reflector returns (up to +34 dB)
     and noise-floor pixels don't distort the histogram. Each threshold is
     sanity-checked against the range typically reported for open water in
     C-band VV (-24 to -12 dB); anything outside is flagged loudly.

5.3. Binary flood / no-flood raster per flood date:
         flood = water(flood scene) AND NOT water(dry reference)
     i.e. permanent water (Arabian Sea, Vembanad backwaters, the normal
     Periyar channel, ponds) is removed using the SAME-GEOMETRY dry scene.
     Two standard clean-up steps follow, each counted in the metadata:
       - pixels with FABDEM slope > 5 deg are removed (low backscatter on
         steep slopes facing away from the sensor is radar shadow, not water);
       - 8-connected specks smaller than MIN_PATCH_PX pixels are removed
         (minimum mapping unit, suppresses residual speckle).
     Outputs (data/processed/flood_inventory/, EPSG:4326, S1 20 m grid):
       flood_inventory_<year>_<yyyymmdd>.tif  uint8  1 = flooded, 0 = not flooded
       water_classes_<year>_<yyyymmdd>.tif    uint8  0 = dry land, 1 = flood,
                                              2 = reference (dry-season) water
     Downstream sampling (Step 8) should draw non-flood samples from class 0
     only, never from class 2.

5.4. Visual validation. For each date: an overview figure (Otsu histogram,
     filtered VV, flood map over Esri World Imagery, NDEM overlay) plus, for
     2018, zoom panels over places that are documented to have flooded in
     Aug 2018 (Aluva, Cochin airport/Chengamanad, North Paravur /
     Puthenvelikkara, Kalady, Eloor-Kalamassery) and a Kochi city control
     panel. Official NDEM (NRSC/ISRO) inundation polygons for the same
     calendar date are drawn as outlines, and a quick POD/FAR/CSI against
     NDEM is printed -- a sanity number only; the formal cross-check is Step 6.
     An interactive Folium map (Esri imagery + flood layer + NDEM) is written
     for 2018.
"""
import os
import sys
import io
import json
import math
import time

import numpy as np
import requests
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy import ndimage
from skimage.filters import threshold_otsu
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

S1_META = "data/raw/sentinel1_gee_metadata.json"
FABDEM = "data/raw/fabdem_30m_aoi.tif"
NDEM = "data/validation/NDEM_KL_Floods_Inundation.parquet"
OUT_DIR = "data/processed/flood_inventory"
MAP_DIR = "outputs/maps"
TILE_CACHE = "cache/esri_world_imagery"
for d in (OUT_DIR, MAP_DIR, TILE_CACHE):
    os.makedirs(d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
LEE_WINDOW = 5
OTSU_RANGE_DB = (-30.0, 5.0)
PLAUSIBLE_WATER_THRESHOLD_DB = (-24.0, -12.0)
SLOPE_MAX_DEG = 5.0
MIN_PATCH_PX = 5          # ~0.2 ha at 20 m

# Colors (dataviz reference palette, categorical slots 1-2, + yellow outline)
C_REF_WATER = "#2a78d6"
C_FLOOD = "#eb6834"
C_NDEM = "#eda100"

# Documented Aug-2018 flood locations (plus one control) for the zoom panels.
# (name, lat, lon, half-size in degrees, note)
SITES_2018 = [
    ("Aluva town & Periyar banks", 10.108, 76.352, 0.03, "flooded 15-20 Aug 2018"),
    ("Cochin Airport / Chengamanad", 10.160, 76.375, 0.03, "airport shut 15-29 Aug 2018"),
    ("N. Paravur / Puthenvelikkara", 10.170, 76.235, 0.035, "among worst-hit, Periyar-Chalakudy confluence"),
    ("Kalady / Kanjoor", 10.160, 76.435, 0.03, "Periyar floodplain upstream of Aluva"),
    ("Eloor - Kalamassery", 10.070, 76.300, 0.03, "industrial belt on Periyar banks"),
    ("Kochi city (Ernakulam)", 9.980, 76.285, 0.035, "control: localised flooding only"),
]

print("=" * 80)
print("STEP 5: FLOOD INVENTORY VIA OTSU THRESHOLDING (Sentinel-1 VV, GEE, 20 m)")
print("=" * 80)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def db_to_linear(db):
    return np.power(10.0, db / 10.0)


def linear_to_db(lin):
    """Base paper Eq. 1: sigma0_dB = 10 * log10(sigma0)."""
    return 10.0 * np.log10(np.maximum(lin, 1e-10))


def ensure_db(arr, name):
    """GEE S1_GRD is already dB; only apply Eq. 1 if the data is clearly linear."""
    med = float(np.nanmedian(arr))
    is_linear = np.nanmin(arr) >= 0 and med < 1.0
    if is_linear:
        print(f"    {name}: values look LINEAR (min>=0, median={med:.4f}) -> applying Eq. 1")
        return linear_to_db(arr), "converted_from_linear_with_eq1"
    print(f"    {name}: values already in dB (median={med:.2f} dB) -> no second log transform")
    return arr, "already_db_from_gee"


def lee_filter(intensity, size):
    """Classic Lee speckle filter on LINEAR intensity."""
    mean = ndimage.uniform_filter(intensity, size)
    sq_mean = ndimage.uniform_filter(intensity ** 2, size)
    var = np.maximum(sq_mean - mean ** 2, 0)
    noise_var = np.mean(var)
    weight = var / (var + noise_var)
    return mean + weight * (intensity - mean)


def load_s1(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        profile = src.profile
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    arr[~valid] = np.nan
    return arr, valid, profile


def preprocess(arr_db_raw, valid, name):
    arr_db, db_status = ensure_db(arr_db_raw, name)
    lin = db_to_linear(np.where(valid, arr_db, np.nanmedian(arr_db)))
    filtered_db = linear_to_db(lee_filter(lin, LEE_WINDOW))
    filtered_db[~valid] = np.nan
    return filtered_db, db_status


def otsu_threshold(arr_db, valid, name):
    vals = arr_db[valid]
    vals = vals[(vals >= OTSU_RANGE_DB[0]) & (vals <= OTSU_RANGE_DB[1])]
    t = float(threshold_otsu(vals, nbins=512))
    lo, hi = PLAUSIBLE_WATER_THRESHOLD_DB
    plausible = lo <= t <= hi
    flag = "OK" if plausible else "!! OUTSIDE TYPICAL WATER RANGE -- inspect histogram"
    print(f"    {name}: Otsu threshold = {t:.2f} dB  [{flag}]")
    return t, plausible


def pixel_area_km2(transform, height):
    dy = abs(transform.e)
    dx = abs(transform.a)
    lats = transform.f - (np.arange(height) + 0.5) * dy
    return (dx * 111.320 * np.cos(np.radians(lats))) * (dy * 110.574)  # per-row km^2


def area_km2(mask, row_area):
    return float((mask.sum(axis=1) * row_area).sum())


def slope_on_grid(profile):
    """FABDEM (30 m) resampled to the S1 grid, slope in degrees."""
    dem = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(FABDEM) as src:
        reproject(rasterio.band(src, 1), dem, dst_transform=profile["transform"],
                  dst_crs=profile["crs"], resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    t = profile["transform"]
    mid_lat = BBOX[1] + (BBOX[3] - BBOX[1]) / 2
    dx_m = abs(t.a) * 111320 * math.cos(math.radians(mid_lat))
    dy_m = abs(t.e) * 110574
    gy, gx = np.gradient(dem.astype("float64"), dy_m, dx_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))), dem


def remove_small_patches(mask, min_px):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask, 0
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab], int((~keep[1:]).sum())


def write_tif(path, arr, profile, nodata=255):
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("uint8"), 1)


def confusion(pred, truth, domain):
    hits = int((pred & truth & domain).sum())
    misses = int((~pred & truth & domain).sum())
    fa = int((pred & ~truth & domain).sum())
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = fa / (hits + fa) if hits + fa else float("nan")
    csi = hits / (hits + misses + fa) if hits + misses + fa else float("nan")
    return {"hits": hits, "misses": misses, "false_alarms": fa,
            "POD": round(pod, 4), "FAR": round(far, 4), "CSI": round(csi, 4)}


# --- Esri World Imagery basemap (XYZ tiles, cached, warped to EPSG:4326) ----
MERC = 20037508.342789244


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _get_tile(z, x, y):
    fp = os.path.join(TILE_CACHE, f"{z}_{x}_{y}.jpg")
    if not os.path.exists(fp):
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": "BTP-flood-research"})
                r.raise_for_status()
                with open(fp, "wb") as f:
                    f.write(r.content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    return np.asarray(Image.open(fp).convert("RGB"))


def basemap(bounds, z, out_w):
    """Return an RGB array (EPSG:4326, extent=bounds) of Esri World Imagery."""
    lon0, lat0, lon1, lat1 = bounds
    x0, y0 = _tile_xy(lon0, lat1, z)
    x1, y1 = _tile_xy(lon1, lat0, z)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), dtype=np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                   (tx - tx0) * 256:(tx - tx0 + 1) * 256] = _get_tile(z, tx, ty)
    tile_m = 2 * MERC / 2 ** z
    src_t = rasterio.transform.from_origin(-MERC + tx0 * tile_m, MERC - ty0 * tile_m,
                                           tile_m / 256, tile_m / 256)
    out_h = int(round(out_w * (lat1 - lat0) / (lon1 - lon0)))
    dst_t = rasterio.transform.from_bounds(lon0, lat0, lon1, lat1, out_w, out_h)
    out = np.zeros((3, out_h, out_w), dtype=np.uint8)
    for b in range(3):
        reproject(mosaic[:, :, b], out[b], src_transform=src_t, src_crs="EPSG:3857",
                  dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
    return np.moveaxis(out, 0, -1)


# ---------------------------------------------------------------------------
# Load inputs shared by all events
# ---------------------------------------------------------------------------
with open(S1_META, encoding="utf-8") as f:
    s1_meta = json.load(f)

ndem_all = gpd.read_parquet(NDEM)
if ndem_all.crs is None or ndem_all.crs.to_epsg() != 4326:
    ndem_all = ndem_all.set_crs("EPSG:4326", allow_override=True)
ndem_all = ndem_all[ndem_all.intersects(box(*BBOX))]
ndem_all["date"] = ndem_all["from_time"].astype(str).str[:10]

metadata = {
    "task": "Step 5 - Flood inventory via Otsu thresholding",
    "base_paper": "Mohamadiazar, Ebrahimian & Hosseiny (2024), J. Hydrology 639:131508",
    "parameters": {
        "eq1": "sigma0_dB = 10*log10(sigma0)",
        "speckle_filter": f"Lee {LEE_WINDOW}x{LEE_WINDOW} on linear intensity",
        "otsu_histogram_range_db": OTSU_RANGE_DB,
        "otsu_bins": 512,
        "plausible_water_threshold_range_db": PLAUSIBLE_WATER_THRESHOLD_DB,
        "flood_rule": "water(flood scene, own Otsu) AND NOT water(dry ref, own Otsu)",
        "slope_mask_deg": SLOPE_MAX_DEG,
        "min_patch_px_8conn": MIN_PATCH_PX,
    },
    "raster_encoding": {
        "flood_inventory": {"1": "flooded", "0": "not flooded", "255": "nodata"},
        "water_classes": {"0": "dry land", "1": "flood (new water)",
                          "2": "reference water (water in dry-season scene)", "255": "nodata"},
    },
    "events": {},
}

slope = dem = None
figs = []

for event_name, ev in s1_meta["events"].items():
    if ev.get("status") != "OK" or ev.get("dry_reference_scene") is None:
        print(f"\n[SKIP] {event_name}: missing flood or dry scene")
        continue
    year = event_name.split("_")[0]
    fdate = ev["flood_scene"]["date"]
    ddate = ev["dry_reference_scene"]["date"]
    tag = f"{year}_{fdate.replace('-', '')}"
    print(f"\n[+] {event_name}: flood {fdate}  vs  dry ref {ddate}")

    flood_raw, fvalid, profile = load_s1(ev["flood_scene"]["file"].replace("\\", "/"))
    dry_raw, dvalid, dprofile = load_s1(ev["dry_reference_scene"]["file"].replace("\\", "/"))
    assert profile["transform"] == dprofile["transform"] and flood_raw.shape == dry_raw.shape, \
        "flood/dry grids differ"
    valid = fvalid & dvalid

    # 5.1 dB check + speckle filter + Eq. 1
    print("  5.1 dB conversion / speckle filtering")
    flood_db, f_status = preprocess(flood_raw, valid, "flood")
    dry_db, d_status = preprocess(dry_raw, valid, "dry")

    # 5.2 Otsu
    print("  5.2 Otsu thresholding on VV")
    t_flood, ok_f = otsu_threshold(flood_db, valid, "flood")
    t_dry, ok_d = otsu_threshold(dry_db, valid, "dry")
    water_flood = valid & (flood_db < t_flood)
    water_dry = valid & (dry_db < t_dry)

    # 5.3 binary flood map
    print("  5.3 binary flood / no-flood raster")
    if slope is None:
        slope, dem = slope_on_grid(profile)
    row_area = pixel_area_km2(profile["transform"], profile["height"])
    raw_new = water_flood & ~water_dry
    steep = raw_new & (slope > SLOPE_MAX_DEG)
    after_slope = raw_new & ~steep
    flood, n_patches_removed = remove_small_patches(after_slope, MIN_PATCH_PX)

    inv = np.where(valid, flood.astype("uint8"), 255)
    cls = np.where(~valid, 255, np.where(water_dry, 2, np.where(flood, 1, 0)))
    inv_path = f"{OUT_DIR}/flood_inventory_{tag}.tif"
    cls_path = f"{OUT_DIR}/water_classes_{tag}.tif"
    write_tif(inv_path, inv, profile)
    write_tif(cls_path, cls, profile)

    a_valid = area_km2(valid, row_area)
    stats = {
        "valid_area_km2": round(a_valid, 2),
        "water_in_flood_scene_km2": round(area_km2(water_flood, row_area), 2),
        "reference_water_km2": round(area_km2(water_dry, row_area), 2),
        "new_water_before_cleanup_km2": round(area_km2(raw_new, row_area), 2),
        "removed_steep_slope_km2": round(area_km2(steep, row_area), 2),
        "removed_small_patches_km2": round(area_km2(after_slope & ~flood, row_area), 2),
        "removed_small_patches_count": n_patches_removed,
        "flood_km2": round(area_km2(flood, row_area), 2),
        "flood_pct_of_aoi": round(100 * area_km2(flood, row_area) / a_valid, 2),
        "flood_px": int(flood.sum()),
    }
    for k, v in stats.items():
        print(f"    {k:32s} {v}")

    # NDEM same-date quick agreement (sanity only; Step 6 is the formal check)
    nd = ndem_all[ndem_all["date"] == f"{fdate[8:10]}-{fdate[5:7]}-{fdate[0:4]}"]
    ndem_stats = None
    ndem_mask = None
    if len(nd):
        ndem_mask = rasterize([(g, 1) for g in nd.geometry], out_shape=flood.shape,
                              transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
        dom = valid & ~water_dry
        ndem_stats = {"n_polygons": int(len(nd)),
                      "ndem_area_outside_ref_water_km2": round(area_km2(ndem_mask & dom, row_area), 2),
                      **confusion(flood, ndem_mask, dom)}
        print(f"    NDEM same-day ({len(nd)} polygons): POD={ndem_stats['POD']}  "
              f"FAR={ndem_stats['FAR']}  CSI={ndem_stats['CSI']}  "
              f"(NDEM area {ndem_stats['ndem_area_outside_ref_water_km2']} km2 vs ours {stats['flood_km2']} km2)")
    else:
        print("    NDEM: no polygons for this date")

    metadata["events"][event_name] = {
        "flood_scene": ev["flood_scene"]["id"], "flood_date": fdate,
        "dry_reference_scene": ev["dry_reference_scene"]["id"], "dry_date": ddate,
        "db_check": {"flood": f_status, "dry": d_status},
        "otsu_threshold_db": {"flood": round(t_flood, 3), "dry": round(t_dry, 3)},
        "otsu_threshold_plausible": {"flood": ok_f, "dry": ok_d},
        "areas": stats,
        "ndem_same_day_sanity_check": ndem_stats,
        "outputs": {"flood_inventory": inv_path, "water_classes": cls_path},
    }

    # -----------------------------------------------------------------------
    # 5.4 Visual validation figures
    # -----------------------------------------------------------------------
    print("  5.4 visual validation figures")
    extent = (BBOX[0], BBOX[2], BBOX[1], BBOX[3])
    base = basemap(BBOX, 12, 1600)
    ovl_cmap = ListedColormap([(0, 0, 0, 0), C_FLOOD, C_REF_WATER])
    cls_plot = np.where(cls == 255, 0, np.where(cls == 1, 1, np.where(cls == 2, 2, 0)))

    fig = plt.figure(figsize=(18, 13), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-30, 5, 176)
    ax.hist(flood_db[valid], bins=bins, histtype="step", lw=2, color=C_FLOOD,
            label=f"Flood {fdate}  (Otsu {t_flood:.2f} dB)")
    ax.hist(dry_db[valid], bins=bins, histtype="step", lw=2, color=C_REF_WATER,
            label=f"Dry ref {ddate}  (Otsu {t_dry:.2f} dB)")
    ax.axvline(t_flood, color=C_FLOOD, ls="--", lw=1.5)
    ax.axvline(t_dry, color=C_REF_WATER, ls="--", lw=1.5)
    ax.axvspan(*PLAUSIBLE_WATER_THRESHOLD_DB, color="#999999", alpha=0.12,
               label="typical water threshold range")
    ax.set_xlabel("VV backscatter, Lee-filtered (dB)")
    ax.set_ylabel("Pixel count")
    ax.set_title("5.2  Otsu thresholds on VV histograms", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(flood_db, cmap="gray", vmin=-25, vmax=0, extent=extent)
    ax.contour(np.flipud(flood_db < t_flood), levels=[0.5], colors=[C_FLOOD], linewidths=0.3,
               extent=extent, origin="lower")
    ax.set_title(f"Lee-filtered VV {fdate} (dB), Otsu water edge", loc="left", fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(base, extent=extent)
    ax.imshow(cls_plot, cmap=ovl_cmap, vmin=0, vmax=2, extent=extent, interpolation="nearest")
    for name, lat, lon, hs, _ in SITES_2018:
        ax.add_patch(plt.Rectangle((lon - hs, lat - hs), 2 * hs, 2 * hs, fill=False,
                                   ec="white", lw=1.0, ls="--"))
        below = name.startswith("Kalady")   # avoids collision with the airport box label
        ax.text(lon - hs if not below else lon + hs, lat - hs - 0.003 if below else lat + hs + 0.003,
                name, color="white", fontsize=7.5, fontweight="bold",
                va="top" if below else "bottom", ha="right" if below else "left")
    ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_title(f"5.3  Flood inventory {fdate} over satellite imagery  "
                 f"(flood = {stats['flood_km2']} km², {stats['flood_pct_of_aoi']}% of AOI)",
                 loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=C_FLOOD, label="Flooded (new water)"),
                       Patch(color=C_REF_WATER, label="Reference water (dry-season scene)")],
              loc="lower left", fontsize=9, framealpha=0.9)
    ax.text(0.995, 0.005, "Imagery: Esri, Maxar, Earthstar Geographics", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="white")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(base, extent=extent)
    ax.imshow(np.where(flood, 1, 0), cmap=ListedColormap([(0, 0, 0, 0), C_FLOOD]),
              vmin=0, vmax=1, extent=extent, interpolation="nearest")
    if len(nd):
        nd.boundary.plot(ax=ax, color=C_NDEM, lw=0.6)
        sub = (f"NDEM {len(nd)} polygons · POD {ndem_stats['POD']:.2f} · "
               f"FAR {ndem_stats['FAR']:.2f} · CSI {ndem_stats['CSI']:.2f}")
    else:
        sub = "no NDEM polygons for this date"
    ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_title(f"5.4  Our flood (fill) vs official NRSC/ISRO NDEM same-day extent (outline)\n{sub}",
                 loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=C_FLOOD, label="Our Otsu flood"),
                       Line2D([0], [0], color=C_NDEM, lw=2, label="NDEM inundation")],
              loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Step 5 flood inventory — Sentinel-1 VV {fdate} vs dry reference {ddate} "
                 f"(rel. orbit 165, descending)", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = f"{MAP_DIR}/step5_flood_inventory_{tag}_overview.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    figs.append(fig_path)
    print(f"    [OK] {fig_path}")

    if year == "2018":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12.5), dpi=110)
        t = profile["transform"]
        site_stats = []
        for ax, (name, lat, lon, hs, note) in zip(axes.ravel(), SITES_2018):
            b = (lon - hs, lat - hs, lon + hs, lat + hs)
            img = basemap(b, 15, 900)
            ax.imshow(img, extent=(b[0], b[2], b[1], b[3]))
            col0, row0 = ~t * (b[0], b[3])
            col1, row1 = ~t * (b[2], b[1])
            col0, row0 = int(col0), int(row0)
            col1, row1 = int(math.ceil(col1)), int(math.ceil(row1))
            win = (slice(row0, row1), slice(col0, col1))
            sub_cls = cls_plot[win]
            sub_ext = (t.c + col0 * t.a, t.c + col1 * t.a, t.f + row1 * t.e, t.f + row0 * t.e)
            ax.imshow(sub_cls, cmap=ovl_cmap, vmin=0, vmax=2, extent=sub_ext,
                      interpolation="nearest", alpha=0.75)
            if len(nd):
                nd.clip(box(*b)).boundary.plot(ax=ax, color=C_NDEM, lw=1.0)
            land = sub_cls != 2
            pct = 100 * (sub_cls == 1).sum() / max(land.sum(), 1)
            nd_pct = (100 * (ndem_mask[win] & land).sum() / max(land.sum(), 1)
                      if ndem_mask is not None else float("nan"))
            site_stats.append({"site": name, "our_flood_pct_of_land": round(float(pct), 1),
                               "ndem_flood_pct_of_land": round(float(nd_pct), 1)})
            ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
            ax.set_title(f"{name}\nours {pct:.1f}% of land flooded · NDEM {nd_pct:.1f}%  ({note})",
                         loc="left", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
        fig.legend(handles=[Patch(color=C_FLOOD, label="Our Otsu flood (new water)"),
                            Patch(color=C_REF_WATER, label="Reference water (dry-season)"),
                            Line2D([0], [0], color=C_NDEM, lw=2, label="NDEM 21-08-2018 inundation")],
                   loc="lower center", ncol=3, fontsize=11, frameon=False)
        fig.suptitle(f"5.4  Visual sanity check at documented Aug-2018 flood sites — S1 {fdate}  "
                     f"(Imagery: Esri, Maxar, Earthstar Geographics)",
                     fontsize=14, fontweight="bold", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig_path = f"{MAP_DIR}/step5_flood_inventory_{tag}_site_checks.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figs.append(fig_path)
        metadata["events"][event_name]["site_checks"] = site_stats
        print(f"    [OK] {fig_path}")
        for s in site_stats:
            print(f"      {s['site']:32s} ours {s['our_flood_pct_of_land']:5.1f}%   "
                  f"NDEM {s['ndem_flood_pct_of_land']:5.1f}%")

        # Interactive map
        import folium
        rgba = np.zeros(cls.shape + (4,), dtype=np.uint8)
        rgba[cls == 1] = (235, 104, 52, 210)
        rgba[cls == 2] = (42, 120, 214, 150)
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
        import base64
        png_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        m = folium.Map(location=[10.08, 76.35], zoom_start=11, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics", name="Esri World Imagery").add_to(m)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.raster_layers.ImageOverlay(png_uri, bounds=[[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]],
                                          name=f"Otsu flood {fdate} (orange) / reference water (blue)",
                                          interactive=False).add_to(m)
        if len(nd):
            folium.GeoJson(nd[["geometry"]].to_json(), name="NDEM 21-08-2018 inundation",
                           style_function=lambda _: {"color": C_NDEM, "weight": 1.2,
                                                     "fillOpacity": 0}).add_to(m)
        for name, lat, lon, _, note in SITES_2018:
            folium.Marker([lat, lon], tooltip=f"{name}: {note}").add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        html_path = f"{MAP_DIR}/step5_flood_inventory_{tag}_interactive.html"
        m.save(html_path)
        figs.append(html_path)
        print(f"    [OK] {html_path}")

metadata["figures"] = figs
meta_path = f"{OUT_DIR}/flood_inventory_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\n[OK] Metadata -> {meta_path}")

print("\n" + "=" * 80)
print("STEP 5 SUMMARY")
print("=" * 80)
for name, e in metadata["events"].items():
    nd = e["ndem_same_day_sanity_check"]
    nd_txt = f"NDEM POD {nd['POD']} / FAR {nd['FAR']} / CSI {nd['CSI']}" if nd else "no NDEM"
    print(f"  {e['flood_date']}: Otsu {e['otsu_threshold_db']['flood']} dB (dry {e['otsu_threshold_db']['dry']} dB)"
          f" -> flood {e['areas']['flood_km2']} km2 ({e['areas']['flood_pct_of_aoi']}%) | {nd_txt}")
print("=" * 80)
