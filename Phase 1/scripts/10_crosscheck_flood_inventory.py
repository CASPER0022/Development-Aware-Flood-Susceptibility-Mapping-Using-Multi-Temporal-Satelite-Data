"""
Step 6 - Cross-check the Otsu flood inventory against independent sources
(Week 2, Phase1_Implementation_Plan).

6.1. Independent sources. The plan names Copernicus EMS rapid mapping or the
     Dartmouth Flood Observatory (DFO). Both were checked first:
       - Copernicus EMS: no Rapid Mapping (EMSR) activation exists for the
         Aug-2018 Kerala floods; the event was mapped under the International
         Charter (activation 582), whose Indian products are NRSC/ISRO's NDEM
         maps -- source A below.
       - DFO: the DFO / Cloud-to-Street Global Flood Database on GEE
         (GLOBAL_FLOOD_DB/MODIS_EVENTS/V1) has 24 events over this AOI but the
         latest is DFO 4378 (2016); it contains NO 2018 India event (queried
         live, see metadata). The DFO archive site itself returns HTTP 410.
         (MODIS 250 m would also be ~150x coarser than our 20 m pixels.)
     So the script uses:
       A. NDEM (NRSC/ISRO) official flood-inundation polygons -- the national
          agency's own SAR maps (RADARSAT-2 / Sentinel-1), produced by a
          different team, chain and thresholds. Two ways:
            A1 same-day  : NDEM passes on our S1 date (21-08-2018 evening etc.)
            A2 envelope  : union of every NDEM pass of that event (2018: 17-28
                           Aug, incl. the near-peak 17 & 18 Aug maps). A pixel
                           we see flooded on 21 Aug should lie inside the area
                           the event flooded at some point; residual water on
                           the 21st is a subset of the peak.
       B. Sentinel-2 MNDWI (optical, different physics) on 2018-08-22, ~28 h
          after our S1 pass, the first low-cloud S2 scene after the peak.
          new water = MNDWI>0 on 22 Aug AND NOT MNDWI>0 in a Jan-Mar 2018
          median; clouds (s2cloudless prob > 40, +100 m) and every pixel in the
          projected cloud-shadow zone (2 km) are excluded from the comparison.
          2019 has no usable S2 scene (Aug 2019 tiles 73-100 % cloud).
          The comparison is only reported if >= 20 % of the domain is clear.
          RESULT: on 22 Aug 2018 s2cloudless flags ~82 % of the AOI as cloud
          (the 13 % CLOUDY_PIXEL_PERCENTAGE is a whole-tile figure), so B is
          recorded as rejected. The next clear scene (2018-09-04) is two weeks
          after our pass and would test flood persistence, not agreement.

6.2. Agreement numbers. The base paper's Ground Truth Index (84.05 %) is "how
     often the model's flood predictions matched real, reported flood
     locations", i.e. the share of our flood calls confirmed by the reference
     (a precision). Reported here:
       GTI_px      share of our flood pixels inside the reference flood extent
       GTI_px_40m  same, counting a pixel as confirmed if a reference flood
                   pixel lies within 40 m (2 px) -- absorbs the resolution /
                   geolocation mismatch between two independently made maps
       GTI_patch   share of our flood patches (8-connected, >= 10 px) that
                   touch the reference extent -- closest to the base paper's
                   point-report check
       random baseline = reference prevalence in the comparison domain (what
                   a random map of the same size would score); lift = GTI/that
     plus the full contingency table: POD (recall), FAR, CSI, F1, OA, kappa,
     area bias.
     Comparison domain: valid pixels, excluding OUR reference water (class 2)
     -- permanent water is not part of the flood/no-flood question -- and,
     for S2, excluding S2 cloud/shadow and S2 dry-season water.
"""
import os
import sys
import json
import math
import time

import numpy as np
import requests
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import ee

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

INV_DIR = "data/processed/flood_inventory"
NDEM = "data/validation/NDEM_KL_Floods_Inundation.parquet"
S2_DIR = "data/processed/crosscheck"
MAP_DIR = "outputs/maps"
METRICS_DIR = "outputs/metrics"
TILE_CACHE = "cache/esri_world_imagery"
for d in (S2_DIR, MAP_DIR, METRICS_DIR, TILE_CACHE):
    os.makedirs(d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
TOL_PX = 2                 # 40 m tolerance at 20 m
MIN_PATCH_EVAL_PX = 10     # patches counted for GTI_patch (~0.4 ha)
S2_CLOUD_PROB = 40
S2_CLOUD_BUFFER_M = 100
S2_SHADOW_PROJ_KM = 2
S2_MIN_CLEAR_SHARE = 0.20  # below this the S2 comparison is rejected, not reported
EDGE_PX = 3                # "water-edge" diagnostic: within 60 m of reference water

EVENTS = {
    "2018_flood": {
        "tag": "2018_20180821",
        "ndem_same_day": ["21-08-2018"],
        "ndem_envelope": ["17-08-2018", "18-08-2018", "21-08-2018", "24-08-2018",
                          "27-08-2018", "28-08-2018"],
        "s2": {"date": ("2018-08-22", "2018-08-23"), "ref": ("2018-01-01", "2018-04-01")},
    },
    "2019_flood": {
        "tag": "2019_20190810",
        "ndem_same_day": ["10-08-2019"],
        "ndem_envelope": ["10-08-2019", "12-08-2019"],
        "s2": None,
    },
    "2021_flood": {
        "tag": "2021_20211016",
        "ndem_same_day": ["16-10-2021"],
        "ndem_envelope": ["16-10-2021", "19-10-2021"],
        "s2": None,
    },
}

# Agreement-map colors (dataviz reference palette, categorical slots 1-3)
C_MISS = "#2a78d6"      # reference only
C_FA = "#eb6834"        # ours only
C_HIT = "#1baf7a"       # both
C_BAR = "#2a78d6"
C_BASE = "#52514e"

print("=" * 80)
print("STEP 6: CROSS-CHECK OF THE OTSU FLOOD INVENTORY AGAINST INDEPENDENT SOURCES")
print("=" * 80)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pixel_area_km2(transform, height):
    dy, dx = abs(transform.e), abs(transform.a)
    lats = transform.f - (np.arange(height) + 0.5) * dy
    return (dx * 111.320 * np.cos(np.radians(lats))) * (dy * 110.574)


def area_km2(mask, row_area):
    return float((mask.sum(axis=1) * row_area).sum())


def dilate(mask, px):
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=px)


def agreement(ours, ref, domain, row_area):
    o, r = ours & domain, ref & domain
    tp = int((o & r).sum()); fp = int((o & ~r).sum())
    fn = int((~o & r).sum()); tn = int((domain & ~o & ~r).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    pod = tp / (tp + fn) if tp + fn else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * prec * pod / (prec + pod) if prec + pod else float("nan")
    oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else float("nan")
    prevalence = (tp + fn) / n

    # tolerant precision: our pixel confirmed if ref flood within TOL_PX
    r_tol = dilate(ref, TOL_PX) & domain
    gti_tol = int((o & r_tol).sum()) / max(int(o.sum()), 1)
    prevalence_tol = int(r_tol.sum()) / n

    # patch-level: share of our patches (>= MIN_PATCH_EVAL_PX) touching the ref
    lab, nlab = ndimage.label(o, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    big = np.flatnonzero(sizes >= MIN_PATCH_EVAL_PX)
    big = big[big > 0]
    touched = np.unique(lab[o & r_tol])
    n_hit_patches = int(np.isin(big, touched).sum())
    gti_patch = n_hit_patches / len(big) if len(big) else float("nan")

    r4 = lambda v: round(float(v), 4)
    return {
        "domain_km2": round(area_km2(domain, row_area), 2),
        "ours_km2": round(area_km2(o, row_area), 2),
        "reference_km2": round(area_km2(r, row_area), 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "GTI_px": r4(prec), "GTI_px_40m": r4(gti_tol), "GTI_patch": r4(gti_patch),
        "n_patches_evaluated": int(len(big)), "n_patches_confirmed": n_hit_patches,
        "random_baseline": r4(prevalence), "random_baseline_40m": r4(prevalence_tol),
        "lift_px": r4(prec / prevalence) if prevalence else None,
        "lift_px_40m": r4(gti_tol / prevalence_tol) if prevalence_tol else None,
        "POD": r4(pod), "FAR": r4(1 - prec), "CSI": r4(csi), "F1": r4(f1),
        "OA": r4(oa), "kappa": r4(kappa),
        "area_bias": r4((tp + fp) / (tp + fn)) if tp + fn else None,
    }


def show(name, m):
    print(f"    {name:28s} GTI {100*m['GTI_px']:5.1f}% | 40m {100*m['GTI_px_40m']:5.1f}% "
          f"(random {100*m['random_baseline_40m']:4.1f}%, lift {m['lift_px_40m']}) | "
          f"patch {100*m['GTI_patch']:5.1f}% of {m['n_patches_evaluated']} | "
          f"POD {m['POD']:.2f} CSI {m['CSI']:.2f} k {m['kappa']:.2f} | "
          f"ours {m['ours_km2']} vs ref {m['reference_km2']} km2")


# --- Esri World Imagery basemap (same helper as script 09) ------------------
MERC = 20037508.342789244


def _tile_xy(lon, lat, z):
    n = 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


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
# 6.1 Source availability (logged, so the EMS/DFO decision is reproducible)
# ---------------------------------------------------------------------------
ee.Initialize(project="btp-flood")
region = ee.Geometry.Rectangle(BBOX)

print("\n[6.1] Checking the plan's named sources")
gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
gfd_aoi = gfd.filterBounds(region)
gfd_ids = gfd_aoi.aggregate_array("id").getInfo()
gfd_last = ee.Date(gfd_aoi.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo()
gfd_2018_ind = gfd.filterDate("2018-01-01", "2019-01-01").filter(ee.Filter.eq("cc", "IND")).size().getInfo()
print(f"    DFO Global Flood Database: {len(gfd_ids)} events over AOI, latest {gfd_last}; "
      f"2018 India events in DB: {gfd_2018_ind}  -> not usable for 2018/2019/2021")
print("    Copernicus EMS: no Rapid Mapping activation for Kerala Aug 2018 "
      "(mapped via International Charter 582 / NRSC-NDEM) -> use NDEM")

sources = {
    "copernicus_ems": {"usable": False,
                       "reason": "No EMSR Rapid Mapping activation for the Aug-2018 Kerala floods; the event "
                                 "was covered by International Charter activation 582, whose Indian products "
                                 "are NRSC/ISRO NDEM maps (used here as source A)."},
    "dfo_global_flood_database": {"usable": False, "gee_asset": "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1",
                                  "events_over_aoi": len(gfd_ids), "latest_event_date_over_aoi": gfd_last,
                                  "india_events_in_2018": gfd_2018_ind,
                                  "reason": "No 2018/2019/2021 event over the AOI; also MODIS 250 m. "
                                            "DFO archive website returns HTTP 410 Gone."},
    "A_ndem": {"usable": True, "file": NDEM,
               "provider": "NRSC/ISRO National Database for Emergency Management (see data/validation/SOURCE.md)"},
    "B_sentinel2_mndwi": {"usable": True, "asset": "COPERNICUS/S2_HARMONIZED + COPERNICUS/S2_CLOUD_PROBABILITY",
                          "events": ["2018_flood"],
                          "note": "Aug 2019 S2 tiles 73-100 % cloudy; 2021 not attempted (near-null event)"},
}


# ---------------------------------------------------------------------------
# Source B: Sentinel-2 MNDWI water on our exact S1 grid (GEE)
# ---------------------------------------------------------------------------
def s2_masked(img_col, prob_col):
    joined = ee.Join.saveFirst("prob").apply(
        img_col, prob_col, ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def per_image(img):
        img = ee.Image(img)
        prob = ee.Image(img.get("prob")).select("probability")
        cloud = prob.gt(S2_CLOUD_PROB).focalMax(S2_CLOUD_BUFFER_M, "circle", "meters")
        az = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
        shadow_zone = (cloud.directionalDistanceTransform(az, S2_SHADOW_PROJ_KM * 10)
                       .reproject(crs=img.select("B2").projection(), scale=100)
                       .select("distance").mask())
        bad = cloud.Or(shadow_zone)
        mndwi = img.normalizedDifference(["B3", "B11"]).rename("mndwi")
        return mndwi.updateMask(bad.Not())

    return ee.ImageCollection(joined.map(per_image))


def fetch_s2_water(cfg, profile, out_path):
    if os.path.exists(out_path):
        print(f"    cached {out_path}")
        return
    col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region)
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(region)
    d0, d1 = cfg["date"]
    r0, r1 = cfg["ref"]
    flood_m = s2_masked(col.filterDate(d0, d1), prob.filterDate(d0, d1)).mosaic()
    ref_m = s2_masked(col.filterDate(r0, r1), prob.filterDate(r0, r1)).median()
    scenes = col.filterDate(d0, d1).aggregate_array("system:index").getInfo()
    n_ref = col.filterDate(r0, r1).size().getInfo()
    print(f"    S2 flood scenes {scenes}; dry-season reference composite from {n_ref} scenes")
    clear = flood_m.mask().And(ref_m.mask()).rename("clear")
    img = (flood_m.gt(0).unmask(0).rename("water_flood")
           .addBands(ref_m.gt(0).unmask(0).rename("water_ref"))
           .addBands(clear.unmask(0))
           .toByte())
    t = profile["transform"]
    url = img.getDownloadURL({
        "crs": "EPSG:4326",
        "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "dimensions": f"{profile['width']}x{profile['height']}",
        "format": "GEO_TIFF",
    })
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"    [OK] {out_path} ({len(resp.content)/1e6:.1f} MB)")
    return {"flood_scenes": scenes, "reference_scene_count": n_ref}


# ---------------------------------------------------------------------------
# Load NDEM
# ---------------------------------------------------------------------------
ndem = gpd.read_parquet(NDEM)
if ndem.crs is None or ndem.crs.to_epsg() != 4326:
    ndem = ndem.set_crs("EPSG:4326", allow_override=True)
ndem = ndem[ndem.intersects(box(*BBOX))].copy()
ndem["date"] = ndem["from_time"].astype(str).str[:10]


def ndem_mask(dates, profile):
    sel = ndem[ndem["date"].isin(dates)]
    if not len(sel):
        return np.zeros((profile["height"], profile["width"]), bool), 0
    m = rasterize([(g, 1) for g in sel.geometry], out_shape=(profile["height"], profile["width"]),
                  transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    return m, int(len(sel))


# ---------------------------------------------------------------------------
# 6.2 Agreement per event and source
# ---------------------------------------------------------------------------
results = {"task": "Step 6 - Independent cross-check of the Otsu flood inventory",
           "gti_definition": "share of our flood pixels (or patches) confirmed by the reference extent; "
                             "base paper Ground Truth Index = 84.05 %",
           "tolerance_m": TOL_PX * 20, "min_patch_eval_px": MIN_PATCH_EVAL_PX,
           "sources": sources, "events": {}}
maps_2018 = {}

for ev_name, cfg in EVENTS.items():
    print(f"\n[6.2] {ev_name}")
    with rasterio.open(f"{INV_DIR}/water_classes_{cfg['tag']}.tif") as src:
        cls = src.read(1)
        profile = src.profile
    row_area = pixel_area_km2(profile["transform"], profile["height"])
    valid = cls != 255
    ours = cls == 1
    domain = valid & (cls != 2)
    ev_res = {}

    same, n_same = ndem_mask(cfg["ndem_same_day"], profile)
    env, n_env = ndem_mask(cfg["ndem_envelope"], profile)
    ev_res["A1_ndem_same_day"] = {"ndem_dates": cfg["ndem_same_day"], "n_polygons": n_same,
                                  **agreement(ours, same, domain, row_area)}
    ev_res["A2_ndem_event_envelope"] = {"ndem_dates": cfg["ndem_envelope"], "n_polygons": n_env,
                                        **agreement(ours, env, domain, row_area)}
    show("A1 NDEM same-day", ev_res["A1_ndem_same_day"])
    show("A2 NDEM event envelope", ev_res["A2_ndem_event_envelope"])

    if cfg["s2"]:
        s2_path = f"{S2_DIR}/s2_mndwi_water_{cfg['tag']}.tif"
        s2_info = fetch_s2_water(cfg["s2"], profile, s2_path)
        with rasterio.open(s2_path) as src:
            s2_flood, s2_ref, s2_clear = (src.read(i).astype(bool) for i in (1, 2, 3))
        s2_new = s2_flood & ~s2_ref
        s2_dom = domain & s2_clear & ~s2_ref
        clear_share = float(s2_dom.sum() / domain.sum())
        print(f"    S2 comparable (cloud/shadow-free, not S2 dry water): {100*clear_share:.1f}% of domain")
        b = {"s2_date": cfg["s2"]["date"][0], **(s2_info or {}),
             "clear_share_of_domain": round(clear_share, 4)}
        if clear_share >= S2_MIN_CLEAR_SHARE:
            b.update(status="used", **agreement(ours, s2_new, s2_dom, row_area))
            show("B  Sentinel-2 MNDWI", b)
        else:
            b.update(status="rejected",
                     reason=f"only {100*clear_share:.1f}% of the domain is cloud/shadow-free "
                            f"(< {100*S2_MIN_CLEAR_SHARE:.0f}% minimum)")
            results["sources"]["B_sentinel2_mndwi"]["usable"] = False
            print(f"    [REJECTED] Sentinel-2: {b['reason']}")
        ev_res["B_sentinel2_mndwi"] = b

    # Diagnostic: where do the unconfirmed (ours-only) pixels sit?
    near_water = dilate(cls == 2, EDGE_PX) & domain
    unconf = ours & domain & ~dilate(env, TOL_PX)
    ev_res["diagnostic_unconfirmed_vs_envelope"] = {
        "unconfirmed_km2": round(area_km2(unconf, row_area), 2),
        "share_within_60m_of_reference_water": round(float((unconf & near_water).sum() / max(unconf.sum(), 1)), 4),
        "share_of_domain_within_60m_of_reference_water": round(float(near_water.sum() / domain.sum()), 4),
        "GTI_px_40m_excluding_60m_water_edge": agreement(ours, env, domain & ~near_water, row_area)["GTI_px_40m"],
    }
    dg = ev_res["diagnostic_unconfirmed_vs_envelope"]
    print(f"    diag: {dg['unconfirmed_km2']} km2 unconfirmed; "
          f"{100*dg['share_within_60m_of_reference_water']:.1f}% of it within 60 m of permanent water "
          f"(that belt is {100*dg['share_of_domain_within_60m_of_reference_water']:.1f}% of the domain); "
          f"GTI(40 m) outside the belt {100*dg['GTI_px_40m_excluding_60m_water_edge']:.1f}%")

    if ev_name == "2018_flood":
        maps_2018 = {"cls": cls, "profile": profile, "domain": domain, "ours": ours,
                     "same": same, "env": env}
    results["events"][ev_name] = ev_res

# ---------------------------------------------------------------------------
# Headline number
# ---------------------------------------------------------------------------
h = results["events"]["2018_flood"]["A2_ndem_event_envelope"]
results["headline"] = {
    "metric": "GTI analogue (2018, vs official NDEM event envelope, 40 m tolerance)",
    "value_pct": round(100 * h["GTI_px_40m"], 1),
    "strict_pixel_pct": round(100 * h["GTI_px"], 1),
    "patch_pct": round(100 * h["GTI_patch"], 1),
    "random_baseline_pct": round(100 * h["random_baseline_40m"], 1),
    "base_paper_gti_pct": 84.05,
}
out_json = f"{METRICS_DIR}/step6_flood_inventory_crosscheck.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n[OK] {out_json}")

# ---------------------------------------------------------------------------
# Figures (2018)
# ---------------------------------------------------------------------------
print("\n[fig] 2018 agreement maps")
m = maps_2018
extent = (BBOX[0], BBOX[2], BBOX[1], BBOX[3])
base = basemap(BBOX, 12, 1600)
agree_cmap = ListedColormap([(0, 0, 0, 0), C_HIT, C_FA, C_MISS])


def agreement_raster(ours, ref, dom):
    a = np.zeros(ours.shape, np.uint8)
    a[dom & ours & ref] = 1
    a[dom & ours & ~ref] = 2
    a[dom & ~ours & ref] = 3
    return a


r18 = results["events"]["2018_flood"]
panels = [
    ("A1  vs NDEM same-day (21 Aug 2018, evening)", agreement_raster(m["ours"], m["same"], m["domain"]),
     r18["A1_ndem_same_day"]),
    ("A2  vs NDEM event envelope (17–28 Aug 2018)", agreement_raster(m["ours"], m["env"], m["domain"]),
     r18["A2_ndem_event_envelope"]),
]
fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), dpi=110)
for ax, (title, arr, st) in zip(axes, panels):
    ax.imshow(base, extent=extent)
    ax.imshow(arr, cmap=agree_cmap, vmin=0, vmax=3, extent=extent, interpolation="nearest")
    ax.set_xlim(BBOX[0], BBOX[2]); ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_title(f"{title}\nGTI {100*st['GTI_px']:.1f}% · within 40 m {100*st['GTI_px_40m']:.1f}% "
                 f"(random {100*st['random_baseline_40m']:.1f}%) · patches {100*st['GTI_patch']:.0f}% · "
                 f"POD {st['POD']:.2f}",
                 loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
axes[0].text(0.995, 0.005, "Imagery: Esri, Maxar, Earthstar Geographics", transform=axes[0].transAxes,
             ha="right", va="bottom", fontsize=6.5, color="white")
fig.legend(handles=[Patch(color=C_HIT, label="Both flooded (agreement)"),
                    Patch(color=C_FA, label="Ours only"),
                    Patch(color=C_MISS, label="NDEM only")],
           loc="lower center", ncol=3, fontsize=11, frameon=False)
fig.suptitle("Step 6  Sentinel-1 Otsu flood inventory (21 Aug 2018) vs official NRSC/ISRO NDEM flood maps",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
p1 = f"{MAP_DIR}/step6_crosscheck_2018_agreement_maps.png"
fig.savefig(p1, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p1}")

print("[fig] GTI summary chart")
rows = []
for ev_name, ev in results["events"].items():
    for key, label in [("A1_ndem_same_day", "NDEM same-day"),
                       ("A2_ndem_event_envelope", "NDEM event envelope"),
                       ("B_sentinel2_mndwi", "Sentinel-2 MNDWI")]:
        if key in ev and "GTI_px_40m" in ev[key]:
            rows.append((f"{ev_name[:4]} · {label}", ev[key]))
rows = rows[::-1]
fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), dpi=130)
y = np.arange(len(rows))
vals = [100 * r["GTI_px_40m"] for _, r in rows]
ax.barh(y, vals, height=0.55, color=C_BAR, zorder=2)
ax.scatter([100 * r["random_baseline_40m"] for _, r in rows], y, marker="|", s=260, lw=2.5,
           color=C_BASE, zorder=3, label="Random map of same size")
ax.axvline(84.05, color=C_BASE, ls="--", lw=1, zorder=1, label="Base paper GTI (84.05%)")
for yi, v in zip(y, vals):
    ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9, color="#0b0b0b")
ax.set_yticks(y, [n for n, _ in rows], fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of our flood pixels confirmed by the reference (within 40 m), %")
ax.set_title("Ground-Truth-Index analogue by event and reference source", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.tight_layout()
p2 = f"{MAP_DIR}/step6_crosscheck_gti_summary.png"
fig.savefig(p2, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p2}")

print("\n" + "=" * 80)
print("STEP 6 SUMMARY")
print("=" * 80)
hl = results["headline"]
print(f"  Headline: {hl['metric']}: {hl['value_pct']}%  "
      f"(strict pixel {hl['strict_pixel_pct']}%, patch {hl['patch_pct']}%, "
      f"random baseline {hl['random_baseline_pct']}%; base paper GTI {hl['base_paper_gti_pct']}%)")
print("=" * 80)
