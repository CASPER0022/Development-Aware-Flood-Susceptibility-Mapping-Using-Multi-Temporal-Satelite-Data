"""
Shared Phase 2 helpers (Phase 2 plan, Step 1.4).

Reusable code lifted from the verified Phase 1 scripts, unchanged in logic, so
that every Phase 2 step uses one copy:

    grid / areas      row_cell_sizes_m (Step 7), pixel_area_km2, area_km2 (Steps 5-6)
    flow routing      priority_flood_eps, quinn_fd8, route_flow (Step 7)
    labels            dilate, to_30m_fraction (exact 20 m -> 30 m, Step 8),
                      remove_small_patches (Step 5)
    Sentinel-1        load_s1, preprocess (dB check + Lee + Eq. 1), otsu_threshold (Step 5)
    cross-check       agreement (GTI and contingency table, Step 6), NDEM loading
    modelling         LGB_PARAMS, metrics, rounded, cv_summary, block_groups (Step 9)
    figures           basemap (Esri World Imagery, cached), palette constants

Rule: if a function here is changed, Phase 2 Step 1 (01_freeze_baseline_check.py)
must be rerun and must still reproduce the Phase 1 numbers.
"""
import os
import sys
import math
import time
import heapq
import json

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths and project constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PHASE2_ROOT, ".."))
PHASE1_ROOT = os.path.join(PROJECT_ROOT, "Phase 1")


def p1(*parts):
    """Absolute path inside Phase 1."""
    return os.path.join(PHASE1_ROOT, *parts)


def p2(*parts):
    """Absolute path inside Phase 2."""
    return os.path.join(PHASE2_ROOT, *parts)


# Phase 1 inputs reused by Phase 2
P1_FABDEM = p1("data", "raw", "fabdem_30m_aoi.tif")
P1_STACK = p1("data", "processed", "features", "feature_stack.tif")
P1_DOMAIN = p1("data", "processed", "features", "analysis_domain.tif")
P1_FLOWACC = p1("data", "processed", "features", "flow_accumulation_km2.tif")
P1_TRAINING_2018 = p1("data", "processed", "training", "training_samples_2018.csv")
P1_STEP8_JSON = p1("outputs", "metrics", "step8_training_dataset_summary.json")
P1_STEP9_JSON = p1("outputs", "metrics", "step9_baseline_metrics.json")
P1_NDEM = p1("data", "validation", "NDEM_KL_Floods_Inundation.parquet")
P1_INV_DIR = p1("data", "processed", "flood_inventory")
TILE_CACHE = p1("cache", "esri_world_imagery")      # shared with Phase 1, avoids re-downloading

# Phase 2 output folders
RAW_DIR = p2("data", "raw")
PROC_DIR = p2("data", "processed")
MAP_DIR = p2("outputs", "maps")
METRICS_DIR = p2("outputs", "metrics")
MODEL_DIR = p2("outputs", "models")
DOCS_DIR = p2("docs")
for _d in (RAW_DIR, PROC_DIR, MAP_DIR, METRICS_DIR, MODEL_DIR, DOCS_DIR, TILE_CACHE):
    os.makedirs(_d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
UTM = "EPSG:32643"
GEE_PROJECT = "btp-flood"
SEED = 42
FILL_EPS_M = 1e-6            # Priority-Flood+eps step (nextafter() is subnormal at 0 m)

# Palette (same reference palette as Phase 1 figures)
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_GREEN = "#1baf7a"
C_YELLOW = "#eda100"
C_GREY = "#52514e"
C_GRID = "#e5e5e5"


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def banner(title):
    print("=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Grid geometry and areas
# ---------------------------------------------------------------------------
def row_cell_sizes_m(transform, height):
    """True WGS84 cell width/height (m) and area (m^2) for each grid row (Step 7)."""
    a, e2 = 6378137.0, 6.69437999014e-3
    lat = np.radians(transform.f + (np.arange(height) + 0.5) * transform.e)
    w = 1 - e2 * np.sin(lat) ** 2
    m_rad = a * (1 - e2) / w ** 1.5          # meridional radius of curvature
    n_rad = a / np.sqrt(w)                    # prime-vertical radius
    dy = m_rad * np.radians(abs(transform.e))
    dx = n_rad * np.cos(lat) * np.radians(abs(transform.a))
    return dx, dy, dx * dy


def pixel_area_km2(transform, height):
    """Per-row pixel area (km^2), spherical approximation used in Steps 5-6."""
    dy, dx = abs(transform.e), abs(transform.a)
    lats = transform.f - (np.arange(height) + 0.5) * dy
    return (dx * 111.320 * np.cos(np.radians(lats))) * (dy * 110.574)


def area_km2(mask, row_area):
    return float((mask.sum(axis=1) * row_area).sum())


# ---------------------------------------------------------------------------
# Terrain operators, Step 7
# ---------------------------------------------------------------------------
TAN_SLOPE_MIN = 0.001        # TWI floor for tan(beta)


def neighbourhood(z):
    """Z1..Z9 of every 3x3 window (Z1 = NW ... Z9 = SE), odd-reflect padded."""
    zp = np.pad(z.astype(np.float64), 1, mode="reflect", reflect_type="odd")
    return (zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:],
            zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:],
            zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:])


def horn_gradient(z, dx, dy):
    """dz/dx (east +) and dz/dy (north +); dx, dy are per-row arrays (m)."""
    z1, z2, z3, z4, _, z6, z7, z8, z9 = neighbourhood(z)
    p = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * dx[:, None])
    q = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8 * dy[:, None])
    return p, q


def slope_aspect(p, q):
    slope = np.degrees(np.arctan(np.hypot(p, q)))
    aspect = np.degrees(np.arctan2(-p, -q)) % 360.0     # direction of steepest descent
    aspect[(p == 0) & (q == 0)] = -1.0
    return slope, aspect


def twi_from_acc(acc, p, q, dx, dy):
    """TWI = ln((a / w) / max(tan b, 0.001)), w = sqrt(dx*dy) (Step 7)."""
    width = np.sqrt(dx * dy)[:, None]
    with np.errstate(divide="ignore"):
        return np.log((acc / width) / np.maximum(np.hypot(p, q), TAN_SLOPE_MIN))


def open_water_and_land(dem):
    """Step 7 domain: open water = 0 m cells 8-connected to the grid border."""
    from scipy import ndimage
    zero = dem == 0
    lab, _ = ndimage.label(zero, structure=np.ones((3, 3)))
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    border = border[border > 0]
    water = np.isin(lab, border)
    return water, ~water


# ---------------------------------------------------------------------------
# Flow routing (numba), Step 7
# ---------------------------------------------------------------------------
from numba import njit  # noqa: E402

DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)


@njit(cache=True)
def priority_flood_eps(z, outlet, eps):
    """Barnes et al. (2014) Priority-Flood+epsilon. Seeds: grid border and
    outlet (open-water) cells. eps is a fixed step (see Phase 1 Step 7 bug 1).
    Returns the filled float64 DEM and the number of raised cells."""
    H, W = z.shape
    zf = z.astype(np.float64)
    closed = np.zeros((H, W), np.bool_)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    pit = np.empty(H * W, np.int64)
    p_head = 0
    p_tail = 0
    for r in range(H):
        for c in range(W):
            if outlet[r, c] or r == 0 or c == 0 or r == H - 1 or c == W - 1:
                closed[r, c] = True
                heapq.heappush(heap, (zf[r, c], np.int64(r * W + c)))
    n_raised = 0
    while len(heap) > 0 or p_head < p_tail:
        if p_head < p_tail:
            idx = pit[p_head]
            p_head += 1
        else:
            idx = heapq.heappop(heap)[1]
        r = idx // W
        c = idx % W
        z_next = zf[r, c] + eps
        for k in range(8):
            nr = r + DR[k]
            nc = c + DC[k]
            if nr < 0 or nc < 0 or nr >= H or nc >= W or closed[nr, nc]:
                continue
            closed[nr, nc] = True
            if zf[nr, nc] <= z_next:
                if zf[nr, nc] < z_next:
                    n_raised += 1
                zf[nr, nc] = z_next
                pit[p_tail] = np.int64(nr * W + nc)
                p_tail += 1
            else:
                heapq.heappush(heap, (zf[nr, nc], np.int64(nr * W + nc)))
    return zf, n_raised


@njit(cache=True)
def quinn_fd8(zf, land, order, cell_area, dx, dy):
    """Quinn et al. (1991) FD8 accumulation of a per-cell quantity (m^2).
    order: flat indices of land cells, highest filled elevation first.
    cell_area: per-row weights (H,) -- the upslope AREA case of Step 7.
    Returns acc, amount delivered to open water, amount stranded in cells with
    no lower neighbour (border only), and the number of non-border land cells
    without a strictly lower neighbour (must be 0)."""
    H, W = zf.shape
    acc = np.zeros((H, W), np.float64)
    for r in range(H):
        for c in range(W):
            if land[r, c]:
                acc[r, c] = cell_area[r]
    return _fd8_core(zf, land, order, acc, dx, dy)


@njit(cache=True)
def quinn_fd8_weighted(zf, land, order, weight, dx, dy):
    """Same FD8 routing, but each land cell starts with weight[r, c] (H, W)
    instead of its area -- e.g. impervious area for UICA (Phase 2 Step 4)."""
    H, W = zf.shape
    acc = np.zeros((H, W), np.float64)
    for r in range(H):
        for c in range(W):
            if land[r, c]:
                acc[r, c] = weight[r, c]
    return _fd8_core(zf, land, order, acc, dx, dy)


@njit(cache=True)
def _fd8_core(zf, land, order, acc, dx, dy):
    H, W = zf.shape
    to_water = 0.0
    stranded = 0.0
    n_interior_sinks = 0
    wts = np.zeros(8, np.float64)
    for i in range(order.shape[0]):
        idx = order[i]
        r = idx // W
        c = idx % W
        diag = np.sqrt(dx[r] ** 2 + dy[r] ** 2)
        total = 0.0
        for k in range(8):
            wts[k] = 0.0
            nr = r + DR[k]
            nc = c + DC[k]
            if nr < 0 or nc < 0 or nr >= H or nc >= W:
                continue
            drop = zf[r, c] - zf[nr, nc]
            if drop <= 0.0:
                continue
            if DR[k] != 0 and DC[k] != 0:
                wts[k] = drop / diag * 0.354
            elif DR[k] == 0:
                wts[k] = drop / dx[r] * 0.5
            else:
                wts[k] = drop / dy[r] * 0.5
            total += wts[k]
        a = acc[r, c]
        if total == 0.0:
            stranded += a
            if not (r == 0 or c == 0 or r == H - 1 or c == W - 1):
                n_interior_sinks += 1
            continue
        for k in range(8):
            if wts[k] == 0.0:
                continue
            nr = r + DR[k]
            nc = c + DC[k]
            share = a * wts[k] / total
            if land[nr, nc]:
                acc[nr, nc] += share
            else:
                to_water += share
    return acc, to_water, stranded, n_interior_sinks


def route_flow(z, land, dx, dy, cell_area, eps=FILL_EPS_M):
    """Fill + FD8 upslope-area accumulation exactly as Phase 1 Step 7."""
    zf, n_raised = priority_flood_eps(z, ~land, eps)
    flat = np.flatnonzero(land.ravel())
    order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
    acc, to_water, stranded, n_sinks = quinn_fd8(zf, land, order, cell_area, dx, dy)
    return zf, acc, {"n_raised": int(n_raised), "to_water_m2": to_water,
                     "stranded_m2": stranded, "n_interior_sinks": int(n_sinks)}


# ---------------------------------------------------------------------------
# Masks and label aggregation
# ---------------------------------------------------------------------------
def dilate(mask, px):
    from scipy import ndimage
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=px)


def remove_small_patches(mask, min_px):
    """Drop 8-connected patches smaller than min_px; returns (mask, n_removed)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask, 0
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab], int((~keep[1:]).sum())


def to_30m_fraction(mask20, h30, w30):
    """Exact area fraction of a 20 m mask in every 30 m cell (Step 8).
    20 m px -> 2x2 sub-pixels of 10 m; 30 m cell = 3x3 sub-pixels, with the
    30 m grid starting one sub-pixel up-left of the 20 m grid."""
    sub = np.repeat(np.repeat(mask20.astype(np.float32), 2, axis=0), 2, axis=1)
    need_h, need_w = 3 * h30, 3 * w30
    canvas = np.zeros((need_h, need_w), np.float32)
    h = min(sub.shape[0], need_h - 1)
    w = min(sub.shape[1], need_w - 1)
    canvas[1:1 + h, 1:1 + w] = sub[:h, :w]
    return canvas.reshape(h30, 3, w30, 3).mean(axis=(1, 3))


def check_20_30_commensurate(t20, t30):
    """Assert the Phase 1 20 m inventory grid nests exactly in the 30 m grid."""
    ratio = t30.a / t20.a
    off_x = (t20.c - t30.c) / t20.a
    off_y = (t30.f - t20.f) / abs(t20.e)
    assert abs(ratio - 1.5) < 1e-9 and abs(t30.e / t20.e - 1.5) < 1e-9, ratio
    assert abs(off_x - 0.5) < 1e-6 and abs(off_y - 0.5) < 1e-6, (off_x, off_y)


# ---------------------------------------------------------------------------
# Sentinel-1 flood mapping, Step 5
# ---------------------------------------------------------------------------
LEE_WINDOW = 5
OTSU_RANGE_DB = (-30.0, 5.0)
PLAUSIBLE_WATER_THRESHOLD_DB = (-24.0, -12.0)
SLOPE_MAX_DEG = 5.0
MIN_PATCH_PX = 5          # ~0.2 ha at 20 m


def db_to_linear(db):
    return np.power(10.0, db / 10.0)


def linear_to_db(lin):
    """Base paper Eq. 1: sigma0_dB = 10 * log10(sigma0)."""
    return 10.0 * np.log10(np.maximum(lin, 1e-10))


def ensure_db(arr, name, verbose=True):
    """GEE S1_GRD is already dB; only apply Eq. 1 if the data is clearly linear."""
    med = float(np.nanmedian(arr))
    if np.nanmin(arr) >= 0 and med < 1.0:
        if verbose:
            print(f"    {name}: values look LINEAR (median={med:.4f}) -> applying Eq. 1")
        return linear_to_db(arr), "converted_from_linear_with_eq1"
    if verbose:
        print(f"    {name}: values already in dB (median={med:.2f} dB)")
    return arr, "already_db_from_gee"


def lee_filter(intensity, size):
    """Classic Lee speckle filter on LINEAR intensity."""
    from scipy import ndimage
    mean = ndimage.uniform_filter(intensity, size)
    sq_mean = ndimage.uniform_filter(intensity ** 2, size)
    var = np.maximum(sq_mean - mean ** 2, 0)
    noise_var = np.mean(var)
    weight = var / (var + noise_var)
    return mean + weight * (intensity - mean)


def load_s1(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        profile = src.profile
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    arr[~valid] = np.nan
    return arr, valid, profile


def preprocess(arr_db_raw, valid, name, verbose=True):
    arr_db, db_status = ensure_db(arr_db_raw, name, verbose)
    lin = db_to_linear(np.where(valid, arr_db, np.nanmedian(arr_db)))
    filtered_db = linear_to_db(lee_filter(lin, LEE_WINDOW))
    filtered_db[~valid] = np.nan
    return filtered_db, db_status


def otsu_threshold(arr_db, valid):
    """Per-image Otsu on [-30, 5] dB, 512 bins. Returns the threshold, whether
    it is in the typical open-water range, and Otsu's separability
    eta = between-class variance / total variance (0..1, bimodality measure)."""
    from skimage.filters import threshold_otsu
    vals = arr_db[valid]
    vals = vals[(vals >= OTSU_RANGE_DB[0]) & (vals <= OTSU_RANGE_DB[1])]
    t = float(threshold_otsu(vals, nbins=512))
    lo, hi = PLAUSIBLE_WATER_THRESHOLD_DB
    w0 = float((vals < t).mean())
    m0 = float(vals[vals < t].mean()) if w0 > 0 else float("nan")
    m1 = float(vals[vals >= t].mean()) if w0 < 1 else float("nan")
    eta = w0 * (1 - w0) * (m0 - m1) ** 2 / float(vals.var())
    return t, bool(lo <= t <= hi), float(eta), {"water_share": w0, "mean_water_db": m0, "mean_land_db": m1}


# ---------------------------------------------------------------------------
# NDEM and agreement (GTI), Step 6
# ---------------------------------------------------------------------------
TOL_PX = 2                 # 40 m tolerance at 20 m
MIN_PATCH_EVAL_PX = 10     # patches counted for GTI_patch


def load_ndem():
    """NDEM (NRSC/ISRO) polygons over the AOI, with a dd-mm-yyyy 'date' column."""
    import geopandas as gpd
    from shapely.geometry import box
    nd = gpd.read_parquet(P1_NDEM)
    if nd.crs is None or nd.crs.to_epsg() != 4326:
        nd = nd.set_crs("EPSG:4326", allow_override=True)
    nd = nd[nd.intersects(box(*BBOX))].copy()
    nd["date"] = nd["from_time"].astype(str).str[:10]
    return nd


def ndem_mask(ndem, dates, shape, transform):
    from rasterio.features import rasterize
    sel = ndem[ndem["date"].isin(dates)]
    if not len(sel):
        return np.zeros(shape, bool), 0
    m = rasterize([(g, 1) for g in sel.geometry], out_shape=shape, transform=transform,
                  fill=0, dtype="uint8").astype(bool)
    return m, int(len(sel))


def agreement(ours, ref, domain, row_area):
    """Step 6 agreement: GTI (pixel, 40 m tolerant, patch), random baseline,
    lift and the full contingency table."""
    from scipy import ndimage
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

    r_tol = dilate(ref, TOL_PX) & domain
    gti_tol = int((o & r_tol).sum()) / max(int(o.sum()), 1)
    prevalence_tol = int(r_tol.sum()) / n

    lab, _ = ndimage.label(o, structure=np.ones((3, 3)))
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


# ---------------------------------------------------------------------------
# Modelling, Step 9
# ---------------------------------------------------------------------------
THRESHOLD = 0.5
LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 20, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "force_row_wise": True,
    "num_threads": 4, "verbose": -1,
}
META_COLUMNS = {"lon", "lat", "x_utm", "y_utm", "row", "col", "block_id",
                "otsu_flood_frac", "confirmed_flood_frac", "sample_id"}


def metrics(y_true, p, threshold=THRESHOLD):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, confusion_matrix)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
            "auc": roc_auc_score(y_true, p), "specificity": tn / (tn + fp),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def rounded(m, nd=4):
    return {k: (round(float(v), nd) if not isinstance(v, dict) else v) for k, v in m.items()}


def cv_summary(fold_metrics):
    keys = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    return {k: {"mean": round(float(np.mean([m[k] for m in fold_metrics])), 4),
                "sd": round(float(np.std([m[k] for m in fold_metrics])), 4)} for k in keys}


def block_groups(df, block_m):
    """Square UTM block id of every sample (Step 9 spatial CV)."""
    return ((df["x_utm"] // block_m).astype(int).astype(str) + "_"
            + (df["y_utm"] // block_m).astype(int).astype(str)).to_numpy()


# ---------------------------------------------------------------------------
# Esri World Imagery basemap (XYZ tiles, cached, warped to EPSG:4326)
# ---------------------------------------------------------------------------
MERC = 20037508.342789244


def _tile_xy(lon, lat, z):
    n = 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def _get_tile(z, x, y):
    import requests
    from PIL import Image
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
    """RGB array (EPSG:4326, extent=bounds) of Esri World Imagery."""
    import rasterio
    from rasterio.warp import reproject, Resampling
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
# Earth Engine download onto an exact grid
# ---------------------------------------------------------------------------
def ee_init():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    return ee


def ee_download_grid(img, transform, width, height, out_path, tiles=(1, 1), timeout=900, retries=3):
    """Download an ee.Image onto an exact (EPSG:4326) pixel grid, optionally in
    row x col tiles (GEE's per-request size / compute limits), and mosaic the
    tiles into one GeoTIFF that has exactly `transform` and (height, width)."""
    import io
    import requests
    import rasterio
    from rasterio.transform import Affine
    nty, ntx = tiles
    rows = np.linspace(0, height, nty + 1).astype(int)
    cols = np.linspace(0, width, ntx + 1).astype(int)
    out = None
    names = None
    for i in range(nty):
        for j in range(ntx):
            r0, r1, c0, c1 = rows[i], rows[i + 1], cols[j], cols[j + 1]
            t = transform * Affine.translation(c0, r0)
            params = {"crs": "EPSG:4326", "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                      "dimensions": f"{c1 - c0}x{r1 - r0}", "format": "GEO_TIFF"}
            last = None
            for attempt in range(retries):
                try:
                    url = img.getDownloadURL(params)
                    resp = requests.get(url, timeout=timeout)
                    resp.raise_for_status()
                    break
                except Exception as e:      # transient GEE / network errors
                    last = e
                    print(f"      [!] tile {i},{j} attempt {attempt + 1} failed: {str(e)[:160]}")
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"download failed for tile {i},{j}: {last}")
            with rasterio.open(io.BytesIO(resp.content)) as src:
                arr = src.read()
                assert src.width == c1 - c0 and src.height == r1 - r0, (src.shape, r1 - r0, c1 - c0)
                assert all(abs(x - y) < 1e-9 for x, y in zip(src.transform[:6], t[:6])), (src.transform, t)
                if out is None:
                    out = np.zeros((arr.shape[0], height, width), arr.dtype)
                    names = src.descriptions
            out[:, r0:r1, c0:c1] = arr
    prof = {"driver": "GTiff", "height": height, "width": width, "count": out.shape[0],
            "dtype": out.dtype, "crs": "EPSG:4326", "transform": transform, "compress": "lzw"}
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
        if names:
            for b, n in enumerate(names, 1):
                if n:
                    dst.set_band_description(b, n)
    return out, list(names or [])
