"""
Step 7 - Terrain and hydrological feature rasters
(Week 3, Phase1_Implementation_Plan).

Master grid (7.5). Every layer is written on the native FABDEM grid itself:
EPSG:4326, 1 arc-second (~30 m), 1300 x 1486 px, which already coincides with
the AOI bbox [76.15, 9.90, 76.55, 10.25]. The DEM is therefore never resampled
(elevation band == source file, bit for bit). Metric quantities are computed
with the true WGS84 cell size of every row (dx ~ 29.6 m, dy ~ 29.8 m), not a
degree-to-metre constant.

Analysis domain. FABDEM is exactly 0 m over the Arabian Sea, the Vembanad
backwater and the Kochi harbour channels; that surface is one zero-elevation
body connected to the grid border. Those pixels are "open water": they act as
the drainage outlet for flow routing and every feature is NaN there. All other
pixels (land, including the real below-sea-level polders, down to -6 m) form
the analysis domain and must carry a finite value in every layer.

7.1 slope / aspect / curvature
    slope, aspect   Horn (1981) 3x3 gradient.  aspect: compass degrees of the
                    downslope direction (0 = N, 90 = E); -1 where the gradient
                    is exactly zero (flat).
    curvature       Zevenbergen & Thorne (1987) quadratic surface, in 1/100 m
                    (the ArcGIS unit). One sign convention for all three:
                    positive = convex (ridges, shoulders, spurs),
                    negative = concave (valleys, hollows, footslopes).
                      general  -(z_xx + z_yy)
                      profile  along the slope line  (convex: flow accelerates)
                      plan     across the slope      (concave: flow converges)
    Only plan + profile go into the model stack: in this formulation general
    curvature is exactly plan + profile (VIF ~ 10^4 when all three are kept),
    so it is written as an auxiliary layer.
    Border pixels use odd-reflection padding (linear extrapolation), so slope
    is exact on a plane right up to the grid edge.

7.2 TWI = ln(a / tan b)   (Beven & Kirkby 1979)
    1. depressions filled with Priority-Flood+epsilon (Barnes et al. 2014),
       float64, fixed eps = 1e-6 m, outlets = grid border + open-water body.
       Every other cell then has a strictly lower neighbour (checked).
       (Barnes' nextafter() step is subnormal on 0 m flats -- 5e-324 m -- and
       the drop underflows to 0 once divided by the cell length, stranding
       ~14 000 coastal cells; the fixed eps raises flats by mm at most.)
    2. flow accumulation with the multiple-flow-direction FD8 of Quinn et al.
       (1991): weights tan b_i * L_i (L = 0.5 cardinal, 0.354 diagonal), cells
       processed from highest to lowest filled elevation. Mass balance is
       checked: all land area must end in an outlet.
    3. a = upslope area / contour width (width = cell size); tan b from the
       Horn slope of the unfilled DEM, floored at 0.001 so flats stay finite.
    Only the DEM inside the AOI is routed, so the Periyar's upstream catchment
    (outside the AOI) is not counted; the size of that truncation is reported.

7.3 distances (metres)
    Euclidean distance from each pixel centre to the nearest HydroRIVERS line
    (all orders, as in the plan), to the nearest *major* river (Strahler >= 4,
    upland area >= 252 km2: the Periyar main stem and its largest tributaries)
    and to the nearest OSM drive-network edge, computed in UTM 43N with a KD-tree
    over line vertices densified to <= 5 m (error <= 2.5 m). Rivers were
    clipped with a 0.1 deg buffer around the AOI in Step 3, so there is no
    border effect; roads were clipped to the AOI itself, so the share of
    pixels where a road just outside the AOI could be closer is reported.
    Why two river distances: in this wet region HydroRIVERS keeps streams
    down to 1.9 km2 catchment (0.12 m3/s mean flow), so "nearest river" is
    within 1-2 km almost everywhere and mostly means a first-order stream;
    the 2018 flooding followed the main stem.

7.4 drainage density (km / km^2)
    HydroRIVERS line length inside a 2 km radius circle around each pixel,
    divided by the circle area (line-density method), computed on a 30 m UTM
    lattice that extends 2 km beyond the AOI so edge pixels see full circles.

    rainfall (needed for the Step 8 stack): CHIRPS 2017-2023 mean annual total.
    CHIRPS stores ocean cells as 0 mm; those are replaced by the nearest valid
    cell *before* bilinear interpolation, otherwise coastal pixels would be
    dragged toward 0.

7.5 alignment checkpoint: every layer is re-opened from disk and compared with
    the DEM (CRS, transform, shape), NaN counts inside / outside the domain,
    physical ranges; plus analytic self-tests of the terrain operators, the
    flow-routing invariants, a HydroRIVERS vs DEM-channel position check, and
    the feature correlation / VIF table that Step 8-9 will need.

Outputs
    data/processed/features/<layer>.tif        float32, NaN = open water
    data/processed/features/analysis_domain.tif uint8 1 = land, 0 = open water
    data/processed/features/feature_stack.tif  all model features, named bands
    data/processed/features/features_metadata.json
    outputs/metrics/step7_feature_stack_checkpoint.json
    outputs/maps/step7_feature_stack_overview.png
    outputs/maps/step7_river_alignment_check.png
    outputs/maps/step7_feature_correlation.png
"""
import os
import sys
import json
import time
import heapq

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
import geopandas as gpd
import shapely
from shapely.geometry import box
from pyproj import Transformer
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(phase1_root)

FABDEM = "data/raw/fabdem_30m_aoi.tif"
RIVERS = "data/raw/hydrosheds/periyar_rivers_clip.gpkg"
ROADS = "data/raw/osm_roads_aoi.gpkg"
CHIRPS = "data/raw/chirps_mean_annual_rainfall_2017_2023.tif"
S1_CLASSES_2018 = "data/processed/flood_inventory/water_classes_2018_20180821.tif"
OUT_DIR = "data/processed/features"
MAP_DIR = "outputs/maps"
METRICS_DIR = "outputs/metrics"
for d in (OUT_DIR, MAP_DIR, METRICS_DIR):
    os.makedirs(d, exist_ok=True)

BBOX = [76.15, 9.90, 76.55, 10.25]
UTM = "EPSG:32643"
TAN_SLOPE_MIN = 0.001        # TWI floor for tan(beta) (~0.057 deg)
FILL_EPS_M = 1e-6            # Priority-Flood+eps step (nextafter() is subnormal at 0 m)
DENSIFY_M = 5.0              # vertex spacing for distance KD-trees
DD_RADIUS_M = 2000.0         # drainage-density search radius
DD_CELL_M = 30.0             # drainage-density lattice
CHANNEL_KM2 = 5.0            # DEM channel threshold for the river-alignment check
MAJOR_STRAHLER = 4           # HydroRIVERS order for dist_to_major_river_m
SAMPLE_N = 200_000           # pixels for correlation / VIF
RNG = np.random.default_rng(42)

# Model features, in stack band order: (name, unit, description)
FEATURES = [
    ("elevation_m", "m", "FABDEM v1-2 bare-earth elevation (unchanged source values)"),
    ("slope_deg", "deg", "Horn (1981) slope"),
    ("aspect_deg", "deg", "downslope compass direction 0-360 (0 = N); -1 = flat"),
    ("curvature_plan", "1/100 m", "plan (across-slope) curvature, + divergent / - convergent"),
    ("curvature_profile", "1/100 m", "profile (along-slope) curvature, + convex / - concave"),
    ("twi", "-", "ln(a / tan b), Priority-Flood+eps fill, Quinn FD8 accumulation"),
    ("dist_to_river_m", "m", "Euclidean distance to nearest HydroRIVERS v10 line (all orders)"),
    ("dist_to_major_river_m", "m", "Euclidean distance to nearest HydroRIVERS line with Strahler >= 4"),
    ("dist_to_road_m", "m", "Euclidean distance to nearest OSM drive-network edge"),
    ("drainage_density_km_km2", "km/km2", "HydroRIVERS length within 2 km radius / circle area"),
    ("rainfall_mean_annual_mm", "mm/yr", "CHIRPS v2 mean annual total 2017-2023, bilinear"),
]

print("=" * 80)
print("STEP 7: TERRAIN AND HYDROLOGICAL FEATURE RASTERS")
print("=" * 80)
t_start = time.time()


# ---------------------------------------------------------------------------
# Terrain operators
# ---------------------------------------------------------------------------
def row_cell_sizes_m(transform, height):
    """True WGS84 cell width/height (m) and area (m^2) for each grid row."""
    a, e2 = 6378137.0, 6.69437999014e-3
    lat = np.radians(transform.f + (np.arange(height) + 0.5) * transform.e)
    w = 1 - e2 * np.sin(lat) ** 2
    m_rad = a * (1 - e2) / w ** 1.5          # meridional radius of curvature
    n_rad = a / np.sqrt(w)                    # prime-vertical radius
    dy = m_rad * np.radians(abs(transform.e))
    dx = n_rad * np.cos(lat) * np.radians(abs(transform.a))
    return dx, dy, dx * dy


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


def curvatures(z, dx, dy):
    """Zevenbergen & Thorne (1987); returns general, plan, profile in 1/100 m,
    all with + = convex, - = concave."""
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = neighbourhood(z)
    dx = dx[:, None]
    dy = dy[:, None]
    D = ((z4 + z6) / 2 - z5) / dx ** 2              # z_xx / 2
    E = ((z2 + z8) / 2 - z5) / dy ** 2              # z_yy / 2
    F = (-z1 + z3 + z7 - z9) / (4 * dx * dy)        # z_xy
    G = (z6 - z4) / (2 * dx)                        # z_x
    H = (z2 - z8) / (2 * dy)                        # z_y
    g2 = G ** 2 + H ** 2
    safe = np.where(g2 > 0, g2, 1.0)
    general = -2 * (D + E) * 100
    profile = np.where(g2 > 0, -2 * (D * G ** 2 + E * H ** 2 + F * G * H) / safe, 0.0) * 100
    plan = np.where(g2 > 0, -2 * (D * H ** 2 + E * G ** 2 - F * G * H) / safe, 0.0) * 100
    return general, plan, profile


# ---------------------------------------------------------------------------
# Flow routing (numba)
# ---------------------------------------------------------------------------
DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)


@njit(cache=True)
def priority_flood_eps(z, outlet, eps):
    """Barnes et al. (2014) Priority-Flood+epsilon. Seeds: grid border and
    outlet (open-water) cells. eps is a fixed step: Barnes' nextafter() gives
    subnormal steps (5e-324) on 0 m flats, which underflow to a zero drop once
    divided by the cell length. Returns the filled float64 DEM and the number
    of raised cells."""
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
    """Quinn et al. (1991) FD8 accumulation of upslope area (m^2).
    order: flat indices of land cells, highest filled elevation first.
    Returns acc, area delivered to open water, area stranded in cells with no
    lower neighbour (border only), and the number of non-border land cells
    without a strictly lower neighbour (must be 0)."""
    H, W = zf.shape
    acc = np.zeros((H, W), np.float64)
    for r in range(H):
        for c in range(W):
            if land[r, c]:
                acc[r, c] = cell_area[r]
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


def route_flow(z, land, dx, dy, cell_area):
    zf, n_raised = priority_flood_eps(z, ~land, FILL_EPS_M)
    flat = np.flatnonzero(land.ravel())
    order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
    acc, to_water, stranded, n_sinks = quinn_fd8(zf, land, order, cell_area, dx, dy)
    return zf, acc, {"n_raised": int(n_raised), "to_water_m2": to_water,
                     "stranded_m2": stranded, "n_interior_sinks": int(n_sinks)}


# ---------------------------------------------------------------------------
# Analytic self-tests of the operators (run on synthetic surfaces first)
# ---------------------------------------------------------------------------
def self_tests():
    print("\n[1/8] Self-tests of terrain operators on analytic surfaces")
    n, L = 41, 30.0
    dx = np.full(n, L)
    dy = np.full(n, L)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    x = (xx - n // 2) * L                   # east +
    y = -(yy - n // 2) * L                  # north +
    c = n // 2
    res = {}

    # plane rising to the east -> faces west, slope atan(0.1), exact to the edge
    p, q = horn_gradient(0.1 * x, dx, dy)
    s, asp = slope_aspect(p, q)
    res["plane_slope_err_deg"] = float(np.abs(s - np.degrees(np.arctan(0.1))).max())
    res["plane_aspect_deg"] = float(asp[c, c])            # expect 270
    p, q = horn_gradient(-0.05 * x - 0.05 * y, dx, dy)   # falls to NE -> aspect 45
    res["plane_ne_aspect_deg"] = float(slope_aspect(p, q)[1][c, c])

    k = 1e-3
    g, _, _ = curvatures(-k * (x ** 2 + y ** 2), dx, dy)    # dome: convex
    res["dome_general_expect_+0.4"] = float(g[c, c])        # -(z_xx+z_yy)*100 = 4k*100
    u = (x + y) / np.sqrt(2)                                 # diagonal ridge: needs F
    _, pl, pr = curvatures(-k * u ** 2, dx, dy)
    res["diag_ridge_profile_expect_+0.2"] = float(pr[c - 3, c + 5])   # off-crest (u = 8L/sqrt2)
    res["diag_ridge_plan_expect_0"] = float(pl[c - 3, c + 5])
    _, pl, _ = curvatures(0.05 * x - k * y ** 2, dx, dy)    # spur (divergent) -> plan > 0
    res["spur_plan_expect_+0.2"] = float(pl[c, c])
    _, pl, _ = curvatures(0.05 * x + k * y ** 2, dx, dy)    # valley (convergent) -> plan < 0
    res["valley_plan_expect_-0.2"] = float(pl[c, c])

    # inclined plane falling to the south: FD8 specific area at row i = (i+1)*L
    # (checked on rows the side-border influence cannot reach yet: i < n//2)
    z = 1000.0 - 0.02 * (yy * L)
    land = np.ones_like(z, dtype=bool)
    _, acc, fl = route_flow(z, land, dx, dy, np.full(n, L * L))
    a_spec = acc[:, c] / L
    res["plane_fd8_specific_area_max_rel_err"] = float(np.abs(a_spec[:c] / ((np.arange(c) + 1) * L) - 1).max())
    res["plane_fd8_mass_balance_rel_err"] = float(abs(fl["stranded_m2"] + fl["to_water_m2"] - n * n * L * L) / (n * n * L * L))

    # closed pit must be filled and drained
    zpit = 10.0 + 0.001 * (xx + yy)
    zpit[15:25, 15:25] = 5.0
    zf, n_raised = priority_flood_eps(zpit, np.zeros_like(land), FILL_EPS_M)
    res["pit_cells_raised"] = int(n_raised)
    _, _, fl = route_flow(zpit, land, dx, dy, np.full(n, L * L))
    res["pit_interior_sinks_after_fill"] = fl["n_interior_sinks"]

    # 0 m flat (the coastal case): every interior cell must still drain
    _, _, fl = route_flow(np.zeros((n, n), np.float32), land, dx, dy, np.full(n, L * L))
    res["zero_flat_interior_sinks"] = fl["n_interior_sinks"]

    checks = {
        "plane slope exact incl. border": res["plane_slope_err_deg"] < 1e-9,
        "aspect west-facing = 270": abs(res["plane_aspect_deg"] - 270) < 1e-9,
        "aspect NE-facing = 45": abs(res["plane_ne_aspect_deg"] - 45) < 1e-9,
        "dome general curvature = +0.4": abs(res["dome_general_expect_+0.4"] - 0.4) < 1e-9,
        "diagonal ridge profile = +0.2 (F term)": abs(res["diag_ridge_profile_expect_+0.2"] - 0.2) < 1e-9,
        "diagonal ridge plan = 0": abs(res["diag_ridge_plan_expect_0"]) < 1e-9,
        "spur plan > 0, valley plan < 0": abs(res["spur_plan_expect_+0.2"] - 0.2) < 1e-9
                                            and abs(res["valley_plan_expect_-0.2"] + 0.2) < 1e-9,
        "FD8 plane specific area = (i+1)L": res["plane_fd8_specific_area_max_rel_err"] < 1e-9,
        "FD8 mass balance": res["plane_fd8_mass_balance_rel_err"] < 1e-12,
        "pit filled, no interior sinks": res["pit_cells_raised"] == 100 and res["pit_interior_sinks_after_fill"] == 0,
        "0 m flat drains (no underflow)": res["zero_flat_interior_sinks"] == 0,
    }
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit("Self-tests failed - terrain operators are wrong, stopping.")
    return {"values": res, "checks": checks, "all_pass": True}


results = {"step": 7, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
results["self_tests"] = self_tests()

# ---------------------------------------------------------------------------
# Master grid, analysis domain
# ---------------------------------------------------------------------------
print("\n[2/8] Master grid (native FABDEM) and analysis domain")
with rasterio.open(FABDEM) as src:
    dem = src.read(1)
    profile = src.profile.copy()
    T = src.transform
    CRS = src.crs
HGT, WID = dem.shape
dx, dy, cell_area = row_cell_sizes_m(T, HGT)
print(f"    grid {HGT} x {WID}, {CRS}, res {T.a:.8f} deg; dx {dx.min():.2f}-{dx.max():.2f} m, "
      f"dy {dy.mean():.2f} m")

zero = dem == 0
lab, n_lab = ndimage.label(zero, structure=np.ones((3, 3)))
border_labels = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
border_labels = border_labels[border_labels > 0]
water = np.isin(lab, border_labels)
land = ~water
inland_zero = zero & land
km2 = lambda m: float((m.sum(axis=1) * cell_area).sum() / 1e6)
print(f"    open water (border-connected 0 m surface): {km2(water):.1f} km2 "
      f"({100 * water.mean():.1f}% of grid, {len(border_labels)} bodies)")
print(f"    land domain: {km2(land):.1f} km2; inland 0 m pixels kept as land: {km2(inland_zero):.2f} km2; "
      f"below-sea-level land: {km2(land & (dem < 0)):.2f} km2")

# Independent sanity check: Step 5 SAR water on the dry reference inside our open-water mask
with rasterio.open(S1_CLASSES_2018) as src:
    s1cls = np.full(dem.shape, 255, np.uint8)
    reproject(rasterio.band(src, 1), s1cls, dst_transform=T, dst_crs=CRS,
              resampling=Resampling.mode, src_nodata=255, dst_nodata=255)
s1_valid = s1cls != 255
water_s1_agree = float(((s1cls == 2) & water & s1_valid).sum() / max((water & s1_valid).sum(), 1))
print(f"    {100 * water_s1_agree:.1f}% of DEM open-water pixels are also dry-season water in Step 5 SAR (2018)")
results["domain"] = {
    "grid": {"crs": str(CRS), "height": HGT, "width": WID, "transform": list(T)[:6],
             "cell_dx_m_range": [round(float(dx.min()), 3), round(float(dx.max()), 3)],
             "cell_dy_m": round(float(dy.mean()), 3)},
    "open_water_rule": "FABDEM == 0 m, 8-connected components touching the grid border",
    "open_water_km2": round(km2(water), 2),
    "open_water_share_of_grid": round(float(water.mean()), 4),
    "land_km2": round(km2(land), 2),
    "inland_zero_elevation_kept_as_land_km2": round(km2(inland_zero), 3),
    "below_sea_level_land_km2": round(km2(land & (dem < 0)), 3),
    "open_water_confirmed_as_s1_dry_season_water_share": round(water_s1_agree, 4),
}

# ---------------------------------------------------------------------------
# 7.1 slope, aspect, curvature
# ---------------------------------------------------------------------------
print("\n[3/8] 7.1 Slope, aspect, curvature")
p, q = horn_gradient(dem, dx, dy)
slope, aspect = slope_aspect(p, q)
curv_general, curv_plan, curv_profile = curvatures(dem, dx, dy)
tan_b = np.hypot(p, q)
print(f"    slope land median {np.median(slope[land]):.2f} deg, p99 {np.percentile(slope[land], 99):.1f} deg; "
      f"flat (zero-gradient) land pixels: {int(((aspect == -1) & land).sum())}")

# ---------------------------------------------------------------------------
# 7.2 TWI
# ---------------------------------------------------------------------------
print("\n[4/8] 7.2 Topographic Wetness Index (Priority-Flood+eps, Quinn FD8)")
t0 = time.time()
zf, acc, flow = route_flow(dem, land, dx, dy, cell_area)
land_area_m2 = float((land.sum(axis=1) * cell_area).sum())
mass_err = abs(flow["to_water_m2"] + flow["stranded_m2"] - land_area_m2) / land_area_m2
width = np.sqrt(dx * dy)[:, None]
with np.errstate(divide="ignore"):          # open-water cells (acc = 0) are masked later
    twi = np.log((acc / width) / np.maximum(tan_b, TAN_SLOPE_MIN))
acc_km2 = acc / 1e6
raised = land & (zf > dem.astype(np.float64))
print(f"    routed in {time.time() - t0:.1f} s; cells raised by fill: {flow['n_raised']} "
      f"({km2(raised):.1f} km2); interior sinks after fill: {flow['n_interior_sinks']}")
print(f"    mass balance: to open water {flow['to_water_m2'] / 1e6:.1f} km2 + off-grid via border "
      f"{flow['stranded_m2'] / 1e6:.1f} km2 = land {land_area_m2 / 1e6:.1f} km2 (rel err {mass_err:.1e})")
print(f"    TWI land range {np.nanmin(twi[land]):.2f} - {np.nanmax(twi[land]):.2f}, median {np.median(twi[land]):.2f}; "
      f"max accumulation {acc_km2[land].max():.0f} km2")

rivers = gpd.read_file(RIVERS)
aoi_box = box(*BBOX)
entering = rivers[rivers.intersects(aoi_box.boundary)]
flow_trunc = {
    "hydrorivers_max_upland_km2_crossing_aoi_border": round(float(entering["UPLAND_SKM"].max()), 1),
    "dem_max_accumulation_km2_inside_aoi": round(float(acc_km2[land].max()), 1),
}
print(f"    upstream truncation: HydroRIVERS upland area at the AOI border up to "
      f"{flow_trunc['hydrorivers_max_upland_km2_crossing_aoi_border']:.0f} km2 "
      f"vs {flow_trunc['dem_max_accumulation_km2_inside_aoi']:.0f} km2 routed inside the AOI")
results["twi"] = {
    "fill": f"Priority-Flood+epsilon (Barnes et al. 2014), float64, eps = {FILL_EPS_M} m, outlets = border + open water",
    "routing": "Quinn et al. (1991) FD8, weights tan(b)*L, L = 0.5 / 0.354",
    "tan_beta": f"Horn slope of the unfilled DEM, floored at {TAN_SLOPE_MIN}",
    "cells_raised_by_fill": flow["n_raised"],
    "area_raised_by_fill_km2": round(km2(raised), 2),
    "interior_cells_without_lower_neighbour": flow["n_interior_sinks"],
    "mass_balance_rel_error": float(mass_err),
    "area_to_open_water_km2": round(flow["to_water_m2"] / 1e6, 2),
    "area_leaving_via_grid_border_km2": round(flow["stranded_m2"] / 1e6, 2),
    "pixels_at_tan_beta_floor_share": round(float((tan_b[land] < TAN_SLOPE_MIN).mean()), 4),
    "upstream_truncation": flow_trunc,
}

# ---------------------------------------------------------------------------
# 7.3 distances, 7.4 drainage density
# ---------------------------------------------------------------------------
print("\n[5/8] 7.3 Distance to river / road, 7.4 drainage density")
to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
cols = T.c + (np.arange(WID) + 0.5) * T.a
rows = T.f + (np.arange(HGT) + 0.5) * T.e
lon_g, lat_g = np.meshgrid(cols, rows)
ex, ny = to_utm.transform(lon_g.ravel(), lat_g.ravel())
centres = np.column_stack([ex, ny])


def densified_vertices(gdf, spacing):
    g = shapely.segmentize(gdf.to_crs(UTM).geometry.values, spacing)
    return shapely.get_coordinates(g)


def distance_raster(gdf, name):
    t0 = time.time()
    pts = densified_vertices(gdf, DENSIFY_M)
    d, _ = cKDTree(pts).query(centres, workers=-1)
    print(f"    {name}: {len(gdf)} lines, {len(pts):,} vertices, {time.time() - t0:.1f} s")
    return d.reshape(HGT, WID)


dist_river = distance_raster(rivers, "rivers (HydroRIVERS, all orders)")
major = rivers[rivers["ORD_STRA"] >= MAJOR_STRAHLER]
dist_major = distance_raster(major, f"major rivers (Strahler >= {MAJOR_STRAHLER})")
roads = gpd.read_file(ROADS, layer="edges")
dist_road = distance_raster(roads, "roads (OSM drive)")

# roads were clipped to the AOI: where could an outside road be closer?
aoi_utm = gpd.GeoSeries([aoi_box], crs="EPSG:4326").to_crs(UTM).iloc[0]
d_edge = shapely.distance(shapely.points(centres), aoi_utm.boundary).reshape(HGT, WID)
road_edge_risk = float((dist_road[land] > d_edge[land]).mean())
print(f"    land pixels where a road outside the AOI could be nearer: {100 * road_edge_risk:.2f}%")

# drainage density on a padded 30 m UTM lattice
seg_g = shapely.segmentize(rivers.explode(index_parts=False).to_crs(UTM).geometry.values, DENSIFY_M)
coords, gi = shapely.get_coordinates(seg_g, return_index=True)
same = gi[1:] == gi[:-1]
mid = ((coords[1:] + coords[:-1]) / 2)[same]
seg_len = np.hypot(*(coords[1:] - coords[:-1]).T)[same]
x0 = centres[:, 0].min() - DD_RADIUS_M - DD_CELL_M
y1 = centres[:, 1].max() + DD_RADIUS_M + DD_CELL_M
nx = int(np.ceil((centres[:, 0].max() + DD_RADIUS_M + DD_CELL_M - x0) / DD_CELL_M))
ny_ = int(np.ceil((y1 - (centres[:, 1].min() - DD_RADIUS_M - DD_CELL_M)) / DD_CELL_M))
ci = np.floor((mid[:, 0] - x0) / DD_CELL_M).astype(int)
ri = np.floor((y1 - mid[:, 1]) / DD_CELL_M).astype(int)
ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny_)
length_grid = np.bincount(ri[ok] * nx + ci[ok], weights=seg_len[ok], minlength=nx * ny_).reshape(ny_, nx)
rk = int(np.ceil(DD_RADIUS_M / DD_CELL_M))
ky, kx = np.mgrid[-rk:rk + 1, -rk:rk + 1]
disk = ((kx * DD_CELL_M) ** 2 + (ky * DD_CELL_M) ** 2 <= DD_RADIUS_M ** 2).astype(float)
disk_area_km2 = disk.sum() * DD_CELL_M ** 2 / 1e6
dd_lattice = np.clip(fftconvolve(length_grid, disk, mode="same"), 0, None) / 1000.0 / disk_area_km2
qc = np.floor((centres[:, 0] - x0) / DD_CELL_M).astype(int)
qr = np.floor((y1 - centres[:, 1]) / DD_CELL_M).astype(int)
drainage_density = dd_lattice[qr, qc].reshape(HGT, WID)
dd_check = float(length_grid.sum() / 1000)
print(f"    drainage density: {dd_check:.0f} km of river on lattice, AOI land mean "
      f"{drainage_density[land].mean():.3f} km/km2, max {drainage_density[land].max():.3f}")
results["distances"] = {
    "method": f"KD-tree on vertices densified to <= {DENSIFY_M} m in {UTM}; max vertex error {DENSIFY_M / 2} m",
    "n_river_lines": int(len(rivers)),
    "major_river_rule": f"ORD_STRA >= {MAJOR_STRAHLER}",
    "n_major_river_lines": int(len(major)),
    "major_river_min_upland_km2": round(float(major["UPLAND_SKM"].min()), 1),
    "land_share_within_1km_of_any_river": round(float((dist_river[land] <= 1000).mean()), 4),
    "land_share_within_1km_of_major_river": round(float((dist_major[land] <= 1000).mean()), 4),
    "n_road_edges": int(len(roads)),
    "road_border_effect_share_of_land": round(road_edge_risk, 4),
}
results["drainage_density"] = {
    "method": f"HydroRIVERS line length within {DD_RADIUS_M:.0f} m radius / circle area, "
              f"{DD_CELL_M:.0f} m UTM lattice padded by the radius",
    "disk_area_km2": round(disk_area_km2, 4),
    "river_km_on_lattice": round(dd_check, 1),
}

# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------
print("\n[6/8] Rainfall (CHIRPS mean annual, ocean cells filled before bilinear)")
with rasterio.open(CHIRPS) as src:
    ch = src.read(1).astype(np.float64)
    tc = src.transform
ch_valid = ch > 0
fill_idx = ndimage.distance_transform_edt(~ch_valid, return_distances=False, return_indices=True)
ch_filled = ch[tuple(fill_idx)]
c_lon = tc.c + (np.arange(ch.shape[1]) + 0.5) * tc.a
c_lat = tc.f + (np.arange(ch.shape[0]) + 0.5) * tc.e
interp = RegularGridInterpolator((c_lat[::-1], c_lon), ch_filled[::-1], method="linear")
q_lat = np.clip(lat_g, c_lat.min(), c_lat.max())
q_lon = np.clip(lon_g, c_lon.min(), c_lon.max())
rainfall = interp(np.column_stack([q_lat.ravel(), q_lon.ravel()])).reshape(HGT, WID)
# which land pixels sit in a CHIRPS no-data (ocean) cell and therefore use filled values
cr = np.clip(((lat_g - tc.f) / tc.e).astype(int), 0, ch.shape[0] - 1)
cc = np.clip(((lon_g - tc.c) / tc.a).astype(int), 0, ch.shape[1] - 1)
in_filled_cell = land & ~ch_valid[cr, cc]
print(f"    CHIRPS valid cells {int(ch_valid.sum())}/{ch.size}, range {ch[ch_valid].min():.0f}-{ch[ch_valid].max():.0f} mm; "
      f"land pixels inside a no-data cell: {100 * in_filled_cell[land].mean():.2f}%")
results["rainfall"] = {
    "source": CHIRPS,
    "chirps_valid_cells": int(ch_valid.sum()),
    "chirps_nodata_cells_(ocean,_stored_as_0)": int((~ch_valid).sum()),
    "valid_range_mm": [round(float(ch[ch_valid].min()), 1), round(float(ch[ch_valid].max()), 1)],
    "land_pixels_in_nodata_cell_share": round(float(in_filled_cell[land].mean()), 4),
    "note": "5.5 km source: smooth, near-linear in location - treat as a location proxy in Step 9 importance.",
}

# ---------------------------------------------------------------------------
# Write rasters
# ---------------------------------------------------------------------------
print("\n[7/8] Writing rasters")
layers = {
    "elevation_m": dem.astype(np.float64),
    "slope_deg": slope,
    "aspect_deg": aspect,
    "curvature_plan": curv_plan,
    "curvature_profile": curv_profile,
    "twi": twi,
    "dist_to_river_m": dist_river,
    "dist_to_major_river_m": dist_major,
    "dist_to_road_m": dist_road,
    "drainage_density_km_km2": drainage_density,
    "rainfall_mean_annual_mm": rainfall,
}
aux = {"flow_accumulation_km2": acc_km2, "curvature_general": curv_general}
fprof = profile.copy()
fprof.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate",
             predictor=3, tiled=True, blockxsize=256, blockysize=256)
paths = {}
for name, arr in {**layers, **aux}.items():
    out = np.where(land, arr, np.nan).astype(np.float32)
    path = f"{OUT_DIR}/{name}.tif"
    with rasterio.open(path, "w", **fprof) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, name)
    paths[name] = path
dprof = profile.copy()
dprof.update(driver="GTiff", dtype="uint8", count=1, nodata=None, compress="deflate")
with rasterio.open(f"{OUT_DIR}/analysis_domain.tif", "w", **dprof) as dst:
    dst.write(land.astype(np.uint8), 1)
    dst.set_band_description(1, "1 = land (analysis domain), 0 = open water")
sprof = fprof.copy()
sprof.update(count=len(FEATURES), interleave="band")
stack_path = f"{OUT_DIR}/feature_stack.tif"
with rasterio.open(stack_path, "w", **sprof) as dst:
    for b, (name, unit, _) in enumerate(FEATURES, start=1):
        dst.write(np.where(land, layers[name], np.nan).astype(np.float32), b)
        dst.set_band_description(b, name)
        dst.update_tags(b, unit=unit)
print(f"    {len(paths)} single-layer GeoTIFFs + analysis_domain.tif + {stack_path} ({len(FEATURES)} bands)")

# ---------------------------------------------------------------------------
# 7.5 Checkpoint: re-open everything from disk
# ---------------------------------------------------------------------------
print("\n[8/8] 7.5 Alignment checkpoint (re-read from disk)")
RANGES = {
    "elevation_m": (-50, 3000), "slope_deg": (0, 90), "aspect_deg": (-1, 360),
    "curvature_general": (-1e4, 1e4), "curvature_plan": (-1e4, 1e4), "curvature_profile": (-1e4, 1e4),
    "twi": (-5, 40), "dist_to_river_m": (0, 60000), "dist_to_major_river_m": (0, 60000), "dist_to_road_m": (0, 60000),
    "drainage_density_km_km2": (0, 50),
    "rainfall_mean_annual_mm": tuple(results["rainfall"]["valid_range_mm"]),
    "flow_accumulation_km2": (0, 1e5),
}
with rasterio.open(f"{OUT_DIR}/analysis_domain.tif") as src:
    dom_disk = src.read(1).astype(bool)
per_layer = {}
all_ok = bool((dom_disk == land).all())
for name, path in {**paths, "feature_stack": stack_path}.items():
    with rasterio.open(path) as src:
        arrs = src.read()
        geo_ok = (src.crs == CRS and src.transform == T and src.shape == (HGT, WID))
        bands = list(src.descriptions)
    for b, arr in enumerate(arrs):
        lname = bands[b] if name == "feature_stack" else name
        key = f"feature_stack[{b + 1}] {lname}" if name == "feature_stack" else name
        nan_in = int(np.isnan(arr[dom_disk]).sum())
        fin_out = int(np.isfinite(arr[~dom_disk]).sum())
        v = arr[dom_disk].astype(np.float64)
        lo, hi = RANGES[lname]
        in_range = bool(v.min() >= lo - 1e-6 and v.max() <= hi + 1e-6)
        ok = geo_ok and nan_in == 0 and fin_out == 0 and in_range
        if name == "elevation_m":
            ok = ok and bool((arr[dom_disk] == dem[dom_disk]).all())
        all_ok &= ok
        if name != "feature_stack":
            per_layer[name] = {
                "same_crs_transform_shape_as_dem": bool(geo_ok), "nan_inside_domain": nan_in,
                "finite_outside_domain": fin_out, "in_physical_range": in_range, "pass": bool(ok),
                "min": round(float(v.min()), 4), "p01": round(float(np.percentile(v, 1)), 4),
                "median": round(float(np.median(v)), 4), "mean": round(float(v.mean()), 4),
                "p99": round(float(np.percentile(v, 99)), 4), "max": round(float(v.max()), 4),
            }
        print(f"    [{'PASS' if ok else 'FAIL'}] {key:<34} nan_in={nan_in} finite_out={fin_out} "
              f"range=[{v.min():.3g}, {v.max():.3g}]")
stack_names_ok = bands == [f[0] for f in FEATURES]
all_ok &= stack_names_ok
flow_ok = flow["n_interior_sinks"] == 0 and mass_err < 1e-9
all_ok &= flow_ok
print(f"    [{'PASS' if stack_names_ok else 'FAIL'}] stack band names/order")
print(f"    [{'PASS' if flow_ok else 'FAIL'}] flow routing: no interior sinks, mass balance")

# independent spot-check of the KD-tree distances and the lattice drainage
# density against exact shapely geometry at random land pixels
spot_idx = RNG.choice(np.flatnonzero(land.ravel()), size=2000, replace=False)
spot_pts = shapely.points(centres[spot_idx])
spot = {}
for lname, gdf, arr in [("dist_to_river_m", rivers, dist_river), ("dist_to_major_river_m", major, dist_major),
                        ("dist_to_road_m", roads, dist_road)]:
    tree = shapely.STRtree(gdf.to_crs(UTM).geometry.values)
    _, exact = tree.query_nearest(spot_pts, return_distance=True, all_matches=False)
    err = np.abs(arr.ravel()[spot_idx] - exact)
    spot[lname] = {"n": int(len(spot_idx)), "max_abs_err_m": round(float(err.max()), 3),
                   "tolerance_m": DENSIFY_M / 2}
    spot[lname]["pass"] = bool(err.max() <= DENSIFY_M / 2 + 1e-6)
riv_utm = shapely.union_all(rivers.to_crs(UTM).geometry.values)
dd_idx = spot_idx[:300]
circles = shapely.buffer(shapely.points(centres[dd_idx]), DD_RADIUS_M, quad_segs=64)
dd_exact = shapely.length(shapely.intersection(circles, riv_utm)) / 1000 / (np.pi * DD_RADIUS_M ** 2 / 1e6)
dd_err = np.abs(drainage_density.ravel()[dd_idx] - dd_exact)
spot["drainage_density_km_km2"] = {
    "n": int(len(dd_idx)), "mean_exact": round(float(dd_exact.mean()), 4),
    "median_abs_err": round(float(np.median(dd_err)), 4), "max_abs_err": round(float(dd_err.max()), 4),
    "tolerance_max_abs_err": 0.05}
spot["drainage_density_km_km2"]["pass"] = bool(dd_err.max() <= 0.05)
spot_ok = all(v["pass"] for v in spot.values())
all_ok &= spot_ok
for lname, v in spot.items():
    e = v.get("max_abs_err_m", v.get("max_abs_err"))
    print(f"    [{'PASS' if v['pass'] else 'FAIL'}] exact-geometry spot check {lname:<24} max |err| {e}")

# HydroRIVERS vs DEM-derived channels (positional check of both sources)
chan = land & (acc_km2 >= CHANNEL_KM2)
chan_tree = cKDTree(centres[chan.ravel()])
riv_in = gpd.clip(rivers, aoi_box).explode(index_parts=False)
riv_align = {}
for label, sub in [("all_orders", riv_in), ("strahler_ge_4", riv_in[riv_in["ORD_STRA"] >= 4])]:
    pts = densified_vertices(sub, 30.0)
    lon_p, lat_p = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(pts[:, 0], pts[:, 1])
    pr = ((np.asarray(lat_p) - T.f) / T.e).astype(int)
    pc = ((np.asarray(lon_p) - T.c) / T.a).astype(int)
    on_land = land[np.clip(pr, 0, HGT - 1), np.clip(pc, 0, WID - 1)]
    d, _ = chan_tree.query(pts[on_land], workers=-1)
    riv_align[label] = {
        "n_points": int(on_land.sum()), "median_m": round(float(np.median(d)), 1),
        "p75_m": round(float(np.percentile(d, 75)), 1), "p90_m": round(float(np.percentile(d, 90)), 1),
        "within_100m_share": round(float((d <= 100).mean()), 3),
        "within_250m_share": round(float((d <= 250).mean()), 3),
    }
    print(f"    HydroRIVERS ({label}) -> nearest DEM channel (>= {CHANNEL_KM2:g} km2): median "
          f"{riv_align[label]['median_m']} m, p90 {riv_align[label]['p90_m']} m, "
          f"within 250 m {100 * riv_align[label]['within_250m_share']:.0f}%")

# correlation / VIF on a random land sample
idx = RNG.choice(np.flatnonzero(land.ravel()), size=min(SAMPLE_N, int(land.sum())), replace=False)
names = [f[0] for f in FEATURES]
X = np.column_stack([layers[n].ravel()[idx] for n in names])
corr = np.corrcoef(X, rowvar=False)
vif = np.diag(np.linalg.inv(corr))
spear = np.corrcoef(np.argsort(np.argsort(X, axis=0), axis=0), rowvar=False)
high = [(names[i], names[j], round(float(corr[i, j]), 3))
        for i in range(len(names)) for j in range(i + 1, len(names)) if abs(corr[i, j]) >= 0.7]
print("    VIF: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(names, vif)))
print(f"    |r| >= 0.7 pairs: {high if high else 'none'}")

results["checkpoint_7_5"] = {
    "all_pass": bool(all_ok),
    "layers": per_layer,
    "stack_band_order": names,
    "stack_band_names_ok": bool(stack_names_ok),
    "flow_routing_ok": bool(flow_ok),
    "exact_geometry_spot_checks": spot,
}
results["hydrorivers_vs_dem_channels"] = {"dem_channel_threshold_km2": CHANNEL_KM2, **riv_align}
results["collinearity"] = {
    "n_sample_pixels": int(len(idx)),
    "pearson": {a: {b: round(float(corr[i, j]), 3) for j, b in enumerate(names)} for i, a in enumerate(names)},
    "spearman": {a: {b: round(float(spear[i, j]), 3) for j, b in enumerate(names)} for i, a in enumerate(names)},
    "vif": {n: round(float(v), 2) for n, v in zip(names, vif)},
    "pairs_abs_r_ge_0.7": high,
}
with open(f"{METRICS_DIR}/step7_feature_stack_checkpoint.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

meta = {
    "grid": results["domain"]["grid"],
    "nodata": "NaN (float32) = open water outside the analysis domain",
    "analysis_domain": f"{OUT_DIR}/analysis_domain.tif",
    "feature_stack": {"path": stack_path, "bands": [
        {"band": b, "name": n, "unit": u, "description": d} for b, (n, u, d) in enumerate(FEATURES, start=1)]},
    "auxiliary": {"flow_accumulation_km2": {"path": paths["flow_accumulation_km2"],
                                            "description": "Quinn FD8 upslope area, AOI-internal only"}},
    "inputs": {"dem": FABDEM, "rivers": RIVERS, "roads": f"{ROADS}:edges", "rainfall": CHIRPS},
    "parameters": {"tan_beta_floor": TAN_SLOPE_MIN, "densify_m": DENSIFY_M,
                   "drainage_density_radius_m": DD_RADIUS_M, "utm_crs": UTM},
}
with open(f"{OUT_DIR}/features_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[fig] feature stack overview")
extent = (T.c, T.c + WID * T.a, T.f + HGT * T.e, T.f)
WATER_BG = "#d6dde3"
C_RIVER = "#2a78d6"
C_CHAN = "#eb6834"


def hillshade(z, az=315, alt=45):
    s = np.radians(90 - alt)
    a = np.radians(az)
    slope_r = np.arctan(tan_b)
    asp_r = np.arctan2(-p, -q)                       # downslope azimuth
    hs = np.cos(s) * np.cos(slope_r) + np.sin(s) * np.sin(slope_r) * np.cos(a - asp_r)
    return np.clip(hs, 0, 1)


hs = hillshade(dem)
panels = [
    ("elevation_m", "Elevation (m)", "Greens", None, (np.percentile(dem[land], 1), np.percentile(dem[land], 99))),
    ("slope_deg", "Slope (°)", "Oranges", None, (0, np.percentile(slope[land], 99))),
    ("aspect_deg", "Aspect (° from N; 6×6 circular mean for display)", "twilight", None, (0, 360)),
    ("curvature_plan", "Plan curvature (+ divergent)", "PuOr_r", None, "sym"),
    ("curvature_profile", "Profile curvature (+ convex)", "PuOr_r", None, "sym"),
    ("twi", "TWI  ln(a / tan β)", "Blues", None, (np.percentile(twi[land], 1), np.percentile(twi[land], 99))),
    ("dist_to_river_m", "Distance to river, all HydroRIVERS (m)", "Purples_r", None, (0, np.percentile(dist_river[land], 99))),
    ("dist_to_major_river_m", "Distance to major river, Strahler ≥ 4 (m)", "Purples_r", None,
     (0, np.percentile(dist_major[land], 99))),
    ("dist_to_road_m", "Distance to road (m)", "Purples_r", None, (0, np.percentile(dist_road[land], 99))),
    ("drainage_density_km_km2", "Drainage density (km/km², r = 2 km)", "Blues", None, (0, drainage_density[land].max())),
    ("rainfall_mean_annual_mm", "Mean annual rainfall 2017–23 (mm)", "Blues", None, None),
    ("flow_accumulation_km2", "Flow accumulation (km², log)", "Blues", "log", None),
]
fig, axes = plt.subplots(3, 4, figsize=(20, 13.4), dpi=100)
for ax, (name, title, cmap, norm, lim) in zip(axes.ravel(), panels):
    arr = {**layers, **aux}[name].astype(np.float64).copy()
    ax.set_facecolor(WATER_BG)
    ax.imshow(np.where(land, hs, np.nan), cmap="Greys_r", extent=extent, vmin=0, vmax=1, alpha=0.9,
              interpolation="nearest")
    arr[~land] = np.nan
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0)
    kw = {}
    if norm == "log":
        kw["norm"] = LogNorm(vmin=max(np.nanpercentile(arr, 1), 1e-4), vmax=np.nanmax(arr))
    elif lim == "sym":
        v = np.nanpercentile(np.abs(arr), 98)
        kw.update(vmin=-v, vmax=v)
    elif lim is not None:
        kw.update(vmin=lim[0], vmax=lim[1])
    if name == "aspect_deg":
        # a cyclic, pixel-scale field aliases into false bands when drawn at
        # thumbnail size; show the circular mean of 6x6 blocks instead
        flat = np.where(land & (aspect == -1), 1.0, np.nan)
        rad = np.radians(np.where(aspect < 0, 0, aspect))
        w8 = (aspect >= 0).astype(float)
        s_m = ndimage.uniform_filter(np.sin(rad) * w8, 6)
        c_m = ndimage.uniform_filter(np.cos(rad) * w8, 6)
        arr = np.degrees(np.arctan2(s_m, c_m)) % 360
        arr[~land | (aspect == -1)] = np.nan
    im = ax.imshow(arr, cmap=cm, extent=extent, alpha=0.82 if name != "rainfall_mean_annual_mm" else 0.9,
                   interpolation="nearest", **kw)
    if name == "aspect_deg":
        ax.imshow(flat, cmap=ListedColormap(["#8c8c8c"]), extent=extent, interpolation="nearest")
    if name == "dist_to_river_m":
        riv_in.plot(ax=ax, color=C_RIVER, lw=0.6)
    if name == "dist_to_major_river_m":
        riv_in[riv_in["ORD_STRA"] >= MAJOR_STRAHLER].plot(ax=ax, color=C_RIVER, lw=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
fig.suptitle("Step 7  Terrain & hydrological feature stack — FABDEM 1″ grid (1300 × 1486), "
             "open water (grey-blue) outside the analysis domain",
             fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
p_over = f"{MAP_DIR}/step7_feature_stack_overview.png"
fig.savefig(p_over, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_over}")

print("[fig] river alignment check")
fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), dpi=110,
                         gridspec_kw={"width_ratios": [1.15, 1]})
zoom = (76.30, 76.55, 10.05, 10.25)     # Periyar main stem, Kalady - Aluva - Paravur
for ax, lims, ttl in [(axes[0], extent, "Whole AOI"), (axes[1], zoom, "Zoom: Periyar main stem (Kalady → Aluva)")]:
    ax.set_facecolor(WATER_BG)
    ax.imshow(np.where(land, hs, np.nan), cmap="Greys_r", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.where(chan, 1.0, np.nan), cmap=ListedColormap([C_CHAN]), extent=extent, interpolation="nearest")
    riv_in.plot(ax=ax, color=C_RIVER, lw=1.1 if ax is axes[1] else 0.8)
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_title(ttl, loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)
a = riv_align["all_orders"]
fig.legend(handles=[Line2D([], [], color=C_RIVER, lw=2, label="HydroRIVERS v10 (15″ network)"),
                    Line2D([], [], color=C_CHAN, lw=5, label=f"FABDEM channels (FD8 accumulation ≥ {CHANNEL_KM2:g} km²)")],
           loc="lower center", ncol=2, fontsize=11, frameon=False)
fig.suptitle(f"Step 7  River-network position check — HydroRIVERS to nearest FABDEM channel: median {a['median_m']:.0f} m, "
             f"p90 {a['p90_m']:.0f} m, {100 * a['within_250m_share']:.0f}% within 250 m",
             fontsize=13, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
p_riv = f"{MAP_DIR}/step7_river_alignment_check.png"
fig.savefig(p_riv, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_riv}")

print("[fig] feature correlation")
short = ["Elevation", "Slope", "Aspect", "Curv. plan", "Curv. profile", "TWI",
         "Dist. river", "Dist. major river", "Dist. road", "Drain. density", "Rainfall"]
fig, ax = plt.subplots(figsize=(10.5, 9), dpi=120)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr[i, j]) > 0.6 else "#0b0b0b")
ax.set_xticks(range(len(names)), short, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(names)), [f"{s}  (VIF {v:.1f})" for s, v in zip(short, vif)], fontsize=9)
ax.set_title(f"Pearson correlation of Step 7 features ({len(idx):,} random land pixels)",
             loc="left", fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.outline.set_visible(False)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
p_cor = f"{MAP_DIR}/step7_feature_correlation.png"
fig.savefig(p_cor, bbox_inches="tight")
plt.close(fig)
print(f"    [OK] {p_cor}")

print("\n" + "=" * 80)
print("STEP 7 SUMMARY")
print("=" * 80)
print(f"  Grid: native FABDEM {HGT} x {WID} ({CRS}); land domain {results['domain']['land_km2']} km2, "
      f"open water {results['domain']['open_water_km2']} km2")
print(f"  Stack: {stack_path} ({len(FEATURES)} bands)")
print(f"  Checkpoint 7.5: {'PASS' if all_ok else 'FAIL'}  (self-tests PASS, flow routing "
      f"{'PASS' if flow_ok else 'FAIL'})")
print(f"  Runtime {time.time() - t_start:.0f} s")
print("=" * 80)
if not all_ok:
    raise SystemExit("Checkpoint 7.5 failed - see the table above.")
