"""
Phase 2 Step 4 - Full-basin routing, UICA and dUICA.

4.1 Full upstream basin. HydroBASINS level 10 (GEE WWF/HydroSHEDS/v1): every
    basin touching the AOI plus every basin whose NEXT_DOWN chain ends in one
    of them. FABDEM is fetched for the bounding box of that catchment on the
    SAME pixel lattice as the Phase 1 grid (0.000269494585 deg, integer
    offset), so the AOI is an exact sub-window (checked: identical DEM).
    Routing = the Phase 1 Priority-Flood+eps and Quinn FD8 from common.py.
4.2 UICA (upslope impervious contributing area) = FD8 accumulation of
    impervious_fraction x cell area (km2), i.e. flow accumulation with paved
    area as the "water". UICA ratio = UICA / upslope area (paved share of the
    catchment draining to the cell).
    Imperviousness on the extended grid: inside the AOI the Step 2 hybrid layer
    (exact); outside it the same Option A formula built at 30 m in GEE:
        imp_y = clip(GHSL2020 + k (DW_y - DW_2020), 0, 1), k from Step 2,
    with DW_y = 30 m fraction of 10 m Dynamic World "built" (annual mode),
    temporally smoothed by a 3-year median (interior years).
4.3 dUICA for 2018 -> 2025 (plan minimum), 2017 -> 2025 (whole record) and
    2018 -> 2024 (between the first and last kept flood events).
4.4 Self-tests (Phase 1 Step 7 style):
    T1 weighted FD8 with weight = cell area reproduces plain accumulation;
    T2 inclined plane, one fully paved cell: UICA downstream equals the
       analytic FD8 spreading (repeated 3-tap convolution), mass conserved;
    T3 0 <= UICA <= upslope area everywhere (real grid, all years);
    T4 UICA = 0 wherever no impervious cell lies upstream (real grid);
    T5 UICA never decreases downstream along channels (upslope area >= 10 km2,
       steepest-descent step). FD8 splits flow between several lower cells, so
       the exact statement tested is the FD8 invariant: the receiver holds at
       least the share of the cell's UICA routed to it, and UICA does not drop
       where the flow is not split. The raw drop rate of plain upslope area on
       the same steps is reported to show that drops are a routing property.

Outputs
    data/processed/basin/{fabdem_ext.tif, upstream_catchment.geojson, zf_ext.npy, acc_ext.npy, ext_grid.json}
    data/processed/landcover/impervious_fraction_ext_30m.tif   (bands 2017-2025, uint8 x 200)
    data/processed/uica/uica_km2_aoi.tif, uica_ratio_aoi.tif, duica_km2_aoi.tif, upslope_area_full_km2_aoi.tif
    outputs/metrics/step4_uica.json
    outputs/maps/step4_catchment.png, step4_uica_maps.png, step4_self_tests.png
"""
import os
import json
import time

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource
from matplotlib.lines import Line2D

import common as C

BASIN = C.p2("data", "processed", "basin")
LC = C.p2("data", "processed", "landcover")
UO = C.p2("data", "processed", "uica")
for d in (BASIN, LC, UO):
    os.makedirs(d, exist_ok=True)

HYBAS_LEVEL = 10
BUFFER_DEG = 0.02
YEARS = list(range(2017, 2026))
ANCHOR = 2020
DUICA_PAIRS = [(2018, 2025), (2017, 2025), (2018, 2024)]
CHANNEL_KM2 = 10.0
STEP2_JSON = f"{C.METRICS_DIR}/step2_builtup_timeseries.json"
STEP2_IMP = f"{LC}/impervious_fraction_30m.tif"
SCALE = 200.0          # uint8 storage of fractions (0..1 -> 0..200, 255 = no data)

C.banner("PHASE 2 STEP 4: FULL-BASIN ROUTING, UICA AND dUICA")
t_start = time.time()
res = {"step": 4, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

with rasterio.open(C.P1_FABDEM) as src:
    T30 = src.transform
    H30, W30 = src.shape
    dem_aoi_p1 = src.read(1)
with rasterio.open(C.P1_DOMAIN) as src:
    land_aoi_p1 = src.read(1).astype(bool)
step2 = json.load(open(STEP2_JSON, encoding="utf-8"))
K = step2["hybrid_impervious"]["k"]
A = T30.a

# ---------------------------------------------------------------------------
# 4.1 Upstream catchment and extended grid
# ---------------------------------------------------------------------------
print("\n[1/8] 4.1 Upstream catchment (HydroBASINS level 10)")
ee = C.ee_init()
aoi_g = ee.Geometry.Rectangle(C.BBOX)
search = ee.Geometry.Rectangle([75.5, 8.8, 77.8, 11.2])
catch_path = f"{BASIN}/upstream_catchment.geojson"
fc = ee.FeatureCollection(f"WWF/HydroSHEDS/v1/Basins/hybas_{HYBAS_LEVEL}").filterBounds(search)
rows = fc.reduceColumns(ee.Reducer.toList(4), ["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "UP_AREA"]).get("list").getInfo()
nd = {r[0]: r[1] for r in rows}
sub = {r[0]: r[2] for r in rows}
touch = set(fc.filterBounds(aoi_g).aggregate_array("HYBAS_ID").getInfo())
ids = set(touch)
while True:
    add = {h for h, n in nd.items() if n in ids and h not in ids}
    if not add:
        break
    ids |= add
if not os.path.exists(catch_path):
    gj = fc.filter(ee.Filter.inList("HYBAS_ID", list(ids))).getInfo()
    with open(catch_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
import geopandas as gpd
catch = gpd.read_file(catch_path)
cb = catch.total_bounds
print(f"    {len(touch)} basins touch the AOI, {len(ids) - len(touch)} more drain into them; "
      f"catchment {sum(sub[h] for h in ids):.0f} km2, bounds {np.round(cb, 3).tolist()}")

# extended grid on the Phase 1 lattice
n_left = int(np.ceil((T30.c - (cb[0] - BUFFER_DEG)) / A))
n_top = int(np.ceil(((cb[3] + BUFFER_DEG) - T30.f) / A))
TX = Affine(A, 0, T30.c - n_left * A, 0, -A, T30.f + n_top * A)
WX = int(np.ceil(((cb[2] + BUFFER_DEG) - TX.c) / A))
HX = int(np.ceil((TX.f - (cb[1] - BUFFER_DEG)) / A))
R0, C0 = n_top, n_left                       # AOI window inside the extended grid
AOI = (slice(R0, R0 + H30), slice(C0, C0 + W30))
print(f"    extended grid {HX} x {WX} = {HX * WX / 1e6:.1f} M cells; AOI window rows {R0}:{R0 + H30}, cols {C0}:{C0 + W30}")
res["catchment"] = {"hybas_level": HYBAS_LEVEL, "n_basins_touching_aoi": len(touch), "n_basins_total": len(ids),
                    "area_km2_hydrobasins": round(sum(sub[h] for h in ids), 1), "bounds": np.round(cb, 4).tolist()}
res["ext_grid"] = {"height": HX, "width": WX, "transform": list(TX)[:6], "aoi_row0": R0, "aoi_col0": C0}
C.save_json(res["ext_grid"], f"{BASIN}/ext_grid.json")

# ---------------------------------------------------------------------------
# Downloads: FABDEM, GHSL 2020, DW yearly built fraction (extended grid)
# ---------------------------------------------------------------------------
print("\n[2/8] Extended-grid inputs from Earth Engine (cached)")
dem_path = f"{BASIN}/fabdem_ext.tif"
if not os.path.exists(dem_path):
    dem_img = (ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().unmask(0)
               .toFloat().rename("elevation_m"))
    t = time.time()
    C.ee_download_grid(dem_img, TX, WX, HX, dem_path, tiles=(3, 3), workers=6)
    print(f"    FABDEM {time.time() - t:.0f} s")
with rasterio.open(dem_path) as src:
    dem = src.read(1)
same_dem = np.array_equal(dem[AOI], dem_aoi_p1)
print(f"    AOI window of the extended DEM identical to Phase 1 DEM: {same_dem}")
assert same_dem, "extended grid is not aligned with the Phase 1 grid"

ghsl_path = f"{LC}/ghsl2020_ext_30m.tif"
if not os.path.exists(ghsl_path):
    g = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filter(ee.Filter.eq("system:index", str(ANCHOR)))
         .first().select("built_surface").divide(10000).clamp(0, 1).toFloat().rename("ghsl"))
    C.ee_download_grid(g, TX, WX, HX, ghsl_path, tiles=(2, 2), workers=4)
with rasterio.open(ghsl_path) as src:
    ghsl_x = src.read(1).astype(np.float32)

utm10 = ee.Projection("EPSG:32643").atScale(10)
dwf = {}
for y in YEARS:
    p = f"{LC}/dw_builtfrac_ext_30m_{y}.tif"
    if not os.path.exists(p):
        col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(search)
               .filterDate(f"{y}-01-01", f"{y + 1}-01-01").select("label"))
        img = (col.mode().eq(6).reproject(utm10)
               .reduceResolution(ee.Reducer.mean(), maxPixels=256)
               .multiply(SCALE).round().unmask(255).toByte().rename("dw_built"))
        t = time.time()
        C.ee_download_grid(img, TX, WX, HX, p, tiles=(6, 6), workers=8, retries=6)
        print(f"    DW {y} built fraction: {time.time() - t:.0f} s")
    with rasterio.open(p) as src:
        dwf[y] = src.read(1)

# temporal cleaning at 30 m: gap fill from neighbouring years, 3-year median
F = np.stack([dwf[y] for y in YEARS]).astype(np.float32)
F[F == 255] = np.nan
F /= SCALE
for i in range(len(YEARS)):
    for k in range(1, len(YEARS)):
        for j in (i - k, i + k):
            if 0 <= j < len(YEARS):
                m = np.isnan(F[i]) & ~np.isnan(F[j])
                F[i][m] = F[j][m]
F = np.nan_to_num(F, nan=0.0)
Fs = F.copy()
for i in range(1, len(YEARS) - 1):
    Fs[i] = np.median(F[i - 1:i + 2], axis=0)
del F

with rasterio.open(STEP2_IMP) as src:
    imp_aoi = {y: src.read(i + 1) for i, y in enumerate(YEARS)}
imp_x = {}
for i, y in enumerate(YEARS):
    v = np.clip(ghsl_x + K * (Fs[i] - Fs[YEARS.index(ANCHOR)]), 0, 1)
    gee_aoi = v[AOI].copy()
    a = imp_aoi[y]
    m = land_aoi_p1 & np.isfinite(a)
    v[AOI][m] = a[m]
    if y in (2018, 2025):
        r = float(np.corrcoef(gee_aoi[m], a[m])[0, 1])
        res.setdefault("imperviousness_ext", {})[f"aoi_gee_vs_step2_r_{y}"] = round(r, 4)
    imp_x[y] = v.astype(np.float32)
del Fs
_, _, cellx = C.row_cell_sizes_m(TX, HX)
outside = np.ones((HX, WX), bool)
outside[AOI] = False
res.setdefault("imperviousness_ext", {}).update({
    "k_from_step2": K, "aoi_cells_use_step2_layer": True,
    "impervious_km2_outside_aoi_by_year": {y: round(float(((imp_x[y] * outside).sum(axis=1) * cellx).sum() / 1e6), 2)
                                           for y in YEARS}})
print(f"    imperviousness outside AOI: {res['imperviousness_ext']['impervious_km2_outside_aoi_by_year']}")
print(f"    AOI check, GEE route vs Step 2 layer: {res['imperviousness_ext']}")
with rasterio.open(f"{LC}/impervious_fraction_ext_30m.tif", "w", driver="GTiff", height=HX, width=WX,
                   count=len(YEARS), dtype="uint8", crs="EPSG:4326", transform=TX, compress="deflate") as dst:
    for i, y in enumerate(YEARS, 1):
        dst.write(np.round(imp_x[y] * SCALE).astype(np.uint8), i)
        dst.set_band_description(i, f"impervious_{y}_x200")

# ---------------------------------------------------------------------------
# 4.4 Synthetic self-tests (before touching real data)
# ---------------------------------------------------------------------------
print("\n[3/8] 4.4 Synthetic self-tests")
tests = {}
rng = np.random.default_rng(C.SEED)
n, L = 120, 30.0
dxs = np.full(n, L); dys = np.full(n, L); area = dxs * dys
zr = rng.random((n, n)) * 5 + np.add.outer(np.arange(n) * -0.3, np.zeros(n)) + 100
landr = np.ones((n, n), bool)
zf_r, acc_r, _ = C.route_flow(zr, landr, dxs, dys, area)
flat = np.flatnonzero(landr.ravel())
order_r = flat[np.argsort(-zf_r.ravel()[flat], kind="stable")]
acc_w, *_ = C.quinn_fd8_weighted(zf_r, landr, order_r, np.broadcast_to(area[:, None], (n, n)).copy(), dxs, dys)
tests["T1_weighted_equals_plain_accumulation"] = {"max_abs_diff_m2": float(np.abs(acc_w - acc_r).max()),
                                                  "pass": bool(np.allclose(acc_w, acc_r, rtol=1e-12, atol=1e-6))}

s = 0.02
zp = 1000.0 - s * L * np.add.outer(np.arange(n), np.zeros(n))      # plane falling to the south
zf_p = zp                                       # a plane has no pits: route it as is, no fill
order_p = flat[np.argsort(-zf_p.ravel()[flat], kind="stable")]
wpave = np.zeros((n, n)); r0, c0 = 10, n // 2
wpave[r0, c0] = L * L
u_plane, *_ = C.quinn_fd8_weighted(zf_p, landr, order_p, wpave, dxs, dys)
wc, ws = 0.5, 0.354 / np.sqrt(2)                                    # FD8 weights to S and to SE/SW
kern = np.array([ws, wc, ws]) / (wc + 2 * ws)
prof = np.array([1.0])
worst = 0.0
for k in range(1, 40):
    prof = np.convolve(prof, kern)
    exp = np.zeros(n); exp[c0 - k:c0 + k + 1] = prof * L * L
    worst = max(worst, float(np.abs(u_plane[r0 + k] - exp).max() / (L * L)))
mass = float(abs(u_plane[r0 + 39].sum() - L * L) / (L * L))
tests["T2_inclined_plane_analytic"] = {"max_rel_error": worst, "mass_error": mass,
                                       "pass": bool(worst < 1e-9 and mass < 1e-9)}
print(f"    T1 weighted == plain accumulation: {tests['T1_weighted_equals_plain_accumulation']}")
print(f"    T2 inclined plane vs analytic FD8 spreading: {tests['T2_inclined_plane_analytic']}")

# ---------------------------------------------------------------------------
# 4.1 Routing on the extended grid
# ---------------------------------------------------------------------------
print("\n[4/8] 4.1 Routing the full basin (Priority-Flood+eps, Quinn FD8)")
dx, dy, cell_area = C.row_cell_sizes_m(TX, HX)
water, land = C.open_water_and_land(dem)
same_land = np.array_equal(land[AOI], land_aoi_p1)
t = time.time()
zf, acc, flow = C.route_flow(dem, land, dx, dy, cell_area)
flat = np.flatnonzero(land.ravel())
order = flat[np.argsort(-zf.ravel()[flat], kind="stable")]
land_m2 = float((land.sum(axis=1) * cell_area).sum())
mass_err = abs(flow["to_water_m2"] + flow["stranded_m2"] - land_m2) / land_m2
print(f"    routed {HX * WX / 1e6:.1f} M cells in {time.time() - t:.0f} s; interior sinks {flow['n_interior_sinks']}; "
      f"mass balance rel err {mass_err:.1e}; AOI land mask identical to Phase 1: {same_land}")
np.save(f"{BASIN}/zf_ext.npy", zf)
np.save(f"{BASIN}/acc_ext.npy", acc)

acc_km2 = acc / 1e6
acc_aoi = acc_km2[AOI]
with rasterio.open(C.P1_FLOWACC) as src:
    acc_p1 = src.read(1)
gain = acc_aoi[land_aoi_p1] / np.maximum(acc_p1[land_aoi_p1], 1e-9)
# inflow across each AOI edge: FD8 spreads a wide river over several adjacent
# cells, so a crossing = contiguous run of cells >= 50 km2 (gaps <= 3 cells
# bridged), and its inflow is the SUM over the run, not the largest cell
from scipy import ndimage as ndi
inflows = {}
for side, v, ax_len in (("east", acc_aoi[:, -2], H30), ("north", acc_aoi[1, :], W30),
                        ("south", acc_aoi[-2, :], W30)):
    big = v >= 50
    lab, nlab = ndi.label(ndi.binary_closing(big, np.ones(4)) & np.ones(len(v), bool))
    segs = []
    for k in range(1, nlab + 1):
        idx = np.flatnonzero((lab == k) & big)
        if len(idx):
            segs.append({"cells": int(len(idx)), "inflow_km2": round(float(v[idx].sum()), 1),
                         "position_index": int(idx.mean())})
    inflows[side] = sorted(segs, key=lambda s: -s["inflow_km2"])
entry = inflows["east"][0]["inflow_km2"] if inflows["east"] else 0.0
res["routing"] = {"cells": int(HX * WX), "land_km2": round(land_m2 / 1e6, 1), "n_interior_sinks": flow["n_interior_sinks"],
                  "mass_balance_rel_err": mass_err, "aoi_land_mask_identical": bool(same_land),
                  "max_upslope_area_in_aoi_km2": round(float(acc_aoi[land_aoi_p1].max()), 1),
                  "periyar_inflow_at_east_edge_km2": round(entry, 1),
                  "hydrorivers_reference_km2": 4081,
                  "periyar_inflow_vs_hydrorivers_rel_diff": round(entry / 4081 - 1, 4),
                  "edge_inflows_km2": inflows,
                  "aoi_cells_with_more_upslope_area_than_phase1_share": round(float((gain > 1.000001).mean()), 4),
                  "aoi_cells_with_10x_more_share": round(float((gain > 10).mean()), 4)}
print(f"    largest upslope area in AOI {res['routing']['max_upslope_area_in_aoi_km2']} km2 "
      f"(Phase 1: {float(acc_p1[land_aoi_p1].max()):.0f} km2); Periyar inflow at east edge {entry:.0f} km2 "
      f"(HydroRIVERS ~4,081 km2)")
print(f"    AOI cells whose upslope area grew vs Phase 1: {100 * res['routing']['aoi_cells_with_more_upslope_area_than_phase1_share']:.1f}%")


# ---------------------------------------------------------------------------
# 4.2 / 4.3 UICA per year and dUICA
# ---------------------------------------------------------------------------
DR, DC = C.DR, C.DC


@njit(cache=True)
def d8_receiver(zf, land, dx, dy):
    """Index of the steepest-descent neighbour (on the filled DEM) or -1."""
    H, W = zf.shape
    rec = np.full(H * W, -1, np.int64)
    for r in range(H):
        diag = np.sqrt(dx[r] ** 2 + dy[r] ** 2)
        for c in range(W):
            if not land[r, c]:
                continue
            best = 0.0
            for k in range(8):
                nr = r + DR[k]
                nc = c + DC[k]
                if nr < 0 or nc < 0 or nr >= H or nc >= W:
                    continue
                dist = diag if (DR[k] != 0 and DC[k] != 0) else (dx[r] if DR[k] == 0 else dy[r])
                g = (zf[r, c] - zf[nr, nc]) / dist
                if g > best:
                    best = g
                    rec[r * W + c] = nr * W + nc
    return rec


@njit(cache=True)
def fd8_fraction_to(zf, land, idx, rec_idx, dx, dy):
    """Share of each cell's FD8 outflow that goes to the given receiver."""
    H, W = zf.shape
    out = np.zeros(idx.shape[0])
    for n in range(idx.shape[0]):
        i = idx[n]
        r = i // W
        c = i % W
        diag = np.sqrt(dx[r] ** 2 + dy[r] ** 2)
        total = 0.0
        to_rec = 0.0
        for k in range(8):
            nr = r + DR[k]
            nc = c + DC[k]
            if nr < 0 or nc < 0 or nr >= H or nc >= W:
                continue
            drop = zf[r, c] - zf[nr, nc]
            if drop <= 0.0:
                continue
            if DR[k] != 0 and DC[k] != 0:
                wgt = drop / diag * 0.354
            elif DR[k] == 0:
                wgt = drop / dx[r] * 0.5
            else:
                wgt = drop / dy[r] * 0.5
            total += wgt
            if nr * W + nc == rec_idx[n]:
                to_rec = wgt
        out[n] = to_rec / total if total > 0 else 0.0
    return out


print("\n[5/8] 4.2-4.3 UICA per year, UICA ratio, dUICA")
uica_aoi, tests_real = {}, {"T3_bounds": {}, "T4_zero_without_upstream_imperviousness": {}}
rec = d8_receiver(zf, land, dx, dy)
ch = land & (acc_km2 >= CHANNEL_KM2)
ch_idx = np.flatnonzero(ch.ravel())
r_idx = rec[ch_idx]
ok = r_idx >= 0
ch_idx, r_idx = ch_idx[ok], r_idx[ok]
both = ch.ravel()[r_idx]
ch_idx, r_idx = ch_idx[both], r_idx[both]
frac = fd8_fraction_to(zf, land, ch_idx, r_idx, dx, dy)
single = frac >= 0.999                     # steps where FD8 does not split the flow
af = acc_km2.ravel()
acc_drop = af[r_idx] < af[ch_idx] * (1 - 1e-9)
t5 = {"definition": "FD8 invariant: UICA(receiver) >= f x UICA(cell), f = share of the cell's FD8 outflow sent "
                    "to its steepest receiver; and UICA never decreases where the flow is not split (f >= 0.999)",
      "channel_threshold_km2": CHANNEL_KM2, "channel_steps": int(len(ch_idx)),
      "steps_with_unsplit_flow": int(single.sum()),
      "upslope_area_raw_decrease_share": round(float(acc_drop.mean()), 4)}
for y in YEARS:
    t = time.time()
    w = imp_x[y].astype(np.float64) * cell_area[:, None] * land
    u, to_w, strand, _ = C.quinn_fd8_weighted(zf, land, order, w, dx, dy)
    uk = u / 1e6
    viol_hi = int(((uk - acc_km2) > 1e-9 * np.maximum(acc_km2, 1)).sum())
    viol_lo = int((uk < -1e-12).sum())
    tests_real["T3_bounds"][y] = {"cells_above_upslope_area": viol_hi, "cells_negative": viol_lo,
                                  "pass": viol_hi == 0 and viol_lo == 0}
    if y in (2018, 2025):
        ind = (imp_x[y] > 0).astype(np.float64) * land
        cnt, *_ = C.quinn_fd8_weighted(zf, land, order, ind, dx, dy)
        none_up = land & (cnt == 0)
        bad = int((none_up & (u > 0)).sum())
        tests_real["T4_zero_without_upstream_imperviousness"][y] = {
            "cells_without_upstream_imperviousness": int(none_up.sum()), "of_which_uica_nonzero": bad,
            "pass": bad == 0}
        uf = uk.ravel()
        tol = 1e-9 * np.maximum(uf[ch_idx], 1e-12)
        inv_viol = int((uf[r_idx] < frac * uf[ch_idx] - tol).sum())
        drop = uf[r_idx] < uf[ch_idx] * (1 - 1e-9)
        t5[y] = {"invariant_violations": inv_viol,
                 "decreases_where_flow_unsplit": int((single & (uf[r_idx] < uf[ch_idx] * (1 - 1e-3))).sum()),
                 "raw_decrease_share_all_steps": round(float(drop.mean()), 4),
                 "pass": inv_viol == 0 and int((single & (uf[r_idx] < uf[ch_idx] * (1 - 1e-3))).sum()) == 0}
    uica_aoi[y] = uk[AOI].astype(np.float32)
    print(f"    {y}: UICA max in AOI {uica_aoi[y][land_aoi_p1].max():8.1f} km2, "
          f"paved area routed to sea {to_w / 1e6:6.1f} km2  ({time.time() - t:.0f} s)")
tests_real["T5_monotonic_downstream_along_channels"] = t5
tests.update(tests_real)
t5_pass = all(v["pass"] for k, v in t5.items() if isinstance(k, int))
tests["T5_monotonic_downstream_along_channels"]["pass"] = bool(t5_pass)
print(f"    T3 bounds: {all(v['pass'] for v in tests['T3_bounds'].values())}; "
      f"T4 zero-without-upstream: {all(v['pass'] for v in tests['T4_zero_without_upstream_imperviousness'].values())}; "
      f"T5 monotonic along channels: {t5}")
del imp_x

ratio_aoi = {y: np.where(land_aoi_p1, uica_aoi[y] / np.maximum(acc_aoi, 1e-12), np.nan).astype(np.float32)
             for y in YEARS}
duica = {f"{a}-{b}": (uica_aoi[b] - uica_aoi[a]).astype(np.float32) for a, b in DUICA_PAIRS}


def write(path, bands, names, nodata=np.nan):
    with rasterio.open(path, "w", driver="GTiff", height=H30, width=W30, count=len(bands), dtype="float32",
                       crs="EPSG:4326", transform=T30, nodata=nodata, compress="deflate", predictor=3) as dst:
        for i, (b, nme) in enumerate(zip(bands, names), 1):
            dst.write(np.where(land_aoi_p1, b, np.nan).astype(np.float32), i)
            dst.set_band_description(i, nme)


write(f"{UO}/uica_km2_aoi.tif", [uica_aoi[y] for y in YEARS], [f"uica_km2_{y}" for y in YEARS])
write(f"{UO}/uica_ratio_aoi.tif", [ratio_aoi[y] for y in YEARS], [f"uica_ratio_{y}" for y in YEARS])
write(f"{UO}/duica_km2_aoi.tif", list(duica.values()), [f"duica_km2_{k}" for k in duica])
write(f"{UO}/upslope_area_full_km2_aoi.tif", [acc_aoi], ["upslope_area_full_km2"])

lm = land_aoi_p1
res["uica"] = {
    "units": "km2 of impervious area draining to the cell (incl. the cell itself)",
    "aoi_summary_by_year": {y: {"median_km2": round(float(np.median(uica_aoi[y][lm])), 5),
                                "p99_km2": round(float(np.percentile(uica_aoi[y][lm], 99)), 3),
                                "max_km2": round(float(uica_aoi[y][lm].max()), 2),
                                "ratio_median": round(float(np.nanmedian(ratio_aoi[y][lm])), 4),
                                "ratio_p90": round(float(np.nanpercentile(ratio_aoi[y][lm], 90)), 4)} for y in YEARS},
    "duica": {k: {"median_km2": round(float(np.median(v[lm])), 5), "p99_km2": round(float(np.percentile(v[lm], 99)), 3),
                  "max_km2": round(float(v[lm].max()), 3), "min_km2": round(float(v[lm].min()), 3),
                  "share_cells_increase": round(float((v[lm] > 1e-6).mean()), 4)} for k, v in duica.items()}}
res["self_tests"] = tests
res["all_tests_pass"] = bool(all(v["pass"] for k, v in tests.items() if "pass" in v)
                             and all(x["pass"] for x in tests["T3_bounds"].values())
                             and all(x["pass"] for x in tests["T4_zero_without_upstream_imperviousness"].values()))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
print("\n[6/8] Figures")
ls = LightSource(azdeg=315, altdeg=45)
step = 4
dem_s = dem[::step, ::step].astype(float)
extx = (TX.c, TX.c + WX * A, TX.f - HX * A, TX.f)
fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), dpi=110)
ax = axes[0]
hs = ls.hillshade(np.where(dem_s > 0, dem_s, 0), vert_exag=0.05, dx=A * step * 111000, dy=A * step * 111000)
ax.imshow(hs, cmap="gray", extent=extx, alpha=0.9)
ax.imshow(np.ma.masked_where(dem_s <= 0, dem_s), cmap="terrain", vmin=0, vmax=2000, extent=extx, alpha=0.45)
catch.boundary.plot(ax=ax, color=C.C_ORANGE, lw=0.5)
catch.dissolve().boundary.plot(ax=ax, color=C.C_ORANGE, lw=2)
ax.add_patch(plt.Rectangle((C.BBOX[0], C.BBOX[1]), C.BBOX[2] - C.BBOX[0], C.BBOX[3] - C.BBOX[1], fill=False,
                           ec=C.C_BLUE, lw=2.2))
ax.set_title(f"4.1  Upstream catchment: {len(ids)} HydroBASINS L10 units, "
             f"{res['catchment']['area_km2_hydrobasins']:.0f} km²\n(orange) vs Phase 1 AOI (blue)",
             loc="left", fontweight="bold")
ax = axes[1]
hb, wb = (HX // step) * step, (WX // step) * step          # block max keeps the channel lines visible
acc_s = np.where(land, acc_km2, 0)[:hb, :wb].reshape(hb // step, step, wb // step, step).max(axis=(1, 3))
land_s = land[:hb, :wb].reshape(hb // step, step, wb // step, step).any(axis=(1, 3))
im = ax.imshow(np.where(land_s, np.maximum(acc_s, 1e-3), np.nan), cmap="Blues",
               norm=LogNorm(1e-3, 5e3), extent=extx)
ax.add_patch(plt.Rectangle((C.BBOX[0], C.BBOX[1]), C.BBOX[2] - C.BBOX[0], C.BBOX[3] - C.BBOX[1], fill=False,
                           ec=C.C_ORANGE, lw=2.2))
fig.colorbar(im, ax=ax, fraction=0.04, label="upslope area (km², log)")
ax.set_title(f"Full-basin flow accumulation; Periyar enters the AOI with {entry:,.0f} km²\n"
             f"(Phase 1 saw at most {float(acc_p1[land_aoi_p1].max()):,.0f} km² inside the AOI)",
             loc="left", fontweight="bold")
for a in axes:
    a.set_xlim(extx[0], extx[1]); a.set_ylim(extx[2], extx[3]); a.tick_params(labelsize=7)
fig.suptitle("Step 4.1 — routing the full upstream basin", fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{C.MAP_DIR}/step4_catchment.png", bbox_inches="tight")
plt.close(fig)

ext = (C.BBOX[0], C.BBOX[2], C.BBOX[1], C.BBOX[3])
ya, yb = DUICA_PAIRS[0]
fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5), dpi=110)
panels = [
    (np.where(lm, np.maximum(uica_aoi[yb], 1e-4), np.nan), "magma_r", LogNorm(1e-4, 100),
     f"UICA {yb} (km² of paved area draining to each cell, log)", "UICA (km²)"),
    (np.where(lm, ratio_aoi[yb], np.nan), "viridis_r", None, f"UICA ratio {yb} (paved share of the upstream catchment)",
     "UICA / upslope area"),
    (np.where(lm, duica[f'{ya}-{yb}'], np.nan), "RdBu_r", None, f"ΔUICA {ya} → {yb} (km²)", "ΔUICA (km²)"),
    (np.where(lm, np.maximum(acc_aoi, 1e-3), np.nan), "Blues", LogNorm(1e-3, 5e3),
     "Full-basin upslope area (km², log), for comparison", "upslope area (km²)"),
]
for ax, (arr, cm, norm, title, lab) in zip(axes.ravel(), panels):
    if cm == "RdBu_r":
        v = float(np.nanpercentile(np.abs(arr), 99.5))
        im = ax.imshow(arr, cmap=cm, vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    elif norm is None:
        im = ax.imshow(arr, cmap=cm, vmin=0, vmax=0.6, extent=ext, interpolation="nearest")
    else:
        im = ax.imshow(arr, cmap=cm, norm=norm, extent=ext, interpolation="nearest")
    ax.set_facecolor("#d6dde3")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, label=lab)
    ax.tick_params(labelsize=7)
fig.suptitle("Step 4.2–4.3 — UICA, UICA ratio and ΔUICA over the AOI", fontsize=14, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{C.MAP_DIR}/step4_uica_maps.png", bbox_inches="tight")
plt.close(fig)

# self-test figure: inclined plane
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=110)
ax = axes[0]
im = ax.imshow(np.where(u_plane > 0, u_plane / (L * L), np.nan)[:60, c0 - 30:c0 + 31], cmap="magma_r", norm=LogNorm(1e-6, 1))
ax.plot(30, r0, "o", mfc="none", mec=C.C_BLUE, ms=12, mew=2)
ax.set_title("T2  One fully paved cell (circle) on an inclined plane:\nUICA spreads downslope by FD8, total conserved",
             loc="left", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.04, label="UICA / paved cell area")
ax = axes[1]
prof = np.array([1.0])
for k, colr in zip((5, 15, 35), (C.C_BLUE, C.C_ORANGE, C.C_GREEN)):
    pr = np.array([1.0])
    for _ in range(k):
        pr = np.convolve(pr, kern)
    xs = np.arange(-k, k + 1)
    ax.plot(xs, pr, "-", color=colr, lw=2, label=f"analytic, {k} rows down")
    ax.plot(xs, u_plane[r0 + k, c0 - k:c0 + k + 1] / (L * L), "o", color=colr, ms=4, label=f"computed, {k} rows down")
ax.set_xlim(-20, 20)
ax.set_xlabel("columns from the paved cell")
ax.set_ylabel("UICA / paved cell area")
ax.set_title(f"T2  computed = analytic (max error {worst:.1e}); mass error {mass:.1e}", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{C.MAP_DIR}/step4_self_tests.png", bbox_inches="tight")
plt.close(fig)

print("\n[7/8] Metrics")
res["runtime_s"] = round(time.time() - t_start, 1)
C.save_json(res, f"{C.METRICS_DIR}/step4_uica.json")
print(f"[8/8] done in {res['runtime_s']} s; all self-tests pass: {res['all_tests_pass']}")
