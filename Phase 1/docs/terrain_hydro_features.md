# Terrain and Hydrological Feature Stack (Step 7)

Status as of 2026-09-27. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84).

**Script:** `scripts/11_build_terrain_hydro_features.py` (about 1 min end to end)
**Inputs:** FABDEM (`data/raw/fabdem_30m_aoi.tif`), HydroRIVERS (`data/raw/hydrosheds/`),
OSM drive network (`data/raw/osm_roads_aoi.gpkg`), CHIRPS mean annual rainfall (`data/raw/chirps_*`)
**Rasters:** `data/processed/features/` (one GeoTIFF per layer + `feature_stack.tif` + `analysis_domain.tif`)
**Metrics:** `outputs/metrics/step7_feature_stack_checkpoint.json`
**Maps:** `outputs/maps/step7_feature_stack_overview.png`,
`outputs/maps/step7_river_alignment_check.png`, `outputs/maps/step7_feature_correlation.png`

## Master grid and analysis domain (7.5)

Every layer is on the **native FABDEM grid**: EPSG:4326, 1 arc-second, 1300 × 1486 px. That grid already coincides with the AOI bbox, so the DEM is never resampled: the elevation band equals the source file bit for bit. Slopes, distances and areas use the true WGS84 cell size of every row (dx 29.52–29.56 m, dy 29.81 m), not a single degree-to-metre constant.

FABDEM is exactly **0 m** over the Arabian Sea, the Vembanad backwater and the Kochi harbour channels. That surface is one body connected to the grid border (256.3 km², 15.1 % of the grid), and 96.4 % of it is also dry-season water in the Step 5 Sentinel-1 data. These pixels are **open water**:
- They are the drainage outlet for flow routing.
- Every feature is NaN there.

The remaining **1,444.8 km² of land** is the analysis domain. It includes 15.1 km² of genuine below-sea-level polders (down to −6 m) and 10.1 km² of inland 0 m pixels. Every layer has a finite value on every land pixel.

## The stack

`feature_stack.tif` has 11 bands, in this order:

| # | Band | Unit | Method | Land median (p1–p99) |
|---|---|---|---|---|
| 1 | `elevation_m` | m | FABDEM v1-2, unchanged | 9.6 (−0.2 – 108.6) |
| 2 | `slope_deg` | ° | Horn (1981) 3×3 | 1.0 (0 – 18.4) |
| 3 | `aspect_deg` | ° | downslope compass direction, −1 = flat | – |
| 4 | `curvature_plan` | 1/100 m | Zevenbergen & Thorne (1987), + divergent / − convergent | 0.0 (−0.60 – 0.67) |
| 5 | `curvature_profile` | 1/100 m | Zevenbergen & Thorne (1987), + convex / − concave | 0.0 (−0.85 – 0.78) |
| 6 | `twi` | – | ln(a / tan β), see 7.2 | 9.1 (5.4 – 17.9) |
| 7 | `dist_to_river_m` | m | to nearest HydroRIVERS line, all orders | 648 (12 – 4,435) |
| 8 | `dist_to_major_river_m` | m | to nearest HydroRIVERS line with Strahler ≥ 4 | 4,072 (72 – 13,316) |
| 9 | `dist_to_road_m` | m | to nearest OSM drive-network edge | 50 (1 – 654) |
| 10 | `drainage_density_km_km2` | km/km² | HydroRIVERS length within 2 km radius / circle area | 0.40 (0 – 0.91) |
| 11 | `rainfall_mean_annual_mm` | mm/yr | CHIRPS 2017–2023 mean annual total, bilinear | 3,737 (3,570 – 3,899) |

Auxiliary layers are written but **not** in the stack:
- `flow_accumulation_km2`, the input to TWI.
- `curvature_general`. In the Zevenbergen–Thorne formulation general curvature is *exactly* plan + profile. With all three in the stack the VIF was about 10⁴, so only plan and profile are model features.

### 7.1 Slope, aspect, curvature
- **Horn gradient** with odd-reflection padding (linear extrapolation), so slope is exact on a plane right up to the grid edge.
- **Aspect** is the compass direction of steepest descent. 41,035 land pixels have exactly zero gradient and are coded −1.
- **Curvature** uses one sign convention for all layers: **positive = convex** (ridges, spurs), **negative = concave** (valleys, hollows).
- **Analytic self-tests** run before the real data and gate the run:
  - plane slope and aspect;
  - dome curvature;
  - a 45° diagonal ridge, which exercises the cross-derivative term;
  - spur vs. valley plan curvature.

### 7.2 Topographic Wetness Index
1. **Depression filling:** Priority-Flood+ε (Barnes et al. 2014) in float64. Outlets are the grid border and the open-water body. 313 k cells (275.5 km², mostly coastal flats and paddy land) are raised, by millimetres in flat areas.
2. **Flow accumulation:** multiple-flow-direction FD8 of Quinn et al. (1991), with weights tan β × contour length (0.5 cardinal, 0.354 diagonal).
3. **TWI = ln(a / tan β)**:
   - a = upslope area / cell width;
   - tan β from the Horn slope of the unfilled DEM, floored at 0.001;
   - 6.0 % of land pixels are at the floor.

Checks:
- **No interior cell is left without a strictly lower neighbour.**
- **Mass balance closes to 2 × 10⁻¹⁵.** 476.8 km² drains to the sea/backwater and 967.9 km² leaves through the grid edge. Most of the edge flow is two real rivers crossing the AOI box: the Periyar's northern arm into the Kodungallur backwater (~560 km², north edge at 76.19° E, 0.5 m) and the Muvattupuzha near Piravom (~170 km², south edge at 76.48° E).
- **Self-tests:** on an inclined plane the FD8 specific area equals the analytic (i+1)·L, and a closed pit is filled and drained.

**Bug found and fixed by these checks.** Barnes' algorithm uses `nextafter()` as ε. On a 0 m flat that step is 5 × 10⁻³²⁴ (a subnormal float). Once divided by the 30 m cell length it underflows to exactly 0, so 14,262 coastal cells had "no lower neighbour" and their flow was stranded. The fill now uses a fixed ε = 10⁻⁶ m, and a 0 m-flat self-test covers the case.

### 7.3 Distances
- **Method:** Euclidean distance from each pixel centre, computed in UTM 43N with a KD-tree over line vertices densified to ≤ 5 m. The error bound is 2.5 m.
- **Exact spot-check:** against exact shapely geometry at 2,000 random land pixels, the max error is 1.1 m (rivers), 0.3 m (major rivers) and 2.2 m (roads).
- **Rivers:** clipped with a 0.1° buffer in Step 3, so there is no edge effect.
- **Roads:** clipped to the AOI itself. On 1.5 % of land pixels a road just outside the AOI *could* be nearer; this is negligible.

**Why two river distances.** In this wet climate HydroRIVERS keeps streams down to 1.9 km² catchment (0.12 m³/s mean flow). 72 % of the land is within 1 km of some "river", so `dist_to_river_m` mostly measures distance to first-order streams. `dist_to_major_river_m` uses Strahler ≥ 4: upland area ≥ 252 km², 99 km of channel in the AOI, the Periyar main stem and its largest tributaries. Only 13.5 % of the land is within 1 km of one. The two are correlated at r = 0.48 (VIF 1.6), so each carries its own information.

**Position check** (`step7_river_alignment_check.png`): HydroRIVERS vertices were compared with channels traced from FABDEM (FD8 ≥ 5 km²).
- **Major rivers:** median offset 117 m, 71 % within 250 m, which is good for a 15″ (~460 m) product.
- **All orders:** median 220 m, p90 1.4 km. Small HydroRIVERS streams come from the coarse, stream-burned HydroSHEDS DEM, and in the flat coastal plain they do not follow FABDEM channels. Treat `dist_to_river_m` as approximate at sub-500 m scales.

### 7.4 Drainage density (done, not skipped)
- **Method:** HydroRIVERS length (all orders) inside a 2 km circle around each pixel, divided by the circle area. It is computed on a 30 m UTM lattice padded by 2 km, so edge pixels see full circles.
- **Result:** land mean 0.41 km/km², max 1.11.
- **Exact check:** against clipping the lines with a true circle at 300 pixels, the median error is 0.0013 km/km² and the max 0.026 km/km².

### Rainfall
CHIRPS stores 12 of its 72 cells over the AOI (all ocean) as **0 mm**. Those cells are filled from the nearest valid cell *before* bilinear interpolation. Plain resampling would have pulled coastal rainfall toward 0. No land pixel sits inside a no-data cell. The result stays within the source range (3,566–3,922 mm).

## 7.5 Checkpoint: PASS

Every layer is re-opened from disk and checked:

| Check | Result |
|---|---|
| Same CRS, transform and shape as the DEM (12 layers + 11 stack bands) | PASS |
| NaN inside the land domain | 0 in every layer |
| Finite values outside the domain (open water) | 0 in every layer |
| Physical ranges (slope 0–90°, distances ≥ 0, rainfall within CHIRPS range, …) | PASS |
| Elevation band identical to the FABDEM source | PASS |
| Stack band names and order | PASS |
| Analytic self-tests of slope / aspect / curvature / fill / FD8 (11) | PASS |
| Flow routing: no interior sinks, mass balance | PASS |
| Distances and drainage density vs exact geometry | PASS |

**Collinearity** (200,000 random land pixels, `step7_feature_correlation.png`):
- The largest |r| is 0.69, between elevation and slope.
- The largest VIF is 2.4 (slope).
- No pair reaches |r| ≥ 0.7.
- Physically sensible structure: TWI vs slope −0.54, TWI vs elevation −0.44, drainage density vs river distance −0.58.

## Known limitations

- **Upstream catchment truncated at the AOI edge.** HydroRIVERS gives up to 4,081 km² of upland area where the Periyar enters the AOI, but only 643 km² is routed inside it. TWI along the main stem is therefore lower than a basin-wide TWI; `dist_to_major_river_m` carries that signal instead. Fixing it would need the DEM for the whole upstream basin (roughly 4,000 km² more).
- **HydroRIVERS is a tree, not a delta.** Distributaries such as the Periyar's southern branch toward the Kochi backwater are not in it.
- **FABDEM in the coastal plain.** Relief there is centimetres to metres. Slope, aspect and curvature mostly pick out the real beach-ridge / swale pattern parallel to the coast (checked: no periodic DEM artifact in the spectrum). In near-flat ground they are dominated by DEM noise.
- **Rainfall is 5.5 km CHIRPS**, 60 cells with a ~10 % range across the AOI. It is smooth and close to a linear function of location, so in Step 9 a high importance for rainfall would more likely mean "location" than rain physics.
- **Aspect is circular** (359° is next to 0°). It is kept as degrees for comparability with the literature. Step 8 can convert it to sin/cos if it matters.

## Implications for Step 8

1. **Label grid:** sample on this 30 m grid. The Step 5/6 flood inventory is on the 20 m S1 grid, so aggregate the labels onto this grid (or sample points and read both rasters). Do not resample features onto the 20 m grid.
2. **Exclude open water:** drop any sample outside `analysis_domain.tif` (open water), as well as the Step 5 class-2 reference water already excluded in Step 6.
3. **Feature list:** the plan's minimum set is elevation, slope, TWI, dist-to-river, dist-to-road and rainfall. All are in the stack, plus aspect, plan/profile curvature, major-river distance and drainage density. None is redundant by VIF.
