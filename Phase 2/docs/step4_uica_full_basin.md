# Step 4: Full-basin routing, UICA and ΔUICA

Script: `scripts/04_uica_full_basin.py` · numbers: `outputs/metrics/step4_uica.json` · figures: `outputs/maps/step4_catchment.png`, `step4_uica_maps.png`, `step4_self_tests.png` · rasters (gitignored): `data/processed/uica/{uica_km2_aoi, uica_ratio_aoi, duica_km2_aoi, upslope_area_full_km2_aoi}.tif`, `data/processed/basin/*`, `data/processed/landcover/impervious_fraction_ext_30m.tif`

## 4.1 Full upstream basin

- **Catchment.** We took HydroBASINS level 10 from GEE: the 19 basins that touch the AOI, plus the 42 basins whose `NEXT_DOWN` chain ends in one of them. That is **61 basins, 9,130 km²**, with bounds 76.02–77.40 E and 9.28–10.55 N.
- **Extended grid.** We built a 4,881 × 5,267 grid (**25.7 M cells**) on the *same pixel lattice* as Phase 1 (0.000269494585°, integer offset). FABDEM was fetched for the whole grid. The AOI window of this DEM is **identical to the Phase 1 DEM**, checked with `np.array_equal`, and the AOI land mask is also identical.
- **Routing.** The Phase 1 Priority-Flood+ε and Quinn FD8 code from `common.py` routes the full grid in 15 s. It leaves 0 interior sinks, and the mass-balance error is 7.7 × 10⁻¹⁵.

| Check | Value |
|---|---|
| Largest upslope area inside the AOI | **6,378 km²** (Phase 1: 643 km²) |
| Periyar inflow across the east edge (sum over its channel cells) | **4,103 km²**, vs HydroRIVERS 4,081 km² (+0.5 %) |
| Other inflows | North edge 1,346 km² (Chalakudy side); south edge 1,340 km² (Muvattupuzha side) |
| AOI cells whose upslope area grew vs Phase 1 | 5.2 % (1.1 % grew more than 10×), i.e. exactly the river corridors |

FD8 spreads a wide river over several neighbouring cells. The inflow is therefore the *sum* across the river crossing, not the largest single cell: one cell holds 2,522 km² and the next four cells another 1,581 km².

## 4.2 UICA and the UICA ratio

- **UICA** (upslope impervious contributing area) is the FD8 accumulation of *impervious fraction × cell area*. It is flow accumulation with paved area as the "water", and its unit is km² of paved surface draining through the cell.
- **UICA ratio** = UICA / upslope area, i.e. the paved share of the cell's catchment.

**Imperviousness on the extended grid:**

- **Inside the AOI:** the Step 2 hybrid layer (Option A), used exactly as it is.
- **Outside the AOI:** the same Option A formula, `imp = clip(GHSL2020 + 0.214 · (DW_y − DW_2020), 0, 1)`, built at 30 m in GEE. Here DW_y is the 30 m share of 10 m Dynamic World "built" pixels (annual mode), smoothed with a 3-year median.
- **Consistency check.** Inside the AOI, the GEE-route layer agrees with the Step 2 layer at r = 0.9996 (2018) and 0.9991 (2025).
- **Impervious area outside the AOI on the extended grid:** 337 km² (2017) → 448 km² (2025). This part grows faster (+33 %) than the AOI (+15 %), partly because of real towns along the Western Ghats foothills and partly because of DW "built" drift in plantations. Most of this area drains away from the AOI.

| Year | UICA max in AOI (km²) | UICA p99 (km²) | UICA ratio median | UICA ratio p90 |
|---|---|---|---|---|
| 2017 | 114.6 | 1.03 | 0.057 | 0.322 |
| 2018 | 118.9 | 1.06 | 0.064 | 0.323 |
| 2024 | 135.5 | 1.23 | 0.090 | 0.325 |
| 2025 | 132.9 | 1.22 | 0.089 | 0.325 |

The largest values sit on the Periyar main stem, which carries the paved area of the whole upstream basin. The UICA ratio is highest across Kochi city (more than 0.5) and close to 0 in the forested east.

## 4.3 ΔUICA

| Pair | Median (km²) | p99 (km²) | Max (km²) | Cells that increased |
|---|---|---|---|---|
| 2018 → 2025 | 0.00001 | 0.149 | 15.06 | 54.9 % |
| 2017 → 2025 | 0.00002 | 0.188 | 18.28 | 60.7 % |
| 2018 → 2024 | 0.00001 | 0.156 | 16.59 | 56.7 % |

The growth concentrates along the drainage lines downstream of the areas that were built up. The small negative values (min −0.015 km²) come from cells where cleaned DW built-up decreased.

## 4.4 Self-tests: all pass

| Test | Result |
|---|---|
| T1: weighted FD8 with weight = cell area equals plain accumulation | max difference 0.0 m² ✅ |
| T2: inclined plane, one fully paved cell; UICA downstream vs analytic FD8 spreading (repeated 3-tap convolution) | max relative error 6 × 10⁻¹⁷, mass error 3 × 10⁻¹⁶ ✅ |
| T3: 0 ≤ UICA ≤ upslope area, every cell, every year | 0 violations ✅ |
| T4: UICA = 0 where no impervious cell lies upstream | 8.4 M such cells, 0 non-zero ✅ |
| T5: UICA never decreases downstream along channels (≥ 10 km², 179,693 steepest-descent steps) | 0 violations of the FD8 invariant; 0 decreases where the flow is not split (63,374 steps) ✅ |

**About T5.** Checked naïvely along the steepest-descent path, UICA "decreases" at 29.6 % of channel steps. The same check on *plain upslope area* gives 30.7 %, so this is how FD8 behaves, not an error in UICA: FD8 splits each cell's outflow between several lower neighbours, and one receiver gets only part of it. The test was therefore written as the exact FD8 statement:

- the receiver holds at least the share *f* of the cell's UICA that FD8 sends to it;
- where the flow is not split (*f* ≥ 0.999), UICA never drops.

Both parts hold everywhere.

## Limitations

- **Natural topographic routing.** The Idukki–Moolamattom inter-basin transfer and reservoir operations are not represented. Upstream paved area is routed as if the rivers flowed freely.
- **Dynamic World drift.** The extension outside the AOI uses the same `k` as the AOI and a 30 m median filter, not the 10 m majority filter, so DW drift in the hills is only partly removed.
- **ΔUICA depends on the yearly DW signal.** Its fine pattern inherits the Step 2 speckle. It is meaningful when summed along drainage lines, as here, not as single cells.
