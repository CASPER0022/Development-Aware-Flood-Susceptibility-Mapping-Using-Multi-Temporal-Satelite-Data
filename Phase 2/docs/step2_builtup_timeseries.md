# Step 2: Multi-year land-cover and imperviousness record

Script: `scripts/02_built_up_timeseries.py` · numbers: `outputs/metrics/step2_builtup_timeseries.json` · figures: `outputs/maps/step2_source_comparison.png`, `step2_builtup_first_last.png`, `step2_growth_sites.png` · rasters (gitignored): `data/processed/landcover/builtup_fraction_30m_{DW,ESRI,clean}.tif`, `ghsl_builtup_fraction_30m.tif`

**Status: [CHECKPOINT] closed. The team chose Option A, the hybrid (section 2.5). The final layer is `data/processed/landcover/impervious_fraction_30m.tif`.**

## Method

- **Sources, all from GEE:**
  - Dynamic World V1 (DW): annual mode of the per-image label, class "built".
  - Esri 10 m annual LULC (ESRI): class "built area".
  - ESA WorldCover (WC): 2020 (v100) and 2021 (v200), fetched for the whole AOI. The Phase 1 tile stopped at 10.008° N.
  - GHSL GHS-BUILT-S P2023A: built surface per 100 m cell for 2015 and 2020, plus the 2025 and 2030 projections.
- **Grid.** Every 10 m source is sampled onto a 1/3″ grid, which nests exactly 3 × 3 inside the Phase 1 FABDEM 1″ grid. The 30 m built-up fraction is therefore an exact mean of 9 sub-pixels, the same idea as the Step 8 label aggregation. GHSL is sampled directly at 30 m.
- **Cleaning (DW).**
  - Years with no observation take the nearest valid year.
  - A width-3 temporal majority filter is applied at 10 m.
  - 2017 and 2025 have only one neighbouring year, so they are not filtered.

## Results

| Source | Built on land (km²) | Share of land (1,444.8 km²) | vs WorldCover 2021 at 30 m: r / F1 | Mean yearly built "loss", raw |
|---|---|---|---|---|
| DW | 570 (2017) → 723 (2025) | 39 → 50 % | 0.46 / 0.41 | 3.5 % |
| ESRI | 786 (2017) → 974 (2025) | 54 → 67 % | 0.33 / 0.33 | 2.4 % |
| WorldCover | 140 (2020), 191 (2021) | 10–13 % | – | – (the 2020→2021 jump is a v100→v200 algorithm change) |
| GHSL | 140 (2015), 147 (2020), 153 (2025 proj.), 155 (2030 proj.) | 10–11 % | 0.74 / 0.30 (2020 epoch) | – |

- **DW after cleaning:** 570.1 → 722.9 km² (+152.8 km², 3.0 %/yr). The yearly built loss falls from 3.5 % (raw) to 1.5 %, but it is still 3.4 % for 2024→2025, which is an unfiltered endpoint. Growth slows from +8.9 % (2017→18) to about 0 % after 2023. The fast early growth is partly the classifier settling in, since DW has few images in 2017.
- **Validation sites (DW cleaned, 2017 → 2025):**

  | Site | Built-up 2017 → 2025 |
  |---|---|
  | Kakkanad–Infopark | 53.0 → 63.5 % |
  | Cochin Airport | 47.0 → 57.3 % |
  | Kalamassery | 73.2 → 76.2 % |
  | Aluva fringe | 59.5 → 64.3 % |

  Some of the new built-up area is real: the large cleared construction area south of Infopark, and the extensions around the airport terminal. A lot of it, though, is the "built" patch edge creeping into tree canopy.

## Main finding: DW and ESRI "built" is settlement extent, not impervious surface

The site panels show the reason. In Kerala, houses sit under continuous coconut and tree canopy, and DW and ESRI label that whole mixed fabric "built". That is why they put 40–67 % of the AOI in the built class. GHSL (built surface m² per cell) and WorldCover measure actual roofs and paving, about 10–13 % of land. Runoff, and therefore UICA, depends on the impervious **area**, so the DW fraction used directly would overstate imperviousness about four times.

The automatic rule in the script (rank on agreement with WC and on temporal stability) picks DW. But that choice does not answer the question that matters, which is whether the layer represents impervious area. **The source decision is therefore left open for the review.**

## Options for the checkpoint decision

| Option | Imperviousness level | Year-to-year change | Problem |
|---|---|---|---|
| **A. Hybrid (recommended)** | GHSL 2020 built-surface fraction, i.e. physical roof and paving area | DW cleaned yearly series: a cell's GHSL fraction is moved forward or back in time with the change in DW built fraction, scaled by the AOI-wide ratio of impervious to DW built (≈ 147 / 687 ≈ 0.21) | Two sources are combined. The scaling ratio is an assumption, to be stated. |
| B. GHSL only | GHSL | 2015 / 2020 epochs, linearly interpolated per year; 2025 and 2030 are projections | No observed change after 2020, so ΔUICA becomes a model output rather than an observation |
| C. DW only (current script choice) | DW fraction | DW cleaned | Imperviousness about 4× too high, which inflates UICA |

A gives a physically realistic amount of imperviousness with observed annual timing. B is the cleanest single source but is not observed after 2020. C is simplest but tells the model that half of the AOI is paved.

## 2.5 Decision: Option A (hybrid) implemented

`imp_y = clip(GHSL2020 + k · (DW_y − DW_2020), 0, 1)` with `k = ΣGHSL2020 / ΣDW2020 = 0.214`. Figure: `outputs/maps/step2_impervious_hybrid.png`.

| Year | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| Impervious area (km²) | 137.8 | 142.0 | 145.2 | 146.8 | 150.0 | 154.5 | 158.2 | 159.2 | 158.9 |

- **Growth.** +21.1 km² from 2017 to 2025, or 1.79 %/yr. The 2025 value (158.9 km²) is close to GHSL's own 2025 projection (152.9 km²), which is an independent check.
- **Agreement with WorldCover 2021.** r = 0.74 and MAE = 0.12, compared with r = 0.46 and MAE = 0.36 for DW alone. The hybrid is therefore much closer to an impervious-surface map.
- **Limitation.** The change map is speckled, because single 10 m DW pixels that switch class add small increments everywhere. ΔUICA sums this change over whole upstream basins, which averages the speckle out. For the same reason, the per-cell change should not be read as a construction map.
