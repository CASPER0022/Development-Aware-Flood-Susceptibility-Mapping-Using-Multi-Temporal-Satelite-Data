# Step 5: Feature stack v2

Script: `scripts/05_feature_stack_v2.py` · numbers: `outputs/metrics/step5_feature_stack_v2.json` · figures: `outputs/maps/step5_twi_hand.png`, `step5_event_rainfall.png`, `step5_stack_v2_overview.png`, `step5_correlation_vif.png`, `step5_uica_sanity.png` · rasters (gitignored): `data/processed/features_v2/stack_v2_<event>.tif` + `features_v2_metadata.json`

## 5.1 HAND (height above nearest drainage)

- **Method.** Each cell follows its D8 steepest-descent path on the Step 4 full-basin filled DEM until it reaches a drainage cell. HAND is the height of the cell above that drainage cell, using FABDEM elevation.
- **Drainage** means cells with upslope area ≥ **1 km²** (fixed before looking at any labels), plus open water. Every land cell reaches a drainage cell.
- **Single-feature AUC on the Phase 1 2018 samples.** This is a diagnostic only; nothing was tuned on it.

| Feature | All 30,568 samples | Floodplain only (elev ≤ 5.7 m, slope ≤ 2°) |
|---|---|---|
| Elevation | 0.915 | 0.764 |
| TWI (Phase 1) | 0.896 | – |
| TWI (full basin) | 0.896 | 0.810 |
| **HAND (1 km²)** | **0.951** | **0.907** |

HAND does what the plan expected: it beats raw elevation, and the gain is largest on the flat floodplain, where elevation alone is weak. **Sensitivity:** thresholds of 0.25, 5 and 25 km² give AUC 0.934, 0.965 and 0.960. The 1 km² value was kept rather than picking the best one, to avoid tuning a feature on the training labels.

## 5.2 Event and antecedent rainfall

- **Source and windows.** CHIRPS daily (0.05°). For each kept flood event: a 5-day total ending on the Sentinel-1 date, and a 30-day antecedent total before that.
- **Interpolation.** Ocean cells were filled from the nearest land cell, then interpolated bilinearly (the Phase 1 rule).

| Event | 5-day total (AOI mean, range) | 30-day antecedent (AOI mean) |
|---|---|---|
| 2018-08 (S1 21 Aug) | 155 mm (127–177) | 971 mm |
| 2019-08 (S1 10 Aug) | 517 mm (445–612) | 572 mm |
| 2020-08 (S1 10 Aug) | 206 mm (166–239) | 887 mm |
| 2024-08 (S1 1 Aug) | 221 mm (138–307) | 754 mm |

**Note on 2018.** The window ends on the satellite date, so it misses the 14–17 Aug peak (the main rain fell before the 5-day window). The 30-day antecedent total captures it. 2019 is the reverse: a short, very intense burst on dry-ish ground.

Rainfall varies mostly *between* events and only a little *within* the AOI, so these features carry event-level information. Phase 1's mean annual rainfall, by contrast, acted as a location label.

## 5.3 Full-basin TWI, imperviousness, land cover

- **TWI.** Recomputed from the Step 4 accumulation. 5.8 % of AOI cells change, all along the river corridors, by up to +15 units. The AOI median is unchanged (9.14 vs 9.13).
- **Impervious fraction.** The Step 2 hybrid layer for the event year.
- **Land-cover class.** Dynamic World annual mode, aggregated to a 30 m majority. In 2024: trees 38 %, built 51 % (settlement extent, see Step 2), water 6 %, crops 3 %.
- **Soil texture (5.4, optional).** Not added: the 250 m OpenLandMap layer has only a handful of classes over the AOI.

## 5.5 Feature stack v2: one stack per event year

| Group | Features |
|---|---|
| Terrain and hydrology (Phase 1, kept) | elevation, slope, aspect, plan / profile curvature, distance to river, distance to major river, distance to road, drainage density |
| Terrain and hydrology (new) | **twi_full** (replaces twi), **hand_m** |
| Event rainfall (new) | **rain_event_5d_mm**, **rain_antecedent_30d_mm** (replace mean annual rainfall) |
| Development (new) | **impervious_frac**, **uica_km2**, **uica_ratio**, **duica_since_2017_km2**, **landcover_class** (categorical) |

That is 18 features, in four files: `stack_v2_2018-08_ndem.tif`, `stack_v2_2019-08_ndem.tif`, `stack_v2_2020-08_ndem.tif` and `stack_v2_2024-08_ndem.tif`.

**Phase 1 Step 7.5 checks (all events): PASS.** Same grid as Phase 1, 0 gaps on land, and 0 values outside the physical ranges.

**Correlation and VIF.** These were computed on 200,000 pooled cells, with UICA as log10. Max VIF is **9.5** (elevation), below the threshold of 10. Pairs with |r| ≥ 0.7:

| Pair | r | What to do about it |
|---|---|---|
| elevation – HAND | 0.93 | HAND is the better flood feature. In Step 8, test HAND instead of elevation as well as both together. |
| impervious_frac – uica_ratio | 0.88 | Local paving and upstream paving go together in the city. Keep both for the ablation, but read importances carefully. |
| rain_event_5d – rain_antecedent_30d | −0.89 | An artefact of only 4 events (wet-antecedent years had smaller bursts). It is not a physical law. |
| twi_full – uica_km2 | 0.73 | Both grow with upslope area along channels. |

## Step 4 checkpoint: is UICA physically right?

| Area (2024) | Median UICA ratio | Mean impervious fraction | Tree share |
|---|---|---|---|
| Kochi city core | **0.449** | 0.40 | 4 % |
| Eloor–Kalamassery (below Aluva) | **0.238** | 0.21 | 22 % |
| Eastern forested hills | **0.006** | 0.04 | 75 % |

By land-cover class, the median UICA ratio is 0.169 for built cells, 0.065 for crops and 0.037 for trees. **UICA is high below Kochi's built-up areas and near zero in forested catchments, as required.**
