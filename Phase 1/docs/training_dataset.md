# Training / Testing Dataset (Step 8)

Status as of 2026-09-28. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84).

**Script:** `scripts/12_build_training_dataset.py` (about 25 s)
**Inputs:** Step 5 `water_classes_2018_20180821.tif`, NDEM polygons (`data/validation/`),
Step 7 `feature_stack.tif` + `analysis_domain.tif`
**Dataset:** `data/processed/training/training_samples_2018.csv` (+ `label_pools_2018.tif`)
**Metrics:** `outputs/metrics/step8_training_dataset_summary.json`
**Maps:** `outputs/maps/step8_sample_locations.png`, `outputs/maps/step8_feature_distributions.png`

## Event and label grid

- **Event:** Aug 2018 only. Step 6 rated 2019 weaker (GTI 37 %) and 2021 unusable (15 %).
- **Grid:** samples are on the Step 7 feature grid (~30 m). The Step 5 inventory is on the 20 m Sentinel-1 grid. The two grids are exactly commensurate: the pixel ratio is 1.5, and the 30 m grid starts half a 20 m pixel up-left (both asserted in the script). Each 20 m pixel is split into 2 × 2 sub-pixels of 10 m, so every 30 m cell is exactly 3 × 3 sub-pixels.
- **Labels:** every cell therefore gets the *exact* area fraction of each Step 5 class, with no resampling approximation. The aggregation is checked to conserve flood area.

## 8.1 Label rules and sampling

| Class | Rule | Pool |
|---|---|---|
| **Flood (1)** | ≥ 50 % of the cell is Otsu flood that (a) lies in a patch of ≥ 10 px and (b) is confirmed by the NRSC/ISRO NDEM 2018 event envelope within 40 m | 15,284 cells, 13.45 km² |
| **Non-flood (0)** | no Otsu flood in the cell or within 60 m; outside the NDEM envelope (+40 m); > 60 m from reference (dry-season) water | 1,241,604 cells, 1,092.8 km² |
| Not sampled | unconfirmed or small-patch Otsu flood, mixed cells, the water fringe | – |

Why these rules come straight from Step 6:
- **NDEM confirmation of flood cells.** 58 % of our Otsu flood is NDEM-confirmed; the rest concentrates at water edges and in small patches. Using only the confirmed part removes most of the doubtful positives. Of the 24,789 cells that are ≥ 50 % Otsu flood, 15,284 qualify.
- **Envelope exclusion for non-flood cells.** Our scene (21 Aug) is 3–4 days after the 17–18 Aug peak. Land NDEM mapped as flooded at the peak but dry by the 21st would otherwise be labelled "non-flood". This removes 174.3 km² from the non-flood pool.
- **60 m water fringe.** Step 6 showed the river/backwater edge is where the inventory is least reliable. This removes another 84.6 km².

**Sampling.** Every flood-pool cell is used, plus the same number of non-flood cells drawn uniformly at random (class-stratified, seed 42). That gives **15,284 + 15,284 = 30,568 samples**, the base paper's balanced design.

## 8.2 Dataset

| Columns | Content |
|---|---|
| `label` | 1 = flood, 0 = non-flood |
| 11 features | `elevation_m`, `slope_deg`, `aspect_deg`, `curvature_plan`, `curvature_profile`, `twi`, `dist_to_river_m`, `dist_to_major_river_m`, `dist_to_road_m`, `drainage_density_km_km2`, `rainfall_mean_annual_mm` |
| metadata (not features) | `sample_id`, `row`, `col`, `lon`, `lat`, `x_utm`, `y_utm`, `otsu_flood_frac`, `confirmed_flood_frac`, `block_id` |

`block_id` is a 2 km UTM block (398 blocks; 98 contain flood samples). It is kept so Phase 2 can run spatial block cross-validation without redoing this step.

## 8.3 Data-quality pass: PASS

The pass is run on the CSV as written to disk.

| Check | Result |
|---|---|
| Missing / non-finite values | 0 |
| Negative distances, slope or drainage density | none |
| Aspect in [−1, 360] | yes |
| Every value within the Step 7 layer ranges | yes |
| Class balance | 15,284 / 15,284 |
| Duplicate cells | none |
| All samples on land (Step 7 domain) | yes |
| Labels agree with the pool raster | yes |

**Single-feature separability** (AUC of one feature alone; 0.5 = useless):

| Feature | AUC | Higher in |
|---|---|---|
| Elevation | 0.915 | non-flood |
| TWI | 0.896 | flood |
| Distance to road | 0.832 | flood |
| Slope | 0.817 | non-flood |
| Distance to major river | 0.801 | non-flood |
| Distance to river | 0.781 | non-flood |
| Drainage density | 0.692 | flood |
| Rainfall | 0.653 | non-flood |
| Profile / plan curvature, aspect | 0.51–0.52 | – |

**Leakage check.** No feature is derived from Sentinel-1, so none can encode the label directly, and no single feature exceeds AUC 0.95. The directions are physically sensible: flood cells are low, flat, wet (high TWI) and near the major rivers.

## Caveats for Step 9 (and the report)

1. **Flood samples are spatially clustered. The random 70/30 split in Step 9 will be optimistic.** Every flood sample has a neighbouring flood sample within 45 m (median spacing 29 m), versus 127 m for non-flood samples. A random split puts near-identical neighbours in both train and test. The plan uses this naive split on purpose, for comparability with the reference papers. `block_id` allows a spatial-block check alongside it.
2. **Distance to road (AUC 0.83) is partly a Sentinel-1 artifact.** Flooded paddy and wetland genuinely has few roads. But SAR also cannot see floodwater among buildings, so flood near roads and in built-up areas rarely enters the inventory. In Step 9, a high importance for this feature must be read with that caveat. Correcting this bias is what PU-learning (Phase 2) is for.
3. **Rainfall (AUC 0.65) acts as a location proxy.** CHIRPS has only 60 cells here, and its spiky class distributions follow where the flood patches happen to be, not rain physics (predicted in Step 7).
4. **Non-flood samples include easy hill-country negatives.** They are drawn from the whole AOI, including the eastern hills, which is standard in susceptibility studies. It inflates elevation/slope separability relative to a floodplain-only comparison.
5. **Negatives are "not observed flooded", not "cannot flood".** The envelope and fringe exclusions remove the worst cases, but urban flooding missed by SAR can remain in the non-flood pool. This is the positive-unlabelled problem that Phase 2 targets.
