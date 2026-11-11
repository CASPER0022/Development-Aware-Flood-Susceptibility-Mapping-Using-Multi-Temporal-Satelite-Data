# Step 1: Freeze the baseline and set up Phase 2

Script: `scripts/01_freeze_baseline_check.py` · numbers: `outputs/metrics/step1_baseline_reproduction.json`

## What was done

1. **Git tag.** `phase1-final` (annotated, on commit `cd1284e`) marks the Phase 1 state that every Phase 2 gain is measured against. The tag is local and is pushed after review (`git push origin phase1-final`).
2. **Folders.** The Phase 2 layout mirrors Phase 1: `scripts/`, `docs/`, `data/raw/`, `data/processed/` (both gitignored), and `outputs/{maps,metrics,models,presentation}/`.
3. **Shared module.** `scripts/common.py` holds every piece of Phase 1 code that Phase 2 reuses. The logic is unchanged and there is one copy of each:

   | Group | Functions | From |
   |---|---|---|
   | Grid and areas | `row_cell_sizes_m`, `pixel_area_km2`, `area_km2`, `open_water_and_land` | Steps 5–7 |
   | Terrain | `horn_gradient`, `slope_aspect`, `twi_from_acc` | Step 7 |
   | Flow routing | `priority_flood_eps`, `quinn_fd8`, `route_flow`, and a new `quinn_fd8_weighted` for UICA | Step 7 |
   | Labels | `to_30m_fraction`, `check_20_30_commensurate`, `dilate`, `remove_small_patches` | Steps 5, 8 |
   | Sentinel-1 | `load_s1`, `preprocess` (dB check, Lee filter, Eq. 1), `otsu_threshold` (now also returns the separability η) | Step 5 |
   | Cross-check | `load_ndem`, `ndem_mask`, `agreement` (GTI and contingency table) | Step 6 |
   | Modelling | `LGB_PARAMS`, `metrics`, `rounded`, `cv_summary`, `block_groups` | Step 9 |
   | Figures and GEE | `basemap` (Esri tiles; the Phase 1 tile cache is shared), `ee_download_grid` (downloads onto an exact pixel grid, in tiles, and asserts the grid) | Steps 5–6, new |

   `quinn_fd8` now calls the shared `_fd8_core`, so the weighted version needed for UICA (Step 4) uses exactly the same routing.

## [CHECKPOINT] Phase 1 reproduced through `common.py`

| Check | Result | Phase 1 value |
|---|---|---|
| A1 land domain | identical, 1,640,815 cells | same |
| A2 flow accumulation | max relative difference 0.0 | Step 7 raster |
| A3 TWI | max absolute difference 0.0 | Step 7 raster |
| B1 20 m → 30 m area conservation | error 0 sub-pixels | exact |
| B2 2018 flood fraction at all 30,568 samples | max diff 4.4 × 10⁻⁵ (CSV rounding to 6 digits) | Step 8 CSV |
| C1 GTI 2018 (NDEM envelope, 40 m) | 58.44 %, lift 4.38 | 58.44 % |
| D1 boosting rounds (CV inside train) | 648 | 648 |
| D2 test metrics | AUC 0.9993, OA 99.06 %, F1 99.07 %, confusion TN 4516 / FP 70 / FN 16 / TP 4569 | identical |
| D3 every one of 9,171 test predictions | max difference 0.0 | Step 9 CSV |
| D4 10 km spatial blocks | AUC 0.9788, OA 88.49 % | identical |

**10 / 10 checks pass (about 30 s).** The refactor changed no number. If anything in `common.py` changes later, this script must be rerun.

## Review 2 feedback

`docs/review2_feedback.md` holds the table for the panel's comments. The comments themselves still have to be entered by the team.
