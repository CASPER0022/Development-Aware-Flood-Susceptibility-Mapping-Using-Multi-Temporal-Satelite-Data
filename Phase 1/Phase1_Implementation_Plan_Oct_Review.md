# Phase 1 Implementation Plan — Base Paper Reproduction
**Target: Review 2, October 2nd week (Oct 12–18, 2026)**
**Today: September 13, 2026 → ~4.5 weeks available**

## How to use this document
- Work through the numbered steps **in order** — each step depends on the previous one's output.
- After you finish each step, message me with what you did, your code/output, and any numbers/plots you got. I will give you an honest review: what's correct, what's wrong, what's weak, and what to fix before moving on.
- Steps marked **[CHECKPOINT]** are hard stops — don't move past them until the output looks right, since every later step builds on them.
- Realistic scope for this review: **reproduce the base paper's pipeline (Mohamadiazar et al., 2024) adapted to the Periyar basin**, using a LightGBM/XGBoost baseline instead of full U-Net (explained in Step 0). The novel contributions (UICA, PU-learning, dual prediction) start *after* this review, in Phase 2.

---

## Step 0 — Scope decision: why not U-Net first (read this before starting)

The base paper uses a U-Net CNN trained on 226 Sentinel-1 images over 10 years for Miami-Dade. Replicating that exactly, for Periyar basin, in 4 weeks, without an existing labeled dataset, is not realistic — image collection, patch generation, and CNN training/tuning alone would eat the whole timeline, leaving nothing to show at review.

**Decision:** For this review, reproduce the base paper's *logic* (satellite → preprocessing → flood inventory → ML model → susceptibility/flood map → accuracy metrics) using a tabular ML model (LightGBM or XGBoost) on pixel/point-based features, the same approach used in your reference papers (Kendagannaswamy et al., Waleed & Sajjad). This is faster to build, easier to debug, and still directly comparable to your literature review table.

U-Net / deep learning can be revisited in a later phase once the pipeline and data are proven to work. **State this decision explicitly to the panel** — it shows you made a deliberate, justified engineering trade-off, not that you skipped the base paper.

---

## WEEK 1 (Sep 14 – Sep 20): Study area, environment, and data acquisition

### 1. Define and finalize the study area boundary
1.1. Decide the exact Periyar basin extent (either full basin from HydroSHEDS, or a focused sub-area like Kochi + surrounding urbanizing wards — smaller is safer given your timeline).
1.2. Export the boundary as a shapefile/GeoJSON. This is your master AOI (Area of Interest) used in every later step.
1.3. Confirm the boundary against the 2018 Kerala flood extent — make sure your AOI actually contains flooded area, or your flood inventory will be empty.

**[CHECKPOINT]** Share the AOI map before proceeding — a wrong boundary invalidates everything downstream.

### 2. Set up the working environment
2.1. Set up Google Earth Engine (GEE) account (needed for Sentinel-1/2 and CHIRPS access without downloading huge files manually).
2.2. Set up a Python environment (recommend: `conda` or `venv`) with: `geopandas`, `rasterio`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `shap`, `matplotlib`.
2.3. Set up QGIS or ArcGIS Pro (whichever you have license/access to) for visual inspection and manual QA of rasters — don't skip this, visually checking your data catches mistakes that code alone won't.
2.4. Create a clean folder structure, e.g.:
```
project/
  data/raw/
  data/processed/
  notebooks/
  outputs/maps/
  outputs/models/
  docs/
```

### 3. Acquire flood-influencing datasets
3.1. **DEM:** Download FABDEM (30 m, forest/building-removed) for the AOI — this fixes the SRTM canopy-bias limitation you noted from Kendagannaswamy et al.
3.2. **Rainfall:** Pull CHIRPS daily rainfall (via GEE) for the AOI, at least for 2017–2023.
3.3. **Rivers/drainage:** Download HydroSHEDS river network + drainage basins for Periyar.
3.4. **Roads:** Download OSM road network for the AOI.
3.5. **Land cover:** Identify source — either Sentinel-2 (for you to classify yourself later) or ESA WorldCover / ESRI LULC as a ready-made reference layer.
3.6. **Soil (optional but recommended):** Check if NBSS&LUP soil data is accessible for Kerala; if not, note this as a limitation (like the base paper's HSG).

**[CHECKPOINT]** Confirm every dataset actually downloads and opens correctly (no corrupt files, correct projection) before Week 2.

---

## WEEK 2 (Sep 21 – Sep 27): Sentinel-1 acquisition and flood inventory generation

### 4. Acquire Sentinel-1 SAR imagery
4.1. Identify known flood dates in your AOI — primarily August 2018, and any other major flood events (2019, 2021 if applicable) — using news reports, Kerala SDMA records, or Copernicus EMS as reference.
4.2. In GEE, pull Sentinel-1 GRD (VV polarization, matching the base paper) scenes closest to each flood date, plus a "dry" reference scene from a non-flood period for comparison.
4.3. Export these scenes (or process directly in GEE) clipped to your AOI.

### 5. Build the flood inventory using Otsu thresholding
5.1. Convert Sentinel-1 backscatter to dB (log transform), as done in the base paper (Eq. 1 in their paper).
5.2. Apply Otsu's thresholding method on the VV band to separate water/flooded pixels from dry pixels — this reproduces the base paper's core flood-detection step.
5.3. Generate a binary flood/no-flood raster for each flood date.
5.4. Visually validate: overlay your binary flood map on a basemap/satellite image — do the "flooded" pixels actually correspond to known flooded areas (e.g., low-lying Kochi wards, riverbanks)? Do NOT skip this — a wrong threshold silently ruins every later step.

**[CHECKPOINT]** Share the flood inventory map for at least one date (ideally Aug 2018) with a visual sanity check before continuing.

### 6. Cross-check the flood inventory against an independent source
6.1. Compare your Otsu-derived flood extent against at least one independent source — Copernicus Emergency Management Service (EMS) rapid mapping products for the 2018 Kerala floods, or the Dartmouth Flood Observatory (DFO), if available for your AOI.
6.2. Compute a rough agreement percentage (like the base paper's "Ground Truth Index") — this becomes one of your first real evaluation numbers to report.

---

## WEEK 3 (Sep 28 – Oct 4): Feature engineering and baseline model training

### 7. Generate terrain and hydrological features
7.1. From the FABDEM: compute slope, aspect, and curvature.
7.2. Compute Topographic Wetness Index (TWI) — used in 3 of your 4 reference papers, an essential baseline feature.
7.3. Compute Euclidean distance-to-river and distance-to-road rasters (from HydroSHEDS/OSM layers).
7.4. Compute drainage density if time allows (optional for this review, but note it as pending if skipped).
7.5. Resample/align **all** feature rasters to the same resolution and grid as your DEM (this alignment step is where most bugs happen — check pixel counts and extents match exactly across all layers).

**[CHECKPOINT]** Confirm all feature rasters stack correctly (same shape, same CRS, no NaN gaps) before generating your training dataset.

### 8. Build the training/testing dataset
8.1. Sample flooded pixels (label = 1) from your flood inventory (Step 5) and an equal number of non-flooded pixels (label = 0) — stratified random sampling, matching the base paper's balanced-sampling approach.
8.2. Extract the feature values (elevation, slope, TWI, distance-to-river, distance-to-road, rainfall) at each sampled point into a single tabular dataset (CSV or DataFrame).
8.3. Do a basic data-quality pass: check for missing values, obviously wrong values (e.g., negative distances), and class balance.

### 9. Train the baseline ML model
9.1. Split data using a **simple random 70/30 split first** (deliberately reproducing the same "naive" evaluation method used in your reference papers — this gives you a directly comparable number).
9.2. Train a LightGBM (or XGBoost) binary classifier on the features.
9.3. Evaluate using Accuracy, Precision, Recall, F1, and AUC-ROC — the same metrics reported in the base paper and Kendagannaswamy et al.
9.4. Generate a feature importance plot (built into LightGBM/XGBoost) — compare it against the base paper's Table 4 findings (rainfall, elevation, distance-to-river typically dominate).

**[CHECKPOINT]** Share your metrics table and feature importance plot — I will check whether the numbers are plausible (suspiciously perfect scores like 0.999 AUC usually mean data leakage, not a good model) and whether feature importance makes physical sense.

---

## WEEK 4 (Oct 5 – Oct 11): Validation write-up, susceptibility map, and review prep

### 10. Generate the full-area susceptibility map
10.1. Apply the trained model to every pixel in the AOI (not just sampled points) to generate a continuous 0–1 flood susceptibility raster.
10.2. Classify into 5 susceptibility classes (Very Low → Very High) using quantile breaks, matching the convention used in your reference papers.
10.3. Produce a clean, labeled susceptibility map (with legend, scale bar, north arrow) for the AOI.

### 11. Document honest limitations of this baseline
11.1. Write a short "Known Limitations" note for yourself (and the panel): random split (not spatial CV — this comes in Phase 2), no PU-learning yet, no UICA/land-use-change features yet, single/few flood dates rather than multi-year inventory, LightGBM baseline instead of U-Net.
11.2. This is not a weakness to hide — stating it clearly shows the panel you understand exactly where Phase 1 ends and your novel Phase 2 contributions begin.

### 12. Prepare Review 2 materials
12.1. Update your architecture diagram to mark which blocks are **done** (data acquisition, preprocessing, flood inventory, baseline model) vs. **planned** (UICA/ΔUICA, PU-learning, spatial CV, dual prediction, explainability).
12.2. Prepare 3–4 result slides: AOI map, flood inventory map, susceptibility map, metrics table + feature importance plot.
12.3. Prepare answers for likely panel questions:
   - "Why LightGBM and not U-Net like the base paper?" → Step 0 justification.
   - "How do you know your flood inventory is correct?" → Step 6 cross-validation number.
   - "Is your model actually good, or just overfit?" → mention the random-split limitation (Step 11) and that spatial CV is the immediate next step.
   - "What's left to do?" → Phase 2 roadmap (UICA, PU-learning, dual prediction).

**[CHECKPOINT — FINAL]** Send me the complete set: AOI, flood inventory validation, feature stack, model metrics, susceptibility map, and limitations note, by **Oct 9–10** at the latest, so there's buffer time for fixes before your Oct 12 review.

---

## Quick reference: what "done" looks like by review date

| # | Deliverable | Must-have for review |
|---|---|---|
| 1 | AOI boundary | Yes |
| 2 | Flood inventory (≥1 date, Otsu-derived) | Yes |
| 3 | Independent cross-check of flood inventory | Yes (even if approximate) |
| 4 | Feature stack (DEM, slope, TWI, dist-to-river, dist-to-road, rainfall) | Yes |
| 5 | Baseline LightGBM/XGBoost model + metrics | Yes |
| 6 | Full-AOI susceptibility map | Yes |
| 7 | Feature importance plot | Yes |
| 8 | Written limitations note | Yes |
| 9 | UICA/ΔUICA features | No — Phase 2 |
| 10 | PU-learning | No — Phase 2 |
| 11 | Spatial block CV | No — Phase 2 (mention as immediate next step) |
| 12 | Dual (current vs. post-development) prediction | No — Phase 2 |

---

**Reminder:** send updates after each checkpoint, not just at the end. Catching a bad flood-inventory threshold in Week 2 costs you an afternoon to fix; catching it in Week 4 costs you the whole review.
