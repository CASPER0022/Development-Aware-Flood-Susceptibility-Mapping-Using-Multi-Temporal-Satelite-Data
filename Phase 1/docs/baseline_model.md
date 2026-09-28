# Baseline LightGBM Model (Step 9)

Status as of 2026-09-28. AOI: Lower Periyar & Aluva-Kochi Urban Corridor.

**Script:** `scripts/13_train_baseline_lightgbm.py` (about 1 min)
**Input:** Step 8 `data/processed/training/training_samples_2018.csv` (30,568 samples, 11 features)
**Model:** `outputs/models/step9_lightgbm_baseline.txt` (LightGBM booster, 648 trees)
**Metrics:** `outputs/metrics/step9_baseline_metrics.json`
**Figures:** `outputs/maps/step9_model_evaluation.png`, `outputs/maps/step9_feature_importance.png`

## Set-up (9.1, 9.2)

- **Split:** simple random 70/30, stratified by class, seed 42 (21,397 train / 9,171 test). This is deliberately the "naive" evaluation used in the reference papers, so the numbers are directly comparable to them.
- **Model:** LightGBM binary classifier with fixed, conventional settings: learning rate 0.05, 31 leaves, 20 samples per leaf, 90 % feature and 80 % row bagging, L2 1.0. **Nothing is tuned on the test set.**
- **Number of trees (648):** chosen by 5-fold CV with early stopping on AUC *inside the training 70 % only* (CV AUC 0.9993), then the model is refit on the whole training part.
- **Classification threshold:** 0.5.

## Results (9.3)

| | Accuracy | Precision | Recall | F1 | AUC | Specificity |
|---|---|---|---|---|---|---|
| Train (70 %) | 100.00 | 100.00 | 100.00 | 100.00 | 1.0000 | 100.00 |
| **Test (30 %), random split: plan headline** | **99.06** | **98.49** | **99.65** | **99.07** | **0.9993** | 98.47 |
| Random 5-fold CV (all data) | – | – | – | 99.18 ± 0.11 | 0.9995 ± 0.0002 | – |
| Base paper, U-Net test (Table 3) | 93.73 | 94.07 | 86.00 | 89.85 | 0.93 | – |

Test confusion matrix: TN 4,516 · FP 70 · FN 16 · TP 4,569.

## Checkpoint: is 0.999 AUC leakage?

The plan warns that "suspiciously perfect scores like 0.999 AUC usually mean data leakage". We hit exactly that, so it was investigated. **It is not feature leakage. It is spatial autocorrelation plus an easy task, and the honest performance on unseen areas is lower.**

1. **No leakage by construction.** No feature is derived from Sentinel-1 (the source of the labels). No metadata column is used as a feature: coordinates, ids, block and label fractions are all excluded, and this is asserted in the script.
2. **The random split tests on neighbours of training pixels.** 98.7 % of flood test samples have a training flood sample in the adjacent 30 m cell; the median distance is 29 m. For non-flood samples the share is only 6.7 % (median 153 m). Because flood patches are sampled completely (Step 8), the test set is mostly near-duplicates of training pixels.
3. **Holding out whole areas.** StratifiedGroupKFold over square UTM blocks:

   | Held-out unit | Blocks | Accuracy | Precision | Recall | F1 | AUC |
   |---|---|---|---|---|---|---|
   | Random 70/30 (headline) | – | 99.06 | 98.49 | 99.65 | 99.07 | 0.999 |
   | 2 km blocks | 398 | 93.5 ± 2.8 | 98.3 ± 0.5 | 88.6 ± 5.6 | 93.1 ± 3.2 | 0.993 ± 0.003 |
   | 5 km blocks | 73 | 89.8 ± 7.6 | 97.3 ± 1.6 | 81.5 ± 14.8 | 88.0 ± 10.5 | 0.986 ± 0.012 |
   | **10 km blocks** | 22 | **88.5 ± 9.5** | **95.2 ± 5.2** | **78.5 ± 19.0** | **84.5 ± 13.3** | **0.979 ± 0.016** |

   AUC stays high: the model genuinely *ranks* flood-prone terrain well in areas it has never seen. But accuracy and F1 drop by 10–15 points, mostly through recall (floods in a new region are under-called at the 0.5 threshold). **Unseen-region performance (≈ 88 % accuracy, F1 ≈ 84 %) is the honest number, and it is comparable to the base paper's 93.7 % / 89.9 %.** The large fold-to-fold spread at 10 km (only 22 blocks) shows performance depends on which region is held out.
4. **The easy part of the task.** Non-flood samples include the whole AOI, including the eastern hills; floods lie on low ground. Elevation alone gives AUC 0.915. On the **hard subset**, low flat floodplain only (elevation ≤ 5.67 m, the flood 95th percentile, and slope ≤ 2°; 18,559 samples), the unseen-10 km predictions still reach **AUC 0.935, accuracy 81.0 %**. Elevation alone manages only 0.76 there. The model has learned more than "low ground floods".
5. **Overfitting.** Train metrics are perfect (648 trees of 31 leaves memorise the training pixels). On the random split this barely shows (train–test AUC gap 0.0007) because the test pixels are neighbours; it shows up in the spatial-hold-out numbers above. Stronger regularisation or fewer trees is a Phase 2 tuning question, to be decided under spatial CV rather than on this split.

## Feature importance (9.4)

| Feature | Gain share | Permutation ΔAUC (test) |
|---|---|---|
| Elevation | 42.1 % | 0.137 |
| TWI | 15.4 % | 0.002 |
| Distance to road | 12.9 % | 0.004 |
| Rainfall (mean annual) | 11.8 % | 0.021 |
| Distance to major river | 9.3 % | 0.010 |
| Drainage density | 3.6 % | 0.002 |
| Distance to river | 2.9 % | 0.001 |
| Plan / profile curvature, slope, aspect | ≤ 0.6 % each | ≈ 0 |

How to read it:
- **Elevation dominates by both measures.** It is physically expected for fluvial flooding on a coastal floodplain.
- **TWI** has high gain but low permutation drop, because it overlaps with elevation. When TWI is shuffled, elevation compensates.
- **Rainfall ranks #2 by permutation, but for the wrong reason.** On its own it scores AUC 0.82 on the random split but **0.58 on unseen 10 km regions** (near chance). Removing it *improves* unseen-region accuracy, from 88.5 % to 89.7 %. It acts as a location label (60 CHIRPS cells), as predicted in Steps 7–8, not as rain physics. The plan's list includes rainfall, so it stays in the baseline, but this should be said when presenting it.
- **Distance to road** is useful when transferring to new regions: removing it lowers unseen-region AUC from 0.979 to 0.961. Part of its signal is SAR's blindness to urban flooding (Step 8 caveat 2).
- **Slope** matters little *given elevation*, although on its own it separates the classes (AUC 0.82 in Step 8).

## Comparison with base-paper Table 4

The plan describes Table 4 as a feature ranking, but in the paper it is an **input ablation**: the U-Net retrained on 8 combinations of slope, HSG (soil groups), imperviousness and rainfall. Test values span:
- Precision 90.03–94.07 %
- Recall 83.27–86.00 %
- F1 86.52–89.85 %
- Overall accuracy 91.57–93.73 %

The best row is all four inputs; the worst row is rainfall only. The paper names slope and imperviousness as the most useful. The same kind of ablation here:

| Inputs | P | R | F1 | OA | AUC | Unseen 10 km: AUC | Unseen 10 km: OA |
|---|---|---|---|---|---|---|---|
| All 11 features | 98.49 | 99.65 | 99.07 | 99.06 | 0.999 | 0.979 | 88.5 |
| Plan minimum set (elev, slope, TWI, d-river, d-road, rain) | 97.66 | 99.17 | 98.41 | 98.40 | 0.998 | 0.976 | 86.8 |
| Without terrain | 97.32 | 98.95 | 98.13 | 98.11 | 0.998 | 0.938 | 78.9 |
| Without hydrology | 97.16 | 98.52 | 97.83 | 97.82 | 0.997 | 0.967 | 86.5 |
| Without distance to road | 98.66 | 99.69 | 99.18 | 99.17 | 0.999 | 0.961 | 86.9 |
| Without rainfall | 97.02 | 99.41 | 98.20 | 98.18 | 0.997 | **0.983** | **89.7** |
| Slope + rainfall | 80.96 | 87.46 | 84.08 | 83.45 | 0.911 | 0.729 | 64.9 |
| Rainfall only | 71.63 | 79.15 | 75.20 | 73.91 | 0.816 | 0.575 | 55.1 |

**Agreements with the base paper:**
- More inputs give better results.
- Rainfall alone is the weakest case.

**Differences:**
1. Here terrain (elevation), not slope, carries the model. Miami is flat pluvial flooding; the Periyar floodplain is fluvial.
2. The base paper's second key input, **imperviousness, has no counterpart yet**. That is exactly where the Phase 2 UICA / land-use-change features come in.
3. Under the random split every feature set with elevation looks near-perfect. Only the unseen-region columns separate them. The base paper's Table 4 ablation would likely also look different under spatial hold-out.

## What to tell the panel

- "Random-split test: 99.1 % accuracy, F1 99.1 %, AUC 0.999. We checked why it is so high. Test pixels are neighbours of training pixels, which is the known optimism of random splits. On whole 10 km regions the model never saw, accuracy is 88.5 % and F1 84.5 %, with AUC still 0.98. That is comparable to the base paper's 93.7 % / 89.9 %."
- "Feature importance: elevation dominates, then TWI, distance to road, rainfall and distance to the Periyar. Rainfall's rank is a location artifact of 5.5 km CHIRPS; dropping it improves transfer."
- "Spatial CV is therefore not optional. It becomes the evaluation method in Phase 2, and the 2 km `block_id` column is already in the dataset."

## Open decision for Step 10

The susceptibility map (Step 10) should use one of two models:
- **(a)** this baseline with all 11 features, which reproduces the plan exactly; or
- **(b)** the same model without rainfall, which transfers better to unseen areas (89.7 % vs 88.5 %; AUC 0.983 vs 0.979) and avoids 5.5 km blockiness in the map.

The recommendation is (a) as the headline, for comparability, with (b) shown alongside.
