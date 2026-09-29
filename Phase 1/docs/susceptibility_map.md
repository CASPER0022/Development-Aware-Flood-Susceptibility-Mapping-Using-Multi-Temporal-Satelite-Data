# Flood Susceptibility Map (Step 10)

Status as of 2026-09-29. AOI: Lower Periyar & Aluva-Kochi Urban Corridor.

**Script:** `scripts/14_susceptibility_map.py` (about 30 s)
**Inputs:** Step 7 feature stack + domain, Step 9 model, Step 8 samples (for the no-rainfall variant), Step 5 2019 inventory, NDEM 2019 polygons
**Rasters:** `data/processed/susceptibility/susceptibility_{prob,class}_{all,no_rainfall}.tif`
**Model (variant):** `outputs/models/step10_lightgbm_no_rainfall.txt`
**Metrics:** `outputs/metrics/step10_susceptibility_summary.json`
**Maps:**
- `outputs/maps/step10_susceptibility_map.png`: headline, all 11 features
- `outputs/maps/step10_susceptibility_map_no_rainfall.png`
- `outputs/maps/step10_susceptibility_comparison.png`

## 10.1 Continuous susceptibility

The Step 9 LightGBM baseline is applied to all **1,640,815 land pixels** (1,444.8 km², ~30 m). The model is reloaded from disk, and the script checks it reproduces the Step 9 test AUC exactly (0.9993). Open water stays NoData.

As agreed at Step 9, a **no-rainfall variant** is mapped alongside: the same split, parameters and 648 trees, with rainfall removed. Its random-split test AUC is 0.9973 and accuracy 98.18 %.

**Scores are a relative index, not flood probabilities.**
- The model was trained on balanced 50/50 samples, although flooding covers about 1–2 % of the land.
- It is also overconfident: it fits the training data perfectly (Step 9).
- As a result, 87 % of land scores below 0.01, and only 7 % above 0.5.

The ranking is what carries the information, and the validation below tests exactly that.

## 10.2 Five quantile classes

Breaks are the 20/40/60/80th percentiles of the land pixels (reference-paper convention), so every class covers exactly 20 % of the land.

| Class | Score range, all 11 features | Score range, without rainfall |
|---|---|---|
| Very Low | 0 – 2.2×10⁻⁶ | 0 – 3.4×10⁻⁶ |
| Low | 2.2×10⁻⁶ – 7.4×10⁻⁶ | 3.4×10⁻⁶ – 1.4×10⁻⁵ |
| Moderate | 7.4×10⁻⁶ – 3.8×10⁻⁵ | 1.4×10⁻⁵ – 8.3×10⁻⁵ |
| High | 3.8×10⁻⁵ – 8.6×10⁻⁴ | 8.3×10⁻⁵ – 3.9×10⁻³ |
| Very High | 8.6×10⁻⁴ – 1 | 3.9×10⁻³ – 1 |

Because of the saturated scores, the lower four classes are carved out of scores the model considers "not flood". The two models put 63.8 % of land in the same class and 97.5 % within one class (Spearman r = 0.92).

## 10.3 Map

The publication map (`step10_susceptibility_map.png`) has:
- a legend with score ranges, a scale bar and a north arrow;
- major rivers and open water;
- the grid and source credits.

It is drawn at **true scale** (aspect 1/cos latitude). Plain lon/lat axes would stretch it about 1.5 % east–west.

## Validation on an unseen flood: Aug 2019

Labels came only from the Aug 2018 flood, so the Aug 2019 flood is an independent test. The comparison domain is land minus 2019 dry-season water.

**Frequency ratio** = share of 2019 flood in a class ÷ share of land in that class. 1 means no better than chance.

| Reference (2019) | Model | AUC | In High + Very High (40 % of land) | FR: VL / L / M / H / VH |
|---|---|---|---|---|
| NDEM flood, 10 + 12 Aug (32.7 km²) | all 11 | **0.830** | **83.8 %** | 0.07 / 0.23 / 0.48 / 0.99 / **3.54** |
| | no rainfall | 0.805 | 80.1 % | 0.12 / 0.32 / 0.53 / 0.97 / 3.35 |
| NDEM flood **outside the whole 2018 envelope** (9.1 km²) | all 11 | 0.779 | 78.1 % | 0.11 / 0.33 / 0.62 / 1.09 / 3.12 |
| | no rainfall | 0.762 | 76.0 % | 0.17 / 0.39 / 0.60 / 1.10 / 2.98 |
| Our Otsu inventory, 10 Aug (26.8 km²) | all 11 | 0.962 | 98.8 % | 0.00 / 0.01 / 0.05 / 0.23 / 5.20 |

What this shows:
- **The map predicts a different year's flood.** 84 % of the official 2019 flood falls in the 40 % of land rated High/Very High. The Very High class holds 3.5× its area share, and the frequency ratio rises steadily from Very Low to Very High.
- **It also works where 2018 did not flood.** For 2019 flooding in places outside the entire 2018 NDEM envelope, the AUC is still 0.78. The model has learned terrain, not just last year's footprint.
- **Only Very High is strongly predictive.** The High class sits at FR ≈ 1.0, which is chance. This is the saturation effect: quantile classes force 40 % of the land into High/Very High, while the model flags about 7 %.
- **The Otsu 2019 AUC (0.96) is higher than NDEM's.** That comparison uses the same sensor and method as the training labels, so it is not fully independent. NDEM is the fair test.

## Physical plausibility: all-feature map vs no-rainfall map

| Share of such land rated High / Very High | All 11 features | Without rainfall |
|---|---|---|
| Land above 50 m | 23.4 % | **13.4 %** |
| Slopes steeper than 10° | 24.1 % | **13.9 %** |

**In the all-feature map, rainfall imprints CHIRPS cells.** This is clearest as a rectangle in the NE hills with a straight edge near 10.21° N (see panel (a) of the comparison figure). The top CHIRPS row has the AOI's lowest rainfall, the same range as the 2018 flood patches, so the model rates hills there as susceptible.

Removing rainfall roughly halves the high-ground false alarms. The cost is small: the 2019 AUC falls from 0.830 to 0.805. The 2019 flood lies in the same low-rainfall northern floodplain, so for *this* AOI the location proxy slightly helps temporal validation, while Step 9 showed it hurts spatial transfer.

## Recommendation for the review

Present **both**:
- **Headline:** the all-feature map, reproducing the plan (best 2019 AUC 0.83).
- **Alongside it:** the no-rainfall map, which is physically cleaner and transfers better spatially.

Explain the rainfall artifact as a finding, not an error. It motivates Phase 2's finer rainfall data (event-based and higher-resolution) and spatial CV.

## Known limitations of this map

1. **Scores are uncalibrated,** because of balanced training and overfitting. Read classes as relative ranks, not probabilities. Calibrating the scores to a realistic flood rate is a Phase 2 task.
2. **Quantile classes are equal-area by construction.** "High" does not mean likely to flood; only Very High is strongly predictive on 2019.
3. **Urban flooding is under-represented.** Sentinel-1 cannot see water among buildings (Steps 5 and 8), so Kochi city tends toward lower classes. PU-learning in Phase 2 targets this.
4. **One training event** (2018). No land-use or imperviousness feature is included yet (the Phase 2 UICA hook).
5. **Rainfall artifact** in the headline map, as described above.
