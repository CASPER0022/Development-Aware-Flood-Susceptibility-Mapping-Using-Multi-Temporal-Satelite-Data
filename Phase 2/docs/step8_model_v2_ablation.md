# Step 8: Model v2 and the development ablation

Scripts: `scripts/08_model_v2_ablation.py`, `scripts/08b_uica_diagnostics.py` · numbers: `outputs/metrics/step8_model_v2_ablation.json`, `step8b_uica_diagnostics.json` · models: `outputs/models/model_v2_{A,C,D}.txt` · figures: `outputs/maps/step8_ablation.png`, `step8_tuning_gap.png`, `step8_calibration.png`, `step8_maps_A_vs_chosen.png`, `step8b_uica_diagnostics.png`

## 8.1 Tuning under spatial CV

- **Grid:** 18 settings: `num_leaves` {15, 31, 63} × `min_data_in_leaf` {20, 200, 1000} × `lambda_l2` {1, 10}. Learning rate 0.05; rounds by early stopping.
- **Data:** tuning used **only 2018 + 2019** with the 10 km folds, so 2020 and 2024 stay unseen.
- **Result:** every setting scored between 0.993 and 0.994. The best is 31 leaves, min 1000 per leaf, L2 = 1, 336 rounds.
- **Train–test gap:** 0.0071 with Phase 1 settings and 648 rounds, 0.0065 tuned. Train AUC is still about 1.0.

So tuning hardly matters. The data, not the settings, limit the model. We keep the tuned settings, but the "smaller gap" goal is met only marginally.

## 8.3 Ablation: same folds, same tuned settings

| Model | Features | 10 km CV AUC | Strict 2020 (flood vs all land), standard / PU bagging | Strict 2024, standard / PU bagging | Random-split AUC |
|---|---|---|---|---|---|
| A | Phase 1 (11) | 0.967 | 0.956 / 0.969 | 0.960 / 0.968 | 0.998 |
| B | A + HAND, full TWI, event rain (15) | 0.985 | 0.963 / 0.976 | 0.965 / 0.973 | 0.999 |
| C | B + impervious fraction, land cover (17) | 0.992 | 0.977 / **0.985** | 0.978 / **0.985** | 0.999 |
| D | C + UICA, UICA ratio, ΔUICA (20) | 0.992 | 0.977 / **0.985** | 0.978 / 0.984 | 0.999 |
| D − annual rain | D without mean annual rainfall | 0.993 | 0.970 / 0.983 | 0.969 / 0.982 | 0.999 |
| XGBoost D (8.2) | D | 0.991 | 0.976 | 0.975 | – |

**Reading.**
1. **Each Phase 2 upgrade except UICA adds real, held-out skill.**
   - Hydrology (B) lifts the strict AUC by about 0.006.
   - Land cover and imperviousness (C) add about 0.013.
   - Total A → C: 0.956 → 0.977 (standard) and 0.969 → 0.985 (PU bagging).
2. **PU bagging beats the standard design for every feature set,** by 0.007–0.013 on the strict test.
3. **D = C.** UICA and ΔUICA add **no measurable skill**: −0.0001 standard, −0.0002 PU. As the plan asks, we report this and investigate it (8b below).
4. **The random split hides everything:** all models score about 0.999 there. The differences appear only on held-out blocks and years. This is the plan's location-proxy rule in action.
5. **Annual rainfall is still a mild location proxy.** Removing it costs 0.007 on the strict test, though not on spatial CV. We keep it in the model and flag it.
6. **XGBoost on D ≈ standard LightGBM on D** (0.976 / 0.975). The model family is not the bottleneck.

**Chosen model:** **PU bagging on feature set C** (strict AUC 0.985 / 0.985). Model D is kept for the Week 4 development scenarios.

## 8b Why UICA adds nothing (diagnostics)

| | Flood cells (labels) | Random land | Kochi core |
|---|---|---|---|
| Median UICA ratio | 0.050 | 0.073 | **0.450** |
| Share with UICA ratio ≥ 0.10 | 11 % | 41 % | 96 % |
| Median impervious fraction | 0.00 | 0.03 | 0.43 |

- **UICA's share of model D's gain is 0.8 %.**
- **On urban-influenced test land** (UICA ratio ≥ 0.10: 52,865 cells, 890 floods): C 0.962 vs D 0.963.

**Cause.** The radar labels sit almost entirely where UICA is near zero: rural floodplain, rice fields, river banks. The places where UICA is large (paved city catchments) never appear as flooded, because Sentinel-1 cannot see water between buildings. A feature cannot earn skill on labels that never vary with it.

This is a **label limitation, not a UICA failure**. UICA is physically right (Step 4 checkpoint). The fix is urban flood labels: SAR double-bounce, NDEM urban polygons, or reported waterlogging points. That is the top Phase 3 candidate. Until then, Week 4's dual prediction shows the effect of development through impervious fraction and land cover, with UICA reported as a scenario quantity.

## 8.4 Calibration (chosen model)

- **Fitted on:** out-of-fold scores of 120,000 random land cells of 2018 + 2019, where 0.81 % is flooded.
- **Tested on:** 120,000 random land cells of 2020 + 2024, where 0.37 % is flooded.

| Method | Brier | ECE | Mean predicted vs observed |
|---|---|---|---|
| Raw balanced score | 0.0308 | 0.0508 | 5.4 % vs 0.37 % |
| **Prior correction** (balanced → land prevalence) | **0.0028** | **0.0001** | 0.38 % vs 0.37 % |
| Platt | 0.0032 | 0.0045 | 0.81 % vs 0.37 % |
| Isotonic | 0.0032 | 0.0044 | 0.81 % vs 0.37 % |

- Prior correction turns the relative index into a probability: *"P(this cell is inside the confirmed flood extent of an event like these)"*.
- Platt and isotonic learn the **training events' flood size** (2018/2019 were twice as large as 2020/2024), so they over-predict by about 2×.
- The near-perfect prior-correction ECE is partly a coincidence of the event sizes. The real lesson: **a calibrated probability depends on event magnitude.** Week 4 / Phase 3 should condition it on the event rainfall.

## 2024 map: A vs chosen model

- 45 % of land changes class between the two maps.
- Built land in High + Very High falls from 29 % (A) to 23 % (chosen).
- The land-cover features have learned "built = not flooded", which is the radar label bias described in 8b. The map is better at predicting the radar-visible flood, and it under-rates city waterlogging. This must be stated with any map shown to planners.
