# Phase 2 limitations (Step 13.2)

Same style as the Phase 1 limitations note: each limitation, how big its effect is (measured where possible), and what follows from it.

| # | Limitation | Measured effect | Consequence / mitigation |
|---|---|---|---|
| L1 | **Radar labels miss urban flooding** (no water visible between buildings) | 0.1 % of flood labels built-up vs 47 % of land; UICA 0.8 % of model gain; unconstrained models learn "paving lowers risk" (ρ = −0.91) | Built-up susceptibility is a lower bound; monotone scenario model (9.0b); NDEM labels (9.0a); Phase 3 WP1 |
| L2 | **Few flood events** | 4 validated events, only two before / two after; natural experiment MDE about 0.5 pp (radar), 1.1 pp (NDEM) | Big-flood development effect below detection; more events in Phase 3 |
| L3 | **Detailed NDEM only for 2018 and 2020** | Urban UICA gain rests on one test event (2020, 458 flooded paved cells) | Report as small, consistent, not proven |
| L4 | **Noisy paving record** (GHSL + 0.214 × Dynamic World change) | r = 0.74 vs WorldCover; some cells show paving falling; 2030 range 299–434 km² (GHSL vs trend) | Two scenarios bracket the result |
| L5 | **One soil group for the basin** (no downloadable soil map) | Groups B / D change runoff numbers about 2× | Pattern unchanged; soil map in Phase 3 if available |
| L6 | **No drains, detention or paddy / wetland storage in SCS-CN** | Not measured; storage loss would make the true lowland effect larger | Extra-runoff numbers are an upper bound where drainage is good and a lower bound where paddy fields were filled |
| L7 | **Runoff volume only, not timing** | Not measured | Paving also speeds water up; a drainage / hydraulic model in Phase 3 (optional) |
| L8 | **Scenario model accuracy cost** | Strict AUC 0.864 → 0.854 (0.011 vs a pre-set limit of 0.01) | Accepted because the alternative gives physically wrong scenarios; guide to confirm |
| L9 | **Probabilities depend on event size** | Platt / isotonic over-predict 2× on smaller 2020 / 2024 events; prior correction matched by partly coincidence | Condition on event rainfall in Phase 3 |
| L10 | **Location proxies** | Removing mean annual rainfall costs 0.007 strict AUC; SHAP 5–8 %; aspect 7.6 % in one model | Flagged; kept for skill |
| L11 | **Waterlogging data small and news-based** | 15 spots, 2 main events, mostly the old city; location ±300 m | First hint only (p = 0.08–0.19); systematic records in Phase 3 |
| L12 | **Choices made after seeing data** | NDEM detail threshold, frequent-storm case, 9.0b acceptance | Stated openly in the Step 9 doc; results reported for all cases |
| L13 | **FABDEM licence** | Non-commercial use only | Copernicus DEM for any deployment |
| L14 | **Tuning did not close the train–test gap** | Train AUC about 1.0; gap 0.0071 → 0.0065 | Confident negatives are easy; the strict test is the honest measure |
