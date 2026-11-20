# Step 7: PU-learning

Script: `scripts/07_pu_learning.py` · numbers: `outputs/metrics/step7_pu_learning.json` · figures: `outputs/maps/step7_pu_comparison.png`, `step7_pu_kochi_maps.png` · model code: `common.make_fit_predict`

## Models

All four use the same features (set D: 20 features incl. UICA) and the Step 6 folds. The settings are LightGBM with Phase 1 parameters and 400 rounds.

| Model | Training rows | Idea |
|---|---|---|
| Standard | flood vs confident non-flood (balanced) | Phase 1 design |
| PU naive | flood vs random land ("unlabelled"), class-weighted | no correction |
| PU Elkan–Noto (2008) | flood vs unlabelled, natural proportions; *c* from 20 % held-out positive blocks | f = g / c |
| PU bagging (Mordelet & Vert 2014) | 25 bags: all flood + equal random unlabelled | average of bags |

## Results (strict test: new year + new place)

| Model | AUC flood vs all land (2020 / 2024) | Held-out flood in VH | NDEM-only in H + VH (2020 / 2024) | Kochi median percentile | Built land in H + VH (2024 map) |
|---|---|---|---|---|---|
| Standard | 0.975 / 0.975 | 99.1 % / 99.8 % | 62 % / 97 % | 60 / 61 | 24.1 % |
| PU naive | 0.981 / 0.982 | 99.3 % / 99.8 % | 67 % / 97 % | 46 / 45 | 25.6 % |
| PU Elkan–Noto | 0.963 / 0.967 | 97.7 % / 97.7 % | 57 % / 96 % | 47 / 46 | 25.1 % |
| **PU bagging** | **0.984 / 0.983** | 99.8 % / 99.9 % | 63 % / 97 % | 60 / 60 | 26.6 % |

- Spatial 10 km CV AUC is about 0.99 for all four.
- Feature set D already beats the Phase 1 baseline on the strict test: 0.975 vs 0.949 and 0.954.

## Reading

1. **PU bagging is the best ranking model.**
   - Flood vs all land: 0.984 / 0.983, against 0.975 for the standard model.
   - It keeps the standard model's view of Kochi.
   - It is used as the PU design in Step 8.
2. **Elkan–Noto is the weakest here.**
   - Its classifier trains on only 80 % of the positive blocks, with no class weighting.
   - *c* = 0.17, which implies that about **7 % of land** is flood-prone. Only 0.3–0.9 % is confirmed in any one event.
   - That implied share sits between a single radar snapshot and the much larger NDEM peak extents (51.6 km² of NDEM-only flood in 2018), so it is plausible.
3. **PU does not fix the Kochi blind spot, and it cannot.**
   - PU-learning stops treating unseen flooding as "dry". But it can only raise places that **look like** the positives.
   - Our positives are 0.1 % built-up, because the radar cannot see urban flooding.
   - So Kochi stays near the 60th percentile, and the naive/Elkan variants even lower it.
   - Built land in High + Very High rises only from 24 % to 27 %.
   - Fixing this needs **urban positives** (SAR double-bounce, NDEM urban polygons, or reported waterlogging points). This goes on the Phase 3 list.
4. **NDEM-only flood** (peak water the satellite missed) is found by all models at a similar rate: about 60–67 % in High + Very High for 2020 and 97 % for 2024.
