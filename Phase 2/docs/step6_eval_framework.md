# Step 6: Evaluation framework

Script: `scripts/06_eval_framework.py` · numbers: `outputs/metrics/step6_eval_framework.json`, `step6_fold_assignment.json` · figures: `outputs/maps/step6_folds_and_labels.png`, `step6_baseline_rescored.png` · data (gitignored, regenerable): `data/processed/training_v2/samples_v2.parquet`, `label_pools_<event>.tif` · shared code: `common.py` (Week 3 section: `label_pools`, `run_protocols`, `event_eval`)

## Dataset v2

Each of the 4 kept events is labelled with the **Phase 1 Step 8 rules**, applied to that event's Sentinel-1 class raster and NDEM envelope:
- **Flood:** ≥ 50 % confirmed flood, in patches ≥ 10 px, within 40 m of NDEM.
- **Confident non-flood:** no radar flood within 60 m, outside NDEM + 40 m, and > 60 m from reference water.

| Event | Confirmed flood (positives) | NDEM-only flood\* | Positives on built land | Flood share of random land |
|---|---|---|---|---|
| 2018-08-21 (train) | 15,370 cells, 13.53 km² | 51.6 km² | 0.1 % | 0.90 % |
| 2019-08-10 (train) | 11,968 cells, 10.53 km² | 15.7 km² | 0.1 % | 0.73 % |
| 2020-08-10 (test) | 6,731 cells, 5.92 km² | 5.4 km² | 0.1 % | 0.40 % |
| 2024-08-01 (test) | 5,485 cells, 4.83 km² | 0.3 km² | 0.0 % | 0.34 % |

\* Inside the NDEM map, but with no radar flood within 60 m: the peak water our single satellite pass missed. It is used **only for evaluation**.

**Rows per event:**
- `pos`: all positives.
- `neg`: an equal number of confident non-flood cells (the standard balanced design).
- `rnd`: 60,000 uniform random land cells. These are the PU "unlabelled" set, and they also give land percentiles.
- `ndem`: up to 20,000 NDEM-only cells.

The table has 363,434 rows. Its features are the 18 stack-v2 bands plus Phase 1 `twi` and `rainfall_mean_annual_mm`, so models A–D read the same table.

**Key fact for Step 7.** Only **0.1 % of positives are built-up**, against 47 % of land. Sentinel-1 cannot see water between buildings, so the labels contain almost no urban flooding.

## 6.1 Fixed spatial folds

- 5-fold StratifiedGroupKFold over **10 km UTM blocks** (23 blocks), seed 42.
- The folds are stratified on the pooled standard rows of all 4 events.
- The block → fold map is saved to `step6_fold_assignment.json`, and **every later model uses it**.
- The same place is always in the same fold, in every event. A test block was therefore never seen in training, in any year.
- Fold 0 holds the dense Aluva–Eloor floodplain, so its fold has the most positives. This cannot be avoided with 10 km blocks.

## 6.2 and 6.3 Protocols and metrics

| Protocol | Train | Test |
|---|---|---|
| **Spatial** | 4 events, folds ≠ k | 4 events, fold k (standard rows) |
| **Temporal** | 2018 + 2019, all blocks | 2020, 2024 |
| **Strict** (new year + new place) | 2018 + 2019, folds ≠ k | 2020, 2024 in fold k (out-of-fold) |

**Metrics:**
- the base-paper set: accuracy, precision, recall, F1, AUC on flood vs confident non-flood;
- **map-level AUC: flood vs all other land.** This is like the Phase 1 2019 check, and harder, because uncertain land is included;
- share of held-out flood in **High + Very High** (top 40 % of land, the Phase 1 quantile classes);
- frequency ratio per class;
- share of NDEM-only flood in High + Very High;
- Kochi core median land percentile.

## 6.4 Phase 1 baseline re-scored (11 features, Phase 1 settings, 648 rounds)

| Test | AUC | Other |
|---|---|---|
| Random 70/30 split, 2018 (Phase 1 protocol) | **0.999** | – |
| 10 km CV, 2018 only | 0.965 ± 0.041 | – |
| **10 km CV, 4 events** | **0.962 ± 0.042** | accuracy 83.0 %, F1 0.706 |
| Temporal 2020 / 2024 (flood vs all land) | 0.988 / 0.989 | H + VH 100 % / 100 % |
| **Strict 2020 / 2024 (flood vs all land)** | **0.949 / 0.954** | H + VH 98.7 % / 99.2 %; NDEM-only H + VH 66 % / 95 % |

**Reading.**
- The temporal test alone is **too easy**. The same floodplain floods every year, so a model that memorises *where* past floods were scores about 0.99 on a new year.
- Only the **strict** test (a new year *and* a new place) measures generalisation. It is the bar for Phase 2: **0.949 (2020) and 0.954 (2024)**.
- The 10 km CV of 0.965 on 2018 is lower than the Phase 1 value (0.979). The folds are now fixed on the pooled 4-event data, and the 2018 inventory uses the Phase 2 dry reference.
