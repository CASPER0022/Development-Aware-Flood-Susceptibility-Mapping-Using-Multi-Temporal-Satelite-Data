# Step 9.0a: Radar + NDEM labels: a fair test of UICA on urban land

Script: `scripts/09a_ndem_labels.py` · numbers: `outputs/metrics/step9a_ndem_labels.json` · figures: `outputs/maps/step9a_label_quality.png`, `step9a_urban_tests.png` · log: `outputs/step9a_run.log`

## Why

Week 3 found that radar flood labels are 0.1 % built-up. UICA therefore could not earn skill (D = C), and the model learned "built = dry". NDEM (NRSC) maps also contain flooding our single radar pass missed, which is far more urban.

## 1. Label quality: only detailed NDEM polygons are used

**Rule, fixed before modelling:** a polygon is *detailed* if it has ≥ 50 vertices per km of outline. Coarse, generalised polygons look like random land and are dropped.

| Event | Polygons | Detailed | Median vertices / km | Radar flood | NDEM flood added |
|---|---|---|---|---|---|
| 2018 (train) | 15,000 | 98 % | 177 | 13.5 km² | **62.2 km²** |
| 2019 (train) | 840 | 0 % | 23 | 10.5 km² | 0 (coarse, dropped) |
| 2020 (test) | 11,038 | 100 % | 329 | 5.9 km² | **8.9 km²** |
| 2024 (test) | 431 | 7 % | 27 | 4.8 km² | 0 (coarse, dropped) |

**Profile of the labels:**

| | Radar flood | NDEM-added flood (2018 / 2020) | Random land |
|---|---|---|---|
| ≥ 20 % paved | 0 % | **9.8 % / 4.5 %** | 19 % |
| UICA ratio ≥ 0.10 | 5–13 % | **30 % / 19 %** | 39–46 % |
| DW built (settlement) | 0.1 % | 30 % / 27 % | 43–50 % |
| Median HAND | 0.0 m | 1.5 m / 1.0 m | 3.8 m |

NDEM-added flood reaches paved, high-UICA land, which radar flood never does. It is still low-lying (HAND 1–1.5 m vs 3.8 m for land), so it looks like real flooding, not polygon noise.

**Limits:**
- Only 2018 (training) and 2020 (test) have usable NDEM.
- **The strict urban test therefore rests on one event (2020).**

## 2. Labels and models

- **Positives:** confirmed radar flood ∪ cells ≥ 50 % inside a detailed NDEM polygon (land, fully observed, away from reference water).
- **Negatives:** the unchanged confident non-flood pool.
- **Random land:** the Step 6 cells, re-flagged.
- **Models:** C and D × {standard, PU bagging (15 bags)} × {radar labels, radar + NDEM labels}. Tuned Step 8 settings, fixed 10 km folds.
- **Scoring:** everything is scored against radar + NDEM truth (flood vs all other land).

## 3. Results

**AUC, PU bagging** (standard in brackets):

| Test | Subset | Radar labels C / D | Radar + NDEM labels C / D |
|---|---|---|---|
| Spatial (4 events) | all land | 0.805 / 0.801 | 0.940 / 0.939 |
| | paved ≥ 20 % | 0.666 / 0.646 | 0.942 / 0.943 |
| **Strict 2020** | all land | 0.853 / 0.853 | 0.848 / 0.852 (std 0.861 / **0.864**) |
| | DW built | 0.694 / 0.698 | 0.691 / 0.702 (std 0.742 / **0.757**) |
| | paved ≥ 20 % (458 floods) | 0.662 / 0.659 | 0.697 / 0.707 (std 0.695 / **0.732**) |
| | UICA ratio ≥ 0.10 | 0.710 / 0.709 | 0.712 / 0.720 (std 0.739 / **0.753**) |

**D − C on strict 2020**, with a paired bootstrap over 2 km blocks (1,000 reps, 95 % CI):

| Labels | Design | All land | DW built | Paved ≥ 20 % | UICA ratio ≥ 0.10 |
|---|---|---|---|---|---|
| Radar | standard | −0.003 [−0.008, −0.000] | −0.002 [−0.018, +0.013] | +0.010 [−0.016, +0.032] | −0.005 [−0.017, +0.007] |
| Radar | PU bagging | −0.000 [−0.002, +0.001] | +0.004 [−0.002, +0.010] | −0.003 [−0.013, +0.007] | −0.001 [−0.006, +0.004] |
| **Radar + NDEM** | **standard** | +0.003 [−0.001, +0.008] | **+0.015 [+0.003, +0.027]** | **+0.037 [+0.021, +0.058]** | +0.013 [−0.004, +0.029] |
| **Radar + NDEM** | **PU bagging** | +0.004 [−0.001, +0.007] | **+0.012 [+0.002, +0.021]** | **+0.010 [+0.000, +0.021]** | **+0.008 [+0.002, +0.014]** |

**2024 map** (trained on 2018 + 2019, PU bagging):

| | Built land in H + VH | Kochi core median percentile |
|---|---|---|
| Radar labels, C (Week 3 choice) | 23.4 % | 55 |
| Radar + NDEM, C | 29.7 % | 46 |
| Radar + NDEM, D | 29.8 % | 44 |

## Reading

1. **Radar-only models miss urban flooding.** Scored against radar + NDEM truth, they reach only 0.65–0.74 on paved / built land. This is the Week 3 blind spot, now measured.
2. **With NDEM labels, UICA starts to matter where it should.**
   - On urban land in a new year and new place, D beats C: +0.015 (built) and +0.037 (≥ 20 % paved), standard design. The confidence intervals exclude zero.
   - On all land the gain stays small (+0.003, CI includes zero), because most flooding is still rural.
   - With radar labels there is no gain anywhere.
   - **This is the first evidence that the development feature adds skill, and it appears only once the labels include urban flooding.**
3. **The gains are modest, and they rest on one test event (2020)** with 458 flooded paved cells. Report them as "consistent, small, positive", not as proof.
4. **The spatial-CV numbers (0.94) are optimistic.** The positives are dominated by 2018's 62 km² of NDEM, so the test is new places in the same flood. The strict number (0.85–0.86) is the honest one.
5. **Built land rises** in High + Very High (23 % → 30 %), so the "built = dry" bias is reduced.
   - **The Kochi core does not rise** (it drops to the 44–46th percentile). The NDEM maps also show little flooding in the dense core.
   - Core waterlogging is short-lived (drains in hours), and neither radar nor NDEM sees it. Phase 3's waterlogging-point list is still needed for the core.
6. **Design:** with NDEM labels, the standard design beats PU bagging on the strict test (0.864 vs 0.852 all land, 0.732 vs 0.707 paved). NDEM fills in many of the missing positives that PU was compensating for.

## Decision (plan 9.0a)

D beats C on built-up land, so the **scenario model is D, trained on radar + NDEM labels, standard design**. Step 9.0b adds the non-decreasing constraints on paving / UICA to this model.

The Week 3 model (PU bagging, C, radar labels) is kept as the "radar-visible flood" map.
