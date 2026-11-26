# Step 10: Natural experiment: did places where development grew actually flood more?

Script: `scripts/10_natural_experiment.py` · numbers: `outputs/metrics/step10_natural_experiment.json` · figure: `outputs/maps/step10_natural_experiment.png` · log: `outputs/step10_run.log`

## Question

Step 9 showed, by physics, that development sends more water to flood-prone places. Step 10 asks the **observed flood record**: between the earlier floods (2018, 2019) and the later ones (2020, 2024), did flooding change more where development grew than in similar places where it did not?

## Design (fixed in the script header before running)

| Part | Choice |
|---|---|
| Cells | Floodplain land (HAND ≤ 5 m; 10 m as sensitivity), seen in every radar scene, away from reference water |
| Outcomes | Radar flood (≥ 50 % of the cell) for each event; radar + detailed NDEM for 2018 and 2020 |
| **D1** | Radar, 4 events: before = 2018/2019, after = 2020/2024 |
| **D2** | Radar + NDEM, 2018 vs 2020 (urban flooding visible) |
| **Placebo P** | 2018 vs 2019, before most of the growth. The "effect" of **future** growth must be about 0 |
| Treatments (top 10 % growth vs bottom 25 %) | **T1** upstream paving (UICA ratio change, the plan's main test); **T2** local paving; **T3** Step 9 modelled extra water |
| Matching | Coarsened exact matching on HAND × TWI × distance to river × event rainfall |
| Effect | Difference-in-differences (percentage points of flooded cells) and the ratio of ratios |
| Uncertainty | 2 km block bootstrap (500 reps), 95 % CI, minimum detectable effect (MDE = 2.8 SE) |
| Validity checks | **Floor effect:** treated cells barely ever flood beforehand (< 0.1 %). **Ratio disagrees:** absolute and relative changes point opposite ways. Either one makes an estimate invalid. **Placebo correction:** D1 minus placebo, paired bootstrap |

## Results (floodplain, HAND ≤ 5 m)

| Design | T1 upstream paving | T2 local paving | T3 modelled extra water |
|---|---|---|---|
| D1 radar, 4 events | +0.42 pp [+0.07, +0.80] | +0.84 pp, *invalid (floor effect)* | +0.33 pp [+0.06, +0.56] |
| D2 radar + NDEM | +0.66 pp [−0.14, +1.49] | +1.05 pp, *invalid (ratio 0.55 disagrees)* | +0.54 pp [−0.57, +1.51] |
| Placebo (future growth, 2018 vs 2019) | **+0.47 pp [+0.03, +0.99]** | invalid | +0.14 pp [−0.12, +0.40] |
| **D1 minus placebo** | **−0.05 pp [−0.71, +0.58]** | invalid | **+0.19 pp [−0.17, +0.56]** |
| Minimum detectable effect (D1 / D2) | 0.50 / 1.12 pp | | 0.35 / 1.51 pp |

**HAND ≤ 10 m (sensitivity):** placebo-corrected T1 +0.07 pp [−0.22, +0.41]; T3 +0.07 pp [−0.11, +0.24]. Same conclusion.

For scale, the flooded share of floodplain cells is 0.4–0.9 % per event with radar labels, and 1–8 % with radar + NDEM.

## Reading

1. **The simple estimates look positive, but they do not survive the checks.**
   - Upstream paving (T1) shows +0.42 pp. But the **placebo shows the same +0.47 pp**: the places that *would later* be developed already behaved differently between 2018 and 2019. So the difference comes from *which places* grew (they respond differently to flood size), not from the growth itself.
   - Placebo-corrected: **−0.05 pp, with an interval of about ±0.6 pp**.
2. **Modelled extra water (T3)** passes the placebo check (+0.14, not significant). Its corrected effect is +0.19 pp [−0.17, +0.56]: **positive, but not distinguishable from zero.**
3. **Local paving (T2) cannot be tested with this data.**
   - With radar labels, newly paved cells are "never flooded" both before and after, because radar cannot see water among buildings (floor effect).
   - With NDEM labels, the absolute and relative changes point in opposite directions.
   - Both checks flag it automatically; it is not evidence either way.
4. **Overall: no detectable effect of development on the observed large floods.**
   - The design could detect about 0.5 pp (radar) or about 1.1 pp (NDEM) changes in flood rate.
   - Step 9.0c predicts that development adds only **+0.2 % runoff** in monsoon flood storms, an effect far below what four big floods can reveal.
   - So the null result is **consistent with the physics**, not a contradiction of it. The development effect lives in frequent storms (urban waterlogging), and the satellite / NDEM record does not observe those at all.

## What this means for the project

| Evidence | Development effect on big riverine floods | Development effect on frequent storms / waterlogging |
|---|---|---|
| ML (Steps 8, 9.0a) | Not visible in radar labels; small gain on urban land with NDEM labels (one test event) | Not observable (no labels) |
| Physics (9.0c, 9) | Very small (+0.2 % runoff) | **Large and growing** (280 → 434 km² with ≥ 5 % more water; 113 → 190 km² hotspots) |
| Natural experiment (10) | **None detected** (CI includes 0; could detect about 0.5 pp) | Cannot be tested with satellite / NDEM data |

**The defensible conclusion:** development does not measurably change *where the big monsoon floods go*. It does add substantial water in *frequent storms*, especially downstream of growing suburbs. That second effect is modelled, not yet observed. **Observing it needs urban waterlogging records**, which is the top Phase 3 item (SAR coherence / double-bounce detection, Kochi waterlogging reports).

## 10.3 Temporal hold-out: Phase 1 vs Phase 2 (AUC, flood vs all other land)

| Model | 2019 (Phase 1 check) | 2020 | 2024 | Test |
|---|---|---|---|---|
| Phase 1 map (Phase 1 Step 10) | 0.83 | – | – | Otsu flood vs all land |
| Phase 1 model, re-scored | – | 0.988 | 0.989 | new year |
| Phase 1 model, re-scored | – | 0.949 | 0.954 | **new year + new place** |
| Phase 2 chosen (PU bagging, set C) | – | 0.992 | 0.993 | new year |
| **Phase 2 chosen (PU bagging, set C)** | – | **0.985** | **0.985** | **new year + new place** |
| Phase 2 scenario model, radar + NDEM truth | – | 0.864 | – | new year + new place, includes NDEM-only flood |

The Phase 1 0.83 used a different flood definition (all Otsu flood, including unconfirmed patches), so it is shown for reference, not as a like-for-like comparison. Like-for-like, under the strict test, Phase 2 improves from 0.949 / 0.954 to 0.985 / 0.985.

## Limits

- Only 4 events, of very different sizes (2018 is about 5× 2020), and only two before and two after.
- Development 2018 → 2024 is gradual (about 1.8 %/yr), so the treatment contrast is weak.
- The radar labels miss urban flooding (floor effect).
- The rainfall-tercile matching uses the 2018 spatial pattern.
- With only one placebo pair, the placebo correction is itself noisy (wide CIs).
