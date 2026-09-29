# Phase 2 Implementation Plan — Development-Aware Flood Susceptibility
**Target: Phase 2 complete by November 30, 2026 (Phase 3 follows; this is not the project end)**
**Start: October 22, 2026 → ~5.5 weeks available (40 days)**

## How to use this document
- Work through the numbered steps **in order**; each step depends on the previous one's output, exactly like Phase 1.
- After each step, message me with what you did, your code/output and the numbers/plots you got. I will review it honestly: what is correct, what is wrong, what is weak, and what to fix before moving on.
- Steps marked **[CHECKPOINT]** are hard stops. Do not move past them until the output looks right; every later step builds on them.
- Scope for this phase: **turn the verified Phase 1 baseline into the development-aware model promised in Review 1.** That means UICA / ΔUICA features, PU-learning, leak-free (spatial + temporal) evaluation, and the dual current vs post-development prediction, with explanations. Everything is measured against the Phase 1 baseline, so the gain from each novelty is a number, not a claim.
- Phase 2 is the middle of the project, not the end. Anything that does not fit in 40 days goes on a **Phase 3 candidate list** (Step 13), not into a rushed half-version.
- Keep the Phase 1 conventions: one script per step in `Phase 2/scripts/`, settings as constants at the top, a JSON of numbers per step in `outputs/metrics/`, one write-up per step in `docs/`, seed 42, and commits only after review.

---

## Step 0 — Starting point and scope (read this before starting)

**What Phase 1 already gives you** (all in `Phase 1/`, verified and pushed):
- 30 m analysis grid (FABDEM 1″), land/open-water domain, 11-feature stack, FD8 flow-routing code (`11_build_terrain_hydro_features.py`).
- 2018 / 2019 / 2021 Otsu flood inventories, NDEM cross-check (GTI 58.4 %).
- A balanced training set with a 2 km `block_id`, a LightGBM baseline, spatial-block diagnostics (88.5 % accuracy on unseen 10 km regions), a 5-class map validated on 2019 (AUC 0.83).
- `docs/limitations.md`: the list of what Phase 2 must fix.

**What Phase 2 must add** (the Review 1 novelty slide):

| Novelty (Review 1) | Phase 1 limitation it fixes |
|---|---|
| Development-aware prediction (current vs post-development) | Model sees current conditions only |
| UICA / ΔUICA (upstream impervious area and its change) | No land use / imperviousness; model is development-blind |
| Positive-Unlabelled learning | "Non-flood" really means "not seen flooded"; urban floods counted as dry |
| Leak-free evaluation (spatial CV + temporal hold-out) | Random split: 99.1 % vs 88.5 % on unseen regions |
| Natural-experiment validation (observed urban growth) | Only one training event, no link between growth and flooding |
| Explanations and counterfactual recommendations | Only global feature importance |

**Scope decision: U-Net stays out of Phase 2.** Even with more flood events, the inventory will be a handful of scenes, far from the 226 images a U-Net needs. Spending the 40 days on a CNN would leave the actual novelty (UICA, PU-learning, dual prediction) unfinished. Keep LightGBM (plus one comparison model) and state this decision explicitly, the same way Step 0 of Phase 1 did. Whether U-Net is worth trying in Phase 3 is decided at the end of this phase (Step 13), once the size of the expanded inventory is known.

**Must-have vs later.** Must-have in Phase 2: Steps 1–13. Stretch, only if on schedule: counterfactual recommendations (Step 11.3). Planner dashboards are a Phase 3 candidate.

---

## WEEK 0 (Oct 22 – Oct 25, Thu–Sun): Kickoff

### 1. Freeze the baseline and set up Phase 2
1.1. Write down every piece of Review 2 panel feedback in `Phase 2/docs/review2_feedback.md`, with one line on how Phase 2 will address it. Feedback that changes scope is decided **now**, not in Week 4.
1.2. Tag the Phase 1 state in git (`git tag phase1-final`) so the baseline can always be reproduced exactly.
1.3. Create the Phase 2 folder structure, mirroring Phase 1:
```
Phase 2/
  scripts/            one script per step
  docs/               one write-up per step
  data/raw/           (gitignored)
  data/processed/     (gitignored)
  outputs/maps/  outputs/metrics/  outputs/models/  outputs/presentation/
```
1.4. Move reusable Phase 1 code into one small shared module (`Phase 2/scripts/common.py`): the per-row cell sizes, Priority-Flood + FD8 routing, the exact 20 m → 30 m label aggregation, the metrics function, and the figure helpers. Import it; do not copy-paste it into every script.

**[CHECKPOINT]** Rerun Phase 1 Step 9 using the shared module and confirm it still gives test AUC 0.9993. If refactoring changed a number, something broke.

---

## WEEK 1 (Oct 26 – Nov 1): Development data and a bigger flood inventory

### 2. Build a multi-year land-cover and imperviousness record
2.1. Choose the source. Compare these on the AOI and pick one (document why):
- **Dynamic World V1** (GEE `GOOGLE/DYNAMICWORLD/V1`, 10 m, 2015 onwards, per-pixel "built" probability): best for yearly change.
- **ESRI 10 m annual land cover** (GEE community catalogue, 2017 onwards): cleaner classes, yearly.
- **ESA WorldCover** 2020 / 2021 (GEE `ESA/WorldCover/v100`, `v200`): best accuracy, but only two years. Fetch it through GEE for the **full** AOI; the Phase 1 tile only covered north of 10.008° N.
- **GHSL built-up surface** (GEE `JRC/GHSL/P2023A/GHS_BUILT_S`, 100 m, epochs to 2020 plus projections for 2025 and 2030): the candidate source for the post-development scenario in Step 9.
2.2. Build a yearly **built-up / impervious fraction** layer (0–1) for at least 2018, 2021 and the latest full year, aggregated onto the Phase 1 30 m grid (area-weighted, like the Step 8 label aggregation).
2.3. Clean the time series: built-up should almost never disappear between years. Flag and smooth single-year flickers (classifier noise, clouds) instead of treating them as real change.
2.4. Validate: overlay on Esri imagery at 3–4 known growth sites (Kakkanad / Infopark, the airport surroundings, Aluva–Kalamassery corridor). Report the built-up area per year and the growth rate.

**[CHECKPOINT]** Share the built-up maps for the first and last year, the growth table, and the validation panels. A noisy land-cover series makes ΔUICA meaningless; fix it here.

### 3. Expand the flood inventory (more events, both orbits)
3.1. Find candidate events 2017–2025 from the CHIRPS daily series (3-day and monthly extremes, as in Phase 1 Step 4.1) **and** check which have NDEM maps for this AOI.
3.2. For each candidate, search Sentinel-1 VV scenes within ±3 days in **both** orbit directions (ascending and descending), each with a same-relative-orbit dry reference. Sentinel-1B stopped in December 2021, so post-2021 events have fewer passes; record what exists.
3.3. Run the Phase 1 Step 5 inventory (Otsu + change detection) and the Step 6 NDEM cross-check on every candidate. **Keep an event only if it passes**: clearly bimodal histogram, plausible threshold, and GTI well above its random baseline. Record rejected events and why, as Phase 1 did for 2021.
3.4. Optional, flagged separately: urban flood positives from the Task 1 double-bounce detector (`Phase 1/scripts/test_urban_double_bounce.py`), only where confirmed by NDEM. These are the positives radar normally misses.

**[CHECKPOINT]** Share the event table: date, orbit, flood area, GTI, kept or rejected. Aim for at least 3 usable events so a temporal hold-out is possible (one event reserved purely for testing).

---

## WEEK 2 (Nov 2 – Nov 8): Development-aware features

### 4. UICA and ΔUICA
4.1. **Route the full upstream basin.** Phase 1 TWI saw only 643 of about 4,081 km² draining into the AOI. Get FABDEM (and the Step 2 land cover) for the whole upstream Periyar catchment (HydroBASINS polygons from Phase 1 Step 3), route water there, and pass the accumulated values into the AOI. This fixes the Phase 1 TWI truncation at the same time.
4.2. **UICA (upslope impervious contributing area)** for each pixel = sum over all upslope cells of (impervious fraction × cell area), routed with the same Quinn FD8 weights as Phase 1. It is flow accumulation with impervious area as the "water" instead of cell area. Also compute the **UICA ratio** = UICA / total upslope area (share of the catchment that is paved).
4.3. **ΔUICA** = UICA(later year) − UICA(earlier year), for the pairs your land-cover series supports (at least 2018 → latest).
4.4. Sanity checks (write them as self-tests, like Phase 1 Step 7):
- 0 ≤ UICA ≤ upslope area, everywhere;
- UICA = 0 where there is no built-up upstream;
- a synthetic test: one fully paved cell upstream on an inclined plane gives the analytic UICA downstream;
- UICA never decreases going downstream along a channel.

### 5. Upgrade the rest of the feature stack
5.1. **HAND (height above nearest drainage)**: height of each pixel above the channel it drains to. It is the standard fluvial-flood feature and should beat raw elevation for Periyar flooding. Compute it from the full-basin routing (4.1).
5.2. **Event rainfall** instead of mean annual rainfall: rainfall over the event (e.g. 3-day or 5-day totals) and antecedent rainfall before it, from CHIRPS daily or GPM IMERG (`NASA/GPM_L3/IMERG_V07`, 0.1°). This addresses the Phase 1 finding that mean annual rainfall only acted as a location label.
5.3. Recompute TWI with full-basin accumulation; add the local impervious fraction and land-cover class for each year.
5.4. Optional: OpenLandMap soil texture (already downloaded in Phase 1 Step 3.6) as an infiltration proxy.
5.5. Build **feature stack v2** per event year (development features use that year's land cover) and run the Phase 1 Step 7.5 checks: same grid, no gaps on land, physical ranges, VIF.

**[CHECKPOINT]** Share the UICA / ΔUICA maps, the self-test output, and the v2 correlation/VIF table. UICA must look physically right (high downstream of Kochi's built-up areas, zero in forested catchments) before any model uses it.

---

## WEEK 3 (Nov 9 – Nov 15): Leak-free modelling and PU-learning

### 6. Build the evaluation framework first
6.1. **Spatial block CV as the standard.** Fixed 5-fold StratifiedGroupKFold over 10 km blocks (the strictest Phase 1 setting), with the **same folds for every model** so comparisons are fair.
6.2. **Temporal hold-out.** Train on the earlier events, test on the event reserved in Step 3. This is the direct test of "does the model predict a future flood?".
6.3. Metric set: accuracy, precision, recall, F1, AUC (base-paper set), plus the 2019-style frequency-ratio check on held-out events.
6.4. Re-score the **Phase 1 baseline** under this framework. That number is the bar every Phase 2 model must beat.

### 7. PU-learning
7.1. **Dataset v2**: positives = confirmed flood cells from all kept events; unlabelled = all other land cells (no longer called "non-flood"). Keep the `block_id` and add an `event` column.
7.2. Implement two PU methods and compare them with the standard classifier:
- **Elkan & Noto (2008)**: train labelled-vs-unlabelled, estimate *c* = P(labelled | flood) on held-out positives, rescale the scores;
- **Bagging PU** (Mordelet & Vert, 2014): many classifiers, each on all positives plus a random unlabelled subsample, averaged.
7.3. Evaluate on what PU cannot fake: recall on held-out positive events, NDEM-only flood areas (the peak water we missed), and urban positives if Step 3.4 produced any. Check specifically whether Kochi city susceptibility rises; it was the main Phase 1 blind spot.

### 8. Model v2 and the development ablation
8.1. Tune LightGBM **under spatial CV** (a small grid over `num_leaves`, `min_data_in_leaf`, `learning_rate`, regularisation). Phase 1 was untuned and had a perfect training fit; aim for a smaller train–test gap.
8.2. Add one comparison model (XGBoost or random forest), same folds.
8.3. **The ablation that proves the novelty**, all under the same spatial and temporal tests:

| Model | Features |
|---|---|
| A | Phase 1 baseline (11 features) |
| B | A + improved hydrology (HAND, full-basin TWI, event rainfall) |
| C | B + local land cover / impervious fraction |
| D | C + UICA / ΔUICA (development-aware) |

If D does not beat C on the held-out tests, say so honestly and investigate (resolution, land-cover noise, too few events). Do not hide it.
8.4. **Calibrate** the chosen model (prior correction for the balanced sampling, then Platt or isotonic scaling on held-out folds), so the map shows probabilities rather than the Phase 1 relative index.

**[CHECKPOINT]** Share the A–D table (spatial CV + temporal hold-out), the PU vs standard comparison, and the calibration curve. I will check that the gain from UICA is real, and not a location effect like Phase 1's rainfall. Rule of thumb: if a feature helps on the random split but not on held-out blocks, it is a location proxy.

---

## WEEK 4 (Nov 16 – Nov 22): Dual prediction, validation and explanation

### 9. Current vs post-development susceptibility
9.1. **Current map**: model D with the latest land cover and UICA.
9.2. **Post-development scenario**: define it explicitly and document it. Use at least one of:
- GHSL 2030 built-up projection;
- your own trend extrapolation of the 2018 → latest built-up growth;
- a planner-style scenario (e.g. specific corridors converted to built-up).
Recompute UICA for the scenario, predict again.
9.3. **Δ susceptibility map** = post − current. Report the area moving up one or more classes, the top hotspots (named places), and how much of the change comes from *upstream* development (ΔUICA) versus local building.

### 10. Natural-experiment and temporal validation
10.1. Use the observed urban growth between events: where ΔUICA grew most between an earlier and a later event, did observed flooding increase compared with similar places where it did not? Compare carefully matched areas (similar HAND and TWI).
10.2. Report it as supporting evidence with its limits (few events, rainfall differs between years). It is not proof of cause.
10.3. Report the temporal hold-out results from Step 6.2 for the Phase 2 model next to Phase 1's 2019 check (AUC 0.83).

### 11. Explanations
11.1. **SHAP** (TreeExplainer) for the Phase 2 model: a global summary plot, and dependence plots for UICA, ΔUICA and HAND. Check that the directions make physical sense.
11.2. Local explanations for 3–4 named places (e.g. Aluva, Paravur, a Kochi ward, a growth site): why is each one high or low?
11.3. *Stretch (otherwise Phase 3):* counterfactual what-ifs, e.g. how much upstream impervious area would have to be avoided to drop a site by one class.

**[CHECKPOINT]** Share the current, post-development and Δ maps, the natural-experiment table, and the SHAP plots.

---

## WEEK 5 (Nov 23 – Nov 29): Consolidation and handover to Phase 3

### 12. Phase 2 maps and review materials
12.1. Maps with legend, scale bar and north arrow (Phase 1 style): current, post-development, Δ susceptibility, and a Phase 1 vs Phase 2 comparison.
12.2. One headline table: Phase 1 baseline vs the Phase 2 model, under the same spatial and temporal tests.
12.3. Update the architecture diagram: mark each block done in Phase 1, done in Phase 2, or planned for Phase 3.
12.4. A Phase 2 deck (reuse the Phase 1 deck builder, `15_build_review2_deck.py`) and an updated study notebook covering Phase 2.

### 13. Phase 2 write-up, limitations and the Phase 3 list
13.1. One doc per step (as in Phase 1), plus a Phase 2 summary: what was built, the headline numbers, and what they mean.
13.2. A **limitations note** in the same style as Phase 1 Step 11: each limitation with its measured effect.
13.3. A **Phase 3 candidate list**: each item with the Phase 2 evidence that motivates it and a rough effort estimate. Likely candidates: counterfactual recommendations (if not done in 11.3), a planner / citizen dashboard, U-Net or another spatial model if the inventory grew enough, more scenarios, finer rainfall, and the full project report.
13.4. Keep the report material ready to reuse: step docs, figures, and one merged numbered IEEE reference list (Review 1 [1]–[13], References 14–15, and the data/method references from the Review 2 deck).
13.5. Prepare answers for the likely panel questions:
- "What exactly does UICA add over imperviousness?" → the ablation C vs D.
- "Is the development effect real or a location effect?" → spatial and temporal hold-out gains, SHAP directions.
- "How do you know the post-development map is realistic?" → scenario definition and its assumptions.
- "Why not U-Net?" → Step 0 of this plan, and the Phase 3 decision in 13.3.
- "What is left for Phase 3?" → the candidate list in 13.3.

**[CHECKPOINT — END OF PHASE 2]** Send me the complete set: event table, UICA maps, A–D ablation, PU comparison, dual maps, SHAP, Phase 2 deck, limitations note and Phase 3 list, by **Nov 26–27** at the latest, so there is buffer for fixes before November 30.

---

## Quick reference: what "done" looks like by November 30

| # | Deliverable | Must-have in Phase 2 |
|---|---|---|
| 1 | Multi-year built-up / impervious record for the full AOI | Yes |
| 2 | Expanded, NDEM-checked flood inventory (≥ 3 events) | Yes |
| 3 | Full-basin routing, UICA and ΔUICA with self-tests | Yes |
| 4 | Feature stack v2 (HAND, event rainfall, land cover) | Yes |
| 5 | Spatial CV + temporal hold-out framework, Phase 1 baseline re-scored | Yes |
| 6 | PU-learning vs standard comparison | Yes |
| 7 | A–D ablation proving (or honestly testing) development awareness | Yes |
| 8 | Calibrated Phase 2 model | Yes |
| 9 | Current, post-development and Δ susceptibility maps | Yes |
| 10 | Natural-experiment / temporal validation | Yes |
| 11 | SHAP explanations | Yes |
| 12 | Phase 2 deck, study notebook update, step docs | Yes |
| 13 | Limitations note + Phase 3 candidate list | Yes |
| 14 | Counterfactual recommendations | Stretch (else Phase 3) |
| 15 | Planner / citizen dashboard | No: Phase 3 candidate |
| 16 | U-Net / spatial deep model | No: decided for Phase 3 in Step 13 |
| 17 | Full project report | No: Phase 3 (material collected in 13.4) |

---

**Reminder:** send updates after each checkpoint, not just at the end. A noisy land-cover series caught in Week 1 costs an afternoon; caught in Week 4, it invalidates UICA, the ablation and the dual maps.
