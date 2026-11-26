# Phase 3 Implementation Plan: Proving the Development Effect, and Finishing the Project
**Target: Phase 3 complete by March 28, 2027 (final project phase: report, final review, hand-over)**
**Start: December 1, 2026 → about 17 weeks, including a 2-week break (Dec 21 – Jan 3)**
*Dates are tentative: set Week 0 and the final checkpoint to the official review calendar as soon as it is announced, and shift the weeks with them.*

## How to use this document
- Work through the numbered steps **in order**, as in Phases 1 and 2. Each step depends on the previous one's output.
- After each step, message me with what you did, your code and output, and the numbers and plots you got. I will review it honestly: what is correct, what is wrong, what is weak, and what to fix before moving on.
- Steps marked **[CHECKPOINT]** are hard stops. Do not move past them until the output looks right.
- **Scope for this phase:** the project title promises *development-aware* flood mapping. Phase 2 showed the development effect **by physics** (Step 9) but could not show it in **observed** flood data (Step 10), because satellite and NDEM maps only record the big riverine floods. Phase 3 has two jobs:
  1. **Work Package 1 (WP1): prove or disprove the development effect with observed data,** mainly urban waterlogging, where Phase 2 says development matters.
  2. **Finish the project:** the planner tool, the full report, the final review and the patent decision.
- Every claim must be a number with an uncertainty, measured against Phase 2. **"No detectable effect" is a valid result**; an unmeasured claim is not.
- Keep the Phase 1 / 2 conventions:
  - one script per step in `Phase 3/scripts/`;
  - settings as constants at the top;
  - one JSON of numbers per step in `outputs/metrics/`;
  - one write-up per step in `docs/`;
  - seed 42;
  - design choices fixed in the script header **before** running;
  - commits only after review, with no Claude attribution.

---

## Step 0: Starting point and scope (read this before starting)

**What Phase 2 already gives you** (all in `Phase 2/`, verified and pushed):
- **Paving and routing:**
  - yearly paved fraction 2017–2025 on the full upstream grid;
  - full-basin routing (9,130 km²);
  - UICA / ΔUICA for every year;
  - HAND;
  - feature stack v2 for 4 validated floods (2018, 2019, 2020, 2024).
- **Evaluation:**
  - the strict test (new year + new place), with fixed 10 km folds (`step6_fold_assignment.json`);
  - Phase 1 baseline 0.949 / 0.954 → Phase 2 model 0.985 / 0.985.
- **Models and scenarios:**
  - radar + NDEM labels (`common.ndem_label_dataset()`);
  - the monotone scenario model (`model_v2_scenario.txt`);
  - SCS-CN extra runoff;
  - 2017 / 2025 / 2030 scenarios and hotspots (113 → 190 km²).
- **Tests and explanations:**
  - natural-experiment code (DiD, matching, placebo, block bootstrap);
  - the waterlogging check with 15 spots (UICA ratio 57th–61st percentile, p = 0.08–0.19);
  - SHAP.
- **Write-up material:**
  - `docs/phase2_limitations.md` (L1–L14): the list of what Phase 3 must address;
  - `docs/phase3_candidates.md`: the work packages;
  - `docs/report_material.md`: merged references [1]–[54].

**What Phase 3 must add**

| Phase 3 work | Phase 2 limitation it addresses |
|---|---|
| Systematic urban waterlogging records (WP1a) | L1 radar labels miss urban flooding; L11 only 15 news spots |
| Urban flood detection from Sentinel-1 (WP1b) | L1 |
| Paddy-field and wetland loss as lost storage (WP1c) | L6 SCS-CN ignores storage |
| More and smaller events (WP1d) | L2 only 4 floods; natural experiment can detect only about 0.5 pp |
| Waterlogging model and the UICA test on waterlogging labels | L1, L3 (urban result rests on one test year) |
| Event-conditioned probabilities | L9 probabilities depend on event size |
| Counterfactual recommendations | Phase 2 stretch item 14 |
| Planner tool, full report, final review | Deliverables |

**Scope decisions**
- **U-Net stays out**, unless WP1 produces several hundred dated urban flood labels (decided at the Step 6 checkpoint). The model family was not the limit in Phase 2 (XGBoost = LightGBM).
- **The drainage / hydraulic model (WP1e) is a stretch.** Do it only if Steps 2–7 are on schedule.
- **Data access is the main risk.** Corporation and KSDMA records may need a formal request or an RTI application. Start those requests in Week 0, and plan so that public sources alone are enough.

**Must-have vs later.**
- Must-have: Steps 1–7 and 10–12.
- Stretch: Step 8 (drainage model) and Step 9 (counterfactuals).

---

## WEEK 0 (Dec 1 – Dec 6): Kickoff

### 1. Freeze Phase 2 and set up Phase 3
1.1. Write every Phase 2 review / guide comment into `Phase 3/docs/phase2_feedback.md`, with one line on how Phase 3 addresses it. Two open items must be decided **now**:
- accepting the 9.0b accuracy cost (0.011 vs 0.01);
- the WP1 scope.

1.2. Tag the Phase 2 state (`git tag phase2-final`).

1.3. Create the folder structure (it mirrors Phase 2):
```
Phase 3/
  scripts/            one script per step (import Phase 2's common.py; do not copy it)
  docs/               one write-up per step
  data/raw/           (gitignored, except small hand-made tables such as waterlogging points)
  data/processed/     (gitignored)
  outputs/maps/  outputs/metrics/  outputs/models/  outputs/presentation/
```

1.4. **Send the data requests this week:**
- Kochi Corporation (waterlogging complaints, drain maps);
- KSDMA / Ernakulam DDMA (incident logs);
- Kerala State Remote Sensing and Environment Centre (paddy / wetland data bank);
- IMD Kochi (hourly or daily station rainfall).

Keep a log in `docs/data_requests.md` (date sent, contact, status).

**[CHECKPOINT]** Rerun the Phase 2 headline numbers from `Phase 3/scripts/01_freeze_check.py`:
- strict AUC 0.985 / 0.985;
- hotspots 113 / 190 km²;
- Step 10b percentiles.

If any number changes, something broke.

---

## WEEKS 1–2 (Dec 7 – Dec 20): Observed urban flooding (WP1a, WP1b)

### 2. Systematic waterlogging records (WP1a)
2.1. **Collect** dated waterlogging reports for Kochi / Ernakulam, 2015–2026:
- news archives (The Hindu, New Indian Express, Times of India, Mathrubhumi, Malayala Manorama, Onmanorama);
- corporation and DDMA records, if they arrive;
- citizen and social-media reports, only with a date and a place name.

The Phase 2 list (15 spots) is the seed.

2.2. **One table, one row per report:** `date, place, lat, lon, geocode_source, location_accuracy_m, rain_event, source_url, notes`. Geocode with OpenStreetMap Nominatim as in Step 10b. Check every doubtful point by hand and record the OSM feature used.

2.3. **Quality rules, fixed before analysis:**
- a point needs a date and a place that can be located to ±500 m;
- one report per place per day (duplicates merged);
- riverine flooding (e.g. airport, Aluva riverbank) is tagged separately from urban waterlogging.

2.4. **Target:** at least 150 dated points over at least 6 rain events. Also collect **"not waterlogged" information** where it exists (e.g. roads reported open during the same event), because Phase 2 lacked true negatives.

### 3. Urban flood detection from Sentinel-1 (WP1b)
3.1. For the dates in Step 2 with a Sentinel-1 pass within 1 day, compute:
- the **coherence drop** between a pre-event SLC pair and a co-event pair;
- the **double-bounce increase** (VV backscatter rise in built-up cells).

3.2. Validate the detection against the Step 2 points (hit rate, false alarms on dry events).
- If it works (hit rate clearly above random with a CI), it becomes a second urban label source.
- If not, report it and rely on Step 2 alone.

**[CHECKPOINT]** Share:
- the waterlogging table (count, events, map);
- the urban radar detection result.

I will check location accuracy, duplicates, and that riverine and urban points are separated. Without at least about 100 usable points, Steps 5–6 will be weak: decide then whether to extend collection by one week.

---

## WEEKS 3–4 (Jan 4 – Jan 17): Storage loss and more events (WP1c, WP1d)

### 4. Paddy-field and wetland loss (WP1c)
4.1. Map paddy fields and wetlands for 2017 and 2025:
- Dynamic World crops / flooded vegetation / water;
- ESA WorldCover;
- the Kerala paddy / wetland data bank if obtained.

Validate against 100 hand-checked points on high-resolution imagery.

4.2. Quantify the conversion 2017 → 2025 (km², where).

4.3. Add **lost storage** to the runoff model: converted paddy / wetland cells lose a depression-storage depth (state the value and a range from the literature). Rerun the Step 9 frequent-storm layer, and report how much the hotspot area changes.

### 5. More events (WP1d)
5.1. Add the 2025 and 2026 monsoons (Sentinel-1A / 1C) and any new NDEM maps with the Phase 2 Step 3 pipeline; keep only events that pass the same checks.

5.2. Add **frequent-storm events**: rain days at the IMD / CHIRPS 90th percentile that match Step 2 waterlogging reports.

5.3. **Power analysis before rerunning the natural experiment:** with the enlarged event set, what is the minimum detectable effect? Write the number down first.

**[CHECKPOINT]** Share:
- the paddy / wetland loss map;
- the storage-corrected hotspot change;
- the new event table;
- the power analysis.

---

## WEEKS 5–7 (Jan 18 – Feb 7): The development test on observed data (WP1 core)

### 6. Waterlogging susceptibility model: does UICA matter for waterlogging?
6.1. **Dataset:**
- positives = Step 2 urban waterlogging points (and Step 3 detections if validated);
- negatives = "not waterlogged" reports plus PU-unlabelled urban land (Phase 2 Step 7 method);
- the domain restricted to urban land (paved ≥ 0.2).

6.2. **Features:**
- HAND, slope, local paved fraction, UICA, UICA ratio, ΔUICA;
- drainage proxies (distance to canal / drain, road density, culvert density from OSM);
- short-duration rainfall intensity.

6.3. **The key ablation:** model C-urban (everything except UICA / ΔUICA) vs D-urban (with them), under a spatial block test and a held-out-event test, with block-bootstrap CIs. **This is the main development test of the project.** Monotone constraints as in Phase 2 Step 9.0b.

6.4. SHAP on the waterlogging model:
- development share;
- direction;
- local vs upstream paving.

### 7. Validate the Step 9 development layer against observations
7.1. **Rank test:** do places with more modelled extra water (Step 9, frequent storm) have more waterlogging reports per km²? Use a Spearman correlation over 1 km blocks, with a block-bootstrap CI.

7.2. **Natural experiment rerun** (Phase 2 Step 10 code) with:
- the enlarged events (Step 5);
- waterlogging outcomes (Step 2);
- the placebo, floor and ratio checks unchanged.

7.3. Write the **development verdict** in one table: for big riverine floods and for waterlogging, what each line of evidence says (ML ablation, physics, natural experiment, rank test), with CIs.

**[CHECKPOINT — DEVELOPMENT VERDICT]** Share the C-urban vs D-urban table, the rank test and the natural-experiment rerun. I will check:
- that the UICA gain survives held-out events and is not a location proxy (rule of thumb: helps on random split but not on held-out blocks = proxy);
- that the waterlogging labels are not concentrated in a few well-reported places (test: drop the top 3 places and rerun).

The report's framing depends on this checkpoint:
- **If the effect is confirmed:** "development-aware" is supported by observation.
- **If not:** the report says development matters by physics but is not yet observable at this data level, and explains why.

---

## WEEKS 8–9 (Feb 8 – Feb 21): Probabilities and recommendations

### 8. Optional: drainage model for one catchment (WP1e, stretch)
8.1. Only if on schedule. Pick one fast-growing catchment (e.g. Kakkanad / Infopark). Build an EPA-SWMM (or HEC-RAS 2D rain-on-grid) model from the DEM, OSM drains and the Step 9 paving scenarios. Compare 2017 vs 2030 peak flow and ponding.

8.2. Check that the model's ponded cells match the Step 2 reports in that catchment.

### 9. Event-conditioned probabilities and counterfactuals
9.1. Make the calibrated probability depend on the event (rainfall depth or intensity), fitted across all events (Phase 2 limitation L9). Report the reliability curve per event.

9.2. **Counterfactual recommendations** (Phase 2 stretch item): for the top Step 9 hotspots, how much upstream paving would have to be avoided (or offset by retention) to drop the site one class, or to bring its extra water below 5 %? Report it in m² of paving and as a retention volume.

**[CHECKPOINT]** Share the event-conditioned reliability curves and the counterfactual table for the top 10 hotspots.

---

## WEEKS 10–11 (Feb 22 – Mar 7): Planner tool

### 10. Planner / citizen tool
10.1. A web map (static, as in the Phase 2 study notebook), with:
- current vs 2030 maps;
- hotspots;
- the waterlogging layer;
- click-to-inspect (paving, UICA, extra water, risk class);
- a simple what-if: add paving in a drawn area and show the extra water downstream (precomputed per 2 km block if live routing is too slow).

10.2. Every layer labelled with source, date and uncertainty. Built-up risk is marked as a lower bound where it still is.

10.3. Test it with 3–5 users (e.g. a classmate in civil engineering, a planner contact if available); record task time and feedback.

**[CHECKPOINT]** The tool runs on a phone and a laptop, and every number shown matches the JSON files.

---

## WEEKS 12–14 (Mar 8 – Mar 28): Report, final review, hand-over

### 11. Full project report
11.1. One report on the institute template: Phases 1–3, built from the step docs. Keep the < 20 % similarity rule; write in your own words, not copied from the docs.

11.2. **References:** start from `Phase 2/docs/report_material.md` ([1]–[54]) and add Phase 3 sources. Check every DOI.

11.3. A **limitations chapter**, each limitation with its measured effect (Phase 1 + 2 + 3).

11.4. **The development chapter:** the Step 7 verdict table, answered honestly.

### 12. Final review materials and patent decision
12.1. Final deck (reuse `Phase 2/scripts/12b_build_phase2_deck.py`), the speaking split, and an updated study notebook covering Phase 3.

12.2. **Patent decision with the guide:** if filing, file the provisional application **before** any public demo. Until then, the code and the method must stay private; the BTP repository is currently public.

12.3. **Hand-over:**
- README;
- a reproduction checklist (every headline number from one command per phase);
- tagged release `phase3-final`.

**[CHECKPOINT — END OF PROJECT]** Send the complete set by **March 21** at the latest, so there is a week of buffer before the final review:
- waterlogging dataset;
- the development verdict;
- maps;
- the tool;
- the report;
- the deck;
- the notebook.

---

## Quick reference: what "done" looks like by March 28

| # | Deliverable | Must-have in Phase 3 |
|---|---|---|
| 1 | Phase 2 frozen and reproduced; data requests sent | Yes |
| 2 | Waterlogging dataset (≥ 150 dated points, ≥ 6 events, with sources) | Yes |
| 3 | Urban radar detection tested (works or not, measured) | Yes |
| 4 | Paddy / wetland loss map and storage-corrected hotspots | Yes |
| 5 | Enlarged event set + power analysis | Yes |
| 6 | Waterlogging model; C-urban vs D-urban UICA test | Yes (the main test) |
| 7 | Development verdict table (all evidence, with CIs) | Yes |
| 8 | Drainage / hydraulic model for one catchment | Stretch |
| 9 | Event-conditioned probabilities; counterfactual recommendations | 9.1 yes, 9.2 stretch |
| 10 | Planner tool, user-tested | Yes |
| 11 | Full project report with merged references | Yes |
| 12 | Final deck, notebook, patent decision, hand-over | Yes |
| 13 | U-Net / spatial deep model | Only if Step 6 has several hundred urban labels |

---

**Reminder:** the whole phase turns on Step 2. A waterlogging dataset with dates, real locations and sources is what lets the project prove (or honestly disprove) the development effect its title promises. Start the data requests in Week 0, not when Step 2 begins.
