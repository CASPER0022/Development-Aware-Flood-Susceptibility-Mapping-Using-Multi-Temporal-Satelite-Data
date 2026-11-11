# Step 3: Expanded flood inventory, 2017–2025

Script: `scripts/03_expand_flood_inventory.py` · numbers: `outputs/metrics/step3_flood_inventory_events.json`, `outputs/metrics/step3_event_table.csv` · figures: `outputs/maps/step3_chirps_candidates.png`, `step3_event_table.png`, `step3_kept_events.png` · rasters: `data/processed/flood_inventory/water_classes_<yyyymmdd>_o<orbit>.tif` (gitignored, regenerable)

## Result in one line

**Four flood events pass every check: Aug 2018, Aug 2019, Aug 2020 and Aug 2024.** Phase 1 had two usable events (2018, 2019). Four events make the temporal hold-out possible: one event can be held back purely for testing.

## 3.0 Regression test

The Step 5 chain now lives in `common.py`. Rerun on the Phase 1 2018 scene pair (21 Aug flood, 30 Mar dry), it gives a class raster **identical to Phase 1, pixel for pixel** (23.54 km² of flood, Otsu −13.66 dB). The script stops if this test fails.

## 3.1 Candidate events

Two independent lists were merged:

- **NDEM (NRSC/ISRO) map dates over the AOI**, 2017–2025, grouped into events when dates are ≤ 14 days apart: Aug 2018, Aug 2019, Aug 2020, Oct 2021, Nov 2021, Aug 2022, Jul 2023, Aug 2024 (8 events).
- **CHIRPS daily AOI-mean rainfall**: the 10 largest 3-day totals, at least 15 days apart. The two largest are the known floods, 8 Aug 2019 (404 mm) and 16 Aug 2018 (311 mm). The others fall in 2017, 2018, 2022, 2024 and 2025.

Rainfall peaks inside an NDEM window were merged into that event, which leaves **16 candidate events**.

## 3.2 Sentinel-1 acquisitions

Each event was searched within ±3 days in both pass directions. One acquisition means all frames from one date and relative orbit. Every acquisition needs a dry reference from the same relative orbit, taken between 15 Jan and 30 Apr of the same year (closest to 20 Mar). All scenes are sampled directly onto the Phase 1 20 m inventory grid.

- **Only descending orbit 165 is usable.** Ascending orbit 173 was imaged over the AOI on 21 and 27 Aug 2018, but it has no dry-season scene in Jan–Apr 2018, so a same-geometry change detection is not possible. These passes are recorded as rejected.
- **Five events have no acquisition in their window:** Jul 2018 (first peak), Nov 2021, Aug 2022 (NDEM date), Jul 2023 and Jun 2025. After Sentinel-1B failed in Dec 2021, the AOI was imaged only every 12 days.
- **Total:** 14 acquisitions, 12 of them mappable.

## 3.3 Mapping, cross-check and keep rule

Every acquisition went through the Phase 1 chain: Lee filter and Eq. 1, per-image Otsu, change against the dry reference, removal of slopes > 5° and patches < 5 px. Each was then cross-checked against its event's NDEM envelope with a 40 m tolerance. The keep rule was fixed before the run and calibrated so that it reproduces the Phase 1 decisions (2018 and 2019 kept, 2021 rejected). An acquisition must pass all five checks:

1. AOI coverage ≥ 80 %.
2. Both Otsu thresholds within −24 to −12 dB.
3. Otsu separability η ≥ 0.75.
4. Flood area ≥ 2 km².
5. An NDEM map exists, with GTI₄₀ ≥ 30 % and lift ≥ 3.

| Event | S1 date (dry ref) | Flood km² | Otsu dB | NDEM km² | GTI₄₀ | Random | Lift | Decision |
|---|---|---|---|---|---|---|---|---|
| 2017-06 rain | 2017-06-03 (03-23) | 7.93 | −13.60 | – | – | – | – | unverified (no NDEM) |
| 2018-07 rain | 2018-07-28 (03-18) | 3.12 | −15.06 | – | – | – | – | unverified (no NDEM) |
| **2018-08** | **2018-08-21 (03-18)** | **24.81** | −13.66 | 75.1 | **55.9 %** | 13.3 % | 4.2 | **kept** |
| 2018-08 | 2018-08-27 (03-18) | 3.79 | −15.41 | 75.1 | 39.2 % | 13.3 % | 2.9 | rejected (lift < 3) |
| **2019-08** | **2019-08-10 (03-19)** | **28.90** | −13.83 | 32.6 | **37.1 %** | 4.5 % | 8.3 | **kept** |
| **2020-08** | **2020-08-10 (03-19)** | **20.38** | −12.87 | 13.3 | **30.0 %** | 2.8 % | 10.6 | **kept (borderline)** |
| 2021-10 | 2021-10-16 (03-14) | 8.86 | −13.44 | 4.1 | 15.2 % | 0.6 % | 26.3 | rejected (GTI < 30 %) |
| 2022-08 rain | 2022-07-31 (03-21) | 1.09 | −15.75 | – | – | – | – | rejected (< 2 km²) |
| 2024-05 rain | 2024-05-21 (03-22) | 8.62 | −13.72 | – | – | – | – | unverified (no NDEM) |
| 2024-06 rain | 2024-06-26 (03-22) | 24.46 | −11.64 | – | – | – | – | rejected (Otsu outside water range) |
| **2024-08** | **2024-08-01 (03-22)** | **10.11** | −14.45 | 5.4 | **49.6 %** | 1.1 % | 44.2 | **kept** |
| 2025-05 rain | 2025-05-28 (03-17) | 55.06 | −10.30 | – | – | – | – | rejected (Otsu −10.3 dB, η 0.75) |

## Points to note for the review

- **2018 area differs slightly from Phase 1 (24.81 vs 23.54 km²).** The single fixed dry-reference rule picks 18 Mar 2018, where Phase 1 picked 30 Mar (that script used a different target date for each year). The GTI is almost the same (55.9 % vs 58.4 %). The regression test above shows the code itself is unchanged. The difference comes only from which dry scene is used.
- **2020 is borderline.** Its GTI₄₀ is 30.04 %, against the 30 % cut-off. Its lift (10.6) is high, and its NDEM map is large (more than 11,000 polygons, 13.3 km² outside permanent water). Treat it as the weakest of the four, and test it as a hold-out event before it is used for training.
- **η does not separate good scenes from bad ones.** It ranges from 0.83 to 0.88 for nearly every scene, because the sea and the backwaters always make the histogram bimodal. It is kept only as a sanity floor. The rejections that matter come from GTI and from the Otsu range.
- **The May 2025 and June 2024 scenes give 55 and 24 km² of "flood"** with thresholds above −12 dB. This is the signature of wet soil and paddy, or of wind-roughened water, rather than a clean flood. The keep rule rejects both.
- **Three events are unverified (2017-06, 2018-07, 2024-05).** They look plausible (thresholds near −13.6 dB, 3–9 km²), but there is no independent map to check them against. They are not used for training. If a later step needs more positives, they could be checked against Sentinel-2.
- **3.4, urban double-bounce positives: not attempted in this step.** It is optional, and it is better done in Step 7 with PU-learning.

## Effect on later steps

- **Step 6 / Step 10 temporal hold-out:** train on 2018 + 2019 (the two largest events), and hold out 2020 and 2024 as independent test years. 2024 is also the first event after most of the 2017–2024 urban growth, which links directly to Step 10.
- **The 20 m label pools for each kept event** use the same grid as Phase 1, so Step 8's exact 20 m → 30 m aggregation applies without change.
