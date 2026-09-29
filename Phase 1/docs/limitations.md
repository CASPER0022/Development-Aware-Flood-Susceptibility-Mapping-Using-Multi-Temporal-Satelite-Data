# Known Limitations of the Phase 1 Baseline (Step 11)

Status as of 2026-09-29. AOI: Lower Periyar & Aluva-Kochi Urban Corridor.

This note states where the Phase 1 baseline stops and Phase 2 begins. Phase 1 reproduces the base paper's logic (satellite → preprocessing → flood inventory → ML model → susceptibility map → accuracy metrics) with a LightGBM baseline. Every limitation below was either chosen deliberately to fit the review timeline, or measured along the way. Where a number exists, it is given, with the step that produced it.

## Headline numbers vs honest numbers

| Question | Headline (plan method) | Honest version | Source |
|---|---|---|---|
| Is the flood inventory right? | – | 58.4 % of our 2018 flood pixels confirmed by NRSC/ISRO NDEM (within 40 m); base paper GTI 84.05 % | Step 6 |
| How accurate is the model? | 99.1 % accuracy, AUC 0.999 (random 70/30 split) | **88.5 % accuracy, F1 84.5 %, AUC 0.979** on 10 km regions the model never saw | Step 9 |
| Does the map predict real floods? | – | 83.8 % of the Aug 2019 NDEM flood (never used in training) falls in the High/Very High 40 % of land; AUC 0.83 | Step 10 |

The random-split number reproduces the reference papers' method, so we report it for comparability. We present it next to the honest numbers, never alone.

## A. The five planned scope limitations (plan 11.1)

### 1. Random split instead of spatial cross-validation
- **What:** Step 9 uses a random 70/30 split, as the reference papers do.
- **Why it matters:** flood pixels come in patches. In the random test set, 98.7 % of flood pixels have a training pixel in the very next 30 m cell, so the test mostly checks near-copies of training data.
- **Measured effect:** accuracy falls from 99.1 % (random) to 93.5 % (unseen 2 km blocks), 89.8 % (5 km) and 88.5 % (10 km). AUC stays at 0.98–0.99, so the model ranks terrain well; the loss is mostly recall in new regions.
- **Phase 2:** spatial block CV becomes the evaluation method. The 2 km `block_id` is already in the Step 8 dataset.

### 2. No PU-learning (non-flood labels are uncertain)
- **What:** label 0 means "not seen flooded by Sentinel-1 on 21 Aug 2018", not "cannot flood".
- **Why it matters:** radar cannot see water among buildings (double-bounce raises backscatter), and anything radar-dark in both scenes, such as the Cochin airport runway, cannot register as flood. So urban flooding sits in the non-flood pool.
- **Measured effect:**
  - Distance to road separates the classes strongly (single-feature AUC 0.83). Part of that is this blindness: floods near roads rarely enter the inventory.
  - Kochi city tends toward the lower susceptibility classes.
  - Mitigation already applied: non-flood samples exclude the whole NDEM 2018 envelope, the 60 m water fringe and anything within 60 m of detected flood (Step 8).
- **Phase 2:** positive-unlabelled learning treats non-flood pixels as unlabelled rather than as confirmed negatives.

### 3. No land-use / imperviousness (UICA / ΔUICA) features
- **What:** the model sees only terrain, hydrology, roads and rainfall. There is no built-up or impervious-surface feature and no land-use change.
- **Why it matters:** in the base paper's input ablation (Table 4), imperviousness is one of the two most useful inputs. Development is the core of this project's title, so the current model is "development-blind".
- **Extra constraint:** the ESA WorldCover raster acquired in Step 3 covers only part of the AOI (north of 10.008° N), so it could not be used as-is.
- **Phase 2:** UICA / ΔUICA features built from multi-temporal land cover, and the dual prediction (current vs post-development).

### 4. One training event, observed after the peak
- **What:** labels come only from the Aug 2018 flood. 2019 is used for validation only, and 2021 is unusable (GTI 15 %, near-null event).
- **Why it matters:**
  - A single event trains the model on one rainfall pattern.
  - The only orbit-165 pass is 21 Aug 2018, about 4–6 days after the 15–17 Aug peak. It shows *residual* water: 23.5 km² detected against a 75 km² NDEM event envelope.
- **Phase 2:** a multi-event inventory (more Sentinel-1 dates, both orbit directions), in the spirit of the base paper's 226 images over 10 years.

### 5. LightGBM instead of U-Net
- **What:** the base paper uses a U-Net CNN on 226 Sentinel-1 images; Phase 1 uses a pixel-based LightGBM (Step 0 decision).
- **Why it matters:** a pixel model sees no spatial context (neighbouring pixels, shapes), and its metrics are not directly comparable with a segmentation network's.
- **Why it was chosen:** without an existing labelled image set, collecting images, generating patches and tuning a CNN would have used up the whole review timeline. LightGBM gave a complete, debuggable pipeline, and it is the method of the Kerala reference papers.
- **Phase 2:** revisit deep learning once the multi-event inventory exists.

## B. Limitations found while building Phase 1

### Flood inventory (Steps 4–6)
- **Agreement with the official map is moderate:** 58.4 % confirmed within 40 m (36.7 % strict per pixel), 4.4× better than chance. Away from the river/backwater fringe it rises to 74.2 %.
- **The reference is also SAR.** NDEM maps are RADARSAT-2 / Sentinel-1 products, so they share radar's blind spots. The independent optical check (Sentinel-2) was impossible: 82 % cloud on 22 Aug 2018. Copernicus EMS had no activation, and the DFO database has no 2018 India event.
- **NDEM same-day passes are ~12 h later** (evening vs our morning pass), so same-day agreement is approximate.

### Features (Step 7)
- **TWI misses the upstream Periyar catchment.** Only 643 of about 4,081 km² of upland area is routed inside the AOI, so TWI along the main river is too low.
- **HydroRIVERS is coarse.** Small streams sit a median 220 m from the DEM channels (major rivers 117 m), and distributaries are missing.
- **Rainfall is 5.5 km CHIRPS** (60 cells, about 10 % range). It acts as a **location label**, not rain physics: rainfall-only AUC is 0.82 on the random split but 0.58 on unseen regions (Step 9), and it paints CHIRPS cells onto the map (Step 10).
- **No soil input.** NBSS&LUP soil maps cannot be downloaded (PDF/WMS only). The 250 m OpenLandMap texture fallback was acquired but is not in the model. This matches the base paper's own HSG data constraint.
- **Flat coastal terrain.** In ground with only metres of relief, slope, aspect and curvature largely reflect beach ridges or DEM noise.
- **Judgment-call parameters:** 2 km drainage-density radius, Strahler ≥ 4 for "major river", TWI slope floor 0.001, aspect kept in degrees. Also, TWI was not cross-checked against SAGA / WhiteboxTools.

### Training data (Step 8)
- **Clustered samples.** Every flood patch is sampled completely (median spacing 29 m), which is the root cause of the random-split optimism.
- **Easy negatives.** Non-flood samples include the eastern hills, which inflates elevation's apparent power (single-feature AUC 0.92). On floodplain-only samples the unseen-region AUC is 0.935 and accuracy 81 % (Step 9).

### Model (Step 9)
- **Overfitting.** Training metrics are 100 %: 648 trees memorise the training pixels. It barely shows on the random split and shows clearly on spatial hold-out.
- **Untuned.** The hyper-parameters are fixed conventional values; only the number of trees was chosen, by CV inside the training part. Tuning belongs under spatial CV (Phase 2), not on this split.
- **Uncalibrated scores.** Training was balanced 50/50 while floods cover about 1–2 % of the land, and the model is overconfident. 87 % of land scores below 0.01, so scores are a relative index, not flood probabilities.

### Susceptibility map (Step 10)
- **Equal-area classes.** Quantile classes put 20 % of land in each class by construction. On the 2019 flood only **Very High** is strongly predictive (3.5× its area share); **High** is at chance (frequency ratio 0.99).
- **Rainfall artifact.** A CHIRPS cell imprints a rectangle on the NE hills. With rainfall, 23 % of land above 50 m is rated High/Very High, versus 13 % without it. A no-rainfall map is provided alongside.
- **Urban under-representation** follows from the inventory (limitation A2).

## C. Limitation → Phase 2 remedy

| Limitation | Effect in Phase 1 | Phase 2 remedy |
|---|---|---|
| Random split | 99.1 % vs 88.5 % on unseen regions | Spatial block CV as the evaluation standard |
| Non-flood = unobserved | Urban floods counted as dry; distance to road inflated | PU-learning |
| No land-use / imperviousness | Development-blind model | UICA / ΔUICA features; dual (current vs post-development) prediction |
| One post-peak event | Residual extent only; one rainfall pattern | Multi-event, multi-orbit Sentinel-1 inventory |
| Pixel LightGBM | No spatial context | Revisit U-Net / deep learning on the larger inventory |
| Coarse rainfall | Location proxy, map artifact | Event rainfall at higher resolution (e.g. IMD gridded, GPM IMERG), or drop it |
| Uncalibrated, overfit scores | Scores are ranks, not probabilities | Regularise and calibrate under spatial CV |
| No explanation of individual predictions | Only global importance | Explainability (SHAP) in Phase 2 |

## D. How to say it to the panel

- **"Why is your accuracy 99 %?"** "That is the random split used by the reference papers, and we show it for comparability. We tested it: test pixels are neighbours of training pixels. On whole 10 km regions the model never saw, accuracy is 88.5 %. That is our honest number, and spatial CV is the first Phase 2 change."
- **"How do you know the flood map is right?"** "58.4 % of our flood pixels are confirmed by the official NRSC/ISRO map, 4.4× better than chance, and 74 % away from water edges. The susceptibility map also predicts a flood it never saw: 84 % of the 2019 flood falls in our top two classes."
- **"Why LightGBM and not U-Net?"** "A deliberate trade-off (Step 0). There was no labelled image set, and a CNN pipeline would have consumed the timeline. LightGBM let us build and verify the whole chain end to end."
- **"What's missing?"** "Land use and imperviousness, which is our Phase 2 novelty (UICA/ΔUICA). Also PU-learning for unobserved urban floods, spatial CV, more flood events, and the dual current-vs-development prediction."
