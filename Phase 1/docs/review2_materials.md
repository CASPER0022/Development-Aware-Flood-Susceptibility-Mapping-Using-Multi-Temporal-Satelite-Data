# Review 2 Materials (Step 12)

Status as of 2026-09-29. Review 2 target: October 2nd week (Oct 12–18, 2026).

**Script:** `scripts/15_build_review2_deck.py` (rebuilds everything below from the step outputs)
**Deck:** `outputs/presentation/BTP_Review2_Phase1_Team67.pptx` (15 slides, 16:9, speaker notes on every slide)
**New figures:** `outputs/maps/step12_architecture_status.png`, `outputs/maps/step12_study_area.png`

Every number in the deck is read from the step metrics JSON files, not typed in, so the deck always matches the pipeline. The style follows the Review 1 deck: white 16:9, left-aligned titles, IIIT Kottayam logo, institute footer, slide numbers.

## 12.1 Architecture diagram: done vs planned

`step12_architecture_status.png` is the Review 1 system architecture redrawn with every block marked:

| Status | Blocks |
|---|---|
| **Done in Phase 1 (12)** | Sentinel-1 SAR · DEM (FABDEM) · rainfall (CHIRPS) · OSM / HydroSHEDS · SAR processing · common grid · flood inventory · elevation / slope / TWI · distance to river / road · flow direction & accumulation · LightGBM · current susceptibility map |
| **Partly done (4)** | Cloud / noise handling (SAR speckle done; Sentinel-2 cloud masking only tested) · multi-temporal alignment (same-orbit flood/dry pairs; land cover pending) · spatial block CV (run as a diagnostic, not yet the evaluation method) · temporal hold-out (2019 flood check of the map) |
| **Planned for Phase 2 (8)** | Sentinel-2 land cover / urban growth · building footprints · land-cover / built-up change · UICA · ΔUICA · PU-learning · post-development susceptibility · Δ susceptibility |

Plus two more blocks:
- **Explainability:** global importance is done (partial); SHAP / counterfactuals and planner dashboards are planned.
- **Independent validation loop:** done, via NRSC/ISRO NDEM.

## 12.2 Slides

| # | Slide | Main content |
|---|---|---|
| 1 | Title | Team, guide, "Review 2 — Phase 1" |
| 2 | From Review 1 to Review 2 | Base paper vs our reproduction, stage by stage; why LightGBM, not U-Net |
| 3 | System architecture — progress | 12.1 diagram |
| 4 | Study area & data | **AOI map** (plan 12.2), dataset table |
| 5 | Flood inventory | **Flood inventory map** (plan 12.2), Otsu histograms, key areas |
| 6 | Is the flood inventory right? | GTI analogue 58.4 % vs NDEM, 4.4× chance, 74 % away from water edges |
| 7 | Features & training data | 11-feature stack, 30,568 balanced samples, no leakage |
| 8 | Model results — metrics | **Metrics table** (plan 12.2): random split, unseen regions, base paper |
| 9 | What drives the model? | **Feature importance plot** (plan 12.2) + Table 4-style ablation |
| 10 | Flood susceptibility map | **Susceptibility map** (plan 12.2) + 2019 validation |
| 11 | Finding: coarse rainfall | With vs without rainfall |
| 12 | Known limitations → Phase 2 | From `limitations.md` |
| 13 | Conclusion | – |
| 14 | Anticipated questions | Backup slide |
| 15 | References | – |

Presenting tip: slides 2, 8 and 10 carry the story. The reproduction is complete (slide 2), the scores are honest (slide 8), and the map predicts a flood it never saw (slide 10).

## 12.3 Anticipated panel questions

**"Why LightGBM and not U-Net like the base paper?"**
No labelled image dataset existed for Kerala. The base paper trained on 226 images, and building patches and tuning a CNN would have consumed the whole review timeline. LightGBM let us build and verify the complete chain, and it is the method of our Kerala reference papers. U-Net is revisited in Phase 2, once a multi-event inventory exists. (Step 0)

**"How do you know your flood inventory is correct?"**
- 58.4 % of our 2018 flood pixels are confirmed by the official NRSC/ISRO NDEM flood maps within 40 m. A random map would score 13.3 %, so this is 4.4× chance.
- Away from river and backwater edges, it is 74.2 %.
- We checked Copernicus EMS (no activation for this event), DFO (no 2018 India event) and Sentinel-2 (82 % cloud) first. (Step 6)

**"Why is your GTI lower than the base paper's 84 %?"**
- Our only usable Sentinel-1 pass (21 Aug) is 4–6 days after the peak, so we see residual water: 23.5 km² vs 75 km² at the peak.
- Most disagreement sits on water edges.
- The base paper checked against 116 point reports taken during the flood, which is a different and easier kind of reference.

**"Is your model actually good, or just overfit?"**
- The 99.1 % comes from the random split used by the reference papers. We tested it: 98.7 % of flood test pixels sit next to a training pixel.
- On whole 10 km regions the model never saw, accuracy is 88.5 %, F1 84.5 %, AUC 0.979. That is our honest number and is comparable to the base paper's 93.7 %.
- It is not feature leakage: no feature comes from the radar image.
- Spatial CV becomes our evaluation method in Phase 2. (Step 9)

**"Does the susceptibility map mean anything?"**
We tested it on the Aug 2019 flood, which was never used in training. 83.8 % of the official 2019 flood falls in the High / Very High classes, which cover 40 % of the land; Very High holds 3.5× its area share; AUC is 0.83. Even for 2019 flooding outside anything that flooded in 2018, AUC is 0.78. (Step 10)

**"Which factors matter most? Does it agree with the base paper?"**
- Elevation dominates, followed by TWI, distance to road, rainfall and distance to the major river. That fits river flooding on a coastal floodplain.
- The base paper's Table 4 is an input ablation (slope, soil, imperviousness, rainfall). Our ablation agrees that more inputs help and rainfall alone is weakest.
- Imperviousness, their key input, is missing in ours; that is exactly our Phase 2 UICA contribution. (Step 9)

**"Why does rainfall rank high if it does not transfer?"**
CHIRPS is 5.5 km, about 60 cells with a 10 % range here. The model uses it to recognise *where* the 2018 flood was, not how much it rained. Alone it scores 55 % accuracy on unseen regions, which is chance. We show a no-rainfall map alongside. (Steps 9–10)

**"Why not use the 2021 flood?"**
It fails the NDEM check (GTI 15 %). NDEM mapped under 1 km² in our AOI, so it is a near-null event and would add noisy labels. (Steps 5–6)

**"Why balanced 50/50 sampling?"**
To match the base paper's balanced design. The consequence is that model scores are a relative index, not flood probabilities, which is why the map uses quantile classes. Calibration is a Phase 2 task. (Steps 8, 10)

**"What exactly is new in your project? What is left?"**
Phase 1 reproduces the base paper. The novelty is Phase 2:
- UICA / ΔUICA (upstream impervious area and its change);
- PU-learning (unobserved urban floods are not "safe");
- spatial and temporal validation;
- a multi-event inventory;
- the dual current vs post-development susceptibility, with explanations.

Each Phase 1 limitation maps onto one of these (`limitations.md`).

## Regenerating

```bash
python "Phase 1/scripts/15_build_review2_deck.py"
```

This requires `python-pptx` (listed in `requirements.txt`) and the outputs of Steps 5–10. Close the deck in PowerPoint before rebuilding: Windows locks open files.
