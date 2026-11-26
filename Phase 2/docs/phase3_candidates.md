# Phase 3 candidate list (Step 13.3)

Each item lists the Phase 2 evidence behind it and a rough effort estimate (team-weeks).

## Work Package 1 (top priority): prove the development effect from observations

The title is development-aware. Phase 2 shows the effect by physics (Step 9) but not in observed flood data (Step 10), because satellite and NDEM maps only record the big floods.

| Item | Phase 2 evidence | Effort |
|---|---|---|
| **1a. Systematic Kochi / Ernakulam waterlogging records:** all news 2015–2026, Kochi Corporation and KSDMA complaint / incident logs, citizen apps; dated points | 15 news spots already give a UICA hint (57th–61st percentile, p = 0.08–0.19); more points give power | 2–3 |
| **1b. Urban flood detection from Sentinel-1:** coherence drop and double-bounce increase between buildings | Radar labels are 0.1 % urban (L1) | 3–4 |
| **1c. Paddy-field and wetland loss** since 2017, added as lost storage to the runoff model | SCS-CN ignores storage (L6) | 2 |
| **1d. More events:** 2025–2026 monsoons (Sentinel-1C), plus frequent-storm events, not only extremes | Natural experiment MDE about 0.5 pp (L2) | 2 |
| **1e. Optional:** a drainage / 2D hydraulic model for one fast-growing catchment (e.g. Kakkanad) | Volume only, no timing or depth (L7) | 3–4 |

**Success test:** places with more modelled extra water (Step 9) show more observed waterlogging (rank correlation or DiD with CI above 0).

## Other candidates

| Item | Motivation | Effort |
|---|---|---|
| Planner / citizen dashboard (current vs 2030 maps, hotspots, what-if paving) | Step 9 outputs exist; the NESFIC viewer is a starting point | 2–3 |
| Counterfactual recommendations: how much upstream paving to avoid to drop a site one class | Plan 11.3 stretch, not done | 1–2 |
| Event-conditioned probabilities (rainfall-dependent calibration) | L9 | 1 |
| Soil map / hydrologic soil groups | L5 | 1 |
| Replace FABDEM with Copernicus DEM for deployment | L13 | 0.5 |
| Full project report (merge Phase 1 + 2 docs, merged references) | Report material ready (13.4) | 3 |

## U-Net decision

**Still no.** The inventory grew from 1 to 4 validated events, which is still far too few scenes for a CNN. XGBoost did no better than LightGBM (Step 8), so the model family is not the limit; the labels are. Revisit only if Work Package 1 produces several hundred dated urban flood labels.
