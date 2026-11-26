# Step 12: Phase 2 maps and review materials

Scripts: `scripts/12_phase2_maps.py`, `scripts/12b_build_phase2_deck.py` · numbers: `outputs/metrics/step12_headline_tables.json`

## 12.1 Report maps

Each map has a legend, scale bar, north arrow, lon/lat ticks and town labels.

| File | Content |
|---|---|
| `outputs/maps/step12_map_development_effect.png` | **Headline.** Extra frequent-storm water 2017 → 2030, and the development-impact classes |
| `outputs/maps/step12_map_current_vs_post_development.png` | Riverine susceptibility 2025 vs 2030 (scenario model, fixed class breaks) and cells moving up a class |
| `outputs/maps/step12_map_phase1_vs_phase2.png` | Phase 1 map vs the Phase 2 chosen model (PU bagging, set C, trained 2018 + 2019, shown for 2024) and class change |

- **Phase 1 vs Phase 2:** 53 % of land is in the same class, and 92 % is within one class.
- **Main visible differences:** Phase 2 is sharper along river corridors and valley floors. Like the radar-label models in general, it rates the dense city lower; see the Step 8 / 9.0b caveat.

The Phase 2 current-susceptibility raster is saved at `data/processed/scenarios/phase2_current_susceptibility.tif` (gitignored, regenerable).

## 12.2 Headline tables

These are in `step12_headline_tables.json` and `outputs/maps/step12_headline_tables.png`.

**Phase 1 vs Phase 2 (AUC):**

| Test | Phase 1 | Phase 2 |
|---|---|---|
| Random 70/30 split | 0.999 | 0.999 |
| Unseen 10 km regions (4 events) | 0.962 | 0.992 |
| New year (2020 / 2024) | 0.988 / 0.989 | 0.992 / 0.993 |
| **New year + new place (strict)** | **0.949 / 0.954** | **0.985 / 0.985** |

**Effect of development:**

| Evidence | Big riverine floods | Frequent storms / waterlogging |
|---|---|---|
| ML, radar labels | none | not observable |
| ML, radar + NDEM labels | paved land +0.037 AUC (one test year) | not observable |
| SHAP development share | 5.3 % | – |
| Physics (SCS-CN) | +0.17 % runoff | +1.4 % runoff; hotspots 113 → 190 km² |
| Natural experiment | none (−0.05 pp [−0.71, +0.58]) | not testable |
| Reported waterlogging spots | – | UICA ratio 57th percentile of urban land (11 / 15 above; not significant) |

## 12.3 Architecture status

`outputs/maps/step12_architecture_status.png`: blue = done in Phase 1, orange = done in Phase 2, dashed grey = Phase 3 (observed proof of the development effect, planner dashboard, report).

## 12.4 Deck and study notebook

- **Deck:** `outputs/presentation/BTP Phase 2 Team 67.pptx`, 22 slides on the official template (`PPT/Review 2/BTP Presentation Template-3.pptx`). Numbers are read from the JSON files. Speaker notes are tagged Albin / Niketh / Jatin / Ranjith.
- **Study notebook (published artifact):** not updated yet. Pending a decision on whether to extend the Review 2 notebook or publish a Phase 2 one.
