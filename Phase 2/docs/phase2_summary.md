# Phase 2 summary (Step 13.1)

**Project:** Development-aware flood susceptibility mapping using multi-temporal satellite data · Lower Periyar (Aluva–Kochi), Kerala<br>
**Phase 2:** 22 Oct – 30 Nov 2026 · Team 67 · Guide: Dr. Rosebell Paul

## What Phase 2 set out to do

Phase 1 built and validated a baseline flood-susceptibility map (LightGBM, 11 features, the 2018 flood). Phase 2 had to deliver the **development-aware** part of the title:
- a paving record;
- UICA / ΔUICA;
- PU-learning;
- leak-free (spatial + temporal) evaluation;
- current vs post-development prediction;
- a test of whether development changes flood risk.

## What was built (Steps 1–12)

| Step | Built | Headline number |
|---|---|---|
| 1 | Shared module; Phase 1 reproduced exactly | 10 / 10 checks, AUC 0.9993 |
| 2 | Yearly paved fraction 2017–2025 (GHSL + Dynamic World) | 137.8 → 158.9 km² (about 1.8 %/yr); r = 0.74 vs WorldCover |
| 3 | Multi-event flood inventory | 16 candidates → 4 validated (2018, 2019, 2020, 2024) |
| 4 | Full-basin routing, UICA, ΔUICA | 9,130 km² basin; Periyar inflow 4,103 vs 4,081 km²; 5 / 5 self-tests |
| 5 | Feature stack v2 | HAND AUC 0.951 vs elevation 0.915; 18 features × 4 events |
| 6 | Fair evaluation (strict: new year + new place) | Phase 1 baseline 0.949 / 0.954 |
| 7 | PU-learning | PU bagging 0.984 vs standard 0.975 |
| 8 | Tuned model, A–D ablation, calibration | Phase 2 model **0.985 / 0.985**; UICA adds nothing with radar labels |
| 9.0a | Radar + NDEM labels | UICA on paved land +0.037 (CI > 0, one test year) |
| 9.0b | Monotone scenario model | "Paving makes places safer": 94 % → 0 % |
| 9.0c | SCS-CN extra runoff | monsoon storm +0.17 %; frequent storm +1.4 %; newly paved +43 % |
| 9 | Current vs post-development, hotspots | ≥ 5 % more water: 280 → 434 km²; hotspots 113 → 190 km² (2025 → 2030) |
| 10 | Natural experiment (DiD + placebo) | big floods: no effect (−0.05 pp [−0.71, +0.58]) |
| 10b | Reported waterlogging spots (15) | low-lying (28th percentile, p = 0.006); UICA ratio 57th–61st percentile (p = 0.08–0.19) |
| 11 | SHAP | terrain 59–66 %; development 4–5 % (0.6 % constrained) |
| 12 | Final maps, tables, deck, notebook | 22-slide deck; Phase 2 study notebook |

## What it means

1. **Better flood prediction.** On a new year and a new place, the Phase 2 model scores 0.985, against 0.949 / 0.954 for Phase 1. The gain comes from HAND, full-basin hydrology, event rainfall and land cover.
2. **Development barely changes the big riverine floods.**
   - Physics: +0.2 % runoff in monsoon storms on soaked ground.
   - Natural experiment: no detectable change.
   - SHAP: development is 4–5 % of the model.
3. **Development adds substantial water in everyday heavy rain**, mostly from paving upstream (87 %).
   - About one fifth of the area already receives ≥ 5 % more storm water than in 2017.
   - Flood-prone land getting more water grows from 113 km² to 190 km² by 2030 on the current trend, mostly in the eastern suburbs.
4. **Observed evidence is partial.**
   - Satellite and NDEM maps record only the big floods.
   - Fifteen reported waterlogging spots sit in low land below more paved land: a first, not yet significant, hint for UICA.
   - Proving the waterlogging effect from observations is Phase 3 Work Package 1.

## One-paragraph conclusion

Phase 2 turned the Phase 1 baseline into a development-aware system: a yearly paving record, upstream paved area (UICA), a physically constrained scenario model, and current vs 2030 maps. Under a strict new-year-and-new-place test, flood prediction improves from 0.95 to 0.985. Development hardly changes where the big monsoon floods go. But it already sends at least 5 % more storm water to about a fifth of the study area in frequent storms, including 113 km² of flood-prone land, rising to 190 km² by 2030. Most of that water comes from paving upstream. This effect is shown by physics and hinted at by real waterlogging reports; confirming it with systematic waterlogging data is the main task of Phase 3.
