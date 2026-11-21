# Step 9: Current vs post-development: the effect of development on flooding

Script: `scripts/09_dual_prediction.py` · numbers: `outputs/metrics/step9_dual_prediction.json`, `step9_hotspot_blocks.csv` · figures: **`outputs/maps/step9_development_effect.png`** (headline), `step9_susceptibility_scenarios.png` · rasters (gitignored): `data/processed/scenarios/scenario_<name>.tif`, `data/processed/landcover/ghsl{2025,2030}_ext_30m.tif`

## How development enters (from Steps 9.0a–c)

The ML model locates riverine floods but cannot size the development effect (9.0b). Physics shows development matters for frequent storms, not extreme ones (9.0c). So each scenario produces two layers and a combined class:

| Layer | Source | Question |
|---|---|---|
| **A. Riverine flood susceptibility** | 9.0b scenario model (D − land cover, monotone in paving / UICA). Static and rainfall features of the 2018 reference flood; development features swapped per scenario. **Fixed class breaks** (quintiles of the 2025 map). | Where do big floods happen, and does development change that? |
| **B. Waterlogging pressure** | 9.0c physics: frequent 50 mm storm on normal soil, SCS-CN (soil group C), routed with FD8 | How much more water reaches each place than in 2017? |
| **Development-impact class** | A × B | **Hotspot** = flood-prone (High / Very High in A) **and** ≥ 5 % more water (B). **Emerging** = ≥ 5 % more water, not flood-prone. **Existing** = flood-prone only. |

## Scenarios (paved fraction, full upstream grid)

| Scenario | Rule | Paved km², study area (full grid) |
|---|---|---|
| 2017 | Step 2 / 4 layer | 137.8 (473.5) |
| 2025 (current) | observed | 158.9 (605.3) |
| **2030 trend** (main) | each cell keeps its own 2017 → 2025 rate for 5 more years (growth only) | 173.5 (700.8) |
| 2030 GHSL (check) | 2025 + growth in the JRC GHS-BUILT-S 2030 projection | 161.3 (627.8) |

For each scenario, UICA and runoff are rerouted through the full basin. Check: UICA 2025 matches Step 4 to within 0.07 km² (the paved layer is stored in steps of 0.005).

## Results

**Headline figure: `step9_development_effect.png`.**

| | 2025 | 2030 trend | 2030 GHSL |
|---|---|---|---|
| **B: land receiving ≥ 5 % more water than 2017** (frequent storm) | **280 km² (19 %)** | **434 km² (30 %)** | 299 km² (21 %) |
| Median increase on that land | +9.6 % | +11.1 % | +9.5 % |
| **Hotspots: flood-prone + ≥ 5 % more water** | **113 km²** | **190 km²** | 122 km² |
| Emerging (more water, not yet flood-prone) | 167 km² | 244 km² | 176 km² |
| Share of the extra water coming from **upstream** paving (median, affected land) | 87 % | 100 % | 88 % |
| Whole-grid storm runoff vs 2017: frequent storm / monsoon storm | +1.4 % / +0.16 % | +2.5 % / +0.27 % | +1.6 % / +0.19 % |
| A: riverine susceptibility, land moving up ≥ 1 class vs 2017 | 1.1 % | 1.2 % | 1.3 % |

The study area has 1,445 km² of land. In A, a further 0.06 % of land moves *down* a class; it is driven by cells where the observed 2025 paving is lower than 2017 (Dynamic World noise), not by the model.

**Top hotspot blocks (2 km, 2030 trend):** mostly the growing eastern belt (Kolenchery, Pallikkara near Kakkanad, Angamaly, Kalady), where paving roughly doubles from a low base (e.g. 3 % → 7 %) on valley floors that are already flood-prone. The full list is in `step9_hotspot_blocks.csv`.

**Absolute size and threshold check.** The % numbers are relative to 2017. In absolute terms, the extra water on affected land is a **median of about 1 mm** (90th percentile 2.8 mm in 2025, 3.9 mm in 2030 trend), spread over each place's whole upstream area, in a 50 mm storm. That storm yields about 10 mm of runoff on unpaved soil, so 1 mm is roughly +10 %.

The rising trend does not depend on the 5 % threshold:

| Threshold | Land with more water, 2025 → 2030 trend | Hotspots, 2025 → 2030 trend |
|---|---|---|
| ≥ 2 % | 521 → 632 km² | 234 → 290 km² |
| ≥ 5 % (used) | 280 → 434 km² | 113 → 190 km² |
| ≥ 10 % | 133 → 242 km² | 50 → 96 km² |

## Reading: the effect of development on flooding

1. **Development measurably increases flood water, but through frequent storms.**
   - In a frequent 50 mm storm, one fifth of the study area (280 km²) already receives ≥ 5 % more water than in 2017. This rises to **30 % (434 km²) by 2030** on the current trend.
   - **113 → 190 km²** of that is land the model already rates flood-prone: development is adding water where flooding already happens.
2. **The added water comes mostly from upstream.** About 87 % of the extra water at affected places comes from paving further up the catchment, not on the spot. This is exactly what UICA was designed to capture. The planning lesson: a plot can be flooded by development it cannot see.
3. **Extreme monsoon floods barely change:** +0.2–0.3 % runoff; riverine susceptibility rises for about 1 % of land. Big floods like 2018 are driven by rain and the river. This matches every label-based result (Steps 8 and 9.0a).
4. **The two 2030 scenarios bracket the outcome.** Trend continuation (434 km² affected) vs the GHSL projection (299 km²). GHSL projects slower growth than our own 2017–2025 record shows.
5. **Where:** the fastest-growing hotspots are the eastern suburbs and panchayats, not the Kochi core, which was already paved before 2017.

## Assumptions and choices made after seeing data (stated openly)

- **NDEM detail rule (9.0a):** 50 vertices per km was set after looking at the polygon outlines, before any model was trained.
- **The frequent-storm case (9.0c) was added after the monsoon case showed almost no effect.** It is physically motivated (paving matters where soil can still absorb water), and both cases are always reported.
- **Scenario-model rule (9.0b):** the pre-set limit of ≤ 0.01 AUC cost was exceeded (0.011). The constrained model was accepted because the alternative makes 94 % of developed cells "safer".
- **The 5 % threshold** is a choice; the table above shows the conclusion holds at 2 % and 10 %.
- **The urban test of UICA (9.0a)** rests on one test flood (2020).

## Limits

- **The result is modelled, not yet observed.** It shows development sends more water to flood-prone places. Whether that produced more observed flooding is tested in Step 10.
- **The paving record is noisy:** hybrid GHSL + Dynamic World, r = 0.74 vs WorldCover, and some cells show paving falling. The trend scenario extends this noise; the GHSL scenario gives about 30 % less effect.
- **Paddy-field and wetland storage is not represented.** In Kerala, much development fills paddy fields and wetlands that used to store floodwater. SCS-CN treats all unpaved land alike, so the lowland effect is probably **under-estimated**. This is a Phase 3 candidate.

- Layer B is physics for one soil group (C). B / D change magnitudes by about 2× (Step 9.0c), but not the pattern.
- No drains or detention, volume only (not peak timing).
- Layer A inherits the radar-label bias, so built-up susceptibility is a lower bound.
- The hotspot threshold (5 %) is a choice. The class areas for other thresholds can be recomputed from the scenario rasters.
- Town labels are nearest-town names within 5 km (approximate centres).
- The **evidence that this extra water actually turns into more flooding** is tested in Step 10 (natural experiment).
