# Step 9.0c: Physics effect size of development (extra runoff)

Script: `scripts/09c_extra_runoff.py` · numbers: `outputs/metrics/step9c_extra_runoff.json` · figure: `outputs/maps/step9c_extra_runoff.png` · rasters (gitignored): `data/processed/runoff/extra_runoff_{monsoon,frequent}_aoi.tif` · log: `outputs/step9c_run.log`

## Why

Step 9.0b showed the ML model cannot give the *size* of the development effect: its constrained response to paving is flat. This step computes the size from physics, **with no flood labels**.

## Method

- **SCS curve number (USDA TR-55), per 30 m cell:**
  - CN = paved × 98 + (1 − paved) × CN_pervious.
  - Runoff Q = (P − 0.2S)² / (P + 0.8S), with S = 25400/CN − 254.
  - Wet monsoon soil uses CN_III = 23 CN / (10 + 0.13 CN).
- **Pervious CN:** TR-55 "woods–grass, fair": soil group B 65, C 76, D 82.
  - There is no basin-wide soil map (NBSS&LUP is not downloadable, and OpenLandMap is too coarse), so one group is used for the whole basin.
  - Central case: **C** (laterite / clay loam). B and D are run as sensitivity.
- **Paving:** the Step 2 / Step 4 paved fraction for 2017 and 2025, on the full upstream grid. The catchment plus AOI land is 8,973 km², of which 263 → 329 km² is paved (+25 %).
- **Routing:** runoff volumes are routed with the Step 4 FD8 routing, exactly like UICA. For every cell this gives the storm runoff arriving from upstream in 2017 and 2025, and the **% increase**.
- **Two storms:**
  - **Monsoon flood storm:** 200 mm on wet soil (the big riverine floods; CHIRPS event totals were 155–517 mm).
  - **Frequent storm:** 50 mm on normal soil (the storms behind urban waterlogging).

## Results

| | Monsoon flood storm (200 mm, wet) | Frequent storm (50 mm, normal) |
|---|---|---|
| Share of rain that runs off: unpaved vs fully paved land | 0.82 vs 0.99 | **0.20 vs 0.89** |
| Whole catchment: extra runoff 2017 → 2025 | +0.17 % (2.5 million m³) | **+1.4 %** (1.4 million m³) |
| Periyar main stem (6,378 km² upstream) | +0.07 % | +0.56 % |
| AOI land with ≥ 5 % more upstream runoff | 0 % | **19 %** |
| Most affected 1 % of AOI land | ≥ 3.7 % (max 4.9 %) | **≥ 33 %** (max 44 %) |
| Cells paved the most themselves (1 %; paved 3 % → 25 %) | +4.8 % | **+43 %** |
| Kochi city core | 0 % | 0 % |

Sensitivity (soil group C, cells paved the most): +43 % (50 mm, normal soil), +21 % (100 mm), +11 % (200 mm), +6 % (400 mm); on wet soil roughly half of each. Soil group B roughly doubles the catchment numbers; D roughly halves them. The pattern is the same in every case.

## Reading

1. **For the big monsoon floods, recent development adds very little water:** +0.2 % over the catchment, +0.07 % on the Periyar.
   - On soaked monsoon soil, unpaved land already sheds 82 % of the rain, so paving it changes little.
   - The large 2018-type floods are driven by rain and the river, not by the last eight years of building.
   - This agrees with the ML results (the radar floods are rural and riverine, and UICA adds nothing there).
2. **For frequent, moderate storms, development matters a lot locally.**
   - Unpaved soil still soaks up 80 % of a 50 mm storm; paved ground sheds 89 %.
   - Where land was newly paved, local runoff rises by **about 40 %**.
   - Downstream of growth areas, one fifth of the AOI receives ≥ 5 % more runoff, and up to 44 %.
   - This is the **urban waterlogging** problem: short, local, and invisible to radar and NDEM. It is why the label-based tests could not see the development effect.
3. **The Kochi core shows 0 %** because it was already paved in 2017; the 2017–2025 growth is in the suburbs and the eastern panchayats. The core's waterlogging is an existing-drainage problem, not a new-development one.
4. **Main conclusion for the project:** development's effect is **storm-dependent and local**. It is negligible for the extreme riverine floods, and large for the frequent storms that cause urban waterlogging. This splits the problem into two hazards and two kinds of evidence:
   - *where* big floods happen: the ML map;
   - *how much more water* development sends in everyday storms: this physics layer.

## Limits

- One soil group for the whole basin (covered by the B / C / D sensitivity).
- No drains or detention ponds. SCS assumes all runoff reaches the next cell, so the numbers are an upper bound for well-drained areas.
- Runoff *volume* only, not peak flow or timing: paving also makes water arrive faster, which this step does not model.
- Paved fraction comes from the Step 2 hybrid layer (r = 0.74 vs WorldCover).

## Use in Step 9

The scenario maps carry two layers:
- the constrained ML Δ susceptibility (9.0b): **where**;
- the extra-runoff Δ for both storms (this step): **how much**.

The frequent-storm layer is the main development-impact layer.
