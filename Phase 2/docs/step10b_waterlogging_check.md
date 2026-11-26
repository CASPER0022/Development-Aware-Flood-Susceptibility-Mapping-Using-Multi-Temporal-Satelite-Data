# Step 10b: Reported waterlogging spots vs the development layers

Script: `scripts/10b_waterlogging_check.py` · data: `data/raw/waterlogging/kochi_waterlogging_reports.csv` (sources), `kochi_waterlogging_spots_geocoded.csv` · numbers: `outputs/metrics/step10b_waterlogging_check.json` · figure: `outputs/maps/step10b_waterlogging_check.png`

## Why

Satellites and NDEM record only the big riverine floods. Urban waterlogging, where Step 9 says development matters, is recorded only in news and official reports. This is the first check of the development layers against **observed waterlogging**.

## Data

- **15 spots** reported waterlogged in news and official sources: Deccan Herald, Onmanorama, The News Minute, Urban Acres, Wikipedia (2019 floods). The source URL for each spot is in the CSV.
- **Main events:**
  - **21 Oct 2019:** about 20 cm of rain in 4 hours; the Kerala High Court ordered a waterlogging task force.
  - **28 May 2024:** cloudburst, about 150 mm at Kalamassery.
  - Recurring reports for MG Road, Ernakulam South station and the KSRTC stand.
- **Locations** come from OpenStreetMap (Nominatim). Six doubtful or missing results were re-queried and fixed to the named OSM feature (station, bus station, junction, town).
- **Uncertainty:** about 300 m, the radius used.

## Test (fixed before running)

- For each spot, take the median of the land cells within 300 m.
- Express it as a percentile among **other urban land** (paved ≥ 0.2, within 15 km of MG Road). 50 means "no different".
- Summarise as the mean over spots, with a bootstrap CI and a Wilcoxon test against 50.

## Results

| Layer | Mean percentile (15 spots) | 95 % CI | Spots above 50 | One-sided p | Without airport (14) |
|---|---|---|---|---|---|
| **HAND (low-lying)** | **28** | 15–42 | 4 / 15 | **0.006 (lower)** | 29, p = 0.012 |
| **UICA ratio 2025 (upstream paving)** | **57** | 43–70 | 11 / 15 | 0.19 (higher) | **61, p = 0.077** |
| Paved fraction 2025 | 51 | 39–62 | 10 / 15 | 0.38 | 55, p = 0.21 |
| More water 2017 → 2025 (Step 9) | 49 | 40–59 | 4 / 15 | 0.64 | 46 |
| Riverine susceptibility (Step 9) | 45 | 33–58 | 7 / 15 | 0.28 (lower) | 41 |

The seven old-core spots (MG Road, South station, KSRTC, Kaloor, Panampilly Nagar, Thammanam, Palarivattom) sit at the **70th–87th percentile of upstream paving**.

## Reading

1. **Waterlogging happens in low-lying spots** (HAND 28th percentile, significant), as expected physically.
2. **Waterlogging spots tend to drain more paved upstream land** (UICA ratio).
   - 11 of 15 spots are above the urban median, and the old core sits at the 70th–87th percentile.
   - It is **not statistically significant** with 15 spots (p = 0.19; 0.08 without the airport, whose 2019 closure was river flooding).
   - This is **the first observed hint that upstream paving (UICA) relates to waterlogging**: suggestive, not proof.
3. **Recent growth (2017 → 2025) is not where the reported waterlogging is.** The reported spots are mostly in the old core, paved long before 2017, so the Step 9 "more water since 2017" layer is flat there.
   - Waterlogging in the core is a problem of **existing** paving and drainage. The new-growth areas (eastern suburbs) have few news reports yet.
4. **The riverine model does not flag waterlogging spots** (45th percentile), as expected: it was trained on river floods. A separate waterlogging layer is needed.

## Limits

- 15 spots from news, concentrated in two events and the city core, so there is reporting bias towards prominent places.
- About 300 m location uncertainty.
- No "not waterlogged" reports, so the controls are ordinary urban land, some of which may also waterlog.
- **Phase 3 WP1:** a larger, systematic waterlogging dataset (corporation complaint logs, all news 2015–2026, citizen reports) to turn this hint into a test with enough power.
