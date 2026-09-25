# Flood Inventory Cross-check against Independent Sources (Step 6)

Status as of 2026-09-25. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84).

**Script:** `scripts/10_crosscheck_flood_inventory.py`
**Inputs:** Step 5 rasters (`data/processed/flood_inventory/water_classes_*.tif`),
NDEM polygons (`data/validation/`), GEE (DFO database, Sentinel-2)
**Metrics:** `outputs/metrics/step6_flood_inventory_crosscheck.json`
**Maps:** `outputs/maps/step6_crosscheck_2018_agreement_maps.png`,
`outputs/maps/step6_crosscheck_gti_summary.png`

## 6.1 Which independent source?

The plan names Copernicus EMS or DFO. Both were checked first. Neither covers our event.

| Source | Usable? | Why |
|---|---|---|
| Copernicus EMS Rapid Mapping | **No** | No EMSR activation for the Aug-2018 Kerala floods. The event was mapped under International Charter activation 582, and the Indian products from that are NRSC/ISRO's NDEM maps. |
| DFO Global Flood Database (GEE `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1`) | **No** | 24 events over the AOI, but the latest is 2016-07-07, and there are **0** India events for 2018 (queried live by the script). The DFO archive website returns HTTP 410. It would also be MODIS 250 m, about 150× coarser than our pixels. |
| **A. NDEM (NRSC/ISRO)**: official flood-inundation polygons | **Yes** | India's national emergency-mapping agency. Its SAR maps (RADARSAT-2 / Sentinel-1) were made by a different team with a different processing chain and thresholds. |
| B. Sentinel-2 MNDWI (optical) | **Tried, rejected** | 22 Aug 2018 is the only low-cloud scene near our pass by tile metadata (13 %), but s2cloudless flags about 82 % of the AOI as cloud. The only cloud-free pixels are the Arabian Sea, so **0 %** of the land can be compared. The next clear scene, 2018-09-04, is two weeks later: it would test how long the flood lasted, not agreement. Aug 2019 tiles are 73–100 % cloud. The script keeps this check and rejects it automatically below 20 % clear land. |

NDEM is used in two ways:
- **A1, same-day.** Only the NDEM passes on our Sentinel-1 date (2018: 21 Aug, 18:00–19:30 IST).
- **A2, event envelope.** The union of all NDEM passes for the event. For 2018 that is 17, 18, 21, 24, 27 and 28 Aug, including the near-peak maps of 17–18 Aug (75 km² in the AOI). Water still standing on 21 Aug should lie inside the area the event flooded at some point, so A2 answers the GTI question best: *is what we call flooded real flooding?*

## 6.2 Agreement numbers

**GTI analogue.** The base paper's Ground Truth Index (84.05 %) is "how often the model's flood predictions matched real, reported flood locations", i.e. the share of *our* flood calls that the reference confirms (a precision). Three versions are reported:
- **GTI_px**: share of our flood pixels inside the reference extent (strict).
- **GTI_40m**: the same, but a pixel counts as confirmed if a reference flood pixel lies within 40 m (2 px). This absorbs the resolution and geolocation mismatch between two independently made maps.
- **GTI_patch**: share of our flood patches (≥ 10 px) that touch the reference within 40 m. This is the closest analogue to the base paper's check against point reports.

**Random baseline.** The share a random map with the same flood area would score, i.e. the reference's prevalence in the domain. **Lift** = GTI ÷ baseline.

**Comparison domain.** Valid pixels, excluding our reference (dry-season) water, which is neither flood nor dry land.

### Headline (report this)

> **2018: 58.4 % of our Otsu flood pixels are confirmed by the official NRSC/ISRO NDEM flood extent (within 40 m), 4.4× a random map (13.3 %). The strict per-pixel figure is 36.7 %. Base paper GTI: 84.05 %.**

### Full table

| Event | Reference | Ours km² | Ref km² | GTI_px | **GTI_40m** | Random (40 m) | Lift | GTI_patch | POD | CSI | κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2018 | A1 NDEM same-day | 23.5 | 11.5 | 20.2 % | 39.7 % | 2.0 % | 20.1× | 19.3 % | 0.41 | 0.16 | 0.26 |
| 2018 | **A2 NDEM envelope** | 23.5 | 75.1 | 36.7 % | **58.4 %** | 13.3 % | 4.4× | 38.6 % | 0.12 | 0.10 | 0.15 |
| 2019 | A1 NDEM same-day | 28.9 | 32.3 | 22.1 % | 36.7 % | 4.5 % | 8.3× | 25.7 % | 0.20 | 0.12 | 0.19 |
| 2019 | A2 NDEM envelope (10+12 Aug) | 28.9 | 32.6 | 22.8 % | 37.1 % | 4.5 % | 8.3× | 25.7 % | 0.20 | 0.12 | 0.20 |
| 2021 | A1 NDEM same-day | 8.9 | 0.9 | 5.1 % | 6.7 % | 0.2 % | 33.9× | 4.2 % | 0.48 | 0.05 | 0.09 |
| 2021 | A2 NDEM envelope (16+19 Oct) | 8.9 | 4.1 | 13.2 % | 15.2 % | 0.6 % | 26.3× | 8.1 % | 0.28 | 0.10 | 0.18 |

Overall accuracy is in the JSON (0.94–0.99). It is **not** reported as a headline because it is dominated by the ~98 % of pixels that both maps call dry. Papers quoting "94 % accuracy" for Kerala 2018 are quoting this inflated number.

### Where the disagreement is

Diagnostic: of our 2018 flood pixels that the NDEM envelope does **not** confirm (9.8 km²), **53 %** lie within 60 m of permanent water (river channel / backwater edges). That belt is only 4.9 % of the land domain. Excluding it, **GTI_40m rises to 74.2 %** for 2018 (48.8 % for 2019).

This is a diagnostic, not the headline. It agrees with what the Step 5 maps showed:
- **River/backwater fringe.** The Periyar is visibly wider in August than in March (the orange band east of Aluva in the agreement map). NDEM appears to mask a generous normal-river-channel zone, so our real overbank water there counts as unconfirmed.
- **Small patches.** Patch-level GTI (38.6 %) is well below pixel-level GTI (58.4 %). Big patches are confirmed and many small ones are not: small isolated patches are the least reliable part of the inventory.
- **Timing (low POD in A2 is expected).** Our 21 Aug scene is 3–4 days after NDEM's 17–18 Aug near-peak maps. We see 23.5 km² of *residual* water against a 75 km² envelope, so we "miss" most of the peak extent by construction. Recall against the envelope is not a fair score for a post-peak image; precision (GTI) is.
- **Evening vs morning.** Same-day NDEM passes are ~12 h after our 06:10 IST pass and cover only 11.5 km² here, hence the lower A1 numbers.

## Verdict

- **2018: usable, with an honest number.** 58.4 % (40 m) / 36.7 % (strict) confirmation by an independent official source, 4.4× better than chance, rising to ~74 % away from water edges. This is below the base paper's 84.05 %. However, that figure was measured on a different kind of reference (point flood reports in Miami), with an image taken during the flood rather than 4–6 days after the peak.
- **2019: weaker.** 37.1 % (8.3× chance). NDEM and our map agree on total area (28.9 vs 32.6 km²) but overlap less. It is still usable as a secondary label source.
- **2021: confirms the Step 5 verdict.** GTI ≤ 15 %, and 80 % of the unconfirmed area hugs water edges. Do not use it for positive labels.

**Implications for Step 8 (sampling):**
1. Draw flood (label 1) samples preferentially from pixels that are both Otsu-flood **and** inside the NDEM 2018 envelope. Patches ≥ 10 px, and outside the 60 m permanent-water belt, are the most trustworthy.
2. Keep class-2 (reference water) and the 60 m water-edge belt out of the non-flood (label 0) pool.
3. Report the 58.4 % (and 36.7 % strict) figure as the inventory's validation number, with the timing caveat stated.
