# Flood Inventory via Otsu Thresholding (Step 5)

Status as of 2026-09-23. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84).

**Script:** `scripts/09_build_flood_inventory_otsu.py`
**Inputs:** Step 4 GEE Sentinel-1 VV scenes (`data/raw/sentinel1_gee/`), FABDEM, NDEM (validation only)
**Rasters:** `data/processed/flood_inventory/` (git-ignored)
**Metadata:** `data/processed/flood_inventory/flood_inventory_metadata.json`
**Maps:** `outputs/maps/step5_flood_inventory_*`

## 5.1 dB conversion (base paper Eq. 1)

Base paper (Mohamadiazar et al., 2024) Eq. 1: `σ⁰_dB = 10·log10(σ⁰)`.
GEE's `COPERNICUS/S1_GRD` is already calibrated, terrain-corrected σ⁰ **in dB**,
so the script *checks* the input is dB (all six scenes: median −7.5 to −9.2 dB)
instead of log-transforming twice. To mirror the base paper's SNAP chain,
each scene is taken dB → linear, speckle-filtered (Lee 5×5, which must run on
linear intensity), and brought back to dB with Eq. 1.

## 5.2 Otsu thresholding (VV)

Otsu is computed per image on the filtered VV histogram, restricted to
[−30, +5] dB (512 bins). Every histogram is clearly bimodal (water mode
≈ −22 dB, land mode ≈ −7 dB), and every threshold falls in the valley and
inside the typical C-band VV water range (−24 to −12 dB):

| Event | Flood scene | Otsu (flood) | Dry ref | Otsu (dry) |
|---|---|---|---|---|
| 2018 | 2018-08-21 | −13.66 dB | 2018-03-30 | −14.55 dB |
| 2019 | 2019-08-10 | −13.83 dB | 2019-03-19 | −15.61 dB |
| 2021 | 2021-10-16 | −13.44 dB | 2021-03-14 | −15.27 dB |

## 5.3 Binary flood / no-flood rasters

`flood = water(flood scene) AND NOT water(same-orbit dry reference)`, then
FABDEM slope > 5° removed (radar shadow) and 8-connected specks < 5 px
(~0.2 ha) removed.

| Event | Water on flood date | Reference water | Flood (final) | % of AOI |
|---|---|---|---|---|
| 2018-08-21 | 315.9 km² | 299.8 km² | **23.5 km²** | 1.39 % |
| 2019-08-10 | 324.4 km² | 296.3 km² | **28.9 km²** | 1.70 % |
| 2021-10-16 | 293.8 km² | 294.1 km² | **8.9 km²** | 0.52 % |

Files per event (EPSG:4326, 20 m S1 grid, nodata 255):
- `flood_inventory_<year>_<date>.tif`: 1 = flooded, 0 = not flooded
- `water_classes_<year>_<date>.tif`: 0 = dry land, 1 = flood, 2 = reference (dry-season) water

**For Step 8 sampling:** draw non-flood samples from class 0 only, never
class 2 (permanent water is neither flood nor safe land).

## 5.4 Visual validation — CHECKPOINT (Aug 2018)

Maps: `step5_flood_inventory_2018_20180821_overview.png`,
`..._site_checks.png` (six zoom panels over Esri World Imagery),
`..._interactive.html` (toggle flood / NDEM layers over imagery).

What the 2018 map shows:
- **Pass — spatial pattern.** Flooded patches cluster on the northern Periyar
  floodplain (Kunnukara / Puthenvelikkara / Chengamanad / east of Aluva),
  the area worst hit in 2018, and line up closely with the official
  NRSC/ISRO NDEM same-day polygons (yellow outlines in the zoom panels).
- **Pass — control.** Kochi city (Ernakulam) shows essentially no new water
  (0.7 %), with NDEM also 0 % there.
- **Pass — river overbank.** The Periyar at Kalady and upstream of Aluva is
  visibly wider than in March (orange fringe along the channel).
- **Known miss — airport runway.** The Cochin airport runway/apron is already
  radar-dark (smooth tarmac) in the March dry scene, so it is classed as
  reference water and cannot register as flood, although NDEM shows it
  flooded. Change detection is blind to anything dark in both scenes.
- **Known miss — urban flooding.** Low-backscatter Otsu cannot detect water
  among buildings (double-bounce *raises* backscatter). This was handled
  separately in `test_urban_double_bounce.py`, not here.
- **Timing caveat.** The nearest orbit-165 pass is 2018-08-21, ~4–6 days
  after the 15–17 Aug peak, so this is *residual* flooding, not peak extent.

Quick same-day agreement with NDEM (sanity only; the formal cross-check is Step 6):

| Event | NDEM area (outside ref. water) | POD | FAR | CSI |
|---|---|---|---|---|
| 2018-08-21 | 11.5 km² | 0.41 | 0.80 | 0.16 |
| 2019-08-10 | 32.3 km² | 0.20 | 0.78 | 0.12 |
| 2021-10-16 | 0.9 km² | 0.48 | 0.95 | 0.05 |

The high FAR is partly by construction: NDEM passes are ~12 h later (evening
IST), NDEM appears to exclude the normal river channel (our overbank fringe
counts against us), and NDEM maps only 11.5 km² in this AOI on 21 Aug.

**Verdict:** the threshold is sound (bimodal histogram, valley threshold, the
right places light up, the control stays clean). 2018 and 2019 are usable as
flood inventories. **2021 is not** — NDEM maps < 1 km² here and our 8.9 km²
is dominated by specks and backwater-edge slivers; treat 2021 as a near-null
event and do not use it as a positive-label source without further filtering.

## Side finding

`data/raw/esa_worldcover_2021_aoi.tif` covers only part of the AOI
(bounds 76.15–76.467 E, 10.008–10.25 N), so it was **not** used for the
permanent-water mask. The earlier pipeline scripts that use it should be
checked before Step 7.
