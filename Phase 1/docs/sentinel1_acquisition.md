# Sentinel-1 SAR Acquisition (Step 4)

Status as of 2026-09-22. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84), ~1,696.6 sq km.

**Script:** `scripts/08_fetch_sentinel1_sar.py`
**Metadata:** `data/raw/sentinel1_gee_metadata.json`
**Rasters:** `data/raw/sentinel1_gee/*.tif`

## 4.1 Flood date selection

Three flood events were checked for applicability to this specific AOI using
the AOI-mean CHIRPS daily rainfall record already on disk
(`data/raw/chirps_daily_timeseries_2017_2023.csv`), not just news reports:

| Event | Monthly AOI total | Peak day | Applicable? |
|---|---|---|---|
| Aug 2018 (catastrophic Kerala flood) | 888.9 mm | 2018-08-15 (115.7 mm) | Yes — primary event |
| Aug 2019 (Kerala's second major flood year) | 1070.9 mm | 2019-08-08 (202.1 mm) | Yes — CHIRPS shows this AOI's rainfall was actually *higher* than 2018, even though the worst landslide damage that year was reported in Wayanad/Malappuram |
| Oct 2021 (Kottayam/Idukki flood-landslide event) | 545.2 mm | 2021-10-16 (55.7 mm) | Yes — Idukki dam sits on the Periyar upstream of this AOI, so elevated reservoir releases plausibly raised river levels downstream through the AOI |

All three exceed this AOI's typical monthly total for their month across
2017-2023, so all three were pulled.

## 4.2 Scene selection

Source: GEE `COPERNICUS/S1_GRD`, IW mode, VV polarization. For each event,
the closest available scene (by date) to the target flood date was picked,
plus a same-year pre-monsoon "dry" reference scene, preferring an exact
`relativeOrbitNumber` match (identical imaging geometry/incidence angle) to
the flood scene.

| Event | Flood scene date | Δ days | Dry scene date | Δ days | Orbit pass | Rel. orbit | Geometry match |
|---|---|---|---|---|---|---|---|
| 2018 | 2018-08-21 | 5 | 2018-03-30 | 0 | DESCENDING | 165 | exact relative orbit |
| 2019 | 2019-08-10 | 1 | 2019-03-19 | 4 | DESCENDING | 165 | exact relative orbit |
| 2021 | 2021-10-16 | 1 | 2021-03-14 | 1 | DESCENDING | 165 | exact relative orbit |

All three years happened to resolve to the same relative orbit (165,
descending), so the flood/dry pairs are on genuinely identical imaging
geometry every year — no incidence-angle mismatch to correct for.

## 4.3 Export

Each scene's VV band was clipped to the AOI and downloaded as a
single-band GeoTIFF (EPSG:4326). The requested native ~10 m/pixel export
exceeded GEE's 50 MB per-request `getDownloadURL` limit
(156 MB uncompressed at 10 m over this AOI), so the script automatically
retried at 20 m/pixel, which fit (~32 MB each). This fallback is recorded
per-file in the metadata (`scale_m`).

**Important — GEE's dB conversion:** `COPERNICUS/S1_GRD` is already
thermal-noise-removed, radiometrically calibrated, and terrain-corrected by
GEE's own preprocessing chain, and the resulting sigma0 backscatter is
delivered **already in decibels**. Step 5.1 ("convert Sentinel-1 backscatter
to dB") is therefore already satisfied by this source — no further log
transform should be applied on top of these pixel values.

Sanity check (percentiles of `s1_vv_2018_flood_20180821.tif`, all valid,
n=4,342,372 px): 1st pct = -26.1 dB, median = -7.5 dB, 99th pct = +1.1 dB —
consistent with a mix of open-water/flooded (very low dB), vegetated/bare
land (mid dB), and urban double-bounce (positive dB) targets. A small tail
above +10 dB (max +34.4 dB, <0.1% of pixels) is consistent with strong
corner-reflector-type urban/industrial returns already seen in this AOI's
earlier SAR analysis (`test_urban_double_bounce.py`), not corrupted data.

## Output files

```
data/raw/sentinel1_gee/
  s1_vv_2018_flood_20180821.tif   s1_vv_2018_dry_20180330.tif
  s1_vv_2019_flood_20190810.tif   s1_vv_2019_dry_20190319.tif
  s1_vv_2021_flood_20211016.tif   s1_vv_2021_dry_20210314.tif
data/raw/sentinel1_gee_metadata.json
```

## Note on the earlier ad hoc 2018 SAR pipeline

`test_urban_double_bounce.py` / `local_ccd_flood_detection.py` /
`01_define_aoi_and_validate_flood.py` already process a 2018 dry/flood pair,
but pulled directly from the raw AWS Sentinel-1 L1C bucket with manual
GCP-based georeferencing and an approximate linear DN→dB calibration
(`20*log10(DN) - 55.0`, explicitly flagged there as uncalibrated). This
step's GEE-sourced scenes are properly calibrated/terrain-corrected by GEE
and cover 2019 and 2021 as well, so they are the inputs Step 5's Otsu
thresholding should use going forward, not the earlier ad hoc arrays.
