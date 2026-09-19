# Dataset Acquisition (Step 3)

Status as of 2026-09-19. AOI: Lower Periyar & Aluva-Kochi Urban Corridor,
bbox `[76.15, 9.90, 76.55, 10.25]` (WGS84), ~1,696.6 sq km
(`Phase 1/data/aoi/periyar_study_area.geojson`).

All datasets below were re-opened from disk and verified (correct CRS,
non-empty, intersects the AOI) by
`scripts/07_dataset_acquisition_checkpoint.py` — see
`data/aoi/step3_dataset_acquisition_checkpoint.json` for the full report.
**Result: 10/10 OK, 0 WARN, 0 FAIL.**

## 3.1 DEM — FABDEM

- **Source:** GEE community catalog, `projects/sat-io/open-datasets/FABDEM`
  (Hawker et al., 2022 — forest/building-height bias removed from
  Copernicus DEM; fixes the SRTM canopy-bias issue noted from
  Kendagannaswamy et al.)
- **Resolution:** 30 m, EPSG:4326
- **Output:** `data/raw/fabdem_30m_aoi.tif` (1300 × 1486 px)
- **Sanity check:** elevation range -6 m to 392 m, mean 13.7 m — consistent
  with the Periyar lower basin / coastal lowlands rising toward the Western
  Ghats foothills.
- **Script:** `scripts/02_fetch_fabdem_dem.py`

## 3.2 Rainfall — CHIRPS Daily

- **Source:** GEE, `UCSB-CHG/CHIRPS/DAILY`, 2017-01-01 to 2023-12-31 (2,556
  daily images)
- **Native resolution:** ~5.5 km (0.05°), EPSG:4326
- **Outputs:**
  - `data/raw/chirps_daily_timeseries_2017_2023.csv` — AOI-mean daily
    rainfall (mm), for antecedent-precipitation-index features around flood
    dates
  - `data/raw/chirps_monsoon_jjas_total_by_year_2017_2023.tif` — 7-band
    (2017-2023) Jun-Sep total rainfall raster
  - `data/raw/chirps_mean_annual_rainfall_2017_2023.tif` — multi-year mean
    annual total (climatology feature)
- **Sanity check:** August 2018 AOI-mean total = **888.9 mm**, a strong
  independent confirmation of the known August 2018 Kerala flood event.
- **Script:** `scripts/03_fetch_chirps_rainfall.py`

## 3.3 Rivers / Drainage — HydroSHEDS

- **Source:** `data.hydrosheds.org`, HydroRIVERS v10 and HydroBASINS
  level-06 v1c, continent = Asia (direct download, clipped to AOI + 0.10°
  buffer so up/downstream segments aren't hard-cut at the AOI edge)
- **Outputs:**
  - `data/raw/hydrosheds/periyar_rivers_clip.gpkg` — 595 river segments (91
    with Strahler order ≥ 4, i.e. the likely Periyar mainstem/major
    tributaries)
  - `data/raw/hydrosheds/periyar_basins_lev06_clip.gpkg` — 3 level-06
    sub-basin polygons
- **Note:** the continental source shapefiles (~500 MB) are downloaded,
  clipped, then deleted automatically by the script — only the small
  AOI-clipped GeoPackages are kept.
- **Script:** `scripts/04_fetch_hydrosheds_rivers_basins.py`

## 3.4 Roads — OpenStreetMap

- **Source:** OSM via Overpass API (`osmnx`), `network_type="drive"`
- **Output:** `data/raw/osm_roads_aoi.gpkg` (layers `edges`, `nodes`)
- **Stats:** 155,985 road edges, 65,828 intersections, ~19,331 km total
  network length (UTM 43N) — a plausible density for the dense
  Kochi–Aluva urban corridor (~11.4 km/km²).
- **Script:** `scripts/05_fetch_osm_roads.py`

## 3.5 Land Cover — ESA WorldCover

- Already acquired in an earlier pass (`scripts/fetch_esa_worldcover.py`):
  `data/raw/esa_worldcover_2021_aoi.tif`, ESA WorldCover 10m v200, 2021,
  clipped to the same AOI. Reused as-is here; re-verified by the Step 3
  checkpoint.

## 3.6 Soil — NBSS&LUP (limitation) + OpenLandMap fallback

- **Checked:** NBSS&LUP (National Bureau of Soil Survey & Land Use
  Planning) and the ICAR/Bhuvan soil portals are reachable as **HTML
  viewers only** — Kerala soil maps are published as static
  district-level PDF/scanned maps or through an interactive WMS viewer.
  **There is no bulk-download API or direct GIS export**, so this dataset
  cannot be acquired programmatically for the AOI.
- **Limitation (documented explicitly, analogous to the base paper's own
  Hydrologic Soil Group / HSG data-access constraint):** NBSS&LUP soil data
  is not accessible for this project. A true HSG layer for infiltration
  behavior is therefore unavailable.
- **Fallback used:** OpenLandMap USDA soil texture class (0–20 cm), a
  global open dataset (`OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02`
  via GEE), 250 m resolution, as an infiltration-capacity proxy.
  `data/raw/soil_openlandmap_texture_class_aoi.tif`. Class `255` in the
  output is an unmasked no-data/water fill value (coastal/backwater
  pixels), not a real texture class.
- **Script:** `scripts/06_fetch_soil_data.py`

## Checkpoint

Run `python scripts/07_dataset_acquisition_checkpoint.py` (from `Phase 1/`,
with the venv activated) to re-verify all datasets open correctly with
consistent WGS84 projection before proceeding to Week 2 (Sentinel-1
acquisition and flood inventory generation).
