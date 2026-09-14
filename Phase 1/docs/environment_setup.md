# Environment Setup (Step 2)

Status as of 2026-09-14. 

## 2.1 Google Earth Engine

`earthengine-api` (v1.7.43) is installed and **fully authenticated & verified working** with Google Cloud Project **`btp-flood`**.

- **Authentication status**: `AUTHENTICATED` (OAuth token saved locally at `~/.config/earthengine/credentials`)
- **Default Cloud Project**: `btp-flood`
- **Verification Script**: `python scripts/test_ee_connection.py btp-flood` (Verified 100% working: USGS SRTM DEM & ESA WorldCover 2021)

```python
import ee
ee.Initialize(project="btp-flood")
print(ee.Image("USGS/SRTMGL1_003").getInfo()["type"])  # Output: "Image"
```

If you don't already have a Cloud project, create one free at
console.cloud.google.com and enable the "Earth Engine API" for it, or let the
`earthengine authenticate` flow create one for you.

## 2.2 Python environment

Using the existing `venv` at the repo root (`E:\Downloads\SEM 7\BTP\venv`, Python
3.13). Installed and verified importable:

| Package | Version |
|---|---|
| geopandas | 1.1.4 |
| rasterio | 1.5.1 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| xgboost | 3.4.1 |
| shap | 0.52.0 |
| matplotlib | 3.11.1 |
| earthengine-api (ee) | 1.7.43 |
| jupyterlab / ipykernel | 4.6.3 / 7.3.0 |

A `requirements.txt` was added at the repo root for reproducibility
(`pip install -r requirements.txt`).

A Jupyter kernel pointing at this venv was registered as **"BTP Phase 1 (venv)"**
(`btp-phase1`) — select it when opening notebooks in `notebooks/`.

To activate the venv directly in a shell:

```bash
source "E:\Downloads\SEM 7\BTP\venv\Scripts\activate"
```

## 2.3 QGIS / ArcGIS

**Not currently installed** on this machine (checked `Program Files\QGIS*` and
`OSGeo4W*` — neither found). This is a desktop GUI install, not something
scriptable from here. Recommended: install QGIS (free, no license needed) from
https://qgis.org/download — the standalone Windows installer is the simplest
option. Use it to open the `.tif`/`.geojson` outputs in `data/` and `outputs/maps/`
for visual QA, per Step 2.3 of the plan (this step is a manual action for you to
complete; nothing else in this phase depends on it being done immediately).

## 2.4 Folder structure

Matches the plan's suggested layout, built on top of what Part 1 already produced
under `Phase 1/`:

```
Phase 1/
  data/
    raw/          (existing — Sentinel-1, ESA WorldCover, flood masks, etc.)
    processed/     (new — for aligned/derived feature rasters)
    aoi/           (existing — AOI shapefile/GeoJSON)
    validation/    (existing — NDEM ground truth)
  notebooks/        (new — exploratory analysis, model training)
  outputs/
    maps/          (existing — AOI/diagnostic maps)
    models/        (new — trained model artifacts, e.g. .pkl/.txt)
  docs/            (new — this file, and future write-ups)
  scripts/         (existing — pipeline scripts from Part 1)
```

`data/raw/`, `data/processed/`, and `data/validation/` are already excluded from
git via the repo's `.gitignore` (large rasters/arrays); `notebooks/`, `docs/`, and
`outputs/models/` are tracked normally.
