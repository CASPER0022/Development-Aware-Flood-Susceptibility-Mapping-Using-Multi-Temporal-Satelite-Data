# Development-Aware Flood Susceptibility Mapping Using Multi-Temporal Satellite Data

A development-aware flood susceptibility and risk mapping framework for the **Lower Periyar Basin and Aluva–Kochi Urban Corridor (Kerala, India)** using multi-temporal Sentinel-1 Synthetic Aperture Radar (SAR) data, FABDEM elevation models, ESA WorldCover 10m LULC, and hydro-meteorological drivers.

---

## 📌 Repository Structure
```text
BTP/
├── README.md                                # Root repository documentation
├── .gitignore                              # Root Git ignore rules
├── requirements.txt                         # Python environment dependencies
├── Implementation Plans/                   # Project implementation plans & review roadmaps
├── Review 1/                               # Review 1 presentations & materials
│
├── Phase 1/                                # Phase 1: Base Paper Reproduction & SAR Verification
│   ├── Task1_Summary.html                  # Plain-language HTML report for Phase 1 & NRSC validation
│   ├── Phase1_Implementation_Plan_Oct_Review.md# Phase 1 review roadmap
│   │
│   ├── docs/                               # Environment & Setup Documentation
│   │   ├── environment_setup.md            # Verified GEE & environment setup specification
│   │   ├── dataset_acquisition.md          # Step 3 flood-influencing dataset acquisition log
│   │   ├── sentinel1_acquisition.md        # Step 4 Sentinel-1 SAR acquisition log
│   │   ├── flood_inventory_otsu.md         # Step 5 Otsu flood inventory & visual validation
│   │   └── flood_inventory_crosscheck.md   # Step 6 independent cross-check (NDEM) & GTI analogue
│   │
│   ├── notebooks/                          # Jupyter Notebooks for exploratory data analysis
│   │
│   ├── scripts/                            # Production Pipeline Scripts
│   │   ├── 01_define_aoi_and_validate_flood.py # Master Phase 1 entry-point pipeline script
│   │   ├── 02_fetch_fabdem_dem.py          # FABDEM 30m forest/building-removed elevation downloader
│   │   ├── 03_fetch_chirps_rainfall.py     # CHIRPS daily precipitation time-series (2017-2023 via GEE)
│   │   ├── 04_fetch_hydrosheds_rivers_basins.py# HydroRIVERS stream network & HydroBASINS extractor
│   │   ├── 05_fetch_osm_roads.py           # OSM road network & junction nodes extractor (osmnx)
│   │   ├── 06_fetch_soil_data.py           # OpenLandMap USDA soil texture class loader
│   │   ├── 07_dataset_acquisition_checkpoint.py# Automated Step 3 dataset verification checkpoint
│   │   ├── 08_fetch_sentinel1_sar.py       # Step 4: Sentinel-1 GRD VV acquisition (2018/2019/2021 flood + dry refs, GEE)
│   │   ├── 09_build_flood_inventory_otsu.py# Step 5: Otsu flood inventory (Eq. 1 dB, Lee filter, change vs dry ref) + validation maps
│   │   ├── 10_crosscheck_flood_inventory.py# Step 6: cross-check vs NRSC/ISRO NDEM (EMS/DFO/S2 availability checked) + GTI analogue
│   │   ├── test_urban_double_bounce.py     # Dual-criterion SAR detector engine
│   │   ├── fetch_esa_worldcover.py         # ESA WorldCover 10m LULC & settlement mask loader
│   │   ├── local_ccd_flood_detection.py    # Change detection & diagnostic evaluation module
│   │   └── validate_against_ndem.py        # ISRO/NRSC NDEM ground truth validation engine
│   │
│   ├── data/                               # Structured Data Assets
│   │   └── aoi/                            # Master Boundaries, Precision Metadata & Checkpoint Reports
│   │       ├── periyar_study_area.geojson  # Master AOI GeoJSON (1,696.64 km²)
│   │       ├── periyar_study_area.shp      # Master AOI Shapefile (EPSG:4326 / EPSG:32643)
│   │       ├── aoi_summary_metadata.json   # Precision JSON metadata report
│   │       └── step3_dataset_acquisition_checkpoint.json# Automated 10/10 dataset audit report
│   │
│   └── outputs/                            # Output Deliverables, Maps & Trained Models
│       ├── maps/                           # Publication PNG maps & interactive HTML maps
│       ├── metrics/                        # Evaluation numbers (JSON), e.g. Step 6 cross-check
│       └── models/                         # Trained model artifacts (.pkl, .txt)
│
└── Phase 2/                                # Phase 2: Feature Extraction & Novel Model Training (Upcoming)
```

---

## 🎯 Empirical Progress Summary

### Phase 1 Task 1: AOI & Flood Verification
- **Master AOI Bounds**: `[76.15°E, 9.90°N, 76.55°E, 10.25°N]` (**1,696.64 km²**, UTM Zone 43N).
- **Spatial Resolution**: Native $3,800 \times 2,900$ grid ($\approx 10\text{m}/\text{pixel}$).
- **Built-Up Land (Class 50)**: $190.47\text{ km}^2$ ($11.23\%$ of AOI).
- **Permanent Pre-existing Water**: $290.34\text{ km}^2$ ($17.11\%$).
- **Open Specular Flood**: $28.77\text{ km}^2$ ($1.70\%$).
- **Urban Double-Bounce Surge**: **$30.85\text{ km}^2$ ($1.82\%$ of AOI, $16.20\%$ of built-up land)**.
- **Hotspot Capture**:
  - **Cochin International Airport (COK)**: 2,711 double-bounce flood pixels ($20.09\%$ of precinct built-up land).
  - **Aluva Town Center**: 5,925 double-bounce flood pixels ($21.47\%$ of precinct built-up land).

### Phase 1 Step 2: Environment & GEE Authentication
- **Python Virtual Environment**: Configured Python 3.13 venv (`geopandas`, `rasterio`, `lightgbm`, `xgboost`, `shap`, `scikit-learn`).
- **Google Earth Engine (GEE)**: Authenticated and linked to Cloud Project **`btp-flood`**. Verified live API access for USGS SRTM DEM & ESA WorldCover 2021.

### Phase 1 Step 3: Flood-Influencing Datasets Acquisition (**10/10 PASS RATE**)
- **FABDEM 30m DEM**: Forest/building-removed bare-earth elevation data (`data/raw/fabdem_30m_aoi.tif`).
- **CHIRPS Daily Precipitation**: 2,556 daily records (2017–2023) + monsoon seasonal totals via GEE API (`data/raw/chirps_*`).
- **HydroSHEDS Drainage**: 595 HydroRIVERS segments + 3 HydroBASINS sub-basin polygons (`data/raw/hydrosheds_*`).
- **OSM Transport Network**: 155,985 road edges + 65,828 junction nodes via `osmnx` (`data/raw/osm_roads_aoi.gpkg`).
- **ESA WorldCover 10m LULC**: Native 10m land cover raster (`data/raw/esa_worldcover_2021_aoi.tif`).
- **OpenLandMap Soil Texture**: USDA soil texture class raster (`data/raw/soil_texture_openlandmap_aoi.tif`).

### Phase 1 Step 4: Sentinel-1 SAR Imagery Acquisition (3/3 Flood Events)
- **Source**: GEE `COPERNICUS/S1_GRD` (IW mode, VV polarization, terrain-corrected sigma0 in dB).
- **Flood events acquired**: Aug 2018 (2018-08-21), Aug 2019 (2019-08-10), Oct 2021 (2021-10-16) — each cross-checked as applicable to this AOI via its own CHIRPS rainfall record, not just news reports.
- **Dry references**: one pre-monsoon scene per year, each an exact `relativeOrbitNumber` match (orbit 165, descending) to that year's flood scene — identical imaging geometry, no incidence-angle correction needed.
- **Output**: `data/raw/sentinel1_gee/*.tif` (6 GeoTIFFs) + `data/raw/sentinel1_gee_metadata.json`.

### Phase 1 Step 5: Flood Inventory via Otsu Thresholding
- **Method**: GEE σ⁰ verified already in dB → Lee 5×5 speckle filter on linear intensity → back to dB via base-paper Eq. 1 → per-image Otsu on VV → flood = water(flood date) AND NOT water(same-orbit dry ref), minus slope > 5° and specks < 5 px.
- **Otsu thresholds** (all in the bimodal valley): 2018 −13.66 dB · 2019 −13.83 dB · 2021 −13.44 dB.
- **Flood extent**: 2018-08-21 **23.5 km²** (1.39%) · 2019-08-10 **28.9 km²** (1.70%) · 2021-10-16 8.9 km² (0.52%).
- **Visual check (Aug 2018)**: flood patches concentrate on the northern Periyar floodplain (Puthenvelikkara/Chengamanad/Aluva) and align with NRSC/ISRO NDEM same-day polygons; Kochi city control stays clean. Known misses: airport runway (dark in both scenes), urban double-bounce water; scene is ~5 days post-peak.
- **Usability**: 2018 and 2019 usable as inventories; 2021 is noise-dominated (NDEM < 1 km² in AOI) and should not be used as positive labels.
- **Output**: `data/processed/flood_inventory/*.tif` + `outputs/maps/step5_flood_inventory_*` (overview PNGs, 2018 site-check PNG, 2018 interactive HTML). Details: `Phase 1/docs/flood_inventory_otsu.md`.

### Phase 1 Step 6: Independent Cross-check of the Flood Inventory
- **Sources checked**: Copernicus EMS has no Rapid Mapping activation for Kerala Aug 2018 (it was mapped via International Charter 582 / NDEM). The DFO Global Flood Database has no 2018 India event (latest over the AOI: 2016). Sentinel-2 on 22 Aug 2018 is about 82% cloud over the AOI, with 0% of land comparable, so it was rejected automatically.
- **Reference used**: official NRSC/ISRO **NDEM** flood polygons, both same-day and the event envelope (all passes 17–28 Aug 2018).
- **Headline (GTI analogue)**: **58.4%** of our 2018 Otsu flood pixels are confirmed by NDEM within 40 m (36.7% strict per-pixel). A random map would score 13.3%, so this is 4.4× lift. Base paper GTI: 84.05%.
- **Other events**: 2019 37.1% (8.3× chance). 2021 15.2%, which confirms it is unusable as positive labels.
- **Where it disagrees**: 53% of the unconfirmed 2018 area lies within 60 m of permanent water (river/backwater fringe). Away from that belt, GTI is 74.2%. NDEM's 17–18 Aug maps are near-peak, while our 21 Aug scene shows residual water, so recall against the envelope is low by construction.
- **Output**: `outputs/metrics/step6_flood_inventory_crosscheck.json` + `outputs/maps/step6_crosscheck_*`. Details: `Phase 1/docs/flood_inventory_crosscheck.md`.

---

## 🛠️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 1 master pipeline script
python "Phase 1/scripts/01_define_aoi_and_validate_flood.py"

# Run Step 3 dataset acquisition verification checkpoint
python "Phase 1/scripts/07_dataset_acquisition_checkpoint.py"

# Run Step 4 Sentinel-1 SAR acquisition (2018/2019/2021 flood + dry refs)
python "Phase 1/scripts/08_fetch_sentinel1_sar.py"

# Run Step 5 Otsu flood inventory + visual validation maps
python "Phase 1/scripts/09_build_flood_inventory_otsu.py"

# Run Step 6 independent cross-check (NDEM) + GTI analogue
python "Phase 1/scripts/10_crosscheck_flood_inventory.py"
```

---

## 📜 License
Academic Research / BTP Project.
