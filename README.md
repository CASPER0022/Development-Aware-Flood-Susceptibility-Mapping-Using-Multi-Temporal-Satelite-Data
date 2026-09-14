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
│   │   └── environment_setup.md            # Verified GEE & environment setup specification
│   │
│   ├── notebooks/                          # Jupyter Notebooks for exploratory data analysis
│   │
│   ├── scripts/                            # Production Pipeline Scripts
│   │   ├── 01_define_aoi_and_validate_flood.py # Master Phase 1 entry-point pipeline script
│   │   ├── test_urban_double_bounce.py     # Dual-criterion SAR detector engine
│   │   ├── fetch_esa_worldcover.py         # ESA WorldCover 10m LULC & settlement mask loader
│   │   ├── local_ccd_flood_detection.py    # Change detection & diagnostic evaluation module
│   │   └── validate_against_ndem.py        # ISRO/NRSC NDEM ground truth validation engine
│   │
│   ├── data/                               # Structured Data Assets
│   │   └── aoi/                            # Master Boundary Datasets & Precision Metadata
│   │       ├── periyar_study_area.geojson  # Master AOI GeoJSON (1,696.64 km²)
│   │       ├── periyar_study_area.shp      # Master AOI Shapefile (EPSG:4326 / EPSG:32643)
│   │       └── aoi_summary_metadata.json   # Precision JSON metadata report
│   │
│   └── outputs/                            # Output Deliverables, Maps & Trained Models
│       ├── maps/                           # Publication PNG maps & interactive HTML maps
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
- **QGIS Setup**: GIS viewer setup for spatial visual QA.

---

## 🛠️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 1 master pipeline script
python "Phase 1/scripts/01_define_aoi_and_validate_flood.py"
```

---

## 📜 License
Academic Research / BTP Project.
