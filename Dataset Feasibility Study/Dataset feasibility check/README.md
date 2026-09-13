# Dataset Feasibility & Technical Provenance Check

This directory contains the verification pipelines proving the technical availability, data provenance, and spatial integration feasibility of all multi-source datasets for the **CV-Based Town Planning Risk System**.

---

## 🛰️ Dataset Sources & Provenance

| Data Layer | Actual Source / Protocol | Resolution / Format | Ground Truth Access |
| :--- | :--- | :--- | :--- |
| **Multispectral Satellite Imagery** | Sentinel-2 Level-2A (AWS Earth Search STAC) | 10m Ground Resolution / Cloud-Optimized GeoTIFFs (COGs) | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) |
| **Elevation & Topography** | Copernicus Global DEM (GLO-90/GLO-30) | 30m/90m Global Elevation Grid | Topographic Sinks & Elevation percentiles |
| **Waterways & River Drainage** | OpenStreetMap Hydrography Network | Vector Linestrings & Polygons | Periyar River Channels & Canal Buffers |
| **Building Footprint Labels** | OSM Building Vectors, Microsoft Global ML Footprints, Google Open Buildings v3 | Vector GeoJSON Polygons | Ground-truth masks for fine-tuning segmentation models |

---

## 📁 Directory Structure

```
Dataset feasibility check/
│
├── 01_fetch_sentinel2_satellite.py      # Sentinel-2 Level-2A STAC Multispectral Band Asset fetcher
├── 02_fetch_elevation_and_water.py      # Copernicus DEM & OpenStreetMap River drainage analyzer
├── 03_fetch_building_footprints.py      # Building footprint ground-truth verification (OSM, Google, MS)
├── 04_complete_town_risk_assessment.py  # End-to-end multi-criteria risk engine & dashboard
├── run_all_dataset_demos.py             # Master verification suite with explicit Data Provenance reporting
├── README.md                            # Technical documentation
│
└── output_data/                         # Generated research artifacts
    ├── 01_satellite_comparison.png      # 2019 vs 2024 Level-2A Sentinel-2 comparison map
    ├── 02_elevation_and_water.png       # Copernicus DEM elevation contours & river buffers
    ├── 03_building_footprints.png       # Building footprints vector map
    ├── 04_complete_town_risk_map.png    # 4-panel synthesized town planning risk map
    ├── interactive_risk_dashboard.html  # Interactive Leaflet web map with risk popups
    ├── sentinel2_band_catalog.json      # Direct AWS COG URLs for Red, Green, Blue, NIR bands
    ├── elevation_dem.npy                # Raw Copernicus DEM elevation matrix
    ├── water_distance_matrix.npy        # Distance-to-water matrix
    ├── osm_water_bodies.json            # Fetched OSM river channels
    └── osm_buildings_sample.json        # Fetched OSM building polygons
```

---

## 🚀 How to Run

From the root project directory:

```powershell
# Run the complete verification suite with provenance reporting:
.\venv\Scripts\python.exe "Dataset feasibility check\run_all_dataset_demos.py"

# Or run individual modules:
cd "Dataset feasibility check"
..\venv\Scripts\python.exe 01_fetch_sentinel2_satellite.py
..\venv\Scripts\python.exe 02_fetch_elevation_and_water.py
..\venv\Scripts\python.exe 03_fetch_building_footprints.py
..\venv\Scripts\python.exe 04_complete_town_risk_assessment.py
```
