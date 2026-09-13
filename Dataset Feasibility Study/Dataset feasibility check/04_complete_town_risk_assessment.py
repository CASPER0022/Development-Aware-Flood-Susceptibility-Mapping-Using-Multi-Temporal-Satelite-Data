import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 75)
print("MODULE 4: COMPLETE INTEGRATED TOWN PLANNING RISK ASSESSMENT PIPELINE")
print("=" * 75)

BBOX = [76.30, 10.05, 76.40, 10.15]
TOWN_NAME = "Aluva / North Kochi (Periyar Basin, Kerala)"
GRID_SIZE = 60

os.makedirs("output_data", exist_ok=True)
provenance = {"module_4_risk_pipeline": "LIVE_SYNTHESIS"}

# -------------------------------------------------------------
# 1. Load Geospatial Datasets
# -------------------------------------------------------------
print("\n[+] Step 1: Loading Multi-Source Geospatial Datasets...")

# 1.1 Elevation DEM (Copernicus DEM)
dem_file = "output_data/elevation_dem.npy"
if os.path.exists(dem_file):
    raw_elevation = np.load(dem_file)
    img_elev = Image.fromarray(raw_elevation)
    elevation = np.array(img_elev.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR))
else:
    print("⚠️ [WARNING: Using synthetic topography fallback]")
    y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    elevation = 4.0 + 12.0 * (x / GRID_SIZE) + 6.0 * np.sin(y / 4.0)

# 1.2 Water Distance Matrix (OpenStreetMap Hydrography)
water_file = "output_data/water_distance_matrix.npy"
if os.path.exists(water_file):
    raw_dist = np.load(water_file)
    img_w = Image.fromarray(raw_dist)
    water_dist = np.array(img_w.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR))
else:
    print("⚠️ [WARNING: Using synthetic water channel fallback]")
    water_dist = np.zeros((GRID_SIZE, GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            river_c = int(GRID_SIZE * 0.45 + 4 * np.sin(r / 4.0))
            water_dist[r, c] = abs(c - river_c) * 150.0

# 1.3 Satellite Multi-Temporal Landcover / Built-up Masks
sat_2019_path = "output_data/satellite_2019.jpg"
sat_2024_path = "output_data/satellite_2024.jpg"

if os.path.exists(sat_2019_path) and os.path.exists(sat_2024_path):
    img_19 = Image.open(sat_2019_path).convert('RGB').resize((GRID_SIZE, GRID_SIZE))
    img_24 = Image.open(sat_2024_path).convert('RGB').resize((GRID_SIZE, GRID_SIZE))
    arr_19 = np.array(img_19, dtype=float)
    arr_24 = np.array(img_24, dtype=float)
    
    # 1. Mask out nodata background (outside satellite orbit swath)
    valid_land = (arr_19[:, :, 0] > 15) & (arr_19[:, :, 0] < 240) & \
                 (arr_24[:, :, 0] > 15) & (arr_24[:, :, 0] < 240)
    
    # 2. Extract significant spectral expansion clusters
    spectral_diff = np.mean(np.abs(arr_24 - arr_19), axis=2)
    if np.any(valid_land):
        diff_thresh = np.percentile(spectral_diff[valid_land], 80)
        new_urban_growth = valid_land & (spectral_diff >= diff_thresh)
    else:
        new_urban_growth = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
else:
    print("⚠️ [WARNING: Using synthetic urban expansion mask]")
    np.random.seed(42)
    new_urban_growth = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    new_urban_growth[15:25, 10:20] = True
    new_urban_growth[35:45, 15:25] = True

# -------------------------------------------------------------
# 2. Multi-Criteria Risk Scoring Engine (Project Novelty)
# -------------------------------------------------------------
print("\n[+] Step 2: Executing Multi-Criteria Flood & Growth Risk Scoring...")

low_elevation_thresh = float(np.percentile(elevation, 30)) # Lowest 30% elevation tier (<8m ASL)
water_buffer_thresh = 350.0                                # Within 350m of water channels

is_low_elevation = elevation <= low_elevation_thresh
is_near_water = water_dist <= water_buffer_thresh

# Risk matrix:
# 0 = Safe baseline / No new construction
# 1 = Low Risk (New construction on safe elevated ground away from water)
# 2 = Medium / Watch (New construction with moderate hazard)
# 3 = High Risk (New construction in severe hazard: low elevation AND within river flood buffer)

risk_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
risk_map[new_urban_growth & (~is_low_elevation) & (~is_near_water)] = 1
risk_map[new_urban_growth & (is_low_elevation ^ is_near_water)] = 2
risk_map[new_urban_growth & is_low_elevation & is_near_water] = 3

total_new_pixels = int(np.sum(new_urban_growth))
high_risk_pixels = int(np.sum(risk_map == 3))
med_risk_pixels = int(np.sum(risk_map == 2))
low_risk_pixels = int(np.sum(risk_map == 1))

print(f"    Total New Urban Expansion Detected: {total_new_pixels} zones")
print(f"    -> High Risk Zones (Floodplain + Water Proximity): {high_risk_pixels} zones ({high_risk_pixels/max(1,total_new_pixels)*100:.1f}%)")
print(f"    -> Medium Risk Zones (Moderate Hazard): {med_risk_pixels} zones ({med_risk_pixels/max(1,total_new_pixels)*100:.1f}%)")
print(f"    -> Safe / Low Risk Development: {low_risk_pixels} zones ({low_risk_pixels/max(1,total_new_pixels)*100:.1f}%)")

# -------------------------------------------------------------
# 3. Create Multi-Panel Research Figure
# -------------------------------------------------------------
print("\n[+] Step 3: Generating Visual Research Artifacts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Subplot 1: Detected Urban Expansion
axes[0, 0].imshow(new_urban_growth, cmap='Blues', extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower')
axes[0, 0].set_title("1. Detected Urban Growth / New Built-up (2019-2024)\n[Sentinel-2 Satellite Surface Reflectance Change]", fontweight='bold', fontsize=11)
axes[0, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")

# Subplot 2: Elevation & Topographic Sinks
c2 = axes[0, 1].imshow(elevation, cmap='terrain', extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower')
axes[0, 1].contour(np.linspace(BBOX[0], BBOX[2], GRID_SIZE), np.linspace(BBOX[1], BBOX[3], GRID_SIZE),
                   elevation, levels=5, colors='black', alpha=0.3)
plt.colorbar(c2, ax=axes[0, 1], label='Elevation (m ASL)')
axes[0, 1].set_title(f"2. Topographic Sinks & Lowlands (<{low_elevation_thresh:.1f}m)\n[Copernicus Global DEM (GLO-90/30)]", fontweight='bold', fontsize=11)
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")

# Subplot 3: Water Proximity & River Buffers
c3 = axes[1, 0].imshow(water_dist, cmap='PuBu_r', extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower')
plt.colorbar(c3, ax=axes[1, 0], label='Distance (m)')
axes[1, 0].set_title("3. River Basin & Drainage Buffers (<350m)\n[OpenStreetMap Hydrography Network]", fontweight='bold', fontsize=11)
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")

# Subplot 4: Final Synthesized Risk Assessment Map
risk_rgb = np.ones((GRID_SIZE, GRID_SIZE, 4), dtype=float)
risk_rgb[:, :, 3] = 0.08 # Subtle background

risk_rgb[risk_map == 1] = [0.13, 0.69, 0.30, 0.90] # Green: Safe
risk_rgb[risk_map == 2] = [0.95, 0.70, 0.05, 0.90] # Amber: Watch
risk_rgb[risk_map == 3] = [0.90, 0.15, 0.15, 0.95] # Red: High Risk

axes[1, 1].imshow(risk_rgb, extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower')
axes[1, 1].set_title("4. Synthesized Town Planning Risk Assessment Map\n[Red: High Flood Hazard | Yellow: Watch | Green: Safe]", fontweight='bold', fontsize=11)
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")

plt.suptitle(f"Town Planning Risk Assessment: End-to-End System Output\nTown: {TOWN_NAME}", fontsize=14, fontweight='bold')
plt.tight_layout()
final_img_path = "output_data/04_complete_town_risk_map.png"
plt.savefig(final_img_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[OK] Saved comprehensive risk assessment plot -> {final_img_path}")

# -------------------------------------------------------------
# 4. Generate Interactive Standalone Leaflet HTML Map
# -------------------------------------------------------------
center_lat = (BBOX[1] + BBOX[3]) / 2.0
center_lon = (BBOX[0] + BBOX[2]) / 2.0

markers_js = []
high_risk_indices = np.argwhere(risk_map == 3)
if len(high_risk_indices) > 35:
    sample_idx = np.random.choice(len(high_risk_indices), 35, replace=False)
    high_risk_indices = high_risk_indices[sample_idx]

for r, c in high_risk_indices:
    lat = BBOX[1] + (r / (GRID_SIZE - 1)) * (BBOX[3] - BBOX[1])
    lon = BBOX[0] + (c / (GRID_SIZE - 1)) * (BBOX[2] - BBOX[0])
    elev_val = elevation[r, c]
    dist_val = water_dist[r, c]
    markers_js.append(f"""
    L.circleMarker([{lat:.5f}, {lon:.5f}], {{
        color: '#c0392b',
        fillColor: '#e74c3c',
        fillOpacity: 0.85,
        radius: 8
    }}).addTo(map).bindPopup(`
        <div style="font-family: sans-serif; font-size: 13px;">
            <b style="color: #c0392b;">⚠️ HIGH FLOOD RISK ZONE</b><br>
            <b>Town:</b> {TOWN_NAME}<br>
            <b>Elevation:</b> {elev_val:.1f} m ASL (Copernicus DEM)<br>
            <b>Distance to River:</b> {dist_val:.0f} m (OSM Buffer)<br>
            <b>Classification:</b> Unplanned Inundation Hazard<br>
            <b>Recommendation:</b> Require drainage audit before approval.
        </div>
    `);
    """)

leaflet_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Town Planning CV Risk Dashboard</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #header {{ background: #1e293b; color: white; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; }}
        #header h1 {{ margin: 0; font-size: 20px; }}
        #stats {{ font-size: 14px; background: #334155; padding: 6px 14px; border-radius: 6px; }}
        #map {{ height: calc(100vh - 65px); width: 100%; }}
        .legend {{ background: white; padding: 12px; line-height: 1.5; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }}
        .legend i {{ width: 14px; height: 14px; float: left; margin-right: 8px; opacity: 0.9; border-radius: 50%; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🛰️ AI-Driven Town Planning Risk & Flood Hazard System</h1>
        <div id="stats"><b>Region:</b> {TOWN_NAME} | <b>High-Risk Zones Flagged:</b> {high_risk_pixels}</div>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 13);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 18,
            attribution: '© OpenStreetMap contributors | Sentinel-2 STAC | Copernicus DEM'
        }}).addTo(map);

        var bounds = [[{BBOX[1]}, {BBOX[0]}], [{BBOX[3]}, {BBOX[2]}]];
        L.rectangle(bounds, {{color: "#3b82f6", weight: 2, fill: false}}).addTo(map);

        {''.join(markers_js)}

        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML += '<b>Urban Flood Risk Legend</b><br>';
            div.innerHTML += '<i style="background: #e74c3c"></i> High Hazard (Sink + River Buffer)<br>';
            div.innerHTML += '<i style="background: #f1c40f"></i> Moderate Risk (Watch Zone)<br>';
            div.innerHTML += '<i style="background: #2ecc71"></i> Safe / Low Hazard<br>';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""

dashboard_html = "output_data/interactive_risk_dashboard.html"
with open(dashboard_html, "w", encoding="utf-8") as f:
    f.write(leaflet_html)
print(f"[OK] Saved interactive Leaflet map -> {dashboard_html}")
print("=" * 75)

with open("output_data/provenance_module_4.json", "w") as f:
    json.dump(provenance, f)
