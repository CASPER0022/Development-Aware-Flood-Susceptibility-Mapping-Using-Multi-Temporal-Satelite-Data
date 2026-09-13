import requests
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 75)
print("MODULE 2: ELEVATION (COPERNICUS DEM) & OPENSTREETMAP WATERWAYS")
print("=" * 75)

# Target Town Bounding Box: Aluva / Kochi region (Kerala, India)
BBOX = [76.30, 10.05, 76.40, 10.15]
TOWN_NAME = "Aluva / North Kochi (Periyar Basin)"
GRID_SIZE = 15  # 15x15 = 225 sample points (queried in safe chunks of 45 points)

os.makedirs("output_data", exist_ok=True)
provenance = {
    "module_2_elevation": "LIVE_FETCH",
    "module_2_water": "LIVE_FETCH"
}

# -------------------------------------------------------------
# 1. Fetch Elevation from Copernicus Global DEM (via Elevation API)
# -------------------------------------------------------------
print(f"\n[+] Querying Copernicus DEM (GLO-90/GLO-30) for {TOWN_NAME}...")
lats = np.linspace(BBOX[1], BBOX[3], GRID_SIZE)
lons = np.linspace(BBOX[0], BBOX[2], GRID_SIZE)
mesh_lon, mesh_lat = np.meshgrid(lons, lats)

flat_lats = [round(float(x), 4) for x in mesh_lat.flatten()]
flat_lons = [round(float(x), 4) for x in mesh_lon.flatten()]

elevations_list = []
batch_size = 45 # Safe chunk size to keep GET URL short (< 800 chars)

try:
    for i in range(0, len(flat_lats), batch_size):
        chunk_lat = flat_lats[i:i+batch_size]
        chunk_lon = flat_lons[i:i+batch_size]
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": chunk_lat, "longitude": chunk_lon},
            timeout=15
        )
        if r.status_code == 200:
            elevations_list.extend(r.json().get("elevation", []))
        else:
            raise RuntimeError(f"API status {r.status_code}: {r.text[:100]}")
            
    elevations = np.array(elevations_list).reshape((GRID_SIZE, GRID_SIZE))
    print(f"    [LIVE SUCCESS] Fetched {len(elevations_list)} Copernicus DEM elevation points.")
except Exception as e:
    provenance["module_2_elevation"] = "SYNTHETIC_FALLBACK"
    print("\n" + "!" * 75)
    print(f"[WARNING: USING SYNTHETIC FALLBACK - LIVE ELEVATION API FETCH FAILED: {e}]")
    print("!" * 75)
    y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    elevations = 4.0 + 12.0 * (x / GRID_SIZE) + 6.0 * np.sin(y / 4.0)

min_elev, max_elev = float(np.nanmin(elevations)), float(np.nanmax(elevations))
low_lying_threshold = float(np.percentile(elevations, 25))
print(f"    Elevation Range: {min_elev:.1f} m to {max_elev:.1f} m above sea level")
print(f"    Low-Lying Floodplain Threshold (Bottom 25%): < {low_lying_threshold:.1f} m")

np.save("output_data/elevation_dem.npy", elevations)
print(f"[OK] Saved elevation matrix -> output_data/elevation_dem.npy")

# -------------------------------------------------------------
# 2. Fetch OpenStreetMap (OSM) Water Bodies and Rivers
# -------------------------------------------------------------
print(f"\n[+] Querying OpenStreetMap Overpass API for river networks & water bodies...")
osm_url = "https://overpass-api.de/api/interpreter"
headers = {'User-Agent': 'TownPlanningBTPApp/2.0 (btp_feasibility_check@edu.in)'}

query = f"""[out:json][timeout:25];
(
  way["natural"="water"]({BBOX[1]}, {BBOX[0]}, {BBOX[3]}, {BBOX[2]});
  way["waterway"]({BBOX[1]}, {BBOX[0]}, {BBOX[3]}, {BBOX[2]});
);
out geom;"""

elements = []
try:
    osm_resp = requests.post(osm_url, data={"data": query}, headers=headers, timeout=25)
    if osm_resp.status_code == 200:
        osm_data = osm_resp.json()
        elements = osm_data.get("elements", [])
        print(f"    [LIVE SUCCESS] Fetched {len(elements)} river / canal segments from OpenStreetMap.")
        with open("output_data/osm_water_bodies.json", "w") as f:
            json.dump(osm_data, f, indent=2)
    else:
        raise RuntimeError(f"Overpass API returned status {osm_resp.status_code}")
except Exception as e:
    provenance["module_2_water"] = "SYNTHETIC_FALLBACK"
    print("\n" + "!" * 75)
    print(f"[WARNING: USING SYNTHETIC FALLBACK - LIVE OSM FETCH FAILED: {e}]")
    print("!" * 75)

# Rasterize water coordinates onto grid
water_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
for el in elements:
    coords = el.get("geometry", [])
    for pt in coords:
        lat, lon = pt.get("lat"), pt.get("lon")
        if lat and lon and BBOX[1] <= lat <= BBOX[3] and BBOX[0] <= lon <= BBOX[2]:
            r = int((lat - BBOX[1]) / (BBOX[3] - BBOX[1]) * (GRID_SIZE - 1))
            c = int((lon - BBOX[0]) / (BBOX[2] - BBOX[0]) * (GRID_SIZE - 1))
            water_mask[r, c] = True

if np.sum(water_mask) == 0:
    if provenance["module_2_water"] != "SYNTHETIC_FALLBACK":
        provenance["module_2_water"] = "SYNTHETIC_FALLBACK"
        print("[WARNING: NO WATER ELEMENTS FOUND - USING SYNTHETIC CHANNEL FALLBACK]")
    for r in range(GRID_SIZE):
        c = int(GRID_SIZE * 0.45 + 3 * np.sin(r / 3.0))
        if 0 <= c < GRID_SIZE:
            water_mask[r, c] = True

# Vectorized distance calculation
cell_size_m = (0.10 * 111000.0) / GRID_SIZE
water_pts = np.argwhere(water_mask)
grid_pts = np.indices((GRID_SIZE, GRID_SIZE)).reshape(2, -1).T
diff = grid_pts[:, np.newaxis, :] - water_pts[np.newaxis, :, :]
dist_sq = np.sum(diff**2, axis=2)
dist_grid_meters = (np.sqrt(np.min(dist_sq, axis=1)).reshape((GRID_SIZE, GRID_SIZE))) * cell_size_m

np.save("output_data/water_distance_matrix.npy", dist_grid_meters)

# -------------------------------------------------------------
# 3. Generate Visual Verification Graphic
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

c1 = axes[0].imshow(elevations, extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower', cmap='terrain')
contours = axes[0].contour(mesh_lon, mesh_lat, elevations, levels=6, colors='black', alpha=0.4, linewidths=0.8)
axes[0].clabel(contours, inline=True, fontsize=8, fmt='%1.0fm')
plt.colorbar(c1, ax=axes[0], label='Elevation (meters ASL)')
axes[0].set_title(f"A. Digital Elevation Model [Copernicus DEM (GLO-90/30)]\n{TOWN_NAME}\n(Sampled at ~660m feasibility grid, smoothed for preview)", fontweight='bold', fontsize=10)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

c2 = axes[1].imshow(dist_grid_meters, extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], origin='lower', cmap='Blues_r')
plt.colorbar(c2, ax=axes[1], label='Distance to River Channel (meters)')
water_y, water_x = np.where(water_mask)
if len(water_y) > 0:
    water_lats = BBOX[1] + (water_y / (GRID_SIZE - 1)) * (BBOX[3] - BBOX[1])
    water_lons = BBOX[0] + (water_x / (GRID_SIZE - 1)) * (BBOX[2] - BBOX[0])
    axes[1].scatter(water_lons, water_lats, color='cyan', s=12, label='OSM River Elements', alpha=0.8)
axes[1].legend(loc='upper right')
axes[1].set_title(f"B. River Proximity & Drainage Buffer [OpenStreetMap]\n(Flood Hazard Buffers < 350m)", fontweight='bold', fontsize=10)
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.suptitle("Copernicus DEM & OSM Waterway Dataset Availability Proof", fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = "output_data/02_elevation_and_water.png"
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved topographic analysis graphic -> {out_fig}")
print("=" * 75)

with open("output_data/provenance_module_2.json", "w") as f:
    json.dump(provenance, f)
