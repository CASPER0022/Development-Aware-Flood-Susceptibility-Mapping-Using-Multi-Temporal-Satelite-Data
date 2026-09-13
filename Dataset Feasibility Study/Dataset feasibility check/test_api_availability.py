import requests
import json
import os

print("="*60)
print("1. TESTING OPENSTREETMAP (OSM) OVERPASS API")
print("="*60)
# Test area: Aluva / Kochi, Kerala (known for 2018 floods and rapid urban expansion)
# Bounding box: [south: 10.05, west: 76.30, north: 10.15, east: 76.40]
osm_url = "https://overpass-api.de/api/interpreter"
query = """[out:json][timeout:25];
(
  way["natural"="water"](10.05, 76.30, 10.15, 76.40);
  way["waterway"](10.05, 76.30, 10.15, 76.40);
  relation["natural"="water"](10.05, 76.30, 10.15, 76.40);
);
out geom;"""

try:
    r = requests.post(osm_url, data={"data": query}, timeout=20)
    if r.status_code == 200:
        data = r.json()
        elements = data.get("elements", [])
        print(f"[SUCCESS] OpenStreetMap Water Data is Available!")
        print(f"-> Found {len(elements)} water bodies/river segments in target boundary.")
    else:
        print(f"[WARNING] OSM returned status {r.status_code}")
except Exception as e:
    print(f"[ERROR] OSM API error: {e}")

print("\n" + "="*60)
print("2. TESTING SENTINEL-2 SATELLITE IMAGERY (STAC API - AWS Earth Search)")
print("="*60)
stac_url = "https://earth-search.aws.element84.com/v1/search"
stac_query = {
    "collections": ["sentinel-2-l2a"],
    "bbox": [76.30, 10.05, 76.40, 10.15],
    "datetime": "2024-01-01T00:00:00Z/2024-03-31T23:59:59Z",
    "query": {
        "eo:cloud_cover": {"lt": 5}
    },
    "limit": 3
}

try:
    r = requests.post(stac_url, json=stac_query, timeout=20)
    if r.status_code == 200:
        stac_data = r.json()
        features = stac_data.get("features", [])
        print(f"[SUCCESS] Sentinel-2 Satellite Imagery is 100% Available!")
        print(f"-> Found {len(features)} ultra-low cloud scenes for 2024.")
        for f in features:
            print(f"   Scene ID: {f['id']}, Cloud Cover: {f['properties'].get('eo:cloud_cover', 'N/A')}%")
            if "rendered_preview" in f["assets"]:
                print(f"   Preview URL: {f['assets']['rendered_preview']['href']}")
            elif "thumbnail" in f["assets"]:
                print(f"   Thumbnail URL: {f['assets']['thumbnail']['href']}")
    else:
        print(f"[WARNING] STAC returned status {r.status_code}")
except Exception as e:
    print(f"[ERROR] STAC API error: {e}")

print("\n" + "="*60)
print("3. TESTING ELEVATION (DEM) DATA API")
print("="*60)
# Test Open-Meteo Elevation API (free, global, high performance DEM)
elev_url = "https://api.open-meteo.com/v1/elevation"
params = {
    "latitude": [10.05, 10.08, 10.10, 10.12, 10.15],
    "longitude": [76.30, 76.32, 76.35, 76.38, 76.40]
}
try:
    r = requests.get(elev_url, params=params, timeout=15)
    if r.status_code == 200:
        res = r.json()
        print(f"[SUCCESS] Elevation (DEM) Data is 100% Available!")
        elevations = res.get("elevation", [])
        print(f"-> Sample sampled elevations across the town: {elevations} meters above sea level.")
    else:
        print(f"[WARNING] Elevation returned status {r.status_code}")
except Exception as e:
    print(f"[ERROR] Elevation error: {e}")

print("\n" + "="*60)
print("4. TESTING MICROSOFT BUILDING FOOTPRINTS DATASET")
print("="*60)
# Check access to Microsoft Global Building Footprints repository / Azure links
ms_repo_url = "https://raw.githubusercontent.com/microsoft/GlobalMLBuildingFootprints/master/resources/india.geojson"
try:
    r = requests.get(ms_repo_url, timeout=15)
    print(f"Microsoft Building Footprint Index Status: {r.status_code}")
    if r.status_code == 200:
        print(f"[SUCCESS] Microsoft Building Footprints for India dataset is live and accessible!")
        urls = json.loads(r.text)
        print(f"-> Found {len(urls.get('features', []))} partitioned dataset download regions for India.")
    else:
        print(f"[INFO] Checking direct releases url...")
except Exception as e:
    print(f"MS Footprint check: {e}")
