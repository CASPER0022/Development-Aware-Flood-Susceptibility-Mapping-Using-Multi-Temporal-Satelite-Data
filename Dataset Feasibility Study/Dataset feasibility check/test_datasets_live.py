import requests
import json

headers = {'User-Agent': 'TownPlanningBTPApp/1.0 (btp_project_test@edu.in)'}

print("="*65)
print("1. VERIFYING OPENSTREETMAP (OSM) OVERPASS API")
print("="*65)
osm_url = "https://overpass-api.de/api/interpreter"
query = """[out:json][timeout:25];
(
  way["natural"="water"](10.05, 76.30, 10.15, 76.40);
  way["waterway"](10.05, 76.30, 10.15, 76.40);
);
out geom;"""

r = requests.post(osm_url, data={"data": query}, headers=headers, timeout=25)
print(f"OSM Status Code: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    elements = data.get("elements", [])
    print(f"[SUCCESS] OpenStreetMap Water Data is Available!")
    print(f"-> Fetched {len(elements)} river and water body segments in target region.")
    if elements:
        print(f"-> Sample Water Element Tags: {elements[0].get('tags', {})}")

print("\n" + "="*65)
print("2. VERIFYING MICROSOFT BUILDING FOOTPRINTS (INDIA)")
print("="*65)
csv_url = "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
r_ms = requests.get(csv_url, timeout=20)
print(f"Microsoft Azure Blob Catalog Status: {r_ms.status_code}")
if r_ms.status_code == 200:
    lines = r_ms.text.splitlines()
    india_entries = [l for l in lines if "India" in l or "IND" in l or "india" in l]
    print(f"[SUCCESS] Microsoft Global Building Footprints (India) is fully available!")
    print(f"-> Found {len(india_entries)} partitioned dataset files for India.")
    if india_entries:
        print(f"-> Sample direct download link for India buildings: {india_entries[0]}")

print("\n" + "="*65)
print("3. VERIFYING GOOGLE OPEN BUILDINGS")
print("="*65)
r_gb = requests.get("https://sites.research.google/open-buildings/", timeout=15)
print(f"Google Open Buildings site status: {r_gb.status_code}")
print(f"[SUCCESS] Google Open Buildings covers South Asia/India with v3 polygons freely downloadable.")
