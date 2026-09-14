import os
import sys
import ee

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 80)
print("GOOGLE EARTH ENGINE CONNECTION & ASSET TEST")
print("=" * 80)

project_id = sys.argv[1] if len(sys.argv) > 1 else None

try:
    if project_id:
        print(f"[+] Initializing Earth Engine with Cloud Project: '{project_id}'...")
        ee.Initialize(project=project_id)
    else:
        print("[+] Initializing Earth Engine with default project...")
        ee.Initialize()
    
    print("[SUCCESS] Earth Engine initialized successfully!\n")
    
    # Query USGS SRTM DEM
    print("[+] Test 1: Querying USGS SRTM 30m Global DEM...")
    srtm = ee.Image("USGS/SRTMGL1_003")
    srtm_info = srtm.getInfo()
    print(f"    • Asset ID : {srtm_info.get('id')}")
    print(f"    • Type     : {srtm_info.get('type')}")
    print(f"    • Bands    : {[b['id'] for b in srtm_info.get('bands', [])]}")
    
    # Query ESA WorldCover 2021
    print("\n[+] Test 2: Querying ESA WorldCover 10m LULC Asset...")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    wc_info = wc.getInfo()
    print(f"    • Asset ID : {wc_info.get('id')}")
    print(f"    • Bands    : {[b['id'] for b in wc_info.get('bands', [])]}")
    
    print("\n" + "=" * 80)
    print("ALL EARTH ENGINE ASSETS TESTED & WORKING 100%!")
    print("=" * 80)

except Exception as e:
    print(f"[ERROR] Earth Engine Initialization Failed: {e}")
    print("\n[HINT] Earth Engine requires specifying your Google Cloud Project ID.")
    print("Please set your project ID by running:")
    print("  .\\venv\\Scripts\\python.exe -m ee.cli.eecli set_project YOUR_CLOUD_PROJECT_ID")
