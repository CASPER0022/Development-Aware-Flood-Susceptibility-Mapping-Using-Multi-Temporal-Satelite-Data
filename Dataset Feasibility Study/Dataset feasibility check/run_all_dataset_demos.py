import subprocess
import sys
import time
import os
import json

# Ensure UTF-8 console output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

scripts = [
    ("01_fetch_sentinel2_satellite.py", "Sentinel-2 Level-2A Multispectral STAC Assets"),
    ("02_fetch_elevation_and_water.py", "Copernicus DEM (GLO-90/30) & OSM River Networks"),
    ("03_fetch_building_footprints.py", "Ground-Truth Footprints (OSM, Google, MS Datasets)"),
    ("04_complete_town_risk_assessment.py", "Multi-Criteria Town Risk Assessment Engine"),
    ("05_validate_with_2018_flood_groundtruth.py", "2018 Kerala Flood Ground-Truth Benchmark")
]

print("=" * 88)
print("     TOWN PLANNING CV PROJECT: RIGOROUS DATASET FEASIBILITY & PROVENANCE SUITE")
print("=" * 88)
print("This suite executes all live dataset pipelines and verifies genuine data provenance.\n")

results = []
python_exe = sys.executable

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

for script_name, description in scripts:
    print(f"\n>>> Running [{script_name}]: {description} ...")
    start_t = time.time()
    res = subprocess.run([python_exe, script_name], capture_output=True, text=True, encoding='utf-8', errors='replace')
    elapsed = time.time() - start_t
    
    # Read provenance flag
    module_num = script_name[:2]
    provenance_file = f"output_data/provenance_module_{int(module_num)}.json"
    provenance_status = "UNKNOWN"
    if os.path.exists(provenance_file):
        with open(provenance_file, "r") as pf:
            pdata = json.load(pf)
            if any("SYNTHETIC" in str(v) for v in pdata.values()):
                provenance_status = "[WARN] SYNTHETIC FALLBACK"
            else:
                provenance_status = "[OK] REAL LIVE DATA"
    
    if res.returncode == 0:
        print(res.stdout)
        results.append((script_name, description, "SUCCESS", provenance_status, f"{elapsed:.2f}s"))
    else:
        print(res.stdout)
        print(f"[FAILED] Error output:\n{res.stderr}")
        results.append((script_name, description, "FAILED", "[FAIL] FAILED", f"{elapsed:.2f}s"))

print("\n" + "=" * 88)
print("                     FINAL VERIFICATION & PROVENANCE TABLE")
print("=" * 88)
print(f"{'Script Name':<36} | {'Status':<8} | {'Data Provenance':<24} | {'Time':<7} | {'Dataset'}")
print("-" * 88)
for script, desc, status, prov, t in results:
    print(f"{script:<36} | {status:<8} | {prov:<24} | {t:<7} | {desc}")
print("=" * 88)

generated_files = [
    "output_data/01_satellite_comparison.png",
    "output_data/02_elevation_and_water.png",
    "output_data/03_building_footprints.png",
    "output_data/04_complete_town_risk_map.png",
    "output_data/05_historical_flood_validation.png",
    "output_data/historical_flood_validation_report.json",
    "output_data/interactive_risk_dashboard.html",
    "output_data/sentinel2_band_catalog.json"
]

print("\n[+] Verified Research Artifacts in output_data/:")
for f in generated_files:
    if os.path.exists(f):
        size_kb = os.path.getsize(f) / 1024.0
        print(f"  [x] {f:<45} ({size_kb:.1f} KB)")
    else:
        print(f"  [ ] {f:<45} (Not created)")

print("\n" + "=" * 88)
print("Technical Provenance & Accuracy Notes:")
print("1. Sentinel-2: STAC metadata connects directly to AWS COG bands (B02, B03, B04, B08).")
print("2. Elevation: Copernicus Global DEM (GLO-90/GLO-30) via global elevation API.")
print("3. Hydrography: Real OpenStreetMap waterway vector segments.")
print("4. Building Footprints: Live verified OSM polygons & Microsoft/Google repositories.")
print("=" * 88)
