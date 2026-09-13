import requests
import json
import os
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

print("=" * 75)
print("MODULE 1: SENTINEL-2 MULTI-YEAR MULTISPECTRAL SATELLITE IMAGERY (STAC)")
print("=" * 75)

# Target Town Bounding Box: Aluva / Kochi region (Kerala, India)
# Coordinates: [min_lon, min_lat, max_lon, max_lat]
BBOX = [76.30, 10.05, 76.40, 10.15]
TOWN_NAME = "Aluva / North Kochi (Periyar Basin)"

os.makedirs("output_data", exist_ok=True)
stac_search_url = "https://earth-search.aws.element84.com/v1/search"

def query_sentinel2_stac(year, bbox):
    """
    Queries Sentinel-2 Level-2A STAC catalog on AWS Earth Search.
    Extracts direct URLs to calibrated COG bands (B02 Blue, B03 Green, B04 Red, B08 NIR, TCI).
    """
    print(f"\n[+] Querying Sentinel-2 Level-2A STAC catalog for {year} over {TOWN_NAME}...")
    query_payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{year}-01-01T00:00:00Z/{year}-04-30T23:59:59Z", # Dry season for cloud-free surface reflectance
        "query": {
            "eo:cloud_cover": {"lt": 5.0}
        },
        "limit": 1,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]
    }
    
    response = requests.post(stac_search_url, json=query_payload, timeout=25)
    if response.status_code != 200:
        raise RuntimeError(f"STAC search failed with status {response.status_code}: {response.text}")
    
    features = response.json().get("features", [])
    if not features:
        # Fallback to broader date window
        query_payload["datetime"] = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
        query_payload["query"]["eo:cloud_cover"] = {"lt": 15.0}
        response = requests.post(stac_search_url, json=query_payload, timeout=25)
        features = response.json().get("features", [])
    
    if not features:
        raise RuntimeError(f"No Sentinel-2 scenes found for {year} in target bounding box.")
    
    feature = features[0]
    scene_id = feature["id"]
    cloud_cover = feature["properties"].get("eo:cloud_cover", 0.0)
    datetime_str = feature["properties"].get("datetime", "")[:10]
    assets = feature["assets"]
    
    print(f"    Scene ID: {scene_id}")
    print(f"    Acquisition Date: {datetime_str} | Cloud Cover: {cloud_cover:.2f}%")
    
    # Extract Real Multispectral Band Asset URLs (Cloud-Optimized GeoTIFFs)
    band_urls = {
        "B02_Blue": assets.get("blue", {}).get("href"),
        "B03_Green": assets.get("green", {}).get("href"),
        "B04_Red": assets.get("red", {}).get("href"),
        "B08_NIR": assets.get("nir", {}).get("href"),
        "TCI_Visual": assets.get("visual", {}).get("href")
    }
    
    print(f"    Found Calibrated Multispectral COG Bands:")
    for band_name, url in band_urls.items():
        print(f"      - {band_name:<12}: {url}")
    
    # Download visual image asset for baseline visualization
    preview_url = assets.get("rendered_preview", {}).get("href") or assets.get("thumbnail", {}).get("href")
    if not preview_url:
        preview_url = band_urls["TCI_Visual"]
        
    img_resp = requests.get(preview_url, timeout=30)
    img_resp.raise_for_status()
    img = Image.open(BytesIO(img_resp.content))
    
    return {
        "image": img,
        "scene_id": scene_id,
        "date": datetime_str,
        "cloud_cover": cloud_cover,
        "band_urls": band_urls,
        "is_live": True
    }

provenance = {"module_1": "LIVE_FETCH"}

try:
    # 1. Fetch Baseline 2019
    data_2019 = query_sentinel2_stac(2019, BBOX)
    img_2019 = data_2019["image"]
    img_2019.save("output_data/satellite_2019.jpg")
    
    # 2. Fetch Recent 2024
    data_2024 = query_sentinel2_stac(2024, BBOX)
    img_2024 = data_2024["image"]
    img_2024.save("output_data/satellite_2024.jpg")
    
    # Save STAC Band URLs catalog to JSON for Phase 6 model training
    band_catalog = {
        "town": TOWN_NAME,
        "bbox": BBOX,
        "baseline_2019": {
            "scene_id": data_2019["scene_id"],
            "date": data_2019["date"],
            "bands": data_2019["band_urls"]
        },
        "current_2024": {
            "scene_id": data_2024["scene_id"],
            "date": data_2024["date"],
            "bands": data_2024["band_urls"]
        }
    }
    with open("output_data/sentinel2_band_catalog.json", "w") as f:
        json.dump(band_catalog, f, indent=2)
    print(f"\n[OK] Saved calibrated band URLs catalog -> output_data/sentinel2_band_catalog.json")
    
    # 3. Side-by-Side Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_2019)
    axes[0].set_title(f"Baseline Sentinel-2 Level-2A ({data_2019['date']})\nCloud Cover: {data_2019['cloud_cover']:.2f}% | Scene: {data_2019['scene_id'][:15]}...", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_2024)
    axes[1].set_title(f"Recent Sentinel-2 Level-2A ({data_2024['date']})\nCloud Cover: {data_2024['cloud_cover']:.2f}% | Scene: {data_2024['scene_id'][:15]}...", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(f"Sentinel-2 Satellite Data Verification: {TOWN_NAME}\n(Calibrated Level-2A BOA Surface Reflectance via AWS STAC)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    comparison_path = "output_data/01_satellite_comparison.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Generated satellite comparison figure -> {comparison_path}")
    print("\n" + "=" * 75)
    print("[SUCCESS] Sentinel-2 STAC Multispectral Dataset verified live!")
    print("=" * 75)

except Exception as err:
    provenance["module_1"] = "SYNTHETIC_FALLBACK"
    print("\n" + "!" * 75)
    print(f"⚠️ [WARNING: USING SYNTHETIC FALLBACK - LIVE API FETCH FAILED: {err}]")
    print("!" * 75)

# Save provenance flag
with open("output_data/provenance_module_1.json", "w") as f:
    json.dump(provenance, f)
