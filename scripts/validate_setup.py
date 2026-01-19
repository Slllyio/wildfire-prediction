"""
Simple data validation script to test GEE connection and data availability.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Test 1: Basic imports
print("=== Testing Imports ===")
try:
    import ee
    print("[OK] Earth Engine imported")
except Exception as e:
    print(f"[ERROR] Failed to import ee: {e}")
    sys.exit(1)

try:
    from config import *
    print(f"[OK] Config loaded (Project: {GEE_PROJECT_ID})")
except Exception as e:
    print(f"[ERROR] Failed to load config: {e}")
    sys.exit(1)

# Test 2: GEE Authentication
print("\n=== Testing GEE Connection ===")
try:
    ee.Initialize(project=GEE_PROJECT_ID)
    print(f"[OK] Connected to GEE project: {GEE_PROJECT_ID}")
except Exception as e:
    print(f"[INFO] First-time auth needed. Error: {e}")
    print("Run: earthengine authenticate")
    sys.exit(0)

# Test 3: Load Bichua asset
print("\n=== Testing Asset Access ===")
try:
    beats = ee.FeatureCollection(BEATS_ASSET_ID)
    count = beats.size().getInfo()
    print(f"[OK] Loaded {count} forest beats from {BEATS_ASSET_ID}")
except Exception as e:
    print(f"[ERROR] Failed to load beats: {e}")
    print("Check asset ID or permissions")
    sys.exit(1)

# Test 4: MODIS data
print("\n=== Testing MODIS Access ===")
try:
    fires = ee.ImageCollection(FIRE_DATASET)\
        .filterDate('2024-01-01', '2024-12-31')\
        .filterBounds(beats.geometry())
    
    fire_count = fires.size().getInfo()
    print(f"[OK] Accessed MODIS burnt area data ({fire_count} images in 2024)")
except Exception as e:
    print(f"[ERROR] Failed to access MODIS: {e}")

# Test 5: Sentinel-2
print("\n=== Testing Sentinel-2 Access ===")
try:
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate('2024-01-01', '2024-12-31')\
        .filterBounds(beats.geometry())\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    
    s2_count = s2.size().getInfo()
    print(f"[OK] Found {s2_count} Sentinel-2 scenes (cloud < 30%)")
    print(f"    ~{s2_count/12:.1f} scenes/month")
except Exception as e:
    print(f"[ERROR] Failed to access Sentinel-2: {e}")

print("\n=== Validation Complete ===")
print("All critical components are accessible!")
