"""
Compare fire detection datasets at different resolutions.

MODIS MCD64A1: 500m (monthly burnt area)
VIIRS: 375m (active fire detection)
FIRMS: Point data (active fire hotspots)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ee
from config import *

ee.Initialize(project=GEE_PROJECT_ID)

beats = ee.FeatureCollection(BEATS_ASSET_ID)
roi = beats.geometry()

print("=== Fire Dataset Comparison ===\n")

# 1. MODIS (500m)
print("1. MODIS MCD64A1 (500m resolution):")
modis = ee.ImageCollection('MODIS/061/MCD64A1')\
    .filterDate('2021-01-01', '2021-12-31')\
    .filterBounds(roi)

modis_pixels = modis.select('BurnDate').map(lambda img: img.gt(0)).sum()\
    .reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=500, maxPixels=1e9)\
    .getInfo()

print(f"   2021 burnt pixels: {modis_pixels.get('BurnDate', 0)}")
print(f"   Resolution: 500m x 500m (25 hectares/pixel)")
print(f"   Area burned: {modis_pixels.get('BurnDate', 0) * 25:.0f} hectares\n")

# 2. VIIRS (375m) - Better resolution
print("2. VIIRS Active Fire (375m resolution):")
try:
    viirs = ee.ImageCollection('FIRMS')\
        .filterDate('2021-01-01', '2021-12-31')\
        .filterBounds(roi)\
        .select('T21')
    
    viirs_count = viirs.size().getInfo()
    print(f"   2021 detections: {viirs_count}")
    print(f"   Resolution: 375m x 375m (14 hectares/pixel)")
except Exception as e:
    print(f"   [INFO] FIRMS/VIIRS not directly available in GEE")
    print(f"   Alternative: Use ee.FeatureCollection('FIRMS') as points\n")

# 3. Active Fire Points (highest precision)
print("3. MODIS/VIIRS Active Fire Points:")
try:
    # MODIS Thermal Anomalies (1km)
    modis_fire = ee.ImageCollection('MODIS/061/MOD14A1')\
        .filterDate('2021-01-01', '2021-12-31')\
        .filterBounds(roi)\
        .select('FireMask')
    
    fire_pixels = modis_fire.map(lambda img: img.gt(7)).sum()\
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=1000, maxPixels=1e9)\
        .getInfo()
    
    print(f"   MODIS thermal anomalies: {fire_pixels.get('FireMask', 0)}")
    print(f"   Resolution: 1km (100 hectares/pixel)")
except Exception as e:
    print(f"   Error: {e}")

print("\n=== Recommendation ===")
print("For 20m resolution training:")
print("1. Use MODIS MCD64A1 (500m) as COARSE labels")
print("2. Supplement with Sentinel-2 NBR differencing for 20m precision")
print("3. Or manually digitize burns from Sentinel-2 imagery")
print("\nCurrent approach is valid - MODIS labels will be upsampled during training.")
