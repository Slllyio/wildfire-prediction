"""
Training Dataset Generator

Extracts balanced training samples (burnt/non-burnt pixels) with:
- Sentinel-2 time series
- Weather data
- Proximity features
- Seasonal encoding
"""
import ee
import sys
import os
import json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from scripts.gee_utils import *

def generate_training_sample(year, output_dir='../data/training'):
    """
    Generate training data for one year.
    
    Steps:
    1. Sample burnt pixels
    2. Sample non-burnt pixels (balanced)
    3. For each pixel, extract:
        - 6-month Sentinel-2 time series
        - Weather data (ERA5)
        - Proximity features
        - Seasonal encoding
    4. Export as GeoJSON + metadata
    """
    initialize_gee()
    
    print(f"Generating training data for {year}...")
    
    # 1. Get samples
    print("  Sampling burnt pixels...")
    burnt_pixels = get_burnt_pixels(year)
    burnt_count = burnt_pixels.size().getInfo()
    print(f"    Found {burnt_count} burnt pixels")
    
    print("  Sampling non-burnt pixels...")
    non_burnt_pixels = get_non_burnt_pixels(year)
    non_burnt_count = non_burnt_pixels.size().getInfo()
    print(f"    Found {non_burnt_count} non-burnt pixels")
    
    # 2. Merge samples and balance strictly
    limit_per_class = 1000
    burnt_subset = burnt_pixels.limit(limit_per_class)
    non_burnt_subset = non_burnt_pixels.limit(limit_per_class)
    all_samples = burnt_subset.merge(non_burnt_subset)
    
    # 3. For each sample, we need to extract time series
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/samples_{year}.geojson"
    
    # Export to local
    sample_data = all_samples.getInfo() # Now safe (max 2000)
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f)
    
    print(f"[OK] Exported {len(sample_data['features'])} samples to {output_file}")
    
    return output_file

def export_timeseries_task(year, sample_file, folder='wildfire_training'):
    """
    Create GEE Export tasks for time series data.
    
    This is the REAL workflow - export via GEE tasks to Drive.
    """
    initialize_gee()
    
    # Load samples
    with open(sample_file, 'r') as f:
        samples_geojson = json.load(f)
    
    samples = ee.FeatureCollection(samples_geojson)
    
    # For each sample, we build a composite image with time series "bands"
    # This is complex - Prithvi expects temporal sequences, not single images
    
    # Alternative: Export raw Sentinel-2 stack and process with Python/PyTorch
    
    # Get time range (6 months before fire season peak)
    fire_season_peak = f"{year}-04-15"
    start_date = f"{year-1}-10-15"  # 6 months prior
    
    # Get S2 composite
    s2_stack = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start_date, fire_season_peak)\
        .filterBounds(samples.geometry())\
        .select(PRITHVI_INPUT_BANDS)\
        .median()  # Simplified - should keep temporal dimension
    
    # Export task
    task = ee.batch.Export.image.toDrive(
        image=s2_stack,
        description=f'S2_Stack_{year}',
        folder=folder,
        region=samples.geometry().bounds().getInfo()['coordinates'],
        scale=OUTPUT_RESOLUTION_M,
        crs='EPSG:4326',
        maxPixels=1e9
    )
    
    task.start()
    print(f"✓ Export task started: S2_Stack_{year}")
    print(f"  Check status at: https://code.earthengine.google.com/tasks")
    
    return task

if __name__ == '__main__':
    # Generate for training AND validation years
    for year in range(TRAINING_START_YEAR, VALIDATION_YEAR + 1):
        sample_file = generate_training_sample(year)
        print(f"Year {year} complete.\n")
