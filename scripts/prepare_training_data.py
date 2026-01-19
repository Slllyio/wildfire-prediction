"""
Prepare Training Data from FIRMS Fire History
Downloads fire locations and creates labeled dataset for fine-tuning.
"""
import ee
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEE_PROJECT_ID, PRITHVI_INPUT_BANDS
from gee_utils import get_multisource_features

# Region bounds for West Chhindwara
BOUNDS = {
    'min_lon': 78.2449,
    'max_lon': 79.0312,
    'min_lat': 21.8803,
    'max_lat': 22.7003
}

print_lock = threading.Lock()
def safe_print(msg):
    with print_lock:
        print(msg)

def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"[OK] GEE initialized: {GEE_PROJECT_ID}")
    except:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)


def get_firms_fire_locations(start_date, end_date, bounds, max_points=500):
    """
    Get fire locations from FIRMS (MODIS/VIIRS Active Fire).
    Returns list of dicts with lon, lat, date.
    """
    print(f"Fetching FIRMS fire data: {start_date} to {end_date}")
    
    region = ee.Geometry.Rectangle([
        bounds['min_lon'], bounds['min_lat'],
        bounds['max_lon'], bounds['max_lat']
    ])
    
    # FIRMS dataset
    firms = ee.ImageCollection("FIRMS")\
        .filterDate(start_date, end_date)\
        .filterBounds(region)
    
    # helper to extract points from each image
    def extract_features(image):
        date = image.date().format('yyyy-MM-dd')
        # T21 is brightness temp. select it and mask low confidence
        hotspots = image.select('T21').gt(300)
        
        # Use sample() instead of reduceToVectors to avoid band issues.
        # This samples the centroids of pixels where mask is 1.
        # Increased numPixels to cap higher, but we want ALL fires if possible.
        points = hotspots.updateMask(hotspots).sample(
            region=region,
            scale=1000,
            numPixels=500, # Much higher limit
            geometries=True
        )
        return points.map(lambda f: f.set('date', date, 'label', 1))

    # Flatten the collection of feature collections
    all_fires = firms.map(extract_features).flatten()
    
    # Limit locally to avoid timeout/too many points
    # We download 'max_points' features
    fire_features = all_fires.limit(max_points).getInfo().get('features', [])
    
    fire_locations = []
    for f in fire_features:
        coords = f['geometry']['coordinates']
        props = f.get('properties', {})
        fire_locations.append({
            'lon': coords[0],
            'lat': coords[1],
            'date': props.get('date'),
            'label': 1
        })
    
    print(f"  Extracted {len(fire_locations)} fire locations with dates")
    return fire_locations


def get_non_fire_locations(fire_locations, bounds, num_samples=500, year=2024):
    """
    Generate non-fire locations (negative samples).
    Avoids proximity to known fire locations.
    Assigns random date during fire season.
    """
    print(f"Generating {num_samples} non-fire locations...")
    
    import random
    random.seed(42)
    
    # Create exclusion zones
    fire_coords = [(f['lon'], f['lat']) for f in fire_locations]
    
    non_fire = []
    attempts = 0
    max_attempts = num_samples * 10
    
    # Date range for random sampling
    start_dt = datetime(year, 2, 1)
    end_dt = datetime(year, 6, 30)
    days_range = (end_dt - start_dt).days
    
    while len(non_fire) < num_samples and attempts < max_attempts:
        lon = random.uniform(bounds['min_lon'], bounds['max_lon'])
        lat = random.uniform(bounds['min_lat'], bounds['max_lat'])
        
        # Check buffer
        too_close = False
        for fx, fy in fire_coords:
            if abs(lon - fx) < 0.045 and abs(lat - fy) < 0.045:
                too_close = True
                break
        
        if not too_close:
            # Random date
            rand_day = random.randint(0, days_range)
            rand_date = (start_dt + timedelta(days=rand_day)).strftime('%Y-%m-%d')
            
            non_fire.append({
                'lon': lon,
                'lat': lat,
                'date': rand_date,
                'label': 0
            })
        
        attempts += 1
    
    print(f"  Generated {len(non_fire)} non-fire locations")
    return non_fire


def download_training_patch(point, output_dir, idx, year):
    """
    Download S2 time-series patch for a training point.
    Uses 6 months before expected fire date (fire season: March-May).
    """
    # Fire season is March-May, so use Oct-Mar for pre-fire data
    end_date = f"{year}-03-01"
    start_date = f"{year-1}-09-01"
    
    lon, lat = point['lon'], point['lat']
    point_geom = ee.Geometry.Point([lon, lat])
    
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start_date, end_date)\
        .filterBounds(point_geom.buffer(1000))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))\
        .select(PRITHVI_INPUT_BANDS)
    
    count = s2_col.size().getInfo()
    if count == 0:
        return None
    
    # Monthly composites
    fake_empty = ee.Image.constant([0]*6).rename(PRITHVI_INPUT_BANDS).cast(dict(zip(PRITHVI_INPUT_BANDS, ['float']*6)))
    
    steps = []
    curr_date = ee.Date(start_date)
    for i in range(6):
        d1 = curr_date.advance(i, 'month')
        d2 = d1.advance(1, 'month')
        img = s2_col.filterDate(d1, d2).median()
        img = img.addBands(fake_empty, overwrite=False).select(PRITHVI_INPUT_BANDS).unmask(0)
        steps.append(img)
    
    stacked = ee.ImageCollection(steps).toBands()
    region = point_geom.buffer(640).bounds()  # ~64 pixels at 20m
    
    try:
        url = stacked.getDownloadURL({
            'scale': 20,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF',
            'region': region.getInfo()['coordinates']
        })
        
        import requests
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            label = point['label']
            filename = f"{'fire' if label == 1 else 'nofire'}_{idx:04d}.tif"
            path = os.path.join(output_dir, filename)
            with open(path, 'wb') as f:
                f.write(response.content)
            return path
    except Exception as e:
        safe_print(f"  Error downloading patch {idx}: {e}")
    
    return None


def download_training_data_parallel(points, output_dir, year, max_workers=8):
    """
    Download all training patches in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    def download_one(idx_point):
        idx, point = idx_point
        result = download_training_patch(point, output_dir, idx, year)
        if result:
            safe_print(f"  [OK] Sample {idx+1}/{len(points)}")
            
            # Extract environmental features
            try:
                # Use Lag 1 (yesterday's weather) to predict today's fire
                fire_date = point.get('date', f"{year}-06-01")
                feats = get_multisource_features(ee.Geometry.Point([point['lon'], point['lat']]), 
                                              fire_date)
            except Exception as e:
                safe_print(f"  Warning: Feature extraction failed for {idx}: {e}")
                feats = {}
            
        else:
            safe_print(f"  [FAIL] Sample {idx+1}/{len(points)}")
            feats = {}
            
        return idx, result, point['label'], feats
    
    print(f"Downloading {len(points)} training patches with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, (i, p)): i for i, p in enumerate(points)}
        for future in as_completed(futures):
            idx, path, label, feats = future.result()
            if path:
                record = {'idx': idx, 'path': path, 'label': label}
                record.update(feats)
                results.append(record)
    
    print(f"Downloaded {len(results)}/{len(points)} patches successfully")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024, help='Fire season year')
    parser.add_argument('--fire-samples', type=int, default=200, help='Number of fire samples')
    parser.add_argument('--nofire-samples', type=int, default=200, help='Number of non-fire samples')
    parser.add_argument('--output-dir', default='data/training', help='Output directory')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    args = parser.parse_args()
    
    initialize_gee()
    
    # 1. Get fire locations from FIRMS
    start_date = f"{args.year}-02-01"  # Start of fire season
    end_date = f"{args.year}-06-30"    # End of fire season
    
    fire_locations = get_firms_fire_locations(start_date, end_date, BOUNDS, args.fire_samples)
    
    if not fire_locations:
        print("ERROR: No fire locations found in FIRMS!")
        print("Trying previous year...")
        start_date = f"{args.year-1}-02-01"
        end_date = f"{args.year-1}-06-30"
        fire_locations = get_firms_fire_locations(start_date, end_date, BOUNDS, args.fire_samples)
    
    if not fire_locations:
        print("ERROR: Still no fire locations found!")
        return
    
    # 2. Generate non-fire locations
    non_fire_locations = get_non_fire_locations(fire_locations, BOUNDS, args.nofire_samples, year=args.year)
    
    # 3. Combine and shuffle
    all_points = fire_locations + non_fire_locations
    import random
    random.seed(42)
    random.shuffle(all_points)
    
    print(f"\nTotal training samples: {len(all_points)}")
    print(f"  Fire: {sum(1 for p in all_points if p['label'] == 1)}")
    print(f"  Non-fire: {sum(1 for p in all_points if p['label'] == 0)}")
    
    # 4. Download training patches
    results = download_training_data_parallel(all_points, args.output_dir, args.year, args.workers)
    
    # 5. Save labels CSV
    # 5. Save labels CSV
    # Construct DF from results, ensuring filename is basename
    final_records = []
    for r in results:
        rec = r.copy()
        rec['filename'] = os.path.basename(r['path'])
        if 'path' in rec: del rec['path']
        if 'idx' in rec: del rec['idx']
        final_records.append(rec)
        
    labels_df = pd.DataFrame(final_records)
    labels_path = os.path.join(args.output_dir, 'labels.csv')
    labels_df.to_csv(labels_path, index=False)
    print(f"\nLabels saved to {labels_path}")
    
    # Stats
    fire_count = sum(1 for r in results if r['label'] == 1)
    nofire_count = len(results) - fire_count
    print(f"\n[OK] Training data ready!")
    print(f"  Fire samples: {fire_count}")
    print(f"  Non-fire samples: {nofire_count}")
    print(f"  Location: {args.output_dir}/")


if __name__ == "__main__":
    main()
