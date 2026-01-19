"""
TEST: Extract features for first 10 samples only
"""
import pandas as pd
import ee
import numpy as np
from datetime import datetime, timedelta
import math
import sys

print("=" * 60, flush=True)
print("TEST: Processing 10 samples only", flush=True)
print("=" * 60, flush=True)

# Initialize Earth Engine
try:
    print("\n1. Initializing Earth Engine...", flush=True)
    ee.Initialize()
    print("   ✓ Success!", flush=True)
except Exception as e:
    print(f"   ✗ Failed: {e}", flush=True)
    sys.exit(1)

def get_simple_features(lat, lon, date_str):
    """Simplified feature extraction - just basic data"""
    print(f"   → Extracting for {lat:.3f}, {lon:.3f} on {date_str}", flush=True)
    
    features = {
        'lat': lat,
        'lon': lon,
        'date': date_str
    }
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        point = ee.Geometry.Point([lon, lat])
        buffer_200m = point.buffer(200)
        
        # Just get ONE simple dataset: ERA5 temperature
        print("      - Fetching ERA5 temp...", flush=True)
        era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .filterBounds(buffer_200m)
                .filterDate(date_str, (date + timedelta(days=1)).strftime('%Y-%m-%d'))
                .select(['temperature_2m'])
                .mean())
        
        stats = era5.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer_200m,
            scale=100
        ).getInfo()
        
        if stats and 'temperature_2m' in stats:
            features['temperature'] = stats['temperature_2m']
            print(f"      ✓ Got temp: {stats['temperature_2m']:.1f}K", flush=True)
        else:
            features['temperature'] = None
            print("      ✗ No temp data", flush=True)
            
    except Exception as e:
        print(f"      ✗ Error: {e}", flush=True)
        features['temperature'] = None
    
    return features

# Process first 10 samples
print("\n2. Loading data...", flush=True)
from pathlib import Path

base_path = Path('data/training')
year_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()])

if not year_dirs:
    print("   ✗ No year directories found!", flush=True)
    sys.exit(1)

print(f"   ✓ Found {len(year_dirs)} years", flush=True)

all_samples = []
sample_count = 0
MAX_SAMPLES = 10

print(f"\n3. Processing up to {MAX_SAMPLES} samples...", flush=True)

for year_dir in year_dirs:
    if sample_count >= MAX_SAMPLES:
        break
        
    labels_file = year_dir / 'labels.csv'
    if not labels_file.exists():
        continue
    
    df = pd.read_csv(labels_file)
    print(f"\n  Year {year_dir.name}: {len(df)} samples available", flush=True)
    
    for idx, row in df.iterrows():
        if sample_count >= MAX_SAMPLES:
            break
        
        sample_count += 1
        print(f"\n  Sample {sample_count}/{MAX_SAMPLES}:", flush=True)
        
        sample = {
            'year': int(year_dir.name),
            'label': row['label'],
            'filename': row['filename']
        }
        
        # Get location
        lat = row.get('latitude', 22.5)
        lon = row.get('longitude', 78.5)
        
        # Get date
        if 'date' in row:
            date_str = row['date']
        else:
            import re
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', row['filename'])
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else f"{year_dir.name}-03-15"
        
        # Extract features
        features = get_simple_features(lat, lon, date_str)
        sample.update(features)
        
        all_samples.append(sample)

# Save results
print(f"\n4. Creating CSV...", flush=True)
df_result = pd.DataFrame(all_samples)

outfile = 'test_features_10.csv'
df_result.to_csv(outfile, index=False)

print("\n" + "=" * 60, flush=True)
print(f"✅ Test Complete!", flush=True)
print(f"   Processed: {len(df_result)} samples", flush=True)
print(f"   Saved to: {outfile}", flush=True)
print(f"   Columns: {list(df_result.columns)}", flush=True)
print("=" * 60, flush=True)
