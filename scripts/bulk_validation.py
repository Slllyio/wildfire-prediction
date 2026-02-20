"""
Bulk Validation of Fire Model for ALL Fire Dates in May 2025.
"""
import ee
import os
import sys
import pandas as pd
import numpy as np
import joblib
import rasterio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Hardcode config
GEE_PROJECT_ID = "monkhub-internal-enetra-dev"
try:
    ee.Initialize(project=GEE_PROJECT_ID)
except:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)

BOUNDS = {
    'min_lon': 78.2449,
    'max_lon': 79.0312,
    'min_lat': 21.8803,
    'max_lat': 22.7003
}
REGION = ee.Geometry.Rectangle([BOUNDS['min_lon'], BOUNDS['min_lat'], BOUNDS['max_lon'], BOUNDS['max_lat']])

def get_actual_fires_dates(month=5, year=2025):
    """Find all dates with fires in the month."""
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-31"
    
    print(f"Scanning for fire dates: {start} to {end}...")
    fires = ee.ImageCollection('FIRMS').filterDate(start, end).filterBounds(REGION)
    
    # We need to iterate daily because GEE 'max()' over a month merges everything
    active_dates = []
    
    curr = datetime(year, month, 1)
    end_dt = datetime(year, month, 31)
    
    while curr <= end_dt:
        d_str = curr.strftime("%Y-%m-%d")
        next_d = (curr + timedelta(days=1)).strftime("%Y-%m-%d")
        
        daily = fires.filterDate(d_str, next_d)
        if daily.size().getInfo() > 0:
            count = daily.select('T21').max().gt(300).selfMask().reduceRegion(
                reducer=ee.Reducer.count(), geometry=REGION, scale=1000, maxPixels=1e9
            ).getInfo().get('T21', 0)
            
            if count > 0:
                print(f"  {d_str}: {count} fire pixels")
                active_dates.append({'date': d_str, 'pixels': count})
        
        curr += timedelta(days=1)
        
    return active_dates

def download_tile(date, region, output_path):
    img = get_sentinel2_image(date, region)
    if not img:
        return False
    
    img = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    try:
        url = img.getDownloadURL({
            'scale': 20, 
            'crs': 'EPSG:4326', 
            'region': region.getInfo()['coordinates'], 
            'format': 'GEO_TIFF'
        })
        import requests
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return True
        else:
            print(f"    Download HTTP Error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"    GEE Download URL Error: {e}")
        return False

def get_sentinel2_image(date, region):
    """Get mosaic for date and region."""
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")
    
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start, end)\
        .filterBounds(region)\
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    
    if col.size().getInfo() == 0:
        return None
    return col.mosaic()

def predict_on_tile(tile_path, model):
    with rasterio.open(tile_path) as src:
        data = src.read()
    
    # Fast prediction (vectorized if possible, or simple loop)
    # Reusing the loop logic for simplicity but minimizing overhead
    # ... (same loop as before) ...
    # To speed up, we'll just sample 200 random points? No, we need Max Risk.
    
    # Vectorized approach for speed:
    # 1. Reshape to (C, H, W) -> (C, N)
    c, h, w = data.shape
    # We need sliding window... vectorizing sliding window is tricky without unfold
    # We'll stick to loop but with step=64 (fast stride)
    
    window_size = 64
    step = 64
    
    max_risk = 0
    mean_risk = 0
    count = 0
    
    for y in range(0, h-window_size, step):
        for x in range(0, w-window_size, step):
            window = data[:, y:y+window_size, x:x+window_size].astype(float)
            if window.shape[1] != window_size or window.shape[2] != window_size: continue
            
            means = np.mean(window, axis=(1, 2))
            stds = np.std(window, axis=(1, 2))
            
            ndvi = (means[3]-means[2])/(means[3]+means[2]+1e-6)
            nbr = (means[3]-means[5])/(means[3]+means[5]+1e-6)
            si = means[4]/(means[3]+1e-6)
            
            feats = np.concatenate([means, stds, [ndvi, nbr, si]]).reshape(1, -1)
            feats = np.nan_to_num(feats)
            
            prob = model.predict_proba(feats)[0][1]
            if prob > max_risk: max_risk = prob
            mean_risk += prob
            count += 1
            
    if count > 0:
        mean_risk /= count
    
    return max_risk, mean_risk

def main():
    os.makedirs("outputs/bulk_validation", exist_ok=True)
    model = joblib.load("models/checkpoints/rf_fire_model.pkl")
    
    # 1. Identify all fire dates
    fire_dates = get_actual_fires_dates()
    
    if not fire_dates:
        print("No fires found to validate.")
        return
    
    results = []
    
    print(f"\nRunning validation on {len(fire_dates)} events...")
    
    for event in fire_dates:
        date = event['date']
        pixels = event['pixels']
        
        # Get actual fire locations to define download region
        fires = ee.ImageCollection('FIRMS').filterDate(date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")).filterBounds(REGION)
        fire_img = fires.select('T21').max().gt(300).selfMask()
        
        # Buffer around fire locations (10km or whole forest if small)
        fire_region = fire_img.reduceToVectors(geometry=REGION, scale=1000, geometryType='centroid').geometry().buffer(5000).bounds()
        
        print(f"  Validating {date} ({pixels} pixels)...")
        
        tile_path = f"outputs/bulk_validation/tile_{date}.tif"
        if not os.path.exists(tile_path):
            if not download_tile(date, fire_region, tile_path):
                print("    Skipping (Download failed)")
                continue
        
        max_r, mean_r = predict_on_tile(tile_path, model)
        print(f"    -> Max Risk: {max_r:.4f}, Mean Risk: {mean_r:.4f}")
        
        results.append({
            'date': date,
            'actual_fires': pixels,
            'max_risk': max_r,
            'mean_risk': mean_r
        })
        
    # Stats
    df = pd.DataFrame(results)
    df.to_csv("outputs/bulk_validation/report.csv", index=False)
    
    print("\n=== VALIDATION SUMMARY ===")
    print(df)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(df['date'], df['max_risk'], color='salmon', label='Predicted Max Risk')
    plt.plot(df['date'], df['mean_risk'], marker='o', color='blue', label='Mean Area Risk')
    plt.axhline(y=0.40, color='r', linestyle='--', label='Warning Threshold (0.40)')
    plt.ylabel('Risk Probability')
    plt.title('Model Performance Across All May 2025 Fire Events')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/bulk_validation/performance_chart.png")

if __name__ == "__main__":
    main()
