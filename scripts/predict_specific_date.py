"""
Predict Fire Risk for a Specific Date (Validation).
Downloads data for the target date and runs RF prediction.
"""
import ee
import os
import sys
import shutil
import joblib
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import zipfile
import re
from datetime import datetime, timedelta

# from scripts.config import GEE_PROJECT_ID
GEE_PROJECT_ID = "van-suraksha-alert"

# Hardcode bounds for West Chhindwara
BOUNDS = {
    'min_lon': 78.2449,
    'max_lon': 79.0312,
    'min_lat': 21.8803,
    'max_lat': 22.7003
}

def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)

def get_sentinel2_image(date, region):
    """Get the closest Sentinel-2 image to the date."""
    target_date = datetime.strptime(date, "%Y-%m-%d")
    start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=5)).strftime("%Y-%m-%d")
    
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start, end)\
        .filterBounds(region)\
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    
    count = col.size().getInfo()
    if count == 0:
        return None
    
    # Mosaic the best images
    return col.mosaic()

def download_tile(region, date, output_path):
    """Download a single tile for the date."""
    img = get_sentinel2_image(date, region)
    if not img:
        return False
        
    # Select bands: B2, B3, B4, B8, B11, B12
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
        with open(output_path, 'wb') as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def parse_kmz(kmz_path):
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            content = kml_file.read().decode('utf-8')
            coord_pattern = re.compile(r'<coordinates>(.*?)</coordinates>', re.DOTALL)
            matches = coord_pattern.findall(content)
            polygons = []
            for coords_str in matches:
                coords_str = coords_str.strip()
                points = []
                for tupl in coords_str.split():
                    parts = tupl.split(',')
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            points.append((lon, lat))
                        except:
                            continue
                if len(points) > 2:
                    polygons.append(Polygon(points))
            if not polygons:
                raise ValueError("No polygons found")
            return unary_union(polygons)

def predict_on_tile(tile_path, model, output_list):
    """Run sliding window prediction on a tile."""
    try:
        with rasterio.open(tile_path) as src:
            data = src.read() # (Bands, H, W)
            transform = src.transform
            
        # Simplified features: Mean of bands + Indices
        # RF expects: [B2,B3,B4,B8,B11,B12, stds..., NDVI, NBR, SI]
        
        # We need to replicate the feature extraction from train_rf_model.py
        # There we had (Time=6, Bands=6). Here we have (Bands=6) single image.
        # We'll treat Time=1.
        
        # Sliding window 64x64
        window_size = 64
        step = 32
        h, w = data.shape[1], data.shape[2]
        
        for y in range(0, h-window_size, step):
            for x in range(0, w-window_size, step):
                window = data[:, y:y+window_size, x:x+window_size].astype(float)
                if window.shape[1] != window_size or window.shape[2] != window_size:
                    continue
                
                # Features
                means = np.mean(window, axis=(1, 2))
                stds = np.std(window, axis=(1, 2))
                
                # Indices
                # B2=0, B3=1, B4=2, B8=3, B11=4, B12=5
                ndvi = (means[3]-means[2])/(means[3]+means[2]+1e-6)
                nbr = (means[3]-means[5])/(means[3]+means[5]+1e-6)
                si = means[4]/(means[3]+1e-6)
                
                feats = np.concatenate([means, stds, [ndvi, nbr, si]])
                feats = np.nan_to_num(feats).reshape(1, -1)
                
                prob = model.predict_proba(feats)[0][1]
                
                # Center
                cx, cy = rasterio.transform.xy(transform, y+window_size//2, x+window_size//2)
                output_list.append((cx, cy, prob))
                
    except Exception as e:
        print(f"Error processing {tile_path}: {e}")

def main():
    target_date = "2025-05-05"
    kmz_path = "data/West Chhindwara.kmz"
    output_dir = f"outputs/validation_{target_date}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Validating prediction for date: {target_date}")
    
    initialize_gee()
    model = joblib.load('models/checkpoints/rf_fire_model.pkl')
    forest = parse_kmz(kmz_path)
    
    # 1. Define area of interest (AOI) around the FIRE location found
    # From previous step: [78.28907532630042, 22.204108035224273]
    fire_lon, fire_lat = 78.289075, 22.204108
    aoi_geom = ee.Geometry.Point([fire_lon, fire_lat]).buffer(5000).bounds() # 10km box
    
    # Download just this AOI tile for speed
    print("Downloading Sentinel-2 tile for fire location...")
    tile_path = os.path.join(output_dir, "fire_tile.tif")
    if download_tile(aoi_geom, target_date, tile_path):
        print("  Tile downloaded.")
    else:
        print("  Failed to download tile!")
        return

    # 2. Predict
    print("Running prediction...")
    results = []
    predict_on_tile(tile_path, model, results)
    
    # 3. Analyze
    df = pd.DataFrame(results, columns=['lon', 'lat', 'risk'])
    print(f"\nPrediction Stats for {target_date}:")
    print(f"  Max Risk: {df['risk'].max():.4f}")
    print(f"  Mean Risk: {df['risk'].mean():.4f}")
    
    high_risk = df[df['risk'] > 0.5]
    print(f"  High Risk Points (>0.5): {len(high_risk)}")
    
    # Save map
    if not df.empty:
        plt.figure(figsize=(10, 8))
        plt.scatter(df['lon'], df['lat'], c=df['risk'], cmap='RdYlGn_r', vmin=0, vmax=1, s=50)
        plt.colorbar(label='Risk')
        plt.scatter([fire_lon], [fire_lat], c='blue', marker='x', s=200, label='Actual Fire', linewidth=3)
        plt.legend()
        plt.title(f"Prediction vs Reality ({target_date})")
        plt.savefig(os.path.join(output_dir, "validation_map.png"))
        print(f"Map saved to {output_dir}/validation_map.png")

if __name__ == "__main__":
    main()
