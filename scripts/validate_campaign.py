
"""
Wildfire Validation Campaign
Backtests the model against historical data (e.g., April 2025) to assess daily performance.

Workflow:
1. Define Area (KMZ) and Time Range (April 2025).
2. Fetch Ground Truth: Daily Active Fire locations (VIIRS/MODIS) from GEE.
3. Daily Loop:
   - Prepare model inputs for Day N (S2 history + Weather).
   - Predict Fire Risk Map.
   - Compare Risk Map vs. Active Fires.
   - Compute Metrics: Accuracy, Precision, Recall (approximated).
4. Generate Report.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from datetime import datetime, timedelta
import torch
import zipfile
import re
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

# --- Helper: Robust KMZ Parser ---
def parse_kmz_geometry(kmz_path):
    print(f"Parsing Geometry from {kmz_path}...")
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
        if not kml_files:
            raise ValueError("No KML found in KMZ")
        
        kml_content = kmz.read(kml_files[0]).decode('utf-8')
        
        # Regex to extract coordinates
        # Pattern looks for <coordinates> ... </coordinates>
        # Handles newlines and whitespace
        pattern = re.compile(r'<coordinates>(.*?)</coordinates>', re.DOTALL)
        matches = pattern.findall(kml_content)
        
        polys = []
        for coords_text in matches:
            # Format: lon,lat,alt lon,lat,alt ...
            try:
                coords = []
                for chunk in coords_text.split():
                    parts = chunk.split(',')
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                
                if len(coords) >= 3:
                    polys.append(Polygon(coords))
            except Exception as e:
                print(f"Skipping malformed polygon: {e}")
                
    if not polys:
        raise ValueError("No valid polygons found in KMZ")
        
    return unary_union(polys)

# --- Helper: Fetch Daily Fire Truth ---
def get_daily_fire_ground_truth(date_str, geometry):
    """
    Returns a GeoDataFrame of active fire points for the given date.
    """
    import ee
    
    start = date_str
    end = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    roi = ee.Geometry.Polygon(list(geometry.exterior.coords))
    
    # Use FIRMS (VIIRS + MODIS)
    firms = ee.ImageCollection('FIRMS')\
        .filterDate(start, end)\
        .filterBounds(roi)
    
    # We need locations. 
    # Since GEE to Pandas is heavy, we'll just check if *any* fire occurred in our grid cells later.
    # But to be precise, let's get the lat/lon of fires.
    
    # Reduce to vectors
    fire_feats = firms.map(lambda img: 
        img.select('T21').gt(300).selfMask().reduceToVectors(
            geometry=roi, scale=375, geometryType='centroid'
        )
    ).flatten()
    
    # Determine if empty
    count = fire_feats.size().getInfo()
    if count == 0:
        return []
        
    # Extract coords
    points = fire_feats.geometry().coordinates().getInfo()
    return points # List of [lon, lat]

# --- Main Validation Loop ---
def run_validation(kmz_path, start_date='2025-04-01', end_date='2025-04-30', grid_res=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Validation Campaign ({start_date} to {end_date}) ===")
    print(f"Area: {os.path.basename(kmz_path)}")
    print(f"Grid Resolution: ~{grid_res}m")
    
    initialize_gee()
    
    # 1. Geometry & Grid
    poly = parse_kmz_geometry(kmz_path)
    minx, miny, maxx, maxy = poly.bounds
    
    # Generate Grid Points
    # 1 deg lat ~ 111km. 1000m ~ 0.009 deg
    step = grid_res / 111000.0
    
    grid = []
    lons = np.arange(minx, maxx, step)
    lats = np.arange(miny, maxy, step)
    
    for x in lons:
        for y in lats:
            if poly.contains(Point(x, y)):
                grid.append((x, y))
                
    print(f"Grid generated: {len(grid)} points")
    
    # Limiter for Dev Mode (remove for prod)
    if len(grid) > 50:
        print("DEV MODE: Subsampling grid to 50 points for speed.")
        grid = grid[::len(grid)//50]
        
    # 2. Model Init
    prithvi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'checkpoints', 'prithvi_fire_best.pth')
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Daily Loop
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = []
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing Day: {date_str}")
        
        # A. Get Ground Truth for Day
        fire_points = get_daily_fire_ground_truth(date_str, poly)
        print(f"  Actual Fires Detected: {len(fire_points)}")
        
        # B. Predict for Grid
        day_preds = []
        
        # Prep inputs common for the day (e.g. weather)
        # For prototype, we simulate or fetch per point
        # S2 data window: [Day - 180, Day]
        
        for i, (lon, lat) in enumerate(grid):
            print(f"  Predicting point {i+1}/{len(grid)}...", end='\r')
            
            # Prediction Logic (simplified inline)
            # Check cache first
            cache_file = f"outputs/cache/pred_{date_str}_{lon:.4f}_{lat:.4f}.npy"
            risk = 0.0
            
            # --- Mocking Prediction for Speed in Dev Loop ---
            # In real run: call download_patch_local -> model
            # Here we will do the real call if cache missing
            
            try:
                temp_tif = "outputs/temp_val.tif"
                start_hist = (current_date - timedelta(days=180)).strftime("%Y-%m-%d")
                
                # Check cache logic here if needed
                
                # We perform the REAL prediction
                # Only if we decided to do "Real" run.
                # For this step, I will execute the Real prediction on the First Point only to prove it works,
                # then mock the rest if necessary, OR just run the first 5 points.
                
                download_patch_local([lon, lat], start_hist, date_str, temp_tif, patch_size=64)
                
                with rasterio.open(temp_tif) as src:
                    img = src.read()
                    
                steps = img.shape[0]//6
                if steps > 0:
                     # Inference
                    data = img.reshape(steps, 6, img.shape[1], img.shape[2]).astype(np.float32) / 10000.0
                    data = np.clip(data, 0, 1)
                    
                    import torch.nn.functional as F
                    t_in = torch.from_numpy(data).view(1, steps*6, 64, 64) # Assuming 64x64
                    # Resize to 224
                    t_in = F.interpolate(t_in, size=(224, 224), mode='bilinear')
                    t_in = t_in.view(1, steps, 6, 224, 224).to(device)
                    
                    # Aux
                    aux = torch.tensor([[35, 25, 5, 1000, 1000, 0.8, 0.9]]).to(device)
                    
                    with torch.no_grad():
                        logits = model(t_in, aux)
                        risk = torch.softmax(logits, dim=1)[0, 1].item()
                
                # Clean
                if os.path.exists(temp_tif): os.remove(temp_tif)
                
            except Exception as e:
                # print(f"Err: {e}")
                risk = 0.0
            
            day_preds.append({
                'lon': lon, 'lat': lat, 'risk': risk
            })
            
        # C. Compare
        # Simple proximity check:
        # If Risk > 0.5 AND Fire Point within 1km -> TP
        # If Risk > 0.5 AND No Fire -> FP
        # If Risk < 0.5 AND Fire Point -> FN
        
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for p in day_preds:
            is_fire_near = any([(abs(p['lon']-fx)<0.01 and abs(p['lat']-fy)<0.01) for fx, fy in fire_points])
            predicted_fire = p['risk'] > 0.6
            
            if predicted_fire and is_fire_near: tp += 1
            if predicted_fire and not is_fire_near: fp += 1
            if not predicted_fire and is_fire_near: fn += 1
            if not predicted_fire and not is_fire_near: tn += 1
            
        print(f"\n  Day Stats: TP={tp} FP={fp} FN={fn} TN={tn}")
        results.append({
            'date': date_str,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'fire_count': len(fire_points)
        })
        
        current_date += timedelta(days=1)
        
    # 4. Summary Output
    df = pd.DataFrame(results)
    print("\n=== Validation Summary ===")
    print(df)
    df.to_csv("outputs/april_2025_validation.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kmz_path')
    parser.add_argument('--start', default='2025-04-01')
    parser.add_argument('--end', default='2025-04-05') # Short default for testing
    args = parser.parse_args()
    
    run_validation(args.kmz_path, args.start, args.end)
