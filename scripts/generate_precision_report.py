"""
Comprehensive Spatial Precision Analysis for May 2025 Fires.
Generates Maps, Distance Errors (m), and a Summary CSV for each fire event.
"""
import ee
import os
import sys
import pandas as pd
import numpy as np
import joblib
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

# Config
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

def haversine_m(lon1, lat1, lon2, lat2):
    """Calculate distance in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371 * 1000

def get_fire_cluster_centroid(date):
    """Get centroid of actual fires on date."""
    fires = ee.ImageCollection('FIRMS').filterDate(date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")).filterBounds(REGION)
    if fires.size().getInfo() == 0:
        return None, 0
    
    fire_img = fires.select('T21').max().gt(300).selfMask()
    # Centroid of all pixels
    centroid_geom = fire_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=REGION, scale=500).getInfo()
    if 'longitude' in centroid_geom and 'latitude' in centroid_geom:
        return (centroid_geom['longitude'], centroid_geom['latitude']), fires.size().getInfo()
    
    # Fallback to reduceToVectors
    try:
        vec = fire_img.reduceToVectors(geometry=REGION, scale=500, geometryType='centroid')
        feats = vec.getInfo()['features']
        if feats:
            coords = feats[0]['geometry']['coordinates']
            return (coords[0], coords[1]), len(feats)
    except:
        pass
    return None, 0

def predict_weighted_centroid(tile_path, model):
    """Predict and find weighted risk centroid."""
    with rasterio.open(tile_path) as src:
        data = src.read()
        transform = src.transform
    
    window_size = 64
    step = 32
    h, w = data.shape[1], data.shape[2]
    
    results = []
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
            
            cx, cy = rasterio.transform.xy(transform, y+window_size//2, x+window_size//2)
            results.append((cx, cy, prob))
            
    df = pd.DataFrame(results, columns=['lon', 'lat', 'risk'])
    high_risk = df[df['risk'] > 0.35]
    if not high_risk.empty:
        pw_lon = np.average(high_risk['lon'], weights=high_risk['risk'])
        pw_lat = np.average(high_risk['lat'], weights=high_risk['risk'])
        return (pw_lon, pw_lat), df['risk'].max(), df
    return None, df['risk'].max(), df

def main():
    output_dir = "outputs/precision_analysis"
    os.makedirs(output_dir, exist_ok=True)
    model = joblib.load("models/checkpoints/rf_fire_model.pkl")
    
    # Dates from previous scan
    dates = ["2025-05-02", "2025-05-04", "2025-05-05", "2025-05-07", "2025-05-09", 
             "2025-05-10", "2025-05-11", "2025-05-12", "2025-05-20"]
    
    report = []
    
    for date in dates:
        print(f"\nProcessing {date}...")
        
        # 1. Actual
        actual_centroid, num_pixels = get_fire_cluster_centroid(date)
        if not actual_centroid:
            print(f"  No fire data found for {date}")
            continue
            
        # 2. Predicted
        tile_path = f"outputs/bulk_validation/tile_{date}.tif"
        if not os.path.exists(tile_path):
            print(f"  Tile {tile_path} missing. Skipping.")
            continue
            
        pred_centroid, max_risk, full_df = predict_weighted_centroid(tile_path, model)
        
        # 3. Stats
        dist_m = None
        if pred_centroid:
            dist_m = haversine_m(actual_centroid[0], actual_centroid[1], pred_centroid[0], pred_centroid[1])
            print(f"  Actual: {actual_centroid}")
            print(f"  Pred:   {pred_centroid}")
            print(f"  Error:  {dist_m:.2f} meters")
        
        report.append({
            'date': date,
            'actual_lon': actual_centroid[0],
            'actual_lat': actual_centroid[1],
            'pred_lon': pred_centroid[0] if pred_centroid else None,
            'pred_lat': pred_centroid[1] if pred_centroid else None,
            'error_meters': dist_m,
            'max_risk': max_risk
        })
        
        # 4. Map
        plt.figure(figsize=(10, 8))
        plt.scatter(full_df['lon'], full_df['lat'], c=full_df['risk'], cmap='YlOrRd', s=40, alpha=0.6)
        plt.colorbar(label='Risk')
        plt.scatter([actual_centroid[0]], [actual_centroid[1]], c='blue', marker='x', s=200, label='Actual Fire', linewidth=3)
        if pred_centroid:
            plt.scatter([pred_centroid[0]], [pred_centroid[1]], c='green', marker='+', s=200, label='Predicted Centroid', linewidth=3)
        plt.title(f"Fire Prediction Accuracy: {date}\nError: {dist_m/1000:.2f} km")
        plt.legend()
        plt.savefig(f"{output_dir}/map_{date}.png")
        plt.close()

    # Save CSV
    df_report = pd.DataFrame(report)
    df_report.to_csv(f"{output_dir}/precision_report.csv", index=False)
    print(f"\nFinal report saved to {output_dir}/precision_report.csv")

if __name__ == "__main__":
    main()
