"""
Analyze High Risk Locations
Identifies the coordinates of the highest risk predictions and compares with actual fire location.
"""
import rasterio
import numpy as np
import joblib
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Actual fire location (May 5, 2025)
ACTUAL_FIRE_LON = 78.289075
ACTUAL_FIRE_LAT = 22.204108

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in km
    return c * r

def main():
    tile_path = "outputs/validation_2025-05-05/fire_tile.tif"
    model_path = "models/checkpoints/rf_fire_model.pkl"
    
    print(f"Analyzing tile: {tile_path}")
    model = joblib.load(model_path)
    
    with rasterio.open(tile_path) as src:
        data = src.read()
        transform = src.transform
        
    # Sliding window prediction
    window_size = 64
    step = 32
    results = []
    
    h, w = data.shape[1], data.shape[2]
    
    for y in range(0, h-window_size, step):
        for x in range(0, w-window_size, step):
            window = data[:, y:y+window_size, x:x+window_size].astype(float)
            
            means = np.mean(window, axis=(1, 2))
            stds = np.std(window, axis=(1, 2))
            
            # B2=0, B3=1, B4=2, B8=3, B11=4, B12=5
            ndvi = (means[3]-means[2])/(means[3]+means[2]+1e-6)
            nbr = (means[3]-means[5])/(means[3]+means[5]+1e-6)
            si = means[4]/(means[3]+1e-6)
            
            feats = np.concatenate([means, stds, [ndvi, nbr, si]]).reshape(1, -1)
            feats = np.nan_to_num(feats)
            
            prob = model.predict_proba(feats)[0][1]
            
            cx, cy = rasterio.transform.xy(transform, y+window_size//2, x+window_size//2)
            results.append((cx, cy, prob))
            
    df = pd.DataFrame(results, columns=['lon', 'lat', 'risk'])
    df = df.sort_values('risk', ascending=False)
    
    # Top 5 Locations
    print("\n=== TOP 5 PREDICTED RISK LOCATIONS ===")
    print(f"{'Rank':<5} {'Risk':<10} {'Location (Lon, Lat)':<25} {'Dist to Actual Fire':<20}")
    print("-" * 65)
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        dist = haversine(row['lon'], row['lat'], ACTUAL_FIRE_LON, ACTUAL_FIRE_LAT)
        print(f"{i+1:<5} {row['risk']:.4f}     {row['lon']:.5f}, {row['lat']:.5f}       {dist:.2f} km")
        
    # Analyze alignment
    peak = df.iloc[0]
    peak_dist = haversine(peak['lon'], peak['lat'], ACTUAL_FIRE_LON, ACTUAL_FIRE_LAT)
    
    print(f"\nAnalysis:")
    print(f"Actual Fire Location: {ACTUAL_FIRE_LON:.5f}, {ACTUAL_FIRE_LAT:.5f}")
    if peak_dist < 2.0:
        print(f"SUCCESS: The highest risk prediction was within {peak_dist:.2f} km of the actual fire!")
    else:
        print(f"OFFSET: The predicted peak was {peak_dist:.2f} km away from the actual fire.")

if __name__ == "__main__":
    main()
