"""
Generate forensic maps for the Top 5 most accurate fire events.
Outputs: map_forensic_{date}.png for each.
"""
import ee
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
import os
from datetime import datetime, timedelta

# Config
GEE_PROJECT_ID = "monkhub-internal-enetra-dev"
try: ee.Initialize(project=GEE_PROJECT_ID)
except: ee.Authenticate(); ee.Initialize(project=GEE_PROJECT_ID)

BOUNDS = {'min_lon': 78.2449, 'max_lon': 79.0312, 'min_lat': 21.8803, 'max_lat': 22.7003}
REGION = ee.Geometry.Rectangle([BOUNDS['min_lon'], BOUNDS['min_lat'], BOUNDS['max_lon'], BOUNDS['max_lat']])

def get_actual_fires(date):
    fires = ee.ImageCollection('FIRMS').filterDate(date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")).filterBounds(REGION)
    if fires.size().getInfo() == 0: return np.array([])
    fire_img = fires.select('T21').max().gt(300).selfMask()
    vec = fire_img.reduceToVectors(geometry=REGION, scale=500, geometryType='centroid')
    feats = vec.getInfo()['features']
    return np.array([f['geometry']['coordinates'] for f in feats])

def run_prediction_grid(tile_path, model):
    with rasterio.open(tile_path) as src:
        data = src.read(); transform = src.transform
    window_size = 64; step = 16
    results = []
    h, w = data.shape[1], data.shape[2]
    for y in range(0, h-window_size, step):
        for x in range(0, w-window_size, step):
            window = data[:, y:y+window_size, x:x+window_size].astype(float)
            if window.shape[1] != window_size or window.shape[2] != window_size: continue
            means = np.mean(window, axis=(1, 2)); stds = np.std(window, axis=(1, 2))
            ndvi = (means[3]-means[2])/(means[3]+means[2]+1e-6)
            nbr = (means[3]-means[5])/(means[3]+means[5]+1e-6)
            si = means[4]/(means[3]+1e-6)
            feats = np.concatenate([means, stds, [ndvi, nbr, si]]).reshape(1, -1)
            prob = model.predict_proba(np.nan_to_num(feats))[0][1]
            cx, cy = rasterio.transform.xy(transform, y+window_size//2, x+window_size//2)
            results.append((cx, cy, prob))
    return pd.DataFrame(results, columns=['lon', 'lat', 'risk'])

def plot_forensic(df, fire_locs, date, output_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['lon'], df['lat'], c='lightgrey', s=5, alpha=0.2)
    risk_pts = df[df['risk'] > 0.25]
    sc = plt.scatter(risk_pts['lon'], risk_pts['lat'], c=risk_pts['risk'], cmap='YlOrRd', vmin=0.25, vmax=0.75, s=40, alpha=0.7)
    plt.colorbar(sc, label='Risk')
    if len(fire_locs) > 0:
        plt.scatter(fire_locs[:,0], fire_locs[:,1], c='blue', marker='x', s=150, linewidth=2, label='Actual Fire')
    plt.title(f"Detailed Forensic Overlay: {date}")
    plt.legend(); plt.savefig(output_path, dpi=200); plt.close()

def main():
    report = pd.read_csv('outputs/precision_analysis/precision_report.csv')
    top_dates = report.sort_values('error_meters').head(5)['date'].tolist()
    model = joblib.load('models/checkpoints/rf_fire_model.pkl')
    output_dir = "outputs/precision_analysis/top5_forensics"
    os.makedirs(output_dir, exist_ok=True)
    
    for date in top_dates:
        print(f"Generating forensic map for {date}...")
        tile_path = f"outputs/bulk_validation/tile_{date}.tif"
        if not os.path.exists(tile_path): continue
        fire_locs = get_actual_fires(date)
        df_pred = run_prediction_grid(tile_path, model)
        plot_forensic(df_pred, fire_locs, date, f"{output_dir}/forensic_{date}.png")

if __name__ == "__main__": main()
