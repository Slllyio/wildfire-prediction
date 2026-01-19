"""
Detailed Forensic Analysis of May 5, 2025 Fire.
Compares Predicted Risk Extent vs Actual Fire Clusters.
"""
import ee
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint
import geopandas as gpd
# import contextily as ctx
# from scripts.config import GEE_PROJECT_ID
GEE_PROJECT_ID = "van-suraksha-alert"

# Initialize GEE
try:
    ee.Initialize(project=GEE_PROJECT_ID)
except:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)

def get_actual_fires(date):
    """Fetch actual fire pixels for the date."""
    print(f"Fetching actual fires for {date}...")
    bounds = {'min_lon': 78.2449, 'max_lon': 79.0312, 'min_lat': 21.8803, 'max_lat': 22.7003}
    region = ee.Geometry.Rectangle([bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat']])
    
    start = date
    end = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    fires = ee.ImageCollection('FIRMS').filterDate(start, end).filterBounds(region)
    count = fires.size().getInfo()
    
    locations = []
    if count > 0:
        fire_img = fires.select('T21').max().gt(300).selfMask()
        vec = fire_img.reduceToVectors(geometry=region, scale=500, geometryType='centroid')
        feats = vec.getInfo()['features']
        for f in feats:
            locations.append(f['geometry']['coordinates'])
            
    print(f"  Found {len(locations)} fire pixels.")
    return np.array(locations)

def run_prediction_grid(tile_path, model_path):
    """Run prediction to get a dense grid."""
    print("Running prediction on tile...")
    model = joblib.load(model_path)
    
    with rasterio.open(tile_path) as src:
        data = src.read()
        transform = src.transform
        
    window_size = 64
    step = 16  # Finer step for detailed map (was 32)
    results = []
    
    h, w = data.shape[1], data.shape[2]
    
    for y in range(0, h-window_size, step):
        for x in range(0, w-window_size, step):
            window = data[:, y:y+window_size, x:x+window_size].astype(float)
            
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
            
    return pd.DataFrame(results, columns=['lon', 'lat', 'risk'])

def analyze_overlap(df, fire_locs):
    """Analyze spatial offset and overlap."""
    if len(fire_locs) == 0:
        return
        
    # 1. Define High Risk Area (Risk > 0.35) - Adaptive threshold
    # Since peak is 0.43, 0.35 captures the "core" risk area
    high_risk_pts = df[df['risk'] > 0.35]
    
    print(f"\nAnalysis Stats:")
    print(f"  Predicted At-Risk Area (>0.35): {len(high_risk_pts)} points")
    
    # 2. Fire Centroid
    fire_center = np.mean(fire_locs, axis=0)
    print(f"  Actual Fire Center: {fire_center[0]:.5f}, {fire_center[1]:.5f}")
    
    # 3. Predicted Center (Weighted by risk)
    if not high_risk_pts.empty:
        pred_center_lon = np.average(high_risk_pts['lon'], weights=high_risk_pts['risk'])
        pred_center_lat = np.average(high_risk_pts['lat'], weights=high_risk_pts['risk'])
        print(f"  Predicted Risk Center: {pred_center_lon:.5f}, {pred_center_lat:.5f}")
        
        # Distance
        from math import radians, cos, sin, asin, sqrt
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return c * 6371
            
        dist = haversine(fire_center[0], fire_center[1], pred_center_lon, pred_center_lat)
        print(f"  Centroid Offset: {dist:.2f} km")
    else:
        print("  No high risk points predicted > 0.35")

    return high_risk_pts

def plot_comparison(df, fire_locs, output_path):
    """Create a detailed comparison map."""
    plt.figure(figsize=(12, 10))
    
    # Plot predicted risk contour/heatmap
    # Use griddata to interpolate for smoother look? 
    # Scatter is fine for now
    
    # Background: Low risk points (grey/transparent)
    plt.scatter(df['lon'], df['lat'], c='lightgrey', s=10, alpha=0.3, label='Low Risk')
    
    # Risk Gradient
    risk_pts = df[df['risk'] > 0.25]
    sc = plt.scatter(risk_pts['lon'], risk_pts['lat'], c=risk_pts['risk'], 
                     cmap='YlOrRd', vmin=0.25, vmax=0.45, s=60, alpha=0.8, edgecolors='none', label='Predicted Risk')
    plt.colorbar(sc, label='Fire Probability')
    
    # Actual Fires
    if len(fire_locs) > 0:
        plt.scatter(fire_locs[:, 0], fire_locs[:, 1], c='blue', marker='x', s=150, linewidth=2, label='Actual Fire (MODIS)')
        
        # Draw Hull around fire
        if len(fire_locs) >= 3:
            hull = MultiPoint(fire_locs).convex_hull
            if hull.geom_type == 'Polygon':
                x, y = hull.exterior.xy
                plt.plot(x, y, 'b--', linewidth=1.5, alpha=0.7)
    
    plt.title("Forensic Analysis: May 5, 2025 Fire\nprediction (Heatmap) vs Reality (Blue Crosses)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison map saved to {output_path}")

def main():
    date = "2025-05-05"
    tile_path = "outputs/validation_2025-05-05/fire_tile.tif"
    model_path = "models/checkpoints/rf_fire_model.pkl"
    output_dir = "outputs/validation_2025-05-05"
    
    # 1. Get Actual
    fire_locs = get_actual_fires(date)
    
    # 2. Get Prediction
    df = run_prediction_grid(tile_path, model_path)
    
    # 3. Analyze
    analyze_overlap(df, fire_locs)
    
    # 4. Plot
    plot_comparison(df, fire_locs, f"{output_dir}/forensic_map_may5.png")

if __name__ == "__main__":
    main()
