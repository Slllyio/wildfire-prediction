"""
Fast KMZ Fire Prediction using Random Forest.
Uses pre-trained RF model on extracted spectral features.
"""
import rasterio
import numpy as np
import pandas as pd
import os
import joblib
import argparse
from datetime import datetime
import zipfile
import re
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt

def parse_kmz(kmz_path):
    print(f"Parsing KMZ: {kmz_path}")
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
                raise ValueError("No polygons found in KML")
            return unary_union(polygons)

def extract_features_from_data(data):
    """
    Extract features from a (Bands, H, W) numpy array.
    """
    # Reshape to (Time, Bands, H, W)
    n_channels = 6
    n_steps = data.shape[0] // n_channels
    
    if n_steps == 0:
        return None
    
    data = data.reshape(n_steps, n_channels, data.shape[1], data.shape[2])
    
    # 1. Temporal mean
    time_mean = np.mean(data, axis=0) # (6, H, W)
    
    # We need pixel-wise features for prediction?
    # NO, the training was done on PATCH-level statistics.
    # So we predict risk for the WHOLE TILE (or sliding window).
    # But user wants a grid of predictions.
    
    # Wait, simple RF trained on patch stats gives one prediction per patch.
    # To get pixel-wise map, we should use a sliding window or pixel-wise RF.
    # BUT, extracting 15 features per pixel is fast.
    
    # Let's check training script:
    # `band_means = np.mean(time_mean, axis=(1, 2))`
    # Training was on GLOBAL patch statistics.
    # So the model predicts "Does this 64x64 patch contain fire?"
    
    # We can apply this to sliding windows or just predict for the 5x5km tile?
    # 5km is too coarse.
    # We should run a sliding window of size ~64px (1.2km) over the tile.
    
    return time_mean

def predict_sliding_window(tile_path, model, window_size=64, step=32):
    """
    Run RF prediction on sliding windows of the tile.
    """
    results = []
    
    try:
        with rasterio.open(tile_path) as src:
            data = src.read()
            transform = src.transform
            width = src.width
            height = src.height
        
        # Precompute time mean to save time
        n_channels = 6
        n_steps = data.shape[0] // n_channels
        
        if n_steps == 0:
            return results
            
        data = data.reshape(n_steps, n_channels, height, width).astype(np.float32)
        # Handle nodata
        
        # Sliding window
        for y in range(0, height - window_size, step):
            for x in range(0, width - window_size, step):
                # Extract window
                window = data[:, :, y:y+window_size, x:x+window_size]
                
                # Compute features
                # 1. Time mean
                time_mean = np.mean(window, axis=0) # (6, 64, 64)
                
                # 2. Spatial mean/std
                band_means = np.mean(time_mean, axis=(1, 2))
                band_stds = np.std(time_mean, axis=(1, 2))
                
                # 3. Indices
                # B2, B3, B4, B8, B11, B12 -> 0,1,2,3,4,5
                ndvi_mean = (band_means[3] - band_means[2]) / (band_means[3] + band_means[2] + 1e-6)
                nbr_mean = (band_means[3] - band_means[5]) / (band_means[3] + band_means[5] + 1e-6)
                si_mean = band_means[4] / (band_means[3] + 1e-6)
                
                features = np.concatenate([band_means, band_stds, [ndvi_mean, nbr_mean, si_mean]])
                features = np.nan_to_num(features).reshape(1, -1)
                
                # Predict
                prob = model.predict_proba(features)[0][1]
                
                # Get center coordinate
                cx_px = x + window_size/2
                cy_px = y + window_size/2
                lon, lat = rasterio.transform.xy(transform, cy_px, cx_px)
                
                results.append((lon, lat, prob))
                
    except Exception as e:
        print(f"Error predicting {tile_path}: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kmz_path', help='Path to KMZ')
    parser.add_argument('--output-dir', default='outputs/rf_campaign', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Model
    model_path = 'models/checkpoints/rf_fire_model.pkl'
    print(f"Loading model: {model_path}")
    rf_model = joblib.load(model_path)
    
    # 2. Parse KMZ for bounds
    forest_poly = parse_kmz(args.kmz_path)
    
    # 3. Find tiles
    tile_dir = 'outputs/kmz_campaign/tiles' # Use existing downloaded tiles
    if not os.path.exists(tile_dir):
        # Fallback to v2 dir
        tile_dir = 'outputs/kmz_campaign_v2/tiles'
    
    print(f"Using tiles from: {tile_dir}")
    tile_files = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('.tif')]
    print(f"Found {len(tile_files)} tiles")
    
    # 4. Predict
    print("Running RF predictions (sliding window 1.2km)...")
    all_results = []
    
    for i, tile_path in enumerate(tile_files):
        # Quick intersection check (optional but speeds up)
        # Just process all for now
        res = predict_sliding_window(tile_path, rf_model)
        for r in res:
            # Filter by KMZ
            if forest_poly.contains(Point(r[0], r[1])):
                all_results.append(r)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(tile_files)} tiles... ({len(all_results)} points)")
            
    print(f"\nTotal points inside forest: {len(all_results)}")
    
    # 5. Save and Visualize
    if all_results:
        # Save CSV
        df = pd.DataFrame(all_results, columns=['lon', 'lat', 'predicted_risk'])
        csv_path = os.path.join(args.output_dir, 'rf_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Map
        lons = df['lon']
        lats = df['lat']
        risks = df['predicted_risk']
        
        plt.figure(figsize=(12, 10))
        plt.scatter(lons, lats, c=risks, cmap='RdYlGn_r', vmin=0, vmax=1, s=30, alpha=0.8, edgecolors='none')
        plt.colorbar(label='Fire Risk Probability')
        plt.title(f"Fire Risk Assessment (May 2025 Prediction)\nRandom Forest Model (Trained on 2022-2024 Data)")
        
        # Plot boundary
        if forest_poly.geom_type == 'Polygon':
            x, y = forest_poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=1)
        elif forest_poly.geom_type == 'MultiPolygon':
            for geom in forest_poly.geoms:
                x, y = geom.exterior.xy
                plt.plot(x, y, 'b-', linewidth=1)
                
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        map_path = os.path.join(args.output_dir, 'rf_map_preview.png')
        plt.savefig(map_path, dpi=200)
        print(f"Saved Map: {map_path}")
        
        # Stats
        print("\nRisk Statistics:")
        print(f"  Mean Risk: {risks.mean():.4f}")
        print(f"  Max Risk: {risks.max():.4f}")
        print(f"  High Risk Points (>0.5): {sum(risks > 0.5)}")
        print(f"  Very High Risk Points (>0.8): {sum(risks > 0.8)}")

if __name__ == "__main__":
    main()
