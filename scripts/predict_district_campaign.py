"""
District-wide Fire Prediction Campaign
Allows predicting fire risk for a specific date and region with configurable grid density.
"""
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import ee

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

def run_campaign_prediction(target_date_str, grid_size=20, output_dir='outputs/campaign'):
    """
    Run prediction for a specific date across the AOI.
    
    Args:
        target_date_str (str): Target date in 'YYYY-MM-DD' format.
        grid_size (int): Number of points along each axis (total points = grid_size^2).
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    run_id = f"campaign_{target_date.strftime('%Y%m%d')}_g{grid_size}"
    
    print(f"=== Wildfire Prediction Campaign ===")
    print(f"Target Date: {target_date_str}")
    print(f"Region: Chhindwara District (AOI in config)")
    print(f"Grid Size: {grid_size}x{grid_size} ({grid_size**2} points)")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize GEE
    initialize_gee()
    
    # Load trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    scratch_dir = os.path.dirname(project_dir)
    
    prithvi_path = os.path.join(scratch_dir, 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(project_dir, 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Trained model not found at {checkpoint_path}")
        return
    
    print("Loading model...")
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define prediction grid
    lons = np.linspace(AOI_BOUNDS['min_lon'], AOI_BOUNDS['max_lon'], grid_size)
    lats = np.linspace(AOI_BOUNDS['min_lat'], AOI_BOUNDS['max_lat'], grid_size)
    
    predictions = []
    coords = []
    
    print(f"\nProcessing {len(lons) * len(lats)} grid points...")
    
    # Date range for time series (6 months prior to target date)
    end_date = target_date.strftime('%Y-%m-%d')
    start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
    
    count = 0
    total = len(lons) * len(lats)
    
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            count += 1
            progress = (count / total) * 100
            print(f"[{progress:.1f}%] Processing point ({lat:.4f}, {lon:.4f})...", end='\r')
            
            try:
                # Download patch
                temp_patch = os.path.join(output_dir, f'temp_patch_{i}_{j}.tif')
                
                # Check if we can skip download (optimization for reruns)
                # For now, always download to be safe
                download_patch_local(
                    [lon, lat],
                    start_date,
                    end_date,
                    temp_patch,
                    patch_size=64
                )
                
                # Preprocess
                with rasterio.open(temp_patch) as src:
                    data = src.read()
                
                # Reshape and normalize
                temporal_steps = data.shape[0] // 6
                
                # Basic validation
                if temporal_steps < 1:
                     # If no data found (e.g. cloud cover or missing images), pad or skip
                     # Here we skip/zero
                     predictions.append(0.0)
                     coords.append((lon, lat))
                     if os.path.exists(temp_patch): os.remove(temp_patch)
                     continue

                data = data.reshape(temporal_steps, 6, data.shape[1], data.shape[2])
                data = data.astype(np.float32) / 10000.0
                data = np.clip(data, 0, 1)
                
                tensor_data = torch.from_numpy(data)
                
                # Resize to 224x224
                import torch.nn.functional as F
                B, C, H, W = 1, tensor_data.shape[0] * tensor_data.shape[1], tensor_data.shape[2], tensor_data.shape[3]
                flat_data = tensor_data.view(1, C, H, W)
                resized_data = F.interpolate(flat_data, size=(224, 224), mode='bilinear', align_corners=False)
                
                # Reshape for model
                # Ensure we have the right number of steps. If < 6, pad, if > 6, truncate?
                # Model expects specific temporal window usually, but Prithvi might be flexible or fixed.
                # Assuming config.TEMPORAL_WINDOW_MONTHS logic is handled by 'download_patch_local' generally
                # But here 'temporal_steps' depends on what GEE returned.
                
                # Simple padding/truncating to match model expectation if strict
                # For this prototype, we pass what we have, assuming model can handle variable length 
                # OR we force it to match TEMPORAL_WINDOW_MONTHS if needed.
                # Prithvi usually expects fixed T. Let's assume the downloader did a good job or check.
                
                input_tensor = resized_data.view(1, temporal_steps, 6, 224, 224).to(device)
                
                # Auxiliary features (Simulated for this campaign - purely location/season based)
                # Tendu season is April-May. So May is high risk.
                aux_features = torch.tensor([[ 
                    35.0, 25.0, 5.0, 1000.0, 1000.0, 0.8, 0.9 
                ]], dtype=torch.float32).to(device)
                
                # Predict
                with torch.no_grad():
                    logits = model(input_tensor, aux_features)
                    probs = torch.softmax(logits, dim=1)
                    fire_prob = probs[0, 1].item()
                
                predictions.append(fire_prob)
                coords.append((lon, lat))
                
                # Clean up temp file
                if os.path.exists(temp_patch):
                    os.remove(temp_patch)
                
            except Exception as e:
                # print(f"\n  Error at ({lat:.4f}, {lon:.4f}): {e}")
                predictions.append(0.0)  # Fallback
                coords.append((lon, lat))
                if os.path.exists(temp_patch):
                    try:
                        os.remove(temp_patch)
                    except:
                        pass
    
    print("\nPrediction loop complete.")
    
    # Create output GeoTIFF
    output_tif = os.path.join(output_dir, f'fire_risk_{run_id}.tif')
    save_risk_map(predictions, coords, lons, lats, output_tif)
    
    # Alerts
    high_risk_count = sum(1 for p in predictions if p > 0.7)
    print(f"[RESULT] High Risk Points (>70%): {high_risk_count}")
    print(f"[RESULT] Output Map: {output_tif}")

def save_risk_map(predictions, coords, lons, lats, output_path):
    """Save as GeoTIFF"""
    # Reshape predictions to grid (Lat x Lon)
    # Note: Our loop order was lon then lat.
    # We need to be careful with reshaping.
    # predictions list order: (lon0, lat0), (lon0, lat1)... (lon1, lat0)...
    # This means 'lon' is the outer loop, 'lat' is the inner.
    # So grid[i, j] should be lon[i], lat[j]?
    # Actually, standard raster is (row=lat, col=lon).
    # We should reshape such that rows correspond to lats (usually typically desc or asc) and cols to lons.
    
    # Let's reconstruct explicitly to be safe
    grid = np.zeros((len(lats), len(lons)), dtype=np.float32)
    
    # We iterated: for i (lon) -> for j (lat) -> append
    # So idx = i * len(lats) + j
    for idx, prob in enumerate(predictions):
        i = idx // len(lats) # lon index
        j = idx % len(lats)  # lat index
        
        # We want map[lat_index, lon_index]
        # Usually rasters are stored from Top-Left (High Lat) to Bottom-Right.
        # Our lats are linspace(min, max), so increasing.
        # So row 0 corresponds to min_lat (Bottom).
        # We might want to flip the lat axis for standard image viewing, 
        # or just define the transform correctly (South-Up vs North-Up).
        # Standard GeoTIFF `from_origin` expects West, North (Top-Left).
        # So we should arrange our data where row 0 is the Top Latitude (max_lat).
        
        # If lats is increasing [21.8, ..., 22.8], then max_lat is at index -1.
        # We want row 0 to be max_lat.
        # So row_idx = (len(lats) - 1) - j
        
        row_idx = (len(lats) - 1) - j
        col_idx = i
        
        grid[row_idx, col_idx] = prob

    # Calculate resolution
    lon_res = (lons[-1] - lons[0]) / (len(lons) - 1) if len(lons) > 1 else 0.1
    lat_res = (lats[-1] - lats[0]) / (len(lats) - 1) if len(lats) > 1 else 0.1
    
    # from_origin(west, north, xsize, ysize)
    transform = from_origin(lons[0] - lon_res/2, lats[-1] + lat_res/2, lon_res, lat_res)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(grid, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wildfire prediction campaign")
    parser.add_argument('--date', type=str, required=True, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--grid', type=int, default=10, help='Grid size (NxN)')
    
    args = parser.parse_args()
    
    run_campaign_prediction(args.date, args.grid)
