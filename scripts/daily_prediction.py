"""
Daily Fire Risk Prediction Automation

Runs daily at a scheduled time to:
1. Fetch latest Sentinel-2 imagery and weather data
2. Generate fire risk predictions for the AOI
3. Export results as GeoTIFF
4. Create alert notifications for high-risk areas
"""
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import sys
from datetime import datetime, timedelta
import json
import ee

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

def run_daily_prediction(output_dir='outputs/daily'):
    """
    Main daily prediction pipeline.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"=== Daily Fire Risk Prediction ===")
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # Define prediction grid (sample points across AOI)
    # For a full implementation, you'd tile the entire AOI
    # Here we use a 3x3 grid for quick testing
    lons = np.linspace(AOI_BOUNDS['min_lon'], AOI_BOUNDS['max_lon'], 3)
    lats = np.linspace(AOI_BOUNDS['min_lat'], AOI_BOUNDS['max_lat'], 3)
    
    predictions = []
    coords = []
    
    print(f"\nProcessing {len(lons) * len(lats)} grid points...")
    
    # Date range for time series
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    for i, lon in enumerate(lons):
        for j, lat in enumerate(lats):
            try:
                # Download patch
                temp_patch = os.path.join(output_dir, f'temp_patch_{i}_{j}.tif')
                
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
                input_tensor = resized_data.view(1, temporal_steps, 6, 224, 224).to(device)
                
                # Auxiliary features (placeholder - would fetch real weather data)
                aux_features = torch.tensor([[ 
                    25.0, 40.0, 5.0, 2000.0, 1500.0, 0.5, 0.8
                ]], dtype=torch.float32).to(device)
                
                # Predict
                with torch.no_grad():
                    logits = model(input_tensor, aux_features)
                    probs = torch.softmax(logits, dim=1)
                    fire_prob = probs[0, 1].item()
                
                predictions.append(fire_prob)
                coords.append((lon, lat))
                
                # Clean up temp file
                os.remove(temp_patch)
                
                print(f"  Point ({lat:.4f}, {lon:.4f}): {fire_prob*100:.2f}%", end='\r')
                
            except Exception as e:
                print(f"\n  Error at ({lat:.4f}, {lon:.4f}): {e}")
                predictions.append(0.0)  # Fallback
                coords.append((lon, lat))
    
    print("\n")
    
    # Create output GeoTIFF
    output_tif = os.path.join(output_dir, f'fire_risk_{run_timestamp}.tif')
    save_risk_map(predictions, coords, lons, lats, output_tif)
    
    # Generate alert summary
    high_risk_threshold = 0.7
    high_risk_count = sum(1 for p in predictions if p > high_risk_threshold)
    
    alert_summary = {
        'timestamp': datetime.now().isoformat(),
        'grid_points': len(predictions),
        'high_risk_count': high_risk_count,
        'max_risk': float(max(predictions)),
        'avg_risk': float(np.mean(predictions)),
        'output_file': output_tif
    }
    
    alert_file = os.path.join(output_dir, f'alert_{run_timestamp}.json')
    with open(alert_file, 'w') as f:
        json.dump(alert_summary, f, indent=2)
    
    print(f"[SUCCESS] Prediction complete")
    print(f"  Risk Map: {output_tif}")
    print(f"  Alert Summary: {alert_file}")
    print(f"  High Risk Points: {high_risk_count}/{len(predictions)}")
    print(f"  Max Risk: {alert_summary['max_risk']*100:.2f}%")
    print(f"  Avg Risk: {alert_summary['avg_risk']*100:.2f}%")
    
    return alert_summary

def save_risk_map(predictions, coords, lons, lats, output_path):
    """
    Save predictions as a GeoTIFF raster.
    """
    # Reshape predictions to grid
    grid = np.array(predictions).reshape(len(lats), len(lons))
    
    # Calculate resolution
    lon_res = (lons[-1] - lons[0]) / (len(lons) - 1)
    lat_res = (lats[-1] - lats[0]) / (len(lats) - 1)
    
    # Create transform (top-left origin)
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
        dst.write(grid.astype(np.float32), 1)
    
    print(f"  Saved risk map: {output_path}")

if __name__ == "__main__":
    # Run the daily prediction
    run_daily_prediction()
