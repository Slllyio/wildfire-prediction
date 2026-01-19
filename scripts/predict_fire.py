"""
Daily Prediction Script for Wildfire Risk
Generates a 20m resolution fire risk map using the fine-tuned Prithvi-EO model.
"""
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import from_origin
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import ee

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import get_timeseries_patch, initialize_gee, download_patch_local

def get_latest_weather(region):
    """Fetch GFS weather forecast for the region"""
    # GFS 0.25 Degree 6-Hourly
    gfs = ee.ImageCollection('NOAA/GFS0P25') \
        .filterBounds(region) \
        .sort('system:time_start', False) \
        .first()
    
    # Scale: Temperature (K), Humidity (%), Wind (m/s)
    # We select key bands: 
    # - temperature_2m_above_ground
    # - relative_humidity_2m_above_ground
    # - u_component_of_wind_10m_above_ground
    # - v_component_of_wind_10m_above_ground
    
    weather = gfs.select([
        'temperature_2m_above_ground', 
        'relative_humidity_2m_above_ground',
        'u_component_of_wind_10m_above_ground',
        'v_component_of_wind_10m_above_ground'
    ])
    
    return weather

def predict_aoi(model, device):
    """Run inference over the entire AOI"""
    initialize_gee()
    
    # Define AOI geometry
    region = ee.Geometry.Rectangle([
        AOI_BOUNDS['min_lon'], AOI_BOUNDS['min_lat'],
        AOI_BOUNDS['max_lon'], AOI_BOUNDS['max_lat']
    ])
    
    print("Fetching meteorological forecast...")
    weather = get_latest_weather(region)
    
    print(f"Fetching latest Sentinel-2 time series for AOI...")
    # Full AOI implementation would use Export.image.toDrive or local tiling.
    
    print("Ready to generate fire risk map.")
    
    # --- PROTOTYPE: Single Point Inference ---
    # Fetch a single patch to demonstrate the pipeline
    center_lon = (AOI_BOUNDS['min_lon'] + AOI_BOUNDS['max_lon']) / 2
    center_lat = (AOI_BOUNDS['min_lat'] + AOI_BOUNDS['max_lat']) / 2
    
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching sample patch for ({center_lat:.4f}, {center_lon:.4f})...")
    # Using a temp file for the patch
    temp_patch_path = os.path.join("data", "temp_prediction_patch.tif")
    
    try:
        # 1. Download Patch
        # Using a small buffer for the center point -> ~1km box approx
        # patch_size=64 at 20m resolution is ~1.28km x 1.28km
        # We need to ensure we download enough to resize to 224 without too much upsampling artifacting
        # but the model was trained on resized patches, so we follow that.
        download_patch_local(
            [center_lon, center_lat], 
            start_date, 
            end_date, 
            temp_patch_path, 
            patch_size=64
        )
        
        # 2. Preprocess
        with rasterio.open(temp_patch_path) as src:
            data = src.read() # (36, H, W)
            
        print(f"Patch retrieved. Shape: {data.shape}")
        
        # Reshape to (T, C, H, W)
        temporal_steps = data.shape[0] // 6
        if temporal_steps != TEMPORAL_WINDOW_MONTHS:
             # Handle cases where GEE returns different band counts (though download_patch_local handles padding)
             # If we have 36 bands, that's 6 steps.
             pass
             
        data = data.reshape(temporal_steps, 6, data.shape[1], data.shape[2])
        
        # Normalize
        data = data.astype(np.float32) / 10000.0
        data = np.clip(data, 0, 1)
        
        tensor_data = torch.from_numpy(data) # (T, C, H, W)
        
        # Resize to 224x224 (B=1)
        B, C, H, W = 1, tensor_data.shape[0] * tensor_data.shape[1], tensor_data.shape[2], tensor_data.shape[3]
        flat_data = tensor_data.view(1, C, H, W)
        
        import torch.nn.functional as F
        resized_data = F.interpolate(
            flat_data, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape for model: (B, T, C, H, W)
        # Model expects batch dim
        input_tensor = resized_data.view(1, temporal_steps, 6, 224, 224).to(device)
        
        # Auxiliary features (Placeholder for prototype)
        # Using the same dummy features as dataset.py for now
        # Ideally, we extract these from 'weather' and location
        aux_features = torch.tensor([[
            25.0, 40.0, 5.0, 2000.0, 1500.0, 0.5, 0.8
        ]], dtype=torch.float32).to(device)
        
        # 3. Model Inference
        print("Running inference...")
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor, aux_features)
            probs = torch.softmax(logits, dim=1)
            fire_prob = probs[0, 1].item()
            
        print(f"\n[PREDICTION] Fire Probability for AOI Center: {fire_prob*100:.2f}%")
        
        if fire_prob > 0.5:
            print("ALERT: High probability of fire detection!")
        else:
            print("Status: Low risk.")
            
        # 4. Export Result
        output_tif = os.path.join(project_dir, 'fire_risk_center.tif')
        save_risk_geotiff(fire_prob, center_lat, center_lon, output_tif)
        print(f"Risk map saved to: {output_tif}")

    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        import traceback
        traceback.print_exc()

def save_risk_geotiff(probability, lat, lon, output_path):
    """Save the prediction as a 1-pixel GeoTIFF (Prototype for full map)"""
    # 20m resolution in degrees approx 0.0002
    res = 0.0002
    transform = from_origin(lon - res/2, lat + res/2, res, res)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=1,
        width=1,
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(np.array([[probability]], dtype=np.float32), 1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    # We must point to where Prithvi-EO 2.0 folder is
    # Model is in ../../models/prithvi-eo-2.0 relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir) # wildfire_v2
    scratch_dir = os.path.dirname(project_dir) # scratch
    
    prithvi_path = os.path.join(scratch_dir, 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(project_dir, 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Trained model not found at {checkpoint_path}. Please finish training first.")
        # For dev, if no checkpoint, maybe skip? But user wants to continue.
        # sys.exit(1)
        
    print("Initializing predictor...")
    # Initialize the wrapper
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    
    # Load our fine-tuned weights
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("WARNING: Using pre-trained weights only (no fine-tuning checkpoint found).")

    model.to(device)
    
    predict_aoi(model, device)
