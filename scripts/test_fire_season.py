"""
Fire Season Test - March-May 2025

Tests the model's predictions during the actual fire season (March-May 2025)
to validate if it can detect elevated risk during the critical period.
"""
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

def test_fire_season_2025():
    """
    Test predictions for fire season 2025 (March-May).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Fire Season Test (March-May 2025) ===")
    print(f"Device: {device}\n")
    
    # Initialize GEE
    initialize_gee()
    
    # Load trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    scratch_dir = os.path.dirname(project_dir)
    
    prithvi_path = os.path.join(scratch_dir, 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(project_dir, 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    print("Loading model...")
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Test different dates during fire season
    test_dates = [
        ('2025-03-15', 'Early Season (Mid-March)'),
        ('2025-04-15', 'Peak Season (Mid-April)'),
        ('2025-05-15', 'Late Season (Mid-May)')
    ]
    
    # Test center point of AOI
    center_lon = (AOI_BOUNDS['min_lon'] + AOI_BOUNDS['max_lon']) / 2
    center_lat = (AOI_BOUNDS['min_lat'] + AOI_BOUNDS['max_lat']) / 2
    
    results = []
    
    for target_date, description in test_dates:
        print(f"\n--- Testing: {description} ({target_date}) ---")
        
        # Calculate date range (6 months before target date)
        from datetime import datetime, timedelta
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        start_dt = target_dt - timedelta(days=180)
        
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = target_date
        
        print(f"  Fetching imagery: {start_date} to {end_date}")
        
        # Download patch
        temp_patch = os.path.join('outputs', f'test_patch_{target_date}.tif')
        os.makedirs('outputs', exist_ok=True)
        
        try:
            download_patch_local(
                [center_lon, center_lat],
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
            
            # Auxiliary features for fire season
            # Temperature: Higher in April (~35°C vs 25°C baseline)
            # Humidity: Lower in April (~25% vs 40% baseline)
            # Wind: Moderate increase (~7 m/s vs 5 m/s)
            if 'April' in description:
                temp, rh, wind = 35.0, 25.0, 7.0
            elif 'May' in description:
                temp, rh, wind = 38.0, 20.0, 8.0
            else:  # March
                temp, rh, wind = 32.0, 30.0, 6.0
            
            # Calculate day of year
            doy = target_dt.timetuple().tm_yday
            sin_doy = np.sin(2 * np.pi * doy / 365)
            cos_doy = np.cos(2 * np.pi * doy / 365)
            
            aux_features = torch.tensor([[
                temp, rh, wind, 2000.0, 1500.0, sin_doy, cos_doy
            ]], dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(input_tensor, aux_features)
                probs = torch.softmax(logits, dim=1)
                fire_prob = probs[0, 1].item()
            
            print(f"  Weather: Temp={temp}°C, RH={rh}%, Wind={wind} m/s")
            print(f"  🔥 Fire Probability: {fire_prob*100:.2f}%")
            
            if fire_prob > 0.5:
                print(f"  ⚠️  HIGH RISK ALERT")
            elif fire_prob > 0.3:
                print(f"  ⚠️  MODERATE RISK")
            else:
                print(f"  ✓ Low Risk")
            
            results.append({
                'date': target_date,
                'description': description,
                'fire_probability': float(fire_prob),
                'temp': temp,
                'humidity': rh,
                'wind': wind
            })
            
            # Clean up
            os.remove(temp_patch)
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'date': target_date,
                'description': description,
                'error': str(e)
            })
    
    # Save results
    output_file = 'outputs/fire_season_test_2025.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Test complete. Results saved to: {output_file}")
    
    # Summary
    print("\n=== SUMMARY ===")
    for r in results:
        if 'error' not in r:
            risk_level = "HIGH" if r['fire_probability'] > 0.5 else "MODERATE" if r['fire_probability'] > 0.3 else "LOW"
            print(f"{r['description']:30s} {r['fire_probability']*100:6.2f}%  [{risk_level}]")

if __name__ == "__main__":
    test_fire_season_2025()
