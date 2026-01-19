
"""
Single Point Prediction (Fast)
"""
import torch
import numpy as np
import rasterio
import os
import sys
import argparse
from datetime import datetime, timedelta
import ee

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

def predict_point(date_str, lat, lon):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"Predicting for {lat}, {lon} on {date_str}")
    
    initialize_gee()
    
    # Load Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    scratch_dir = os.path.dirname(project_dir)
    prithvi_path = os.path.join(scratch_dir, 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(project_dir, 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Data Prep
    end_date = target_date.strftime('%Y-%m-%d')
    start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
    
    temp_path = os.path.join("outputs", "temp_point.tif")
    
    try:
        download_patch_local([lon, lat], start_date, end_date, temp_path, patch_size=64)
        
        with rasterio.open(temp_path) as src:
            data = src.read()
            
        temporal_steps = data.shape[0] // 6
        if temporal_steps == 0:
            print("No data found.")
            return

        data = data.reshape(temporal_steps, 6, data.shape[1], data.shape[2])
        data = data.astype(np.float32) / 10000.0
        data = np.clip(data, 0, 1)
        
        tensor_data = torch.from_numpy(data)
        import torch.nn.functional as F
        flat = tensor_data.view(1, -1, tensor_data.shape[2], tensor_data.shape[3])
        resized = F.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
        input_tensor = resized.view(1, temporal_steps, 6, 224, 224).to(device)
        
        aux = torch.tensor([[35.0, 25.0, 5.0, 1000.0, 1000.0, 0.8, 0.9]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor, aux)
            probs = torch.softmax(logits, dim=1)
            risk = probs[0, 1].item()
            
        print(f"Risk: {risk*100:.2f}%")
        
        # Save result
        with open("outputs/single_point_result.txt", "w") as f:
            f.write(f"Date: {date_str}\nLocation: {lat}, {lon}\nRisk: {risk*100:.2f}%\n")
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--lat', type=float, default=22.15)
    parser.add_argument('--lon', type=float, default=78.85)
    args = parser.parse_args()
    predict_point(args.date, args.lat, args.lon)
