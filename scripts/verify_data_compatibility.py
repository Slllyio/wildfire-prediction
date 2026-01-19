"""
Verify Data Compatibility for Prithvi-EO
Loads a generated sample and runs a dummy forward pass through the model.
"""
import torch
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dataset import FireDataset, collate_fn
from models.prithvi_fire import PrithviFirePredictor

def verify_compatibility(data_dir):
    print(f"Checking data in: {data_dir}")
    labels_file = os.path.join(data_dir, 'labels.csv')
    
    if not os.path.exists(labels_file):
        print("ERROR: labels.csv not found")
        return
        
    df = pd.read_csv(labels_file)
    print(f"Found {len(df)} samples")
    
    # Check features
    needed_cols = [
        'elevation', 'slope', 'aspect',
        'era5_temp_d1', 'era5_wind_speed_d1', 'era5_precip_d1', 'era5_vpd_d1', 'era5_soil_moisture_d1',
        'modis_ndvi_d1', 'modis_evi_d1'
    ]
    
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        return
    else:
        print("[OK] All environmental feature columns present.")

    # Prepare dataset
    paths = [os.path.join(data_dir, f) for f in df['filename']]
    labels = df['label'].values
    aux = df[needed_cols].values.astype(np.float32)
    
    dataset = FireDataset(paths, labels, aux)
    
    # Test loading one Item
    try:
        img, aux_t, label = dataset[0]
        print(f"\nSample 0 stats:")
        print(f"  Image shape: {img.shape} (Time, Bands, H, W)")
        print(f"  Aux shape: {aux_t.shape}")
        print(f"  Label: {label}")
        
        # Expected: (T, 6, 224, 224)
        if img.shape[1] != 6 or img.shape[2] != 224 or img.shape[3] != 224:
            print("WARNING: Image shape might be incorrect for Prithvi! Expected (?, 6, 224, 224)")
            
    except Exception as e:
        print(f"ERROR loading sample: {e}")
        return

    # Test Model Forward Pass
    print("\nInitializing Prithvi Model (Dry Run)...")
    try:
        # Load model with correct num_aux
        # Note: We need the config path usually, but let's try purely shape-based
        # We assume the user has the model weights or we run without loading weights just for shape check
        
        # Point to dummy path if real one not found, we just want class init
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        prithvi_path = os.path.join(project_dir, 'models', 'prithvi-eo-2.0')
        
        model = PrithviFirePredictor(prithvi_model_path=prithvi_path, num_aux_features=len(needed_cols))
        
        # Batch of 2
        dl = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        imgs, auxs, lbls = next(iter(dl))
        
        print(f"Batch shapes: Img {imgs.shape}, Aux {auxs.shape}")
        
        # Forward
        print("Running forward pass...")
        with torch.no_grad():
            logits = model(imgs, auxs)
        
        print(f"Output shape: {logits.shape}")
        print("[OK] Forward pass successful!")
        
    except Exception as e:
        print(f"ERROR during model check: {e}")
        # Typical error might be missing model weights file, which is fine for data check, 
        # but verifies code logic.

if __name__ == "__main__":
    # Check test_gen or training_new
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        d = "data/test_gen" 
        
    verify_compatibility(d)
