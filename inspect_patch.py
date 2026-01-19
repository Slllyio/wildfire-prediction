
import rasterio
import numpy as np
import os

patch_path = 'outputs/temp_point.tif'
if os.path.exists(patch_path):
    with rasterio.open(patch_path) as src:
        data = src.read()
        print(f"File: {patch_path}")
        print(f"Shape: {data.shape}")
        print(f"Bands: {src.count}")
        print(f"Min: {np.min(data)}, Max: {np.max(data)}")
        
        # Check if empty (all zeros)
        if np.max(data) == 0:
            print("WARNING: Data is all zeros")
        else:
            print("Data content looks valid (non-zero).")
            
        # Check temporal steps logic
        t_steps = data.shape[0] // 6
        print(f"Implied Temporal Steps: {t_steps}")
else:
    print(f"File not found: {patch_path}")
