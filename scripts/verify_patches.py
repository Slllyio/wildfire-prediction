"""
Verify the contents of the downloaded training patches.
"""
import rasterio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def verify_patch(patch_path):
    print(f"Inspecting patch: {patch_path}")
    with rasterio.open(patch_path) as src:
        data = src.read()
        profile = src.profile
        
    print(f"  Shape: {data.shape}")
    print(f"  Bands: {profile['count']}")
    
    # Check for 36 bands (6 months * 6 bands)
    if data.shape[0] != 36:
        print(f"  [WARNING] Expected 36 bands, found {data.shape[0]}")
    
    # Reshape and check stats
    data_reshaped = data.reshape(6, 6, data.shape[1], data.shape[2])
    
    for t in range(6):
        month_data = data_reshaped[t]
        print(f"  Month {t} stats:")
        for b in range(6):
            band_data = month_data[b]
            print(f"    Band {b}: min={band_data.min():.1f}, max={band_data.max():.1f}, mean={band_data.mean():.1f}")
            
    # Visualize the first month RGB
    # Sentinel-2 RGB: B4, B3, B2 (Indices 2, 1, 0)
    # Scaled 0-2000 for display
    rgb = data_reshaped[0, [2, 1, 0]].transpose(1, 2, 0)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.title(f"RGB Preview - {os.path.basename(patch_path)}")
    plt.axis('off')
    
    output_preview = patch_path.replace('.tif', '_preview.png')
    plt.savefig(output_preview)
    print(f"  [OK] Saved preview to {output_preview}")

if __name__ == '__main__':
    search_path = 'data/training/patches/2018/*.tif'
    files = glob.glob(search_path)
    if files:
        verify_patch(files[0])
    else:
        print("No patches found to verify yet.")
