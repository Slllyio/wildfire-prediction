
import rasterio
import numpy as np

try:
    with rasterio.open('outputs/campaign/fire_risk_campaign_20250515_g5.tif') as src:
        data = src.read(1)
        print(f"Shape: {data.shape}")
        print(f"Min: {np.min(data)}, Max: {np.max(data)}")
        print(f"Mean: {np.mean(data)}")
        count = np.count_nonzero(data)
        print(f"Non-zero pixels: {count}")
except Exception as e:
    print(f"Error: {e}")
