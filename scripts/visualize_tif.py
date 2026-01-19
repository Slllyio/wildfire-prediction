
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def visualize_risk_map(tif_path, output_png=None):
    if not output_png:
        output_png = tif_path.replace('.tif', '_preview.png')
        
    print(f"Visualizing {tif_path}...")
    
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=1) # Red (High) to Green (Low)
            plt.colorbar(label='Fire Risk Probability')
            plt.title(f'Fire Risk Map: {os.path.basename(tif_path)}')
            
            # Save
            plt.savefig(output_png, dpi=300, bbox_inches='tight')
            print(f"Preview saved to: {output_png}")
            plt.close()
            
    except Exception as e:
        print(f"Error visualizing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to .tif file')
    args = parser.parse_args()
    
    visualize_risk_map(args.file)
