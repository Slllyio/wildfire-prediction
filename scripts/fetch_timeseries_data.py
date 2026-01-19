"""
Fetch Sentinel-2 patches for generated samples.
"""
import sys
import os
import json
import geopandas as gpd
from tqdm import tqdm
import ee
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from scripts.gee_utils import *

def fetch_data_for_year(year, limit=100, output_dir=None):
    """Fetch image patches for a specific year's samples"""
    initialize_gee()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', 'patches')
    
    sample_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', f'samples_{year}.geojson')
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        return
        
    samples = gpd.read_file(sample_file)
    print(f"Loaded {len(samples)} samples for {year}")
    
    # Create output dir
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    
    # Fetch subset for prototype (limit to fewer for testing download speed)
    subset = samples.sample(min(limit, len(samples)), random_state=42)
    
    success_count = 0
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Fetching {year}"):
        coords = [row.geometry.x, row.geometry.y]
        label = row.label
        
        # Output filename
        patch_file = os.path.join(year_dir, f"patch_{idx}_L{label}.tif")
        if os.path.exists(patch_file):
            success_count += 1
            continue
            
        # Time range: 6 months before May (peak fire)
        start_date = f"{year}-01-01"
        end_date = f"{year}-06-30"
        
        try:
            download_patch_local(coords, start_date, end_date, patch_file, patch_size=64)
            success_count += 1
        except Exception as e:
            print(f"Error fetching sample {idx}: {e}")
            pass
            
    print(f"Year {year} finished: {success_count}/{len(subset)} successful")

if __name__ == '__main__':
    # Phase 1: Get 10 samples per year for 2018, 2021, 2024, 2025 (Val data fast)
    print("=== PHASE 1: Fetching validation subset (10 per year) ===")
    for year in [2018, 2021, 2024, 2025]:
        fetch_data_for_year(year, limit=10)
        
    # Phase 2: Get more for training (2018, 2021, 2024)
    print("\n=== PHASE 2: Fetching training data (additional 40 per year) ===")
    for year in [2018, 2021, 2024]:
        fetch_data_for_year(year, limit=50)


