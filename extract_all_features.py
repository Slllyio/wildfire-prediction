"""
Feature Extraction for Wildfire Training
Extracts features from:
1. HLS optical data (.tif files)
2. Sentinel-1 SAR data (from Earth Engine)
3. Combines with existing terrain/weather features
"""

import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import ee
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    ee.Initialize()
    print("✓ Earth Engine initialized")
except:
    print("⚠ Earth Engine not initialized - SAR features will be skipped")
    ee = None

def extract_optical_features(tif_path):
    """Extract statistical features from HLS .tif file"""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # Shape: (bands, height, width)
            
            # Calculate statistics across spatial dimensions
            features = {}
            for band_idx in range(min(6, data.shape[0])):  # First 6 bands
                band_data = data[band_idx].flatten()
                band_data = band_data[band_data > 0]  # Remove no-data values
                
                if len(band_data) > 0:
                    features[f'band{band_idx+1}_mean'] = np.mean(band_data)
                    features[f'band{band_idx+1}_std'] = np.std(band_data)
                    features[f'band{band_idx+1}_min'] = np.min(band_data)
                    features[f'band{band_idx+1}_max'] = np.max(band_data)
                else:
                    for stat in ['mean', 'std', 'min', 'max']:
                        features[f'band{band_idx+1}_{stat}'] = 0
                        
            return features
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return {f'band{i+1}_{stat}': 0 
                for i in range(6) 
                for stat in ['mean', 'std', 'min', 'max']}

def extract_sar_features(lat, lon, date_str):
    """Extract Sentinel-1 SAR features from Earth Engine"""
    if ee is None:
        return {'sar_vv_mean': 0, 'sar_vh_mean': 0, 'sar_vv_vh_ratio': 0}
    
    try:
        # Parse date
        from datetime import datetime, timedelta
        date = datetime.strptime(date_str, '%Y-%m-%d')
        start_date = (date - timedelta(days=3)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])
        
        # Get Sentinel-1 data
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(point)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
              .select(['VV', 'VH'])
              .mean())
        
        # Sample at point
        sample = s1.sample(point, 10).first().getInfo()
        
        if sample and 'properties' in sample:
            props = sample['properties']
            vv = props.get('VV', 0)
            vh = props.get('VH', 0)
            
            return {
                'sar_vv_mean': vv,
                'sar_vh_mean': vh,
                'sar_vv_vh_ratio': vv / vh if vh != 0 else 0
            }
    except Exception as e:
        print(f"SAR extraction error for {lat}, {lon}: {e}")
    
    return {'sar_vv_mean': 0, 'sar_vh_mean': 0, 'sar_vv_vh_ratio': 0}

def process_all_data(base_dir):
    """Process all training data and create feature CSV"""
    base_path = Path(base_dir)
    all_features = []
    
    # Find all year directories
    year_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"\n📊 Processing {len(year_dirs)} years of data...")
    
    for year_dir in tqdm(year_dirs, desc="Years"):
        labels_file = year_dir / 'labels.csv'
        if not labels_file.exists():
            continue
            
        # Load labels
        df = pd.read_csv(labels_file)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), 
                             desc=f"  {year_dir.name}", leave=False):
            
            # Get .tif path
            tif_path = year_dir / row['filename']
            if not tif_path.exists():
                continue
            
            # Extract optical features
            optical_features = extract_optical_features(tif_path)
            
            # Extract SAR features (if coordinates available)
            sar_features = {}
            if 'latitude' in row and 'longitude' in row and 'date' in row:
                sar_features = extract_sar_features(
                    row['latitude'], row['longitude'], row['date']
                )
            else:
                sar_features = {'sar_vv_mean': 0, 'sar_vh_mean': 0, 'sar_vv_vh_ratio': 0}
            
            # Combine all features
            sample = {
                'year': year_dir.name,
                'filename': row['filename'],
                'label': row['label'],
                **optical_features,
                **sar_features
            }
            
            # Add existing features from labels.csv
            for col in df.columns:
                if col not in ['filename', 'label']:
                    sample[col] = row.get(col, 0)
            
            all_features.append(sample)
    
    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    
    return features_df

if __name__ == '__main__':
    print("🔥 Wildfire Feature Extraction")
    print("=" * 50)
    
    base_dir = 'data/training'
    output_file = 'training_features_complete.csv'
    
    # Process all data
    df = process_all_data(base_dir)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print(f"✅ Feature extraction complete!")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns) - 3}")  # Exclude year, filename, label
    print(f"   Fire samples: {(df['label'] == 1).sum()}")
    print(f"   Non-fire samples: {(df['label'] == 0).sum()}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print("\n🚀 Ready to upload to Colab!")
