"""
Quick test of 3-tier labeling with just 3 samples
"""
import pandas as pd
import ee
from datetime import datetime, timedelta
from pathlib import Path
import math

# Initialize
ee.Initialize(project='van-suraksha-alert')

def get_modis_fire_truth(lat, lon, date_str):
    """Check if VIIRS point coincides with MODIS Burnt Area"""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=3)).strftime('%Y-%m-%d')
        
        point = ee.Geometry.Point([lon, lat])
        look_region = point.buffer(2000) 
        
        fire_col = (ee.ImageCollection('MODIS/061/MCD64A1')
                    .filterDate(start_date, end_date)
                    .filterBounds(look_region))
        
        fire_img = fire_col.first()
        if fire_img:
            burnt_mask = fire_img.select('BurnDate').gt(0)
            vectors = burnt_mask.reduceToVectors(geometry=look_region, scale=500, maxPixels=1e9)
            target_poly = vectors.filterBounds(point.buffer(200)).first()
            
            if target_poly:
                geom = target_poly.geometry()
                centroid = geom.centroid()
                stats = geom.area().getInfo()
                burnt_area_ha = stats / 10000
                
                cent_coords = centroid.coordinates().getInfo()
                d_lon = cent_coords[0] - lon
                d_lat = cent_coords[1] - lat
                angle = math.atan2(d_lat, d_lon)
                
                return {
                    'geometry': geom,
                    'is_modis_verified': 1,
                    'fire_ros_proxy': math.sqrt(stats),
                    'fire_spread_dir': angle,
                    'fire_burnt_area_ha': burnt_area_ha
                }
    except Exception as e:
        pass
    
    return {
        'geometry': ee.Geometry.Point([lon, lat]).buffer(200),
        'is_modis_verified': 0,
        'fire_ros_proxy': float('nan'),
        'fire_spread_dir': float('nan'),
        'fire_burnt_area_ha': 12.5
    }

print("🔥 Quick Test: 3-Tier Labeling System")
print("="*60)

# Test samples
base_path = Path('C:/Users/dfogu/.gemini/antigravity/scratch/wildfire_v2/data/training')

# Get 1 fire and 1 no-fire from 2019
year_dir = base_path / '2019'
df = pd.read_csv(year_dir / 'labels.csv')

# Get one fire and one no-fire
fire_sample = df[df['label'] == 1].iloc[0]
nofire_sample = df[df['label'] == 0].iloc[0]

results = []

print("\nProcessing samples...")
for idx, (row, sample_type) in enumerate([(fire_sample, 'Fire'), (nofire_sample, 'No Fire')]):
    print(f"\n{idx+1}. {sample_type} sample: {row['filename']}")
    
    lat, lon = row.get('latitude', 22.5), row.get('longitude', 78.5)
    date_str = '2019-03-15'
    
    if row['label'] == 1:
        # Fire sample - check MODIS verification
        print("   Checking MODIS verification...")
        truth = get_modis_fire_truth(lat, lon, date_str)
        # 3-tier labeling: 2=MODIS verified, 1=VIIRS only
        final_label = 2 if truth['is_modis_verified'] == 1 else 1
        print(f"   MODIS verified: {truth['is_modis_verified']}")
        print(f"   Final label: {final_label}")
    else:
        # No fire sample
        truth = {
            'geometry': ee.Geometry.Point([lon, lat]).buffer(200),
            'is_modis_verified': 0, 'fire_ros_proxy': 0.0, 
            'fire_spread_dir': 0.0, 'fire_burnt_area_ha': 0.0
        }
        final_label = 0
        print(f"   Final label: {final_label}")
    
    results.append({
        'filename': row['filename'],
        'original_label': row['label'],
        'final_label': final_label,
        'is_modis_verified': truth['is_modis_verified'],
        'burnt_area_ha': truth['fire_burnt_area_ha']
    })

# Summary
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

print("\n✅ 3-Tier Labeling Test Complete!")
print(f"\nLabel Distribution:")
print(f"  label=0 (No Fire): {(df_results['final_label']==0).sum()}")
print(f"  label=1 (VIIRS only): {(df_results['final_label']==1).sum()}")
print(f"  label=2 (MODIS verified): {(df_results['final_label']==2).sum()}")
