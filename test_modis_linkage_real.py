"""
Test MODIS linkage with REAL fire coordinates from fire_dynamics_analysis.csv
"""
import pandas as pd
import ee
from tqdm import tqdm
from datetime import datetime, timedelta
import math

# Initialize
ee.Initialize(project='monkhub-internal-enetra-dev')

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
                    'is_modis_verified': 1,
                    'fire_ros_proxy': math.sqrt(stats),
                    'fire_spread_dir': angle,
                    'fire_burnt_area_ha': burnt_area_ha
                }
    except Exception as e:
        print(f"    Error: {e}")
        pass
    
    return {
        'is_modis_verified': 0,
        'fire_ros_proxy': float('nan'),
        'fire_spread_dir': float('nan'),
        'fire_burnt_area_ha': 12.5
    }

# Load REAL fire data
fire_csv = 'c:/Users/dfogu/.gemini/antigravity/scratch/wildfire_v2/outputs/fire_dynamics_analysis.csv'
df = pd.read_csv(fire_csv)

# Take 20 samples
fires = df.head(20)

print(f"🔥 Testing MODIS linkage for {len(fires)} REAL fire samples...\n")

results = []
for idx, row in tqdm(fires.iterrows(), total=len(fires), desc="Processing"):
    lat = row['latitude']
    lon = row['longitude']
    date_str = row['fire_date']
    
    truth = get_modis_fire_truth(lat, lon, date_str)
    
    results.append({
        'fire_date': date_str,
        'lat': lat,
        'lon': lon,
        **truth
    })

# Summary
df_results = pd.DataFrame(results)
verified = (df_results['is_modis_verified']==1).sum()

print(f"\n{'='*60}")
print(f"RESULTS: MODIS Linkage Test")
print(f"{'='*60}")
print(f"Total fire samples tested: {len(results)}")
print(f"MODIS-verified fires: {verified}")
print(f"Verification rate: {100*verified/len(results):.1f}%")

if verified > 0:
    print(f"\n✅ MODIS linkage IS working!")
    print(f"\nVerified fire statistics:")
    verified_df = df_results[df_results['is_modis_verified']==1]
    print(f"  - Average burnt area: {verified_df['fire_burnt_area_ha'].mean():.1f} hectares")
    print(f"  - Min burnt area: {verified_df['fire_burnt_area_ha'].min():.1f} ha")
    print(f"  - Max burnt area: {verified_df['fire_burnt_area_ha'].max():.1f} ha")
    print(f"\n📝 Sample verified fires:")
    print(verified_df[['fire_date', 'lat', 'lon', 'fire_burnt_area_ha']].to_string())
else:
    print("\n⚠️ No MODIS matches found")
    print("Possible reasons:")
    print("  1. Fires are too small (< 0.5 ha) for MODIS 500m resolution")
    print("  2. MODIS temporal coverage issue")
    print("  3. Fire coordinates are outside MODIS coverage area")

# Save results
df_results.to_csv('modis_linkage_test_results.csv', index=False)
print(f"\n💾 Results saved to: modis_linkage_test_results.csv")
