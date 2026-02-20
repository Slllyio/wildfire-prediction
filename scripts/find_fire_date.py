
import ee
from datetime import datetime, timedelta

# Hardcoded project ID
GEE_PROJECT_ID = "monkhub-internal-enetra-dev"

try:
    ee.Initialize(project=GEE_PROJECT_ID)
    print(f"[OK] GEE initialized: {GEE_PROJECT_ID}")
except:
    print("Authenticating...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)

bounds = {'min_lon': 78.2449, 'max_lon': 79.0312, 'min_lat': 21.8803, 'max_lat': 22.7003}
region = ee.Geometry.Rectangle([bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat']])

# Search period: May 2025
start_date = datetime(2025, 5, 1)
end_date = datetime(2025, 5, 31)

print(f"Scanning for fires in West Chhindwara: {start_date.date()} to {end_date.date()}...")

current = start_date
max_fires = 0
best_date = None

while current <= end_date:
    d_str = current.strftime("%Y-%m-%d")
    next_d = (current + timedelta(days=1)).strftime("%Y-%m-%d")
    
    fires = ee.ImageCollection('FIRMS').filterDate(d_str, next_d).filterBounds(region)
    count = fires.size().getInfo()
    
    if count > 0:
        fire_img = fires.select('T21').max().gt(300).selfMask()
        stats = fire_img.reduceRegion(reducer=ee.Reducer.count(), geometry=region, scale=1000).getInfo()
        pixel_count = stats.get('T21', 0)
        
        if pixel_count > 0:
            print(f"  [FIRE FOUND] {d_str}: {pixel_count} pixels")
            if pixel_count > max_fires:
                max_fires = pixel_count
                best_date = d_str
        else:
            # print(f"  {d_str}: 0 fires")
            pass
    
    current += timedelta(days=1)

print("\nSearch Complete.")
if best_date:
    print(f"=== BEST DATE FOUND: {best_date} ({max_fires} pixels) ===")
    
    # Get coordinates for the best date to visualize
    fire_img = ee.ImageCollection('FIRMS').filterDate(best_date, (datetime.strptime(best_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")).filterBounds(region).select('T21').max().gt(300).selfMask()
    points = fire_img.reduceToVectors(geometry=region, scale=500, geometryType='centroid', maxPixels=1e9)
    feats = points.getInfo()['features']
    print(f"Fire locations on {best_date}:")
    for f in feats:
        print(f"  {f['geometry']['coordinates']}")
        
else:
    print("No fires found in May 2025.")
