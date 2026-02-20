
import ee
# Hardcoded project ID to avoid import issues
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

start = '2025-05-15'
end = '2025-05-16'

print(f"Checking FIRMS (MODIS/VIIRS) for actual fires on {start}...")
fires = ee.ImageCollection('FIRMS').filterDate(start, end).filterBounds(region)
count = fires.size().getInfo()

if count > 0:
    # T21 > 300K
    fire_img = fires.select('T21').max().gt(300).selfMask()
    
    stats = fire_img.reduceRegion(reducer=ee.Reducer.count(), geometry=region, scale=1000).getInfo()
    pixel_count = stats.get('T21', 0)
    
    if pixel_count > 0:
        points = fire_img.reduceToVectors(geometry=region, scale=500, geometryType='centroid', maxPixels=1e9)
        feats = points.getInfo()['features']
        print(f"ACTUAL FIRES FOUND: {len(feats)}")
        for f in feats:
            print(f"  - Coordinate: {f['geometry']['coordinates']}")
    else:
        print("RESULT: No active fires detected.")
else:
    print("RESULT: No FIRMS data available for this date.")
