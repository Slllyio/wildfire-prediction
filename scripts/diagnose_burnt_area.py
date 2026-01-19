import ee

GEE_PROJECT_ID = 'van-suraksha-alert'
ee.Initialize(project=GEE_PROJECT_ID)

# West Chhindwara approximate bounds
region = ee.Geometry.Rectangle([78.2, 21.4, 79.5, 22.9])

print("=" * 60)
print("MODIS MCD64A1 Burnt Area Diagnostic")
print("=" * 60)

# Check what MCD64A1 data is available
print("\n1. Checking MCD64A1 availability for 2025...")
mcd64_2025 = ee.ImageCollection('MODIS/061/MCD64A1')\
    .filterDate('2025-02-01', '2025-06-30')\
    .filterBounds(region)

count = mcd64_2025.size().getInfo()
print(f"   Found {count} MCD64A1 images for Feb-Jun 2025")

if count > 0:
    dates = mcd64_2025.aggregate_array('system:time_start').getInfo()
    import datetime
    print("\n   Available image dates:")
    for ts in dates:
        dt = datetime.datetime.fromtimestamp(ts/1000)
        print(f"   - {dt.strftime('%Y-%m-%d')}")
    
    # Check March 2025 image in detail (since March 23 worked)
    print("\n2. Analyzing March 2025 burnt area...")
    march_img = mcd64_2025.filterDate('2025-03-01', '2025-04-01').first()
    
    if march_img:
        burn_stats = march_img.select('BurnDate').reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=500
        ).getInfo()
        
        print(f"   BurnDate range: {burn_stats.get('BurnDate_min')} to {burn_stats.get('BurnDate_max')}")
        print("   (Day of year when burning was detected)")
        
        # Check if day 82 (March 23) exists
        day_82_pixels = march_img.select('BurnDate').eq(82).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=500
        ).getInfo()
        print(f"\n   Pixels burnt on day 82 (March 23): {day_82_pixels.get('BurnDate', 0)}")
        
else:
    print("\n   ❌ NO MCD64A1 data available for 2025!")
    print("   This explains why .geo is empty.")

# Check alternative: MODIS Thermal Anomalies (older monthly product)
print("\n3. Checking MODIS Monthly Burned Area (MCD64A1)...")
print("   Note: This product typically has 2-3 month lag time")

# Check latest available data
latest = ee.ImageCollection('MODIS/061/MCD64A1')\
    .filterBounds(region)\
    .limit(5, 'system:time_start', False)

latest_dates = latest.aggregate_array('system:time_start').getInfo()
print("\n   Latest 5 available images:")
import datetime
for ts in latest_dates:
    dt = datetime.datetime.fromtimestamp(ts/1000)
    print(f"   - {dt.strftime('%Y-%m-%d')}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("If 2025 data is unavailable, consider:")
print("1. Use FIRMS 'confidence' as a proxy for burn intensity")
print("2. Buffer active fire points to estimate burnt area")
print("3. Wait for MCD64A1 2025 data to become available")
print("4. Use alternative: MODIS/006/MOD14A1 (daily thermal anomalies)")
