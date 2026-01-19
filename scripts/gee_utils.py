"""
GEE Helper Functions for Data Extraction
"""
import ee
from config import *

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"✓ GEE initialized: {GEE_PROJECT_ID}")
    except:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)

def get_burnt_pixels(year, region=None):
    """
    Extract burnt pixel locations for a given year.
    
    Args:
        year (int): Year to extract data from
        region (ee.Geometry): Optional region filter
        
    Returns:
        ee.FeatureCollection: Burnt pixels as point features
    """
    if region is None:
        beats = ee.FeatureCollection(BEATS_ASSET_ID)
        region = beats.geometry()
    
    # Get MODIS burnt area for the year
    fires = ee.ImageCollection(FIRE_DATASET)\
        .filterDate(f'{year}-01-01', f'{year}-12-31')\
        .select('BurnDate')\
        .filterBounds(region)
    
    # Create binary burnt/not-burnt
    burnt_mask = fires.map(lambda img: img.gt(0)).sum().gt(0)
    
    # Sample burnt pixels
    burnt_points = burnt_mask.selfMask().sample(
        region=region,
        scale=OUTPUT_RESOLUTION_M,
        numPixels=10000,  # Limit for memory
        seed=42,
        geometries=True
    )
    
    return burnt_points.map(lambda f: f.set('year', year, 'label', 1))

def get_non_burnt_pixels(year, region=None, burnt_pixels=None):
    """
    Sample non-burnt pixels (negative samples).
    
    Args:
        year (int): Year to extract data from
        region (ee.Geometry): Optional region filter
        burnt_pixels (ee.FeatureCollection): Burnt pixels to avoid
        
    Returns:
        ee.FeatureCollection: Non-burnt pixels as point features
    """
    if region is None:
        beats = ee.FeatureCollection(BEATS_ASSET_ID)
        region = beats.geometry()
    
    # Get burnt areas
    fires = ee.ImageCollection(FIRE_DATASET)\
        .filterDate(f'{year}-01-01', f'{year}-12-31')\
        .select('BurnDate')\
        .filterBounds(region)
    
    burnt_mask = fires.map(lambda img: img.gt(0)).sum().gt(0).unmask(0).clip(region)
    
    # Invert to get non-burnt areas
    non_burnt_mask = burnt_mask.Not().selfMask() # selfMask to only sample where it's 1 (not burnt)
    
    # Sample with higher count (to balance class)
    non_burnt_points = non_burnt_mask.sample(
        region=region,
        scale=OUTPUT_RESOLUTION_M,
        numPixels=1000 * CLASS_BALANCE_RATIO, # Sample 10x more non-burnt
        seed=42,
        geometries=True
    )
    
    return non_burnt_points.map(lambda f: f.set('year', year, 'label', 0))

def get_sentinel_timeseries(point, start_date, end_date):
    """
    Get Sentinel-2 time series for a point location.
    
    Args:
        point (ee.Feature): Point with .geometry()
        start_date (str): Start date 'YYYY-MM-DD'
        end_date (str): End date 'YYYY-MM-DD'
        
    Returns:
        ee.ImageCollection: Time series of S2 data
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start_date, end_date)\
        .filterBounds(point.geometry())\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))\
        .select(PRITHVI_INPUT_BANDS)
    
    return s2

def calculate_proximity_features(region):
    """
    Calculate distance to roads and villages.
    
    (Placeholder - requires OSM or other road/settlement data)
    """
    # TODO: Implement using OSM data or Indian Census data
    # For now, return dummy image
    return ee.Image.constant([5000, 2000]).rename(['dist_road', 'dist_village'])

def get_timeseries_patch(point_geom, start_date, end_date, patch_size=64):
    """
    Extract a temporal patch of HLS/Sentinel-2 data.
    """
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start_date, end_date)\
        .filterBounds(point_geom.buffer(1000))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))\
        .select(PRITHVI_INPUT_BANDS)
    
    # Check if empty
    count = s2_col.size().getInfo()
    if count == 0:
        return None
    
    # Sort and take latest if many, or just keep all and toBands
    # For Prithvi, we need consistent time steps. 
    # Let's just take the first 18 available or repeat last
    
    # Create monthly medians to ensure consistent steps
    fake_empty = ee.Image.constant([0]*len(PRITHVI_INPUT_BANDS)).rename(PRITHVI_INPUT_BANDS).cast(dict(zip(PRITHVI_INPUT_BANDS, ['float']*6)))
    
    steps = []
    curr_date = ee.Date(start_date)
    for i in range(6):
        d1 = curr_date.advance(i, 'month')
        d2 = d1.advance(1, 'month')
        # Take median of the month, add empty bands to ensure schema consistency
        img = s2_col.filterDate(d1, d2).median()
        img = img.addBands(fake_empty, overwrite=False).select(PRITHVI_INPUT_BANDS).unmask(0)
        steps.append(img.set('step', i))

    stacked = ee.ImageCollection(steps).toBands()
    
    # Region for export
    region = point_geom.buffer(patch_size * 20 / 2).bounds()
    
    return stacked.clip(region)

def download_patch_local(point_coords, start_date, end_date, output_path, patch_size=64):
    """
    Download a S2 time-series patch.
    """
    point = ee.Geometry.Point(point_coords)
    stacked = get_timeseries_patch(point, start_date, end_date, patch_size)
    
    if stacked is None:
        raise ValueError("No S2 data found for this period")
        
    region = point.buffer(patch_size * 20 / 2).bounds().getInfo()['coordinates']
    
    # Download as GeoTIFF
    url = stacked.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'format': 'GEO_TIFF',
        'region': region
    })
    
    import requests
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Download failed with status {response.status_code}: {response.text}")
    
    return output_path

    return output_path

# ==========================================
# Feature Extraction Utilities (Moved from extract_training_features.py)
# ==========================================
from datetime import datetime, timedelta
import math

def get_era5_weather(point, date_gee):
    """ERA5-Land: Reanalysis (Weather + Soil)"""
    try:
        w = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")\
              .filterBounds(point).filterDate(date_gee, date_gee.advance(1, 'day')).first()
        if not w:
            return {}
        s = w.select(['temperature_2m', 'dewpoint_temperature_2m', 'u_component_of_wind_10m', 
                      'v_component_of_wind_10m', 'total_precipitation', 'volumetric_soil_water_layer_1'])\
             .reduceRegion(ee.Reducer.mean(), point.buffer(1000), 11132).getInfo()
        
        u, v = s.get('u_component_of_wind_10m', 0), s.get('v_component_of_wind_10m', 0)
        temp = s.get('temperature_2m', 273) - 273.15
        dew = s.get('dewpoint_temperature_2m', 273) - 273.15
        return {
            'era5_temp': temp,
            'era5_wind_speed': math.sqrt(u**2 + v**2),
            'era5_precip': s.get('total_precipitation', 0) * 1000,
            'era5_vpd': 6.11 * (math.exp(17.27*temp/(temp+237.3)) - math.exp(17.27*dew/(dew+237.3))),
            'era5_soil_moisture': s.get('volumetric_soil_water_layer_1', 0)
        }
    except:
        return {}

def get_topography(point):
    """SRTM: Elevation, Slope, Aspect"""
    try:
        srtm = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(srtm)
        s = terrain.select(['elevation', 'slope', 'aspect'])\
                   .reduceRegion(ee.Reducer.mean(), point.buffer(1000), 90).getInfo()
        return {
            'elevation': s.get('elevation', 0),
            'slope': s.get('slope', 0),
            'aspect': s.get('aspect', 0)
        }
    except:
        return {}

def get_vegetation(point, date_gee):
    """MODIS: NDVI (Vegetation Health)"""
    try:
        # MOD13Q1: 16-Day Vegetation Indices
        # Updated to V6.1
        ndvi = ee.ImageCollection("MODIS/061/MOD13Q1")\
                 .filterBounds(point)\
                 .filterDate(date_gee.advance(-16, 'day'), date_gee.advance(16, 'day'))\
                 .first()
        if not ndvi:
            return {}
        s = ndvi.select(['NDVI', 'EVI'])\
                .reduceRegion(ee.Reducer.mean(), point.buffer(1000), 250).getInfo()
        return {
            'modis_ndvi': s.get('NDVI', 0) * 0.0001,
            'modis_evi': s.get('EVI', 0) * 0.0001
        }
    except:
        return {}

def get_multisource_features(point, date_str, lags=[1]):
    """Extract all environmental features for a point."""
    all_features = {}
    fire_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Static
    all_features.update(get_topography(point))
    
    # Dynamic (Lagged)
    for lag in lags:
        lag_date = fire_date - timedelta(days=lag)
        lag_gee = ee.Date(lag_date.strftime('%Y-%m-%d'))
        
        era5 = get_era5_weather(point, lag_gee)
        veg = get_vegetation(point, lag_gee)
        
        # We only really need lagged weather, not necessarily lagged topo obviously
        suffix = f"_d{lag}" if lag > 0 else ""
        
        for k, v in {**era5, **veg}.items():
            all_features[f'{k}{suffix}'] = v
            
    return all_features
