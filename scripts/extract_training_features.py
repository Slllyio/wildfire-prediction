"""
Extract fire training data with MULTI-SOURCE weather features.

Compare datasets:
1. ERA5-Land (reanalysis, 11km)
2. GFS (forecast-like, 25km) 
3. CHIRPS (precipitation, 5km)
4. MODIS LST (thermal, 1km)

This helps determine which dataset to use for operational prediction.
"""
import ee
import geopandas as gpd
import pandas as pd
import zipfile
import os
import shutil
import math
from fiona.drvsupport import supported_drivers
from datetime import datetime, timedelta
from shapely.geometry import mapping
from shapely.ops import transform

supported_drivers['KML'] = 'rw'
supported_drivers['LIBKML'] = 'rw'

# ── Dynamic paths ────────────────────────────────────────────────────────────
# __file__  →  .../wildfire-prediction/scripts/extract_training_features.py
# SCRIPT_DIR →  .../wildfire-prediction/scripts/
# PROJECT_DIR → .../wildfire-prediction/
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# KMZ file is expected at:  <project_root>/data/West Chhindwara.kmz
KMZ_PATH   = os.path.join(PROJECT_DIR, 'data', 'West Chhindwara.kmz')

# Output CSV written to:    <project_root>/outputs/training_features_multisource.csv
OUTPUT_CSV = os.path.join(PROJECT_DIR, 'outputs', 'training_features_multisource.csv')
# ─────────────────────────────────────────────────────────────────────────────

START_DATE    = '2025-02-15'
END_DATE      = '2025-06-16'
GEE_PROJECT_ID = 'monkhub-internal-enetra-dev'
BUFFER_METERS = 1000

# Ensure output directory exists so CSV write never fails
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

def extract_kmz_geometry(kmz_path):
    print(f"Reading KMZ: {kmz_path}")
    temp_dir = 'temp_kml_dir'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kml_files = [f for f in z.namelist() if f.endswith('.kml')]
        z.extract(kml_files[0], temp_dir)
        kml_path = os.path.join(temp_dir, kml_files[0])
        import fiona
        dfs = [gpd.read_file(kml_path, driver='KML', layer=l) for l in fiona.listlayers(kml_path)]
        gdf = pd.concat(dfs, ignore_index=True)
        shutil.rmtree(temp_dir)
        geom = gdf.union_all()
        return transform(lambda x, y, z=None: (x, y), geom)

def get_era5_weather(point, date_gee):
    """ERA5-Land: Reanalysis (what actually happened)"""
    try:
        w = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")\
              .filterBounds(point).filterDate(date_gee, date_gee.advance(1, 'day')).first()
        if not w:
            return {}
        s = w.select(['temperature_2m', 'dewpoint_temperature_2m', 'u_component_of_wind_10m', 
                    'v_component_of_wind_10m', 'total_precipitation', 'volumetric_soil_water_layer_1'])\
             .reduceRegion(ee.Reducer.mean(), point.buffer(BUFFER_METERS), 11132).getInfo()
        
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
                   .reduceRegion(ee.Reducer.mean(), point.buffer(BUFFER_METERS), 90).getInfo()
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
        # We take the image closest to the date
        ndvi = ee.ImageCollection("MODIS/006/MOD13Q1")\
                 .filterBounds(point)\
                 .filterDate(date_gee.advance(-16, 'day'), date_gee.advance(16, 'day'))\
                 .first()
        
        if not ndvi:
            return {}
            
        s = ndvi.select(['NDVI', 'EVI'])\
                .reduceRegion(ee.Reducer.mean(), point.buffer(BUFFER_METERS), 250).getInfo()
        
        return {
            'modis_ndvi': s.get('NDVI', 0) * 0.0001,
            'modis_evi': s.get('EVI', 0) * 0.0001
        }
    except:
        return {}

def get_gfs_weather(point, date_gee):
    """GFS: Forecast (simulates operational deployment)"""
    try:
        # Use GFS 0.25 degree
        gfs = ee.ImageCollection("NOAA/GFS0P25")\
                .filterBounds(point).filterDate(date_gee, date_gee.advance(1, 'day')).first()
        if not gfs:
            return {}
        s = gfs.select(['temperature_2m_above_ground', 'u_component_of_wind_10m_above_ground',
                        'v_component_of_wind_10m_above_ground', 'precipitable_water_entire_atmosphere'])\
               .reduceRegion(ee.Reducer.mean(), point.buffer(BUFFER_METERS), 27830).getInfo()
        
        u, v = s.get('u_component_of_wind_10m_above_ground', 0), s.get('v_component_of_wind_10m_above_ground', 0)
        return {
            'gfs_temp': s.get('temperature_2m_above_ground', 273) - 273.15,
            'gfs_wind_speed': math.sqrt(u**2 + v**2),
            'gfs_precip_water': s.get('precipitable_water_entire_atmosphere', 0)
        }
    except:
        return {}

def get_chirps_precip(point, date_gee):
    """CHIRPS: High-res precipitation"""
    try:
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                   .filterBounds(point).filterDate(date_gee, date_gee.advance(1, 'day')).first()
        if not chirps:
            return {}
        precip = chirps.select('precipitation').reduceRegion(
            ee.Reducer.mean(), point.buffer(BUFFER_METERS), 5566).getInfo()
        return {'chirps_precip': precip.get('precipitation', 0)}
    except:
        return {}

def get_modis_lst(point, date_gee):
    """MODIS: Land Surface Temperature (thermal stress)"""
    try:
        # MOD11A1: Daily LST
        lst = ee.ImageCollection("MODIS/061/MOD11A1")\
                .filterBounds(point).filterDate(date_gee, date_gee.advance(1, 'day')).first()
        if not lst:
            return {}
        s = lst.select(['LST_Day_1km', 'LST_Night_1km'])\
               .reduceRegion(ee.Reducer.mean(), point.buffer(BUFFER_METERS), 1000).getInfo()
        return {
            'modis_lst_day': s.get('LST_Day_1km', 0) * 0.02 - 273.15,  # Scale + K to C
            'modis_lst_night': s.get('LST_Night_1km', 0) * 0.02 - 273.15
        }
    except:
        return {}

def get_multisource_weather(point, date_str, lags=[1, 3, 7]):
    """Extract from ALL sources with time lags"""
    all_features = {}
    fire_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 1. Topography (Static - no date needed)
    topo = get_topography(point)
    all_features.update(topo)
    
    for lag in lags:
        lag_date = fire_date - timedelta(days=lag)
        lag_gee = ee.Date(lag_date.strftime('%Y-%m-%d'))
        
        # Get from all sources
        era5 = get_era5_weather(point, lag_gee)
        gfs = get_gfs_weather(point, lag_gee)
        chirps = get_chirps_precip(point, lag_gee)
        modis = get_modis_lst(point, lag_gee)
        veg = get_vegetation(point, lag_gee)
        
        # Prefix with lag
        for k, v in {**era5, **gfs, **chirps, **modis, **veg}.items():
            all_features[f'{k}_d{lag}'] = v
    
    return all_features

def main():
    try:
        geom = extract_kmz_geometry(KMZ_PATH)
        ee.Initialize(project=GEE_PROJECT_ID)
        ee_geom = ee.Geometry(mapping(geom))
        
        print(f"Fetching FIRMS ({START_DATE} to {END_DATE})...")
        firms = ee.ImageCollection('FIRMS').filterDate(START_DATE, END_DATE).filterBounds(ee_geom)
        
        def extract_features(image):
            date = image.date().format('yyyy-MM-dd')
            b_names = image.bandNames()
            def get_band(names, target):
                matches = b_names.filter(ee.Filter.inList('item', names))
                return ee.Image(ee.Algorithms.If(matches.size().gt(0),
                    image.select(ee.String(matches.get(0))).rename(target),
                    ee.Image(0).rename(target)))
            
            temp = get_band(['T21', 'bright_ti4'], 'temp').toInt()
            combined = ee.Image([temp, get_band(['FRP', 'frp'], 'frp'), get_band(['Confidence', 'confidence'], 'conf')])
            points = combined.updateMask(temp.gt(0)).reduceToVectors(
                geometry=ee_geom, scale=500, geometryType='centroid', maxPixels=1e9,
                reducer=ee.Reducer.mean(), labelProperty='temp_val')
            return points.map(lambda f: f.copyProperties(image).set('date', date))

        all_fires = firms.map(extract_features).flatten()
        print("Requesting fire points...")
        features = all_fires.getInfo().get('features', [])
        
        print(f"\nProcessing {len(features)} points with MULTI-SOURCE weather...")
        results = []
        
        for idx, f in enumerate(features):
            props, coords = f.get('properties', {}), f['geometry']['coordinates']
            fire_date_str = props.get('date')
            fire_point = ee.Geometry.Point(coords)
            
            # Multi-source weather
            weather_all = get_multisource_weather(fire_point, fire_date_str, lags=[1, 3, 7])
            
            record = {
                'fire_date': fire_date_str,
                'longitude': coords[0],
                'latitude': coords[1],
                'day_of_year': datetime.strptime(fire_date_str, '%Y-%m-%d').timetuple().tm_yday,
                'brightness': props.get('temp'),
                'frp': props.get('frp'),
                'confidence': props.get('conf'),
                'label': 1
            }
            record.update(weather_all)
            results.append(record)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(features)}]")

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ Multi-source training data: {OUTPUT_CSV}")
        print(f"  Shape: {df.shape}")
        print(f"  Sources: ERA5, GFS, CHIRPS, MODIS LST")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
