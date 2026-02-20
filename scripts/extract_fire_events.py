"""
Per-Point Burnt Area Matching (3-day temporal + spatial proximity)

Each VIIRS fire point gets only the burnt area:
1. Detected within 3 days of the fire (same day + next 3 days)
2. Within 1km of the fire point location

This creates a causal link: Fire → Burn scar
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
from shapely.geometry import mapping, shape, Point
from shapely.ops import transform

supported_drivers['KML'] = 'rw'
supported_drivers['LIBKML'] = 'rw'

KMZ_PATH = r"c:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2\data\West Chhindwara.kmz"
START_DATE = '2025-02-15'
END_DATE = '2025-06-16'
GEE_PROJECT_ID = 'monkhub-internal-enetra-dev'
OUTPUT_DIR = r"c:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2\outputs\training_data_final"
BUFFER_METERS = 1000  # 1km buffer around fire point
TEMPORAL_WINDOW_DAYS = 3

def extract_kmz_geometry(kmz_path):
    print(f"Reading KMZ: {kmz_path}")
    temp_dir = 'temp_kml_dir'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kml_files = [f for f in z.namelist() if f.endswith('.kml')]
        if not kml_files:
            raise ValueError("No KML file found in KMZ")
        z.extract(kml_files[0], temp_dir)
        kml_path = os.path.join(temp_dir, kml_files[0])
        import fiona
        layers = fiona.listlayers(kml_path)
        dfs = []
        for layer in layers:
            dfs.append(gpd.read_file(kml_path, driver='KML', layer=layer))
        gdf = pd.concat(dfs, ignore_index=True)
        shutil.rmtree(temp_dir)
        geom = gdf.union_all()
        def flatten_coords(x, y, z=None):
            return (x, y)
        return transform(flatten_coords, geom)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        geom = extract_kmz_geometry(KMZ_PATH)
        ee.Initialize(project=GEE_PROJECT_ID)
        ee_geom = ee.Geometry(mapping(geom))
        
        print(f"Fetching FIRMS for Chhindwara ({START_DATE} to {END_DATE})...")
        firms = ee.ImageCollection('FIRMS').filterDate(START_DATE, END_DATE).filterBounds(ee_geom)
        
        def extract_features(image):
            date = image.date().format('yyyy-MM-dd')
            b_names = image.bandNames()
            
            def get_band(name_list, target_name):
                dummy = ee.Image(0).rename(target_name)
                matches = b_names.filter(ee.Filter.inList('item', name_list))
                return ee.Image(ee.Algorithms.If(
                    matches.size().gt(0),
                    image.select(ee.String(matches.get(0))).rename(target_name),
                    dummy
                ))

            temp = get_band(['T21', 'bright_ti4'], 'temp').toInt()
            frp = get_band(['FRP', 'frp'], 'fire_power')
            conf = get_band(['Confidence', 'confidence'], 'conf')
            
            combined = ee.Image([temp, frp, conf])
            points = combined.updateMask(temp.gt(0)).reduceToVectors(
                geometry=ee_geom, scale=500, geometryType='centroid', maxPixels=1e9,
                reducer=ee.Reducer.mean(), labelProperty='temp_val'
            )
            
            def set_meta(f):
                return f.copyProperties(image).set('date', date)
            return points.map(set_meta)

        all_fires = firms.map(extract_features).flatten()
        print("Requesting detections...")
        fires_info = all_fires.getInfo()
        features = fires_info.get('features', [])
        
        print(f"\nProcessing {len(features)} fire points with per-point matching...")
        
        # Load MCD64A1 once for the entire period
        mcd64 = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(START_DATE, END_DATE).filterBounds(ee_geom)
        
        results = []
        initiation_map = {}
        
        for idx, f in enumerate(features):
            props = f.get('properties', {})
            coords = f.get('geometry', {}).get('coordinates', [])
            fire_date_str = props.get('date')
            fire_date = datetime.strptime(fire_date_str, '%Y-%m-%d')
            fire_doy = fire_date.timetuple().tm_yday
            
            # Track initiation
            key = (round(coords[0], 4), round(coords[1], 4))
            if key not in initiation_map or fire_date_str < initiation_map[key]:
                initiation_map[key] = fire_date_str
            
            # Create 1km buffer around fire point
            fire_point = ee.Geometry.Point(coords)
            buffer = fire_point.buffer(BUFFER_METERS)
            
            # Get month's MCD64A1 image
            month_start = ee.Date(fire_date_str).update(day=1)
            burnt_img = mcd64.filterDate(month_start, month_start.advance(1, 'month')).first()
            
            burnt_wkt = "POLYGON EMPTY"
            if burnt_img:
                try:
                    # Get burnt pixels within temporal window (DOY: fire_day to fire_day+3)
                    burn_date_band = burnt_img.select('BurnDate')
                    temporal_mask = burn_date_band.gte(fire_doy).And(burn_date_band.lte(fire_doy + TEMPORAL_WINDOW_DAYS))
                    
                    # Apply spatial and temporal filters
                    burnt_mask = temporal_mask.selfMask()
                    
                    # Vectorize only within buffer
                    b_vecs = burnt_mask.reduceToVectors(
                        geometry=buffer, 
                        scale=500, 
                        maxPixels=1e9
                    ).getInfo()
                    
                    if b_vecs.get('features'):
                        polys = [shape(fx['geometry']) for fx in b_vecs['features']]
                        burnt_wkt = polys[0].wkt if len(polys) == 1 else shape({'type': 'MultiPolygon', 'coordinates': [p.__geo_interface__['coordinates'] for p in polys]}).wkt
                except:
                    pass
            
            # Get weather for this date
            weather_date = ee.Date(fire_date_str)
            weather = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")\
                        .filterBounds(fire_point).filterDate(weather_date, weather_date.advance(1, 'day')).first()
            w_stats = {'speed': 0, 'direction': 0}
            if weather:
                try:
                    s = weather.select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])\
                              .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=11132).getInfo()
                    u, v = s.get('u_component_of_wind_10m', 0.0), s.get('v_component_of_wind_10m', 0.0)
                    w_stats = {'speed': math.sqrt(u**2 + v**2), 'direction': (math.atan2(u, v)*180/math.pi + 180)%360}
                except: pass
            
            results.append({
                'date': fire_date_str,
                'longitude': coords[0],
                'latitude': coords[1],
                'satellite': props.get('satellite'),
                'frp': props.get('fire_power'),
                'brightness': props.get('temp'),
                'confidence': props.get('conf'),
                'initiation_date': initiation_map[key],
                'wind_speed': w_stats['speed'],
                'wind_direction': w_stats['direction'],
                '.geo': burnt_wkt
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(features)} points...")

        # Save by date
        df_all = pd.DataFrame(results)
        dates = df_all['date'].unique()
        for d_str in sorted(dates):
            day_df = df_all[df_all['date'] == d_str]
            day_df.to_csv(os.path.join(OUTPUT_DIR, f"fire_events_{d_str}.csv"), index=False)
            print(f"Saved {d_str} ({len(day_df)} events)")
        
        print(f"\n✓ Completed! Per-point burnt area saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
