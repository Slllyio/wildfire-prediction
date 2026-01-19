"""
Download ALL auxiliary data for fire prediction in one shot.
- ERA5 weather (temperature, humidity, wind, precipitation)
- OSM features (roads, villages)
- Computes vegetation indices from existing S2 tiles
"""
import ee
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import rasterio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEE_PROJECT_ID

# Region bounds for West Chhindwara
BOUNDS = {
    'min_lon': 78.2449,
    'max_lon': 79.0312,
    'min_lat': 21.8803,
    'max_lat': 22.7003
}

def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"[OK] GEE initialized: {GEE_PROJECT_ID}")
    except:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)


def download_era5_weather(target_date, output_path):
    """
    Download ERA5 weather data for May 2025.
    Variables: temperature, humidity, wind, precipitation
    """
    print("Downloading ERA5 weather data...")
    
    region = ee.Geometry.Rectangle([
        BOUNDS['min_lon'], BOUNDS['min_lat'],
        BOUNDS['max_lon'], BOUNDS['max_lat']
    ])
    
    # ERA5 daily aggregates
    # Note: ERA5 data has ~2 month lag, so for May 2025 we use climatology/forecast proxy
    target = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Use ERA5-Land for historical data or climatological average
    # For future dates, we'll use the same month from previous year as proxy
    proxy_year = target.year - 1
    start_date = f"{proxy_year}-{target.month:02d}-01"
    end_date = f"{proxy_year}-{target.month:02d}-28"
    
    print(f"  Using proxy period: {start_date} to {end_date}")
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")\
        .filterDate(start_date, end_date)\
        .filterBounds(region)
    
    # Get monthly means
    temp_2m = era5.select('temperature_2m').mean()
    dewpoint = era5.select('dewpoint_temperature_2m').mean()
    precip = era5.select('total_precipitation_sum').mean()
    wind_u = era5.select('u_component_of_wind_10m').mean()
    wind_v = era5.select('v_component_of_wind_10m').mean()
    
    # Calculate wind speed
    wind_speed = wind_u.pow(2).add(wind_v.pow(2)).sqrt()
    
    # Calculate relative humidity from temp and dewpoint (Magnus formula approximation)
    # RH ≈ 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
    # Simplified: just use dewpoint depression as proxy
    humidity_proxy = dewpoint.divide(temp_2m).multiply(100).clamp(0, 100)
    
    # Stack all bands
    weather = ee.Image.cat([
        temp_2m.subtract(273.15).rename('temperature_c'),  # Kelvin to Celsius
        humidity_proxy.rename('humidity_pct'),
        wind_speed.rename('wind_ms'),
        precip.multiply(1000).rename('precip_mm')  # m to mm
    ])
    
    # Sample at grid points
    grid_scale = 0.05  # ~5km
    samples = weather.sampleRectangle(region=region, defaultValue=0)
    
    data = samples.getInfo()
    
    # Convert to DataFrame
    temp = np.array(data['properties']['temperature_c'])
    hum = np.array(data['properties']['humidity_pct'])
    wind = np.array(data['properties']['wind_ms'])
    precip = np.array(data['properties']['precip_mm'])
    
    # Get coordinates
    coords = region.getInfo()['coordinates'][0]
    lons = np.linspace(BOUNDS['min_lon'], BOUNDS['max_lon'], temp.shape[1])
    lats = np.linspace(BOUNDS['max_lat'], BOUNDS['min_lat'], temp.shape[0])
    
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append({
                'lon': lon,
                'lat': lat,
                'temperature_c': temp[i, j] if i < temp.shape[0] and j < temp.shape[1] else 35,
                'humidity_pct': hum[i, j] if i < hum.shape[0] and j < hum.shape[1] else 25,
                'wind_ms': wind[i, j] if i < wind.shape[0] and j < wind.shape[1] else 5,
                'precip_mm': precip[i, j] if i < precip.shape[0] and j < precip.shape[1] else 0
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} weather grid points to {output_path}")
    return df


def download_osm_features(output_path):
    """
    Download roads and villages from OpenStreetMap via Overpass API.
    """
    print("Downloading OSM features (roads, villages)...")
    
    query = f"""
    [out:json][timeout:120];
    (
      // Roads
      way["highway"~"primary|secondary|tertiary|trunk"]
        ({BOUNDS['min_lat']},{BOUNDS['min_lon']},{BOUNDS['max_lat']},{BOUNDS['max_lon']});
      // Villages/settlements
      node["place"~"village|hamlet|town"]
        ({BOUNDS['min_lat']},{BOUNDS['min_lon']},{BOUNDS['max_lat']},{BOUNDS['max_lon']});
    );
    out center;
    """
    
    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={'data': query},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        roads = []
        villages = []
        
        for elem in data.get('elements', []):
            if elem['type'] == 'way':
                center = elem.get('center', {})
                if center:
                    roads.append((center.get('lon'), center.get('lat')))
            elif elem['type'] == 'node':
                villages.append((elem.get('lon'), elem.get('lat')))
        
        print(f"  Found {len(roads)} road segments, {len(villages)} villages")
        
        result = {
            'roads': roads,
            'villages': villages
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f)
        
        print(f"  Saved OSM features to {output_path}")
        return result
        
    except Exception as e:
        print(f"  OSM download failed: {e}")
        # Return empty defaults
        return {'roads': [], 'villages': []}


def compute_vegetation_indices(tile_dir, output_path):
    """
    Compute NDVI, NBR from existing S2 tiles.
    """
    print("Computing vegetation indices from S2 tiles...")
    
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    
    results = []
    for tf in tile_files[:10]:  # Sample first 10 tiles
        try:
            with rasterio.open(os.path.join(tile_dir, tf)) as src:
                data = src.read()
                transform = src.transform
                
                # Bands order: B2, B3, B4, B8, B11, B12 (6 bands per timestep)
                # Take last timestep (most recent)
                n_bands = data.shape[0]
                n_steps = n_bands // 6
                if n_steps == 0:
                    continue
                    
                # Last timestep bands
                offset = (n_steps - 1) * 6
                b3 = data[offset + 1].astype(float)  # Green
                b4 = data[offset + 2].astype(float)  # Red
                b8 = data[offset + 3].astype(float)  # NIR
                b12 = data[offset + 5].astype(float) # SWIR2
                
                # NDVI = (NIR - Red) / (NIR + Red)
                ndvi = np.where(b8 + b4 > 0, (b8 - b4) / (b8 + b4 + 1e-10), 0)
                # NBR = (NIR - SWIR2) / (NIR + SWIR2)
                nbr = np.where(b8 + b12 > 0, (b8 - b12) / (b8 + b12 + 1e-10), 0)
                
                # Get center coord
                cx = transform.c + (data.shape[2] / 2) * transform.a
                cy = transform.f + (data.shape[1] / 2) * transform.e
                
                results.append({
                    'tile': tf,
                    'lon': cx,
                    'lat': cy,
                    'ndvi_mean': float(np.nanmean(ndvi)),
                    'nbr_mean': float(np.nanmean(nbr)),
                    'ndvi_min': float(np.nanmin(ndvi)),
                    'drought_stress': float(1.0 - np.clip(np.nanmean(ndvi), 0, 1))
                })
        except Exception as e:
            print(f"  Error processing {tf}: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"  Saved vegetation indices for {len(df)} tiles to {output_path}")
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2025-05-15', help='Target date YYYY-MM-DD')
    parser.add_argument('--output-dir', default='data/auxiliary', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize GEE
    initialize_gee()
    
    # 1. Download ERA5 weather
    weather_path = os.path.join(args.output_dir, f'weather_{args.date.replace("-", "")}.csv')
    download_era5_weather(args.date, weather_path)
    
    # 2. Download OSM features
    osm_path = os.path.join(args.output_dir, 'osm_features.json')
    download_osm_features(osm_path)
    
    # 3. Compute vegetation indices from existing tiles
    tile_dir = 'outputs/kmz_campaign/tiles'
    if os.path.exists(tile_dir):
        veg_path = os.path.join(args.output_dir, 'vegetation_indices.csv')
        compute_vegetation_indices(tile_dir, veg_path)
    
    print("\n[OK] All auxiliary data downloaded!")
    print(f"Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
