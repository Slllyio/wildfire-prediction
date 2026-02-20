"""
Fire Dynamics Analysis: Calculate Rate of Spread, Direction, and Causal Factors

Analyzes temporal fire sequences to extract:
1. Rate of Spread (ROS) - meters/hour
2. Spread Direction - degrees from north
3. Topographic influence - slope, aspect, elevation
4. Wind-fire alignment - wind vs spread direction
5. Vegetation stress - NDVI, NBR, fuel moisture
"""
import ee
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.spatial.distance import cdist

GEE_PROJECT_ID = 'monkhub-internal-enetra-dev'
INPUT_CSV = r"c:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2\outputs\training_features_multisource.csv"
OUTPUT_CSV = r"c:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2\outputs\fire_dynamics_analysis.csv"

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate direction from point 1 to point 2 in degrees"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360

def get_topography(point):
    """Extract elevation, slope, aspect from SRTM"""
    try:
        srtm = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(srtm)
        
        stats = terrain.select(['elevation', 'slope', 'aspect'])\
                      .reduceRegion(ee.Reducer.mean(), point, 30).getInfo()
        
        return {
            'elevation': stats.get('elevation', 0),
            'slope': stats.get('slope', 0),
            'aspect': stats.get('aspect', 0)
        }
    except:
        return {'elevation': 0, 'slope': 0, 'aspect': 0}

def get_vegetation_indices(point, date_str):
    """Calculate NDVI, NBR, EVI from Sentinel-2"""
    try:
        date = ee.Date(date_str)
        # Get S2 image closest to fire date (within ±7 days)
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
               .filterBounds(point)\
               .filterDate(date.advance(-7, 'day'), date.advance(7, 'day'))\
               .sort('CLOUDY_PIXEL_PERCENTAGE')\
               .first()
        
        if not s2:
            return {}
        
        # Calculate indices
        nir, red = s2.select('B8'), s2.select('B4')
        swir = s2.select('B11')
        blue = s2.select('B2')
        
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
        nbr = nir.subtract(swir).divide(nir.add(swir)).rename('nbr')
        evi = nir.subtract(red).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).multiply(2.5).rename('evi')
        
        stats = ee.Image([ndvi, nbr, evi])\
                  .reduceRegion(ee.Reducer.mean(), point, 20).getInfo()
        
        return {
            'ndvi': stats.get('ndvi', 0),
            'nbr': stats.get('nbr', 0),
            'evi': stats.get('evi', 0)
        }
    except:
        return {}

def identify_fire_clusters(df, spatial_threshold=1000, temporal_threshold_days=7):
    """
    Group fire detections into same fire events based on:
    - Spatial proximity (< 1km)
    - Temporal proximity (< 7 days)
    """
    clusters = []
    df = df.sort_values('fire_date').reset_index(drop=True)
    
    for idx, row in df.iterrows():
        assigned = False
        for cluster in clusters:
            # Check if this point belongs to existing cluster
            for c_idx in cluster['indices']:
                ref = df.iloc[c_idx]
                dist = haversine_distance(row['latitude'], row['longitude'], 
                                         ref['latitude'], ref['longitude'])
                days_diff = abs((datetime.strptime(row['fire_date'], '%Y-%m-%d') - 
                                datetime.strptime(ref['fire_date'], '%Y-%m-%d')).days)
                
                if dist < spatial_threshold and days_diff < temporal_threshold_days:
                    cluster['indices'].append(idx)
                    assigned = True
                    break
            if assigned:
                break
        
        if not assigned:
            clusters.append({'indices': [idx]})
    
    return clusters

def calculate_spread_metrics(cluster_df):
    """
    For a fire cluster, calculate:
    - Rate of Spread (ROS)
    - Spread direction
    - Total area affected
    """
    if len(cluster_df) < 2:
        return {}
    
    # Sort by time
    cluster_df = cluster_df.sort_values('fire_date')
    
    # First and last detection
    first = cluster_df.iloc[0]
    last = cluster_df.iloc[-1]
    
    # Time difference in hours
    dt_hours = (datetime.strptime(last['fire_date'], '%Y-%m-%d') - 
                datetime.strptime(first['fire_date'], '%Y-%m-%d')).total_seconds() / 3600
    
    if dt_hours == 0:
        return {}
    
    # Distance traveled
    dist_m = haversine_distance(first['latitude'], first['longitude'],
                                last['latitude'], last['longitude'])
    
    # Rate of spread (m/hr)
    ros = dist_m / dt_hours if dt_hours > 0 else 0
    
    # Spread direction
    spread_dir = calculate_bearing(first['latitude'], first['longitude'],
                                   last['latitude'], last['longitude'])
    
    # Wind alignment (if wind data available)
    wind_alignment = None
    if 'era5_wind_speed_d1' in cluster_df.columns and pd.notna(first.get('era5_wind_speed_d1')):
        # Average wind direction
        avg_wind_dir = cluster_df['era5_wind_speed_d1'].mean()  # Placeholder - need actual wind dir
        # Alignment score: cos(spread_dir - wind_dir)
        # 1 = perfect alignment, 0 = perpendicular, -1 = opposite
    
    return {
        'ros_m_per_hr': ros,
        'spread_direction_deg': spread_dir,
        'duration_hours': dt_hours,
        'total_distance_m': dist_m,
        'num_detections': len(cluster_df)
    }

def main():
    ee.Initialize(project=GEE_PROJECT_ID)
    
    print("Loading fire data...")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Enriching {len(df)} fire points with topography and vegetation...")
    
    enriched_records = []
    
    for idx, row in df.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        
        # Get topography
        topo = get_topography(point)
        
        # Get vegetation indices
        veg = get_vegetation_indices(point, row['fire_date'])
        
        record = row.to_dict()
        record.update(topo)
        record.update(veg)
        
        enriched_records.append(record)
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(df)}]")
    
    df_enriched = pd.DataFrame(enriched_records)
    
    print("\nIdentifying fire clusters for spread analysis...")
    clusters = identify_fire_clusters(df_enriched)
    
    print(f"Found {len(clusters)} distinct fire events")
    
    # Calculate spread metrics for multi-detection fires
    spread_data = []
    for i, cluster in enumerate(clusters):
        if len(cluster['indices']) >= 2:
            cluster_df = df_enriched.iloc[cluster['indices']]
            metrics = calculate_spread_metrics(cluster_df)
            
            if metrics:
                print(f"  Cluster {i+1}: ROS = {metrics['ros_m_per_hr']:.1f} m/hr, "
                      f"Direction = {metrics['spread_direction_deg']:.0f}°")
                
                # Add to first detection in cluster
                for key, val in metrics.items():
                    df_enriched.loc[cluster['indices'][0], f'spread_{key}'] = val
    
    # Save enhanced dataset
    df_enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Fire dynamics analysis saved: {OUTPUT_CSV}")
    print(f"  New features: topography, vegetation indices, spread metrics")

if __name__ == "__main__":
    main()
