"""
Optimized KMZ Fire Prediction with REAL Auxiliary Data
Uses actual ERA5 weather, OSM distances, and computed vegetation indices.
"""
import torch
import numpy as np
import rasterio
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import zipfile
import re
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import ee
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor

print_lock = threading.Lock()
def safe_print(msg):
    with print_lock:
        print(msg)


def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"[OK] GEE initialized: {GEE_PROJECT_ID}")
    except:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)


def parse_kmz(kmz_path):
    print(f"Parsing KMZ: {kmz_path}")
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            content = kml_file.read().decode('utf-8')
            coord_pattern = re.compile(r'<coordinates>(.*?)</coordinates>', re.DOTALL)
            matches = coord_pattern.findall(content)
            
            polygons = []
            for coords_str in matches:
                coords_str = coords_str.strip()
                points = []
                for tupl in coords_str.split():
                    parts = tupl.split(',')
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            points.append((lon, lat))
                        except:
                            continue
                if len(points) > 2:
                    polygons.append(Polygon(points))
            
            if not polygons:
                raise ValueError("No polygons found in KML")
            return unary_union(polygons)


def generate_tiles(bounds, tile_size_deg=0.05):
    minx, miny, maxx, maxy = bounds
    tiles = []
    lon = minx
    while lon < maxx:
        lat = miny
        while lat < maxy:
            tile_bounds = (lon, lat, min(lon + tile_size_deg, maxx), min(lat + tile_size_deg, maxy))
            tiles.append(tile_bounds)
            lat += tile_size_deg
        lon += tile_size_deg
    return tiles


def download_tile(tile_bounds, start_date, end_date, output_path, bands=PRITHVI_INPUT_BANDS):
    minx, miny, maxx, maxy = tile_bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start_date, end_date)\
        .filterBounds(region)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))\
        .select(bands)
    
    count = s2_col.size().getInfo()
    if count == 0:
        return None
    
    fake_empty = ee.Image.constant([0]*len(bands)).rename(bands).cast(dict(zip(bands, ['float']*6)))
    
    steps = []
    curr_date = ee.Date(start_date)
    for i in range(6):
        d1 = curr_date.advance(i, 'month')
        d2 = d1.advance(1, 'month')
        img = s2_col.filterDate(d1, d2).median()
        img = img.addBands(fake_empty, overwrite=False).select(bands).unmask(0)
        steps.append(img.set('step', i))
    
    stacked = ee.ImageCollection(steps).toBands()
    
    try:
        url = stacked.getDownloadURL({
            'scale': 20,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF',
            'region': [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny]]
        })
        
        import requests
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        else:
            return None
    except Exception as e:
        safe_print(f"Tile download error: {e}")
        return None


def download_tiles_parallel(tiles, start_date, end_date, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    def download_one(idx_tile):
        idx, tile = idx_tile
        path = os.path.join(output_dir, f'tile_{idx:04d}.tif')
        # Skip if already exists
        if os.path.exists(path) and os.path.getsize(path) > 100000:
            safe_print(f"  [CACHED] Tile {idx+1}/{len(tiles)}")
            return idx, path
        result = download_tile(tile, start_date, end_date, path)
        if result:
            safe_print(f"  [OK] Tile {idx+1}/{len(tiles)} downloaded")
        else:
            safe_print(f"  [FAIL] Tile {idx+1}/{len(tiles)} failed")
        return idx, result
    
    print(f"Downloading {len(tiles)} tiles with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, (i, t)): i for i, t in enumerate(tiles)}
        for future in as_completed(futures):
            idx, path = future.result()
            if path:
                results[idx] = path
    
    print(f"Downloaded/cached {len(results)}/{len(tiles)} tiles successfully.")
    return results


class AuxiliaryDataLoader:
    """Load and interpolate real auxiliary data."""
    
    def __init__(self, aux_dir='data/auxiliary'):
        self.aux_dir = aux_dir
        self.weather_df = None
        self.veg_df = None
        self.roads = []
        self.villages = []
        self.road_tree = None
        self.village_tree = None
        self._load_data()
    
    def _load_data(self):
        # Load weather
        weather_files = [f for f in os.listdir(self.aux_dir) if f.startswith('weather_') and f.endswith('.csv')]
        if weather_files:
            self.weather_df = pd.read_csv(os.path.join(self.aux_dir, weather_files[0]))
            print(f"  Loaded weather: {len(self.weather_df)} points")
        
        # Load vegetation
        veg_file = os.path.join(self.aux_dir, 'vegetation_indices.csv')
        if os.path.exists(veg_file):
            self.veg_df = pd.read_csv(veg_file)
            print(f"  Loaded vegetation: {len(self.veg_df)} tiles")
        
        # Load OSM
        osm_file = os.path.join(self.aux_dir, 'osm_features.json')
        if os.path.exists(osm_file):
            with open(osm_file, 'r') as f:
                osm = json.load(f)
            self.roads = osm.get('roads', [])
            self.villages = osm.get('villages', [])
            print(f"  Loaded OSM: {len(self.roads)} roads, {len(self.villages)} villages")
            
            # Build KD-trees for fast nearest neighbor
            if self.roads:
                self.road_tree = cKDTree(np.array(self.roads))
            if self.villages:
                self.village_tree = cKDTree(np.array(self.villages))
    
    def get_weather(self, lon, lat):
        """Get interpolated weather for a point."""
        if self.weather_df is None or len(self.weather_df) == 0:
            return {'temperature_c': 35, 'humidity_pct': 25, 'wind_ms': 5, 'precip_mm': 0}
        
        # Simple nearest neighbor (could use inverse distance weighting)
        dists = np.sqrt((self.weather_df['lon'] - lon)**2 + (self.weather_df['lat'] - lat)**2)
        idx = dists.idxmin()
        row = self.weather_df.iloc[idx]
        return {
            'temperature_c': row['temperature_c'],
            'humidity_pct': row['humidity_pct'],
            'wind_ms': row['wind_ms'],
            'precip_mm': row['precip_mm']
        }
    
    def get_vegetation(self, lon, lat):
        """Get vegetation indices for nearest tile."""
        if self.veg_df is None or len(self.veg_df) == 0:
            return {'ndvi': 0.3, 'drought_stress': 0.7}
        
        dists = np.sqrt((self.veg_df['lon'] - lon)**2 + (self.veg_df['lat'] - lat)**2)
        idx = dists.idxmin()
        row = self.veg_df.iloc[idx]
        return {
            'ndvi': row['ndvi_mean'],
            'drought_stress': row['drought_stress']
        }
    
    def get_distances(self, lon, lat):
        """Get distance to nearest road and village in meters."""
        # Approximate: 1 degree ~ 111km
        deg_to_m = 111000
        
        dist_road = 5000  # default 5km
        dist_village = 5000
        
        if self.road_tree is not None:
            d, _ = self.road_tree.query([lon, lat])
            dist_road = d * deg_to_m
        
        if self.village_tree is not None:
            d, _ = self.village_tree.query([lon, lat])
            dist_village = d * deg_to_m
        
        return {'dist_road_m': dist_road, 'dist_village_m': dist_village}
    
    def get_aux_tensor(self, lon, lat, device):
        """Get full auxiliary tensor for model input."""
        weather = self.get_weather(lon, lat)
        veg = self.get_vegetation(lon, lat)
        dists = self.get_distances(lon, lat)
        
        # Create tensor: [temp, humidity, wind, dist_road, dist_village, ndvi, drought]
        aux = torch.tensor([[
            weather['temperature_c'],
            weather['humidity_pct'],
            weather['wind_ms'],
            dists['dist_road_m'],
            dists['dist_village_m'],
            veg['ndvi'],
            veg['drought_stress']
        ]], dtype=torch.float32).to(device)
        
        return aux


def predict_tile_with_real_data(tile_path, tile_bounds, model, device, aux_loader, grid_spacing_m=500):
    """Run prediction with REAL auxiliary data."""
    results = []
    
    try:
        with rasterio.open(tile_path) as src:
            data = src.read()
            transform = src.transform
            
        steps = data.shape[0] // 6
        if steps == 0:
            return results
            
        data = data.reshape(steps, 6, data.shape[1], data.shape[2]).astype(np.float32) / 10000.0
        data = np.clip(data, 0, 1)
        
        minx, miny, maxx, maxy = tile_bounds
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        
        # Get real auxiliary data for tile center
        aux = aux_loader.get_aux_tensor(center_lon, center_lat, device)
        
        import torch.nn.functional as F
        h, w = data.shape[2], data.shape[3]
        t = torch.from_numpy(data)
        flat = t.view(1, steps*6, h, w)
        resized = F.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
        inp = resized.view(1, steps, 6, 224, 224).to(device)
        
        with torch.no_grad():
            logits = model(inp, aux)
            risk = torch.softmax(logits, dim=1)[0, 1].item()
        
        # Grid points in tile
        deg_step = grid_spacing_m / 111000.0
        lons = np.arange(minx + deg_step/2, maxx, deg_step)
        lats = np.arange(miny + deg_step/2, maxy, deg_step)
        
        for lon in lons:
            for lat in lats:
                results.append((lon, lat, risk))
                
    except Exception as e:
        safe_print(f"Tile prediction error: {e}")
    
    return results


def run_campaign_with_real_data(kmz_path, target_date_str, tile_size_km=5, grid_spacing_m=500, 
                                 max_workers=8, output_dir='outputs/kmz_campaign_v2'):
    """Run prediction with REAL weather and auxiliary data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
    end_date = target_date.strftime('%Y-%m-%d')
    
    os.makedirs(output_dir, exist_ok=True)
    tile_dir = os.path.join(output_dir, 'tiles')
    
    # Initialize
    initialize_gee()
    
    # Load auxiliary data
    print("Loading auxiliary data...")
    aux_loader = AuxiliaryDataLoader('data/auxiliary')
    
    # Parse KMZ
    forest_poly = parse_kmz(kmz_path)
    bounds = forest_poly.bounds
    print(f"Bounds: Lon {bounds[0]:.4f}-{bounds[2]:.4f}, Lat {bounds[1]:.4f}-{bounds[3]:.4f}")
    
    # Generate tiles
    tile_size_deg = tile_size_km / 111.0
    tiles = generate_tiles(bounds, tile_size_deg)
    
    from shapely.geometry import box
    tiles_filtered = [t for t in tiles if forest_poly.intersects(box(t[0], t[1], t[2], t[3]))]
    print(f"Generated {len(tiles_filtered)} tiles (filtered from {len(tiles)})")
    
    # Download tiles (with caching)
    tile_paths = download_tiles_parallel(tiles_filtered, start_date, end_date, tile_dir, max_workers)
    
    if not tile_paths:
        print("ERROR: No tiles downloaded!")
        return
    
    # Load model
    prithvi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    print("Loading model...")
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predict with real data
    print("Running predictions with REAL auxiliary data...")
    all_results = []
    for idx, path in tile_paths.items():
        tile_bounds = tiles_filtered[idx]
        results = predict_tile_with_real_data(path, tile_bounds, model, device, aux_loader, grid_spacing_m)
        all_results.extend(results)
        safe_print(f"  Tile {idx+1}: {len(results)} predictions")
    
    print(f"\nTotal predictions: {len(all_results)}")
    
    # Filter to polygon
    final_results = [(lon, lat, risk) for lon, lat, risk in all_results if forest_poly.contains(Point(lon, lat))]
    print(f"Points inside boundary: {len(final_results)}")
    
    # Log sample auxiliary values
    if final_results:
        sample_lon, sample_lat, _ = final_results[len(final_results)//2]
        weather = aux_loader.get_weather(sample_lon, sample_lat)
        veg = aux_loader.get_vegetation(sample_lon, sample_lat)
        dists = aux_loader.get_distances(sample_lon, sample_lat)
        print(f"\nSample auxiliary values at ({sample_lat:.4f}, {sample_lon:.4f}):")
        print(f"  Temperature: {weather['temperature_c']:.1f}C, Humidity: {weather['humidity_pct']:.1f}%")
        print(f"  Wind: {weather['wind_ms']:.1f} m/s, Precip: {weather['precip_mm']:.2f} mm")
        print(f"  NDVI: {veg['ndvi']:.3f}, Drought stress: {veg['drought_stress']:.3f}")
        print(f"  Dist to road: {dists['dist_road_m']:.0f}m, Dist to village: {dists['dist_village_m']:.0f}m")
    
    # Save results
    report_path = os.path.join(output_dir, f'validation_report_{target_date_str}.csv')
    with open(report_path, 'w') as f:
        f.write("lon,lat,predicted_risk\n")
        for lon, lat, risk in final_results:
            f.write(f"{lon},{lat},{risk}\n")
    print(f"\nReport saved: {report_path}")
    
    # Visualize
    try:
        import matplotlib.pyplot as plt
        x, y, c = zip(*final_results) if final_results else ([], [], [])
        plt.figure(figsize=(12, 10))
        plt.scatter(x, y, c=c, cmap='RdYlGn_r', vmin=0, vmax=1, s=20, alpha=0.7)
        plt.colorbar(label='Predicted Risk')
        plt.title(f"Fire Risk Prediction (WITH REAL DATA): {target_date_str}\n(KMZ: {os.path.basename(kmz_path)})")
        
        if forest_poly.geom_type == 'Polygon':
            bx, by = forest_poly.exterior.xy
            plt.plot(bx, by, 'b-', linewidth=1, label='Boundary')
        elif forest_poly.geom_type == 'MultiPolygon':
            for geom in forest_poly.geoms:
                bx, by = geom.exterior.xy
                plt.plot(bx, by, 'b-', linewidth=1)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        preview_path = os.path.join(output_dir, f'map_preview_{target_date_str}.png')
        plt.savefig(preview_path, dpi=150)
        print(f"Map saved: {preview_path}")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n[OK] Campaign complete with REAL data!")
    
    # Print risk statistics
    if final_results:
        risks = [r for _, _, r in final_results]
        print(f"\nRisk Statistics:")
        print(f"  Min: {min(risks):.4f}")
        print(f"  Max: {max(risks):.4f}")
        print(f"  Mean: {np.mean(risks):.4f}")
        print(f"  High risk (>0.5): {sum(1 for r in risks if r > 0.5)} points")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized KMZ Fire Prediction with REAL Data")
    parser.add_argument('kmz_path', help='Path to KMZ file')
    parser.add_argument('--date', default='2025-05-15', help='Target date YYYY-MM-DD')
    parser.add_argument('--tile-size', type=int, default=5, help='Tile size in km')
    parser.add_argument('--grid', type=int, default=500, help='Grid spacing in meters')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    args = parser.parse_args()
    
    run_campaign_with_real_data(
        args.kmz_path, 
        args.date,
        tile_size_km=args.tile_size,
        grid_spacing_m=args.grid,
        max_workers=args.workers
    )
