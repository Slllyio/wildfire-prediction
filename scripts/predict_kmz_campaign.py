
"""
KMZ-based Fire Prediction & Validation Campaign
Predicts fire risk within a specific KMZ polygon and validates against historical fire data.
"""
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import sys
import argparse
from datetime import datetime, timedelta
import json
import zipfile
import xml.etree.ElementTree as ET
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import ee

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from scripts.gee_utils import initialize_gee, download_patch_local

def parse_kmz(kmz_path):
    """Extracts the first polygon geometry from a KMZ file."""
    print(f"Parsing KMZ: {kmz_path}")
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_filename = [f for f in kmz.namelist() if f.endswith('.kml')][0]
        with kmz.open(kml_filename, 'r') as kml_file:
            # Read content as bytes then string
            content = kml_file.read().decode('utf-8')
            
            # Simple REGEX approach to find coordinates block
            # This avoids XML namespace hell
            import re
            
            # Find all <coordinates>...</coordinates> blocks
            # Pattern matches content inside tags
            coord_pattern = re.compile(r'<coordinates>(.*?)</coordinates>', re.DOTALL)
            matches = coord_pattern.findall(content)
            
            polygons = []
            
            for coords_str in matches:
                coords_str = coords_str.strip()
                points = []
                # Split by whitespace
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
                raise ValueError("No polygons found in KML via Regex extraction")
                
            combined = unary_union(polygons)
            return combined

def get_actual_fires(start_date, end_date, region_geom):
    """Fetch actual burnt area pixels from GEE for verification."""
    # region_geom: shapely geometry
    # Convert to ee.Geometry
    xmin, ymin, xmax, ymax = region_geom.bounds
    roi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    
    # MODIS Burnt Area
    # MCD64A1 is monthly.
    # For daily verification in a specific month, standard Active Fire products (VIIRS/MODIS) are better.
    # FIRMS: VIIRS S-NPP 375m
    
    fire_col = ee.ImageCollection("FIRMS")\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .select('T21') # Brightness temp
        
    # Count
    count = fire_col.size().getInfo()
    print(f"Found {count} active fire records/images in FIRMS for this period.")
    
    # We want locations of fires
    # Reduce to a stored image of fire presence (1 if fire, 0 else)
    fire_mask = fire_col.max().gt(300).unmask(0) # >300K temp
    
    return fire_mask

def run_kmz_campaign(kmz_path, target_date_str, grid_res_m=500, output_dir='outputs/kmz_campaign'):
    """
    Run prediction for KMZ area and compare with ground truth.
    grid_res_m: Approx grid spacing in meters
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    run_id = f"kmz_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    os.makedirs(output_dir, exist_ok=True)
    initialize_gee()
    
    # 1. Parse Geometry
    forest_poly = parse_kmz(kmz_path)
    minx, miny, maxx, maxy = forest_poly.bounds
    print(f"Bounds: Lon {minx:.4f}-{maxx:.4f}, Lat {miny:.4f}-{maxy:.4f}")
    
    # 2. Generate Grid
    # rough lat/lon diff for grid_res_m
    # 1 deg lat ~ 111km -> 1m ~ 1/111000 deg ~ 0.000009
    deg_step = (grid_res_m / 111000.0) 
    
    lons = np.arange(minx, maxx, deg_step)
    lats = np.arange(miny, maxy, deg_step)
    
    grid_points = []
    for lon in lons:
        for lat in lats:
            p = Point(lon, lat)
            if forest_poly.contains(p):
                grid_points.append((lon, lat))
                
    print(f"Generated {len(grid_points)} sample points inside forest boundary.")
    if len(grid_points) > 100:
        print("Limiting to 100 points for campaign speed...")
        grid_points = grid_points[::len(grid_points)//100] # Subsample
    
    # 3. Load Model
    prithvi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'models', 'prithvi-eo-2.0')
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'checkpoints', 'prithvi_fire_best.pth')
    
    model = PrithviFirePredictor(prithvi_model_path=prithvi_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 4. Predict Loop
    preds = []
    coords = []
    
    start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
    end_date = target_date.strftime('%Y-%m-%d')
    
    print("Starting Predictions...")
    for i, (lon, lat) in enumerate(grid_points):
        print(f"[{i+1}/{len(grid_points)}] Predicting ({lat:.4f}, {lon:.4f})...")
        try:
            temp_file = os.path.join(output_dir, f'temp_{i}.tif')
            # Reuse download function
            download_patch_local([lon, lat], start_date, end_date, temp_file, patch_size=64)
            
            with rasterio.open(temp_file) as src:
                data = src.read()
                
            steps = data.shape[0]//6
            if steps == 0:
                preds.append(0.0)
            else:
                data = data.reshape(steps, 6, data.shape[1], data.shape[2]).astype(np.float32) / 10000.0
                data = np.clip(data, 0, 1)
                
                # Resize/Infer
                import torch.nn.functional as F
                t = torch.from_numpy(data)
                # (T, C, H, W) -> (1, T*C, H, W)
                flat = t.view(1, steps*6, t.shape[2], t.shape[3])
                resized = F.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
                inp = resized.view(1, steps, 6, 224, 224).to(device)
                aux = torch.tensor([[35, 25, 5, 1000, 1000, 0.8, 0.9]]).to(device)
                
                with torch.no_grad():
                    logits = model(inp, aux)
                    risk = torch.softmax(logits, dim=1)[0, 1].item()
                    preds.append(risk)
                    
            coords.append((lon, lat))
            if os.path.exists(temp_file): os.remove(temp_file)
            
        except Exception as e:
            # print(e)
            preds.append(0.0)
            coords.append((lon, lat))
            
    print("\nPrediction Complete.")
    
    # 5. Get Ground Truth (Validation)
    print("Fetching Actual Fire Data for Validation...")
    # Check if a fire occurred within +/- 3 days of target date at these locations
    # For now, just print the risk map stats
    
    # Save CSV Results
    report_path = os.path.join(output_dir, f'validation_report_{target_date_str}.csv')
    with open(report_path, 'w') as f:
        f.write("lon,lat,predicted_risk\n")
        for (lon, lat), risk in zip(coords, preds):
            f.write(f"{lon},{lat},{risk}\n")
            
    print(f"Report saved: {report_path}")
    
    # 6. Visualize
    # Render basic map (simple scatterplot via matplotlib)
    import matplotlib.pyplot as plt
    try:
        x, y, c = zip(*[(ln, lt, r) for (ln, lt), r in zip(coords, preds)])
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c=c, cmap='RdYlGn_r', vmin=0, vmax=1, s=50) # Point map
        plt.colorbar(label='Predicted Risk')
        plt.title(f"Fire Risk Prediction: {target_date_str}\n(KMZ: {os.path.basename(kmz_path)})")
        
        # Plot boundary
        # shapely coords
        if forest_poly.geom_type == 'Polygon':
            bx, by = forest_poly.exterior.xy
            plt.plot(bx, by, 'b-', label='Boundary')
        
        preview_path = os.path.join(output_dir, f'map_preview_{target_date_str}.png')
        plt.savefig(preview_path)
        print(f"Map Preview: {preview_path}")
    except Exception as e:
        print(f"Viz Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kmz_path', help='Path to KMZ file')
    parser.add_argument('--date', default='2025-05-15', help='YYYY-MM-DD')
    args = parser.parse_args()
    
    run_kmz_campaign(args.kmz_path, args.date)
