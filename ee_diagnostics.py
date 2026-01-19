import ee
try:
    lat, lon = 22.5, 78.5
    point = ee.Geometry.Point([lon, lat])
    
    print(f"Checking assets at ({lat}, {lon}) from 2016-2024:\n")
    
    for year in range(2016, 2025):
        print(f"--- Year {year} ---")
        
        # 1. Check Embeddings
        res = {}
        for coll in ['GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL', 'GOOGLE/SATELLITE_EMBEDDING/V1']:
            try:
                c = ee.ImageCollection(coll).filterBounds(point)
                if coll.endswith('ANNUAL'):
                    c = c.filterDate(f"{year}-01-01", f"{year}-12-31")
                size = c.size().getInfo()
                res[coll] = size
            except: res[coll] = "Error"
        print(f"  Embeddings: {res}")
        
        # 2. Check SAR
        try:
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(point).filterDate(f"{year}-03-01", f"{year}-04-01")
            print(f"  SAR (March {year}): {s1.size().getInfo()} images")
        except: print(f"  SAR: Error")
        
        # 3. Check HLS
        try:
            hls = ee.ImageCollection('NASA/HLS/HLSL30/v002').filterBounds(point).filterDate(f"{year}-03-01", f"{year}-04-01")
            print(f"  HLS (March {year}): {hls.size().getInfo()} images")
        except: print(f"  HLS: Error")

except Exception as e:
    print(f"Error during overall execution: {e}")
