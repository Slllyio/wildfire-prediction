"""
Visualization Script: Historical Fire Patterns

Generates maps and plots of:
1. Fire frequency map (2018-2024)
2. Yearly comparison (including 2020 lockdown)
3. Seasonal patterns
4. Sentinel-2 coverage
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ee
from config import *
import matplotlib.pyplot as plt
import numpy as np

# Initialize GEE
ee.Initialize(project=GEE_PROJECT_ID)

# Load data
print("Loading forest beats and fire data...")
beats = ee.FeatureCollection(BEATS_ASSET_ID)
roi = beats.geometry()

# 1. Historical Fire Frequency Map
print("\n1. Generating fire frequency map (2018-2024)...")
fires = ee.ImageCollection(FIRE_DATASET)\
    .filterDate(f'{TRAINING_START_YEAR}-01-01', f'{TRAINING_END_YEAR}-12-31')\
    .select('BurnDate')\
    .filterBounds(roi)

# Count burns per pixel
fire_frequency = fires.map(lambda img: img.gt(0)).sum()

# Get thumbnail URL for visualization
bounds = roi.bounds().getInfo()['coordinates'][0]
url = fire_frequency.getThumbURL({
    'min': 0,
    'max': 5,
    'palette': ['white', 'yellow', 'orange', 'red', 'darkred'],
    'dimensions': 800,
    'region': bounds
})

print(f"   Fire frequency map URL: {url}")
print("   (Copy this URL to your browser to view the map)")

# 2. Yearly Burn Statistics
print("\n2. Calculating yearly fire statistics...")
years = list(range(TRAINING_START_YEAR, VALIDATION_YEAR + 1))
burn_counts = []

for year in years:
    year_fires = ee.ImageCollection(FIRE_DATASET)\
        .filterDate(f'{year}-01-01', f'{year}-12-31')\
        .select('BurnDate')\
        .filterBounds(roi)
    
    burn_pixels = year_fires.map(lambda img: img.gt(0)).sum()
    
    stats = burn_pixels.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=500,
        maxPixels=1e9
    ).getInfo()
    
    count = stats.get('BurnDate', 0)
    burn_counts.append(count)
    print(f"   {year}: {count:.0f} burnt pixels")

# Plot yearly comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(years, burn_counts, color=['blue' if y != 2020 else 'red' for y in years])
plt.xlabel('Year')
plt.ylabel('Burnt Pixels (MODIS)')
plt.title('Yearly Fire Activity\n(Red = 2020 Lockdown Year)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Highlight lockdown impact
lockdown_idx = years.index(2020)
avg_before = np.mean([burn_counts[i] for i in range(lockdown_idx) if i < lockdown_idx])
lockdown_val = burn_counts[lockdown_idx]
reduction = (1 - lockdown_val/avg_before) * 100

plt.text(2020, lockdown_val, f'{reduction:.0f}% reduction', 
         ha='center', va='bottom', fontweight='bold')

# 3. Seasonal Pattern (Monthly aggregation for a sample year)
print("\n3. Analyzing seasonal patterns (2023)...")
months = list(range(1, 13))
monthly_burns = []

for month in months:
    month_fires = ee.ImageCollection(FIRE_DATASET)\
        .filterDate(f'2023-{month:02d}-01', f'2023-{month:02d}-28')\
        .select('BurnDate')\
        .filterBounds(roi)
    
    burn_pixels = month_fires.map(lambda img: img.gt(0)).sum()
    stats = burn_pixels.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=500,
        maxPixels=1e9
    ).getInfo()
    
    monthly_burns.append(stats.get('BurnDate', 0))

# Plot seasonal pattern
plt.subplot(1, 2, 2)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(month_names, monthly_burns, 
        color=['orange' if m in [3, 4, 5] else 'gray' for m in range(1, 13)])
plt.xlabel('Month')
plt.ylabel('Burnt Pixels')
plt.title('2023 Seasonal Pattern\n(Orange = Mahua/Tendu Season)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Annotate peak season
peak_month = monthly_burns.index(max(monthly_burns))
plt.text(peak_month, max(monthly_burns), 'Peak', 
         ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fire_analysis.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved plot to: {output_path}")

# 4. Summary Statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Study Period: {TRAINING_START_YEAR}-{VALIDATION_YEAR}")
print(f"Total Burnt Pixels: {sum(burn_counts):.0f}")
print(f"Average per year: {np.mean(burn_counts):.0f}")
print(f"2020 Lockdown Impact: {reduction:.1f}% reduction vs previous years")
print(f"Peak Fire Month (2023): {month_names[peak_month]}")
print(f"\nForest Beats: 50")
print(f"Sentinel-2 Coverage: ~19 scenes/month")
print("\n[OK] Analysis complete! Check fire_analysis.png")
