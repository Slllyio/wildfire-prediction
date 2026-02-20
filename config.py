"""
Wildfire Prediction System V2 - Prithvi-EO Based
Configuration settings
"""

# Google Earth Engine
GEE_PROJECT_ID = 'monkhub-internal-enetra-dev'
BEATS_ASSET_ID = 'projects/ee-akshayr1/assets/bichua'

# Study Area (Chhindwara District)
AOI_BOUNDS = {
    'min_lon': 78.20,
    'max_lon': 79.50,
    'min_lat': 21.40,
    'max_lat': 22.90
}

# Prithvi-EO Model
PRITHVI_MODEL_ID = 'ibm-nasa-geospatial/Prithvi-EO-2.0'
PRITHVI_INPUT_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # HLS compatible S2 bands
TEMPORAL_WINDOW_MONTHS = 6  # How many months of history to feed the model
TEMPORAL_STEP_DAYS = 10  # Sample every N days

# Training Data
TRAINING_START_YEAR = 2018
TRAINING_END_YEAR = 2024
VALIDATION_YEAR = 2025
FIRE_DATASET = 'MODIS/061/MCD64A1'  # Burnt area monthly

# Seasonal Patterns (Central India)
MAHUA_SEASON_START_DOY = 75   # March 16
MAHUA_SEASON_END_DOY = 110    # April 20

TENDU_SEASON_START_DOY = 100  # April 10
TENDU_SEASON_END_DOY = 130    # May 10

# Model Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
CLASS_BALANCE_RATIO = 10  # Negative:Positive samples

# Output Resolution
OUTPUT_RESOLUTION_M = 20  # Sentinel-2 resolution
