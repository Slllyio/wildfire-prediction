"""
Train Random Forest Classifier for Fire Prediction.
Extracts statistical features from S2 patches and trains a robust RF model.
"""
import rasterio
import numpy as np
import pandas as pd
import os
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def extract_features(tif_path):
    """
    Extract statistical features from a Sentinel-2 patch.
    Bands: [Blue, Green, Red, NIR, SWIR1, SWIR2] x 6 time steps
    """
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()  # (Bands, H, W)
            
        # Reshape to (Time, Bands, H, W)
        # Assuming 6 bands per time step
        n_channels = 6
        n_steps = data.shape[0] // n_channels
        
        if n_steps == 0:
            return None
        
        data = data.reshape(n_steps, n_channels, data.shape[1], data.shape[2])
        
        # Calculate statistics per band across time and space
        # We want to capture the "fire signature" (dryness, vegetation loss)
        
        # 1. Temporal mean (average condition)
        time_mean = np.mean(data, axis=0) # (6, H, W)
        
        # 2. Spatial mean of temporal mean (global patch stats)
        band_means = np.mean(time_mean, axis=(1, 2))
        band_stds  = np.std(time_mean, axis=(1, 2))
        
        # 3. Indices
        # Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi_mean = (band_means[3] - band_means[2]) / (band_means[3] + band_means[2] + 1e-6)
        
        # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        nbr_mean = (band_means[3] - band_means[5]) / (band_means[3] + band_means[5] + 1e-6)
        
        # Moisture Stress = SWIR1 / NIR
        si_mean = band_means[4] / (band_means[3] + 1e-6)
        
        features = list(band_means) + list(band_stds) + [ndvi_mean, nbr_mean, si_mean]
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return None

def load_dataset(data_dirs):
    X = []
    y = []
    
    print("Extracting features from training data...")
    count = 0
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue
            
        labels_file = os.path.join(data_dir, 'labels.csv')
        if not os.path.exists(labels_file):
            continue
            
        df = pd.read_csv(labels_file)
        
        for _, row in df.iterrows():
            path = os.path.join(data_dir, row['filename'])
            if os.path.exists(path):
                feats = extract_features(path)
                if feats is not None:
                    # Add dummy auxiliary features for training (Temp, Hum, Wind, etc.)
                    # Ideally we extract these from the location, but for now we rely on spectral signal
                    # Fire pixels have distinct spectral signature regardless of weather
                    X.append(feats)
                    y.append(row['label'])
                    count += 1
                    if count % 50 == 0:
                        print(f"  Processed {count} samples...")
    
    return np.array(X), np.array(y)

def main():
    data_dirs = ['data/training', 'data/training_2023', 'data/training_2022']
    output_model = 'models/checkpoints/rf_fire_model.pkl'
    
    # 1. Load Data
    X, y = load_dataset(data_dirs)
    print(f"Total samples: {len(X)}")
    print(f"Fire samples: {sum(y)}")
    
    if len(X) == 0:
        print("No data found!")
        return

    # Handle NaNs
    X = np.nan_to_num(X)

    # Manual Oversampling of Minority Class
    fire_idx = np.where(y == 1)[0]
    nofire_idx = np.where(y == 0)[0]
    
    # Replicate fire samples to match non-fire count
    if len(fire_idx) > 0:
        multiplier = len(nofire_idx) // len(fire_idx)
        fire_idx_repeated = np.repeat(fire_idx, multiplier)
        
        # Combine
        indices = np.concatenate([nofire_idx, fire_idx_repeated])
        np.random.shuffle(indices)
        
        X = X[indices]
        y = y[indices]
        print(f"Oversampled dataset: {len(X)} samples (Fire: {sum(y)})")
    
    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\nValidation Results:")
    y_pred = rf.predict(X_val)
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # 5. Save
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(rf, output_model)
    print(f"\nModel saved to {output_model}")

if __name__ == "__main__":
    main()
