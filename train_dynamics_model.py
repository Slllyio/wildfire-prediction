import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_name
import joblib
import os

def train_and_impute(input_csv='training_features_hybrid.csv', output_csv='training_features_final.csv'):
    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} not found. Please run the extraction script first.")
        return

    print(f"📖 Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Prepare Columns
    # Identify embedding columns
    embedding_cols = [c for c in df.columns if c.startswith('feature_')]
    feature_cols = ['wind_u', 'wind_v'] + embedding_cols
    
    # Targets
    # We use sin/cos for direction to handle circularity
    df['dir_sin'] = np.sin(df['fire_spread_dir'])
    df['dir_cos'] = np.cos(df['fire_spread_dir'])
    target_cols = ['fire_ros_proxy', 'dir_sin', 'dir_cos']
    
    # 2. Split Data
    # Training set: MODIS verified fires
    train_mask = df['is_modis_verified'] == 1
    # Impute set: VIIRS fires that are not MODIS verified
    impute_mask = (df['label'] == 1) & (df['is_modis_verified'] == 0)
    
    if train_mask.sum() < 5:
        print("⚠ Too few MODIS verified fires to train a dynamics model. Using simple defaults.")
        # Fallback logic could go here
        return

    print(f"🔥 Training on {train_mask.sum()} verified MODIS fire events...")
    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, target_cols].fillna(0)
    
    # 3. Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Dynamics Model Trained.")
    
    # 4. Impute for VIIRS Fires
    if impute_mask.sum() > 0:
        print(f"🔮 Predicting dynamics for {impute_mask.sum()} unverified VIIRS fires...")
        X_impute = df.loc[impute_mask, feature_cols].fillna(0)
        predictions = model.predict(X_impute)
        
        df.loc[impute_mask, 'fire_ros_proxy'] = predictions[:, 0]
        # Reconstruct angle from sin/cos
        df.loc[impute_mask, 'fire_spread_dir'] = np.arctan2(predictions[:, 1], predictions[:, 2])
        
    # 5. Save Final Dataset
    df.to_csv(output_csv, index=False)
    print(f"💾 Final enriched dataset saved to {output_csv}")
    
    # Save model for future inference
    joblib.dump(model, 'fire_dynamics_model.joblib')
    print("💾 Model saved to fire_dynamics_model.joblib")

if __name__ == '__main__':
    train_and_impute()
