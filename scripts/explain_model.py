"""
Script to explain model predictions using SHAP (SHapley Additive exPlanations).
Generates feature importance plots to understand which inputs drive fire risk.
"""
import torch
import shap
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.prithvi_fire import PrithviFirePredictor
from models.dataset import FirePredictionDataset

def explain_model(model, data_loader, num_samples=100):
    """
    Run SHAP analysis on a subset of data.
    """
    print(f"Initializing SHAP explainer with {num_samples} background samples...")
    
    # Get background data
    batch = next(iter(data_loader))
    # New loader returns (img, aux, label) tuple
    background_ts, background_aux, _ = batch
    
    model.eval()
    
    # Create a wrapper function that takes only auxiliary features
    # (fixing timeseries to reduce complexity)
    def model_predict_aux(aux_np):
        """Wrapper that predicts using fixed timeseries and variable auxiliary features"""
        aux_tensor = torch.from_numpy(aux_np).float()
        # Use first sample's timeseries as background
        ts_fixed = background_ts[0:1].repeat(aux_tensor.shape[0], 1, 1, 1, 1)
        
        with torch.no_grad():
            logits = model(ts_fixed, aux_tensor)
            probs = torch.softmax(logits, dim=1)
            # Return probability of fire (class 1)
            return probs[:, 1].cpu().numpy()
    
    # Use KernelExplainer (model-agnostic, more robust)
    # Limit background to 10 samples for speed
    background_aux_np = background_aux[:10].numpy()
    
    print("Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_aux, background_aux_np)
    
    # Get test samples (just 5 for quick demo)
    # Get test samples (just 5 for quick demo)
    batch = next(iter(data_loader))
    _, test_aux, _ = batch
    test_aux = test_aux[0:5]
    test_aux_np = test_aux.numpy()
    
    print("Calculating SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(test_aux_np, nsamples=50)
    
    # Visualization for Auxiliary Features
    feature_names = [
        'Elevation', 'Slope', 'Aspect',
        'Temp (D-1)', 'Wind (D-1)', 'Precip (D-1)', 'VPD (D-1)', 'Soil Moist (D-1)',
        'NDVI (D-1)', 'EVI (D-1)'
    ]
    
    print("Generating summary plot for auxiliary features...")
    # Summary plot expects (N, M) array
    shap.summary_plot(shap_values, test_aux_np, feature_names=feature_names, show=False)
    plt.savefig('shap_auxiliary_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP analysis complete. Plot saved to: shap_auxiliary_summary.png")

if __name__ == "__main__":
    device = torch.device('cpu') # SHAP often simpler on CPU for small batches
    
    # 1. Load Data
    # Import loading logic from finetune_classifier (or duplicate it for simplicity)
    # Let's duplicate briefly to avoid circular deps if finetune imports other things, 
    # but ideally we'd refactor load_all_training_data to utils. For now, manual load.
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, 'data', 'training')
    labels_file = os.path.join(data_dir, 'labels.csv')
    
    if os.path.exists(labels_file):
        import pandas as pd
        df = pd.read_csv(labels_file)
        # Select first 50 samples for SHAP
        df = df.head(50)
        
        paths = []
        labels = []
        aux_list = []
        
        feature_cols = [
            'elevation', 'slope', 'aspect',
            'era5_temp_d1', 'era5_wind_speed_d1', 'era5_precip_d1', 'era5_vpd_d1', 'era5_soil_moisture_d1',
            'modis_ndvi_d1', 'modis_evi_d1'
        ]
        
        for _, row in df.iterrows():
            p = os.path.join(data_dir, row['filename'])
            if os.path.exists(p):
                paths.append(p)
                labels.append(row['label'])
                # Safe extract
                vals = []
                for c in feature_cols:
                    vals.append(row.get(c, 0.0))
                aux_list.append(np.array(vals, dtype=np.float32))
        
        aux_features = np.array(aux_list)
        
        from models.dataset import FireDataset
        dataset = FireDataset(paths, labels, aux_features)
        
        # DataLoader
        # Need to return batch, but dataset returns tuple (img, aux, label)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
        
        # Load model
        prithvi_path = os.path.join(project_dir, 'models', 'prithvi-eo-2.0')
        checkpoint_path = os.path.join(project_dir, 'models', 'checkpoints', 'prithvi_fire_finetuned.pth')
        
        model = PrithviFirePredictor(prithvi_model_path=prithvi_path, num_aux_features=len(feature_cols))
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        else:
            print("Warning: No checkpoint found, using random weights for explainer.")
            
        model.to(device)
        
        explain_model(model, loader)
        
    else:
        print(f"No labels.csv found at {labels_file}. Run prepare_training_data.py first.")
