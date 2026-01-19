"""
Fine-tune Prithvi Fire Classifier Head
Trains the classification head on FIRMS-labeled data while keeping encoder frozen.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import pandas as pd
import os
import sys
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.prithvi_fire import PrithviFirePredictor, FocalLoss


from models.dataset import FireDataset


FEATURE_COLUMNS = [
    'elevation', 'slope', 'aspect',
    'era5_temp_d1', 'era5_wind_speed_d1', 'era5_precip_d1', 'era5_vpd_d1', 'era5_soil_moisture_d1',
    'modis_ndvi_d1', 'modis_evi_d1'
]

def load_all_training_data(data_dirs):
    """Load training data from multiple directories (or recursively find labels.csv)."""
    all_paths = []
    all_labels = []
    all_aux = []
    
    # 1. Expand input directories to find all 'labels.csv' files
    label_files = []
    for d in data_dirs:
        if os.path.isfile(d) and d.endswith('labels.csv'):
            label_files.append(d)
        elif os.path.isdir(d):
            # Check root
            if os.path.exists(os.path.join(d, 'labels.csv')):
                label_files.append(os.path.join(d, 'labels.csv'))
            # Check subdirs (one level deep is usually enough, but let's do walk)
            for root, dirs, files in os.walk(d):
                if 'labels.csv' in files:
                    # Avoid adding twice if root was checked
                    p = os.path.join(root, 'labels.csv')
                    if p not in label_files:
                        label_files.append(p)

    print(f"Found {len(label_files)} label files: {label_files}")

    for labels_file in label_files:
        data_dir = os.path.dirname(labels_file)
        print(f"Loading {labels_file}...")
        
        df = pd.read_csv(labels_file)
        
        # Check if we have feature columns
        has_features = all(col in df.columns for col in FEATURE_COLUMNS)
        if not has_features:
            print(f"  Warning: {labels_file} missing some features. Filling with zeros.")
            # Fill missing with 0
            for col in FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
                    
        for _, row in df.iterrows():
            path = os.path.join(data_dir, row['filename'])
            # Support full path in CSV or relative to dir
            if not os.path.exists(path) and os.path.isabs(row['filename']):
                 path = row['filename']
                 
            if os.path.exists(path):
                all_paths.append(path)
                all_labels.append(row['label'])
                
                # Extract features
                aux = row[FEATURE_COLUMNS].values.astype(np.float32)
                # handle NaNs
                aux = np.nan_to_num(aux)
                all_aux.append(aux)
    
    return all_paths, all_labels, np.array(all_aux)


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4):
    """Train the classifier head."""
    
    # Only train classifier parameters
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    
    # Use focal loss for class imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Higher alpha for minority class
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (imgs, aux, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            aux = aux.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, aux)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_fire_correct = 0
        val_fire_total = 0
        
        with torch.no_grad():
            for imgs, aux, labels in val_loader:
                imgs = imgs.to(device)
                aux = aux.to(device)
                labels = labels.to(device)
                
                logits = model(imgs, aux)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Fire class accuracy (important for imbalanced data)
                fire_mask = labels == 1
                if fire_mask.sum() > 0:
                    val_fire_correct += (preds[fire_mask] == 1).sum().item()
                    val_fire_total += fire_mask.sum().item()
        
        val_acc = val_correct / val_total
        fire_recall = val_fire_correct / val_fire_total if val_fire_total > 0 else 0
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.3f}, "
              f"Val Acc={val_acc:.3f}, "
              f"Fire Recall={fire_recall:.3f}")
        
        # Save best model based on fire recall (we care about detecting fires)
        if fire_recall > best_val_acc:
            best_val_acc = fire_recall
            best_model_state = model.state_dict().copy()
    
    return best_model_state, best_val_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs='+', 
                        default=['data/training', 'data/training_2023', 'data/training_2022'],
                        help='Training data directories')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output', default='models/checkpoints/prithvi_fire_finetuned.pth',
                        help='Output checkpoint path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load training data
    # Load training data
    print("Loading training data...")
    all_paths, all_labels, all_aux = load_all_training_data(args.data_dirs)
    
    print(f"Total samples: {len(all_paths)}")
    print(f"  Fire: {sum(all_labels)}")
    print(f"  Non-fire: {len(all_labels) - sum(all_labels)}")
    print(f"  Aux features: {all_aux.shape[1]}")
    
    if len(all_paths) == 0:
        print("ERROR: No training data found!")
        return
    
    # Train/val split (stratified)
    # We need to split paths, labels AND aux
    indices = np.arange(len(all_paths))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    train_aux = all_aux[train_idx]
    
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    val_aux = all_aux[val_idx]
    
    print(f"\nTrain: {len(train_paths)} (Fire: {sum(train_labels)})")
    print(f"Val: {len(val_paths)} (Fire: {sum(val_labels)})")
    
    # Create datasets
    train_dataset = FireDataset(train_paths, train_labels, train_aux)
    val_dataset = FireDataset(val_paths, val_labels, val_aux)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading Prithvi model...")
    # Script is in wildfire_v2/scripts/
    # Project root is wildfire_v2/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prithvi_path = os.path.join(project_root, 'models', 'prithvi-eo-2.0')
    
    print(f"Loading from: {prithvi_path}")
    if not os.path.exists(prithvi_path):
        print(f"ERROR: Model path does not exist: {prithvi_path}")
        return

    model = PrithviFirePredictor(prithvi_model_path=prithvi_path, 
                               num_aux_features=all_aux.shape[1],
                               freeze_encoder=True)
    model.to(device)
    
    # Train
    print("\nStarting training...")
    best_state, best_acc = train_model(
        model, train_loader, val_loader, device, 
        epochs=args.epochs, lr=args.lr
    )
    
    # Save best model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(best_state, args.output)
    print(f"\n[OK] Best model saved to {args.output}")
    print(f"Best fire recall: {best_acc:.3f}")


if __name__ == "__main__":
    main()
