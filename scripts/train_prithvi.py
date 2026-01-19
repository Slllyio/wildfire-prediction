"""
Training Script for Prithvi-EO Fire Prediction

Fine-tunes Prithvi-EO on Central India wildfire data.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os
from tqdm import tqdm
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from models.dataset import FirePredictionDataset, collate_fn
from models.prithvi_fire import PrithviFirePredictor, FocalLoss

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        timeseries = batch['timeseries'].to(device)
        auxiliary = batch['auxiliary'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(timeseries, auxiliary)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            timeseries = batch['timeseries'].to(device)
            auxiliary = batch['auxiliary'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(timeseries, auxiliary)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'training')
    patches_dir = os.path.join(data_dir, 'patches')
    
    # Load multiple years for training (2018, 2021, 2024)
    train_files = [
        os.path.join(data_dir, 'samples_2018.geojson'),
        os.path.join(data_dir, 'samples_2021.geojson'),
        os.path.join(data_dir, 'samples_2024.geojson')
    ]
    val_files = [os.path.join(data_dir, 'samples_2025.geojson')]

    
    print("Loading datasets...")
    train_datasets = []
    for f in train_files:
        if os.path.exists(f):
            train_datasets.append(FirePredictionDataset(samples_file=f, data_dir=patches_dir))
    
    val_datasets = []
    for f in val_files:
        if os.path.exists(f):
            val_datasets.append(FirePredictionDataset(samples_file=f, data_dir=patches_dir))
            
    if not train_datasets or not val_datasets:
        print("Error: No valid datasets found. Check if patches are downloaded.")
        return
        
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # Better for Windows trial
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing Prithvi-EO model...")
    model = PrithviFirePredictor(
        prithvi_model_path='../models/prithvi-eo-2.0',
        freeze_encoder=False  # Fine-tune encoder
    ).to(device)

    # Resume from checkpoint if available
    checkpoint_path = os.path.join(base_dir, 'models', 'checkpoints', 'prithvi_fire_best.pth')
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Checkpoint loaded. Resuming training (fine-tuning).")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting training from scratch.")
    
    # Loss & optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/checkpoints/prithvi_fire_best.pth')
            print(f"[SAVED] Saved best model (val_acc: {val_acc*100:.2f}%)")
    
    # Save training history
    with open('../models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[SAVED] Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    main()
