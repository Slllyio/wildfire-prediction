"""
PyTorch Dataset for Prithvi-EO Fire Prediction

Loads HLS time series + auxiliary features for training.
"""
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
import torch.nn.functional as F

class FireDataset(Dataset):
    """
    Dataset for fire/non-fire classification using Prithvi-EO.
    
    Args:
        file_paths (list): List of paths to TIF files
        labels (list): List of integer labels (0: non-fire, 1: fire)
        aux_features (np.array): Array of shape (N, num_features) containing environmental variables
    """
    
    def __init__(self, file_paths, labels, aux_features=None):
        self.file_paths = file_paths
        self.labels = labels
        # Default to dummy features if none provided (for legacy support)
        if aux_features is not None:
             self.aux_features = torch.from_numpy(aux_features).float()
        else:
             self.aux_features = torch.zeros((len(labels), 10), dtype=torch.float32)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess patch
        try:
            with rasterio.open(self.file_paths[idx]) as src:
                data = src.read()
        except:
             # Return a zero tensor if file read fails (robustness)
             data = np.zeros((6, 64, 64), dtype=np.float32)
        
        # Reshape: (T*C, H, W) -> (T, C, H, W)
        n_bands = data.shape[0]
        n_steps = n_bands // 6
        
        if n_steps == 0:
            # Fallback for malformed data
            data = np.zeros((6, 6, 64, 64), dtype=np.float32)
            n_steps = 6
        else:
            data = data.reshape(n_steps, 6, data.shape[1], data.shape[2]).astype(np.float32)
        
        # Normalize
        data = data / 10000.0
        data = np.clip(data, 0, 1)
        
        # Resize to 224x224
        t = torch.from_numpy(data)
        flat = t.view(1, n_steps*6, t.shape[2], t.shape[3])
        resized = F.interpolate(flat, size=(224, 224), mode='bilinear', align_corners=False)
        img = resized.view(n_steps, 6, 224, 224)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        aux = self.aux_features[idx]
        
        return img, aux, label

def collate_fn(batch):
    """Custom collate function"""
    imgs = torch.stack([item[0] for item in batch])
    aux = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return imgs, aux, labels
