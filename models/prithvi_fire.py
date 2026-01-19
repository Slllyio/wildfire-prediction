"""
Prithvi-EO Fire Prediction Model

Fine-tunes Prithvi-EO encoder with custom classification head.
"""
import torch
import torch.nn as nn
import os
import sys
import json
from transformers import AutoModel

class PrithviFirePredictor(nn.Module):
    """
    Wildfire prediction model based on Prithvi-EO.
    
    Architecture:
    - Prithvi-EO encoder (frozen or fine-tuned)
    - Custom classification head with auxiliary features
    """
    
    def __init__(self, prithvi_model_path, num_aux_features=7, freeze_encoder=False):
        super().__init__()
        
        # Load pre-trained Prithvi-EO from local path manually
        print(f"Loading Prithvi-EO from {prithvi_model_path}...")
        
        # Add model directory to sys.path to import prithvi_mae
        abs_model_path = os.path.abspath(prithvi_model_path)
        if abs_model_path not in sys.path:
            sys.path.append(abs_model_path)
            
        try:
            from prithvi_mae import PrithviMAE
        except ImportError:
            # Fallback for relative path issues or if run from different dir
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'prithvi-eo-2.0'))
            from prithvi_mae import PrithviMAE
            
        # Load config
        with open(os.path.join(prithvi_model_path, 'config.json'), 'r') as f:
            full_config = json.load(f)
            # Use pretrained_cfg part for init
            model_cfg = full_config.get('pretrained_cfg', full_config)
            
        # Instantiate PrithviMAE
        # Filter kwargs that might not be in __init__ if needed, but assuming match
        # PrithviMAE expects specific args. Let's pass what we see in config.
        # Note: config might have lists for tuples, ensure conversion if needed.
        self.prithvi_model = PrithviMAE(
            img_size=model_cfg.get('img_size', 224),
            num_frames=model_cfg.get('num_frames', 4),
            patch_size=tuple(model_cfg.get('patch_size', [1, 16, 16])),
            in_chans=model_cfg.get('in_chans', 6),
            embed_dim=model_cfg.get('embed_dim', 1024),
            depth=model_cfg.get('depth', 24),
            num_heads=model_cfg.get('num_heads', 16),
            decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512),
            decoder_depth=model_cfg.get('decoder_depth', 8),
            decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
            mlp_ratio=model_cfg.get('mlp_ratio', 4),
            coords_encoding=model_cfg.get('coords_encoding', ['time', 'location']),
            coords_scale_learn=model_cfg.get('coords_scale_learn', True)
        )
        
        # Load weights
        weights_path = os.path.join(prithvi_model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            # Fallback to .pt if not renamed (though we renamed it)
            weights_path = os.path.join(prithvi_model_path, "Prithvi_EO_V2_300M_TL.pt")
            
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Load state dict (might need strict=False if config differs slightly or extra keys)
        msg = self.prithvi_model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded: {msg}")
        
        self.encoder = self.prithvi_model.encoder
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Determine hidden dimension
        encoder_dim = self.encoder.embed_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim + num_aux_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
        
    def forward(self, timeseries, auxiliary):
        """
        Args:
            timeseries: (batch, time, channels, H, W)
            auxiliary: (batch, num_aux_features)
        """
        # Prithvi-EO processes temporal sequences
        # Output shape is typically (batch, seq_len, hidden_dim)
        outputs = self.encoder(timeseries)
        
        # Unpack tuple (latent, mask, ids_restore)
        # We need the latent representation (first element)
        last_hidden_state = outputs[0]
        
        # spatial-temporal global average pooling
        # Prithvi-EO 2.0 uses patch embeddings + temporal embeddings
        # seq_len = T * (H/p) * (W/p)
        # We mean pool across the sequence dimension
        embeddings = last_hidden_state.mean(dim=1)
        
        # Concatenate with auxiliary features
        combined = torch.cat([embeddings, auxiliary], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    Focuses training on hard samples (misclassified).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
