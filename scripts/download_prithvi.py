"""
Prithvi-EO Model Downloader and Setup

Downloads the pre-trained Prithvi-EO-2.0 model from Hugging Face
and sets up the directory structure for fine-tuning.
"""
import os
from huggingface_hub import snapshot_download

def download_prithvi():
    """Download Prithvi-EO-2.0 from Hugging Face"""
    
    model_dir = '../models/prithvi-eo-2.0'
    
    print("Downloading Prithvi-EO-2.0 from Hugging Face...")
    print("This may take several minutes (model is ~600MB)...")
    
    try:
        snapshot_download(
            repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"[OK] Model downloaded to: {model_dir}")
        print(f"[OK] Checkpoint: {model_dir}/model.safetensors")
        
        return model_dir
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nAlternative: Visit https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0")
        print("and click 'Files and versions' to download manually.")
        return None

def verify_installation():
    """Verify PyTorch and dependencies"""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        from transformers import AutoModel
        print("[OK] Transformers library installed")
        
        return True
    except Exception as e:
        print(f"[ERROR] Dependency check failed: {e}")
        print("\nRun: pip install -r requirements.txt")
        return False

if __name__ == '__main__':
    print("=== Prithvi-EO Setup ===\n")
    
    if verify_installation():
        download_prithvi()
    else:
        print("\nPlease install dependencies first.")
