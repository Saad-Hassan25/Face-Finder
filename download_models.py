"""
Script to download required models for Face Finder
"""

import os
import requests
import sys
from pathlib import Path
from typing import Optional

# Model URLs
MODELS = {
    "scrfd": {
        "url": "https://github.com/cospectrum/scrfd/releases/download/v0.2.0/scrfd_10g_bnkps.onnx",
        "filename": "scrfd.onnx",
        "description": "SCRFD face detection model (10G variant with keypoints)"
    },
    "lvface-t": {
        "url": "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-T_Glint360K/vit_t_dp005_mask0.onnx",
        "filename": "lvface.onnx", 
        "description": "LVFace-T (Tiny) face recognition model"
    },
    "lvface-s": {
        "url": "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-S_Glint360K/vit_s_dp005_mask_0.onnx",
        "filename": "lvface-s.onnx",
        "description": "LVFace-S (Small) face recognition model"
    },
    "lvface-b": {
        "url": "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-B_Glint360K/vit_b_dp005_mask_005.onnx",
        "filename": "lvface-b.onnx",
        "description": "LVFace-B (Base) face recognition model - best accuracy"
    }
}


def download_file(url: str, destination: Path, description: str = "") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description of the file being downloaded
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nDownloading: {description or destination.name}")
    print(f"URL: {url}")
    print(f"Destination: {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    bar_length = 50
                    filled = int(bar_length * downloaded / total_size)
                    bar = '=' * filled + '-' * (bar_length - filled)
                    sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)')
                    sys.stdout.flush()
        
        print(f"\n✓ Downloaded successfully: {destination.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Download failed: {e}")
        return False


def setup_models(models_dir: Optional[str] = None, models_to_download: Optional[list] = None):
    """
    Download required models.
    
    Args:
        models_dir: Directory to save models (default: ./models)
        models_to_download: List of model keys to download (default: scrfd and lvface-t)
    """
    # Set up models directory
    if models_dir is None:
        models_dir = Path(__file__).parent / "models"
    else:
        models_dir = Path(models_dir)
    
    models_dir.mkdir(exist_ok=True)
    print(f"Models directory: {models_dir}")
    
    # Default models to download
    if models_to_download is None:
        models_to_download = ["scrfd", "lvface-t"]
    
    # Download each model
    success_count = 0
    for model_key in models_to_download:
        if model_key not in MODELS:
            print(f"\n✗ Unknown model: {model_key}")
            print(f"  Available models: {', '.join(MODELS.keys())}")
            continue
        
        model_info = MODELS[model_key]
        destination = models_dir / model_info["filename"]
        
        if destination.exists():
            print(f"\n✓ Model already exists: {destination.name}")
            success_count += 1
            continue
        
        if download_file(model_info["url"], destination, model_info["description"]):
            success_count += 1
    
    print(f"\n{'=' * 60}")
    print(f"Downloaded {success_count}/{len(models_to_download)} models successfully")
    
    if success_count == len(models_to_download):
        print("\n✓ All models ready! You can now start the Face Finder API.")
        print("\nTo start the API:")
        print("  python main.py")
        print("\nOr with uvicorn:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("\n⚠ Some models failed to download. Please try again or download manually.")


def list_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("=" * 60)
    for key, info in MODELS.items():
        print(f"\n  {key}:")
        print(f"    Description: {info['description']}")
        print(f"    Filename: {info['filename']}")
        print(f"    URL: {info['url']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Face Finder")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=list(MODELS.keys()),
        default=["scrfd", "lvface-t"],
        help="Models to download (default: scrfd lvface-t)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    else:
        models_to_download = list(MODELS.keys()) if args.all else args.models
        setup_models(args.dir, models_to_download)
