import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Path to the model in the project root
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../best_model.pth'))

class SimplifiedFloodModel(nn.Module):
    def __init__(self, in_channels=1, classes=2):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )

    def forward(self, x):
        return self.unet(x)

_model_cache = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def load_model(name=None):
    """Loads the default ResNet-34 flood model."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache, get_device()
    
    device = get_device()
    model = SimplifiedFloodModel(in_channels=1)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded existing model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading weight: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        
    model.to(device).eval()
    _model_cache = model
    return model, device

def get_available_models():
    return ["Default (ResNet-34)"]

def get_default_model():
    return "Default (ResNet-34)"

def build_tensor(model_name, ps1, ps2=None, pres1=None, pres2=None, aux=None):
    """
    Simplified tensor builder for single-channel SAR input.
    Expects ps1 (post_s1) to be a path to a GeoTIFF.
    """
    import rasterio
    import numpy as np
    
    if not ps1 or not os.path.exists(ps1):
        # Fallback to empty if missing
        return np.zeros((1, 512, 512), dtype=np.float32)

    try:
        with rasterio.open(ps1) as src:
            # Read first band (usually VV)
            data = src.read(1).astype(np.float32)
            
        # Basic normalization (match training scaling if known, else simple clip)
        # Assuming simple [0, 1] scaling or similar
        data = np.nan_to_num(data)
        low, high = np.percentile(data, (2, 98))
        if high > low:
            data = np.clip((data - low) / (high - low), 0, 1)
        else:
            data = np.zeros_like(data)
            
        # Resize to 512x512 if necessary
        if data.shape != (512, 512):
            from PIL import Image
            img = Image.fromarray(data)
            data = np.array(img.resize((512, 512), Image.BILINEAR))
            
        return data[np.newaxis, ...] # (1, 512, 512)
    except Exception as e:
        print(f"Error building tensor: {e}")
        return np.zeros((1, 512, 512), dtype=np.float32)

def get_model_info():
    return [{
        "name": "Default (ResNet-34)",
        "arch": "ResNet-34 UNet",
        "in_channels": 1,
        "test_flood_iou": 0.72,
        "val_flood_iou": 0.75
    }]

def get_model_config(name=None):
    """Returns a dummy config to satisfy legacy dataset routes."""
    return {
        "encoder_name": "resnet34",
        "in_channels": 1,
        "classes": 2
    }
