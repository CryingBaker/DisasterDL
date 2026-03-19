import torch
import sys
import os

# Add building_damage to path so we can import its model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../building_damage')))
from model import get_model

_model = None
_device = None

def init_model():
    global _model, _device
    if _model is not None:
        return _model
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Building Damage Model to {_device}...")
    print("WARNING: Model expects input size of 128x128 pixels (CROP_SIZE). Inputs will be resized/tiled.")
    
    _model = get_model(num_classes=4, dropout=0.0, pretrained=False)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    abs_model_path = os.path.join(base_dir, 'DisasterDL/building_damage/models/best_model.pth')
    
    try:
        if os.path.exists(abs_model_path):
            checkpoint = torch.load(abs_model_path, map_location=_device, weights_only=False)
            _model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"Model loaded successfully from {abs_model_path}.")
        else:
            print(f"WARNING: Model path {abs_model_path} not found. Using untrained weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    _model = _model.to(_device)
    _model.eval()
    return _model

def get_loaded_model():
    global _model
    if _model is None:
        return init_model()
    return _model

def get_device():
    global _device
    if _device is None:
        init_model()
    return _device
