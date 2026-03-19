import torch
import sys
import os

# Add flood_segmentation to path so we can import its model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../flood_segmentation')))
from model_training import FloodUNet

_fs_model = None
_fs_device = None

def init_fs_model():
    global _fs_model, _fs_device
    if _fs_model is not None:
        return _fs_model
    
    _fs_device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading Flood Segmentation Model to {_fs_device}...")
    
    _fs_model = FloodUNet(in_channels=20, num_classes=2)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    
    # Check for best model first, then pretrain
    best_finetune = os.path.join(base_dir, 'DisasterDL/flood_segmentation/models/finetune_best.pth')
    best_pretrain = os.path.join(base_dir, 'DisasterDL/flood_segmentation/models/pretrain_best.pth')
    
    abs_model_path = best_finetune if os.path.exists(best_finetune) else best_pretrain
    
    try:
        if os.path.exists(abs_model_path):
            checkpoint = torch.load(abs_model_path, map_location=_fs_device, weights_only=False)
            _fs_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"Flood Model loaded successfully from {abs_model_path}.")
        else:
            print(f"WARNING: Flood Model paths not found. Using untrained weights.")
    except Exception as e:
        print(f"Error loading flood model: {e}")
        
    _fs_model = _fs_model.to(_fs_device)
    _fs_model.eval()
    return _fs_model

def get_loaded_fs_model():
    global _fs_model
    if _fs_model is None:
        return init_fs_model()
    return _fs_model

def get_fs_device():
    global _fs_device
    if _fs_device is None:
        init_fs_model()
    return _fs_device
