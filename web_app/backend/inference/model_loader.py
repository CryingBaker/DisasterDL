"""
Building Damage model loader — singleton pattern.
Loads the Siamese ResNet-34 model once and caches it.
"""

import os
import torch
from inference.bd_model import get_model

_model = None
_device = None

# Default model path — relative to this file
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../building_damage/models/best_model.pth')
)
MODEL_PATH = os.environ.get('BD_MODEL_PATH', _DEFAULT_MODEL_PATH)


def init_model():
    global _model, _device
    if _model is not None:
        return _model

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[bd_model_loader] Loading Building Damage Model to {_device}...")

    _model = get_model(num_classes=4, dropout=0.0, pretrained=False)

    try:
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
            _model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print(f"[bd_model_loader] ✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"[bd_model_loader] ✗ Model not found at {MODEL_PATH}. Using untrained weights!")
    except Exception as e:
        print(f"[bd_model_loader] ✗ Error loading model: {e}")

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
