import torch
from .model_loader import get_loaded_model, get_device
from .preprocess import preprocess_image
from .tiler import tile_image, stitch_predictions

DAMAGE_COLORS = {
    0: (0, 255, 0),       # Green  – No damage
    1: (255, 255, 0),     # Yellow – Minor damage
    2: (255, 165, 0),     # Orange – Major damage
    3: (255, 0, 0),       # Red    – Destroyed
}

def predict_tiles(pre_img, post_img, tile_size=128):
    """
    Run prediction on full images by tiling them, predicting on pairs, and stitching.
    """
    model = get_loaded_model()
    device = get_device()
    
    pre_tiles, width, height = tile_image(pre_img, tile_size)
    post_tiles, _, _ = tile_image(post_img, tile_size)
    
    predictions = []
    
    with torch.no_grad():
        for pre_t, post_t in zip(pre_tiles, post_tiles):
            pre_tensor = preprocess_image(pre_t['image']).unsqueeze(0).to(device)
            post_tensor = preprocess_image(post_t['image']).unsqueeze(0).to(device)
            
            output = model(pre_tensor, post_tensor)
            pred_class = output.argmax(dim=1).item()
            
            predictions.append({
                'box': pre_t['box'],
                'class': pred_class
            })
            
    stitched_mask = stitch_predictions(predictions, width, height, tile_size)
    return stitched_mask, predictions

def mask_to_colored_image(mask):
    """Convert a 2D array of class indices to an RGB image."""
    from PIL import Image
    import numpy as np
    
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in DAMAGE_COLORS.items():
        colored[mask == cls] = color
        
    return Image.fromarray(colored)
