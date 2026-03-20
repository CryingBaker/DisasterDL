import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .model_loader import get_loaded_model, get_device
from .tiler import tile_image, stitch_predictions

DAMAGE_COLORS = {
    0: (0, 255, 0),       # Green  – No damage
    1: (255, 255, 0),     # Yellow – Minor damage
    2: (255, 165, 0),     # Orange – Major damage
    3: (255, 0, 0),       # Red    – Destroyed
}

# ImageNet normalization — must match training
_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TILE_SIZE = 128  # must match training crop size

def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Pad to TILE_SIZE x TILE_SIZE if needed, then normalize."""
    w, h = pil_img.size
    if w != TILE_SIZE or h != TILE_SIZE:
        padded = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (0, 0, 0))
        padded.paste(pil_img, (0, 0))
        pil_img = padded
    return _normalize(pil_img)


def predict_tiles(pre_img: Image.Image, post_img: Image.Image, tile_size: int = TILE_SIZE):
    """
    Run prediction on full images by tiling them, predicting on pairs, and stitching.
    """
    model = get_loaded_model()
    device = get_device()
    model.eval()  # ensure batchnorm/dropout are in inference mode

    pre_tiles, width, height = tile_image(pre_img, tile_size)
    post_tiles, _, _ = tile_image(post_img, tile_size)

    predictions = []

    with torch.no_grad():
        for pre_t, post_t in zip(pre_tiles, post_tiles):
            pre_tensor = _preprocess(pre_t['image']).unsqueeze(0).to(device)
            post_tensor = _preprocess(post_t['image']).unsqueeze(0).to(device)

            output = model(pre_tensor, post_tensor)
            pred_class = output.argmax(dim=1).item()

            predictions.append({
                'box': pre_t['box'],
                'class': pred_class
            })

    stitched_mask = stitch_predictions(predictions, width, height, tile_size)
    return stitched_mask, predictions


def mask_to_colored_image(mask: np.ndarray) -> Image.Image:
    """Convert a 2D array of class indices to an RGB image."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in DAMAGE_COLORS.items():
        colored[mask == cls] = color
    return Image.fromarray(colored)