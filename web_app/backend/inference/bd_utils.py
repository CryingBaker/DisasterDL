"""
Utility functions for Building Damage prediction.
Includes: WKT polygon parsing, polygon masking, building crop extraction.

Copied from building_damage/utils.py to keep the web app self-contained.
"""

import re
import numpy as np
from PIL import Image, ImageDraw


# ───────────────────────── Constants ──────────────────────────────

DAMAGE_LABEL_MAP = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3,
}

DAMAGE_COLORS = {
    0: (0, 255, 0),       # Green  – No damage
    1: (255, 255, 0),     # Yellow – Minor damage
    2: (255, 165, 0),     # Orange – Major damage
    3: (255, 0, 0),       # Red    – Destroyed
}

CLASS_NAMES = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']

COLOR_MAP = {
    'no-damage': '#00C851',
    'minor-damage': '#ffbb33',
    'major-damage': '#ff8800',
    'destroyed': '#CC0000',
    'un-classified': '#aaaaaa',
}

CLASS_NAMES_INV = {
    0: 'no-damage',
    1: 'minor-damage',
    2: 'major-damage',
    3: 'destroyed',
}


# ───────────────────────── Polygon parsing ────────────────────────

def parse_wkt_polygon(wkt_str):
    """Parse a WKT POLYGON string and return list of (x, y) tuples."""
    match = re.search(r'POLYGON\s*\(\((.*?)\)\)', wkt_str)
    if not match:
        return []
    coords_str = match.group(1)
    coords = []
    for pair in coords_str.split(','):
        parts = pair.strip().split()
        if len(parts) == 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def polygon_to_mask(coords, img_size=(1024, 1024)):
    """Create a binary mask from polygon coordinates."""
    mask = Image.new('L', img_size, 0)
    if len(coords) >= 3:
        int_coords = [(int(round(x)), int(round(y))) for x, y in coords]
        ImageDraw.Draw(mask).polygon(int_coords, fill=255)
    return np.array(mask)


def get_building_bbox(coords, img_size=(1024, 1024), padding=5):
    """Get bounding box of a polygon with padding, clipped to image size."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min = max(0, int(np.floor(min(xs))) - padding)
    y_min = max(0, int(np.floor(min(ys))) - padding)
    x_max = min(img_size[0], int(np.ceil(max(xs))) + padding)
    y_max = min(img_size[1], int(np.ceil(max(ys))) + padding)
    return x_min, y_min, x_max, y_max


def extract_building_crop(image, coords, crop_size=128, img_size=(1024, 1024)):
    """
    Extract a building region crop from an image using polygon coordinates.
    Applies polygon mask to isolate only the building pixels.
    Returns a PIL Image of size crop_size x crop_size.
    """
    if len(coords) < 3:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    x_min, y_min, x_max, y_max = get_building_bbox(coords, img_size)

    # Ensure valid bbox
    if x_max <= x_min or y_max <= y_min:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    # Create mask for the full image then crop
    mask = polygon_to_mask(coords, img_size)

    # Convert image to numpy
    img_arr = np.array(image)

    # Apply mask
    masked = img_arr.copy()
    mask_3d = np.stack([mask] * 3, axis=-1)
    masked = np.where(mask_3d > 0, masked, 0)

    # Crop to bounding box
    crop = masked[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    # Resize to target size
    crop_img = Image.fromarray(crop.astype(np.uint8))
    crop_img = crop_img.resize((crop_size, crop_size), Image.BILINEAR)
    return crop_img
