import math
from PIL import Image
import numpy as np

def tile_image(image, tile_size=128):
    """
    Split a large image into tile_size x tile_size crops.
    Returns a list of dictionaries with crop and its bounding box.
    """
    width, height = image.size
    tiles = []
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            crop = image.crop(box)
            
            # Pad if crop is smaller than tile_size (at edges)
            if crop.size != (tile_size, tile_size):
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(crop, (0, 0))
                crop = padded
                
            tiles.append({
                'image': crop,
                'box': (x, y, x + tile_size, y + tile_size)
            })
    return tiles, width, height

def stitch_predictions(predictions, width, height, tile_size=128):
    """
    Stitch tile predictions (scalars 0,1,2,3 representing damage) back into a full-size mask.
    Each tile's predicted class is mapped to the whole tile area.
    """
    stitched = np.zeros((height, width), dtype=np.uint8)
    
    for pred in predictions:
        box = pred['box']
        cls = pred['class']
        
        # Determine actual bounds without padding
        x1, y1 = box[0], box[1]
        x2 = min(box[2], width)
        y2 = min(box[3], height)
        
        stitched[y1:y2, x1:x2] = cls
        
    return stitched
