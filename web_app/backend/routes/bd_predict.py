from flask import Blueprint, request, jsonify
from PIL import Image
import io
import base64
from inference.predict import predict_tiles, mask_to_colored_image
from utils.metrics import CLASS_NAMES
import numpy as np

bd_predict_bp = Blueprint('bd_predict', __name__)

@bd_predict_bp.route('/predict', methods=['POST'])
def run_prediction():
    if 'pre_image' not in request.files or 'post_image' not in request.files:
        return jsonify({'error': 'Missing pre_image or post_image file.'}), 400
        
    pre_file = request.files['pre_image']
    post_file = request.files['post_image']
    
    try:
        pre_img = Image.open(pre_file).convert('RGB')
        post_img = Image.open(post_file).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
    if pre_img.size != post_img.size:
        post_img = post_img.resize(pre_img.size, Image.BILINEAR)

    try:
        stitched_mask, predictions = predict_tiles(pre_img, post_img)
        colored_mask_img = mask_to_colored_image(stitched_mask)
        
        buffered = io.BytesIO()
        colored_mask_img.save(buffered, format="PNG")
        mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        unique, counts = np.unique(stitched_mask, return_counts=True)
        total_pixels = stitched_mask.size
        breakdown = {}
        for cls, count in zip(unique, counts):
            name = CLASS_NAMES[cls]
            breakdown[name] = {
                'count': int(count),
                'percentage': round(float(count) / total_pixels * 100, 2)
            }
            
        return jsonify({
            'mask': f"data:image/png;base64,{mask_b64}",
            'breakdown': breakdown,
            'estimated_area_km2': round(total_pixels * 0.000001, 2)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
