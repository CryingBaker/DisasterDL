"""
bd_predict.py — Building Damage prediction from uploaded files.
Accepts pre/post images + their xBD JSON label files.
Runs per-building polygon-based inference (matching training).
"""

import io
import json
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
from flask import Blueprint, request, jsonify
from PIL import Image

from inference.bd_utils import (
    parse_wkt_polygon, extract_building_crop,
    COLOR_MAP, CLASS_NAMES_INV, DAMAGE_LABEL_MAP,
)
from inference.model_loader import get_loaded_model, get_device

CLASS_DISPLAY_NAMES = {
    0: 'No Damage',
    1: 'Minor Damage',
    2: 'Major Damage',
    3: 'Destroyed',
}

bd_predict_bp = Blueprint('bd_predict', __name__)


@bd_predict_bp.route('/predict', methods=['POST'])
def run_prediction():
    """
    Accepts pre/post images AND their JSON label files.
    Performs per-building polygon-based inference (matching training).
    """
    # ── Validate required files ───────────────────────────────
    required = ['pre_image', 'post_image', 'pre_json', 'post_json']
    missing = [f for f in required if f not in request.files]
    if missing:
        return jsonify({
            'error': f'Missing required files: {", ".join(missing)}. '
                     f'Please upload all four files (pre/post images + pre/post JSON labels).'
        }), 400

    # ── Load images ───────────────────────────────────────────
    try:
        pre_img = Image.open(request.files['pre_image']).convert('RGB')
        post_img = Image.open(request.files['post_image']).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    # ── Load JSON labels ──────────────────────────────────────
    try:
        pre_data = json.load(request.files['pre_json'])
        post_data = json.load(request.files['post_json'])
    except Exception as e:
        return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400

    if pre_img.size != post_img.size:
        post_img = post_img.resize(pre_img.size, Image.BILINEAR)

    img_size = (pre_img.width, pre_img.height)

    try:
        # ── Build UID → pre polygon mapping ───────────────────
        pre_polys = {}
        for feat in pre_data.get('features', {}).get('xy', []):
            uid = feat.get('properties', {}).get('uid')
            if uid:
                pre_polys[uid] = parse_wkt_polygon(feat['wkt'])

        # ── Build lng_lat mapping from post data ──────────────
        lng_lat_map = {}
        for feat in post_data.get('features', {}).get('lng_lat', []):
            uid = feat.get('properties', {}).get('uid')
            if uid:
                lng_lat_map[uid] = parse_wkt_polygon(feat.get('wkt', ''))

        # ── Load cached model ─────────────────────────────────
        model = get_loaded_model()
        device = get_device()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ── Per-building inference ────────────────────────────
        buildings = post_data.get('features', {}).get('xy', [])
        predicted_polygons = []
        gt_polygons = []
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        correct = 0
        total_with_gt = 0

        DAMAGE_SCORE_MAP = {
            'no-damage': 0.0, 'minor-damage': 1.0,
            'major-damage': 2.0, 'destroyed': 3.0,
            'un-classified': 1.5,
        }

        with torch.no_grad():
            for feat in buildings:
                uid = feat.get('properties', {}).get('uid')
                gt_subtype = feat.get('properties', {}).get('subtype', 'un-classified')
                post_coords = parse_wkt_polygon(feat.get('wkt', ''))
                pre_coords = pre_polys.get(uid, post_coords)
                lnglat_coords = lng_lat_map.get(uid, [])

                # Ground-truth entry
                gt_polygons.append({
                    'polygon_coords': post_coords,
                    'lnglat_coords': lnglat_coords,
                    'damage_class': gt_subtype,
                    'damage_score': DAMAGE_SCORE_MAP.get(gt_subtype, 1.5),
                    'color': COLOR_MAP.get(gt_subtype, '#aaaaaa'),
                    'uid': uid,
                })

                # Skip buildings with too few polygon vertices
                if len(post_coords) < 3 or len(pre_coords) < 3:
                    predicted_polygons.append({
                        'polygon_coords': post_coords,
                        'lnglat_coords': lnglat_coords,
                        'damage_class': 'un-classified',
                        'damage_score': 0.0,
                        'color': COLOR_MAP['un-classified'],
                        'uid': uid,
                        'ground_truth': gt_subtype,
                    })
                    continue

                # Extract building crops (polygon-masked, same as training)
                pre_crop = extract_building_crop(pre_img, pre_coords, 128, img_size)
                post_crop = extract_building_crop(post_img, post_coords, 128, img_size)

                pre_tensor = normalize(transforms.ToTensor()(pre_crop)).unsqueeze(0).to(device)
                post_tensor = normalize(transforms.ToTensor()(post_crop)).unsqueeze(0).to(device)

                output = model(pre_tensor, post_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred_class = probs.argmax().item()
                damage_score = sum(probs[i].item() * i for i in range(4))

                subtype = CLASS_NAMES_INV[pred_class]
                class_counts[pred_class] += 1

                # Track accuracy vs ground truth
                gt_label = DAMAGE_LABEL_MAP.get(gt_subtype, -1)
                if gt_label >= 0:
                    total_with_gt += 1
                    if gt_label == pred_class:
                        correct += 1

                predicted_polygons.append({
                    'polygon_coords': post_coords,
                    'lnglat_coords': lnglat_coords,
                    'damage_class': subtype,
                    'damage_score': damage_score,
                    'color': COLOR_MAP.get(subtype, '#aaaaaa'),
                    'uid': uid,
                    'ground_truth': gt_subtype,
                })

        # ── Encode images as base64 ──────────────────────────
        pre_buf = io.BytesIO()
        pre_img.save(pre_buf, format='PNG')
        pre_b64 = base64.b64encode(pre_buf.getvalue()).decode('utf-8')

        post_buf = io.BytesIO()
        post_img.save(post_buf, format='PNG')
        post_b64 = base64.b64encode(post_buf.getvalue()).decode('utf-8')

        # ── Build breakdown ──────────────────────────────────
        total_buildings = len(predicted_polygons)
        breakdown = {}
        for cls_id, count in class_counts.items():
            name = CLASS_DISPLAY_NAMES[cls_id]
            breakdown[name] = {
                'count': count,
                'percentage': round(
                    count / total_buildings * 100, 2
                ) if total_buildings > 0 else 0,
            }

        accuracy = round(correct / total_with_gt * 100, 1) if total_with_gt > 0 else None

        return jsonify({
            'pre_image': f"data:image/png;base64,{pre_b64}",
            'post_image': f"data:image/png;base64,{post_b64}",
            'ground_truth_polygons': gt_polygons,
            'predicted_polygons': predicted_polygons,
            'breakdown': breakdown,
            'total_buildings': total_buildings,
            'accuracy': accuracy,
            'image_size': list(img_size),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
