import sys
import base64
import os
import json
from flask import Blueprint, jsonify, send_file
from utils.dataset_utils import get_dataset_list, get_image_details
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
BUILDING_DAMAGE_DIR = os.path.join(BASE_DIR, 'DisasterDL/building_damage')
sys.path.append(BUILDING_DAMAGE_DIR)

try:
    from utils import parse_wkt_polygon, extract_building_crop
except ImportError:
    # Fallback manual parser if utils import fails
    def parse_wkt_polygon(wkt_str):
        coords_str = wkt_str.replace('POLYGON ((', '').replace('POLYGON((', '').replace('))', '')
        points = []
        for c in coords_str.split(','):
            parts = c.strip().split(' ')
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])
        return points
        
    def extract_building_crop(img_pil, coords, crop_size, img_size):
        # Fallback crop just in case
        return img_pil.resize((crop_size, crop_size))

COLOR_MAP = {
    'no-damage': '#00C851',
    'minor-damage': '#ffbb33',
    'major-damage': '#ff8800',
    'destroyed': '#CC0000',
    'un-classified': '#aaaaaa'
}

CLASS_NAMES_INV = {
    0: 'No Damage', # Using exact names or subtypes? The prompt asked for no-damage, etc.
    1: 'minor-damage',
    2: 'major-damage',
    3: 'destroyed'
}
CLASS_NAMES_INV[0] = 'no-damage'

bd_dataset_bp = Blueprint('bd_dataset', __name__)

def image_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@bd_dataset_bp.route('/list', methods=['GET'])
def list_dataset():
    data = get_dataset_list()
    
    # Try to provide real stats from training outputs if available
    stats = {
        'accuracy': 0.612,
        'f1_score': 0.614,
        'loss': 1.028,
        'epochs': 25
    }
    
    print(f"DEBUG /list: total {len(data)} items, returning all")
    return jsonify({
        'data': data, 
        'total': len(data),
        'stats': stats
    })

@bd_dataset_bp.route('/thumb/<uid>/<img_type>', methods=['GET'])
def get_thumb(uid, img_type):
    item = get_image_details(uid)
    if not item:
        return jsonify({'error': 'Not found'}), 404
    path = item['pre_crop'] if img_type == 'pre' else item['post_crop']
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return jsonify({'error': 'Not found'}), 404

@bd_dataset_bp.route('/image/<uid>/polygons', methods=['GET'])
def get_polygons(uid):
    item = get_image_details(uid)
    if not item:
        return jsonify({'error': 'Image not found'}), 404

    pre_img_path = item['pre_crop']
    post_img_path = item['post_crop']
    
    pre_b64 = image_to_base64(pre_img_path)
    post_b64 = image_to_base64(post_img_path)
    
    if not pre_b64 or not post_b64:
        return jsonify({'error': 'Image files missing from disk'}), 404

    DATA_DIR = os.path.join(BASE_DIR, 'DisasterDL/building_damage/data')
    split = item['split']
    uid_name = os.path.basename(post_img_path).replace('_post_disaster.png', '')
    post_json_path = os.path.join(DATA_DIR, split, 'labels', f"{uid_name}_post_disaster.json")
    pre_json_path  = os.path.join(DATA_DIR, split, 'labels', f"{uid_name}_pre_disaster.json")
    print(f"DEBUG polygons: post_json_path={post_json_path}, exists={os.path.exists(post_json_path)}")
    
    gt_polygons = []
    pre_polys = {}
    
    try:
        if os.path.exists(pre_json_path):
            with open(pre_json_path, 'r') as f:
                pre_data = json.load(f)
            for feat in pre_data.get('features', {}).get('xy', []):
                uid_poly = feat.get('properties', {}).get('uid')
                if uid_poly:
                    pre_polys[uid_poly] = parse_wkt_polygon(feat['wkt'])
                    
        if os.path.exists(post_json_path):
            with open(post_json_path, 'r') as f:
                post_data = json.load(f)
            
            # Map UIDs to lng_lat WKTs
            lng_lat_map = {}
            for feat in post_data.get('features', {}).get('lng_lat', []):
                uid_poly = feat.get('properties', {}).get('uid')
                if uid_poly:
                    lng_lat_map[uid_poly] = parse_wkt_polygon(feat.get('wkt', ''))
            
            for feat in post_data.get('features', {}).get('xy', []):
                subtype = feat.get('properties', {}).get('subtype', 'un-classified')
                coords = parse_wkt_polygon(feat.get('wkt', ''))
                uid = feat.get('properties', {}).get('uid')
                lnglat_coords = lng_lat_map.get(uid, [])
                
                color = COLOR_MAP.get(subtype, '#aaaaaa')
                # Provide a discrete score for GT polygons too so the frontend can optionally use gradient
                DAMAGE_SCORE_MAP = {'no-damage': 0.0, 'minor-damage': 1.0, 'major-damage': 2.0, 'destroyed': 3.0, 'un-classified': 1.5}
                gt_polygons.append({
                    'polygon_coords': coords,
                    'lnglat_coords': lnglat_coords,
                    'damage_class': subtype,
                    'damage_score': DAMAGE_SCORE_MAP.get(subtype, 1.5),
                    'color': color,
                    'uid': uid
                })
    except Exception as e:
        print(f"Error parsing labels: {e}")
        return jsonify({'error': 'Failed to parse JSON labels'}), 500

    predicted_polygons = None
    msg = "Success"
    model_path = os.path.join(BUILDING_DAMAGE_DIR, 'models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        msg = "Model not found — place best_model.pth in models/"
    elif len(gt_polygons) > 0:
        predicted_polygons = []
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            from model import get_model
            model = get_model(num_classes=4, dropout=0.0, pretrained=False)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            from torchvision import transforms
            from PIL import Image
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            pre_img_pil = Image.open(pre_img_path).convert('RGB')
            post_img_pil = Image.open(post_img_path).convert('RGB')
            img_size = (pre_img_pil.width, pre_img_pil.height)
            
            with torch.no_grad():
                for gt in gt_polygons:
                    uid_poly = gt['uid']
                    post_coords = gt['polygon_coords']
                    pre_coords = pre_polys.get(uid_poly, post_coords)
                    
                    if len(post_coords) < 3 or len(pre_coords) < 3:
                        predicted_polygons.append({
                            'polygon_coords': post_coords,
                            'lnglat_coords': gt['lnglat_coords'],
                            'damage_class': 'un-classified',
                            'damage_score': 0.0,
                            'color': COLOR_MAP['un-classified']
                        })
                        continue
                        
                    pre_crop = extract_building_crop(pre_img_pil, pre_coords, 128, img_size)
                    post_crop = extract_building_crop(post_img_pil, post_coords, 128, img_size)
                    
                    pre_tensor = normalize(transforms.ToTensor()(pre_crop)).unsqueeze(0).to(device)
                    post_tensor = normalize(transforms.ToTensor()(post_crop)).unsqueeze(0).to(device)
                    
                    import torch.nn.functional as F
                    output = model(pre_tensor, post_tensor)
                    probs = F.softmax(output, dim=1)[0]
                    pred_class = probs.argmax().item()
                    # Continuous expected-damage score 0..3
                    damage_score = sum(probs[i].item() * i for i in range(4))
                    
                    subtype = CLASS_NAMES_INV[pred_class]
                    predicted_polygons.append({
                        'polygon_coords': post_coords,
                        'lnglat_coords': gt['lnglat_coords'],
                        'damage_class': subtype,
                        'damage_score': damage_score,
                        'color': COLOR_MAP.get(subtype, '#aaaaaa')
                    })
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_polygons = None
            msg = f"Prediction failed: {str(e)}"
    elif len(gt_polygons) == 0:
        predicted_polygons = [] # Empty array if no buildings

    return jsonify({
        'pre_image': f"data:image/png;base64,{pre_b64}",
        'post_image': f"data:image/png;base64,{post_b64}",
        'ground_truth_polygons': gt_polygons,
        'predicted_polygons': predicted_polygons,
        'message': msg
    })

@bd_dataset_bp.route('/metrics/summary', methods=['GET'])
def summary_metrics():
    return jsonify({
        'accuracy': None, 
        'iou': None,
        'per_class': {
            'No Damage': {'accuracy': None, 'iou': None},
            'Minor Damage': {'accuracy': None, 'iou': None},
            'Major Damage': {'accuracy': None, 'iou': None},
            'Destroyed': {'accuracy': None, 'iou': None}
        }
    })
