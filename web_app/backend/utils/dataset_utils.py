import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
DATA_DIR = os.path.join(BASE_DIR, 'DisasterDL/building_damage/data')
print(f"DEBUG: Dataset Utils initialized. DATA_DIR resolved to: {DATA_DIR}")

DAMAGE_MAP = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3,
    'un-classified': 0
}

def get_dataset_list():
    dataset = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"DEBUG: split {split} images directory missing at {images_dir}")
            continue
            
        found = 0

        for file in os.listdir(images_dir):
            if file.endswith('_post_disaster.png'):
                uid = file.replace('_post_disaster.png', '')
                pre_img = os.path.join(images_dir, f"{uid}_pre_disaster.png")
                post_img = os.path.join(images_dir, file)
                label_json = os.path.join(labels_dir, f"{uid}_post_disaster.json")
                
                if not os.path.exists(pre_img):
                    continue
                    
                label = 0
                source = "unknown"
                if os.path.exists(label_json):
                    try:
                        with open(label_json, 'r') as f:
                            data = json.load(f)
                            source = data.get('metadata', {}).get('disaster', 'unknown')
                            max_damage = 0
                            for feature in data.get('features', {}).get('xy', []):
                                subtype = feature.get('properties', {}).get('subtype', 'no-damage')
                                max_damage = max(max_damage, DAMAGE_MAP.get(subtype, 0))
                            label = max_damage
                    except:
                        pass
                        
                dataset.append({
                    'id': uid,
                    'split': split,
                    'pre_crop': pre_img,
                    'post_crop': post_img,
                    'label': label,
                    'source': source
                })
                found += 1
        print(f"DEBUG: Split {split} found {found} images in {images_dir}")
    print(f"DEBUG: get_dataset_list returning total {len(dataset)} items")
    return dataset

def get_image_details(uid):
    dataset = get_dataset_list()
    for item in dataset:
        if item['id'] == uid:
            return item
    return None
