import datetime
import json
import os

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'intelligence_cache.json')

_latest_stats = {
    "flood": {
        "estimated_area_km2": 0,
        "breakdown": {"Flooded": {"percentage": 0}, "Dry/Safe": {"percentage": 0}},
        "timestamp": None
    },
    "damage": {
        "total_buildings": 0,
        "breakdown": {
            "No Damage": {"count": 0, "percentage": 0},
            "Minor Damage": {"count": 0, "percentage": 0},
            "Major Damage": {"count": 0, "percentage": 0},
            "Destroyed": {"count": 0, "percentage": 0}
        },
        "accuracy": None,
        "timestamp": None
    }
}

def load_cache():
    global _latest_stats
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cached = json.load(f)
                # Basic merge to ensure schema compatibility
                if 'flood' in cached: _latest_stats['flood'].update(cached['flood'])
                if 'damage' in cached: _latest_stats['damage'].update(cached['damage'])
            print(f"[shared_state] Intelligence cache loaded from {CACHE_FILE}")
        except Exception as e:
            print(f"[shared_state] Error loading cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(_latest_stats, f, indent=4)
    except Exception as e:
        print(f"[shared_state] Error saving cache: {e}")

# Initial load
load_cache()

def update_flood_stats(area, breakdown):
    _latest_stats['flood']['estimated_area_km2'] = area
    _latest_stats['flood']['breakdown'] = breakdown
    _latest_stats['flood']['timestamp'] = datetime.datetime.now().isoformat()
    save_cache()

def update_damage_stats(total, breakdown, accuracy):
    _latest_stats['damage']['total_buildings'] = total
    _latest_stats['damage']['breakdown'] = breakdown
    _latest_stats['damage']['accuracy'] = accuracy
    _latest_stats['damage']['timestamp'] = datetime.datetime.now().isoformat()
    save_cache()
