from flask import Blueprint, jsonify, request
from .shared_state import _latest_stats

intelligence_bp = Blueprint('intelligence', __name__)

@intelligence_bp.route('/summary', methods=['GET'])
def get_summary():
    """Returns the aggregated intelligence summary."""
    
    # Calculate Crisis Index (0-100)
    # Weights: 40% Flood Area, 60% Severe Damage (Major + Destroyed)
    flood_score = min(_latest_stats['flood']['estimated_area_km2'] * 10, 40) # Max 40 points if 4km2 flooded
    
    damage_breakdown = _latest_stats['damage']['breakdown']
    severe_damage_pct = (
        damage_breakdown.get('Major Damage', {}).get('percentage', 0) + 
        damage_breakdown.get('Destroyed', {}).get('percentage', 0)
    )
    damage_score = (severe_damage_pct / 100) * 60
    
    crisis_index = round(flood_score + damage_score, 1)
    
    # Generate Recommendations
    recommendations = []
    if crisis_index > 70:
        recommendations.append("CRITICAL: Immediate mobilization of heavy rescue equipment and aerial support.")
    elif crisis_index > 40:
        recommendations.append("HIGH ALERT: Deploy regional units for evacuation and medical assistance.")
    else:
        recommendations.append("NORMAL MONITORING: Continue satellite surveillance and state-level readiness.")
        
    if _latest_stats['damage']['breakdown']['Destroyed']['count'] > 5:
        recommendations.append("URGENT: Structural engineers required for assessment of heavily hit quadrants.")

    return jsonify({
        "latest": _latest_stats,
        "crisis_index": crisis_index,
        "level": "Extreme" if crisis_index > 80 else "Critical" if crisis_index > 60 else "High" if crisis_index > 40 else "Moderate" if crisis_index > 20 else "Low",
        "recommendations": recommendations,
        "system_status": "Ready",
        "active_models": ["U-Net segmentation", "ResNet-50 CNN"]
    })
