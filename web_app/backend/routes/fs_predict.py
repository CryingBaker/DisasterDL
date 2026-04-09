"""
fs_predict.py — Live Flood Segmentation Prediction (multi-model)
================================================================
Accepts 5 GeoTIFF uploads (pre_s1, pre_s2, post_s1, post_s2, aux),
builds the input tensor appropriate for the selected model, runs inference,
and returns a flood mask overlay + statistics + geo-bounds for Leaflet.
"""
import base64
import io
import os
import tempfile
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image

fs_predict_bp = Blueprint('fs_predict', __name__)


def _get_bounds(tif_path):
    """Return [[south, west], [north, east]] from a GeoTIFF (for Leaflet ImageOverlay)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(tif_path) as src:
            west, south, east, north = transform_bounds(
                src.crs, 'EPSG:4326',
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top
            )
        return [[south, west], [north, east]]
    except Exception:
        return None


def _encode_rgba_png(rgba):
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _tif_to_rgb_b64(tif_path, band_order=None):
    """Convert a GeoTIFF to a stretched RGB PNG (base64)."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            n = src.count
            if band_order is None:
                band_order = [3, 2, 1] if n >= 4 else [0, 1, 2]
            band_order = [min(b, n - 1) for b in band_order]
            bands = np.stack([src.read(b + 1).astype(np.float32) for b in band_order], axis=-1)
            south_up = src.transform.e > 0
        if south_up:
            bands = np.flipud(bands)
        lo, hi = np.percentile(bands, (2, 98))
        if hi - lo < 1e-6:
            bands = np.zeros_like(bands, dtype=np.uint8)
        else:
            bands = np.clip((bands - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(bands, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def _sar_to_rgb_b64(tif_path):
    """Convert SAR (S1 VV/VH) to a stretched RGB PNG (base64)."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32) if src.count >= 2 else vv.copy()
            south_up = src.transform.e > 0

        def stretch(a):
            lo, hi = np.percentile(a, (2, 98))
            if hi - lo < 1e-9:
                return np.zeros_like(a, dtype=np.uint8)
            return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        ratio = np.where(vh != 0, vv / (vh + 1e-9), 0)
        rgb = np.stack([stretch(vv), stretch(vh), stretch(ratio)], axis=-1)
        if south_up:
            rgb = np.flipud(rgb)
        img = Image.fromarray(rgb, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


@fs_predict_bp.route('/run', methods=['POST'])
def run_prediction():
    """
    Expects a multipart/form-data POST with up to 5 files:
      pre_s1, pre_s2, post_s1, post_s2, aux
    And an optional field:
      model — name of the model to use (defaults to first available)
    """
    from inference.fs_model_loader import (
        load_model, build_tensor, get_default_model, get_available_models
    )

    # ── Determine which model to use ──────────────────────────────────────
    model_name = request.form.get('model', get_default_model())
    if model_name not in get_available_models():
        return jsonify({'error': f"Unknown model: {model_name}"}), 400

    # Save uploaded files to a temp dir
    tmpdir = tempfile.mkdtemp()
    saved  = {}
    try:
        for key in ['pre_s1', 'pre_s2', 'post_s1', 'post_s2', 'aux']:
            file = request.files.get(key)
            if file and file.filename:
                dst = os.path.join(tmpdir, f'{key}.tif')
                file.save(dst)
                saved[key] = dst

        if not saved:
            return jsonify({'error': 'No files uploaded. Please upload at least post_s1 or post_s2.'}), 400

        # ── Build tensor using the model loader ──────────────────────────
        tensor = build_tensor(
            model_name,
            ps1=saved.get('post_s1'),
            ps2=saved.get('post_s2'),
            pres1=saved.get('pre_s1'),
            pres2=saved.get('pre_s2'),
            aux=saved.get('aux'),
        )

        # ── Run model ─────────────────────────────────────────────────────
        import torch
        model, device = load_model(model_name)
        with torch.no_grad():
            t     = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
            preds = model(t)
            pmask = torch.argmax(preds[0], dim=0).cpu().numpy()   # (H,W) 0=dry 1=flood

        # ── Overlays ──────────────────────────────────────────────────────
        pred_rgba = np.zeros((*pmask.shape, 4), dtype=np.uint8)
        pred_rgba[pmask == 1, 0] = 255
        pred_rgba[pmask == 1, 3] = 170
        pred_overlay = f"data:image/png;base64,{_encode_rgba_png(pred_rgba)}"

        # ── Stats ─────────────────────────────────────────────────────────
        total       = pmask.size
        flood_px    = int(np.sum(pmask == 1))
        dry_px      = int(np.sum(pmask == 0))
        area_km2    = round(flood_px * 100 / 1_000_000, 4)   # 10m pixel → 100m²

        breakdown = {
            'Flooded':  {'count': flood_px, 'percentage': round(flood_px / total * 100, 2)},
            'Dry/Safe': {'count': dry_px,   'percentage': round(dry_px   / total * 100, 2)},
        }

        # ── Geo bounds (for Leaflet) ───────────────────────────────────────
        bounds_path = saved.get('post_s2') or saved.get('post_s1') or saved.get('pre_s2') or saved.get('pre_s1')
        bounds = _get_bounds(bounds_path) if bounds_path else None

        # ── Preview images ────────────────────────────────────────────────
        previews = {
            'pre_s1_image':  _sar_to_rgb_b64(saved['pre_s1'])          if 'pre_s1'  in saved else None,
            'pre_s2_image':  _tif_to_rgb_b64(saved['pre_s2'], [3,2,1]) if 'pre_s2'  in saved else None,
            'post_s1_image': _sar_to_rgb_b64(saved['post_s1'])         if 'post_s1' in saved else None,
            'post_s2_image': _tif_to_rgb_b64(saved['post_s2'], [3,2,1])if 'post_s2' in saved else None,
        }

        # ── Update Intelligence Aggregator ────────────────────────────────
        from .shared_state import update_flood_stats
        update_flood_stats(area_km2, breakdown)

        return jsonify({
            **previews,
            'pred_overlay':        pred_overlay,
            'bounds':              bounds,
            'breakdown':           breakdown,
            'estimated_area_km2':  area_km2,
            'model_used':          model_name,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
