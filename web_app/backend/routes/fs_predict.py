"""
fs_predict.py — Live Flood Segmentation Prediction
===================================================
Accepts 5 GeoTIFF uploads (pre_s1, pre_s2, post_s1, post_s2, aux),
builds a 20-channel input tensor (no random augmentations), runs the model,
and returns a flood mask overlay + statistics + geo-bounds for Leaflet.
"""
import base64
import io
import json
import os
import sys
import tempfile
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FS_DIR   = os.path.join(BASE_DIR, 'DisasterDL/flood_segmentation')
sys.path.insert(0, FS_DIR)

from inference.fs_model_loader import get_loaded_fs_model, get_fs_device

fs_predict_bp = Blueprint('fs_predict', __name__)

NORM_JSON = os.path.join(FS_DIR, 'data_analysis', 'normalization_stats.json')


def _read_tif_channels(tif_path, num_chans):
    """Read num_chans bands from a GeoTIFF, centre-crop to 512×512. Returns (C,H,W) float32."""
    import rasterio
    with rasterio.open(tif_path) as src:
        n    = min(src.count, num_chans)
        data = src.read(list(range(1, n + 1))).astype(np.float32)  # (n,H,W)
    if data.shape[0] < num_chans:
        pad  = np.zeros((num_chans - data.shape[0], *data.shape[1:]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)
    _, h, w = data.shape
    if h > 512 or w > 512:
        cy, cx = h // 2, w // 2
        data = data[:, cy-256:cy+256, cx-256:cx+256]
    return data


def _norm(data, key, norm):
    m, s = norm[key]['mean'], norm[key]['std']
    for c in range(data.shape[0]):
        data[c] = (data[c] - m[c]) / (s[c] + 1e-9)
    return data


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
      pre_s1   — pre-event Sentinel-1 GeoTIFF  (2 bands: VV, VH)
      pre_s2   — pre-event Sentinel-2 GeoTIFF  (6 bands: B2,B3,B4,B8,B11,B12)
      post_s1  — post-event Sentinel-1 GeoTIFF (2 bands)
      post_s2  — post-event Sentinel-2 GeoTIFF (6 bands)
      aux      — auxiliary GeoTIFF             (4 bands: SRTM, HAND, JRC occ, JRC season)

    At least one of post_s1 / post_s2 must be provided.
    """
    if not os.path.exists(NORM_JSON):
        return jsonify({'error': f'normalization_stats.json not found at {NORM_JSON}'}), 500

    with open(NORM_JSON) as f:
        norm = json.load(f)

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

        # ── Build 20-channel tensor ──────────────────────────────────────────
        img = np.zeros((20, 512, 512), dtype=np.float32)

        if 'post_s1' in saved:
            img[0:2]  = _norm(_read_tif_channels(saved['post_s1'], 2), 's1', norm)
        if 'post_s2' in saved:
            img[2:8]  = _norm(_read_tif_channels(saved['post_s2'], 6), 's2', norm)
        if 'pre_s1' in saved:
            img[8:10] = _norm(_read_tif_channels(saved['pre_s1'], 2), 'pre_s1', norm)
        if 'pre_s2' in saved:
            img[10:16]= _norm(_read_tif_channels(saved['pre_s2'], 6), 'pre_s2', norm)
        if 'aux' in saved:
            img[16:20]= _read_tif_channels(saved['aux'], 4) / 100.0

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Run model ─────────────────────────────────────────────────────────
        import torch
        model  = get_loaded_fs_model()
        device = get_fs_device()
        with torch.no_grad():
            t     = torch.from_numpy(img).float().unsqueeze(0).to(device)
            preds = model(t)
            pmask = torch.argmax(preds[0], dim=0).cpu().numpy()   # (H,W) 0=dry 1=flood

        # ── Overlays ──────────────────────────────────────────────────────────
        # Red: flood prediction
        pred_rgba = np.zeros((*pmask.shape, 4), dtype=np.uint8)
        pred_rgba[pmask == 1, 0] = 255
        pred_rgba[pmask == 1, 3] = 170
        pred_overlay = f"data:image/png;base64,{_encode_rgba_png(pred_rgba)}"

        # ── Stats ─────────────────────────────────────────────────────────────
        total       = pmask.size
        flood_px    = int(np.sum(pmask == 1))
        dry_px      = int(np.sum(pmask == 0))
        area_km2    = round(flood_px * 100 / 1_000_000, 4)   # 10m pixel → 100m²

        breakdown = {
            'Flooded':  {'count': flood_px, 'percentage': round(flood_px / total * 100, 2)},
            'Dry/Safe': {'count': dry_px,   'percentage': round(dry_px   / total * 100, 2)},
        }

        # ── Geo bounds (for Leaflet) ───────────────────────────────────────────
        bounds_path = saved.get('post_s2') or saved.get('post_s1') or saved.get('pre_s2') or saved.get('pre_s1')
        bounds = _get_bounds(bounds_path) if bounds_path else None

        # ── Preview images ────────────────────────────────────────────────────
        previews = {
            'pre_s1_image':  _sar_to_rgb_b64(saved['pre_s1'])          if 'pre_s1'  in saved else None,
            'pre_s2_image':  _tif_to_rgb_b64(saved['pre_s2'], [3,2,1]) if 'pre_s2'  in saved else None,
            'post_s1_image': _sar_to_rgb_b64(saved['post_s1'])         if 'post_s1' in saved else None,
            'post_s2_image': _tif_to_rgb_b64(saved['post_s2'], [3,2,1])if 'post_s2' in saved else None,
        }

        return jsonify({
            **previews,
            'pred_overlay':        pred_overlay,
            'bounds':              bounds,
            'breakdown':           breakdown,
            'estimated_area_km2':  area_km2,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
