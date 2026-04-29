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
import zlib

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


def _encode_mask_b64(mask_np):
    """Compress a numpy uint8 array to base64 for roundtripping to the frontend."""
    raw = mask_np.tobytes()
    compressed = zlib.compress(raw, level=1)
    return base64.b64encode(compressed).decode('utf-8')


def _decode_mask_b64(b64_str, shape):
    """Decode a base64+zlib numpy uint8 array."""
    compressed = base64.b64decode(b64_str)
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _read_elevation(aux_path, target_shape=(512, 512)):
    """Read SRTM elevation (band 1) from the auxiliary GeoTIFF. Returns raw metres."""
    try:
        import rasterio
        with rasterio.open(aux_path) as src:
            elev = src.read(1).astype(np.float32)
        elev = np.nan_to_num(elev, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = elev.shape
        out = np.zeros(target_shape, dtype=np.float32)
        if h > target_shape[0] or w > target_shape[1]:
            cy, cx = h // 2, w // 2
            hs = max(cy - target_shape[0] // 2, 0)
            ws = max(cx - target_shape[1] // 2, 0)
            elev = elev[hs:hs + target_shape[0], ws:ws + target_shape[1]]
            h, w = elev.shape
        out[:min(h, target_shape[0]), :min(w, target_shape[1])] = elev[:min(h, target_shape[0]), :min(w, target_shape[1])]
        return out
    except Exception as e:
        print(f'[fs_predict] _read_elevation error: {e}')
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

        # ── Encode mask for simulation roundtrip ─────────────────────────
        mask_b64   = _encode_mask_b64(pmask.astype(np.uint8))
        mask_shape = list(pmask.shape)   # [H, W]

        # ── Elevation stats (if aux was provided) ─────────────────────────
        has_elevation = 'aux' in saved
        elev_stats = None
        if has_elevation:
            elev = _read_elevation(saved['aux'], tuple(pmask.shape))
            if elev is not None:
                flooded_elev = elev[pmask == 1]
                if flooded_elev.size > 0:
                    elev_stats = {
                        'min': float(np.min(flooded_elev)),
                        'max': float(np.max(flooded_elev)),
                        'median': float(np.median(flooded_elev)),
                        'mean': float(np.mean(flooded_elev)),
                    }

        return jsonify({
            **previews,
            'pred_overlay':        pred_overlay,
            'bounds':              bounds,
            'breakdown':           breakdown,
            'estimated_area_km2':  area_km2,
            'model_used':          model_name,
            'mask_b64':            mask_b64,
            'mask_shape':          mask_shape,
            'has_elevation':       has_elevation,
            'elevation_stats':     elev_stats,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Water Level Rise Simulation
# ══════════════════════════════════════════════════════════════════════════════

@fs_predict_bp.route('/simulate', methods=['POST'])
def simulate_water_rise():
    """
    Simulate additional flooding if water level rises by X metres.

    Expects multipart/form-data:
      - aux           : the auxiliary GeoTIFF (SRTM + MERIT + JRC)
      - mask_b64      : base64+zlib encoded prediction mask (uint8, 0/1)
      - mask_shape    : JSON string like "[512,512]"
      - water_rise    : float, metres of water level rise (0–10)
    """
    import json as _json

    try:
        # ── Parse inputs ──────────────────────────────────────────────────
        mask_b64   = request.form.get('mask_b64')
        shape_str  = request.form.get('mask_shape')
        water_rise = float(request.form.get('water_rise', 0))
        aux_file   = request.files.get('aux')

        if not mask_b64 or not shape_str:
            return jsonify({'error': 'Missing prediction mask data.'}), 400
        if not aux_file:
            return jsonify({'error': 'Auxiliary GeoTIFF (with elevation) is required.'}), 400
        if water_rise < 0 or water_rise > 15:
            return jsonify({'error': 'Water rise must be between 0 and 15 metres.'}), 400

        mask_shape = tuple(_json.loads(shape_str))  # (H, W)
        pred_mask  = _decode_mask_b64(mask_b64, mask_shape)  # uint8 0/1

        # ── Save & read elevation ─────────────────────────────────────────
        tmpdir = tempfile.mkdtemp()
        aux_path = os.path.join(tmpdir, 'aux.tif')
        aux_file.save(aux_path)

        elev = _read_elevation(aux_path, mask_shape)
        if elev is None:
            return jsonify({'error': 'Failed to read elevation from auxiliary file.'}), 400

        # ── Compute reference water surface elevation ─────────────────────
        # Use the 95th-percentile elevation of pixels the model already
        # predicted as flooded.  This represents the current water surface
        # level (robust against outlier elevation pixels inside the mask).
        flooded_elev = elev[pred_mask == 1]
        if flooded_elev.size == 0:
            ref_elevation = float(np.min(elev[elev > -999]))
        else:
            ref_elevation = float(np.percentile(flooded_elev, 95))

        threshold = ref_elevation + water_rise

        # ── Simulate expanded flooding ────────────────────────────────────
        # Only expand into the elevation band ABOVE the current water
        # surface and AT-OR-BELOW the new threshold.  This guarantees
        # that rise=0 produces exactly 0 additional flooding.
        new_flood = (
            (elev > ref_elevation) &   # above current water level
            (elev <= threshold)    &   # below new water level
            (pred_mask == 0)       &   # not already flooded
            (elev > -999)              # valid elevation data
        )
        combined = np.maximum(pred_mask, new_flood.astype(np.uint8))

        # ── Build overlay ─────────────────────────────────────────────────
        # Original flood → red, newly simulated → orange/amber
        sim_rgba = np.zeros((*mask_shape, 4), dtype=np.uint8)
        # Original flood: red
        sim_rgba[pred_mask == 1, 0] = 255
        sim_rgba[pred_mask == 1, 1] = 0
        sim_rgba[pred_mask == 1, 2] = 0
        sim_rgba[pred_mask == 1, 3] = 170
        # New flood: orange
        sim_rgba[new_flood, 0] = 255
        sim_rgba[new_flood, 1] = 165
        sim_rgba[new_flood, 2] = 0
        sim_rgba[new_flood, 3] = 170

        sim_overlay = f"data:image/png;base64,{_encode_rgba_png(sim_rgba)}"

        # ── Stats ─────────────────────────────────────────────────────────
        total         = combined.size
        orig_flood_px = int(np.sum(pred_mask == 1))
        new_flood_px  = int(np.sum(new_flood))
        total_flood   = int(np.sum(combined == 1))
        dry_px        = int(total - total_flood)

        area_orig     = round(orig_flood_px * 100 / 1_000_000, 4)
        area_new      = round(new_flood_px * 100 / 1_000_000, 4)
        area_total    = round(total_flood * 100 / 1_000_000, 4)

        breakdown = {
            'Original Flood': {
                'pixels': orig_flood_px,
                'percentage': round(orig_flood_px / total * 100, 2),
                'area_km2': area_orig,
            },
            'New Flooding': {
                'pixels': new_flood_px,
                'percentage': round(new_flood_px / total * 100, 2),
                'area_km2': area_new,
            },
            'Total Flooded': {
                'pixels': total_flood,
                'percentage': round(total_flood / total * 100, 2),
                'area_km2': area_total,
            },
            'Dry/Safe': {
                'pixels': dry_px,
                'percentage': round(dry_px / total * 100, 2),
            },
        }

        return jsonify({
            'sim_overlay':        sim_overlay,
            'water_rise':         water_rise,
            'ref_elevation':      round(ref_elevation, 2),
            'threshold':          round(threshold, 2),
            'breakdown':          breakdown,
            'total_area_km2':     area_total,
            'additional_area_km2': area_new,
            'original_area_km2':  area_orig,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
