import sys
import base64
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FS_DIR = os.path.join(BASE_DIR, 'DisasterDL/flood_segmentation')

fs_dataset_bp = Blueprint('fs_dataset', __name__)


def encode_png(img_array):
    img = Image.fromarray(img_array.astype('uint8'))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _need_flip(tif_path):
    """Return True if the GeoTIFF is south-up (positive y-transform → stored bottom-first)."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            return src.transform.e > 0
    except:
        return False


def tif_to_rgb(tif_path, band_order=None):
    """Read a multi-band GeoTIFF → stretched uint8 RGB image, with orientation correction."""
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
        low, high = np.percentile(bands, (2, 98))
        if high - low < 1e-6:
            return np.zeros((*bands.shape[:2], 3), dtype=np.uint8)
        bands = np.clip(bands, low, high)
        return ((bands - low) / (high - low) * 255).astype(np.uint8)
    except Exception as e:
        print(f"tif_to_rgb error: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)


def tif_s1_to_rgb(tif_path):
    """Convert SAR (S1) VV/VH to RGB, with orientation correction."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32) if src.count >= 2 else vv.copy()
            south_up = src.transform.e > 0

        def stretch(arr):
            lo, hi = np.percentile(arr, (2, 98))
            arr = np.clip(arr, lo, hi)
            if hi - lo < 1e-9: return np.zeros_like(arr, dtype=np.uint8)
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

        vv8 = stretch(vv)
        vh8 = stretch(vh)
        ratio8 = stretch(np.where(vh != 0, vv / (vh + 1e-9), 0))
        rgb = np.stack([vv8, vh8, ratio8], axis=-1)
        if south_up:
            rgb = np.flipud(rgb)
        return rgb
    except Exception as e:
        print(f"s1_to_rgb error: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)


def label_to_rgba(label_path, flood_color):
    """
    Turn a label TIFF into a RGBA PNG (flood pixels coloured, everything else transparent).
    Corrects south-up GeoTIFFs with np.flipud so the overlay aligns with the map.
    """
    try:
        import rasterio
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int16)
            south_up = src.transform.e > 0
        if south_up:
            label = np.flipud(label)
        h, w = label.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask = label == 1
        rgba[mask, 0] = flood_color[0]
        rgba[mask, 1] = flood_color[1]
        rgba[mask, 2] = flood_color[2]
        rgba[mask, 3] = 180
        return rgba
    except Exception as e:
        print(f"label_to_rgba error: {e}")
        return np.zeros((512, 512, 4), dtype=np.uint8)


def encode_rgba_png(rgba_array):
    img = Image.fromarray(rgba_array.astype('uint8'), mode='RGBA')
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_geo_bounds(tif_path):
    """Extract [[south, west], [north, east]] from a GeoTIFF."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(tif_path) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    except Exception as e:
        print(f"geo_bounds error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@fs_dataset_bp.route('/models', methods=['GET'])
def list_models():
    """Return all available flood-segmentation models with metrics."""
    try:
        from inference.fs_model_loader import get_available_models, get_default_model, get_model_info
        models = get_available_models()
        default = get_default_model()
        info = get_model_info()
        return jsonify({'models': models, 'default': default, 'model_info': info})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@fs_dataset_bp.route('/training_curves', methods=['GET'])
def training_curves():
    """Return per-epoch training data for all models (for charts)."""
    import re, csv
    result = {}

    # ── Ablation models from log file ─────────────────────────────────
    log_path = os.path.join(FS_DIR, 'models', 'disaster_dl_v15_mps.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m = re.search(
                    r'\[(\w+)\] ([HW])-Ep (\d+) \| Loss: ([\d.]+) \| '
                    r'Flood IoU: ([\d.]+) \| Mean IoU: ([\d.]+)', line)
                if m:
                    name = m.group(1).replace('_', ' ').title() + ' (Ablation)'
                    phase = 'weak' if m.group(2) == 'W' else 'hand'
                    result.setdefault(name, []).append({
                        'phase': phase,
                        'epoch': int(m.group(3)),
                        'loss': round(float(m.group(4)), 4),
                        'flood_iou': round(float(m.group(5)), 4),
                        'mean_iou': round(float(m.group(6)), 4),
                    })

    return jsonify(result)


@fs_dataset_bp.route('/list', methods=['GET'])
def list_dataset():
    try:
        meta_path = os.path.join(FS_DIR, "data_analysis", "dataset_metadata.csv")
        if not os.path.exists(meta_path):
            return jsonify({'error': 'Metadata CSV not found'}), 404

        df = pd.read_csv(meta_path)
        df['region'] = df['tile_id'].str.split('_').str[0]
        data = []
        for _, row in df.iterrows():
            set_type = row.get('set_type', 'weak')
            data.append({
                'uid': row['tile_id'],
                'split': row.get('split', 'train'),
                'set_type': set_type,
                'region': row['region'],
                'label_quality': 'Hand Labelled' if set_type in ['hand', 'both'] else 'Weakly Labelled',
                'flood_ratio': float(row['label_flood_pct']) if pd.notna(row.get('label_flood_pct')) else 0.0,
            })

        # Region breakdown
        region_breakdown = {}
        for region in sorted(df['region'].unique()):
            rdf = df[df['region'] == region]
            region_breakdown[region] = {
                'total': int(len(rdf)),
                'train': int(len(rdf[rdf['split'] == 'train'])),
                'val': int(len(rdf[rdf['split'] == 'val'])),
                'test': int(len(rdf[rdf['split'] == 'test'])),
                'hand': int(len(rdf[rdf['set_type'].isin(['hand', 'both'])])),
                'weak': int(len(rdf[rdf['set_type'] == 'weak'])),
            }

        stats = {
            'training_tiles': int(len(df[df['split'] == 'train'])),
            'val_tiles':      int(len(df[df['split'] == 'val'])),
            'test_tiles':     int(len(df[df['split'] == 'test'])),
            'region_breakdown': region_breakdown,
        }
        return jsonify({'data': data, 'total': len(data), 'stats': stats})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@fs_dataset_bp.route('/image/<uid>', methods=['GET'])
def get_images(uid):
    try:
        # ── Which model to use? ──────────────────────────────────────────
        from inference.fs_model_loader import (
            load_model, build_tensor, get_default_model, get_model_config
        )
        model_name = request.args.get('model', get_default_model())

        meta_path = os.path.join(FS_DIR, "data_analysis", "dataset_metadata.csv")
        df = pd.read_csv(meta_path)
        row = df[df['tile_id'] == uid]
        if row.empty:
            return jsonify({'error': 'Tile not found'}), 404

        r = row.iloc[0]

        def p(col):
            val = r.get(col)
            if val and pd.notna(val):
                return os.path.join(FS_DIR, val)
            return None

        pre_s1_abs   = p('pre_s1_path')
        pre_s2_abs   = p('pre_s2_path')
        s1_hand_abs  = p('s1_hand_path')
        s2_hand_abs  = p('s2_hand_path')
        s1_weak_abs  = p('s1_weak_path')
        s2_weak_abs  = p('s2_weak_path')
        lbl_hand_abs = p('label_hand_path')
        lbl_weak_abs = p('s2_weak_label_path')

        set_type = r.get('set_type', 'weak')
        is_hand  = set_type in ['hand', 'both']

        # Pick best post-event paths
        post_s2_path = s2_hand_abs if (is_hand and s2_hand_abs and os.path.exists(s2_hand_abs)) else s2_weak_abs
        post_s1_path = s1_hand_abs if (is_hand and s1_hand_abs and os.path.exists(s1_hand_abs)) else s1_weak_abs

        # GT label: use hand label if available, ALWAYS fall back to weak label
        if lbl_hand_abs and os.path.exists(lbl_hand_abs):
            lbl_path = lbl_hand_abs
        elif lbl_weak_abs and os.path.exists(lbl_weak_abs):
            lbl_path = lbl_weak_abs
        else:
            lbl_path = None

        # Auxiliary
        aux_rel = r.get('aux_path')
        aux_abs = os.path.join(FS_DIR, aux_rel) if aux_rel and pd.notna(aux_rel) else None

        # ── Preview images (unchanged, independent of model) ─────────────
        def img_b64(arr):
            return f"data:image/png;base64,{encode_png(arr)}"

        pre_s1_img  = img_b64(tif_s1_to_rgb(pre_s1_abs))         if pre_s1_abs  and os.path.exists(pre_s1_abs)  else None
        pre_s2_img  = img_b64(tif_to_rgb(pre_s2_abs, [3, 2, 1])) if pre_s2_abs  and os.path.exists(pre_s2_abs)  else None
        post_s1_img = img_b64(tif_s1_to_rgb(post_s1_path))        if post_s1_path and os.path.exists(post_s1_path) else None
        post_s2_img = img_b64(tif_to_rgb(post_s2_path, [3, 2, 1])) if post_s2_path and os.path.exists(post_s2_path) else None

        # GT mask: BLUE flood pixels, transparent background
        gt_overlay = None
        if lbl_path and os.path.exists(lbl_path):
            gt_rgba = label_to_rgba(lbl_path, flood_color=(0, 100, 255))
            gt_overlay = f"data:image/png;base64,{encode_rgba_png(gt_rgba)}"

        # Geo bounds
        bounds_path = post_s2_path or pre_s2_abs or post_s1_path or pre_s1_abs
        bounds = get_geo_bounds(bounds_path) if bounds_path and os.path.exists(bounds_path) else None

        # ── Model prediction ─────────────────────────────────────────────
        pred_overlay = None
        pred_stats = None
        try:
            import torch, rasterio

            tensor = build_tensor(model_name, post_s1_path, post_s2_path,
                                  pre_s1_abs, pre_s2_abs, aux_abs)
            model, device = load_model(model_name)

            with torch.no_grad():
                inp = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
                preds = model(inp)
                pmask = torch.argmax(preds[0], dim=0).cpu().numpy()

            pred_rgba = np.zeros((*pmask.shape, 4), dtype=np.uint8)
            pred_rgba[pmask == 1, 0] = 255
            pred_rgba[pmask == 1, 3] = 170
            pred_overlay = f"data:image/png;base64,{encode_rgba_png(pred_rgba)}"

            # Per-tile prediction stats
            total_px = pmask.size
            flood_px = int(np.sum(pmask == 1))
            flood_pct = round(flood_px / total_px * 100, 2) if total_px > 0 else 0
            area_km2 = round(flood_px * 100 / 1_000_000, 4)

            pred_stats = {
                'flood_pixels': flood_px,
                'total_pixels': total_px,
                'flood_pct': flood_pct,
                'area_km2': area_km2,
            }

            # IoU vs GT if label exists
            if lbl_path and os.path.exists(lbl_path):
                with rasterio.open(lbl_path) as src:
                    gt = src.read(1).astype(np.int16)
                    if src.transform.e > 0:
                        gt = np.flipud(gt)
                # Crop/pad GT to match pred size
                gh, gw = gt.shape
                ph, pw = pmask.shape
                if gh != ph or gw != pw:
                    tmp = np.zeros((ph, pw), dtype=gt.dtype)
                    mh, mw = min(gh, ph), min(gw, pw)
                    tmp[:mh, :mw] = gt[:mh, :mw]
                    gt = tmp

                gt_mask = (gt == 1)
                pred_mask = (pmask == 1)
                noflood_gt = (gt == 0)
                noflood_pred = (pmask == 0)

                # Flood IoU
                inter = int(np.sum(gt_mask & pred_mask))
                union = int(np.sum(gt_mask | pred_mask))
                flood_iou = round(inter / union, 4) if union > 0 else 0.0

                # Non-flood IoU
                nf_inter = int(np.sum(noflood_gt & noflood_pred))
                nf_union = int(np.sum(noflood_gt | noflood_pred))
                noflood_iou = round(nf_inter / nf_union, 4) if nf_union > 0 else 0.0

                mean_iou = round((flood_iou + noflood_iou) / 2, 4)

                # Accuracy
                correct = int(np.sum(pmask == gt))
                accuracy = round(correct / total_px, 4) if total_px > 0 else 0.0

                # Precision & recall for flood class
                tp = inter
                fp = int(np.sum(pred_mask & ~gt_mask))
                fn = int(np.sum(gt_mask & ~pred_mask))
                precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
                recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
                f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

                gt_flood = int(np.sum(gt_mask))
                gt_pct = round(gt_flood / total_px * 100, 2) if total_px > 0 else 0

                pred_stats['flood_iou'] = flood_iou
                pred_stats['mean_iou'] = mean_iou
                pred_stats['accuracy'] = accuracy
                pred_stats['precision'] = precision
                pred_stats['recall'] = recall
                pred_stats['f1'] = f1
                pred_stats['gt_flood_pixels'] = gt_flood
                pred_stats['gt_flood_pct'] = gt_pct

        except Exception as pe:
            import traceback; traceback.print_exc()
            print(f"Prediction (non-fatal): {pe}")

        return jsonify({
            'pre_s1_image':   pre_s1_img,
            'pre_s2_image':   pre_s2_img,
            'post_s1_image':  post_s1_img,
            'post_s2_image':  post_s2_img,
            'gt_overlay':   gt_overlay,
            'pred_overlay': pred_overlay,
            'bounds':       bounds,
            'set_type':       set_type,
            'model_used':     model_name,
            'pred_stats':     pred_stats,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
