import sys
import base64
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify
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


@fs_dataset_bp.route('/list', methods=['GET'])
def list_dataset():
    try:
        meta_path = os.path.join(FS_DIR, "data_analysis", "dataset_metadata.csv")
        if not os.path.exists(meta_path):
            return jsonify({'error': 'Metadata CSV not found'}), 404

        df = pd.read_csv(meta_path)
        data = []
        for _, row in df.iterrows():
            set_type = row.get('set_type', 'weak')
            data.append({
                'uid': row['tile_id'],
                'split': row.get('split', 'train'),
                'set_type': set_type,
                'label_quality': 'Hand Labelled' if set_type in ['hand', 'both'] else 'Weakly Labelled',
                'flood_ratio': float(row['label_flood_pct']) if pd.notna(row.get('label_flood_pct')) else 0.0,
            })

        stats = {
            'training_tiles': int(len(df[df['split'] == 'train'])),
            'val_tiles':      int(len(df[df['split'] == 'val'])),
            'test_tiles':     int(len(df[df['split'] == 'test'])),
        }
        return jsonify({'data': data, 'total': len(data), 'stats': stats})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@fs_dataset_bp.route('/image/<uid>', methods=['GET'])
def get_images(uid):
    try:
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
        # (show ground truth for ALL tiles, hand or weak)
        if lbl_hand_abs and os.path.exists(lbl_hand_abs):
            lbl_path = lbl_hand_abs
        elif lbl_weak_abs and os.path.exists(lbl_weak_abs):
            lbl_path = lbl_weak_abs
        else:
            lbl_path = None

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

        # Model prediction — build input tensor DIRECTLY from TIFFs (bypasses FloodDataset
        # which applies random H/V-flip + rot90 augmentations on every call when split='train')
        pred_overlay    = None
        overlap_overlay = None
        try:
            from inference.fs_model_loader import get_loaded_fs_model, get_fs_device
            import json, torch, rasterio

            norm_path = os.path.join(FS_DIR, 'data_analysis', 'normalization_stats.json')
            if not os.path.exists(norm_path):
                raise FileNotFoundError(f"norm_stats.json not found at {norm_path}")
            with open(norm_path) as f:
                norm = json.load(f)

            def read_chans(tif_path, num_chans):
                """Read num_chans bands from a GeoTIFF, centre-crop to 512×512."""
                if not tif_path or not os.path.exists(tif_path):
                    return np.zeros((num_chans, 512, 512), dtype=np.float32)
                with rasterio.open(tif_path) as src:
                    n = min(src.count, num_chans)
                    data = src.read(list(range(1, n + 1))).astype(np.float32)
                if data.shape[0] < num_chans:
                    pad = np.zeros((num_chans - data.shape[0], *data.shape[1:]), dtype=np.float32)
                    data = np.concatenate([data, pad], axis=0)
                _, h, w = data.shape
                if h > 512 or w > 512:
                    cy, cx = h // 2, w // 2
                    data = data[:, cy-256:cy+256, cx-256:cx+256]
                return data

            def norm_chans(data, key):
                m, s = norm[key]['mean'], norm[key]['std']
                for c in range(data.shape[0]):
                    data[c] = (data[c] - m[c]) / (s[c] + 1e-9)
                return data

            img = np.zeros((20, 512, 512), dtype=np.float32)
            img[0:2]  = norm_chans(read_chans(post_s1_path, 2), 's1')
            img[2:8]  = norm_chans(read_chans(post_s2_path, 6), 's2')
            img[8:10] = norm_chans(read_chans(pre_s1_abs,   2), 'pre_s1')
            img[10:16]= norm_chans(read_chans(pre_s2_abs,   6), 'pre_s2')
            aux_rel = r.get('aux_path')
            if aux_rel and pd.notna(aux_rel):
                aux_abs = os.path.join(FS_DIR, aux_rel)
                img[16:20] = read_chans(aux_abs, 4) / 100.0
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            model  = get_loaded_fs_model()
            device = get_fs_device()
            with torch.no_grad():
                preds = model(torch.from_numpy(img).float().unsqueeze(0).to(device))
                pmask = torch.argmax(preds[0], dim=0).cpu().numpy()  # (H,W) 0=dry 1=flood

            # Red overlay — predicted flood
            pred_rgba = np.zeros((*pmask.shape, 4), dtype=np.uint8)
            pred_rgba[pmask == 1, 0] = 255
            pred_rgba[pmask == 1, 3] = 170
            pred_overlay = f"data:image/png;base64,{encode_rgba_png(pred_rgba)}"

            # Yellow overlap — where GT and prediction both say flood
            if lbl_path and os.path.exists(lbl_path):
                with rasterio.open(lbl_path) as src:
                    gt_raw = src.read(1).astype(np.int16)
                    south_up_lbl = src.transform.e > 0
                if south_up_lbl:
                    gt_raw = np.flipud(gt_raw)
                overlap = (gt_raw == 1) & (pmask == 1)
                ov_rgba = np.zeros((*pmask.shape, 4), dtype=np.uint8)
                ov_rgba[overlap, 0] = 255
                ov_rgba[overlap, 1] = 220
                ov_rgba[overlap, 3] = 210
                overlap_overlay = f"data:image/png;base64,{encode_rgba_png(ov_rgba)}"

        except Exception as pe:
            import traceback; traceback.print_exc()
            print(f"Prediction (non-fatal): {pe}")

        return jsonify({
            'pre_s1_image':   pre_s1_img,
            'pre_s2_image':   pre_s2_img,
            'post_s1_image':  post_s1_img,
            'post_s2_image':  post_s2_img,
            'gt_overlay':     gt_overlay,       # Blue: ground truth flood
            'pred_overlay':   pred_overlay,     # Red: model prediction flood
            'overlap_overlay':overlap_overlay,  # Yellow: GT ∩ Prediction
            'bounds':         bounds,
            'set_type':       set_type,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
