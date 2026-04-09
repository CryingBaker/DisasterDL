"""
fs_model_loader.py — Flood Segmentation Inference (8 Ablation Models)
=====================================================================
Supports MaskedUNet (smp EfficientNet-B4 UNet + sentinel masking)
with variable channels (2-27) and simple scaling normalisation.

Models are discovered from disk, loaded lazily, and cached.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MaskedUNet (matches train.py) ─────────────────────────────────────────────

SENTINEL = -3.5

class _SentinelMask(nn.Module):
    def forward(self, x):
        bad = (x == SENTINEL).all(dim=(2, 3), keepdim=True)
        return torch.where(bad, torch.zeros_like(x), x)

class MaskedUNet(nn.Module):
    def __init__(self, in_channels, classes=2):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.masking = _SentinelMask()
        self.unet = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    def forward(self, x):
        return self.unet(self.masking(x))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
FS_DIR   = os.path.join(_BASE, 'DisasterDL', 'flood_segmentation')
MODELS_DIR = os.path.join(FS_DIR, 'models')
NORM_PATH  = os.path.join(FS_DIR, 'data_analysis', 'normalization_stats.json')

ABLATION_CHANNELS = {
    "OPTICAL_ONLY":     list(range(2, 15)),
    "SAR_ONLY":         [0, 1],
    "OPTICAL_SAR":      list(range(0, 15)),
    "OPTICAL_AUX":      list(range(2, 15)) + list(range(23, 27)),
    "OPTICAL_TEMPORAL": list(range(2, 23)),
    "SAR_TEMPORAL":     [0, 1, 15, 16],
    "NO_AUX":           list(range(0, 23)),
    "FULL_TEAM":        list(range(0, 27)),
}

# ── Test IoU results (from test_eval.py) ──────────────────────────────────────
TEST_METRICS = {
    'SAR_ONLY':         {'test_flood_iou': 0.6635},
    'SAR_TEMPORAL':     {'test_flood_iou': 0.6580},
    'OPTICAL_SAR':      {'test_flood_iou': 0.7680},
    'OPTICAL_AUX':      {'test_flood_iou': 0.7624},
    'NO_AUX':           {'test_flood_iou': 0.7533},
    'FULL_TEAM':        {'test_flood_iou': 0.7479},
    'OPTICAL_TEMPORAL': {'test_flood_iou': 0.7276},
    'OPTICAL_ONLY':     {'test_flood_iou': 0.6933},
}

def _parse_ablation_metrics():
    """Extract best val Flood IoU per ablation model from training log."""
    import re
    log_path = os.path.join(MODELS_DIR, 'disaster_dl_v15_mps.log')
    best = {}
    if not os.path.exists(log_path):
        return best
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(
                    r'\[(\w+)\] [HW]-Ep \d+ \| Loss: ([\d.]+) \| Flood IoU: ([\d.]+) '
                    r'\| Mean IoU: ([\d.]+) \| Pred/GT Flood Ratio: ([\d.]+)', line)
                if m:
                    name, loss, iou, miou, ratio = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
                    if name not in best or iou > best[name]['flood_iou']:
                        best[name] = dict(flood_iou=iou, mean_iou=miou, pred_gt_ratio=ratio, val_loss=loss)
    except Exception:
        pass
    return best


def _discover():
    reg = {}
    abl_metrics = _parse_ablation_metrics()

    for abl, chs in ABLATION_CHANNELS.items():
        p = os.path.join(MODELS_DIR, f"best_{abl}_model.pth")
        if os.path.exists(p):
            nice = abl.replace("_", " ").title()
            m = abl_metrics.get(abl, {})
            m.update(TEST_METRICS.get(abl, {}))
            reg[f"{nice} (Ablation)"] = dict(
                arch="MaskedUNet", in_channels=len(chs), path=p,
                preprocess="scaling", ckpt_key=None, channel_indices=chs,
                metrics=m)
    return reg

MODEL_REGISTRY = _discover()


def get_model_info():
    """Return a list of model info dicts suitable for the frontend."""
    result = []
    for name, cfg in MODEL_REGISTRY.items():
        m = cfg.get('metrics', {})
        result.append(dict(
            name=name,
            arch=cfg['arch'],
            in_channels=cfg['in_channels'],
            val_flood_iou=round(m.get('flood_iou', 0), 4),
            val_mean_iou=round(m.get('mean_iou', 0), 4),
            test_flood_iou=round(m.get('test_flood_iou', 0), 4),
            pred_gt_ratio=round(m.get('pred_gt_ratio', 0), 3) if 'pred_gt_ratio' in m else None,
        ))
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

_cache = {}
_device = None
_norm = None

def _dev():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available()
                               else 'mps' if torch.backends.mps.is_available()
                               else 'cpu')
    return _device

def _norm_stats():
    global _norm
    if _norm is None and os.path.exists(NORM_PATH):
        with open(NORM_PATH) as f:
            _norm = json.load(f)
    return _norm or {}

def get_available_models():
    return list(MODEL_REGISTRY.keys())

def get_default_model():
    names = list(MODEL_REGISTRY.keys())
    # Default to Full Team if available
    for n in names:
        if 'Full Team' in n:
            return n
    return names[0] if names else None

def get_model_config(name):
    return MODEL_REGISTRY.get(name)

def load_model(name):
    """Return (model, device), loading & caching on first call."""
    if name in _cache:
        return _cache[name]
    cfg = MODEL_REGISTRY.get(name)
    if cfg is None:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    dev = _dev()
    print(f"[fs_loader] Loading '{name}' ({cfg['arch']}, {cfg['in_channels']}ch) → {dev}")
    m = MaskedUNet(in_channels=cfg["in_channels"])
    try:
        raw = torch.load(cfg["path"], map_location=dev, weights_only=False)
        sd = raw if not cfg.get("ckpt_key") else raw.get(cfg["ckpt_key"], raw)
        m.load_state_dict(sd)
        print(f"[fs_loader]   ✓ weights loaded from {os.path.basename(cfg['path'])}")
    except Exception as e:
        print(f"[fs_loader]   ✗ weight load error: {e}")
    m.to(dev).eval()
    _cache[name] = (m, dev)
    return m, dev

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING — builds the right tensor for the chosen model
# ═══════════════════════════════════════════════════════════════════════════════

def _read_bands(path, nbands, fill=0.0):
    if not path or not os.path.exists(path):
        return np.full((nbands, 512, 512), fill, dtype=np.float32)
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
        c, h, w = data.shape
        out = np.full((c, 512, 512), fill, dtype=np.float32)
        hm, wm = min(h, 512), min(w, 512)
        if h > 512 or w > 512:
            cy, cx = h // 2, w // 2
            hs = max(cy - 256, 0); ws = max(cx - 256, 0)
            he = min(hs + 512, h); we = min(ws + 512, w)
            data = data[:, hs:he, ws:we]
            _, hm, wm = data.shape
        out[:c, :hm, :wm] = data[:c, :hm, :wm]
        if c >= nbands:
            return out[:nbands]
        result = np.full((nbands, 512, 512), fill, dtype=np.float32)
        result[:c] = out[:c]
        return result
    except Exception as e:
        print(f"[fs_loader] _read_bands error ({path}): {e}")
        return np.full((nbands, 512, 512), fill, dtype=np.float32)

# ── Scaling functions for ablation models ─────────────────────────────────────

def _scale_sar(arr):
    o = arr.copy(); v = o != SENTINEL
    o[v] = np.clip((o[v] + 25.0) / 25.0, 0, 1); return o

def _scale_optical(arr):
    o = arr.copy(); v = o != SENTINEL
    if v.any() and np.nanmax(o[v]) > 10:
        o[v] = o[v] / 10000.0
    o[v] = np.clip(o[v], 0, 1); return o

def _scale_aux(arr):
    o = arr.copy(); v = o != SENTINEL
    o[v] = np.clip((o[v] - 30.0) / 100.0, 0, 1); return o

# ── Tensor builders ───────────────────────────────────────────────────────────

def _build_27ch(ps1, ps2, pres1, pres2, aux):
    """27-channel tensor for MaskedUNet with simple scaling."""
    img = np.full((27, 512, 512), SENTINEL, dtype=np.float32)
    if ps1 and os.path.exists(ps1):
        img[0:2]   = _scale_sar(_read_bands(ps1, 2, SENTINEL))
    if ps2 and os.path.exists(ps2):
        img[2:15]  = _scale_optical(_read_bands(ps2, 13, SENTINEL))
    if pres1 and os.path.exists(pres1):
        img[15:17] = _scale_sar(_read_bands(pres1, 2, SENTINEL))
    if pres2 and os.path.exists(pres2):
        img[17:23] = _scale_optical(_read_bands(pres2, 6, SENTINEL))
    if aux and os.path.exists(aux):
        img[23:27] = _scale_aux(_read_bands(aux, 4, SENTINEL))
    return img

def build_tensor(model_name, ps1, ps2, pres1, pres2, aux):
    """Build the input tensor for a given model name. Returns (C, H, W) numpy."""
    cfg = MODEL_REGISTRY[model_name]
    full = _build_27ch(ps1, ps2, pres1, pres2, aux)
    return full[cfg["channel_indices"]]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BACKWARD-COMPAT helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_loaded_fs_model():
    name = get_default_model()
    if name is None:
        raise RuntimeError("No flood models found")
    m, _ = load_model(name)
    return m

def get_fs_device():
    return _dev()
