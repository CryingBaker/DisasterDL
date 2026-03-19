# DisasterDL — Deep Learning for Disaster Impact Assessment

<p align="center">
  A dual-model deep learning system for real-time flood segmentation and building damage detection,<br/>
  with an interactive React web interface for near real-time disaster assessment.
</p>

---

## Overview

**DisasterDL** combines two satellite-imagery deep learning models:

| Component | Model | Dataset |
|-----------|-------|---------|
| Flood Segmentation | Custom Siamese U-Net (20-channel input) | [Sen1Floods11](https://github.com/cloudtostreet/sen1floods11) |
| Building Damage Detection | CNN (Xview2-style) | [xView2](https://xview2.org/) |

Both models are served through a unified Flask backend and visualised in a React/Vite frontend with interactive Leaflet maps.

---

## Repository Structure

```
DisasterDL/
├── flood_segmentation/         # Flood segmentation model & data pipeline
│   ├── data_downloader.py      # Download Sen1Floods11 from GCS + GEE
│   ├── data_loader.py          # FloodDataset (PyTorch, 20-channel input)
│   ├── data_analysis.py        # Metadata scan, split generation, audit
│   ├── model_training.py       # Two-phase training (pretrain → finetune)
│   ├── visualize_predictions.py
│   ├── data_analysis/          # Split files & normalization stats (versioned)
│   │   ├── splits_train.txt    ← country-level split (DO NOT delete)
│   │   ├── splits_val.txt
│   │   ├── splits_test.txt
│   │   ├── dataset_metadata.csv
│   │   └── normalization_stats.json
│   └── models/                 # Model weights (gitignored — see below)
│
├── building_damage/            # Building damage model & data pipeline
│   ├── model.py                # CNN model (training + inference)
│   └── ...
│
├── web_app/
│   ├── backend/                # Flask API (port 5000)
│   │   ├── app.py
│   │   ├── routes/
│   │   │   ├── fs_dataset.py   # Flood segmentation dataset API
│   │   │   ├── fs_predict.py   # Flood segmentation prediction API
│   │   │   ├── bd_dataset.py   # Building damage dataset API
│   │   │   └── bd_predict.py
│   │   └── inference/
│   │       ├── fs_model_loader.py
│   │       └── model_loader.py
│   └── frontend/               # React + Vite + Leaflet
│       └── src/
│           └── pages/
│               ├── FSDatasetView.jsx   # Flood seg: dataset browser + map
│               ├── FSPredictView.jsx   # Flood seg: live prediction
│               ├── DatasetView.jsx     # Building damage: dataset browser
│               └── PredictionView.jsx  # Building damage: live prediction
│
├── .gitignore
└── README.md
```

---

## Dataset & Splits

### Flood Segmentation — Sen1Floods11

The flood segmentation model uses the [Sen1Floods11](https://github.com/cloudtostreet/sen1floods11) dataset augmented with:
- Pre-event Sentinel-1 composites (from Google Earth Engine)
- Pre-event Sentinel-2 composites (cloud-masked, 6 bands)
- Auxiliary rasters (SRTM elevation, MERIT HAND, JRC water occurrence)

**Split strategy** — data is split **by country** (no geographic leakage):

| Split | Countries | Approx tiles |
|-------|-----------|-------------|
| Train | Bolivia, Colombia, India, Mekong, Nigeria, Pakistan, Paraguay, USA | 4033 |
| Val   | Ghana, Somalia, Spain | 565 |
| Test  | Sri-Lanka | 232 |

**Input tensor** (20 channels, 512×512):

| Channels | Modality | Description |
|----------|----------|-------------|
| 0–1 | S1 (post) | SAR VV, VH after flood |
| 2–7 | S2 (post) | Optical B2,B3,B4,B8,B11,B12 after flood |
| 8–9 | S1 (pre) | SAR VV, VH before flood |
| 10–15 | S2 (pre) | Optical B2,B3,B4,B8,B11,B12 before flood |
| 16–19 | Aux | SRTM elevation, MERIT HAND, JRC occurrence, JRC seasonality |

---

## Downloading Data

```bash
cd flood_segmentation/

# Step 1 — Sen1Floods11 hand-labelled data (from Google Cloud Storage)
python data_downloader.py --section A --base_dir .

# Step 2 — Parse chip metadata (no external deps beyond geopandas)
python data_downloader.py --section B --base_dir .

# Step 3 — Weakly-labelled data (~4 GB from GCS)
python data_downloader.py --section F --base_dir .

# Step 4 — Combine hand + weak metadata
python data_downloader.py --section G --base_dir .

# Steps 5-7 — Pre-event S1, S2, and auxiliary data from Google Earth Engine
#   Requires: earthengine authenticate && a GEE project ID
python data_downloader.py --section C  --base_dir . --gee_project YOUR_PROJECT_ID
python data_downloader.py --section C2 --base_dir . --gee_project YOUR_PROJECT_ID
python data_downloader.py --section D  --base_dir . --gee_project YOUR_PROJECT_ID

# Or run all sections end-to-end:
python data_downloader.py --section all --base_dir . --gee_project YOUR_PROJECT_ID
```

**Required packages:**
```bash
pip install earthengine-api geemap rasterio geopandas tqdm pandas numpy
```

---

## Training

### Flood Segmentation

```bash
cd flood_segmentation/

# Phase 1: Pre-train on all tiles (hand + weak labels)
python model_training.py --phase pretrain

# Phase 2: Fine-tune on hand-labelled tiles only
python model_training.py --phase finetune

# Resume from checkpoint
python model_training.py --phase finetune --resume

# Debug run (fast, 2 epochs, small batch)
python model_training.py --phase pretrain --debug
```

Model weights are saved to `flood_segmentation/models/`.

### Building Damage

```bash
cd building_damage/
python model.py
```

---

## Web Application

### Backend (Flask, port 5000)

```bash
cd web_app/backend/
pip install -r requirements.txt
python app.py
```

### Frontend (React + Vite, port 5173)

```bash
cd web_app/frontend/
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

---

## Model Weights

Model weights (`.pth`) are **not committed** to Git (see `.gitignore`).

| File | Description |
|------|-------------|
| `flood_segmentation/models/pretrain_best.pth` | Best pretrain checkpoint |
| `flood_segmentation/models/finetune_best.pth` | Best finetune checkpoint |
| `building_damage/models/best_model.pth` | Building damage model |

---

## Contributors

- Atharv Wadadekar

## License

MIT License
