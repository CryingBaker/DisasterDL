"""
DisasterDL — Flood Segmentation Data Downloader
================================================
Downloads all data needed to train the flood segmentation model.

Sections
--------
  A   Sen1Floods11 hand-labelled data (S1, S2, Labels, DEM) — Google Cloud Storage
  B   Parse chip metadata from GeoJSON + build chip_metadata.csv
  C   Pre-event Sentinel-1 composites — Google Earth Engine
  C2  Pre-event Sentinel-2 composites (cloud-masked) — Google Earth Engine
  D   Auxiliary rasters (SRTM elevation, MERIT HAND, JRC water) — GEE
  F   Weakly-labelled data (S1Weak, S2Weak, LabelWeak) — Google Cloud Storage
  G   Combine hand + weak chip metadata → chip_metadata_combined.csv

Usage
-----
Run each section independently:

    python data_downloader.py --section A --base_dir /path/to/data
    python data_downloader.py --section B --base_dir /path/to/data
    python data_downloader.py --section C --base_dir /path/to/data --gee_project YOUR_GEE_PROJECT_ID
    ...

Or run all sections end-to-end (long!):

    python data_downloader.py --section all --base_dir /path/to/data --gee_project YOUR_GEE_PROJECT_ID

Requirements
------------
    pip install earthengine-api geemap rasterio geopandas tqdm pandas numpy gsutil

Notes
-----
- All download functions are idempotent (skip if file already exists).
- GEE sections (C, C2, D) require an authenticated Earth Engine account.
  Run `earthengine authenticate` once before using those sections.
- Section F (weakly-labelled) downloads ~4 GB from Google Cloud Storage.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_BASE_DIR = "."
GCS_BASE = "gs://sen1floods11/v1.1"

# Official event → split mapping (by location prefix in filename)
EVENT_SPLIT_MAP = {
    "Bolivia":   "train",
    "Colombia":  "train",
    "Ghana":     "val",
    "India":     "train",
    "Cambodia":  "train",
    "Mekong":    "train",
    "Nigeria":   "train",
    "Pakistan":  "train",
    "Paraguay":  "train",
    "Somalia":   "val",
    "Spain":     "val",
    "Sri-Lanka": "test",
    "USA":       "train",
}

LOCATION_ALIASES = {
    "Mekong": "Cambodia",
    "USA": "USA",
}


# ─── Section A — Sen1Floods11 hand-labelled data ──────────────────────────────

def section_a(base_dir: str):
    """Download Sen1Floods11 hand-labelled data from Google Cloud Storage."""
    print("\n" + "=" * 60)
    print("Section A — Sen1Floods11 Hand-Labelled Data")
    print("=" * 60)

    sen1_dir = os.path.join(base_dir, "data", "sen1floods11")
    os.makedirs(sen1_dir, exist_ok=True)

    folders = [
        ("data/flood_events/HandLabeled/S1Hand",    os.path.join(sen1_dir, "S1Hand")),
        ("data/flood_events/HandLabeled/S2Hand",    os.path.join(sen1_dir, "S2Hand")),
        ("data/flood_events/HandLabeled/LabelHand", os.path.join(sen1_dir, "LabelHand")),
        ("data/flood_events/HandLabeled/DEM",       os.path.join(sen1_dir, "DEM")),
        ("data/flood_events/HandLabeled/Permanent", os.path.join(sen1_dir, "Permanent")),
    ]

    for remote_suffix, local_dst in folders:
        os.makedirs(local_dst, exist_ok=True)
        src = f"{GCS_BASE}/{remote_suffix}/"
        print(f"\nDownloading {remote_suffix} ...")
        subprocess.run(["gsutil", "-m", "cp", "-r", "-n", f"{src}*", f"{local_dst}/"], check=False)
        print(f"  → saved to {local_dst}")

    # Metadata GeoJSON
    geojson_src = f"{GCS_BASE}/Sen1Floods11_Metadata.geojson"
    geojson_dst = os.path.join(sen1_dir, "Sen1Floods11_Metadata.geojson")
    subprocess.run(["gsutil", "-m", "cp", "-n", geojson_src, geojson_dst], check=False)
    print(f"\nMetadata GeoJSON saved to {geojson_dst}")

    # Sanity check
    for name in ["S1Hand", "S2Hand", "LabelHand"]:
        folder = os.path.join(sen1_dir, name)
        count = len(glob.glob(os.path.join(folder, "*.tif")))
        status = "OK" if count >= 400 else "⚠ LOW FILE COUNT"
        print(f"  {name}: {count} .tif files  [{status}]")

    print("\nSection A complete.")


# ─── Section B — Parse chip metadata ──────────────────────────────────────────

def section_b(base_dir: str):
    """Parse chip metadata from GeoJSON and build chip_metadata.csv."""
    print("\n" + "=" * 60)
    print("Section B — Parse Chip Metadata")
    print("=" * 60)

    import geopandas as gpd
    import pandas as pd

    sen1_dir   = os.path.join(base_dir, "data", "sen1floods11")
    csv_path   = os.path.join(base_dir, "data", "chip_metadata.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"chip_metadata.csv already exists — skipping. ({len(df)} chips)")
        for s in ["train", "val", "test"]:
            print(f"  {s}: {(df['split'] == s).sum()}")
        print("Section B complete (skipped).")
        return

    # --- Step 1: list all chip names from S1Hand ---
    s1_hand_dir = os.path.join(sen1_dir, "S1Hand")
    tif_files = sorted(glob.glob(os.path.join(s1_hand_dir, "**", "*.tif"), recursive=True))
    if not tif_files:
        alt = os.path.join(sen1_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
        tif_files = sorted(glob.glob(os.path.join(alt, "**", "*.tif"), recursive=True))
    print(f"Found {len(tif_files)} S1Hand .tif files.")

    chip_names = [re.sub(r"_S1Hand\.tif$", "", os.path.basename(f), flags=re.IGNORECASE)
                  for f in tif_files]
    print(f"Example chips: {chip_names[:5]}")

    # --- Step 2: load GeoJSON ---
    geojson_path = os.path.join(sen1_dir, "Sen1Floods11_Metadata.geojson")
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} event rows from metadata GeoJSON.")

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            if c != "geometry" and df[c].dtype == "object":
                return c
        return None

    location_col = _find_col(gdf, ["location", "Location", "event", "Event", "name", "Name"])
    date_col     = _find_col(gdf, ["s1_date", "S1Date", "s1date", "date", "Date"])
    country_col  = _find_col(gdf, ["ISO_CC", "iso_cc", "country", "Country", "ISO"])
    print(f"Columns — location: {location_col}, date: {date_col}, country: {country_col}")

    event_lookup = {str(row[location_col]).strip().lower(): row for _, row in gdf.iterrows()}

    # --- Step 3: match each chip to its event ---
    rows, unmatched = [], []
    for chip in chip_names:
        matched_event = matched_loc = None
        for alias_prefix, alias_target in LOCATION_ALIASES.items():
            if chip.lower().startswith(alias_prefix.lower()):
                if alias_target.lower() in event_lookup:
                    matched_event = event_lookup[alias_target.lower()]
                    matched_loc = alias_prefix
                break
        if matched_event is None:
            for loc_lower, row in event_lookup.items():
                if chip.lower().startswith(loc_lower):
                    matched_event = row
                    matched_loc = str(row[location_col]).strip()
                    break
        if matched_event is None:
            unmatched.append(chip)
            continue

        s1_date = str(matched_event[date_col])[:10] if date_col and pd.notna(matched_event[date_col]) else None
        bounds  = matched_event.geometry.bounds
        country = str(matched_event[country_col]) if country_col and pd.notna(matched_event[country_col]) else ""

        rows.append({
            "chip_name": chip,
            "s1_date":   s1_date,
            "lon_min": bounds[0], "lat_min": bounds[1],
            "lon_max": bounds[2], "lat_max": bounds[3],
            "country": country,
            "_event": matched_loc,
        })

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} chips unmatched — {unmatched[:5]} ...")

    df = pd.DataFrame(rows)
    split_lookup = {k.lower(): v for k, v in EVENT_SPLIT_MAP.items()}
    df["split"] = df["_event"].apply(lambda e: split_lookup.get(str(e).lower(), "train"))

    print("\nEvent → split:")
    for event in sorted(df["_event"].unique()):
        n = (df["_event"] == event).sum()
        s = split_lookup.get(event.lower(), "train")
        print(f"  {event:<15s} → {s} ({n} chips)")

    df = df.drop(columns=["_event"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved chip_metadata.csv → {csv_path}  ({len(df)} chips)")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(df['split'] == s).sum()}")
    print("Section B complete.")


# ─── Section C — Pre-event Sentinel-1 from GEE ───────────────────────────────

def section_c(base_dir: str, gee_project: str):
    """Download pre-event Sentinel-1 composites for every chip from Google Earth Engine."""
    print("\n" + "=" * 60)
    print("Section C — Pre-event Sentinel-1 (GEE)")
    print("=" * 60)

    import ee
    import geemap
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from tqdm import tqdm

    ee.Authenticate()
    ee.Initialize(project=gee_project)

    csv_path = os.path.join(base_dir, "data", "chip_metadata_combined.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "chip_metadata.csv")
    df = pd.read_csv(csv_path)

    pre_s1_dir  = os.path.join(base_dir, "data", "pre_s1")
    missing_log = os.path.join(base_dir, "data", "missing_pre_s1.txt")
    os.makedirs(pre_s1_dir, exist_ok=True)

    downloaded = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-event S1"):
        chip     = row["chip_name"]
        out_path = os.path.join(pre_s1_dir, f"{chip}_pre_s1.tif")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            s1_dir_chip = row["s1_dir"]
            suffix = "S1Weak" if row.get("label_quality") == "weak" else "S1Hand"
            s1_path = os.path.join(s1_dir_chip, f"{chip}_{suffix}.tif")
            with rasterio.open(s1_path) as src:
                b = transform_bounds(src.crs, "EPSG:4326",
                                     src.bounds.left, src.bounds.bottom,
                                     src.bounds.right, src.bounds.top)
                lon_min, lat_min, lon_max, lat_max = b

            s1_date  = datetime.strptime(str(row["s1_date"])[:10], "%Y-%m-%d")
            pre_end  = s1_date - timedelta(days=3)
            pre_start = s1_date - timedelta(days=33)
            roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

            collection = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(roi)
                .filterDate(pre_start.strftime("%Y-%m-%d"), pre_end.strftime("%Y-%m-%d"))
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            )
            composite = collection.median().select(["VV", "VH"])
            geemap.ee_export_image(composite, filename=out_path, scale=10,
                                   region=roi, file_per_band=False)
            downloaded += 1
        except Exception as e:
            failed += 1
            with open(missing_log, "a") as f:
                f.write(f"{chip}\n")

    print(f"\nPre-event S1 complete.  downloaded={downloaded}  skipped={skipped}  failed={failed}")
    print("Section C complete.")


# ─── Section C2 — Pre-event Sentinel-2 from GEE ──────────────────────────────

def section_c2(base_dir: str, gee_project: str):
    """Download pre-event Sentinel-2 composites (cloud-masked) from GEE."""
    print("\n" + "=" * 60)
    print("Section C2 — Pre-event Sentinel-2 (GEE)")
    print("=" * 60)

    import ee
    import geemap
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds
    from tqdm import tqdm

    ee.Authenticate()
    ee.Initialize(project=gee_project)

    csv_path = os.path.join(base_dir, "data", "chip_metadata_combined.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "chip_metadata.csv")
    df = pd.read_csv(csv_path)

    pre_s2_dir  = os.path.join(base_dir, "data", "pre_s2")
    missing_log = os.path.join(base_dir, "data", "missing_pre_s2.txt")
    os.makedirs(pre_s2_dir, exist_ok=True)

    def mask_scl(image):
        scl = image.select("SCL")
        mask = (scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)

    downloaded = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-event S2"):
        chip     = row["chip_name"]
        out_path = os.path.join(pre_s2_dir, f"{chip}_pre_s2.tif")
        if os.path.exists(out_path):
            skipped += 1
            continue

        lon_min = lat_min = lon_max = lat_max = None
        try:
            s1_dir_chip = row["s1_dir"]
            suffix = "S1Weak" if row.get("label_quality") == "weak" else "S1Hand"
            s1_path = os.path.join(s1_dir_chip, f"{chip}_{suffix}.tif")
            with rasterio.open(s1_path) as src:
                b = transform_bounds(src.crs, "EPSG:4326",
                                     src.bounds.left, src.bounds.bottom,
                                     src.bounds.right, src.bounds.top)
                lon_min, lat_min, lon_max, lat_max = b

            flood_date = datetime.strptime(str(row["s1_date"])[:10], "%Y-%m-%d")
            pre_start  = flood_date - timedelta(days=180)
            pre_end    = flood_date - timedelta(days=3)
            roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(pre_start.strftime("%Y-%m-%d"), pre_end.strftime("%Y-%m-%d"))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            )
            # Fallback: least-cloudy single scene
            if collection.size().getInfo() == 0:
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(roi)
                    .filterDate(pre_start.strftime("%Y-%m-%d"), pre_end.strftime("%Y-%m-%d"))
                    .sort("CLOUDY_PIXEL_PERCENTAGE")
                )
            if collection.size().getInfo() == 0:
                raise ValueError("No Sentinel-2 images found")

            s2_image = (
                collection.map(mask_scl)
                .select(["B2", "B3", "B4", "B8", "B11", "B12"])
                .median()
                .multiply(0.0001)
            )
            geemap.ee_export_image(s2_image, filename=out_path, scale=10,
                                   region=roi, file_per_band=False)
            downloaded += 1
        except Exception as e:
            failed += 1
            with open(missing_log, "a") as f:
                f.write(f"{chip}\n")
            # Write zero-filled placeholder so the dataset index stays stable
            try:
                if lon_min is not None:
                    size = 512
                    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, size, size)
                    with rasterio.open(out_path, "w", driver="GTiff", dtype="float32",
                                       width=size, height=size, count=6,
                                       crs="EPSG:4326", transform=transform) as dst:
                        for b in range(6):
                            dst.write(np.zeros((size, size), dtype=np.float32), b + 1)
            except Exception:
                pass

    print(f"\nPre-event S2 complete.  downloaded={downloaded}  skipped={skipped}  failed={failed}")
    print("Section C2 complete.")


# ─── Section D — Auxiliary data from GEE ─────────────────────────────────────

def section_d(base_dir: str, gee_project: str):
    """Download 4-band auxiliary rasters (SRTM, MERIT HAND, JRC) from GEE.

    Band layout
    -----------
    1: SRTM elevation (m)
    2: MERIT Hydro HAND (Height Above Nearest Drainage)
    3: JRC occurrence (%)
    4: JRC seasonality
    """
    print("\n" + "=" * 60)
    print("Section D — Auxiliary Data (GEE)")
    print("=" * 60)

    import ee
    import geemap
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from tqdm import tqdm

    ee.Authenticate()
    ee.Initialize(project=gee_project)

    csv_path = os.path.join(base_dir, "data", "chip_metadata_combined.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "data", "chip_metadata.csv")
    df = pd.read_csv(csv_path)

    aux_dir = os.path.join(base_dir, "data", "aux")
    os.makedirs(aux_dir, exist_ok=True)

    downloaded = skipped = failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aux data"):
        chip     = row["chip_name"]
        out_path = os.path.join(aux_dir, f"{chip}_aux.tif")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            s1_dir_chip = row["s1_dir"]
            suffix = "S1Weak" if row.get("label_quality") == "weak" else "S1Hand"
            s1_path = os.path.join(s1_dir_chip, f"{chip}_{suffix}.tif")
            with rasterio.open(s1_path) as src:
                b = transform_bounds(src.crs, "EPSG:4326",
                                     src.bounds.left, src.bounds.bottom,
                                     src.bounds.right, src.bounds.top)
                lon_min, lat_min, lon_max, lat_max = b

            roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

            srtm        = ee.Image("USGS/SRTMGL1_003").select("elevation")
            hand        = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
            jrc         = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            occurrence  = jrc.select("occurrence")
            seasonality = jrc.select("seasonality")

            stacked = srtm.addBands(hand).addBands(occurrence).addBands(seasonality)
            geemap.ee_export_image(stacked, filename=out_path, scale=10,
                                   region=roi, file_per_band=False)
            downloaded += 1
        except Exception as e:
            failed += 1

    print(f"\nAuxiliary data complete.  downloaded={downloaded}  skipped={skipped}  failed={failed}")
    print("Section D complete.")


# ─── Section F — Weakly-labelled data from GCS ────────────────────────────────

def section_f(base_dir: str):
    """Download Sen1Floods11 weakly-labelled data (S1Weak, S2Weak, LabelWeak) from GCS."""
    print("\n" + "=" * 60)
    print("Section F — Weakly-Labelled Data (Google Cloud Storage)")
    print("=" * 60)

    sen1_dir = os.path.join(base_dir, "data", "sen1floods11")

    weak_folders = [
        ("data/flood_events/WeaklyLabeled/S1Weak",    os.path.join(sen1_dir, "S1Weak")),
        ("data/flood_events/WeaklyLabeled/S2Weak",    os.path.join(sen1_dir, "S2Weak")),
        ("data/flood_events/WeaklyLabeled/LabelWeak", os.path.join(sen1_dir, "LabelWeak")),
    ]

    for remote_suffix, local_dst in weak_folders:
        os.makedirs(local_dst, exist_ok=True)
        src = f"{GCS_BASE}/{remote_suffix}/"
        print(f"\nDownloading {remote_suffix} ...")
        subprocess.run(["gsutil", "-m", "cp", "-r", "-n", f"{src}*", f"{local_dst}/"], check=False)

    print("\nWeakly-labelled file counts:")
    for name in ["S1Weak", "S2Weak", "LabelWeak"]:
        folder = os.path.join(sen1_dir, name)
        count  = len(glob.glob(os.path.join(folder, "*.tif")))
        expected = "~4385" if name != "S2Weak" else "varies"
        print(f"  {name:<10s}: {count} .tif files  (expected {expected})")

    print("Section F complete.")


# ─── Section G — Combine hand + weak metadata ─────────────────────────────────

def section_g(base_dir: str):
    """Combine hand-labelled and weakly-labelled chip metadata into one CSV."""
    print("\n" + "=" * 60)
    print("Section G — Combine Hand + Weak Metadata")
    print("=" * 60)

    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from tqdm import tqdm

    csv_path          = os.path.join(base_dir, "data", "chip_metadata.csv")
    combined_csv_path = os.path.join(base_dir, "data", "chip_metadata_combined.csv")

    if os.path.exists(combined_csv_path):
        df = pd.read_csv(combined_csv_path)
        print(f"chip_metadata_combined.csv already exists ({len(df)} chips). Skipping.")
        return

    sen1_dir = os.path.join(base_dir, "data", "sen1floods11")

    # 1. Add dir columns to hand chips
    df_hand = pd.read_csv(csv_path)
    df_hand["label_quality"] = "hand"
    df_hand["label_dir"] = os.path.join(sen1_dir, "LabelHand")
    df_hand["s1_dir"]    = os.path.join(sen1_dir, "S1Hand")
    df_hand["s2_dir"]    = os.path.join(sen1_dir, "S2Hand")

    # 2. List weak chips
    s1weak_dir = os.path.join(sen1_dir, "S1Weak")
    if not os.path.exists(s1weak_dir):
        s1weak_dir = os.path.join(sen1_dir, "v1.1", "data", "flood_events", "WeaklyLabeled", "S1Weak")
    weak_files = sorted(glob.glob(os.path.join(s1weak_dir, "**", "*.tif"), recursive=True))
    print(f"Found {len(weak_files)} S1Weak files.")

    # 3. Load GeoJSON for location/date info
    geojson_path = os.path.join(sen1_dir, "Sen1Floods11_Metadata.geojson")
    gdf = gpd.read_file(geojson_path)

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns: return c
        return None

    location_col = _find_col(gdf, ["location", "Location", "event", "Event", "name", "Name"])
    date_col     = _find_col(gdf, ["s1_date", "S1Date", "date", "Date"])
    country_col  = _find_col(gdf, ["ISO_CC", "iso_cc", "country", "Country"])

    event_lookup = {str(row[location_col]).strip().lower(): row for _, row in gdf.iterrows()}
    split_lookup = {k.lower(): v for k, v in EVENT_SPLIT_MAP.items()}

    # 4. Build weak chip rows
    weak_rows, unmatched = [], []
    for f in tqdm(weak_files, desc="Processing weak chips"):
        basename  = os.path.basename(f)
        chip_name = re.sub(r"_S1Weak\.tif$", "", basename, flags=re.IGNORECASE)

        try:
            with rasterio.open(f) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326",
                                          src.bounds.left, src.bounds.bottom,
                                          src.bounds.right, src.bounds.top)
                lon_min, lat_min, lon_max, lat_max = bounds
        except Exception:
            continue

        matched_event = matched_loc = None
        for alias_prefix, alias_target in LOCATION_ALIASES.items():
            if chip_name.lower().startswith(alias_prefix.lower()):
                if alias_target.lower() in event_lookup:
                    matched_event = event_lookup[alias_target.lower()]
                    matched_loc = alias_prefix
                break
        if matched_event is None:
            for loc_lower, row in event_lookup.items():
                if chip_name.lower().startswith(loc_lower):
                    matched_event = row
                    matched_loc = str(row[location_col]).strip()
                    break
        if matched_event is None:
            unmatched.append(chip_name)
            continue

        s1_date  = str(matched_event[date_col])[:10] if date_col and pd.notna(matched_event[date_col]) else None
        country  = str(matched_event[country_col]) if country_col and pd.notna(matched_event[country_col]) else ""
        split_val = split_lookup.get(str(matched_loc).lower(), "train")

        weak_rows.append({
            "chip_name":    chip_name,
            "s1_date":      s1_date,
            "lon_min": lon_min, "lat_min": lat_min,
            "lon_max": lon_max, "lat_max": lat_max,
            "split":        split_val,
            "country":      country,
            "label_quality": "weak",
            "label_dir":    os.path.join(sen1_dir, "LabelWeak"),
            "s1_dir":       s1weak_dir,
            "s2_dir":       os.path.join(sen1_dir, "S2Weak"),
        })

    if unmatched:
        print(f"WARNING: {len(unmatched)} weak chips unmatched.")

    df_weak    = pd.DataFrame(weak_rows)
    df_combined = pd.concat([df_hand, df_weak], ignore_index=True)
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"\nSaved combined metadata → {combined_csv_path}")
    print(f"  Hand chips : {len(df_hand)}")
    print(f"  Weak chips : {len(df_weak)}")
    print(f"  Total      : {len(df_combined)}")
    print("\nSplit breakdown:")
    print(df_combined["split"].value_counts().to_string())
    print("Section G complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

SECTIONS = {
    "A":  section_a,
    "B":  section_b,
    "C":  section_c,
    "C2": section_c2,
    "D":  section_d,
    "F":  section_f,
    "G":  section_g,
}
GEE_SECTIONS = {"C", "C2", "D"}


def main():
    parser = argparse.ArgumentParser(
        description="Download all Sen1Floods11 data for flood segmentation training."
    )
    parser.add_argument(
        "--section",
        choices=[*SECTIONS.keys(), "all"],
        required=True,
        help="Which section to run. Use 'all' to run A → B → F → G → C → C2 → D in order.",
    )
    parser.add_argument(
        "--base_dir",
        default=DEFAULT_BASE_DIR,
        help="Root directory where data/ will be created. Default: current directory.",
    )
    parser.add_argument(
        "--gee_project",
        default=None,
        help="Your Google Earth Engine project ID (required for sections C, C2, D).",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    os.makedirs(base_dir, exist_ok=True)

    if args.section == "all":
        order = ["A", "B", "F", "G", "C", "C2", "D"]
    else:
        order = [args.section]

    for sec in order:
        if sec in GEE_SECTIONS and not args.gee_project:
            print(f"ERROR: Section {sec} requires --gee_project. Provide your GEE project ID.")
            sys.exit(1)
        fn = SECTIONS[sec]
        if sec in GEE_SECTIONS:
            fn(base_dir, args.gee_project)
        else:
            fn(base_dir)

    print("\n" + "=" * 60)
    print("ALL REQUESTED DOWNLOADS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
