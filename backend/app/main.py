from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
import segmentation_models_pytorch as smp
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration ---
app = FastAPI(title="DisasterDL API")

# Allow CORS for frontend development
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tiling Configuration ---
TILE_SIZE = 512  # The model was trained on 512x512 images
OVERLAP = 64     # Overlap between tiles to reduce edge artifacts


# --- Model and Preprocessing Setup ---
def get_model():
    """Defines the model architecture."""
    model = smp.Unet(
        encoder_name="efficientnet-b2",
        encoder_weights=None,  # We are loading our own trained weights
        in_channels=3,
        classes=1
    )
    return model

# --- Paths and Model Loading ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v1model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = get_model()
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def preprocess_sar(s1_data):
    """Preprocesses the 2-channel SAR data into a 3-channel image."""
    s1_data = np.nan_to_num(s1_data, nan=0.0)
    vv = np.clip(s1_data[0], -40, 0)
    vh = np.clip(s1_data[1], -50, -7)
    ratio = np.clip(vv - vh, 0, 30)
    vv_norm = (vv + 40) / 40.0
    vh_norm = (vh + 50) / 43.0
    ratio_norm = ratio / 30.0
    return np.stack([vv_norm, vh_norm, ratio_norm])

# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the DisasterDL API for Flood Segmentation"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Accepts a large GeoTIFF file, runs prediction in tiles, and returns the stitched flood mask."""
    if not file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a GeoTIFF (.tif or .tiff).")

    try:
        logger.info(f"Received file: {file.filename}")
        content = await file.read()
        
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                height, width = src.height, src.width
                profile = src.profile
                logger.info(f"Processing large image of size {width}x{height}")

                stitched_prediction = np.zeros((height, width), dtype=np.float32)
                overlap_counts = np.zeros((height, width), dtype=np.uint8)
                step = TILE_SIZE - OVERLAP

                for r in range(0, height, step):
                    for c in range(0, width, step):
                        window = rasterio.windows.Window(c, r, TILE_SIZE, TILE_SIZE)
                        tile_data = src.read(window=window)
                        tile_height, tile_width = tile_data.shape[1], tile_data.shape[2]

                        if tile_height == 0 or tile_width == 0:
                            continue

                        padded_tile = np.zeros((tile_data.shape[0], TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype)
                        padded_tile[:, :tile_height, :tile_width] = tile_data
                        
                        s1_processed = preprocess_sar(padded_tile.astype(np.float32))
                        input_img = np.transpose(s1_processed, (1, 2, 0))

                        transform = A.Compose([ToTensorV2()])
                        augmented = transform(image=input_img)
                        image_tensor = augmented['image'].unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(image_tensor)
                            output_sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()

                        prediction_tile = output_sigmoid[:tile_height, :tile_width]
                        stitched_prediction[r:r+tile_height, c:c+tile_width] += prediction_tile
                        overlap_counts[r:r+tile_height, c:c+tile_width] += 1
                
                overlap_counts[overlap_counts == 0] = 1
                averaged_prediction = stitched_prediction / overlap_counts
                prediction_mask = (averaged_prediction > 0.5).astype(np.uint8)
                logger.info("Tiled prediction and stitching complete.")

        logger.info("Encoding final prediction mask to GeoTIFF...")
        with MemoryFile() as memfile_out:
            profile.pop('nodata', None)
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw', width=width, height=height)
            
            with memfile_out.open(**profile) as dst:
                dst.write(prediction_mask, 1)
            
            memfile_out.seek(0)
            logger.info("Sending response.")
            return Response(content=memfile_out.read(), media_type="image/tiff")

    except Exception as e:
        logger.exception("An error occurred during prediction")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
