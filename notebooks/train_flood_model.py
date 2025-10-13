import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import sys
import warnings

# Suppress minor warnings that don't affect training
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent # Assumes the script is in a 'notebooks' or 'scripts' folder

# IMPORTANT: These paths are now based on your project's root folder
DATA_DIR = PROJECT_ROOT / "data" / "flood_sar" / "sen1floods11" / "v1.1"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "flood_training_final.log"

# --- Hyperparameters ---
NUM_EPOCHS = 50
BATCH_SIZE = 4      # Increase if your GPU can handle it, decrease if you run out of memory
LEARNING_RATE = 1e-4 # A stable starting learning rate
NUM_WORKERS = 0      # Keep at 0 for macOS MPS compatibility
POS_WEIGHT = 9.5     # Your calculated positive class weight
ALPHA = 0.5          # Focal loss alpha (0.5 is a neutral starting point)
GAMMA = 2.0          # Focal loss gamma
THRESHOLD = 0.5      # Standard probability threshold for binary classification

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)

# --- Helper Functions & Classes ---

def preprocess_sar(s1_data):
    """Preprocesses the 2-channel SAR data into a 3-channel image."""
    s1_data = np.nan_to_num(s1_data, nan=0.0)
    # Clamp VV and VH bands to a standard db range
    vv = np.clip(s1_data[0], -40, 0)
    vh = np.clip(s1_data[1], -50, -7)
    # Create a ratio band using subtraction, which is often more stable than division
    ratio = np.clip(vv - vh, 0, 30)
    # Normalize each channel to [0, 1] for the model
    vv_norm = (vv + 40) / 40.0
    vh_norm = (vh + 50) / 43.0
    ratio_norm = ratio / 30.0
    return np.stack([vv_norm, vh_norm, ratio_norm])

# <<< CRITICAL FIX: USING YOUR ORIGINAL, CORRECT PATHS >>>
class HandLabeledDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        s1_folder = self.data_dir / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        label_folder = self.data_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand"
        
        split_file_name = f"flood_{split}_data.csv"
        split_file = self.data_dir / "splits" / "flood_handlabeled" / split_file_name

        self.image_pairs = []
        try:
            with open(split_file, 'r') as f:
                next(f, None) # Skip header row
                for row in csv.reader(f):
                    if len(row) >= 2:
                        s1_file = s1_folder / row[0].strip()
                        label_file = label_folder / row[1].strip()
                        if s1_file.exists() and label_file.exists():
                            self.image_pairs.append({'s1': s1_file, 'label': label_file})
        except FileNotFoundError:
            logging.error(f"FATAL: The split file was not found at {split_file}")
            raise

        if not self.image_pairs:
            raise RuntimeError(f"No image pairs were loaded for the '{split}' split. Check paths and CSV content.")
            
        logging.info(f"✅ Loaded {len(self.image_pairs)} images for '{split}' split from {split_file.name}.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        with rasterio.open(pair['s1']) as src:
            s1_data = src.read().astype(np.float32)
        s1_data_processed = preprocess_sar(s1_data)
        
        with rasterio.open(pair['label']) as src:
            label = src.read(1).astype(np.float32)
        
        valid_mask = (label != -1).astype(np.float32)
        flood_mask = (label == 1).astype(np.float32)
        
        input_img = np.transpose(s1_data_processed, (1, 2, 0))
        combined_mask = np.stack([flood_mask, valid_mask], axis=-1)

        if self.split == 'train':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                ToTensorV2(),
            ])
            
        augmented = transform(image=input_img, mask=combined_mask)
        images = augmented['image']
        flood_mask_aug = augmented['mask'][:, :, 0]
        valid_mask_aug = augmented['mask'][:, :, 1]
        
        return images, flood_mask_aug, valid_mask_aug

# <<< CRITICAL FIX: MASKED LOSS FUNCTION >>>
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=1.0, reduction='none'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight)
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
            
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt).pow(self.gamma)
        alpha_term = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_term * focal_term * bce_loss
        return loss

def focal_dice_loss(outputs, masks, valid_masks):
    focal = WeightedFocalLoss(alpha=ALPHA, gamma=GAMMA, pos_weight=POS_WEIGHT, reduction='none')
    dice = smp.losses.DiceLoss(mode="binary")

    outputs = outputs.squeeze(1)

    focal_loss_unmasked = focal(outputs, masks)
    focal_loss_masked = focal_loss_unmasked * valid_masks
    valid_pixels = torch.sum(valid_masks)
    focal_loss = torch.sum(focal_loss_masked) / (valid_pixels + 1e-8)

    outputs_sigmoid = torch.sigmoid(outputs)
    masked_outputs_sigmoid = outputs_sigmoid * valid_masks
    masked_targets = masks * valid_masks
    dice_loss = dice(masked_outputs_sigmoid, masked_targets)
    
    return focal_loss + dice_loss

# <<< ENHANCEMENT: COMPREHENSIVE METRICS >>>
@torch.no_grad()
def manual_metrics(outputs_sigmoid, masks, valid_masks, threshold=THRESHOLD):
    preds = (outputs_sigmoid > threshold).float()
    
    preds_masked = preds * valid_masks
    masks_masked = masks * valid_masks
    
    tp = torch.sum(preds_masked * masks_masked)
    fp = torch.sum(preds_masked * (1 - masks_masked))
    fn = torch.sum((1 - preds_masked) * masks_masked)
    
    iou = (tp / (tp + fp + fn + 1e-8)).item()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return iou, precision, recall, f1

# --- Training & Validation Loops ---
def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, masks, valid_masks) in enumerate(loader):
        images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = focal_dice_loss(outputs, masks, valid_masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 20 == 0:
            logging.info(f"  [Epoch {epoch}, Batch {batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss, all_ious, all_precs, all_recs, all_f1s = 0.0, [], [], [], []
    
    with torch.no_grad():
        for images, masks, valid_masks in loader:
            images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
            outputs = model(images)
            loss = focal_dice_loss(outputs, masks, valid_masks)
            
            total_loss += loss.item()
            iou, prec, rec, f1 = manual_metrics(torch.sigmoid(outputs.squeeze(1)), masks, valid_masks)
            all_ious.append(iou)
            all_precs.append(prec)
            all_recs.append(rec)
            all_f1s.append(f1)
            
    return (total_loss / len(loader), np.mean(all_ious), np.mean(all_precs), np.mean(all_recs), np.mean(all_f1s))

# --- Main Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("=" * 60)
    logging.info("🌊 Flood Mapping Model Training: Final Corrected Script")
    logging.info(f"Using device: {device}")
    logging.info(f"Hyperparameters: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Epochs={NUM_EPOCHS}")
    logging.info("=" * 60)
    
    train_dataset = HandLabeledDataset(DATA_DIR, 'train')
    valid_dataset = HandLabeledDataset(DATA_DIR, 'valid') # Using 'valid' as per your original file names

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = smp.Unet("efficientnet-b2", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_iou = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"\n--- EPOCH {epoch}/{NUM_EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        valid_loss, valid_iou, valid_prec, valid_rec, valid_f1 = validate(model, valid_loader, device)
        
        scheduler.step(valid_iou)
        
        logging.info(f"📊 TRAIN  -> Loss: {train_loss:.4f}")
        logging.info(f"📊 VALID  -> Loss: {valid_loss:.4f} | IoU: {valid_iou:.4f} | F1: {valid_f1:.4f} | Precision: {valid_prec:.4f} | Recall: {valid_rec:.4f}")
        
        if valid_iou > best_iou:
            best_iou = valid_iou
            model_path = MODELS_DIR / "best_flood_model.pth"
            torch.save(model.state_dict(), model_path)
            logging.info(f"🏆 NEW BEST IoU: {best_iou:.4f}. Model saved to {model_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info(f"🎉 Training Complete! Best Validation IoU: {best_iou:.4f}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()