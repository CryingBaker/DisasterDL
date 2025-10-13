# DisasterDL: Deep Learning System for Disaster and Impact Assessment

## Project Description
DisasterDL is a deep learning project that focuses on **flood detection** and **building damage assessment** using **satellite imagery**.  
The goal is to create models that can automatically detect flooded areas and assess the impact on buildings from optical satellite images, such as those from FloodNet and xBD datasets.

---

## Dataset

To get started, you need to prepare your dataset.

### 1. Create the following folder structure:

data/flood/train/
data/flood/test/

### 2. Download and place your dataset into these folders:

- **Training images and masks:** Place inside `data/flood/train/`  
- **Testing images and masks:** Place inside `data/flood/test/`  

### 3. Dataset links:

- **Flood dataset (training/testing):** `Sentinel 1 Dataset`
Install it using gcloud storage cp -r gs://sen1floods11 data/flood_sar 
- **Building damage dataset (optional, for later):** `[Insert link here]`  

> Note: Make sure the images and corresponding masks have the **same filename** for proper training.

---

## Next Steps

- Preprocess images and masks (resize, normalize, optional augmentation)  
- Implement and train the flood detection model (U-Net)  
- Visualize predictions on test images
