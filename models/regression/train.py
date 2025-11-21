"""Training script for Regression UNet (Distance Map Prediction)."""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import model
from models.regression.model import RegressionUNet

class AugmentedDataset(Dataset):
    def __init__(self, file_list, root_dir):
        self.files = file_list
        self.root_dir = Path(root_dir)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # file_info is just the filename of the distance map, e.g. "1_rot0_dist.png"
        dist_name = self.files[idx]
        # Image name is dist_name without "_dist.png" + ".png"
        img_name = dist_name.replace("_dist.png", ".png")
        
        img_path = self.root_dir / img_name
        dist_path = self.root_dir / dist_name
        
        # Load images
        # Input image is grayscale
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        dist = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        if dist is None:
            raise ValueError(f"Could not load dist: {dist_path}")
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        dist = dist.astype(np.float32)
        
        # To Tensor
        image = torch.from_numpy(image).unsqueeze(0) # (1, H, W)
        dist = torch.from_numpy(dist).unsqueeze(0)   # (1, H, W)
        
        return image, dist

def train():
    # Config
    # Use absolute path or relative to script? Let's use absolute for safety based on user input
    DATASET_DIR = Path(r"d:\pore_research_and_ai\RealPoresImages\dataset_augmented")
    CHECKPOINT_DIR = Path(r"d:\pore_research_and_ai\models\regression\checkpoints_augmented")
    BATCH_SIZE = 8 # Smaller batch size for safety
    NUM_EPOCHS = 50
    LR = 1e-3
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data Preparation
    print(f"Scanning {DATASET_DIR}...")
    all_dist_files = [f.name for f in DATASET_DIR.glob("*_dist.png")]
    
    if not all_dist_files:
        print("No data found! Check the path.")
        return

    # Shuffle and Split
    np.random.shuffle(all_dist_files)
    split_idx = int(len(all_dist_files) * 0.8)
    train_files = all_dist_files[:split_idx]
    val_files = all_dist_files[split_idx:]
    
    print(f"Found {len(all_dist_files)} images. Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_ds = AugmentedDataset(train_files, DATASET_DIR)
    val_ds = AugmentedDataset(val_files, DATASET_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = RegressionUNet().to(DEVICE)
    
    # Loss: MSE for regression
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    scaler = GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}
    
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for noisy, dist in pbar:
            noisy, dist = noisy.to(DEVICE), dist.to(DEVICE)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    pred = model(noisy)
                    loss = criterion(pred, dist)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy)
                loss = criterion(pred, dist)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, dist in val_loader:
                noisy, dist = noisy.to(DEVICE), dist.to(DEVICE)
                pred = model(noisy)
                loss = criterion(pred, dist)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print("✅ Saved best model")
            
    print("Training Complete.")
    
    # Save history
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    train()
