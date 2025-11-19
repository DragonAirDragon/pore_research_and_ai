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

# Import model
from model import RegressionUNet

class RegressionDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and (split == "train")
        
        with open(self.root_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        self.files = self.metadata["files"][split]
        self.noisy_dir = self.root_dir / split / "noisy"
        self.dist_dir = self.root_dir / split / "distance"
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_info = self.files[idx]
        
        # Load images
        noisy = cv2.imread(str(self.noisy_dir / file_info["noisy"]), cv2.IMREAD_GRAYSCALE)
        dist = cv2.imread(str(self.dist_dir / file_info["distance"]), cv2.IMREAD_GRAYSCALE)
        
        # Normalize
        # Input: 0-1
        noisy = noisy.astype(np.float32) / 255.0
        
        # Target: Keep as absolute pixel values? Or normalize?
        # Let's keep absolute for easier interpretation, but ensure float32.
        # Note: dist is loaded as uint8, so max value is 255.
        dist = dist.astype(np.float32)
        
        # Augmentation (Simple flip/rot)
        if self.augment:
            k = np.random.randint(0, 4)
            noisy = np.rot90(noisy, k)
            dist = np.rot90(dist, k)
            
            if np.random.rand() > 0.5:
                noisy = np.fliplr(noisy)
                dist = np.fliplr(dist)
                
        # To Tensor
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0) # (1, H, W)
        dist = torch.from_numpy(dist.copy()).unsqueeze(0)   # (1, H, W)
        
        return noisy, dist

def train():
    # Config
    DATASET_DIR = "../dataset_regression"
    CHECKPOINT_DIR = "./checkpoints_regression"
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-3
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Data
    train_ds = RegressionDataset(DATASET_DIR, split="train")
    val_ds = RegressionDataset(DATASET_DIR, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = RegressionUNet().to(DEVICE)
    
    # Loss: MSE for regression
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # GradScaler is primarily for CUDA mixed precision. 
    # MPS supports mixed precision but GradScaler support varies. 
    # For safety/simplicity on Mac, we'll disable it for MPS or CPU.
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
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print("✅ Saved best model")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
