"""Обучение модели для денойзинга и сегментации пор."""

import os
import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt


class PoreDataset(Dataset):
    """Dataset для загрузки пар noisy-clean изображений."""
    
    def __init__(self, root_dir: str, split: str = "train", augment: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment and (split == "train")
        
        # Загружаем метаданные
        with open(self.root_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.files = self.metadata["files"][split]
        self.clean_dir = self.root_dir / split / "clean"
        self.noisy_dir = self.root_dir / split / "noisy"
        
        print(f"Загружен {split} датасет: {len(self.files)} пар изображений")
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_info = self.files[idx]
        
        # Загружаем изображения
        noisy = cv2.imread(str(self.noisy_dir / file_info["noisy"]), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(self.clean_dir / file_info["clean"]), cv2.IMREAD_GRAYSCALE)
        
        if noisy is None or clean is None:
            raise FileNotFoundError(f"Не удалось загрузить изображения для {file_info['id']}")
        
        # Нормализация
        noisy = noisy.astype(np.float32) / 255.0
        clean = (clean < 127).astype(np.float32)  # Бинаризация маски (поры = 1, фон = 0)
        
        # Аугментации
        if self.augment:
            noisy, clean = self._augment(noisy, clean)
        
        # Конвертируем в тензоры (добавляем channel dimension)
        noisy = torch.from_numpy(noisy).unsqueeze(0)  # (1, H, W)
        clean = torch.from_numpy(clean).unsqueeze(0)
        
        return noisy, clean
    
    def _augment(self, noisy: np.ndarray, clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Применяет аугментации."""
        # Случайный поворот на 90 градусов
        k = np.random.randint(0, 4)
        noisy = np.rot90(noisy, k)
        clean = np.rot90(clean, k)
        
        # Случайное отражение
        if np.random.rand() > 0.5:
            noisy = np.fliplr(noisy)
            clean = np.fliplr(clean)
        if np.random.rand() > 0.5:
            noisy = np.flipud(noisy)
            clean = np.flipud(clean)
        
        # Легкая вариация яркости/контраста
        if np.random.rand() > 0.5:
            noisy = noisy * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
            noisy = np.clip(noisy, 0, 1)
        
        return noisy.copy(), clean.copy()


class AttentionBlock(nn.Module):
    """Attention gate для U-Net."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Приводим размеры к одинаковым (на случай несовпадения из-за pooling/upsampling)
        if g1.shape != x1.shape:
            g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetWithAttention(nn.Module):
    """U-Net с Attention gates для сегментации пор."""
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)
        
        # Decoder с Attention
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = self._block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self._block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self._block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = self._block(features * 2, features)
        
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder с Attention
        dec4 = self.upconv4(bottleneck)
        # Выравниваем размеры перед attention
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = torch.nn.functional.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        # Выравниваем размеры перед attention
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = torch.nn.functional.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        # Выравниваем размеры перед attention
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = torch.nn.functional.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        # Выравниваем размеры перед attention
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = torch.nn.functional.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        enc1 = self.att1(dec1, enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)  # Убираем sigmoid - будет в loss


class CombinedLoss(nn.Module):
    """Комбинированный лосс: BCE + Dice + Edge-aware."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.3, edge_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.bce = nn.BCEWithLogitsLoss()  # Безопасно для autocast
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss для лучшей сегментации."""
        # Применяем sigmoid к logits
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = pred_sigmoid.view(-1)
        target = target.view(-1)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        return 1 - dice
    
    def edge_loss(self, pred, target):
        """Edge-aware loss для сохранения границ пор."""
        # Применяем sigmoid к logits
        pred_sigmoid = torch.sigmoid(pred)
        
        # Sobel фильтры для градиентов
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        pred_edge_x = torch.nn.functional.conv2d(pred_sigmoid, sobel_x, padding=1)
        pred_edge_y = torch.nn.functional.conv2d(pred_sigmoid, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        target_edge_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_edge_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        
        return torch.nn.functional.l1_loss(pred_edge, target_edge)
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        total = (self.bce_weight * bce + 
                 self.dice_weight * dice + 
                 self.edge_weight * edge)
        
        return total, {"bce": bce.item(), "dice": dice.item(), "edge": edge.item()}


def calculate_metrics(pred, target, threshold=0.5):
    """Вычисляет метрики качества."""
    # Применяем sigmoid к logits
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = target
    
    # Dice Score
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    
    # IoU
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # Пористость (процент черных пикселей)
    pred_porosity = pred_binary.mean()
    target_porosity = target_binary.mean()
    porosity_error = abs(pred_porosity - target_porosity)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "porosity_error": porosity_error.item(),
        "pred_porosity": pred_porosity.item(),
        "target_porosity": target_porosity.item(),
    }


class Trainer:
    """Класс для обучения модели."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda') if use_amp else None
        
        self.best_val_dice = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    
    def train_epoch(self, epoch: int) -> float:
        """Обучение на одной эпохе."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    pred = self.model(noisy)
                    loss, loss_dict = self.criterion(pred, clean)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(noisy)
                loss, loss_dict = self.criterion(pred, clean)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), **loss_dict)
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> dict:
        """Валидация модели."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {"dice": [], "iou": [], "porosity_error": []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for noisy, clean in pbar:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                pred = self.model(noisy)
                loss, _ = self.criterion(pred, clean)
                total_loss += loss.item()
                
                # Вычисляем метрики
                metrics = calculate_metrics(pred, clean)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                pbar.set_postfix(
                    loss=loss.item(),
                    dice=metrics["dice"],
                    iou=metrics["iou"]
                )
        
        # Усредняем метрики
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        avg_metrics["loss"] = total_loss / len(self.val_loader)
        
        return avg_metrics
    
    def train(self, num_epochs: int, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """Полный цикл обучения."""
        print(f"\n{'='*70}")
        print(f"НАЧАЛО ОБУЧЕНИЯ")
        print(f"{'='*70}\n")
        print(f"Устройство: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Эпох: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Обучение
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            # Валидация
            val_metrics = self.validate(epoch)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            
            # LR scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Сохраняем лучшую модель
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ Новая лучшая модель! Dice: {self.best_val_dice:.4f}")
            
            # Сохраняем последнюю модель
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Выводим статистику
            print(f"\n{'='*70}")
            print(f"Эпоха {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Porosity Error: {val_metrics['porosity_error']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*70}\n")
        
        # Сохраняем историю обучения
        self.plot_history()
        print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! Лучший Dice: {self.best_val_dice:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Сохраняет чекпоинт модели."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "history": self.history,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
    
    def plot_history(self):
        """Строит графики обучения."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.history["train_loss"], label="Train Loss")
        axes[0].plot(self.history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Dice
        axes[1].plot(self.history["val_dice"], label="Val Dice", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title("Validation Dice Score")
        axes[1].legend()
        axes[1].grid(True)
        
        # IoU
        axes[2].plot(self.history["val_iou"], label="Val IoU", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("IoU")
        axes[2].set_title("Validation IoU")
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / "training_history.png", dpi=150)
        print(f"📊 График сохранен: {self.checkpoint_dir / 'training_history.png'}")


def main():
    # Параметры
    DATASET_DIR = "./dataset"
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Инициализация...")
    
    # Создаем датасеты
    train_dataset = PoreDataset(DATASET_DIR, split="train", augment=True)
    val_dataset = PoreDataset(DATASET_DIR, split="val", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Создаем модель
    model = UNetWithAttention(in_channels=1, out_channels=1, init_features=32)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total_params:,}")
    
    # Лосс и оптимизатор
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.3, edge_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # LR Scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Трейнер
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        checkpoint_dir="./checkpoints",
        use_amp=True  # Mixed precision для ускорения
    )
    
    # Запускаем обучение
    trainer.train(num_epochs=NUM_EPOCHS, scheduler=scheduler)


if __name__ == "__main__":
    main()

