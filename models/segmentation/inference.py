"""Инференс обученной модели и визуализация результатов."""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.segmentation.train import UNetWithAttention, PoreDataset, calculate_metrics


def load_model(checkpoint_path: str, device: torch.device):
    """Загружает обученную модель."""
    model = UNetWithAttention(in_channels=1, out_channels=1, init_features=32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ Модель загружена из: {checkpoint_path}")
    print(f"   Эпоха: {checkpoint['epoch']}")
    print(f"   Best Dice: {checkpoint['best_val_dice']:.4f}")
    return model


def predict_image(model, image_path: str, device: torch.device):
    """Предсказание для одного изображения."""
    # Загружаем изображение
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    img_norm = img.astype(np.float32) / 255.0
    
    # Конвертируем в тензор
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    
    # Конвертируем обратно
    pred_mask = pred.squeeze().cpu().numpy()
    pred_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    
    return img, pred_mask, pred_binary


def visualize_results(noisy, mask_prob, mask_binary, ground_truth=None, save_path=None):
    """Визуализирует результаты."""
    num_plots = 3 if ground_truth is None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Noisy Input', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(mask_prob, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Predicted Mask (Probability)', fontsize=14)
    axes[1].axis('off')
    cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)
    cbar1.set_label('Вероятность', rotation=270, labelpad=15)
    
    axes[2].imshow(mask_binary, cmap='gray')
    axes[2].set_title('Predicted Mask (Binary)', fontsize=14)
    axes[2].axis('off')
    
    if ground_truth is not None:
        axes[3].imshow(ground_truth, cmap='gray')
        axes[3].set_title('Ground Truth', fontsize=14)
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Результат сохранен: {save_path}")
    
    plt.close()


def evaluate_test_set(model, dataset_dir: str, device: torch.device, num_samples: int = 10):
    """Оценка на тестовом наборе."""
    test_dataset = PoreDataset(dataset_dir, split="test", augment=False)
    
    output_dir = Path("./artifacts/evaluations/segmentation_test")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
    print(f"{'='*70}\n")
    print(f"Всего примеров в тесте: {len(test_dataset)}")
    print(f"Будет обработано: {min(num_samples, len(test_dataset))}\n")
    
    all_dice_scores = []
    all_iou_scores = []
    
    for i in range(min(num_samples, len(test_dataset))):
        noisy_tensor, gt_tensor = test_dataset[i]
        noisy_tensor = noisy_tensor.unsqueeze(0).to(device)
        gt_tensor = gt_tensor.unsqueeze(0).to(device)
        
        # Предсказание
        with torch.no_grad():
            pred = torch.sigmoid(model(noisy_tensor))
        
        # Вычисляем метрики
        metrics = calculate_metrics(pred, gt_tensor)
        all_dice_scores.append(metrics["dice"])
        all_iou_scores.append(metrics["iou"])
        
        # Конвертируем для визуализации
        noisy_np = (noisy_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_prob = pred.squeeze().cpu().numpy()
        pred_binary = (pred_prob > 0.5).astype(np.uint8) * 255
        gt_np = (gt_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # Визуализируем
        save_path = output_dir / f"test_sample_{i:03d}.png"
        visualize_results(noisy_np, pred_prob, pred_binary, gt_np, save_path=save_path)
        
        print(f"Sample {i:03d}: Dice = {metrics['dice']:.4f}, IoU = {metrics['iou']:.4f}, "
              f"Porosity Error = {metrics['porosity_error']:.4f}")
    
    # Итоговая статистика
    print(f"\n{'='*70}")
    print(f"ИТОГОВЫЕ МЕТРИКИ")
    print(f"{'='*70}")
    print(f"Средний Dice Score: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"Средний IoU:        {np.mean(all_iou_scores):.4f} ± {np.std(all_iou_scores):.4f}")
    print(f"{'='*70}\n")


def predict_custom_image(model, image_path: str, device: torch.device, output_path: str = None):
    """Предсказание для пользовательского изображения."""
    print(f"\nОбработка изображения: {image_path}")
    
    noisy, pred_prob, pred_binary = predict_image(model, image_path, device)
    
    if output_path is None:
        output_path = Path(image_path).stem + "_result.png"
    
    visualize_results(noisy, pred_prob, pred_binary, save_path=output_path)
    
    # Вычисляем пористость
    porosity = (pred_binary == 255).sum() / pred_binary.size
    print(f"Предсказанная пористость: {porosity:.2%}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Инференс модели сегментации пор")
    parser.add_argument(
        "--model",
        type=str,
        default="./artifacts/checkpoints/segmentation/best_model.pth",
        help="Путь к чекпоинту модели"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset",
        help="Путь к датасету для оценки"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Количество примеров для визуализации"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Путь к пользовательскому изображению для обработки"
    )
    
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {DEVICE}\n")
    
    # Загружаем модель
    model = load_model(args.model, DEVICE)
    
    if args.image:
        # Обработка пользовательского изображения
        predict_custom_image(model, args.image, DEVICE)
    else:
        # Оцениваем на тестовом наборе
        evaluate_test_set(model, args.dataset, DEVICE, num_samples=args.num_samples)


if __name__ == "__main__":
    main()

