"""Скрипт для генерации большого датасета с разными конфигурациями."""

import json
import os
import random
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.data_prep.generate_images import PoreImageGenerator


def create_varied_config(base_config: dict, variation_seed: int) -> dict:
    """
    Создает вариацию базовой конфигурации с рандомизацией.
    
    Args:
        base_config: Базовая конфигурация
        variation_seed: Сид для генерации вариации
    
    Returns:
        Вариация конфигурации
    """
    random.seed(variation_seed)
    config = json.loads(json.dumps(base_config))  # Глубокое копирование
    
    # Варьируем количество пор
    for pore_type in ["small_pores", "medium_pores", "large_pores"]:
        if pore_type in config["pore_settings"]:
            orig_range = config["pore_settings"][pore_type]["count_range"]
            variation = random.uniform(0.7, 1.3)
            config["pore_settings"][pore_type]["count_range"] = [
                max(1, int(orig_range[0] * variation)),
                max(2, int(orig_range[1] * variation))
            ]
            
            # Варьируем размер
            orig_radius = config["pore_settings"][pore_type]["radius_mean_relative"]
            config["pore_settings"][pore_type]["radius_mean_relative"] = \
                orig_radius * random.uniform(0.85, 1.15)
            
            # Варьируем растяжение
            orig_stretch = config["pore_settings"][pore_type]["stretch_factor_range"]
            config["pore_settings"][pore_type]["stretch_factor_range"] = [
                orig_stretch[0],
                orig_stretch[1] * random.uniform(0.8, 1.2)
            ]
    
    # Варьируем шум
    config["noise_settings"]["min_gray_value"] = random.randint(80, 120)
    config["noise_settings"]["max_gray_value"] = random.randint(180, 220)
    config["noise_settings"]["noise_intensity"] = random.uniform(0.05, 0.2)
    config["noise_settings"]["pore_noise"]["min_value"] = random.randint(0, 30)
    config["noise_settings"]["pore_noise"]["max_value"] = random.randint(80, 120)
    
    return config


def generate_large_dataset(
    total_images: int = 20000,
    images_per_batch: int = 100,
    base_configs: list = None
):
    """
    Генерирует большой датасет с разнообразными конфигурациями.
    
    Args:
        total_images: Общее количество изображений
        images_per_batch: Количество изображений в одном батче
        base_configs: Список базовых конфигураций для вариаций
    """
    if base_configs is None:
        base_configs = [
            {
                "name": "balanced",
                "config": {
                    "image_settings": {"width": 200, "height": 200, "total_images": images_per_batch},
                    "pore_settings": {
                        "small_pores": {
                            "count_range": [20, 35],
                            "radius_mean_relative": 0.04,
                            "min_distance_relative": 0.015,
                            "stretch_enabled": True,
                            "stretch_factor_range": [1, 1.5],
                            "rotation_enabled": True
                        },
                        "medium_pores": {
                            "count_range": [12, 20],
                            "radius_mean_relative": 0.075,
                            "min_distance_relative": 0.025,
                            "stretch_enabled": True,
                            "stretch_factor_range": [1, 1.5],
                            "rotation_enabled": True
                        },
                        "large_pores": {
                            "count_range": [6, 12],
                            "radius_mean_relative": 0.125,
                            "min_distance_relative": 0.04,
                            "stretch_enabled": True,
                            "stretch_factor_range": [1, 1.5],
                            "rotation_enabled": True
                        }
                    },
                    "noise_settings": {
                        "min_gray_value": 100,
                        "max_gray_value": 200,
                        "noise_intensity": 0.1,
                        "pore_noise": {
                            "enabled": True,
                            "min_value": 0,
                            "max_value": 100,
                            "texture_enabled": True
                        }
                    },
                    "output_settings": {
                        "clean_dir": "./artifacts/generated/synthetic/clean_background",
                        "noisy_dir": "./artifacts/generated/synthetic/noisy_background"
                    }
                }
            }
        ]
    
    total_batches = (total_images + images_per_batch - 1) // images_per_batch
    temp_config_path = "temp_large_dataset_config.json"
    
    print("=" * 70)
    print("ГЕНЕРАЦИЯ БОЛЬШОГО ДАТАСЕТА")
    print("=" * 70)
    print(f"\nПараметры:")
    print(f"  Всего изображений: {total_images}")
    print(f"  Изображений в батче: {images_per_batch}")
    print(f"  Всего батчей: {total_batches}")
    print(f"  Базовых конфигураций: {len(base_configs)}")
    print(f"\n{'=' * 70}\n")
    
    generated_total = 0
    
    for batch_idx in range(total_batches):
        # Выбираем случайную базовую конфигурацию
        base_config_data = random.choice(base_configs)
        
        # Создаем вариацию
        varied_config = create_varied_config(
            base_config_data["config"],
            variation_seed=batch_idx
        )
        
        # Корректируем количество изображений для последнего батча
        remaining = total_images - generated_total
        batch_size = min(images_per_batch, remaining)
        varied_config["image_settings"]["total_images"] = batch_size
        
        print(f"Батч {batch_idx + 1}/{total_batches}")
        print(f"  База: {base_config_data['name']}")
        print(f"  Изображений: {batch_size}")
        print(f"  Прогресс: {generated_total + batch_size}/{total_images}")
        
        # Сохраняем временную конфигурацию
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(varied_config, f, indent=2, ensure_ascii=False)
        
        # Генерируем изображения
        try:
            generator = PoreImageGenerator(temp_config_path)
            generator.generate_images()
            generated_total += batch_size
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
        
        print()
    
    # Удаляем временную конфигурацию
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    print("=" * 70)
    print("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\nСгенерировано изображений: {generated_total}/{total_images}")
    print(f"\n📁 Изображения сохранены в:")
    print(f"  - ./artifacts/generated/synthetic/clean_background/ (чистые маски)")
    print(f"  - ./artifacts/generated/synthetic/noisy_background/ (зашумленные)")


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Генерация большого датасета")
    parser.add_argument(
        "--total",
        type=int,
        default=5000,
        help="Общее количество изображений (по умолчанию: 5000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Количество изображений в батче (по умолчанию: 100)"
    )
    
    args = parser.parse_args()
    
    generate_large_dataset(
        total_images=args.total,
        images_per_batch=args.batch_size
    )


if __name__ == "__main__":
    main()

