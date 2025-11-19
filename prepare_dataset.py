"""Скрипт для упаковки и подготовки датасета для обучения нейросети."""

import json
import os
import shutil
from pathlib import Path
from typing import Literal
import random


class DatasetPreparer:
    """Подготовка датасета для обучения нейросети."""
    
    def __init__(
        self,
        clean_dir: str = "./output/clean_background",
        noisy_dir: str = "./output/noisy_background",
        output_dir: str = "./dataset",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Args:
            clean_dir: Директория с чистыми изображениями
            noisy_dir: Директория с зашумленными изображениями
            output_dir: Выходная директория для датасета
            train_ratio: Доля данных для обучения
            val_ratio: Доля данных для валидации
            test_ratio: Доля данных для тестирования
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.output_dir = Path(output_dir)
        
        # Проверяем, что сумма долей = 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Сумма train_ratio, val_ratio и test_ratio должна быть равна 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def prepare_folder_structure(self) -> dict:
        """
        Создает структуру папок и распределяет изображения.
        
        Returns:
            Статистика по датасету
        """
        print("\n" + "=" * 70)
        print("ПОДГОТОВКА ДАТАСЕТА")
        print("=" * 70)
        
        # Получаем списки файлов
        clean_files = sorted(self.clean_dir.glob("*_clean.png"))
        noisy_files = sorted(self.noisy_dir.glob("*_noisy.png"))
        
        print(f"\nНайдено изображений:")
        print(f"  - Чистых: {len(clean_files)}")
        print(f"  - Зашумленных: {len(noisy_files)}")
        
        # Создаем пары файлов
        pairs = self._create_pairs(clean_files, noisy_files)
        print(f"\nСоздано пар: {len(pairs)}")
        
        if len(pairs) == 0:
            print("❌ Не найдено пар изображений!")
            return {}
        
        # Перемешиваем пары
        random.shuffle(pairs)
        
        # Разделяем на train/val/test
        splits = self._split_data(pairs)
        
        # Создаем структуру папок
        self._create_directories()
        
        # Копируем файлы
        stats = self._copy_files(splits)
        
        # Создаем метаданные
        self._create_metadata(stats, splits)
        
        print("\n" + "=" * 70)
        print("✅ ДАТАСЕТ ПОДГОТОВЛЕН!")
        print("=" * 70)
        self._print_stats(stats)
        
        return stats
    
    def _create_pairs(self, clean_files: list, noisy_files: list) -> list:
        """Создает пары соответствующих файлов."""
        pairs = []
        
        # Создаем словарь для быстрого поиска
        clean_dict = {f.stem.replace("_clean", ""): f for f in clean_files}
        noisy_dict = {f.stem.replace("_noisy", ""): f for f in noisy_files}
        
        # Находим пары
        for key in clean_dict:
            if key in noisy_dict:
                pairs.append({
                    "clean": clean_dict[key],
                    "noisy": noisy_dict[key],
                    "id": key
                })
        
        return pairs
    
    def _split_data(self, pairs: list) -> dict:
        """Разделяет данные на train/val/test."""
        total = len(pairs)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        return {
            "train": pairs[:train_size],
            "val": pairs[train_size:train_size + val_size],
            "test": pairs[train_size + val_size:]
        }
    
    def _create_directories(self):
        """Создает структуру директорий."""
        for split in ["train", "val", "test"]:
            for subdir in ["clean", "noisy"]:
                dir_path = self.output_dir / split / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def _copy_files(self, splits: dict) -> dict:
        """Копирует файлы в соответствующие директории."""
        stats = {"train": 0, "val": 0, "test": 0}
        
        for split_name, pairs in splits.items():
            print(f"\nКопирование {split_name}...")
            for pair in pairs:
                # Копируем clean
                clean_dest = self.output_dir / split_name / "clean" / pair["clean"].name
                shutil.copy2(pair["clean"], clean_dest)
                
                # Копируем noisy
                noisy_dest = self.output_dir / split_name / "noisy" / pair["noisy"].name
                shutil.copy2(pair["noisy"], noisy_dest)
                
                stats[split_name] += 1
        
        return stats
    
    def _create_metadata(self, stats: dict, splits: dict):
        """Создает файлы метаданных."""
        metadata = {
            "total_pairs": sum(stats.values()),
            "splits": stats,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "files": {}
        }
        
        for split_name, pairs in splits.items():
            metadata["files"][split_name] = [
                {
                    "id": pair["id"],
                    "clean": str(pair["clean"].name),
                    "noisy": str(pair["noisy"].name)
                }
                for pair in pairs
            ]
        
        # Сохраняем JSON
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Метаданные сохранены: {metadata_path}")
    
    def _print_stats(self, stats: dict):
        """Выводит статистику."""
        total = sum(stats.values())
        print(f"\n📊 Статистика датасета:")
        print(f"  Всего пар: {total}")
        print(f"  Train: {stats['train']} ({stats['train']/total*100:.1f}%)")
        print(f"  Val: {stats['val']} ({stats['val']/total*100:.1f}%)")
        print(f"  Test: {stats['test']} ({stats['test']/total*100:.1f}%)")
        print(f"\n📁 Датасет сохранен в: {self.output_dir.absolute()}")


def main():
    """Основная функция."""
    preparer = DatasetPreparer(
        clean_dir="./output/clean_background",
        noisy_dir="./output/noisy_background",
        output_dir="./dataset",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    preparer.prepare_folder_structure()


if __name__ == "__main__":
    main()

