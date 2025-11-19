"""Основной модуль для генерации набора данных изображений пор."""

import os
import random
import string

import cv2
from tqdm import tqdm

from src.config_loader import ConfigLoader
from src.image_processor import ImageProcessor
from src.pore_generator import PoreGenerator

_DEFAULT_CONFIG_PATH = "config.json"


def _generate_run_id(length: int = 6) -> str:
    """Генерирует случайный буквенно-цифровой ID для сессии генерации."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


class PoreImageGenerator:
    """Оркестрирует процесс создания датасета изображений пор.

    Загружает конфигурацию, инициализирует компоненты и генерирует пары
    изображений: чистое (маска) и финальное зашумленное.

    Attributes:
        config: Словарь с полной конфигурацией.
        pore_generator: Экземпляр генератора базовых изображений пор.
        image_processor: Экземпляр обработчика изображений.
        _clean_dir: Путь к директории для чистых изображений.
        _noisy_dir: Путь к директории для зашумленных изображений.
        _total_images: Общее количество изображений для генерации.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH):
        """Инициализирует генератор.

        Args:
            config_path: Путь к файлу конфигурации.
        """
        loader = ConfigLoader(config_path)
        self.config = loader.load()

        self.pore_generator = PoreGenerator(self.config)
        self.image_processor = ImageProcessor(self.config)

        output_settings = self.config.get("output_settings", {})
        image_settings = self.config.get("image_settings", {})

        self._clean_dir = output_settings.get("clean_dir", "./output/clean_background")
        self._noisy_dir = output_settings.get("noisy_dir", "./output/noisy_background")
        self._total_images = image_settings.get("total_images", 100)

        self._create_output_directories()

    def _create_output_directories(self) -> None:
        """Создает выходные директории, если они не существуют."""
        os.makedirs(self._clean_dir, exist_ok=True)
        os.makedirs(self._noisy_dir, exist_ok=True)

    def generate_images(self) -> None:
        """Запускает основной цикл генерации и сохранения изображений."""
        run_id = _generate_run_id()
        print(
            f"Starting generation of {self._total_images} image pairs "
            f"(Run ID: {run_id})..."
        )

        for i in tqdm(range(self._total_images), desc="Generating Images"):
            self._generate_and_save_pair(run_id, i)

    def _generate_and_save_pair(self, run_id: str, image_index: int) -> None:
        """Генерирует, обрабатывает и сохраняет одну пару изображений."""
        clean_image = self.pore_generator.generate_image()

        noisy_image = self.image_processor.add_complete_noise(clean_image.copy())

        base_filename = f"{run_id}_{image_index:04d}"
        clean_path = os.path.join(self._clean_dir, f"{base_filename}_clean.png")
        noisy_path = os.path.join(self._noisy_dir, f"{base_filename}_noisy.png")

        cv2.imwrite(clean_path, clean_image)
        cv2.imwrite(noisy_path, noisy_image)


def main() -> None:
    """Основная точка входа для запуска генератора изображений."""
    generator = PoreImageGenerator(_DEFAULT_CONFIG_PATH)
    generator.generate_images()


if __name__ == "__main__":
    main()
