"""Модуль для обработки изображений, включая добавление шума."""

from typing import Any, Sequence

import cv2
import numpy as np

_BACKGROUND_COLOR = 255
_PORE_COLOR = 0
_BINARY_THRESHOLD = 127


class ImageProcessor:
    """Обрабатывает изображения, добавляя различные виды процедурного шума.

    Класс инкапсулирует логику для добавления фонового и порового шума
    на основе конфигурации, предоставленной при инициализации.

    Attributes:
        _min_gray: Минимальное значение серого для фонового шума.
        _max_gray: Максимальное значение серого для фонового шума.
        _noise_intensity: Интенсивность текстурного шума для фона.
        _is_pore_noise_enabled: Включен ли шум для пор.
        _pore_min_val: Минимальное значение серого для шума пор.
        _pore_max_val: Максимальное значение серого для шума пор.
        _is_pore_texture_enabled: Включена ли текстура для шума пор.
    """

    _BACKGROUND_NOISE_SCALES: Sequence[int] = (100, 40, 10)
    _BACKGROUND_NOISE_WEIGHTS: Sequence[float] = (0.5, 0.3, 0.2)
    _PORE_NOISE_SCALES: Sequence[int] = (50, 20, 5)
    _PORE_NOISE_WEIGHTS: Sequence[float] = (0.5, 0.3, 0.2)

    def __init__(self, config: dict[str, Any]):
        """Инициализирует обработчик изображений с заданной конфигурацией.

        Args:
            config: Словарь конфигурации. Ожидаемая структура:
                {
                    "noise_settings": {
                        "min_gray_value": int,
                        "max_gray_value": int,
                        "noise_intensity": float,
                        "pore_noise": {
                            "enabled": bool,
                            "min_value": int,
                            "max_value": int,
                            "texture_enabled": bool
                        }
                    }
                }
        """
        noise_settings = config.get("noise_settings", {})
        pore_noise_settings = noise_settings.get("pore_noise", {})

        self._min_gray = noise_settings.get("min_gray_value", 200)
        self._max_gray = noise_settings.get("max_gray_value", 250)
        self._noise_intensity = noise_settings.get("noise_intensity", 0.5)

        self._is_pore_noise_enabled = pore_noise_settings.get("enabled", False)
        self._pore_min_val = pore_noise_settings.get("min_value", 0)
        self._pore_max_val = pore_noise_settings.get("max_value", 15)
        self._is_pore_texture_enabled = pore_noise_settings.get(
            "texture_enabled", False
        )

    def add_complete_noise(self, image: np.ndarray) -> np.ndarray:
        """Применяет к изображению все сконфигурированные виды шума.

        Сначала добавляется шум к порам, затем к фону.

        Args:
            image: Входное изображение в виде NumPy массива.

        Returns:
            Изображение с добавленным шумом.
        """
        image_with_pore_noise = self.add_pore_noise(image)
        image_with_all_noise = self.add_background_noise(image_with_pore_noise)
        return image_with_all_noise

    def add_background_noise(self, image: np.ndarray) -> np.ndarray:
        """Добавляет шум к фону изображения.

        Args:
            image: Входное изображение.

        Returns:
            Изображение с зашумленным фоном.
        """
        noisy_image = image.copy()
        background_mask = image == _BACKGROUND_COLOR

        random_noise_layer = np.random.randint(
            self._min_gray, self._max_gray + 1, size=image.shape, dtype=np.uint8
        )

        if self._noise_intensity > 0:
            texture_noise = self._generate_fractal_noise(
                image.shape,
                self._BACKGROUND_NOISE_SCALES,
                self._BACKGROUND_NOISE_WEIGHTS,
                self._min_gray,
                self._max_gray,
            )
            combined_noise_layer = cv2.addWeighted(
                random_noise_layer,
                1 - self._noise_intensity,
                texture_noise,
                self._noise_intensity,
                0,
            )
        else:
            combined_noise_layer = random_noise_layer

        noisy_image[background_mask] = combined_noise_layer[background_mask]
        return noisy_image

    def add_pore_noise(self, image: np.ndarray) -> np.ndarray:
        """Добавляет шум к порам изображения.

        Args:
            image: Входное изображение.

        Returns:
            Изображение с зашумленными порами или исходное изображение,
            если шум для пор отключен в конфигурации.
        """
        if not self._is_pore_noise_enabled:
            return image

        noisy_image = image.copy()
        pore_mask = image == _PORE_COLOR

        if self._is_pore_texture_enabled:
            pore_noise = self._generate_fractal_noise(
                image.shape,
                self._PORE_NOISE_SCALES,
                self._PORE_NOISE_WEIGHTS,
                self._pore_min_val,
                self._pore_max_val,
            )
        else:
            pore_noise = np.random.randint(
                self._pore_min_val,
                self._pore_max_val + 1,
                size=image.shape,
                dtype=np.uint8,
            )

        noisy_image[pore_mask] = pore_noise[pore_mask]
        return noisy_image

    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Применяет к изображению размытие по Гауссу и бинаризацию.

        Args:
            image: Входное изображение.

        Returns:
            Обработанное бинарное изображение.
        """
        kernel_size = 3
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.5)
        _, binary_image = cv2.threshold(
            blurred, _BINARY_THRESHOLD, _BACKGROUND_COLOR, cv2.THRESH_BINARY
        )
        return binary_image

    def _generate_fractal_noise(
        self,
        shape: tuple[int, ...],
        scales: Sequence[int],
        weights: Sequence[float],
        min_val: int,
        max_val: int,
    ) -> np.ndarray:
        """Генерирует фрактальный шум.

        Args:
            shape: Форма результирующего массива шума.
            scales: Масштабы для генерации шума разной частоты.
            weights: Веса для каждого масштаба.
            min_val: Минимальное значение для нормализации шума.
            max_val: Максимальное значение для нормализации шума.

        Returns:
            Массив с фрактальным шумом, нормализованный в диапазоне
            [min_val, max_val].
        """
        height, width = shape
        texture = np.zeros(shape, dtype=np.float32)

        for scale, weight in zip(scales, weights):
            rand_h = height // scale + 1
            rand_w = width // scale + 1
            random_field = np.random.randn(rand_h, rand_w)
            resized_noise = cv2.resize(
                random_field, (width, height), interpolation=cv2.INTER_CUBIC
            )
            texture += resized_noise * weight

        texture += np.random.randn(height, width) * 0.05

        tex_min, tex_max = texture.min(), texture.max()
        if np.isclose(tex_max, tex_min):
            return np.full(shape, (min_val + max_val) // 2, dtype=np.uint8)

        normalized_texture = (texture - tex_min) / (tex_max - tex_min)
        normalized_texture = normalized_texture * (max_val - min_val) + min_val
        return normalized_texture.astype(np.uint8)

    def apply_ceramic_texture(self, binary_mask: np.ndarray) -> np.ndarray:
        """Applies realistic ceramic texture to a binary mask.
        
        Converts a binary mask (white bg, black pores) into a realistic
        ceramic image (gray textured bg, dark pores with depth).
        
        Args:
            binary_mask: Binary image where 0=pore, 255=background.
            
        Returns:
            Realistic ceramic-style image.
        """
        height, width = binary_mask.shape
        
        # 1. Create gray textured background (concrete/ceramic-like)
        # Base gray level around 180-220
        base_gray = np.random.randint(180, 220, size=binary_mask.shape, dtype=np.uint8)
        
        # Add fractal texture for surface roughness
        surface_texture = self._generate_fractal_noise(
            binary_mask.shape,
            scales=(80, 30, 10),
            weights=(0.6, 0.3, 0.1),
            min_val=160,
            max_val=230
        )
        
        # Blend base gray with texture
        background = cv2.addWeighted(base_gray, 0.3, surface_texture, 0.7, 0)
        
        # 2. Create dark pores with depth
        pore_mask = binary_mask == _PORE_COLOR
        
        # Base pore color: very dark (10-40)
        pore_base = np.random.randint(10, 40, size=binary_mask.shape, dtype=np.uint8)
        
        # Add slight internal texture/noise to pores
        pore_texture = self._generate_fractal_noise(
            binary_mask.shape,
            scales=(40, 15, 5),
            weights=(0.5, 0.3, 0.2),
            min_val=5,
            max_val=50
        )
        
        # Create depth gradient (darker in center, slightly lighter at edges)
        # Use distance transform to create depth
        pore_uint8 = (pore_mask * 255).astype(np.uint8)
        dist_transform = cv2.distanceTransform(pore_uint8, cv2.DIST_L2, 5)
        
        if dist_transform.max() > 0:
            # Normalize distance map
            depth_map = dist_transform / (dist_transform.max() + 1e-6)
            # Invert for shadow (deeper = darker)
            depth_map = 1.0 - depth_map
            # Apply to pore texture
            pore_shaded = (pore_texture * depth_map).astype(np.uint8)
        else:
            pore_shaded = pore_texture
        
        # 3. Combine background and pores
        result = background.copy()
        result[pore_mask] = pore_shaded[pore_mask]
        
        # 4. Add final surface grain (very subtle uniform noise)
        grain = np.random.randint(-5, 5, size=binary_mask.shape, dtype=np.int16)
        result = np.clip(result.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        return result

