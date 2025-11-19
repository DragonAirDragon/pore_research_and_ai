"""Модуль для загрузки конфигурации из JSON-файла."""

import json
from typing import Any


class ConfigLoader:
    """Загружает и парсит конфигурационные файлы в формате JSON.

    Attributes:
        config_path: Путь к файлу конфигурации.
    """

    def __init__(self, config_path: str):
        """Инициализирует загрузчик конфигурации.

        Args:
            config_path: Путь к файлу конфигурации.
        """
        self.config_path = config_path

    def load(self) -> dict[str, Any]:
        """Загружает и возвращает конфигурацию из файла.

        Метод считывает JSON-файл по указанному пути и возвращает его
        содержимое в виде словаря Python.

        Returns:
            Словарь с данными конфигурации.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            json.JSONDecodeError: Если файл содержит невалидный JSON.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Файл конфигурации '{self.config_path}' не найден."
            ) from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Ошибка парсинга JSON в файле '{self.config_path}': {e.msg}",
                e.doc,
                e.pos,
            ) from e
