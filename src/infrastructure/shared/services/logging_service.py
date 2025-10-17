"""
Servicio de logging centralizado con tipado estricto.
"""

import logging
from typing import Optional
import sys


class LoggingService:
    """Servicio para manejo centralizado de logging."""

    def __init__(
        self, name: str, level: int = logging.INFO, log_file: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Evitar duplicados
        if not self.logger.handlers:
            self._setup_handlers(level, log_file)

    def _setup_handlers(self, level: int, log_file: Optional[str]) -> None:
        """Configura los handlers para el logger."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)

        # Handler para archivo si se especifica
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Obtiene el logger configurado."""
        return self.logger

    def set_level(self, level: int) -> None:
        """Establece el nivel de logging."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
