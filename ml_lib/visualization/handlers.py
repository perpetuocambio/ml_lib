"""
Handlers para componentes de visualización en ml_lib
"""
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
import logging
import numpy as np
import matplotlib.pyplot as plt


P = ParamSpec('P')
R = TypeVar('R')


class VisualizationErrorHandler:
    """Handler para manejo de errores en visualización."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_visualization_error(
        self, 
        func: Callable[P, R]
    ) -> Callable[P, R]:
        """Decorador para manejo de errores en operaciones de visualización."""
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError) as e:
                self.logger.error(
                    f"Error de validación en {func.__name__}: {str(e)}"
                )
                raise
            except (ImportError, ModuleNotFoundError) as e:
                self.logger.error(
                    f"Error de dependencias en {func.__name__}: {str(e)}"
                )
                raise
            except Exception as e:
                self.logger.error(
                    f"Error inesperado en {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper


class VisualizationConfigHandler:
    """Handler para manejo de configuración de visualización."""
    
    @staticmethod
    def validate_data_shape(data: np.ndarray, expected_dims: int = None) -> bool:
        """Valida la forma de los datos para visualización."""
        if expected_dims and data.ndim != expected_dims:
            raise ValueError(f"Se esperaban {expected_dims} dimensiones, se recibieron {data.ndim}")
        return True
    
    @staticmethod
    def sanitize_plot_params(params: dict) -> dict:
        """Sanitiza los parámetros para evitar problemas en los gráficos."""
        sanitized = params.copy()
        
        # Asegurar valores razonables
        if 'alpha' in sanitized:
            sanitized['alpha'] = max(0.0, min(1.0, sanitized['alpha']))
        
        if 'figsize' in sanitized:
            sanitized['figsize'] = tuple(max(1, x) for x in sanitized['figsize'])
        
        return sanitized


class ImageExportHandler:
    """Handler para exportación de imágenes."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def save_figure(
        self,
        figure: plt.Figure,
        filepath: str,
        dpi: int = 300,
        format: str = 'png',
        bbox_inches: str = 'tight'
    ) -> None:
        """Guarda una figura con los parámetros especificados."""
        try:
            figure.savefig(
                filepath,
                dpi=dpi,
                format=format,
                bbox_inches=bbox_inches
            )
            self.logger.info(f"Figura guardada exitosamente: {filepath}")
        except Exception as e:
            self.logger.error(f"Error al guardar figura {filepath}: {str(e)}")
            raise