"""
Handler para manejo centralizado de errores con tipado estricto.
"""
from typing import Callable, TypeVar, ParamSpec, Any
from functools import wraps
import logging
import traceback


P = ParamSpec('P')
R = TypeVar('R')


class ErrorHandler:
    """Handler para manejo centralizado de errores."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_execution_error(
        self, 
        func: Callable[P, R]
    ) -> Callable[P, R]:
        """Decorador para manejo de errores en ejecución."""
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(
                    f"Error in {func.__name__}: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}",
                    exc_info=True
                )
                raise
        
        return wrapper
    
    def handle_validation_error(
        self,
        func: Callable[P, R]
    ) -> Callable[P, R]:
        """Decorador para manejo específico de errores de validación."""
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f"Validation error in {func.__name__}: {str(e)}"
                )
                raise
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    
    def handle_resource_error(
        self,
        func: Callable[P, R]
    ) -> Callable[P, R]:
        """Decorador para manejo de errores de recursos."""
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except (MemoryError, OSError) as e:
                self.logger.critical(
                    f"Resource error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
            except Exception as e:
                # Si no es un error de recursos, lo dejamos pasar
                raise
        
        return wrapper