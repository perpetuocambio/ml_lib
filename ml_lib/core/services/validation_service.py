"""
Servicio para validación de datos y modelos con tipado estricto.
"""
from typing import Any, Dict, Set
import numpy as np
import logging


class ValidationService:
    """Servicio para validación de datos y modelos."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate_input_shape(
        self, 
        X: np.ndarray, 
        expected_dims: int,
        context: str = ""
    ) -> None:
        """Valida las dimensiones de entrada."""
        if X.ndim != expected_dims:
            msg = (
                f"Expected {expected_dims}D array, got {X.ndim}D "
                f"in {context or 'validation'}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
    
    def validate_same_length(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        context: str = ""
    ) -> None:
        """Valida que X e y tengan la misma longitud."""
        if len(X) != len(y):
            msg = (
                f"X and y must have same length: {len(X)} != {len(y)} "
                f"in {context or 'validation'}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
    
    def validate_params(
        self, 
        params: Dict[str, Any], 
        allowed_params: Set[str],
        context: str = ""
    ) -> None:
        """Valida que los parámetros sean permitidos."""
        invalid = set(params.keys()) - allowed_params
        if invalid:
            msg = f"Invalid parameters: {invalid} in {context or 'validation'}"
            self.logger.error(msg)
            raise ValueError(msg)
    
    def validate_not_nan_inf(
        self, 
        array: np.ndarray, 
        context: str = ""
    ) -> None:
        """Valida que el array no contenga NaN o inf."""
        if np.any(np.isnan(array)):
            msg = f"Array contains NaN values in {context or 'validation'}"
            self.logger.error(msg)
            raise ValueError(msg)
        
        if np.any(np.isinf(array)):
            msg = f"Array contains infinite values in {context or 'validation'}"
            self.logger.error(msg)
            raise ValueError(msg)
    
    def validate_positive(
        self, 
        value: float, 
        param_name: str,
        context: str = ""
    ) -> None:
        """Valida que un valor sea positivo."""
        if value <= 0:
            msg = (
                f"Parameter {param_name} must be positive, got {value} "
                f"in {context or 'validation'}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
    
    def validate_between(
        self, 
        value: float, 
        min_val: float, 
        max_val: float, 
        param_name: str,
        context: str = ""
    ) -> None:
        """Valida que un valor esté entre un rango."""
        if not (min_val <= value <= max_val):
            msg = (
                f"Parameter {param_name} must be between {min_val} and {max_val}, "
                f"got {value} in {context or 'validation'}"
            )
            self.logger.error(msg)
            raise ValueError(msg)