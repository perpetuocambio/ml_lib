"""
Interfaces para el Módulo 3: Optimización Numérica Avanzada

Define las interfaces comunes para optimizadores, resultados de optimización,
y configuraciones de optimización.
"""
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any
import numpy as np


class OptimizationResult:
    """
    Resultado de una operación de optimización.
    
    Attributes:
        x: Punto óptimo encontrado
        fun: Valor de la función en el punto óptimo
        success: Indica si la optimización fue exitosa
        message: Mensaje descriptivo sobre el resultado
        niter: Número de iteraciones realizadas
        nit: Alias para niter
        nfev: Número de evaluaciones de la función
        njev: Número de evaluaciones del jacobiano
        nhev: Número de evaluaciones del hessiano
        grad: Gradiente en el punto óptimo (si está disponible)
    """
    
    def __init__(self, x: np.ndarray, fun: float, success: bool = True, 
                 message: str = "Optimización exitosa", niter: int = 0, 
                 nfev: int = 0, njev: int = 0, nhev: int = 0, grad: Optional[np.ndarray] = None):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.niter = niter
        self.nit = niter  # Alias para compatibilidad
        self.nfev = nfev
        self.njev = njev
        self.nhev = nhev
        self.grad = grad


class Optimizer(Protocol):
    """
    Interfaz común para todos los optimizadores del módulo.
    """
    
    def optimize(self, func: callable, x0: np.ndarray, 
                 jac: Optional[callable] = None, 
                 hess: Optional[callable] = None,
                 **options) -> OptimizationResult:
        """
        Realiza la optimización de la función objetivo.
        
        Args:
            func: Función objetivo a minimizar
            x0: Punto inicial para la optimización
            jac: Gradiente de la función objetivo (opcional)
            hess: Hessiano de la función objetivo (opcional)
            **options: Opciones adicionales de optimización
            
        Returns:
            OptimizationResult: Resultado de la optimización
        """
        ...