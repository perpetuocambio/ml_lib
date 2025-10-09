"""
Modelos para el Módulo 3: Optimización Numérica Avanzada

Define modelos de datos para configuración de optimización y
parámetros de control de los algoritmos.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class OptimizationConfig:
    """
    Configuración general para algoritmos de optimización.
    
    Attributes:
        maxiter: Número máximo de iteraciones
        maxfev: Número máximo de evaluaciones de función
        tol: Tolerancia para la convergencia
        ftol: Tolerancia para la función
        xtol: Tolerancia para las variables
        gtol: Tolerancia para el gradiente
        eps: Pequeño valor para evitar divisiones por cero
        verbose: Nivel de detalle en la salida
    """
    maxiter: int = 1000
    maxfev: int = 15000
    tol: float = 1e-6
    ftol: float = 1e-9
    xtol: float = 1e-9
    gtol: float = 1e-6
    eps: float = 1e-8
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a un diccionario."""
        return {
            'maxiter': self.maxiter,
            'maxfev': self.maxfev,
            'tol': self.tol,
            'ftol': self.ftol,
            'xtol': self.xtol,
            'gtol': self.gtol,
            'eps': self.eps,
            'verbose': self.verbose
        }


@dataclass
class NewtonConfig(OptimizationConfig):
    """
    Configuración específica para el optimizador de Newton.
    
    Attributes:
        line_search: Habilitar búsqueda de línea
        trust_region: Habilitar región de confianza
        reg_param: Parámetro de regularización
    """
    line_search: bool = True
    trust_region: bool = False
    reg_param: float = 1e-4
    update_method: str = 'bfgs'  # 'bfgs', 'dfp', 'sr1'


@dataclass
class GradientDescentConfig(OptimizationConfig):
    """
    Configuración específica para el optimizador de descenso de gradiente.
    
    Attributes:
        learning_rate: Tasa de aprendizaje
        momentum: Factor de momentum
        adaptive: Usar tasa de aprendizaje adaptativa
        method: Método de descenso ('fixed', 'momentum', 'adagrad', 'adam')
    """
    learning_rate: float = 0.01
    momentum: float = 0.9
    adaptive: bool = True
    method: str = 'adam'  # 'fixed', 'momentum', 'adagrad', 'adam', 'rmsprop'


@dataclass
class LBFGSConfig(OptimizationConfig):
    """
    Configuración específica para el optimizador LBFGS.
    
    Attributes:
        m: Número de correcciones para aproximar el hessiano inverso
        line_search: Tipo de búsqueda de línea ('armijo', 'wolfe', 'strong_wolfe')
        maxcor: Máximo número de correcciones (alias para m)
    """
    m: int = 10
    line_search: str = 'wolfe'
    maxcor: int = 10