"""
Módulo 3: Optimización Numérica Avanzada

Este módulo implementa optimizadores numéricos avanzados para problemas de
minimización y maximización de funciones mediante métodos de primer y segundo orden.
"""
from .interfaces.optimization_interfaces import Optimizer, OptimizationResult
from .services.newton_optimizer import NewtonOptimizer
from .services.gradient_descent_optimizer import GradientDescentOptimizer
from .services.lbfgs_optimizer import LBFGSOptimizer

__all__ = [
    'Optimizer',
    'OptimizationResult',
    'NewtonOptimizer',
    'GradientDescentOptimizer',
    'LBFGSOptimizer'
]