"""
Handlers para el Módulo 3: Optimización Numérica Avanzada

Implementa lógica de manejo de convergencia, errores y monitoreo
de los algoritmos de optimización.
"""
from typing import Optional, Tuple
import numpy as np


class ConvergenceChecker:
    """
    Verificador de convergencia para algoritmos de optimización.
    """
    
    def __init__(self, config):
        self.config = config
    
    def check_convergence(self, x_old: np.ndarray, x_new: np.ndarray, 
                         f_old: float, f_new: float, 
                         grad: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Verifica si el algoritmo ha convergido según varios criterios.
        
        Args:
            x_old: Vector de variables en la iteración anterior
            x_new: Vector de variables en la iteración actual
            f_old: Valor de la función en la iteración anterior
            f_new: Valor de la función en la iteración actual
            grad: Gradiente en el punto actual (opcional)
            
        Returns:
            Tuple[bool, str]: (indicador de convergencia, mensaje)
        """
        # Criterio de tolerancia en x
        x_diff = np.linalg.norm(x_new - x_old)
        if x_diff < self.config.xtol:
            return True, f"Convergencia alcanzada en x (diff={x_diff:.2e})"
        
        # Criterio de tolerancia en f
        f_diff = abs(f_new - f_old)
        if f_diff < self.config.ftol:
            return True, f"Convergencia alcanzada en f (diff={f_diff:.2e})"
        
        # Criterio de tolerancia en gradiente
        if grad is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.config.gtol:
                return True, f"Convergencia alcanzada en gradiente (norm={grad_norm:.2e})"
        
        # Criterio de tolerancia general
        if abs(f_new) > self.config.eps:
            rel_improvement = abs(f_new - f_old) / abs(f_new)
            if rel_improvement < self.config.tol:
                return True, f"Convergencia alcanzada (mejora relativa={rel_improvement:.2e})"
        
        return False, "No converge"
    
    def check_stagnation(self, history: list, threshold: float = 1e-6) -> Tuple[bool, str]:
        """
        Verifica si la optimización se ha estancado.
        
        Args:
            history: Historial de valores de función
            threshold: Umbral para detectar estancamiento
            
        Returns:
            Tuple[bool, str]: (indicador de estancamiento, mensaje)
        """
        if len(history) < 10:
            return False, "No hay suficiente historial para detectar estancamiento"
        
        recent_values = history[-10:]
        improvement = max(recent_values) - min(recent_values)
        
        if improvement < threshold:
            return True, f"Posible estancamiento detectado (mejora={improvement:.2e})"
        
        return False, "No hay estancamiento"


class ErrorHandler:
    """
    Manejador de errores para algoritmos de optimización.
    """
    
    @staticmethod
    def handle_numerical_error(error: Exception, iteration: int) -> str:
        """
        Maneja errores numéricos durante la optimización.
        
        Args:
            error: Excepción capturada
            iteration: Iteración actual
            
        Returns:
            Mensaje de error detallado
        """
        error_type = type(error).__name__
        if error_type == "OverflowError":
            return f"Error de desbordamiento en iteración {iteration}: {str(error)}"
        elif error_type == "ZeroDivisionError":
            return f"División por cero en iteración {iteration}: {str(error)}"
        elif error_type == "ValueError":
            return f"Valor inválido en iteración {iteration}: {str(error)}"
        else:
            return f"Error numérico en iteración {iteration} ({error_type}): {str(error)}"
    
    @staticmethod
    def handle_convergence_failure(maxiter: int, current_iter: int) -> str:
        """
        Maneja el caso de fallo en convergencia.
        
        Args:
            maxiter: Número máximo de iteraciones permitidas
            current_iter: Iteración actual
            
        Returns:
            Mensaje de error por fallo de convergencia
        """
        return f"No se alcanzó la convergencia en {maxiter} iteraciones. Última iteración: {current_iter}"