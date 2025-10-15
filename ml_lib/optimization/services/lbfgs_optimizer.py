"""
Servicio de optimización LBFGS para el Módulo 3.

Implementa el algoritmo Limited-memory BFGS para optimización
de funciones con gran cantidad de variables.
"""

import numpy as np
from typing import Optional, Callable
from ..interfaces.optimization_interfaces import Optimizer, OptimizationResult
from ...optimization.models.optimization_models import LBFGSConfig
from ..handlers.optimization_handlers import ConvergenceChecker, ErrorHandler


class LBFGSOptimizer(Optimizer):
    """
    Optimizador LBFGS (Limited-memory BFGS) para problemas de optimización.

    Implementa el algoritmo LBFGS que aproxima el hessiano usando
    información de iteraciones previas, ideal para problemas de alta dimensionalidad.
    """

    def __init__(self, config: Optional[LBFGSConfig] = None):
        self.config = config or LBFGSConfig()
        self.convergence_checker = ConvergenceChecker(self.config)
        self.error_handler = ErrorHandler()

    def optimize(
        self,
        func: Callable,
        x0: np.ndarray,
        jac: Optional[Callable] = None,
        hess: Optional[Callable] = None,
        **options,
    ) -> OptimizationResult:
        """
        Realiza la optimización usando el algoritmo LBFGS.

        Args:
            func: Función objetivo a minimizar
            x0: Punto inicial para la optimización
            jac: Gradiente de la función objetivo (opcional)
            hess: Hessiano de la función objetivo (opcional)
            **options: Opciones adicionales de optimización

        Returns:
            OptimizationResult: Resultado de la optimización
        """
        # Actualizar configuración con opciones
        for key, value in options.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        x = np.array(x0, dtype=float)
        n = len(x)
        nfev = 0
        njev = 0

        # Inicializar historia de BFGS (almacenamiento limitado)
        s_list = []  # Historia de cambios en x
        y_list = []  # Historia de cambios en gradientes
        rho_list = []  # Historia de 1/(y^T * s)

        # Obtener gradiente inicial
        if jac is not None:
            grad = jac(x)
            njev += 1
        else:
            grad = self._approximate_gradient(func, x)
            nfev += n * 2  # Aproximación del gradiente

        f_old = func(x)
        nfev += 1

        success = False
        message = "Convergencia no alcanzada"

        for iteration in range(self.config.maxiter):
            try:
                # Calcular dirección de búsqueda usando el algoritmo two-loop
                p = self._two_loop_recursion(grad, s_list, y_list, rho_list)

                # Invertir dirección para minimización
                p = -p

                # Aplicar búsqueda de línea
                alpha = self._line_search(func, x, p, grad, jac)

                # Almacenar la actualización
                s = alpha * p
                x_new = x + s

                # Calcular nuevo gradiente
                if jac is not None:
                    grad_new = jac(x_new)
                    njev += 1
                else:
                    grad_new = self._approximate_gradient(func, x_new)
                    nfev += n * 2

                # Calcular cambio en el gradiente
                y = grad_new - grad

                # Calcular rho
                rho = 1.0 / (y @ s)

                # Actualizar listas de historia (mantener solo m últimas)
                s_list.append(s)
                y_list.append(y)
                rho_list.append(rho)

                # Mantener solo los m vectores más recientes
                if len(s_list) > self.config.m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

                # Evaluar la función en el nuevo punto
                f_new = func(x_new)
                nfev += 1

                # Verificar convergencia
                converged, conv_msg = self.convergence_checker.check_convergence(
                    x, x_new, f_old, f_new, grad_new
                )

                if self.config.verbose and iteration % 10 == 0:
                    print(
                        f"Iteración {iteration}: f(x) = {f_new:.6e}, ||grad|| = {np.linalg.norm(grad_new):.6e}"
                    )

                if converged:
                    success = True
                    message = conv_msg
                    x = x_new
                    grad = grad_new
                    f_old = f_new
                    break

                # Actualizar variables para la siguiente iteración
                x = x_new
                grad = grad_new
                f_old = f_new

            except Exception as e:
                message = self.error_handler.handle_numerical_error(e, iteration)
                break

        else:
            # Se alcanzó el número máximo de iteraciones
            message = self.error_handler.handle_convergence_failure(
                self.config.maxiter, iteration
            )

        return OptimizationResult(
            x=x,
            fun=f_new if "f_new" in locals() else func(x),
            success=success,
            message=message,
            niter=iteration + 1,
            nfev=nfev,
            njev=njev,
            grad=grad,
        )

    def _two_loop_recursion(
        self, grad: np.ndarray, s_list: list, y_list: list, rho_list: list
    ) -> np.ndarray:
        """
        Implementa el algoritmo two-loop para calcular H^(-1) * grad.
        """
        q = grad.copy()
        alphas = []

        # Paso hacia atrás (backward loop)
        for i in range(len(s_list) - 1, -1, -1):
            alpha = rho_list[i] * (s_list[i] @ q)
            q = q - alpha * y_list[i]
            alphas.append(alpha)

        # Calcular H0
        if s_list:
            # Aproximar H0 como escalar basado en la última actualización
            s_last = s_list[-1]
            y_last = y_list[-1]
            H0 = (s_last @ y_last) / (y_last @ y_last)
        else:
            H0 = 1.0

        r = H0 * q

        # Paso hacia adelante (forward loop)
        for i in range(len(s_list)):
            beta = rho_list[i] * (y_list[i] @ r)
            r = r + s_list[i] * (alphas[len(s_list) - 1 - i] - beta)

        return r

    def _approximate_gradient(self, func: Callable, x: np.ndarray) -> np.ndarray:
        """
        Aproxima el gradiente usando diferencias finitas.
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            h = self.config.eps * (1 + abs(x[i]))
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
        return grad

    def _line_search(
        self,
        func: Callable,
        x: np.ndarray,
        p: np.ndarray,
        grad: np.ndarray,
        jac: Optional[Callable] = None,
        c1: float = 1e-4,
        c2: float = 0.9,
    ) -> float:
        """
        Implementa búsqueda de línea Wolfe para LBFGS.
        """
        alpha = 1.0  # Tamaño de paso inicial
        max_ls = 10  # Número máximo de iteraciones de búsqueda de línea
        f0 = func(x)
        grad_p = grad @ p  # Producto punto

        for i in range(max_ls):
            x_new = x + alpha * p
            f_new = func(x_new)

            # Verificar condición de Armijo
            if f_new <= f0 + c1 * alpha * grad_p:
                # Calcular nuevo gradiente para verificar condición de curvatura
                if jac is not None:
                    grad_new = jac(x_new)
                else:
                    grad_new = self._approximate_gradient(func, x_new)

                if grad_new @ p >= c2 * grad_p:
                    # Ambas condiciones de Wolfe satisfechas
                    return alpha
                else:
                    # Solo condición de Armijo satisfecha, reducir alpha
                    alpha *= 0.5
            else:
                # Condición de Armijo no satisfecha, reducir alpha
                alpha *= 0.5

        # Si no se encuentra un paso adecuado, devolver el mejor encontrado
        return alpha
