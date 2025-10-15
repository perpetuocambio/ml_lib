"""
Servicio de optimización de Newton para el Módulo 3.

Implementa el algoritmo de Newton con diferentes estrategias para
manejar el hessiano y mejorar la estabilidad.
"""

import numpy as np
from typing import Optional, Callable
from ..interfaces.optimization_interfaces import Optimizer, OptimizationResult
from ...optimization.models.optimization_models import NewtonConfig
from ..handlers.optimization_handlers import ConvergenceChecker, ErrorHandler


class NewtonOptimizer(Optimizer):
    """
    Optimizador de Newton para problemas de optimización.

    Implementa el método de Newton con posibles extensiones como búsqueda de línea
    y región de confianza para mejorar la convergencia y estabilidad.
    """

    def __init__(self, config: Optional[NewtonConfig] = None):
        self.config = config or NewtonConfig()
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
        Realiza la optimización usando el método de Newton.

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
        nfev = 0
        njev = 0
        nhev = 0

        # Obtener el gradiente y hessiano iniciales
        if jac is not None:
            grad = jac(x)
            njev += 1
        else:
            grad = self._approximate_gradient(func, x)
            nfev += len(x) * 2

        if hess is not None:
            hess_matrix = hess(x)
            nhev += 1
        else:
            hess_matrix = self._approximate_hessian(func, x, grad)
            nfev += len(x) * len(x) * 2  # Aproximación del hessiano

        f_old = func(x)
        nfev += 1
        success = False
        message = "Convergencia no alcanzada"

        for iteration in range(self.config.maxiter):
            try:
                # Aplicar regularización si el hessiano no es definido positivo
                hess_reg = self._regularize_hessian(hess_matrix)

                # Calcular dirección de Newton
                try:
                    # Resolver H * p = -grad para obtener la dirección de paso
                    p = np.linalg.solve(hess_reg, -grad)
                except np.linalg.LinAlgError:
                    # Si el sistema no se puede resolver, usar pseudo-inversa
                    p = np.linalg.pinv(hess_reg) @ (-grad)

                # Aplicar búsqueda de línea si está habilitada
                if self.config.line_search:
                    alpha = self._line_search(func, x, p, grad)
                else:
                    alpha = 1.0

                # Actualizar posición
                x_new = x + alpha * p

                # Evaluar la función en el nuevo punto
                f_new = func(x_new)
                nfev += 1

                # Calcular gradiente y hessiano en el nuevo punto
                if jac is not None:
                    grad_new = jac(x_new)
                    njev += 1
                else:
                    grad_new = self._approximate_gradient(func, x_new)
                    nfev += len(x) * 2

                if hess is not None:
                    hess_matrix_new = hess(x_new)
                    nhev += 1
                else:
                    hess_matrix_new = self._approximate_hessian(func, x_new, grad_new)
                    nfev += len(x) * len(x) * 2

                # Verificar convergencia
                converged, conv_msg = self.convergence_checker.check_convergence(
                    x, x_new, f_old, f_new, grad_new
                )

                if self.config.verbose and iteration % 5 == 0:
                    print(
                        f"Iteración {iteration}: f(x) = {f_new:.6e}, ||grad|| = {np.linalg.norm(grad_new):.6e}"
                    )

                if converged:
                    success = True
                    message = conv_msg
                    x = x_new
                    grad = grad_new
                    hess_matrix = hess_matrix_new
                    f_old = f_new
                    break

                # Actualizar variables para la siguiente iteración
                x = x_new
                grad = grad_new
                hess_matrix = hess_matrix_new
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
            nhev=nhev,
            grad=grad,
        )

    def _regularize_hessian(self, hess_matrix: np.ndarray) -> np.ndarray:
        """
        Regulariza el hessiano para asegurar que sea definido positivo.
        """
        # Obtener autovalores y autovectores
        eigenvals, eigenvecs = np.linalg.eigh(hess_matrix)

        # Si hay autovalores negativos, regularizar
        min_eigenval = np.min(eigenvals)
        if min_eigenval < 0:
            # Añadir un múltiplo de la identidad para hacerlo definido positivo
            reg_param = max(self.config.reg_param, -min_eigenval + 1e-8)
            hess_reg = hess_matrix + reg_param * np.eye(hess_matrix.shape[0])
        else:
            hess_reg = hess_matrix

        return hess_reg

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

    def _approximate_hessian(
        self, func: Callable, x: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """
        Aproxima el hessiano usando diferencias finitas.
        """
        n = len(x)
        hess = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                h_i = self.config.eps * (1 + abs(x[i]))
                h_j = self.config.eps * (1 + abs(x[j]))

                if i == j:
                    # Elemento diagonal
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += h_i
                    x_minus[i] -= h_i
                    grad_plus = self._approximate_gradient(func, x_plus)
                    grad_minus = self._approximate_gradient(func, x_minus)
                    hess[i, j] = (grad_plus[i] - grad_minus[i]) / (2 * h_i)
                else:
                    # Elementos fuera de diagonal
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()

                    x_pp[i] += h_i
                    x_pp[j] += h_j
                    x_pm[i] += h_i
                    x_pm[j] -= h_j
                    x_mp[i] -= h_i
                    x_mp[j] += h_j
                    x_mm[i] -= h_i
                    x_mm[j] -= h_j

                    f_pp = func(x_pp)
                    f_pm = func(x_pm)
                    f_mp = func(x_mp)
                    f_mm = func(x_mm)

                    hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_i * h_j)
                    hess[j, i] = hess[i, j]  # Simetría

        return hess

    def _line_search(
        self,
        func: Callable,
        x: np.ndarray,
        p: np.ndarray,
        grad: np.ndarray,
        c1: float = 1e-4,
        c2: float = 0.9,
    ) -> float:
        """
        Implementa búsqueda de línea Wolfe para encontrar un paso adecuado.
        """
        alpha = 1.0  # Tamaño de paso inicial
        max_ls = 10  # Número máximo de iteraciones de búsqueda de línea

        f0 = func(x)
        grad_p = grad @ p  # Producto punto

        # Verificar la condición de Armijo
        for i in range(max_ls):
            x_new = x + alpha * p
            f_new = func(x_new)

            if f_new <= f0 + c1 * alpha * grad_p:
                # Condición de Armijo satisfecha, verificar condición de curvatura
                grad_new = self._approximate_gradient(func, x_new)
                if grad_new @ p >= c2 * grad_p:
                    # Ambas condiciones de Wolfe satisfechas
                    return alpha
                else:
                    # La condición de curvatura no se satisface, reducir alpha
                    alpha *= 0.5
            else:
                # Condición de Armijo no satisfecha, reducir alpha
                alpha *= 0.5

        return alpha
