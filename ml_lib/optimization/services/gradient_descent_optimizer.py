"""
Servicio de optimización por descenso de gradiente para el Módulo 3.

Implementa diferentes variantes del algoritmo de descenso de gradiente.
"""

import numpy as np
from typing import Optional, Callable
from ..interfaces.optimization_interfaces import Optimizer, OptimizationResult
from ...optimization.models.optimization_models import GradientDescentConfig
from ..handlers.optimization_handlers import ConvergenceChecker, ErrorHandler


class GradientDescentOptimizer(Optimizer):
    """
    Optimizador de descenso de gradiente con diferentes variantes.

    Implementa descenso de gradiente simple, con momentum, Adam y otros métodos adaptativos.
    """

    def __init__(self, config: Optional[GradientDescentConfig] = None):
        self.config = config or GradientDescentConfig()
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
        Realiza la optimización usando descenso de gradiente.

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

        # Inicializar variables según el método
        if self.config.method == "adam":
            m = np.zeros_like(x)  # Momento de primer orden
            v = np.zeros_like(x)  # Momento de segundo orden
            t = 0  # Contador de iteraciones
        elif self.config.method == "momentum":
            velocity = np.zeros_like(x)
        elif self.config.method == "adagrad":
            g_squared = np.zeros_like(x)
        elif self.config.method == "rmsprop":
            g_squared = np.zeros_like(x)

        f_old = func(x)
        nfev += 1
        grad = None
        success = False
        message = "Convergencia no alcanzada"

        for iteration in range(self.config.maxiter):
            try:
                # Calcular gradiente
                if jac is not None:
                    grad = jac(x)
                    njev += 1
                else:
                    # Aproximar gradiente numéricamente
                    grad = self._approximate_gradient(func, x)
                    nfev += len(x) * 2  # Evaluaciones para gradiente numérico

                # Aplicar el método de descenso de gradiente
                if self.config.method == "adam":
                    x, m, v, t = self._adam_update(x, grad, m, v, t)
                elif self.config.method == "momentum":
                    x, velocity = self._momentum_update(x, grad, velocity)
                elif self.config.method == "adagrad":
                    x, g_squared = self._adagrad_update(x, grad, g_squared)
                elif self.config.method == "rmsprop":
                    x, g_squared = self._rmsprop_update(x, grad, g_squared)
                else:  # 'fixed' - descenso de gradiente simple
                    x = x - self.config.learning_rate * grad

                # Evaluar la función en el nuevo punto
                f_new = func(x)
                nfev += 1

                # Verificar convergencia
                converged, conv_msg = self.convergence_checker.check_convergence(
                    x - self.config.learning_rate * grad if grad is not None else x0,
                    x,
                    f_old,
                    f_new,
                    grad,
                )

                if self.config.verbose and iteration % 10 == 0:
                    print(
                        f"Iteración {iteration}: f(x) = {f_new:.6e}, ||grad|| = {np.linalg.norm(grad):.6e}"
                    )

                if converged:
                    success = True
                    message = conv_msg
                    break

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

    def _adam_update(
        self, x: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, t: int
    ) -> tuple:
        """
        Actualización Adam para descenso de gradiente.
        """
        t += 1
        beta1, beta2 = 0.9, 0.999
        eps = self.config.eps

        # Actualizar momentos
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        # Corrección de sesgo
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Actualizar parámetros
        x_new = x - self.config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        return x_new, m, v, t

    def _momentum_update(
        self, x: np.ndarray, grad: np.ndarray, velocity: np.ndarray
    ) -> tuple:
        """
        Actualización con momentum para descenso de gradiente.
        """
        velocity = self.config.momentum * velocity - self.config.learning_rate * grad
        x_new = x + velocity
        return x_new, velocity

    def _adagrad_update(
        self, x: np.ndarray, grad: np.ndarray, g_squared: np.ndarray
    ) -> tuple:
        """
        Actualización Adagrad para descenso de gradiente.
        """
        g_squared += grad**2
        x_new = x - self.config.learning_rate * grad / (
            np.sqrt(g_squared) + self.config.eps
        )
        return x_new, g_squared

    def _rmsprop_update(
        self, x: np.ndarray, grad: np.ndarray, g_squared: np.ndarray
    ) -> tuple:
        """
        Actualización RMSprop para descenso de gradiente.
        """
        rho = 0.9
        g_squared = rho * g_squared + (1 - rho) * (grad**2)
        x_new = x - self.config.learning_rate * grad / (
            np.sqrt(g_squared) + self.config.eps
        )
        return x_new, g_squared

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
