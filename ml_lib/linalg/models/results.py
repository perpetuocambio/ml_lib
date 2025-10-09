"""
Result classes para operaciones de álgebra lineal.

Estas clases reemplazan tuplas confusas con tipos fuertemente tipados
que proporcionan validación, documentación inline y mejor experiencia
de desarrollo.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QRDecompositionResult:
    """Resultado de la descomposición QR.

    La descomposición QR factoriza una matriz A en:
        A = Q @ R

    donde Q es ortogonal (Q.T @ Q = I) y R es triangular superior.

    Attributes:
        Q: Matriz ortogonal (m x n)
        R: Matriz triangular superior (n x n)

    Example:
        >>> result = linalg_service.qr_decomposition(A)
        >>> print(f"Q shape: {result.Q.shape}")
        >>> print(f"R shape: {result.R.shape}")
        >>> reconstructed = result.reconstruct()
        >>> assert result.verify_orthogonality()
    """

    Q: np.ndarray
    R: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones y propiedades."""
        if self.Q.ndim != 2:
            raise ValueError(f"Q debe ser 2D, got {self.Q.ndim}D")
        if self.R.ndim != 2:
            raise ValueError(f"R debe ser 2D, got {self.R.ndim}D")
        if self.Q.shape[1] != self.R.shape[0]:
            raise ValueError(
                f"Dimensiones incompatibles: Q.shape[1]={self.Q.shape[1]} "
                f"!= R.shape[0]={self.R.shape[0]}"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz original A = Q @ R.

        Returns:
            Matriz A reconstruida

        Example:
            >>> A_reconstructed = result.reconstruct()
            >>> np.allclose(A, A_reconstructed)
            True
        """
        return self.Q @ self.R

    def verify_orthogonality(self, atol: float = 1e-10) -> bool:
        """Verifica que Q sea ortogonal (Q.T @ Q ≈ I).

        Args:
            atol: Tolerancia absoluta para la comparación

        Returns:
            True si Q es ortogonal dentro de la tolerancia

        Example:
            >>> result.verify_orthogonality()
            True
        """
        product = self.Q.T @ self.Q
        identity = np.eye(self.Q.shape[1])
        return np.allclose(product, identity, atol=atol)


@dataclass
class LUDecompositionResult:
    """Resultado de la descomposición LU con pivoting.

    La descomposición LU con pivoting factoriza:
        PA = LU

    donde P es una permutación, L es triangular inferior con diagonal
    unitaria y U es triangular superior.

    Attributes:
        L: Matriz triangular inferior con diagonal de 1s (n x n)
        U: Matriz triangular superior (n x n)
        P: Matriz de permutación (n x n)

    Example:
        >>> result = linalg_service.lu_decomposition(A)
        >>> PA = result.reconstruct()
        >>> x = result.solve(b)  # Resuelve Ax = b
    """

    L: np.ndarray
    U: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.L.ndim != 2 or self.U.ndim != 2 or self.P.ndim != 2:
            raise ValueError("L, U, P deben ser matrices 2D")

        n = self.L.shape[0]
        if not (self.L.shape == (n, n) and self.U.shape == (n, n) and self.P.shape == (n, n)):
            raise ValueError(
                f"L, U, P deben ser cuadradas del mismo tamaño. "
                f"Got L:{self.L.shape}, U:{self.U.shape}, P:{self.P.shape}"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye PA = LU.

        Returns:
            Producto PA (no A directamente, debido al pivoting)

        Note:
            Para obtener A, necesitas: A = P.T @ (L @ U)
        """
        return self.L @ self.U

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Resuelve Ax = b usando la descomposición LU.

        Algoritmo:
            1. PAx = Pb (aplicar permutación)
            2. LUx = Pb
            3. Ly = Pb (forward substitution)
            4. Ux = y (backward substitution)

        Args:
            b: Vector del lado derecho (n,) o matriz (n, k)

        Returns:
            Solución x del sistema Ax = b

        Example:
            >>> result = linalg_service.lu_decomposition(A)
            >>> x = result.solve(b)
            >>> np.allclose(A @ x, b)
            True
        """
        # PA = LU, entonces Ax = b => PAx = Pb => LUx = Pb
        Pb = self.P @ b
        # Forward substitution: Ly = Pb
        y = self._forward_substitution(self.L, Pb)
        # Backward substitution: Ux = y
        x = self._backward_substitution(self.U, y)
        return x

    @staticmethod
    def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución forward para Ly = b con L triangular inferior."""
        n = L.shape[0]
        y = np.zeros_like(b, dtype=float)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
            if L[i, i] != 0:
                y[i] /= L[i, i]
        return y

    @staticmethod
    def _backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución backward para Ux = b con U triangular superior."""
        n = U.shape[0]
        x = np.zeros_like(b, dtype=float)
        for i in range(n - 1, -1, -1):
            x[i] = b[i] - np.dot(U[i, i + 1:], x[i + 1:])
            if U[i, i] != 0:
                x[i] /= U[i, i]
        return x


@dataclass
class SVDDecompositionResult:
    """Resultado de la descomposición en valores singulares (SVD).

    La descomposición SVD factoriza:
        A = U @ S @ Vt

    donde U y Vt son ortogonales y S es diagonal con valores singulares.

    Attributes:
        U: Matriz de vectores singulares izquierdos (m x m o m x k)
        s: Vector de valores singulares (min(m,n),) en orden descendente
        Vt: Matriz de vectores singulares derechos transpuesta (n x n o k x n)

    Example:
        >>> result = linalg_service.svd_decomposition(A)
        >>> A_reconstructed = result.reconstruct()
        >>> A_rank5 = result.low_rank_approximation(rank=5)
        >>> condition = result.condition_number()
    """

    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.U.ndim != 2 or self.Vt.ndim != 2:
            raise ValueError("U y Vt deben ser matrices 2D")
        if self.s.ndim != 1:
            raise ValueError(f"s debe ser vector 1D, got {self.s.ndim}D")

        # Verificar que s esté ordenado en orden descendente
        if len(self.s) > 1 and not np.all(self.s[:-1] >= self.s[1:]):
            raise ValueError("Los valores singulares deben estar en orden descendente")

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz original A = U @ S @ Vt.

        Returns:
            Matriz reconstruida

        Example:
            >>> A_reconstructed = result.reconstruct()
            >>> np.allclose(A, A_reconstructed)
            True
        """
        k = len(self.s)
        m, n = self.U.shape[0], self.Vt.shape[1]
        S = np.zeros((m, n))
        S[:k, :k] = np.diag(self.s)
        return self.U @ S @ self.Vt

    def low_rank_approximation(self, rank: int) -> np.ndarray:
        """Calcula aproximación de bajo rango de A.

        La aproximación de rango k minimiza ||A - A_k||_F entre todas
        las matrices de rango k (teorema de Eckart-Young).

        Args:
            rank: Número de valores singulares a mantener

        Returns:
            Aproximación de rango 'rank' de A

        Example:
            >>> # Compresión manteniendo 95% de la información
            >>> A_compressed = result.low_rank_approximation(rank=10)
            >>> error = np.linalg.norm(A - A_compressed)
        """
        if rank > len(self.s):
            rank = len(self.s)
        if rank <= 0:
            raise ValueError(f"rank debe ser positivo, got {rank}")

        U_k = self.U[:, :rank]
        s_k = self.s[:rank]
        Vt_k = self.Vt[:rank, :]

        S_k = np.diag(s_k)
        return U_k @ S_k @ Vt_k

    def condition_number(self) -> float:
        """Calcula el número de condición de la matriz.

        El número de condición es κ(A) = σ_max / σ_min, donde σ_max y σ_min
        son los valores singulares máximo y mínimo.

        Un número de condición alto indica una matriz mal condicionada.

        Returns:
            Número de condición, np.inf si el menor valor singular es 0

        Example:
            >>> cond = result.condition_number()
            >>> if cond > 1e10:
            ...     print("Matriz mal condicionada")
        """
        if len(self.s) == 0 or self.s[-1] == 0:
            return np.inf
        return self.s[0] / self.s[-1]

    def effective_rank(self, threshold: float = 1e-10) -> int:
        """Calcula el rango efectivo de la matriz.

        El rango efectivo cuenta cuántos valores singulares son mayores
        que el threshold (relativo al mayor valor singular).

        Args:
            threshold: Threshold relativo para considerar un valor singular significativo

        Returns:
            Número de valores singulares > threshold * σ_max

        Example:
            >>> eff_rank = result.effective_rank(threshold=1e-6)
            >>> print(f"Rango efectivo: {eff_rank}/{len(result.s)}")
        """
        if len(self.s) == 0:
            return 0
        cutoff = threshold * self.s[0]
        return int(np.sum(self.s > cutoff))


@dataclass
class EigenDecompositionResult:
    """Resultado de la descomposición en valores propios.

    Para una matriz A:
        A @ v = λ @ v

    donde λ es un valor propio y v es su vector propio correspondiente.

    Para matrices diagonalizables:
        A = V @ D @ V^(-1)

    Attributes:
        eigenvalues: Vector de valores propios (n,)
        eigenvectors: Matriz de vectores propios como columnas (n x n)

    Example:
        >>> result = linalg_service.eigen_decomposition(A)
        >>> dominant_eigenvalue = result.dominant_eigenvalue()
        >>> A_reconstructed = result.reconstruct()
        >>> is_diagonalizable = result.is_diagonalizable()
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues debe ser 1D, got {self.eigenvalues.ndim}D")
        if self.eigenvectors.ndim != 2:
            raise ValueError(f"eigenvectors debe ser 2D, got {self.eigenvectors.ndim}D")

        if len(self.eigenvalues) != self.eigenvectors.shape[1]:
            raise ValueError(
                f"Número de eigenvalues ({len(self.eigenvalues)}) debe coincidir "
                f"con número de eigenvectors ({self.eigenvectors.shape[1]})"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz si es diagonalizable: A = V @ D @ V^(-1).

        Returns:
            Matriz reconstruida

        Raises:
            np.linalg.LinAlgError: Si la matriz de eigenvectors no es invertible

        Example:
            >>> A_reconstructed = result.reconstruct()
            >>> np.allclose(A, A_reconstructed)
            True
        """
        V = self.eigenvectors
        D = np.diag(self.eigenvalues)
        V_inv = np.linalg.inv(V)
        return V @ D @ V_inv

    def dominant_eigenvalue(self) -> complex:
        """Retorna el valor propio dominante (mayor valor absoluto).

        Returns:
            Eigenvalue con mayor magnitud

        Example:
            >>> lambda_max = result.dominant_eigenvalue()
            >>> spectral_radius = abs(lambda_max)
        """
        idx = np.argmax(np.abs(self.eigenvalues))
        return self.eigenvalues[idx]

    def is_diagonalizable(self, atol: float = 1e-10) -> bool:
        """Verifica si la matriz es diagonalizable.

        Una matriz es diagonalizable si sus eigenvectors forman una base,
        es decir, si la matriz de eigenvectors es invertible.

        Args:
            atol: Tolerancia para considerar el determinante como cero

        Returns:
            True si la matriz de eigenvectors es invertible

        Example:
            >>> if result.is_diagonalizable():
            ...     A_reconstructed = result.reconstruct()
        """
        det = np.linalg.det(self.eigenvectors)
        return abs(det) > atol

    def spectral_radius(self) -> float:
        """Calcula el radio espectral ρ(A) = max|λ_i|.

        Returns:
            Radio espectral de la matriz

        Example:
            >>> rho = result.spectral_radius()
            >>> if rho < 1:
            ...     print("Iteraciones convergen")
        """
        return float(np.max(np.abs(self.eigenvalues)))


@dataclass
class CholeskyDecompositionResult:
    """Resultado de la descomposición de Cholesky.

    Para una matriz simétrica definida positiva A:
        A = L @ L.T

    donde L es triangular inferior.

    Attributes:
        L: Matriz triangular inferior (n x n)

    Example:
        >>> result = linalg_service.cholesky_decomposition(A)
        >>> A_reconstructed = result.reconstruct()
        >>> x = result.solve(b)  # Más eficiente que LU para matrices SPD
    """

    L: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.L.ndim != 2:
            raise ValueError(f"L debe ser 2D, got {self.L.ndim}D")
        if self.L.shape[0] != self.L.shape[1]:
            raise ValueError(f"L debe ser cuadrada, got shape {self.L.shape}")

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz original A = L @ L.T.

        Returns:
            Matriz simétrica reconstruida

        Example:
            >>> A_reconstructed = result.reconstruct()
            >>> np.allclose(A, A_reconstructed)
            True
        """
        return self.L @ self.L.T

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Resuelve Ax = b usando la descomposición de Cholesky.

        Algoritmo:
            1. Ly = b (forward substitution)
            2. L.T x = y (backward substitution)

        Args:
            b: Vector del lado derecho (n,) o matriz (n, k)

        Returns:
            Solución x del sistema Ax = b

        Example:
            >>> result = linalg_service.cholesky_decomposition(A)
            >>> x = result.solve(b)
            >>> np.allclose(A @ x, b)
            True
        """
        # Forward substitution: Ly = b
        y = self._forward_substitution(self.L, b)
        # Backward substitution: L.T x = y
        x = self._backward_substitution(self.L.T, y)
        return x

    @staticmethod
    def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución forward para Ly = b."""
        n = L.shape[0]
        y = np.zeros_like(b, dtype=float)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
            if L[i, i] != 0:
                y[i] /= L[i, i]
        return y

    @staticmethod
    def _backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución backward para Ux = b."""
        n = U.shape[0]
        x = np.zeros_like(b, dtype=float)
        for i in range(n - 1, -1, -1):
            x[i] = b[i] - np.dot(U[i, i + 1:], x[i + 1:])
            if U[i, i] != 0:
                x[i] /= U[i, i]
        return x

    def determinant(self) -> float:
        """Calcula el determinante usando la descomposición de Cholesky.

        Para A = L @ L.T:
            det(A) = det(L)^2 = (∏ L_ii)^2

        Returns:
            Determinante de la matriz original

        Example:
            >>> det_A = result.determinant()
        """
        diag_product = np.prod(np.diag(self.L))
        return diag_product ** 2
