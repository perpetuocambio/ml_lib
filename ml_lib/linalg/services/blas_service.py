"""
Servicios para operaciones de álgebra lineal con tipado estricto.
"""
from typing import Optional
import numpy as np
from ml_lib.linalg.models import Matrix, SparseMatrix, DecompositionResult


class BLASService:
    """Servicio para operaciones BLAS optimizadas."""
    
    @staticmethod
    def gemm(A: np.ndarray, B: np.ndarray, alpha: float = 1.0, beta: float = 0.0, C: Optional[np.ndarray] = None) -> np.ndarray:
        """Producto matricial general (GEMM)."""
        result = alpha * np.dot(A, B)
        if C is not None:
            result += beta * C
        return result
    
    @staticmethod
    def gemv(A: np.ndarray, x: np.ndarray, alpha: float = 1.0, beta: float = 0.0, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Producto matriz-vector (GEMV)."""
        result = alpha * np.dot(A, x)
        if y is not None:
            result += beta * y
        return result


class DecompositionService:
    """Servicio para descomposiciones matriciales."""
    
    @staticmethod
    def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Realiza la descomposición QR."""
        from scipy.linalg import qr
        Q, R = qr(A)
        return Q, R
    
    @staticmethod
    def svd_decomposition(A: np.ndarray, full_matrices: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Realiza la descomposición SVD."""
        from scipy.linalg import svd
        U, s, Vh = svd(A, full_matrices=full_matrices)
        # Convertir s a matriz diagonal si es necesario
        S = np.zeros((A.shape[0], A.shape[1]))
        np.fill_diagonal(S, s)
        return U, S, Vh
    
    @staticmethod
    def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
        """Realiza la descomposición de Cholesky."""
        from scipy.linalg import cholesky
        return cholesky(A)
    
    @staticmethod
    def lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Realiza la descomposición LU."""
        from scipy.linalg import lu
        P, L, U = lu(A)
        return P, L, U


class SparseService:
    """Servicio para operaciones con matrices dispersas."""
    
    @staticmethod
    def to_csr(sparse_matrix: SparseMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convierte una matriz dispersa al formato CSR."""
        if sparse_matrix.format == "CSR":
            return sparse_matrix.data, sparse_matrix.row_indices, sparse_matrix.col_indices
        
        # Convertir de COO a CSR
        if sparse_matrix.format == "COO":
            # Ordenar índices de columna
            sorted_idx = np.lexsort((sparse_matrix.col_indices, sparse_matrix.row_indices))
            data = sparse_matrix.data[sorted_idx]
            row_indices = sparse_matrix.row_indices[sorted_idx]
            col_indices = sparse_matrix.col_indices[sorted_idx]
            
            # Crear array de índices de fila
            indptr = np.zeros(sparse_matrix.shape[0] + 1, dtype=int)
            for i in range(len(row_indices)):
                indptr[row_indices[i] + 1] += 1
            
            # Acumular
            for i in range(1, len(indptr)):
                indptr[i] += indptr[i-1]
            
            return data, indptr, col_indices
        
        raise ValueError(f"Conversion from {sparse_matrix.format} to CSR not implemented")