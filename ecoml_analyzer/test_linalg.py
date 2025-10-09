"""
Prueba del módulo de álgebra lineal de ml_lib
"""
import numpy as np
import unittest
from typing import Any, Dict, List, Tuple

# Importar componentes de ml_lib
from ml_lib.linalg import (
    LinearAlgebraFactory,
    Matrix,
    SparseMatrix,
    MatrixOperationConfig,
    BLASService,
    LAPACKService,
    LinearAlgebraEngine,
    LinearAlgebraErrorHandler
)
from ml_lib.core import LoggingService


class TestLinearAlgebraModels(unittest.TestCase):
    """Pruebas para modelos de álgebra lineal."""
    
    def setUp(self):
        """Configuración previa a las pruebas."""
        self.logger_service = LoggingService("TestLinearAlgebra")
        self.logger = self.logger_service.get_logger()
    
    def test_matrix_creation(self):
        """Prueba la creación de matrices."""
        data = np.random.randn(5, 5)
        matrix = Matrix(data=data, metadata={"test": True})
        
        self.assertEqual(matrix.shape, (5, 5))
        self.assertTrue(matrix.is_square())
        self.assertIsInstance(matrix.data, np.ndarray)
        self.assertEqual(matrix.metadata["test"], True)
    
    def test_sparse_matrix_creation(self):
        """Prueba la creación de matrices dispersas."""
        data = np.array([1.0, 2.0, 3.0])
        rows = np.array([0, 1, 2])
        cols = np.array([0, 1, 2])
        shape = (3, 3)
        
        sparse_matrix = LinearAlgebraFactory.create_sparse_matrix(
            data, rows, cols, shape, "COO"
        )
        
        self.assertEqual(sparse_matrix.shape, shape)
        self.assertEqual(sparse_matrix.nnz, 3)
        self.assertAlmostEqual(sparse_matrix.density, 1.0/3.0)
        self.assertEqual(sparse_matrix.format, "COO")
    
    def test_matrix_validation(self):
        """Prueba la validación de matrices."""
        # Matriz válida
        valid_data = np.random.randn(4, 4)
        matrix = Matrix(data=valid_data)
        matrix.validate_matrix()  # No debería lanzar excepción
        
        # Matriz con NaN
        invalid_data = np.array([[1.0, np.nan], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            invalid_matrix = Matrix(data=invalid_data)
            invalid_matrix.validate_matrix()
        
        # Matriz con infinito
        invalid_data2 = np.array([[1.0, np.inf], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            invalid_matrix2 = Matrix(data=invalid_data2)
            invalid_matrix2.validate_matrix()


class TestBLASService(unittest.TestCase):
    """Pruebas para el servicio BLAS."""
    
    def setUp(self):
        """Configuración previa a las pruebas."""
        self.logger_service = LoggingService("TestBLAS")
        self.logger = self.logger_service.get_logger()
        self.blas_service = BLASService(self.logger)
    
    def test_gemm_operation(self):
        """Prueba la operación GEMM."""
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)
        C = np.zeros((3, 2))
        
        # C = 1.0 * A * B + 0.0 * C
        result = BLASService.gemm(1.0, A, B, 0.0, C)
        expected = A @ B
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_gemv_operation(self):
        """Prueba la operación GEMV."""
        A = np.random.randn(4, 3)
        x = np.random.randn(3)
        y = np.zeros(4)
        
        # y = 1.0 * A * x + 0.0 * y
        result = BLASService.gemv(1.0, A, x, 0.0, y)
        expected = A @ x
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestLAPACKService(unittest.TestCase):
    """Pruebas para el servicio LAPACK."""
    
    def setUp(self):
        """Configuración previa a las pruebas."""
        self.logger_service = LoggingService("TestLAPACK")
        self.logger = self.logger_service.get_logger()
        self.lapack_service = LAPACKService(self.logger)
    
    def test_qr_factorization(self):
        """Prueba la factorización QR."""
        A = np.random.randn(5, 4)
        Q, R = LAPACKService.qr_factorize(A)
        
        # Verificar que A = Q @ R
        np.testing.assert_allclose(A, Q @ R, rtol=1e-10)
        # Verificar que Q es ortogonal
        np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), rtol=1e-10)
        # Verificar que R es triangular superior
        self.assertTrue(np.allclose(R, np.triu(R)))
    
    def test_lu_factorization(self):
        """Prueba la factorización LU."""
        A = np.random.randn(4, 4)
        P, L, U = LAPACKService.lu_factorize(A)
        
        # Verificar que P @ A = L @ U
        np.testing.assert_allclose(P @ A, L @ U, rtol=1e-10)
        # Verificar que L es triangular inferior
        np.testing.assert_allclose(L, np.tril(L), rtol=1e-10)
        # Verificar que U es triangular superior
        np.testing.assert_allclose(U, np.triu(U), rtol=1e-10)
    
    def test_cholesky_factorization(self):
        """Prueba la factorización de Cholesky."""
        # Crear matriz simétrica definida positiva
        A = np.random.randn(4, 4)
        A = A.T @ A + np.eye(4)  # Asegurar definida positiva
        
        L = LAPACKService.cholesky_factorize(A)
        
        # Verificar que A = L @ L.T
        np.testing.assert_allclose(A, L @ L.T, rtol=1e-10)
        # Verificar que L es triangular inferior
        np.testing.assert_allclose(L, np.tril(L), rtol=1e-10)


class TestLinearAlgebraEngine(unittest.TestCase):
    """Pruebas para el motor de álgebra lineal."""
    
    def setUp(self):
        """Configuración previa a las pruebas."""
        self.engine = LinearAlgebraFactory.create_engine()
    
    def test_engine_creation(self):
        """Prueba la creación del motor."""
        self.assertIsInstance(self.engine, LinearAlgebraEngine)
        self.assertIsNotNone(self.engine.blas_service)
        self.assertIsNotNone(self.engine.lapack_service)
    
    def test_solve_linear_system(self):
        """Prueba la resolución de sistemas lineales."""
        # Sistema lineal: Ax = b
        A = np.random.randn(5, 5)
        x_true = np.random.randn(5)
        b = A @ x_true
        
        # Resolver
        x_computed = self.engine.solve_linear_system(A, b)
        
        # Verificar solución
        np.testing.assert_allclose(x_computed, x_true, rtol=1e-10)
    
    def test_svd_decomposition(self):
        """Prueba la descomposición SVD."""
        A = np.random.randn(6, 4)
        U, s, Vt = self.engine.svd_decomposition(A, full_matrices=False)
        
        # Reconstruir matriz
        S = np.diag(s)
        A_reconstructed = U @ S @ Vt
        
        # Verificar reconstrucción
        np.testing.assert_allclose(A, A_reconstructed, rtol=1e-10)
    
    def test_eigen_decomposition(self):
        """Prueba la descomposición de valores propios."""
        # Matriz simétrica
        A = np.random.randn(5, 5)
        A = A + A.T  # Hacer simétrica
        
        eigenvalues, eigenvectors = self.engine.eigen_decomposition(A)
        
        # Verificar que Av = λv para cada valor/vector propio
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            Av = A @ v
            lambda_v = eigenvalues[i] * v
            np.testing.assert_allclose(Av, lambda_v, rtol=1e-10)


def run_tests():
    """Ejecuta todas las pruebas."""
    print("🧪 Ejecutando pruebas del módulo de álgebra lineal")
    print("=" * 50)
    
    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Agregar pruebas
    suite.addTests(loader.loadTestsFromTestCase(TestLinearAlgebraModels))
    suite.addTests(loader.loadTestsFromTestCase(TestBLASService))
    suite.addTests(loader.loadTestsFromTestCase(TestLAPACKService))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearAlgebraEngine))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Reportar resultados
    print(f"\n📊 Resultados:")
    print(f"   Pruebas ejecutadas: {result.testsRun}")
    print(f"   Fallos: {len(result.failures)}")
    print(f"   Errores: {len(result.errors)}")
    print(f"   Éxito: {'✅' if result.wasSuccessful() else '❌'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print("\n❌ Algunas pruebas fallaron.")
        exit(1)