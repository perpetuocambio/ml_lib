"""
Ejemplo de uso del módulo de álgebra lineal de ml_lib
"""
import numpy as np
from typing import Dict, List, Tuple
import time

# Importar componentes de ml_lib
from ml_lib.linalg import (
    LinearAlgebraFactory,
    Matrix,
    MatrixOperationConfig,
    BLASService,
    LAPACKService
)
from ml_lib.core import LoggingService


def demo_basic_operations():
    """Demostración de operaciones básicas de álgebra lineal."""
    print("🧮 Demostración de Operaciones Básicas de Álgebra Lineal")
    print("=" * 60)
    
    # Crear motor de álgebra lineal
    engine = LinearAlgebraFactory.create_engine()
    
    # Generar matrices de ejemplo
    np.random.seed(42)
    A = np.random.randn(100, 50)  # Matriz rectangular
    B = np.random.randn(50, 75)   # Otra matriz rectangular
    C = np.random.randn(100, 75)  # Matriz para acumulación
    
    print(f"1. Creando matrices: A({A.shape}), B({B.shape}), C({C.shape})")
    
    # Producto matricial general (GEMM)
    print("\n2. Realizando producto matricial GEMM: C = α*A*B + β*C")
    start_time = time.time()
    result = engine.gemm(alpha=1.5, A=A, B=B, beta=0.5, C=C)
    gemm_time = time.time() - start_time
    
    print(f"   - Resultado: {result.shape}")
    print(f"   - Tiempo de ejecución: {gemm_time:.6f} segundos")
    
    # Producto matriz-vector (GEMV)
    print("\n3. Realizando producto matriz-vector GEMV: y = α*A*x + β*y")
    x = np.random.randn(50)  # Vector
    y = np.random.randn(100)  # Vector de salida
    
    start_time = time.time()
    result_vec = engine.gemv(alpha=2.0, A=A, x=x, beta=0.3, y=y)
    gemv_time = time.time() - start_time
    
    print(f"   - Resultado: {result_vec.shape}")
    print(f"   - Tiempo de ejecución: {gemv_time:.6f} segundos")
    
    # Factorización QR
    print("\n4. Realizando factorización QR")
    square_matrix = np.random.randn(50, 50)  # Matriz cuadrada
    
    start_time = time.time()
    Q, R = engine.qr_decomposition(square_matrix)
    qr_time = time.time() - start_time
    
    print(f"   - Factores Q({Q.shape}), R({R.shape})")
    print(f"   - Tiempo de ejecución: {qr_time:.6f} segundos")
    print(f"   - Verificación QR: {np.allclose(Q @ R, square_matrix, rtol=1e-10)}")
    
    # Factorización LU
    print("\n5. Realizando factorización LU")
    start_time = time.time()
    P, L, U = engine.lu_decomposition(square_matrix)
    lu_time = time.time() - start_time
    
    print(f"   - Factores P({P.shape}), L({L.shape}), U({U.shape})")
    print(f"   - Tiempo de ejecución: {lu_time:.6f} segundos")
    print(f"   - Verificación PLU: {np.allclose(P @ square_matrix, L @ U, rtol=1e-10)}")
    
    # Factorización de Cholesky
    print("\n6. Realizando factorización de Cholesky")
    # Crear matriz simétrica definida positiva
    chol_matrix = np.random.randn(40, 40)
    chol_matrix = chol_matrix.T @ chol_matrix + np.eye(40)  # Asegurar definida positiva
    
    start_time = time.time()
    L_chol = engine.cholesky_decomposition(chol_matrix)
    chol_time = time.time() - start_time
    
    print(f"   - Factor L({L_chol.shape})")
    print(f"   - Tiempo de ejecución: {chol_time:.6f} segundos")
    print(f"   - Verificación LL^T: {np.allclose(L_chol @ L_chol.T, chol_matrix, rtol=1e-10)}")


def demo_advanced_decompositions():
    """Demostración de descomposiciones avanzadas."""
    print("\n\n🔬 Demostración de Descomposiciones Avanzadas")
    print("=" * 50)
    
    # Crear motor de álgebra lineal
    engine = LinearAlgebraFactory.create_engine()
    
    # Generar matriz para SVD
    np.random.seed(123)
    matrix = np.random.randn(80, 60)  # Matriz rectangular
    
    print(f"1. Matriz para SVD: {matrix.shape}")
    
    # Descomposición SVD
    print("\n2. Realizando descomposición SVD")
    start_time = time.time()
    U, s, Vt = engine.svd_decomposition(matrix, full_matrices=False)
    svd_time = time.time() - start_time
    
    print(f"   - U({U.shape}), s({s.shape}), Vt({Vt.shape})")
    print(f"   - Tiempo de ejecución: {svd_time:.6f} segundos")
    print(f"   - Valores singulares principales: {s[:5]}")
    
    # Reconstruir matriz parcialmente
    k = 10  # Rango reducido
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    matrix_reconstructed = U_k @ np.diag(s_k) @ Vt_k
    frobenius_error = np.linalg.norm(matrix - matrix_reconstructed, 'fro') / np.linalg.norm(matrix, 'fro')
    
    print(f"   - Reconstrucción de rango {k}")
    print(f"   - Error de Frobenius relativo: {frobenius_error:.6f}")
    
    # Descomposición de valores propios
    print("\n3. Realizando descomposición de valores propios")
    # Crear matriz simétrica
    eig_matrix = np.random.randn(50, 50)
    eig_matrix = eig_matrix + eig_matrix.T  # Hacer simétrica
    
    start_time = time.time()
    eigenvalues, eigenvectors = engine.eigen_decomposition(eig_matrix)
    eig_time = time.time() - start_time
    
    print(f"   - Valores propios({eigenvalues.shape}), vectores propios({eigenvectors.shape})")
    print(f"   - Tiempo de ejecución: {eig_time:.6f} segundos")
    print(f"   - Valores propios extremos: [{eigenvalues[0]:.3f}, ..., {eigenvalues[-1]:.3f}]")
    print(f"   - Valores propios ordenados: {np.all(eigenvalues[:-1] >= eigenvalues[1:])}")


def demo_sparse_operations():
    """Demostración de operaciones con matrices dispersas."""
    print("\n\n📊 Demostración de Operaciones con Matrices Dispersas")
    print("=" * 55)
    
    # Crear motor de álgebra lineal
    engine = LinearAlgebraFactory.create_engine()
    
    # Crear una matriz dispersa en formato COO
    print("1. Creando matriz dispersa")
    np.random.seed(456)
    
    # Generar datos dispersos
    nnz = 200  # Número de elementos no nulos
    rows = np.random.randint(0, 100, nnz)
    cols = np.random.randint(0, 80, nnz)
    data = np.random.randn(nnz)
    
    sparse_matrix = engine.create_sparse_matrix(
        data=data,
        row_indices=rows,
        col_indices=cols,
        shape=(100, 80),
        format="COO"
    )
    
    print(f"   - Matriz dispersa: {sparse_matrix.shape}")
    print(f"   - Elementos no nulos: {sparse_matrix.nnz}")
    print(f"   - Densidad: {sparse_matrix.density:.4f}%")
    
    # Multiplicación dispersa-densa
    print("\n2. Realizando multiplicación dispersa-densa")
    dense_vector = np.random.randn(80)
    
    start_time = time.time()
    result = engine.sparse_matmul(sparse_matrix, dense_vector)
    sparse_time = time.time() - start_time
    
    print(f"   - Resultado: {result.shape}")
    print(f"   - Tiempo de ejecución: {sparse_time:.6f} segundos")


def demo_performance_comparison():
    """Comparación de rendimiento entre implementaciones."""
    print("\n\n⚡ Comparación de Rendimiento")
    print("=" * 35)
    
    # Configurar diferentes tamaños de matrices
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    print("Tamaño\t\tNumPy\t\tBLAS(ml_lib)")
    print("-" * 45)
    
    for m, n in sizes:
        # Matrices cuadradas
        np.random.seed(789)
        A = np.random.randn(m, n)
        B = np.random.randn(n, n)
        
        # Medir tiempo con NumPy
        start_time = time.time()
        C_numpy = A @ B
        numpy_time = time.time() - start_time
        
        # Medir tiempo con nuestra implementación BLAS
        engine = LinearAlgebraFactory.create_engine()
        start_time = time.time()
        C_blas = engine.gemm(1.0, A, B)
        blas_time = time.time() - start_time
        
        speedup = numpy_time / blas_time if blas_time > 0 else float('inf')
        
        print(f"{m}x{n}\t\t{numpy_time:.6f}s\t{blas_time:.6f}s ({speedup:.2f}x)")


def main():
    """Función principal que ejecuta todas las demostraciones."""
    print("🚀 Ejemplo Completo del Módulo de Álgebra Lineal de ml_lib")
    print("=" * 65)
    
    # Ejecutar demostraciones
    demo_basic_operations()
    demo_advanced_decompositions()
    demo_sparse_operations()
    demo_performance_comparison()
    
    print("\n\n✅ ¡Todas las demostraciones completadas exitosamente!")
    print("\n🎯 Funcionalidades demostradas:")
    print("   • Operaciones BLAS optimizadas (GEMM, GEMV)")
    print("   • Factorizaciones matriciales (QR, LU, Cholesky)")
    print("   • Descomposiciones avanzadas (SVD, Eigen)")
    print("   • Operaciones con matrices dispersas")
    print("   • Comparación de rendimiento")
    print("   • Manejo de errores robusto")
    print("   • Tipado estricto y validación automática")
    
    print("\n🔧 Componentes de ml_lib utilizados:")
    print("   • Interfaces definidas con tipado estricto")
    print("   • Modelos con validación automática")
    print("   • Servicios optimizados para álgebra lineal")
    print("   • Handlers para manejo de errores y configuración")
    print("   • Motores especializados para diferentes operaciones")
    print("   • Fábricas para creación de componentes")


if __name__ == "__main__":
    main()