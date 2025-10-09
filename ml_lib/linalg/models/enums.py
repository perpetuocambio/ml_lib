"""
Enumeraciones para el módulo de álgebra lineal de ml_lib.
"""

from enum import Enum, auto


class MatrixPrecision(Enum):
    """Precisión numérica para operaciones matriciales."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT128 = "float128"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


class MemoryLayout(Enum):
    """Layout de memoria para arrays."""
    ROW_MAJOR = "C"  # C-contiguous
    COLUMN_MAJOR = "F"  # Fortran-contiguous
    ANY = "A"  # Any (no specific layout)


class DecompositionMethod(Enum):
    """Métodos de descomposición matricial."""
    LU = "lu"
    QR = "qr"
    SVD = "svd"
    CHOLESKY = "cholesky"
    EIGEN = "eigen"
    SCHUR = "schur"
    QZ = "qz"
    HESSENBERG = "hessenberg"


class SolverMethod(Enum):
    """Métodos para resolver sistemas lineales."""
    DIRECT = "direct"
    ITERATIVE = "iterative"
    CONJUGATE_GRADIENT = "cg"
    GMRES = "gmres"
    BICGSTAB = "bicgstab"
    MINRES = "minres"
    LU_SOLVE = "lu_solve"
    CHOLESKY_SOLVE = "cholesky_solve"


class SparseFormat(Enum):
    """Formatos de matrices dispersas."""
    CSR = "csr"  # Compressed Sparse Row
    CSC = "csc"  # Compressed Sparse Column
    COO = "coo"  # Coordinate format
    DIA = "dia"  # Diagonal format
    LIL = "lil"  # List of Lists
    DOK = "dok"  # Dictionary of Keys
    BSR = "bsr"  # Block Sparse Row


class NormType(Enum):
    """Tipos de norma."""
    FROBENIUS = "fro"
    NUCLEAR = "nuc"
    L1 = 1
    L2 = 2
    INFINITY = float("inf")
    NEG_INFINITY = float("-inf")


class EigenSolver(Enum):
    """Algoritmos para cálculo de eigenvalues."""
    STANDARD = "standard"
    GENERALIZED = "generalized"
    SPARSE = "sparse"
    POWER_METHOD = "power"
    LANCZOS = "lanczos"
    ARNOLDI = "arnoldi"


class MatrixProperty(Enum):
    """Propiedades especiales de matrices."""
    SYMMETRIC = "symmetric"
    HERMITIAN = "hermitian"
    POSITIVE_DEFINITE = "positive_definite"
    ORTHOGONAL = "orthogonal"
    UNITARY = "unitary"
    DIAGONAL = "diagonal"
    TRIANGULAR_UPPER = "triangular_upper"
    TRIANGULAR_LOWER = "triangular_lower"
    SPARSE = "sparse"
    DENSE = "dense"


class ConvergenceCriterion(Enum):
    """Criterios de convergencia para métodos iterativos."""
    RESIDUAL_NORM = "residual_norm"
    RELATIVE_RESIDUAL = "relative_residual"
    MAX_ITERATIONS = "max_iterations"
    TOLERANCE = "tolerance"


class PreconditionerType(Enum):
    """Tipos de precondicionadores."""
    NONE = "none"
    JACOBI = "jacobi"
    ILU = "ilu"  # Incomplete LU
    ICC = "icc"  # Incomplete Cholesky
    SSOR = "ssor"  # Symmetric Successive Over-Relaxation
