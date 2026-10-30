"""
Linear Algebra Module

Pure Python implementations of linear algebra operations:
- solve: Gaussian elimination with partial pivoting
- lstsq: Least squares via normal equations
- lu_decompose: LU decomposition
- cholesky: Cholesky decomposition
- qr_decompose: QR decomposition
- svd: Singular value decomposition
- Matrix operations: dot, norm, matvec, transpose
"""

from finceptquantlib.numerical.linalg.decomposition import (
    lu_decompose,
    cholesky,
    qr_decompose,
    svd,
)

from finceptquantlib.numerical.linalg.solve import (
    solve,
    lstsq,
    matrix_inverse,
)

from finceptquantlib.numerical.linalg.matrix import (
    dot,
    norm,
    matvec,
    transpose,
    matmul,
    eye,
    zeros,
    frobenius_norm,
    outer,
)

__all__ = [
    # Decompositions
    "lu_decompose",
    "cholesky",
    "qr_decompose",
    "svd",
    # Solvers
    "solve",
    "lstsq",
    "matrix_inverse",
    # Matrix operations
    "dot",
    "norm",
    "matvec",
    "transpose",
    "matmul",
    "eye",
    "zeros",
    "frobenius_norm",
    "outer",
]
