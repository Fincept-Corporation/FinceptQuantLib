"""
Linear Algebra Utilities

Pure Python implementations of:
- Gaussian elimination with partial pivoting
- Least squares via normal equations
- LU decomposition
"""

from __future__ import annotations

from typing import List


def solve(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve linear system Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: Square matrix (n x n)
        b: Right-hand side vector (n)

    Returns:
        Solution vector x

    Raises:
        ValueError: If matrix is singular
    """
    n = len(b)
    if n == 0:
        return []

    # Create augmented matrix
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row

        # Swap rows
        M[col], M[max_row] = M[max_row], M[col]

        # Check for singular matrix
        if abs(M[col][col]) < 1e-14:
            raise ValueError("Singular matrix")

        # Eliminate column
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


def lstsq(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve least squares problem min ||Ax - b||² via normal equations.

    Args:
        A: Matrix (m x n)
        b: Right-hand side vector (m)

    Returns:
        Solution vector x (n)
    """
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])

    # A^T A
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    # A^T b
    Atb = [sum(A[k][i] * b[k] for k in range(m)) for i in range(n)]

    # Add small regularization for numerical stability
    for i in range(n):
        AtA[i][i] += 1e-12

    return solve(AtA, Atb)


def lu_decompose(A: List[List[float]]) -> tuple[List[List[float]], List[List[float]], List[int]]:
    """
    LU decomposition with partial pivoting: PA = LU

    Args:
        A: Square matrix (n x n)

    Returns:
        (L, U, pivot) where pivot is permutation indices
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[A[i][j] for j in range(n)] for i in range(n)]
    pivot = list(range(n))

    for k in range(n):
        # Find pivot
        max_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i

        # Swap
        U[k], U[max_row] = U[max_row], U[k]
        L[k], L[max_row] = L[max_row], L[k]
        pivot[k], pivot[max_row] = pivot[max_row], pivot[k]

        if abs(U[k][k]) < 1e-14:
            continue

        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    for i in range(n):
        L[i][i] = 1.0

    return L, U, pivot


def dot(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(v: List[float]) -> float:
    """Euclidean norm of vector."""
    from math import sqrt
    return sqrt(sum(x * x for x in v))


def matvec(A: List[List[float]], x: List[float]) -> List[float]:
    """Matrix-vector product Ax."""
    return [dot(row, x) for row in A]


def transpose(A: List[List[float]]) -> List[List[float]]:
    """Matrix transpose."""
    if not A:
        return []
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]
