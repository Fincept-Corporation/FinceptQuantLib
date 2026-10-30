"""
Matrix Operations

Basic matrix and vector operations in pure Python.
"""

from __future__ import annotations

from typing import List
import math


def dot(a: List[float], b: List[float]) -> float:
    """
    Dot product of two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Scalar dot product
    """
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(v: List[float], p: int = 2) -> float:
    """
    Vector norm.

    Args:
        v: Vector
        p: Norm order (1, 2, or inf)

    Returns:
        Norm value
    """
    if p == 1:
        return sum(abs(x) for x in v)
    elif p == 2:
        return math.sqrt(sum(x * x for x in v))
    elif p == float('inf'):
        return max(abs(x) for x in v) if v else 0.0
    else:
        return sum(abs(x) ** p for x in v) ** (1.0 / p)


def matvec(A: List[List[float]], x: List[float]) -> List[float]:
    """
    Matrix-vector product Ax.

    Args:
        A: Matrix (m x n)
        x: Vector (n)

    Returns:
        Result vector (m)
    """
    return [dot(row, x) for row in A]


def transpose(A: List[List[float]]) -> List[List[float]]:
    """
    Matrix transpose.

    Args:
        A: Matrix (m x n)

    Returns:
        Transposed matrix (n x m)
    """
    if not A:
        return []
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Matrix multiplication AB.

    Args:
        A: Matrix (m x k)
        B: Matrix (k x n)

    Returns:
        Result matrix (m x n)
    """
    if not A or not B:
        return []

    m = len(A)
    k = len(A[0])
    n = len(B[0])

    if len(B) != k:
        raise ValueError(f"Incompatible dimensions: {m}x{k} and {len(B)}x{n}")

    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def eye(n: int) -> List[List[float]]:
    """
    Identity matrix.

    Args:
        n: Matrix dimension

    Returns:
        n x n identity matrix
    """
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def zeros(m: int, n: int) -> List[List[float]]:
    """
    Zero matrix.

    Args:
        m: Number of rows
        n: Number of columns

    Returns:
        m x n zero matrix
    """
    return [[0.0] * n for _ in range(m)]


def frobenius_norm(A: List[List[float]]) -> float:
    """
    Frobenius norm of a matrix.

    Args:
        A: Matrix

    Returns:
        Frobenius norm
    """
    return math.sqrt(sum(sum(x * x for x in row) for row in A))


def outer(a: List[float], b: List[float]) -> List[List[float]]:
    """
    Outer product of two vectors.

    Args:
        a: First vector (m)
        b: Second vector (n)

    Returns:
        Outer product matrix (m x n)
    """
    return [[ai * bj for bj in b] for ai in a]
