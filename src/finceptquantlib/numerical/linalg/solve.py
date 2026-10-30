"""
Linear System Solvers

Solvers for linear systems Ax = b:
- Gaussian elimination with partial pivoting
- Least squares via normal equations
- Matrix inverse
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
    Solve least squares problem min ||Ax - b||^2 via normal equations.

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


def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """
    Compute matrix inverse using Gauss-Jordan elimination.

    Args:
        A: Square matrix (n x n)

    Returns:
        Inverse matrix A^(-1)

    Raises:
        ValueError: If matrix is singular
    """
    n = len(A)
    if n == 0:
        return []

    # Create augmented matrix [A | I]
    M = [[A[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
         for i in range(n)]

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

        # Scale pivot row
        pivot = M[col][col]
        for j in range(2 * n):
            M[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row != col:
                factor = M[row][col]
                for j in range(2 * n):
                    M[row][j] -= factor * M[col][j]

    # Extract inverse from right half
    return [[M[i][j] for j in range(n, 2 * n)] for i in range(n)]
