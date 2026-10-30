"""
Matrix Decompositions

Pure Python implementations:
- LU decomposition with partial pivoting
- Cholesky decomposition for positive definite matrices
- QR decomposition via Gram-Schmidt
- SVD (power iteration approximation)
"""

from __future__ import annotations

from typing import List, Tuple
import math


def lu_decompose(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
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


def cholesky(A: List[List[float]]) -> List[List[float]]:
    """
    Cholesky decomposition: A = LL^T

    Args:
        A: Symmetric positive definite matrix (n x n)

    Returns:
        Lower triangular matrix L

    Raises:
        ValueError: If matrix is not positive definite
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                if abs(L[j][j]) < 1e-14:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


def qr_decompose(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    QR decomposition via modified Gram-Schmidt: A = QR

    Args:
        A: Matrix (m x n), m >= n

    Returns:
        (Q, R) where Q is orthogonal (m x n) and R is upper triangular (n x n)
    """
    m = len(A)
    if m == 0:
        return [], []
    n = len(A[0])

    # Copy A to work array (column-major for efficiency)
    V = [[A[i][j] for i in range(m)] for j in range(n)]

    Q = [[0.0] * n for _ in range(m)]
    R = [[0.0] * n for _ in range(n)]

    for j in range(n):
        # v_j = a_j
        v = V[j][:]

        # Orthogonalize against previous columns
        for i in range(j):
            # R[i][j] = q_i^T v_j
            R[i][j] = sum(Q[k][i] * v[k] for k in range(m))
            # v_j = v_j - R[i][j] * q_i
            for k in range(m):
                v[k] -= R[i][j] * Q[k][i]

        # R[j][j] = ||v_j||
        norm_v = math.sqrt(sum(x * x for x in v))
        R[j][j] = norm_v

        # q_j = v_j / ||v_j||
        if norm_v > 1e-14:
            for k in range(m):
                Q[k][j] = v[k] / norm_v
        else:
            for k in range(m):
                Q[k][j] = 0.0

    return Q, R


def svd(
    A: List[List[float]],
    max_iter: int = 100,
    tol: float = 1e-10
) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Singular Value Decomposition: A = U S V^T

    Uses power iteration method for computing singular values.
    Suitable for small to medium matrices.

    Args:
        A: Matrix (m x n)
        max_iter: Maximum power iterations per singular value
        tol: Convergence tolerance

    Returns:
        (U, S, Vt) where:
        - U: Left singular vectors (m x k)
        - S: Singular values (k)
        - Vt: Right singular vectors transposed (k x n)
        where k = min(m, n)
    """
    m = len(A)
    if m == 0:
        return [], [], []
    n = len(A[0])
    k = min(m, n)

    # Work with A^T A for V, then compute U
    AtA = [[sum(A[p][i] * A[p][j] for p in range(m)) for j in range(n)] for i in range(n)]

    U = []
    S = []
    V = []

    # Deflation method: extract singular values one by one
    for _ in range(k):
        # Power iteration to find dominant eigenvector of AtA
        v = [1.0 / math.sqrt(n)] * n
        sigma_sq = 0.0

        for _ in range(max_iter):
            # w = AtA @ v
            w = [sum(AtA[i][j] * v[j] for j in range(n)) for i in range(n)]

            # New eigenvalue estimate
            sigma_sq_new = sum(w[i] * v[i] for i in range(n))

            # Normalize
            norm_w = math.sqrt(sum(x * x for x in w))
            if norm_w < 1e-14:
                break
            v = [x / norm_w for x in w]

            if abs(sigma_sq_new - sigma_sq) < tol * max(abs(sigma_sq_new), 1.0):
                sigma_sq = sigma_sq_new
                break
            sigma_sq = sigma_sq_new

        sigma = math.sqrt(max(sigma_sq, 0.0))

        if sigma < tol:
            break

        S.append(sigma)
        V.append(v[:])

        # Compute u = A @ v / sigma
        u = [sum(A[i][j] * v[j] for j in range(n)) / sigma for i in range(m)]
        U.append(u)

        # Deflate: AtA = AtA - sigma^2 * v @ v^T
        for i in range(n):
            for j in range(n):
                AtA[i][j] -= sigma_sq * v[i] * v[j]

    # Convert U to column-major format
    if U:
        U_matrix = [[U[j][i] for j in range(len(U))] for i in range(m)]
    else:
        U_matrix = []

    # V is already in row format (Vt)
    Vt = V

    return U_matrix, S, Vt
