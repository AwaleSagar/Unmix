"""Lightweight pure-Python matrix helpers (lists of lists)."""

import math


def zeros(rows, cols):
    """Create a *rows* × *cols* matrix of zeros."""
    return [[0.0] * cols for _ in range(rows)]


def full(rows, cols, value):
    """Create a *rows* × *cols* matrix filled with *value*."""
    return [[value] * cols for _ in range(rows)]


def shape(mat):
    """Return (rows, cols) of matrix *mat*."""
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    return rows, cols


def transpose(mat):
    """Transpose a matrix."""
    rows, cols = shape(mat)
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def mat_add(a, b):
    """Element-wise addition of two matrices."""
    rows, cols = shape(a)
    return [[a[r][c] + b[r][c] for c in range(cols)] for r in range(rows)]


def mat_sub(a, b):
    """Element-wise subtraction of two matrices."""
    rows, cols = shape(a)
    return [[a[r][c] - b[r][c] for c in range(cols)] for r in range(rows)]


def mat_scale(mat, s):
    """Multiply every element of *mat* by scalar *s*."""
    rows, cols = shape(mat)
    return [[mat[r][c] * s for c in range(cols)] for r in range(rows)]


def mat_mul(a, b):
    """Matrix multiplication A @ B."""
    ra, ca = shape(a)
    rb, cb = shape(b)
    if ca != rb:
        raise ValueError(f"Incompatible shapes: ({ra},{ca}) @ ({rb},{cb})")
    result = zeros(ra, cb)
    for i in range(ra):
        for j in range(cb):
            s = 0.0
            for k in range(ca):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def frobenius_norm(mat):
    """Frobenius norm of a matrix."""
    s = 0.0
    for row in mat:
        for v in row:
            s += v * v
    return math.sqrt(s)


def nuclear_norm(mat):
    """Nuclear norm (sum of singular values) of a matrix."""
    _, sigma, _ = svd(mat)
    return sum(sigma)


def soft_threshold(mat, tau):
    """Element-wise soft-thresholding: sign(x) * max(|x| - tau, 0)."""
    rows, cols = shape(mat)
    result = zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            v = mat[r][c]
            av = abs(v)
            result[r][c] = (1 if v >= 0 else -1) * max(av - tau, 0.0)
    return result


def svd_shrink(mat, tau):
    """Singular-value soft-thresholding (proximal operator for nuclear norm).

    Returns a matrix whose singular values are each reduced by *tau*
    (clamped to zero).
    """
    U, sigma, Vt = svd(mat)
    sigma_shrunk = [max(s - tau, 0.0) for s in sigma]
    # Reconstruct:  U @ diag(sigma_shrunk) @ Vt
    rows = len(U)
    cols = len(Vt[0]) if Vt else 0
    k = len(sigma_shrunk)
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for d in range(k):
                s += U[i][d] * sigma_shrunk[d] * Vt[d][j]
            result[i][j] = s
    return result


# ---------------------------------------------------------------------------
# SVD via one-sided Jacobi rotations
# ---------------------------------------------------------------------------

def svd(mat):
    """Compute a thin SVD of *mat* using one-sided Jacobi rotations.

    Returns (U, sigma, Vt) where *sigma* is a list of singular values
    (descending) and U, Vt are dense matrices such that mat ≈ U @ diag(σ) @ Vt.
    """
    m, n = shape(mat)
    # Work on the smaller dimension for efficiency
    if m < n:
        U, s, Vt = svd(transpose(mat))
        return transpose(Vt), s, transpose(U)

    # Copy A
    A = [row[:] for row in mat]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    max_iter = 100 * n
    tol = 1e-10
    for _ in range(max_iter):
        converged = True
        for p in range(n):
            for q in range(p + 1, n):
                # Compute 2x2 sub-problem on columns p, q
                alpha = sum(A[i][p] * A[i][p] for i in range(m))
                beta = sum(A[i][q] * A[i][q] for i in range(m))
                gamma = sum(A[i][p] * A[i][q] for i in range(m))

                if abs(gamma) < tol * math.sqrt(alpha * beta + 1e-30):
                    continue
                converged = False

                # Jacobi rotation angle
                zeta = (beta - alpha) / (2.0 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # Rotate columns of A
                for i in range(m):
                    ap = A[i][p]
                    aq = A[i][q]
                    A[i][p] = c * ap - s * aq
                    A[i][q] = s * ap + c * aq

                # Rotate columns of V
                for i in range(n):
                    vp = V[i][p]
                    vq = V[i][q]
                    V[i][p] = c * vp - s * vq
                    V[i][q] = s * vp + c * vq

        if converged:
            break

    # Extract singular values and build U
    sigma = []
    for j in range(n):
        s = math.sqrt(sum(A[i][j] ** 2 for i in range(m)))
        sigma.append(s)

    # Sort by descending singular value
    order = sorted(range(n), key=lambda j: -sigma[j])
    sigma_sorted = [sigma[order[j]] for j in range(n)]

    U = zeros(m, n)
    Vt = zeros(n, n)
    eps = 1e-15  # machine-precision guard for division by near-zero
    for j_new, j_old in enumerate(order):
        s = sigma[j_old]
        for i in range(m):
            U[i][j_new] = A[i][j_old] / s if s > eps else 0.0
        for i in range(n):
            Vt[j_new][i] = V[i][j_old]

    return U, sigma_sorted, Vt
