"""MEMBER 2 - SVD and rank-k.

Contract:  svd_from_scratch(M) -> U (m, n), s (n,), Vt (n, n)   thin SVD, s sorted descending
"""
import numpy as np


def svd_from_scratch(M):
    """G = M^T M -> eigh -> sigma = sqrt(lambda), U = M V / sigma. Must match np.linalg.svd."""
    M = np.asarray(M, dtype=float)
    m, n = M.shape

    # 1. Gram matrix (n x n, symmetric positive semi-definite)
    G = M.T @ M

    # 2. Eigen-decomposition (eigh returns eigenvalues in ASCENDING order)
    lam, V = np.linalg.eigh(G)

    # 3. Sort descending
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]

    # 4. Singular values (clip tiny negatives caused by round-off)
    s = np.sqrt(np.clip(lam, 0.0, None))

    # 5. U = M V / sigma  (skip directions where sigma is ~0)
    # M^T M squares the condition number, so values below ~sqrt(eps)*sigma_1 are noise
    tol = 100 * np.sqrt(np.finfo(float).eps) * (s[0] if n > 0 else 1.0)
    U = np.zeros((m, n))
    good = s > tol
    U[:, good] = (M @ V[:, good]) / s[good]

    # For zero singular values, complete U with orthonormal vectors (QR completion)
    if not np.all(good):
        k_good = int(good.sum())
        rng = np.random.default_rng(0)
        A = np.hstack([U[:, good], rng.standard_normal((m, n - k_good))])
        Q, _ = np.linalg.qr(A)
        U[:, ~good] = Q[:, k_good:n]

    Vt = V.T
    return U, s, Vt


def rank_k_approx(U, s, Vt, k):
    """sum_{i<=k} sigma_i u_i v_i^T."""
    k = int(max(0, min(k, len(s))))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def eckart_young_error(s, k):
    """sqrt(sum_{i>k} sigma_i^2)  (Frobenius-norm error of the best rank-k approximation)."""
    s = np.asarray(s, dtype=float)
    return float(np.sqrt(np.sum(s[k:] ** 2)))


def randomized_svd(M, k, oversample=8):
    """Random Gaussian sketch -> QR -> SVD of small matrix. Used by live mode."""
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    r = min(k + oversample, n)

    rng = np.random.default_rng(0)
    Omega = rng.standard_normal((n, r))      # random test matrix
    Y = M @ Omega                            # sketch of the column space (m x r)
    Q, _ = np.linalg.qr(Y)                   # orthonormal basis (m x r)

    B = Q.T @ M                              # small matrix (r x n)
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub                               # lift back to m dimensions

    return U[:, :k], s[:k], Vt[:k, :]


def compression_table(M, U, s, Vt, ks=(1, 2, 5, 10)):
    """Rows of dict(k, numbers, ratio, psnr). numbers = k*(m+n+1)."""
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    rows = []
    peak = float(np.max(np.abs(M))) if M.size else 1.0
    if peak == 0:
        peak = 1.0
    for k in ks:
        if k > len(s):
            continue
        Mk = rank_k_approx(U, s, Vt, k)
        numbers = k * (m + n + 1)
        ratio = (m * n) / numbers
        mse = float(np.mean((M - Mk) ** 2))
        psnr = float("inf") if mse == 0 else float(20 * np.log10(peak) - 10 * np.log10(mse))
        rows.append(dict(k=k, numbers=numbers, ratio=ratio, psnr=psnr))
    return rows
