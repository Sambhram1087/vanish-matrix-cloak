"""MEMBER 2 - SVD and rank-k.

Contract:  svd_from_scratch(M) -> U (m, n), s (n,), Vt (n, n)   thin SVD, s sorted descending
"""
import numpy as np


def svd_from_scratch(M):
    """TODO(M2): G = M^T M -> eigh -> sigma = sqrt(lambda), U = M V / sigma. Must match np.linalg.svd."""
    m, n = M.shape
    return np.zeros((m, n)), np.zeros(n), np.zeros((n, n))


def rank_k_approx(U, s, Vt, k):
    """TODO(M2): sum_{i<=k} sigma_i u_i v_i^T."""
    return np.zeros((U.shape[0], Vt.shape[1]))


def eckart_young_error(s, k):
    """TODO(M2): sqrt(sum_{i>k} sigma_i^2)."""
    return 0.0


def randomized_svd(M, k, oversample=8):
    """TODO(M2): random Gaussian sketch -> QR -> SVD of small matrix. Used by live mode."""
    m, n = M.shape
    return np.zeros((m, k)), np.zeros(k), np.zeros((k, n))


def compression_table(M, U, s, Vt, ks=(1, 2, 5, 10)):
    """TODO(M2): rows of dict(k, numbers, ratio, psnr). numbers = k*(m+n+1)."""
    return []
