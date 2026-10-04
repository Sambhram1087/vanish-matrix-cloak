"""MEMBER 3 - Robust PCA  (M = L + S).

Contract:  rpca_ialm(M) -> (L, S), both the same shape as M
"""
import numpy as np


def soft_threshold(X, tau):
    """TODO(M3): sign(x) * max(|x| - tau, 0)."""
    return X.copy()


def svt(X, tau):
    """TODO(M3): SVD, subtract tau from each singular value, clip at 0, rebuild. Returns (L, rank)."""
    return X.copy(), 0


def rpca_ialm(M, lam=None, tol=1e-2, max_iter=150):
    """TODO(M3): inexact ALM loop. Returns (L, S, history) where history is a dict of lists:
    'error', 'rank', 'sparsity' (one entry per iteration)."""
    return M.copy(), np.zeros_like(M), {"error": [], "rank": [], "sparsity": []}
