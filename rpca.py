"""MEMBER 3 - Robust PCA  (M = L + S).

Contract:  rpca_ialm(M) -> (L, S, history), L and S the same shape as M

Solves      minimise  ||L||_* + lam * ||S||_1     subject to   L + S = M
with the Inexact Augmented Lagrange Multiplier (IALM) method.

    ||L||_*  nuclear norm (sum of singular values)  -> promotes LOW RANK  (background)
    ||S||_1  sum of absolute entries                -> promotes SPARSITY  (moving people)

Each iteration:
    L <- SVT_{1/mu}( M - S + Y/mu )        singular value thresholding
    S <- soft_{lam/mu}( M - L + Y/mu )     soft thresholding
    Y <- Y + mu (M - L - S)                dual (Lagrange multiplier) update
    mu <- rho * mu
"""
import numpy as np


def soft_threshold(X, tau):
    """Proximal operator of the L1 norm: sign(x) * max(|x| - tau, 0).

    Shrinks every entry towards 0 by tau; entries smaller than tau become exactly 0.
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def svt(X, tau):
    """Singular value thresholding (proximal operator of the nuclear norm).

    SVD of X, subtract tau from every singular value, clip at 0, rebuild.
    Returns (L, rank) where rank = number of singular values that survived.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_shrunk = np.maximum(s - tau, 0.0)
    rank = int(np.sum(s_shrunk > 0))
    if rank == 0:
        return np.zeros_like(X), 0
    L = (U[:, :rank] * s_shrunk[:rank]) @ Vt[:rank]      # U diag(s') Vt using only the kept part
    return L, rank


def rpca_ialm(M, lam=None, tol=1e-2, max_iter=150, rho=1.5):
    """Robust PCA by inexact ALM.

    Parameters
    ----------
    M        (m, n) data matrix (video: pixels x frames)
    lam      weight of the L1 term; default 1/sqrt(max(m, n))  (Candes et al.)
    tol      stop when ||M - L - S||_F / ||M||_F < tol.
             Use ~1e-2 for noisy video (noise floor is 1-2 %); use 1e-7 for exact planted tests.
    max_iter safety cap on the number of iterations
    rho      growth factor of the penalty mu (>1)

    Returns
    -------
    L, S, history
    history = {'error': [...], 'rank': [...], 'sparsity': [...]}   (one entry per iteration)
    """
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_M = np.linalg.norm(M, "fro")
    if norm_M == 0:                                       # all-zero input: nothing to split
        return np.zeros_like(M), np.zeros_like(M), {"error": [], "rank": [], "sparsity": []}

    spectral = np.linalg.norm(M, 2)                       # largest singular value
    J = max(spectral, np.abs(M).max() / lam)              # scaling for the initial dual variable
    Y = M / J
    mu = 1.25 / spectral
    mu_max = mu * 1e7                                     # keep mu from blowing up

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    history = {"error": [], "rank": [], "sparsity": []}

    for _ in range(max_iter):
        # 1) low-rank update: singular value thresholding
        L, rank = svt(M - S + Y / mu, 1.0 / mu)
        # 2) sparse update: soft thresholding
        S = soft_threshold(M - L + Y / mu, lam / mu)
        # 3) dual update
        Z = M - L - S                                     # constraint violation
        Y = Y + mu * Z
        mu = min(mu * rho, mu_max)

        error = np.linalg.norm(Z, "fro") / norm_M
        history["error"].append(float(error))
        history["rank"].append(rank)
        history["sparsity"].append(float(np.mean(np.abs(S) > 1e-8)))

        if error < tol:
            break

    return L, S, history


# ----------------------------------------------------------------------------
# Self-test on a planted problem:  python rpca.py
# ----------------------------------------------------------------------------
def planted_problem(m=120, n=60, rank=3, density=0.05, seed=0):
    """Random rank-`rank` L0 plus sparse S0 with large entries. Returns (M, L0, S0)."""
    rng = np.random.default_rng(seed)
    L0 = rng.standard_normal((m, rank)) @ rng.standard_normal((rank, n))
    S0 = np.zeros((m, n))
    hit = rng.random((m, n)) < density
    S0[hit] = rng.choice([-10.0, 10.0], hit.sum())
    return L0 + S0, L0, S0


if __name__ == "__main__":
    # soft threshold by hand
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    print("soft_threshold(x, 1) =", soft_threshold(x, 1.0), "(expect [-2. 0. 0. 0. 2.])")

    # SVT shrinks singular values by exactly tau
    A = np.random.default_rng(1).standard_normal((30, 12))
    L_, r_ = svt(A, 1.5)
    s_before = np.linalg.svd(A, compute_uv=False)
    s_after = np.linalg.svd(L_, compute_uv=False)
    print("SVT shrink check:", np.allclose(s_after[:r_], s_before[:r_] - 1.5), "| rank kept:", r_)

    # planted problem
    M, L0, S0 = planted_problem()
    L, S, hist = rpca_ialm(M, tol=1e-7, max_iter=300)
    print(f"planted: iterations={len(hist['error'])}  final rank(L)={hist['rank'][-1]} (true 3)")
    print(f"         rel. error L = {np.linalg.norm(L - L0) / np.linalg.norm(L0):.2e}")
    print(f"         rel. error S = {np.linalg.norm(S - S0) / np.linalg.norm(S0):.2e}   (target < 1e-3)")
