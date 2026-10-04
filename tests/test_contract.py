"""Shape tests for the shared data contract. These must ALWAYS pass on main."""
import numpy as np
import frames as F, svd_tools as V, rpca as R, masks as K

T, H, W = 20, 72, 96


def test_frames_contract():
    fr, truth = F.synthetic_video(T=T, height=H, width=W)
    assert fr.shape == (T, H, W) and truth.shape == (T, H, W) and truth.dtype == bool
    M = F.frames_to_matrix(fr)
    assert M.shape == (H * W, T)
    assert F.matrix_to_frames(M, H, W).shape == (T, H, W)


def test_svd_contract():
    M = np.random.default_rng(0).random((H * W, T))
    U, s, Vt = V.svd_from_scratch(M)
    assert U.shape == (H * W, T) and s.shape == (T,) and Vt.shape == (T, T)
    assert V.rank_k_approx(U, s, Vt, 2).shape == M.shape


def test_rpca_contract():
    M = np.random.default_rng(0).random((200, 30))
    L, S, hist = R.rpca_ialm(M)
    assert L.shape == M.shape and S.shape == M.shape
    assert set(hist) >= {"error", "rank", "sparsity"}


def test_mask_contract():
    m = K.clean_mask(np.zeros((H, W)))
    assert m.shape == (H, W) and m.dtype == bool
    assert K.composite(np.zeros((H, W)), np.zeros((H, W)), m).shape == (H, W)


def test_full_chain_runs():
    import main
    assert main.run().shape[1:] == (72, 96)
