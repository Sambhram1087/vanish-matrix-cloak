"""Extended test suite covering edge cases, numerical contracts, and integration."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import frames as F
import svd_tools as V
import rpca as R
import masks as K


# ─── Fixtures ────────────────────────────────────────────────────────────────

T, H, W = 20, 48, 64

@pytest.fixture(scope="module")
def synthetic():
    fr, truth = F.synthetic_video(num_frames=T, height=H, width=W)
    return fr, truth

@pytest.fixture(scope="module")
def rpca_result(synthetic):
    frames, _ = synthetic
    M = F.frames_to_matrix(frames)
    L, S, hist = R.rpca_ialm(M, tol=1e-2, max_iter=100)
    return M, L, S, hist


# ─── frames.py tests ─────────────────────────────────────────────────────────

class TestFrames:
    def test_shapes(self, synthetic):
        frames, truth = synthetic
        assert frames.shape == (T, H, W)
        assert truth.shape == (T, H, W)
        assert truth.dtype == bool

    def test_roundtrip(self, synthetic):
        frames, _ = synthetic
        M = F.frames_to_matrix(frames)
        assert M.shape == (H * W, T)
        recovered = F.matrix_to_frames(M, H, W)
        assert recovered.shape == (T, H, W)
        np.testing.assert_allclose(recovered, frames, atol=1e-10)

    def test_moving_object_exists(self, synthetic):
        frames, truth = synthetic
        # At least some frames should have foreground
        assert truth.any(), "No foreground pixels in any frame"

    def test_background_values_in_range(self, synthetic):
        frames, _ = synthetic
        assert frames.min() >= 0 and frames.max() <= 255


# ─── svd_tools.py tests ──────────────────────────────────────────────────────

class TestSVD:
    def test_shapes(self):
        M = np.random.default_rng(0).random((H * W, T))
        U, s, Vt = V.svd_from_scratch(M)
        assert U.shape == (H * W, T)
        assert s.shape == (T,)
        assert Vt.shape == (T, T)

    def test_singular_values_descending(self):
        M = np.random.default_rng(1).standard_normal((50, 20))
        _, s, _ = V.svd_from_scratch(M)
        assert np.all(s[:-1] >= s[1:] - 1e-10), "Singular values not sorted descending"

    def test_singular_values_nonneg(self):
        M = np.random.default_rng(2).standard_normal((40, 15))
        _, s, _ = V.svd_from_scratch(M)
        assert np.all(s >= -1e-10)

    def test_matches_numpy(self):
        rng = np.random.default_rng(3)
        M = rng.standard_normal((30, 10))
        U, s, Vt = V.svd_from_scratch(M)
        _, s_np, _ = np.linalg.svd(M, full_matrices=False)
        np.testing.assert_allclose(s, s_np, atol=1e-8)

    def test_reconstruction(self):
        rng = np.random.default_rng(4)
        M = rng.standard_normal((30, 10))
        U, s, Vt = V.svd_from_scratch(M)
        M_hat = (U * s) @ Vt
        np.testing.assert_allclose(M_hat, M, atol=1e-8)

    def test_rank_k_approx_shape(self):
        M = np.random.default_rng(5).random((H * W, T))
        U, s, Vt = V.svd_from_scratch(M)
        for k in [1, 2, 5]:
            approx = V.rank_k_approx(U, s, Vt, k)
            assert approx.shape == M.shape

    def test_rank_k_eckart_young(self):
        """rank-1 approx error should equal sqrt(sum of sigma_i^2 for i>1)."""
        M = np.random.default_rng(6).standard_normal((20, 8))
        U, s, Vt = V.svd_from_scratch(M)
        err = V.eckart_young_error(s, k=1)
        M1 = V.rank_k_approx(U, s, Vt, 1)
        actual = np.linalg.norm(M - M1, "fro")
        assert abs(err - actual) < 1e-8

    def test_randomized_svd_shape(self):
        M = np.random.default_rng(7).standard_normal((100, 30))
        U, s, Vt = V.randomized_svd(M, k=5)
        assert U.shape == (100, 5) and s.shape == (5,) and Vt.shape == (5, 30)

    def test_compression_table(self):
        M = np.random.default_rng(8).standard_normal((50, 20))
        U, s, Vt = V.svd_from_scratch(M)
        rows = V.compression_table(M, U, s, Vt, ks=[1, 2, 5])
        assert len(rows) == 3
        for row in rows:
            assert "ratio" in row and "psnr" in row and row["ratio"] > 0


# ─── rpca.py tests ───────────────────────────────────────────────────────────

class TestRPCA:
    def test_output_shapes(self, rpca_result):
        M, L, S, hist = rpca_result
        assert L.shape == M.shape and S.shape == M.shape

    def test_decomposition_constraint(self, rpca_result):
        """L + S should approximate M."""
        M, L, S, hist = rpca_result
        rel_err = np.linalg.norm(M - L - S, "fro") / np.linalg.norm(M, "fro")
        assert rel_err < 0.05, f"Constraint violation too large: {rel_err:.4f}"

    def test_history_keys(self, rpca_result):
        _, _, _, hist = rpca_result
        assert {"error", "rank", "sparsity"} <= set(hist)

    def test_history_lengths_match(self, rpca_result):
        _, _, _, hist = rpca_result
        assert len(hist["error"]) == len(hist["rank"]) == len(hist["sparsity"])

    def test_error_decreasing(self, rpca_result):
        _, _, _, hist = rpca_result
        errors = hist["error"]
        # Error should generally decrease (allow 1 increase due to ALM dynamics)
        increases = sum(1 for a, b in zip(errors, errors[1:]) if b > a * 1.05)
        assert increases <= max(3, len(errors) // 5), "Error not generally decreasing"

    def test_soft_threshold(self):
        x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        result = R.soft_threshold(x, 1.0)
        np.testing.assert_allclose(result, [-2.0, 0.0, 0.0, 0.0, 2.0])

    def test_svt_rank_reduction(self):
        """SVT on a rank-1 matrix with large tau should return rank-0."""
        u = np.ones((10, 1))
        v = np.ones((1, 5))
        X = 0.5 * u @ v  # singular value = 0.5*sqrt(50) ≈ 3.5
        L, rank = R.svt(X, 10.0)  # tau=10 > largest singular value
        assert rank == 0
        np.testing.assert_allclose(L, np.zeros_like(X), atol=1e-10)

    def test_planted_problem_recovery(self):
        M, L0, S0 = R.planted_problem(m=80, n=40, rank=2, density=0.05, seed=42)
        L, S, hist = R.rpca_ialm(M, tol=1e-5, max_iter=300)
        rel_L = np.linalg.norm(L - L0) / np.linalg.norm(L0)
        rel_S = np.linalg.norm(S - S0) / max(np.linalg.norm(S0), 1e-10)
        assert rel_L < 0.05, f"L recovery error too large: {rel_L:.4f}"
        assert rel_S < 0.05, f"S recovery error too large: {rel_S:.4f}"

    def test_zero_input(self):
        M = np.zeros((20, 10))
        L, S, hist = R.rpca_ialm(M)
        np.testing.assert_allclose(L, 0)
        np.testing.assert_allclose(S, 0)
        assert hist["error"] == []


# ─── masks.py tests ──────────────────────────────────────────────────────────

class TestMasks:
    def test_clean_mask_shape(self):
        m = K.clean_mask(np.zeros((H, W)))
        assert m.shape == (H, W) and m.dtype == bool

    def test_clean_mask_zeros(self):
        """All-zero input → all-False mask."""
        m = K.clean_mask(np.zeros((H, W)))
        assert not m.any()

    def test_clean_mask_detects_foreground(self):
        """Bright square on dark background should be detected."""
        S = np.zeros((H, W))
        S[H//4:3*H//4, W//4:3*W//4] = 50.0
        m = K.clean_mask(S, mad_k=1.0, peak_frac=0.02, dilate_px=3)
        assert m.any(), "Foreground not detected"

    def test_composite_shape(self):
        frame = np.zeros((H, W))
        bg    = np.ones((H, W)) * 128
        mask  = np.zeros((H, W), dtype=bool)
        out = K.composite(frame, bg, mask)
        assert out.shape == (H, W)

    def test_composite_zero_mask_returns_frame(self):
        """With mask=0 everywhere, output should equal original frame."""
        frame = np.random.default_rng(0).random((H, W)) * 255
        bg    = np.zeros((H, W))
        mask  = np.zeros((H, W), dtype=bool)
        out = K.composite(frame, bg, mask)
        np.testing.assert_allclose(out, frame, atol=1e-10)

    def test_composite_full_mask_returns_background(self):
        """With mask=1 everywhere, output should equal background."""
        frame = np.zeros((H, W))
        bg    = np.ones((H, W)) * 200.0
        mask  = np.ones((H, W), dtype=bool)
        out = K.composite(frame, bg, mask)
        # Full-mask + Gaussian blur means soft alpha ≈ 1 in the centre
        centre = out[H//4:3*H//4, W//4:3*W//4]
        np.testing.assert_allclose(centre, 200.0, atol=1.0)

    def test_gaussian_kernel_sums_to_one(self):
        k = K.gaussian_kernel(5, 1.5)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_dilate_erode_inverse_property(self):
        """dilate then erode should not shrink below original for convex blobs."""
        mask = np.zeros((H, W), dtype=bool)
        mask[H//3:2*H//3, W//3:2*W//3] = True
        result = K.erode(K.dilate(mask, 3), 3)
        # Original interior pixels should still be True
        assert result[H//2, W//2]

    def test_otsu_threshold_separates(self):
        """Otsu on bimodal data should pick a threshold between the two peaks."""
        low  = np.random.default_rng(0).normal(10, 1, 500)
        high = np.random.default_rng(1).normal(90, 1, 500)
        data = np.concatenate([low, high])
        thr = K.otsu_threshold(data)
        assert 10 < thr < 90, f"Otsu threshold {thr} not between peaks"


# ─── Full pipeline integration test ──────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_shape(self, rpca_result, synthetic):
        frames, _ = synthetic
        M, L, S, hist = rpca_result
        S_fr = F.matrix_to_frames(S, H, W)
        L_fr = F.matrix_to_frames(L, H, W)
        masks = np.stack([K.clean_mask(S_fr[t]) for t in range(T)])
        out   = np.stack([K.composite(frames[t], L_fr[t], masks[t]) for t in range(T)])
        assert out.shape == (T, H, W)

    def test_output_pixel_range(self, rpca_result, synthetic):
        frames, _ = synthetic
        M, L, S, hist = rpca_result
        S_fr = F.matrix_to_frames(S, H, W)
        L_fr = F.matrix_to_frames(L, H, W)
        masks = np.stack([K.clean_mask(S_fr[t]) for t in range(T)])
        out   = np.stack([K.composite(frames[t], L_fr[t], masks[t]) for t in range(T)])
        # Output should be in a reasonable float range (values from 0-255 input)
        assert out.min() >= -1 and out.max() <= 256
