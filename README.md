# Vanish · Matrix Cloak

> **An invisibility-cloak effect powered entirely by linear algebra — no deep learning, no background subtraction library.**

Mini project for **UE25MA242A — Mathematical Foundation for AI & Data Science (MFAD 2026)**.

Video frames are stacked as columns of a matrix **M** and decomposed into **M = L + S** (low-rank background + sparse moving foreground) using SVD and Robust PCA. The foreground is then masked and replaced with the recovered background, producing an invisibility-cloak effect.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Project Architecture](#project-architecture)
4. [Module Reference](#module-reference)
5. [Setup & Usage](#setup--usage)
6. [Streamlit App](#streamlit-app)
7. [Pipeline Parameters](#pipeline-parameters)
8. [Troubleshooting](#troubleshooting)
9. [Team](#team)

---

## How It Works

```
Input Video
    │
    ▼
Stack T frames as columns  ──► M  (pixels × frames)
    │
    ▼
Robust PCA (IALM)
    │  minimise ‖L‖_* + λ‖S‖₁   s.t. L + S = M
    │
    ├──► L  (low-rank)   ── static background
    └──► S  (sparse)     ── moving foreground
    │
    ▼
Mask Generation
    │  |S| ──► Gaussian blur ──► adaptive threshold ──► morphology ──► dilation
    │  (optionally OR'd with Median background subtraction)
    │
    ▼
Compositing
    out[t] = (1 − α) · frame[t] + α · L[t]
    where α = soft mask (Gaussian-blurred binary mask)
    │
    ▼
Output Video  (foreground vanished, background intact)
```

---

## Mathematical Foundation

### 1. Matrix Representation of Video

A video of **T** frames, each **H × W** pixels, is flattened into a matrix:

```
M ∈ ℝ^(HW × T)
```

Each column `M[:, t]` is the vectorised t-th grayscale frame. A static background makes most columns nearly identical, so `rank(M) ≈ 1`. A moving person adds sparse, high-magnitude perturbations.

### 2. Singular Value Decomposition (SVD)

For any matrix `M`:

```
M = U Σ Vᵀ        (thin SVD)
```

- `U ∈ ℝ^(m×n)` — left singular vectors (spatial patterns)
- `Σ = diag(σ₁ ≥ σ₂ ≥ … ≥ 0)` — singular values
- `Vᵀ ∈ ℝ^(n×n)` — right singular vectors (temporal weights)

The **rank-k approximation** retains only the top-k terms:

```
M_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
```

By the Eckart–Young theorem, `M_k` is the best rank-k approximation in Frobenius norm.

Our custom SVD (`svd_tools.py`) computes the Gram matrix `G = MᵀM`, applies `eigh`, and recovers `U = MV / σ` — matching `numpy.linalg.svd` exactly.

### 3. Robust PCA — M = L + S

Standard low-rank approximation fails when the sparse component (moving person) has large magnitude. **Robust PCA** solves the convex relaxation:

```
minimise   ‖L‖_*  +  λ ‖S‖₁
subject to  L + S = M
```

| Term | Meaning |
|---|---|
| `‖L‖_*` | Nuclear norm (sum of singular values) → promotes **low rank** → recovers background |
| `‖S‖₁` | L1 norm (sum of absolute values) → promotes **sparsity** → recovers foreground |
| `λ` | Trade-off weight; default `1 / √max(m, n)` (Candès et al., 2011) |

### 4. IALM Algorithm (Inexact Augmented Lagrange Multiplier)

Each iteration of `rpca_ialm` performs:

```
L  ←  SVT_{1/μ}( M − S + Y/μ )       # singular value thresholding
S  ←  soft_{λ/μ}( M − L + Y/μ )      # soft thresholding (element-wise)
Y  ←  Y + μ (M − L − S)              # dual variable update
μ  ←  ρ · μ                           # penalty growth (ρ = 1.5)
```

Where:
- **Soft threshold**: `soft(x, τ) = sign(x) · max(|x| − τ, 0)` — proximal operator of L1
- **SVT** (Singular Value Thresholding): applies soft threshold to singular values — proximal operator of nuclear norm
- Convergence criterion: `‖M − L − S‖_F / ‖M‖_F < tol`

### 5. Mask Generation

Given a sparse frame `S[t]`:

1. **Gaussian blur** — suppresses isolated noise pixels
2. **Adaptive threshold** — `thr = max(Otsu(S), median + k·MAD, peak_frac · max(S))`
3. **Morphological opening** (erode → dilate) — removes specks
4. **Morphological closing** (dilate → erode) — fills holes
5. **Connected-component filter** — drops blobs smaller than `0.05%` of frame area
6. **Dilation** — grows mask outward to fully cover object edges

### 6. Compositing

```
out[t] = (1 − α) · frame[t]  +  α · L[t]
```

`α` is the mask softened by a Gaussian blur — preventing hard edges around the replaced region.

---

## Project Architecture

```
vanish-matrix-cloak/
├── app.py            # Streamlit UI — upload, controls, results, animated preview, downloads
├── frames.py         # Video ↔ matrix reshaping + synthetic video generator
├── rpca.py           # Robust PCA (IALM): svt, soft_threshold, rpca_ialm
├── svd_tools.py      # From-scratch SVD, rank-k approx, randomised SVD, PSNR table
├── masks.py          # Mask pipeline: Gaussian, convolve2d, erode/dilate, Otsu, clean_mask, composite
├── main.py           # CLI entry point (demo on synthetic video)
├── tests/            # pytest test suite
├── samples/          # Sample input/output videos
└── requirements.txt
```

---

## Module Reference

### `frames.py` — Video ↔ Matrix (Member 1)

| Function | Description |
|---|---|
| `synthetic_video(T, H, W)` | Generates a static-background + moving-box video for testing |
| `frames_to_matrix(frames)` | `(T, H, W)` → `(H·W, T)` column matrix |
| `matrix_to_frames(M, H, W)` | `(H·W, T)` → `(T, H, W)` video tensor |

### `svd_tools.py` — SVD & Rank-k Approximation (Member 2)

| Function | Description |
|---|---|
| `svd_from_scratch(M)` | Full thin SVD via Gram matrix + `eigh`; matches `numpy.linalg.svd` |
| `rank_k_approx(U, s, Vt, k)` | Best rank-k approximation: `Σᵢ≤k σᵢ uᵢ vᵢᵀ` |
| `eckart_young_error(s, k)` | Frobenius-norm reconstruction error for rank-k |
| `randomized_svd(M, k)` | Fast randomised SVD via Gaussian sketch + QR |
| `compression_table(M, U, s, Vt, ks)` | Compression ratio and PSNR for each rank-k |

### `rpca.py` — Robust PCA (Member 3)

| Function | Description |
|---|---|
| `soft_threshold(X, τ)` | Element-wise soft thresholding: proximal operator of L1 norm |
| `svt(X, τ)` | Singular value thresholding: proximal operator of nuclear norm |
| `rpca_ialm(M, lam, tol, max_iter, rho)` | Full IALM solver; returns `(L, S, history)` |
| `planted_problem(m, n, rank, density)` | Synthetic test: known low-rank + sparse ground truth |

### `masks.py` — Masking & Compositing (Member 4)

| Function | Description |
|---|---|
| `gaussian_kernel(size, sigma)` | 2D Gaussian kernel (normalised outer product) |
| `convolve2d(image, kernel)` | Pure-NumPy 2D convolution with zero-padding |
| `erode(mask, size)` / `dilate(mask, size)` | Morphological min/max over neighbourhood |
| `otsu_threshold(values)` | Maximises between-class variance (256-bin histogram) |
| `clean_mask(S_frame, mad_k, peak_frac, min_area, dilate_px)` | Full mask pipeline on one sparse frame |
| `composite(frame, background, mask)` | Soft alpha compositing using blurred mask |

### `app.py` — Streamlit UI (Member 4)

Full browser-based pipeline with:
- **Video upload** (MP4, AVI, MOV, MKV, WebM, FLV, 3GP)
- **Sidebar controls** for all pipeline parameters
- **Frame explorer** — slider-driven per-frame view of all 4/5 panels
- **Animated MP4 preview** — Original / Background / Foreground / Composite
- **RPCA convergence charts** — error, rank, sparsity per iteration
- **One-click downloads** for all output videos

---

## Setup & Usage

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone <repo-url>
cd vanish-matrix-cloak
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest -q
```

### CLI Demo (synthetic video)

```bash
python main.py
```

### Launch the Streamlit App

```bash
streamlit run app.py
```

---

## Streamlit App

After uploading a video and clicking **▶ Run Pipeline**, the app shows:

| Panel | Description |
|---|---|
| **Original** | Raw input frame |
| **Background (L)** | Recovered low-rank background |
| **Foreground (S)** | Sparse residual (colourmap applied) |
| **Composite (out)** | Final invisibility-cloak result |
| **Mask (binary)** | Raw binary foreground mask (toggleable) |

---

## Pipeline Parameters

### Video Loading

| Parameter | Default | Effect |
|---|---|---|
| `Max frames` | 60 | More frames → better background estimation. Use ≥ 60 for real videos. |
| `Resize height (px)` | 96 | Larger = more detail, slower. 96–120 px is a good balance. |

### RPCA

| Parameter | Default | Effect |
|---|---|---|
| `Convergence tolerance` | `1e-2` | Stop when `‖M−L−S‖/‖M‖ < tol`. Lower = more accurate, slower. |
| `Max iterations` | 150 | Hard cap on IALM iterations. |

### Mask Mode

| Mode | Description |
|---|---|
| **RPCA sparse (S)** | Uses the S matrix directly. Best for small/fast-moving objects. |
| **Median background subtractor** | Computes median frame over time and subtracts. Very robust for large/slow objects. |
| **RPCA + Median (union)** | OR of both masks — catches the most foreground. **(Recommended)** |

### Mask Tuning

| Parameter | Default | Effect |
|---|---|---|
| `Sensitivity (mad_k)` | 2.0 | Lower → more pixels flagged as foreground. Original value was 8.0 (too conservative). |
| `Peak fraction floor` | 0.04 | Threshold ≥ `peak_frac × max(|S|)`. Lower → more foreground. |
| `Mask dilation (px)` | 7 | Grows mask outward to fully cover object edges. |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hand not disappearing at all | Switch to **RPCA + Median (union)** mask mode |
| Hand partially visible | Lower **Sensitivity (mad_k)** below 2.0; increase **Mask dilation** |
| Background flickering | Keep camera **completely still** — any shake breaks background estimation |
| Background poorly estimated | Increase **Max frames** to 80–120; record 1–2 s of empty background first |
| Processing too slow | Lower **Resize height** to 64–80 px; reduce **Max frames** |
| Mask coverage near zero | Lower `mad_k` below 1.5; switch to Median or Union mode |

---

## Team

| # | Name | GitHub | Module |
|---|---|---|---|
| 1 | Rahul P | [RahulP2007](https://github.com/RahulP2007) | Data + live pipeline (`frames.py`) |
| 2 | Rohith M | [Rohith-developer14](https://github.com/Rohith-developer14) | SVD + rank-k (`svd_tools.py`) |
| 3 | Sambhram Laxman Sattigeri | [Sambhram1087](https://github.com/Sambhram1087) | Robust PCA (`rpca.py`) |
| 4 | S Banuteja Reddy | [banuteja2007](https://github.com/banuteja2007) | Masks + UI + integration (`masks.py`, `app.py`) |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow.

---

## References

- Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). *Robust principal component analysis?* Journal of the ACM, 58(3), 1–37.
- Lin, Z., Chen, M., & Ma, Y. (2010). *The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices.* arXiv:1009.5055.
- Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank.* Psychometrika, 1(3), 211–218.
