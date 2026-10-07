"""MEMBER 4 - Masks, compositing (UI lives in app.py).

Contract:  clean_mask(S_frame) -> bool array (H, W)
"""
import numpy as np
from collections import deque


def convolve2d(image, kernel):
    """Pad, slide window, multiply by FLIPPED kernel, sum. Same output size (zero padding)."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    H, W = image.shape
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]

    before_r, after_r = kh // 2, kh - 1 - kh // 2
    before_c, after_c = kw // 2, kw - 1 - kw // 2
    padded = np.pad(image, ((before_r, after_r), (before_c, after_c)), mode="constant")

    out = np.zeros((H, W), dtype=float)
    for a in range(kh):
        for b in range(kw):
            out += flipped[a, b] * padded[a:a + H, b:b + W]
    return out


def gaussian_kernel(size=5, sigma=1.0):
    """Outer product of a normalised 1-D Gaussian with itself."""
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def _neighbourhood(mask, size, reducer, pad_value):
    mask = np.asarray(mask)
    H, W = mask.shape
    before, after = size // 2, size - 1 - size // 2
    padded = np.pad(mask, ((before, after), (before, after)),
                    mode="constant", constant_values=pad_value)
    out = None
    for a in range(size):
        for b in range(size):
            window = padded[a:a + H, b:b + W]
            out = window.copy() if out is None else reducer(out, window)
    return out


def erode(mask, size=3):
    """Minimum over neighbourhood."""
    mask = np.asarray(mask)
    pad_value = True if mask.dtype == bool else mask.max() if mask.size else 0
    return _neighbourhood(mask, size, np.minimum, pad_value)


def dilate(mask, size=3):
    """Maximum over neighbourhood."""
    mask = np.asarray(mask)
    pad_value = False if mask.dtype == bool else mask.min() if mask.size else 0
    return _neighbourhood(mask, size, np.maximum, pad_value)


def otsu_threshold(values):
    """Threshold maximising between-class variance."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return lo

    bins = 256
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(float)

    w0 = np.cumsum(hist)                     # weight of the dark class
    w1 = w0[-1] - w0                         # weight of the bright class
    cum_mean = np.cumsum(hist * centers)
    mu0 = cum_mean / np.where(w0 == 0, 1, w0)
    mu1 = (cum_mean[-1] - cum_mean) / np.where(w1 == 0, 1, w1)

    between = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(between))
    return float(edges[idx + 1])             # boundary between bin idx and idx+1


def _drop_small_blobs(mask, min_area):
    """Remove 4-connected components smaller than min_area pixels."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                comp = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if len(comp) >= min_area:
                    ys, xs = zip(*comp)
                    out[list(ys), list(xs)] = True
    return out


def clean_mask(S_frame):
    """|S| -> blur -> threshold -> opening -> closing -> drop tiny blobs -> dilate."""
    S = np.abs(np.asarray(S_frame, dtype=float))
    if S.ndim == 3:                          # colour frame: combine channels
        S = S.mean(axis=2)
    H, W = S.shape
    if S.size == 0 or S.max() <= 1e-12:
        return np.zeros((H, W), dtype=bool)

    # blur (edge-padded so borders are not darkened)
    k = gaussian_kernel(5, 1.5)
    r = k.shape[0] // 2
    blurred = convolve2d(np.pad(S, r, mode="edge"), k)[r:-r, r:-r]

    # threshold: Otsu, but never below a noise floor (median + 8*MAD) or 10% of the peak,
    # so a frame with no person (pure noise) gives an empty mask instead of a random one
    med = np.median(blurred)
    mad = 1.4826 * np.median(np.abs(blurred - med))
    thr = max(otsu_threshold(blurred), med + 8 * mad, 0.1 * blurred.max())
    mask = blurred > thr

    # opening (erode -> dilate) removes specks, closing (dilate -> erode) fills holes
    mask = dilate(erode(mask, 3), 3)
    mask = erode(dilate(mask, 5), 5)

    # drop tiny blobs
    min_area = max(10, int(0.0005 * H * W))
    mask = _drop_small_blobs(mask, min_area)

    # grow slightly so the person's edges are fully covered
    return dilate(mask, 5).astype(bool)


def composite(frame, background, mask):
    """out = (1-alpha)*frame + alpha*background, alpha = slightly blurred mask."""
    frame_arr = np.asarray(frame)
    f = frame_arr.astype(float)
    b = np.asarray(background, dtype=float)

    k = gaussian_kernel(5, 1.5)
    r = k.shape[0] // 2
    m = np.asarray(mask, dtype=float)
    alpha = convolve2d(np.pad(m, r, mode="edge"), k)[r:-r, r:-r]
    alpha = np.clip(alpha, 0.0, 1.0)
    if f.ndim == 3:
        alpha = alpha[..., None]

    out = (1.0 - alpha) * f + alpha * b
    if np.issubdtype(frame_arr.dtype, np.integer):
        info = np.iinfo(frame_arr.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(frame_arr.dtype)
    return out
