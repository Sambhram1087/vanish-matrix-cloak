"""MEMBER 4 - Masks, compositing (UI lives in app.py).

Contract:  clean_mask(S_frame) -> bool array (H, W)
"""
import numpy as np


def convolve2d(image, kernel):
    """TODO(M4): pad, slide window, multiply by FLIPPED kernel, sum. Same output size."""
    return image.copy()


def gaussian_kernel(size=5, sigma=1.0):
    """TODO(M4): outer product of a normalised 1-D Gaussian with itself."""
    k = np.zeros((size, size)); k[size // 2, size // 2] = 1.0
    return k


def erode(mask, size=3):
    """TODO(M4): minimum over neighbourhood."""
    return mask.copy()


def dilate(mask, size=3):
    """TODO(M4): maximum over neighbourhood."""
    return mask.copy()


def otsu_threshold(values):
    """TODO(M4): threshold maximising between-class variance."""
    return 0.1


def clean_mask(S_frame):
    """TODO(M4): |S| -> blur -> threshold -> opening -> closing -> drop tiny blobs -> dilate."""
    return np.zeros(S_frame.shape[:2], dtype=bool)


def composite(frame, background, mask):
    """TODO(M4): out = (1-alpha)*frame + alpha*background, alpha = slightly blurred mask."""
    return frame.copy()
