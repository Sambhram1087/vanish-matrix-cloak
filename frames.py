"""MEMBER 1 - Data and live pipeline.

Contract:  frames -> float array (T, H, W), values in [0, 1], grayscale
           M      -> (H*W, T), one flattened frame per column
"""
import numpy as np


def load_video(path, size=(96, 72), max_frames=100):
    """TODO(M1): read with OpenCV, resize to `size` (width, height), grayscale, scale to 0-1."""
    w, h = size
    return np.zeros((max_frames, h, w))          # dummy so others can import


def frames_to_matrix(frames):
    """TODO(M1): (T, H, W) -> (H*W, T). One column per frame."""
    T = frames.shape[0]
    return np.zeros((frames.shape[1] * frames.shape[2], T))


def matrix_to_frames(M, height, width):
    """TODO(M1): inverse of frames_to_matrix -> (T, H, W)."""
    return np.zeros((M.shape[1], height, width))


def synthetic_video(T=60, height=72, width=96, seed=0):
    """TODO(M1): static room + moving shapes.
    Returns (frames, true_masks) where true_masks is bool (T, H, W)."""
    return np.zeros((T, height, width)), np.zeros((T, height, width), dtype=bool)


def write_video(frames, path, fps=15):
    """TODO(M1): save frames as mp4."""
    pass
