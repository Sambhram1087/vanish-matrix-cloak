"""MEMBER 4 - Masks, compositing (UI lives in app.py).

Contract:  clean_mask(S_frame) -> bool array (H, W)
"""
import numpy as np


def gaussian_kernel(size=5, sigma=1.0):
    """Outer product of a normalised 1-D Gaussian with itself."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def convolve2d(image, kernel):
    """Pad, slide window, multiply by FLIPPED kernel, sum. Same output size."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    # Pad image to maintain same output size
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
    output = np.zeros_like(image, dtype=float)
    
    # Flip kernel as per convolution definition
    flipped_kernel = np.flip(kernel)
    
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * flipped_kernel)
            
    return output


def _morphology_filter(mask, size=3, mode='min'):
    """Helper for erosion (min) and dilation (max) using sliding windows."""
    ph, pw = size // 2, size // 2
    padded = np.pad(mask.astype(float), ((ph, ph), (pw, pw)), mode='constant', constant_values=(1.0 if mode=='min' else 0.0))
    output = np.zeros_like(mask, dtype=bool)
    
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            if mode == 'min':
                output[i, j] = np.min(region) > 0.5
            else:
                output[i, j] = np.max(region) > 0.5
    return output


def erode(mask, size=3):
    """Minimum over neighbourhood."""
    return _morphology_filter(mask, size, mode='min')


def dilate(mask, size=3):
    """Maximum over neighbourhood."""
    return _morphology_filter(mask, size, mode='max')


def otsu_threshold(values):
    """Threshold maximising between-class variance."""
    values = values.ravel()
    hist, bin_edges = np.histogram(values, bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    total_pixels = values.size
    current_max, threshold = 0.0, 0.0
    
    sum_total = np.sum(hist * bin_centers)
    sum_background = 0.0
    weight_background = 0.0
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += hist[i] * bin_centers[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calculate between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > current_max:
            current_max = variance
            threshold = bin_centers[i]
            
    return threshold


def clean_mask(S_frame):
    """|S| -> blur -> threshold -> opening -> closing -> drop tiny blobs -> dilate."""
    # 1. Magnitude / absolute value if complex/multi-channel
    if S_frame.ndim == 3:
        magnitude = np.mean(np.abs(S_frame), axis=2)
    else:
        magnitude = np.abs(S_frame)
        
    # Normalize to [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
        
    # 2. Blur
    kernel = gaussian_kernel(size=5, sigma=1.0)
    blurred = convolve2d(magnitude, kernel)
    
    # 3. Threshold using Otsu
    thresh = otsu_threshold(blurred)
    binary = blurred > thresh
    
    # 4. Opening (Erode then Dilate) & Closing (Dilate then Erode)
    opened = dilate(erode(binary, size=3), size=3)
    closed = erode(dilate(opened, size=3), size=3)
    
    # 5. Dilate final mask slightly for smooth compositing boundaries
    final_mask = dilate(closed, size=3)
    
    return final_mask


def composite(frame, background, mask):
    """out = (1-alpha)*frame + alpha*background, alpha = slightly blurred mask."""
    # Create smooth alpha channel by blurring the boolean mask
    alpha_kernel = gaussian_kernel(size=7, sigma=2.0)
    alpha = convolve2d(mask.astype(float), alpha_kernel)
    
    # Expand alpha dimensions for multi-channel frames (RGB)
    if frame.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[..., np.newaxis]
        
    # Blend frame and background
    out = (1.0 - alpha) * frame + alpha * background
    return np.clip(out, 0, 255).astype(frame.dtype)