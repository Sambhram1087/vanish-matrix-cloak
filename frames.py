"""MEMBER 1 - Video generation and matrix reshaping utilities."""
import numpy as np
def synthetic_video(num_frames=30, height=64, width=64):
    """Generates a synthetic video sequence (static background + moving object) and ground truth."""
    # Static background (e.g., a soft gradient or random texture)
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    bg = 0.5 * (xx + yy) * 255.0
    
    frames = np.zeros((num_frames, height, width), dtype=float)
    truth = np.zeros((num_frames, height, width), dtype=bool)
    
    # Moving square (the foreground object)
    box_size = 12
    for t in range(num_frames):
        frame = bg.copy()
        # Calculate moving position
        x_pos = int(10 + t * 1.2) % (width - box_size)
        y_pos = int(20 + np.sin(t / 3.0) * 5) % (height - box_size)
        
        # Add moving box to frame
        frame[y_pos:y_pos+box_size, x_pos:x_pos+box_size] = 255.0
        
        frames[t] = frame
        
        # Mark truth mask
        mask = np.zeros((height, width), dtype=bool)
        mask[y_pos:y_pos+box_size, x_pos:x_pos+box_size] = True
        truth[t] = mask
        
    return frames, truth


def frames_to_matrix(frames):
    """Reshapes (T, H, W) video tensor into an (H * W, T) matrix where each column is a flattened frame."""
    T, H, W = frames.shape
    # Reshape to (T, H*W) then transpose to (H*W, T)
    return frames.reshape(T, H * W).T


def matrix_to_frames(M, H, W):
    """Reshapes an (H * W, T) matrix back into a (T, H, W) video tensor."""
    # Transpose to (T, H*W) then reshape to (T, H, W)
    return M.T.reshape(-1, H, W)
