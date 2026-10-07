"""Integration script (owned by Member 4): runs the whole chain end to end and visualizes it."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from frames import synthetic_video, frames_to_matrix, matrix_to_frames
from svd_tools import svd_from_scratch
from rpca import rpca_ialm
from masks import clean_mask, composite


def run():
    print("Running pipeline...")
    frames, truth = synthetic_video()
    T, H, W = frames.shape
    M = frames_to_matrix(frames)
    
    # Core algorithms
    U, s, Vt = svd_from_scratch(M)
    L, S, hist = rpca_ialm(M)
    
    S_fr = matrix_to_frames(S, H, W)
    L_fr = matrix_to_frames(L, H, W)
    
    masks = np.stack([clean_mask(S_fr[t]) for t in range(T)])
    out = np.stack([composite(frames[t], L_fr[t], masks[t]) for t in range(T)])
    print("Pipeline ran successfully:", out.shape)
    
    # --- Visualization Setup ---
    print("Opening video animation window...")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    titles = ["Original Frame", "Background (L)", "Foreground (S)", "Composite (Out)"]
    
    # Initialize plots
    ims = []
    ims.append(axes[0].imshow(frames[0], cmap='gray', vmin=0, vmax=255))
    ims.append(axes[1].imshow(L_fr[0], cmap='gray', vmin=0, vmax=255))
    ims.append(axes[2].imshow(np.abs(S_fr[0]), cmap='hot'))
    ims.append(axes[3].imshow(out[0], cmap='gray', vmin=0, vmax=255))
    
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    # Animation update function
    def update(t):
        ims[0].set_array(frames[t])
        ims[1].set_array(L_fr[t])
        ims[2].set_array(np.abs(S_fr[t]))
        ims[3].set_array(out[t])
        return ims

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return out


if __name__ == "__main__":
    run()