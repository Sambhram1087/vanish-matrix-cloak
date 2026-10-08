"""Script to generate sample images and GIF for the samples/ directory."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from frames import synthetic_video, frames_to_matrix, matrix_to_frames
from rpca import rpca_ialm
from masks import clean_mask, composite

OUT = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT, exist_ok=True)

print("Generating synthetic video …")
frames, truth = synthetic_video(num_frames=40, height=64, width=96)
T, H, W = frames.shape
M = frames_to_matrix(frames)

print("Running Robust PCA (IALM) …")
L, S, hist = rpca_ialm(M, tol=1e-3, max_iter=200)

S_fr = matrix_to_frames(S, H, W)
L_fr = matrix_to_frames(L, H, W)

masks = np.stack([clean_mask(S_fr[t], mad_k=2.0, peak_frac=0.04, dilate_px=5) for t in range(T)])
out   = np.stack([composite(frames[t], L_fr[t], masks[t]) for t in range(T)])

# ── 1. Static 4-panel sample frame (mid-point) ──────────────────────────────
mid = T // 2
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#f8fafc")
fig.suptitle("Vanish · Matrix Cloak — Pipeline Output (Synthetic Video)", fontsize=12,
             fontweight="bold", color="#0f172a", y=1.02)

panel_data = [
    ("Original Frame",    frames[mid],        "gray",  0,   255),
    ("Background (L)",    L_fr[mid],           "gray",  0,   255),
    ("Foreground |S|",    np.abs(S_fr[mid]),   "hot",   None, None),
    ("Composite (out)",   out[mid],            "gray",  0,   255),
]

for ax, (title, img, cmap, vmin, vmax) in zip(axes, panel_data):
    ax.imshow(np.clip(img, 0, 255).astype(np.uint8) if vmin is not None else img,
              cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="600", color="#0f172a", pad=5)
    ax.axis("off")

plt.tight_layout()
path_panel = os.path.join(OUT, "pipeline_output.png")
fig.savefig(path_panel, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path_panel}")

# ── 2. RPCA convergence plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor="#f8fafc")
fig.suptitle("RPCA Convergence History", fontsize=11, fontweight="bold", color="#0f172a")

plots = [
    ("Relative Error ‖M−L−S‖/‖M‖", hist["error"],    "#4f46e5"),
    ("Rank of L",                    hist["rank"],     "#0891b2"),
    ("Sparsity of S",                hist["sparsity"], "#059669"),
]
for ax, (title, data, color) in zip(axes, plots):
    ax.plot(data, color=color, linewidth=2)
    ax.set_title(title, fontsize=9, fontweight="600", color="#0f172a")
    ax.set_xlabel("Iteration", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")

plt.tight_layout()
path_conv = os.path.join(OUT, "rpca_convergence.png")
fig.savefig(path_conv, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path_conv}")

# ── 3. Mask evolution strip (every 8th frame) ────────────────────────────────
step = max(1, T // 8)
sample_frames = list(range(0, T, step))[:8]
fig, axes = plt.subplots(2, len(sample_frames), figsize=(len(sample_frames) * 2, 4.5),
                         facecolor="#f8fafc")
fig.suptitle("Mask Evolution Across Frames", fontsize=11, fontweight="bold", color="#0f172a")

for col, t in enumerate(sample_frames):
    axes[0, col].imshow(np.clip(frames[t], 0, 255).astype(np.uint8), cmap="gray",
                        vmin=0, vmax=255, interpolation="nearest")
    axes[0, col].set_title(f"t={t}", fontsize=7, color="#64748b")
    axes[0, col].axis("off")

    axes[1, col].imshow(masks[t].astype(np.uint8) * 255, cmap="hot",
                        vmin=0, vmax=255, interpolation="nearest")
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=8, color="#0f172a", labelpad=4)
axes[1, 0].set_ylabel("Mask",     fontsize=8, color="#0f172a", labelpad=4)
plt.tight_layout()
path_masks = os.path.join(OUT, "mask_evolution.png")
fig.savefig(path_masks, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path_masks}")

# ── 4. Animated GIF of the composite output ──────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 2.8), facecolor="#0f172a")
ax.axis("off")
im = ax.imshow(np.clip(out[0], 0, 255).astype(np.uint8), cmap="gray",
               vmin=0, vmax=255, animated=True, interpolation="nearest")
ax.set_title("Composite (invisibility cloak)", fontsize=8, color="#e2e8f0", pad=4)
fig.tight_layout(pad=0.4)

def update_gif(t):
    im.set_array(np.clip(out[t], 0, 255).astype(np.uint8))
    return [im]

ani = animation.FuncAnimation(fig, update_gif, frames=T, interval=80, blit=True)
path_gif = os.path.join(OUT, "composite_preview.gif")
ani.save(path_gif, writer="pillow", fps=12, dpi=100)
plt.close(fig)
print(f"  Saved: {path_gif}")

# ── 5. SVD singular value spectrum ──────────────────────────────────────────
norm_M = np.linalg.norm(M, "fro")
s_all = np.linalg.svd(M, compute_uv=False)
energy = np.cumsum(s_all**2) / np.sum(s_all**2)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#f8fafc")
fig.suptitle("SVD Singular Value Analysis", fontsize=11, fontweight="bold", color="#0f172a")

axes[0].bar(range(1, len(s_all)+1), s_all, color="#4f46e5", alpha=0.8, width=0.8)
axes[0].set_title("Singular Value Spectrum", fontsize=9, fontweight="600", color="#0f172a")
axes[0].set_xlabel("Rank index", fontsize=8)
axes[0].set_ylabel("σᵢ", fontsize=9)
axes[0].set_xlim(0.5, min(20, len(s_all)) + 0.5)
axes[0].set_facecolor("#ffffff")
for spine in axes[0].spines.values():
    spine.set_color("#e2e8f0")

axes[1].plot(range(1, len(energy)+1), energy * 100, color="#059669", linewidth=2)
axes[1].axhline(95, color="#ef4444", linestyle="--", linewidth=1, label="95% energy")
axes[1].axhline(99, color="#f59e0b", linestyle="--", linewidth=1, label="99% energy")
axes[1].set_title("Cumulative Energy (σᵢ²)", fontsize=9, fontweight="600", color="#0f172a")
axes[1].set_xlabel("Rank k", fontsize=8)
axes[1].set_ylabel("% energy captured", fontsize=8)
axes[1].legend(fontsize=8)
axes[1].set_facecolor("#ffffff")
for spine in axes[1].spines.values():
    spine.set_color("#e2e8f0")

plt.tight_layout()
path_svd = os.path.join(OUT, "svd_spectrum.png")
fig.savefig(path_svd, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved: {path_svd}")

print("\nAll samples generated successfully.")
print(f"Files in samples/: {sorted(os.listdir(OUT))}")
