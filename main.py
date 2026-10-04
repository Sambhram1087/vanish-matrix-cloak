"""Integration script (owned by Member 4): runs the whole chain end to end."""
import numpy as np

from frames import synthetic_video, frames_to_matrix, matrix_to_frames
from svd_tools import svd_from_scratch
from rpca import rpca_ialm
from masks import clean_mask, composite


def run():
    frames, truth = synthetic_video()
    T, H, W = frames.shape
    M = frames_to_matrix(frames)
    U, s, Vt = svd_from_scratch(M)
    L, S, hist = rpca_ialm(M)
    S_fr = matrix_to_frames(S, H, W)
    L_fr = matrix_to_frames(L, H, W)
    masks = np.stack([clean_mask(S_fr[t]) for t in range(T)])
    out = np.stack([composite(frames[t], L_fr[t], masks[t]) for t in range(T)])
    print("pipeline ran:", out.shape)
    return out


if __name__ == "__main__":
    run()
