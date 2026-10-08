"""MEMBER 4 — app.py: Streamlit UI for the Vanish Matrix Cloak pipeline.

Accepts any video file, runs SVD + Robust PCA to separate background (L) from
foreground (S), applies a tunable mask, and composites the result.

Troubleshooting invisibility cloak:
- Use "MOG2 mask" mode if RPCA mask misses your hand (common for large/slow objects)
- Lower "Mask sensitivity (mad_k)" to detect more foreground
- Increase "Max frames" and keep camera COMPLETELY still
- Start recording BEFORE putting your hand in frame (so RPCA sees the background first)
"""

import tempfile
import time
import os

import cv2
import numpy as np
import streamlit as st

from frames import frames_to_matrix, matrix_to_frames
from rpca import rpca_ialm
from masks import clean_mask, composite

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vanish · Matrix Cloak",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root{--bg0:#f8fafc;--surface:#ffffff;--glass:#ffffff;--accent:#4f46e5;--accent2:#0891b2;--accent3:#059669;--text:#0f172a;--muted:#64748b;--border:rgba(79,70,229,0.2);--border-light:#e2e8f0}

html,body,[data-testid="stAppViewContainer"]{background:var(--bg0)!important;font-family:'Inter',sans-serif;color:var(--text)}

[data-testid="stAppViewContainer"]::before{content:'';position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse 70% 50% at 0% 0%,rgba(79,70,229,.05) 0%,transparent 60%),
             radial-gradient(ellipse 50% 40% at 100% 5%,rgba(8,145,178,.04) 0%,transparent 55%),
             radial-gradient(ellipse 40% 30% at 50% 100%,rgba(5,150,105,.03) 0%,transparent 55%);
  pointer-events:none}

[data-testid="stAppViewContainer"]::after{content:'';position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(rgba(79,70,229,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(79,70,229,.025) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none}

[data-testid="stMain"],[data-testid="stMainBlockContainer"]{position:relative;z-index:1}

[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid var(--border-light)!important;box-shadow:2px 0 16px rgba(15,23,42,.06)}

.hero{text-align:center;padding:3rem 1rem 2rem;position:relative}
.hero::before{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);width:600px;height:200px;
  background:radial-gradient(ellipse,rgba(79,70,229,.07) 0%,transparent 70%);filter:blur(40px);pointer-events:none}
.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(79,70,229,.07);border:1px solid rgba(79,70,229,.2);
  border-radius:999px;padding:.25rem .85rem;font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4f46e5;margin-bottom:1.1rem}
.hero-badge span{width:6px;height:6px;background:#4f46e5;border-radius:50%;display:inline-block;animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,5vw,3.4rem);font-weight:700;letter-spacing:-.02em;line-height:1.1;margin:0 0 .8rem;
  background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 35%,#0891b2 70%,#059669 100%);background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer 6s linear infinite}
@keyframes shimmer{0%{background-position:0% 50%}100%{background-position:200% 50%}}
.hero p{color:var(--muted);font-size:1rem;letter-spacing:.01em;margin:0}
.hero-divider{width:80px;height:2px;background:linear-gradient(90deg,transparent,#4f46e5,#0891b2,transparent);margin:1.4rem auto 0;border-radius:999px}

.card{background:#ffffff;border:1px solid var(--border-light);border-radius:16px;
  padding:1.3rem 1.5rem;margin-bottom:1rem;box-shadow:0 1px 4px rgba(15,23,42,.06),0 4px 16px rgba(15,23,42,.04);transition:border-color .2s,box-shadow .2s}
.card:hover{border-color:rgba(79,70,229,.3);box-shadow:0 4px 20px rgba(79,70,229,.08)}
.card-title{font-family:'Space Grotesk',sans-serif;font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin-bottom:.6rem}

.warn-card{background:linear-gradient(135deg,rgba(254,243,199,1) 0%,rgba(255,251,235,1) 100%);
  border:1px solid rgba(217,119,6,.25);border-radius:16px;padding:1.1rem 1.4rem;margin-bottom:1.2rem;box-shadow:0 2px 12px rgba(245,158,11,.06)}
.warn-card ul{margin:.5rem 0 0 1.1rem;color:#92400e;font-size:.875rem;line-height:1.8}
.warn-card b{color:#b45309}

.stat-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}
.stat-pill{display:inline-flex;align-items:center;gap:.3rem;background:rgba(79,70,229,.07);border:1px solid rgba(79,70,229,.18);
  border-radius:999px;padding:.22rem .8rem;font-size:.78rem;font-weight:500;color:#4338ca;letter-spacing:.01em}
.stat-pill.green{background:rgba(5,150,105,.07);border-color:rgba(5,150,105,.2);color:#065f46}
.stat-pill.cyan{background:rgba(8,145,178,.07);border-color:rgba(8,145,178,.2);color:#155e75}
.stat-pill.orange{background:rgba(245,158,11,.07);border-color:rgba(245,158,11,.2);color:#92400e}
.stat-pill.red{background:rgba(220,38,38,.07);border-color:rgba(220,38,38,.2);color:#991b1b}

.panel-label{font-family:'Space Grotesk',sans-serif;font-size:.72rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.12em;color:var(--muted);text-align:center;margin-bottom:.4rem;
  padding:.25rem .6rem;background:#f1f5f9;border:1px solid var(--border-light);border-radius:8px}

.stProgress>div>div>div>div{background:linear-gradient(90deg,#4f46e5,#0891b2,#059669);background-size:200% 100%;
  animation:progressShimmer 2s linear infinite;border-radius:999px}
@keyframes progressShimmer{0%{background-position:0% 50%}100%{background-position:200% 50%}}

[data-testid="stFileUploader"]{background:#f8fafc!important;
  border:2px dashed rgba(79,70,229,.3)!important;border-radius:16px!important;transition:border-color .2s,box-shadow .2s!important}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;box-shadow:0 0 20px rgba(79,70,229,.1)!important}

.stButton>button{background:linear-gradient(135deg,#4f46e5 0%,#2563eb 50%,#0891b2 100%)!important;background-size:200% auto!important;
  color:#fff!important;border:none!important;border-radius:12px!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:600!important;font-size:.95rem!important;letter-spacing:.01em!important;padding:.6rem 2rem!important;
  transition:background-position .4s,transform .15s,box-shadow .2s!important;box-shadow:0 4px 16px rgba(79,70,229,.25)!important}
.stButton>button:hover{background-position:right center!important;transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(79,70,229,.4)!important}

h4{font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;letter-spacing:-.01em!important;color:var(--text)!important;margin-top:1.8rem!important}

details{background:#ffffff!important;border:1px solid var(--border-light)!important;border-radius:12px!important;padding:.2rem .5rem!important;box-shadow:0 1px 4px rgba(15,23,42,.05)!important}

::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#f1f5f9}::-webkit-scrollbar-thumb{background:rgba(79,70,229,.3);border-radius:999px}

/* ─── Sidebar: force all text dark on white background ─── */
[data-testid="stSidebar"] *{color:#0f172a !important}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h3,[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] .stMarkdown b{color:#0f172a !important}
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stSelectSlider"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label{color:#0f172a !important;font-weight:500}
[data-testid="stSidebar"] [role="radiogroup"] label{color:#1e293b !important}
[data-testid="stSidebar"] small,[data-testid="stSidebar"] small *{color:#475569 !important}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"]{color:#64748b !important}
[data-testid="stSidebar"] hr{border-color:#e2e8f0 !important}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#0f172a !important}

#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span></span>SVD &nbsp;·&nbsp; RPCA &nbsp;·&nbsp; Computer Vision</div>
    <h1>👁️ Vanish · Matrix Cloak</h1>
    <p>Background / Foreground separation powered by Robust PCA &amp; Median subtraction</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline settings")

    st.markdown("**📹 Video loading**")
    max_frames = st.slider("Max frames", 20, 150, 60, 5,
        help="More frames → better background estimation. Use ≥60 for real videos.")
    resize_h = st.slider("Resize height (px)", 48, 240, 96, 8,
        help="Larger = more detail but slower. 96–120 px is a good balance.")

    st.markdown("---")
    st.markdown("**🧮 RPCA**")
    tol = st.select_slider("Convergence tolerance",
        options=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3], value=1e-2,
        format_func=lambda v: f"{v:.0e}")
    max_iter = st.slider("Max iterations", 20, 300, 150, 10)

    st.markdown("---")
    st.markdown("**🎭 Mask mode**")
    mask_mode = st.radio(
        "Foreground detection method",
        ["RPCA sparse (S)", "Median background subtractor", "RPCA + Median (union)"],
        index=2,
        help=(
            "**RPCA sparse**: uses the S matrix from RPCA. Works when the hand is small "
            "and moves quickly.\n\n"
            "**Median**: Calculates the median frame over time and subtracts it. "
            "Extremely robust for static cameras and large/slow moving objects.\n\n"
            "**Union**: OR of both — catches the most foreground."
        ),
    )

    st.markdown("---")
    st.markdown("**🎚 Mask tuning**")
    mad_k = st.slider(
        "Sensitivity (mad_k)",
        0.5, 8.0, 2.0, 0.5,
        help=(
            "Lower = more pixels flagged as foreground (hand more likely to vanish). "
            "Original hard-coded value was 8.0 — far too conservative for real video."
        ),
    )
    peak_frac = st.slider("Peak fraction floor", 0.01, 0.20, 0.04, 0.01,
        help="Threshold ≥ peak_frac × max(|S|). Lower = more foreground.")
    dilate_px = st.slider("Mask dilation (px)", 1, 15, 7, 1,
        help="Grow the mask outward to fully cover object edges.")

    st.markdown("---")
    colormap = st.selectbox("Foreground colormap",
        ["hot", "plasma", "viridis", "magma", "inferno"], index=0)
    show_mask_debug = st.checkbox("Show mask debug panel", value=True,
        help="Adds a 5th panel showing the raw binary mask.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>UE25MA242A — MFAD 2026<br>"
        "Rahul P · Rohith M · Sambhram S · Banuteja R</small>",
        unsafe_allow_html=True,
    )

# ─── Tips banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="warn-card">
    <b style="color:#92400e">💡 For best invisibility-cloak results:</b>
    <ul>
        <li>Keep your camera <b>completely still</b> — any camera shake breaks the background estimation</li>
        <li>Record 1–2 seconds of <b>empty background</b> before moving your hand into frame</li>
        <li>Use <b>Median + RPCA union</b> mask mode (sidebar) for large/slow objects</li>
        <li>Increase <b>Max frames</b> to 80–120 so RPCA has more background data</li>
        <li>Lower <b>Sensitivity (mad_k)</b> if the mask is missing your hand</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
COLORMAPS = {
    "hot":    cv2.COLORMAP_HOT,   "plasma":  cv2.COLORMAP_PLASMA,
    "viridis":cv2.COLORMAP_VIRIDIS,"magma":  cv2.COLORMAP_MAGMA,
    "inferno":cv2.COLORMAP_INFERNO,
}


def load_video_frames(path: str, max_frames: int, target_h: int):
    """Read video, convert to grayscale, resize → (T, H, W) float + fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames, frame_idx = [], 0
    while len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            new_w = int(w * target_h / h)
            gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(float))
        frame_idx += 1

    cap.release()
    return np.stack(frames), fps


def build_median_masks(frames: np.ndarray, thresh: float = 20.0) -> np.ndarray:
    """Robust background subtraction using the median over time.
    Returns (T, H, W) bool masks where pixel > thresh from median.
    """
    T, H, W = frames.shape
    median_bg = np.median(frames, axis=0)
    
    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for t in range(T):
        diff = np.abs(frames[t] - median_bg)
        mask = (diff > thresh).astype(np.uint8) * 255
        
        # Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        masks.append(mask > 127)
        
    return np.stack(masks)


def apply_colormap(frame_abs: np.ndarray, cmap_name: str) -> np.ndarray:
    norm = cv2.normalize(frame_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr  = cv2.applyColorMap(norm, COLORMAPS[cmap_name])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frames_to_gif_bytes(rgb_frames, fps: float = 10.0) -> bytes:
    """Encode a list of (H, W, 3) RGB uint8 frames to an animated GIF using Pillow.
    GIFs are universally supported in browsers via st.image().
    """
    from PIL import Image
    import io

    duration_ms = int(1000 / max(fps, 1))
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in rgb_frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=pil_frames[1:],
        loop=0, duration=duration_ms, optimize=False
    )
    return buf.getvalue()


def frames_to_mp4_bytes(rgb_frames, fps: float = 10.0) -> bytes:
    """Encode a list of (H, W, 3) RGB uint8 frames to an MP4 video."""
    import tempfile
    import cv2
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
        
    H, W, _ = rgb_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
    
    for frame in rgb_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()
    
    with open(tmp_path, 'rb') as f:
        data = f.read()
        
    os.unlink(tmp_path)
    return data


def gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


# ─── Upload ──────────────────────────────────────────────────────────────────
st.markdown("#### 📂 Upload your video")
uploaded = st.file_uploader(
    "Drag & drop or browse — MP4, AVI, MOV, MKV, WebM, FLV, 3GP …",
    type=["mp4", "avi", "mov", "mkv", "webm", "m4v", "wmv", "flv", "3gp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2.5rem;">
        <div style="font-size:3rem;margin-bottom:.75rem">🎬</div>
        <div style="color:#94a3b8;font-size:1rem">
            Upload a video above to run the<br>
            <strong style="color:#a78bfa">SVD + Robust PCA</strong> invisibility-cloak pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("▶ Run Pipeline", use_container_width=True)
with col_info:
    st.markdown(
        f"<div style='padding:.5rem 0;color:#64748b;font-size:.9rem'>"
        f"<b style='color:#e2e8f0'>{uploaded.name}</b> &nbsp;·&nbsp; "
        f"{uploaded.size/1_000_000:.2f} MB &nbsp;·&nbsp; "
        f"max <b style='color:#a78bfa'>{max_frames}</b> frames &nbsp;·&nbsp; "
        f"resize <b style='color:#22d3ee'>{resize_h}px</b> &nbsp;·&nbsp; "
        f"mask: <b style='color:#fbbf24'>{mask_mode}</b>"
        f"</div>", unsafe_allow_html=True,
    )

if not run:
    st.stop()

# ─── Pipeline ────────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="Saving upload …")

try:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    progress_bar.progress(8, text="Loading frames …")
    t0 = time.perf_counter()

    frames, fps = load_video_frames(tmp_path, max_frames, resize_h)
    T, H, W = frames.shape

    progress_bar.progress(20, text=f"Loaded {T} frames ({H}×{W}) — running RPCA …")

    M = frames_to_matrix(frames)
    L, S, hist = rpca_ialm(M, tol=tol, max_iter=max_iter)

    progress_bar.progress(70, text="Reconstructing frames …")
    S_fr = matrix_to_frames(S, H, W)
    L_fr = matrix_to_frames(L, H, W)

    # ── Mask computation ───────────────────────────────────────────────────
    progress_bar.progress(78, text="Computing masks …")

    rpca_masks = np.stack([
        clean_mask(S_fr[t], mad_k=mad_k, peak_frac=peak_frac, dilate_px=dilate_px)
        for t in range(T)
    ])

    if "Median" in mask_mode:
        progress_bar.progress(84, text="Running Median background subtractor …")
        median_masks = build_median_masks(frames)
        
        if mask_mode == "Median background subtractor":
            final_masks = median_masks
        else:  # union
            final_masks = rpca_masks | median_masks
    else:
        final_masks = rpca_masks

    progress_bar.progress(90, text="Compositing …")
    out = np.stack([composite(frames[t], L_fr[t], final_masks[t]) for t in range(T)])

    elapsed = time.perf_counter() - t0
    progress_bar.progress(100, text="Done ✓")
    os.unlink(tmp_path)

except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.stop()

# ─── Stats ───────────────────────────────────────────────────────────────────
final_rank     = hist["rank"][-1]     if hist["rank"]     else "—"
final_sparsity = hist["sparsity"][-1] if hist["sparsity"] else 0.0
n_iter         = len(hist["error"])
avg_mask_cov   = float(np.mean(final_masks))

st.markdown(f"""
<div class="card">
    <div class="card-title">📊 Pipeline results</div>
    <div class="stat-row">
        <span class="stat-pill">⏱ {elapsed:.1f}s</span>
        <span class="stat-pill cyan">🎞 {T} frames · {H}×{W}</span>
        <span class="stat-pill green">🔢 rank(L) = {final_rank}</span>
        <span class="stat-pill orange">✨ sparsity(S) = {final_sparsity:.1%}</span>
        <span class="stat-pill">🔄 {n_iter} RPCA iters</span>
        <span class="stat-pill {'red' if avg_mask_cov < 0.01 else 'green'}">🎭 mask coverage = {avg_mask_cov:.1%}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if avg_mask_cov < 0.005:
    st.warning(
        "⚠️ **Mask coverage is near zero** — your hand is not being detected as foreground. "
        "Try: switching to **RPCA + Median (union)** mask mode, lowering **mad_k** below 2.0, "
        "or increasing **Max frames** so more background frames are captured."
    )

# ─── Frame slider ────────────────────────────────────────────────────────────
st.markdown("#### 🎞 Frame explorer")
frame_idx = st.slider("Frame", 0, T - 1, min(T - 1, T // 2), 1, label_visibility="collapsed")

orig_disp = frames[frame_idx].astype(np.uint8)
bg_disp   = np.clip(L_fr[frame_idx], 0, 255).astype(np.uint8)
fg_disp   = apply_colormap(np.abs(S_fr[frame_idx]), colormap)
out_disp  = np.clip(out[frame_idx], 0, 255).astype(np.uint8)
mask_disp = (final_masks[frame_idx].astype(np.uint8) * 255)

n_panels = 5 if show_mask_debug else 4
cols = st.columns(n_panels)
panel_data = [
    ("Original",        gray_to_rgb(orig_disp)),
    ("Background (L)",  gray_to_rgb(bg_disp)),
    ("Foreground (S)",  fg_disp),
    ("Composite (out)", gray_to_rgb(out_disp)),
]
if show_mask_debug:
    panel_data.append(("Mask (binary)", gray_to_rgb(mask_disp)))

for col, (label, img) in zip(cols, panel_data):
    with col:
        st.markdown(f'<div class="panel-label">{label}</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

# ─── Animated Previews ───────────────────────────────────────────────────────────
st.markdown("#### 🎬 Animated preview")

anim_labels = ["Original", "Background", "Foreground", "Composite"]
with st.spinner("Encoding animated previews …"):
    orig_rgb = [gray_to_rgb(frames[t].astype(np.uint8)) for t in range(T)]
    bg_rgb   = [gray_to_rgb(np.clip(L_fr[t], 0, 255).astype(np.uint8)) for t in range(T)]
    fg_rgb   = [apply_colormap(np.abs(S_fr[t]), colormap) for t in range(T)]
    out_rgb  = [gray_to_rgb(np.clip(out[t], 0, 255).astype(np.uint8)) for t in range(T)]
    gif_fps  = min(fps, 12.0)
    gifs = [frames_to_gif_bytes(seq, gif_fps) for seq in [orig_rgb, bg_rgb, fg_rgb, out_rgb]]

acols = st.columns(4)
for col, label, gif in zip(acols, anim_labels, gifs):
    with col:
        st.markdown(f'<div class="panel-label">{label}</div>', unsafe_allow_html=True)
        st.image(gif, use_container_width=True)

# ─── RPCA convergence ────────────────────────────────────────────────────────
with st.expander("📈 RPCA convergence history", expanded=False):
    import pandas as pd
    df = pd.DataFrame(hist)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Relative error ‖M-L-S‖/‖M‖**")
        st.line_chart(df["error"], color="#7c3aed")
    with c2:
        st.markdown("**Rank of L**")
        st.line_chart(df["rank"], color="#06b6d4")
    with c3:
        st.markdown("**Sparsity of S**")
        st.line_chart(df["sparsity"], color="#10b981")

# ─── Downloads ───────────────────────────────────────────────────────────────
st.markdown("#### 💾 Downloads")
with st.spinner("Preparing MP4 downloads ..."):
    vid_fps = min(fps, 30.0)
    mp4s = [frames_to_mp4_bytes(seq, vid_fps) for seq in [orig_rgb, bg_rgb, fg_rgb, out_rgb]]

dcols = st.columns(4)
fnames = ["original.mp4", "background.mp4", "foreground.mp4", "composite.mp4"]
for col, label, mp4, fname in zip(dcols, anim_labels, mp4s, fnames):
    with col:
        st.download_button(f"⬇ {label}", data=mp4, file_name=fname,
                           mime="video/mp4", use_container_width=True)
