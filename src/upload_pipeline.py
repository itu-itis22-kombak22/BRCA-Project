"""Extract images from a user upload (PNG/JPG/TIF/ZIP/PDF), tile them into
96×96 patches, score them with the PCam classifier, and produce heatmap
overlays + aggregate statistics.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.inference import PCAM_PATCH, predict_batch


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# File -> list of PIL images
# ---------------------------------------------------------------------------

def extract_images(name: str, data: bytes) -> list[tuple[str, Image.Image]]:
    """Return ``[(label, image), ...]`` from an uploaded file.

    ``name`` is the original filename (used to detect the format), ``data``
    is the raw bytes. ZIP members are enumerated; PDF pages are rendered
    at ~150 DPI; single images are returned as one item.
    """
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [(name, _load_image(data))]
    if suffix == ".zip":
        return _extract_zip(data)
    if suffix == ".pdf":
        return _extract_pdf(data)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _extract_zip(data: bytes) -> list[tuple[str, Image.Image]]:
    out: list[tuple[str, Image.Image]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                out.append((info.filename, _load_image(z.read(info))))
            except Exception:
                continue
    return out


def _extract_pdf(data: bytes, dpi: int = 150) -> list[tuple[str, Image.Image]]:
    import fitz  # pymupdf, imported lazily
    out: list[tuple[str, Image.Image]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append((f"page_{i + 1}", img))
    return out


# ---------------------------------------------------------------------------
# Image -> patches + scores
# ---------------------------------------------------------------------------

def _is_tissue(arr: np.ndarray, bg_threshold: float) -> bool:
    return float(arr.mean()) < bg_threshold


def score_image(model,
                image: Image.Image,
                *,
                device=None,
                patch_size: int = PCAM_PATCH,
                stride: int | None = None,
                bg_threshold: float = 220.0,
                batch_size: int = 128) -> dict:
    """Tile an image, score every tissue-rich patch, return stats + grid.

    Tiles smaller than ``patch_size`` are center-cropped / padded and
    treated as a single patch. For large images the tiles form a grid and
    the returned overlay shows a smoothed P(tumor) heatmap.
    """
    stride = stride or patch_size
    w, h = image.size

    # Tiny image: treat as a single patch (resize to PCam scale).
    if min(w, h) < patch_size:
        square = min(w, h)
        img_sq = image.crop(((w - square) // 2, (h - square) // 2,
                             (w + square) // 2, (h + square) // 2))
        img_sq = img_sq.resize((patch_size, patch_size), Image.LANCZOS)
        prob = float(predict_batch(model, [img_sq], device=device,
                                   batch_size=batch_size)[0])
        return {
            "mode": "single",
            "n_tissue": 1,
            "n_total": 1,
            "mean": prob,
            "max": prob,
            "suspicious_ratio": float(prob >= 0.5),
            "grid": np.array([[prob]], dtype=np.float32),
            "mask": np.array([[True]]),
            "overlay": img_sq,
        }

    n_cols = (w - patch_size) // stride + 1
    n_rows = (h - patch_size) // stride + 1
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    buf_imgs: list[Image.Image] = []
    buf_idx: list[tuple[int, int]] = []
    n_total = 0

    def _flush():
        if not buf_imgs:
            return
        probs = predict_batch(model, buf_imgs, device=device,
                              batch_size=batch_size)
        for (r, c), p in zip(buf_idx, probs):
            grid[r, c] = p
            mask[r, c] = True
        buf_imgs.clear()
        buf_idx.clear()

    for r in range(n_rows):
        for c in range(n_cols):
            x, y = c * stride, r * stride
            tile = image.crop((x, y, x + patch_size, y + patch_size))
            n_total += 1
            arr = np.asarray(tile)
            if not _is_tissue(arr, bg_threshold):
                continue
            buf_imgs.append(tile)
            buf_idx.append((r, c))
            if len(buf_imgs) >= batch_size:
                _flush()
    _flush()

    tissue = grid[mask]
    n_tissue = int(tissue.size)
    if n_tissue:
        median = float(np.median(tissue))
        # Relative detection: patches whose probability is notably above
        # the image's own median. Robust to the domain-shifted-model
        # tendency to push all absolute probabilities toward zero.
        rel_thresh = max(median + 0.02, median * 3.0, 0.05)
        rel_ratio = float((tissue >= rel_thresh).mean())
        stats = {
            "mean": float(tissue.mean()),
            "median": median,
            "max": float(tissue.max()),
            "suspicious_ratio": float((tissue >= 0.5).mean()),
            "relative_ratio": rel_ratio,
            "relative_threshold": rel_thresh,
        }
    else:
        stats = {"mean": 0.0, "median": 0.0, "max": 0.0,
                 "suspicious_ratio": 0.0,
                 "relative_ratio": 0.0, "relative_threshold": 0.0}

    overlay = _render_overlay(image, grid, mask)

    return {
        "mode": "grid",
        "n_tissue": n_tissue,
        "n_total": n_total,
        **stats,
        "grid": grid,
        "mask": mask,
        "overlay": overlay,
    }


def _render_overlay(image: Image.Image,
                    grid: np.ndarray,
                    mask: np.ndarray,
                    sigma: float = 1.0,
                    alpha: float = 0.45) -> Image.Image:
    """Render a heatmap on top of the image with a matplotlib colorbar.

    ``jet`` colormap: dark blue → cyan → green → yellow → **red**.
    Red means high predicted P(tumor). Auto-scales to the 99.5th
    percentile of tissue-patch probabilities, because absolute values
    under a domain-shifted PCam model can be systematically low.
    """
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = image.size

    tissue = grid[mask]
    vmax = float(np.percentile(tissue, 99.5)) if tissue.size else 1.0
    vmax = max(vmax, 1e-3)

    smoothed = np.where(mask, grid, 0.0)
    if sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=sigma)

    fig_w_in = 8.0
    fig_h_in = fig_w_in * h / w
    fig = plt.figure(figsize=(fig_w_in + 1.2, fig_h_in),
                     facecolor="#0D1117")
    gs = fig.add_gridspec(1, 2, width_ratios=[fig_w_in, 0.5],
                          wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    heat = np.ma.array(smoothed, mask=~mask)
    ax.imshow(heat, extent=(0, w, h, 0), cmap="jet",
              vmin=0.0, vmax=vmax, alpha=alpha,
              interpolation="bilinear")
    ax.set_axis_off()

    cax = fig.add_subplot(gs[0, 1])
    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("P(tumor) — red = high, blue = low",
                   color="#E6EDF3", fontsize=9)
    cbar.ax.tick_params(colors="#8B949E", labelsize=8)
    cbar.outline.set_edgecolor("#21262D")

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0.05, facecolor="#0D1117", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
