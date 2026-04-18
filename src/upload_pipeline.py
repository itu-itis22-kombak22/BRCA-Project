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
        stats = {
            "mean": float(tissue.mean()),
            "max": float(tissue.max()),
            "suspicious_ratio": float((tissue >= 0.5).mean()),
        }
    else:
        stats = {"mean": 0.0, "max": 0.0, "suspicious_ratio": 0.0}

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
                    alpha: int = 115) -> Image.Image:
    """Alpha-blend a jet-colored P(tumor) heatmap on top of the image."""
    import matplotlib.cm as cm

    base = image.convert("RGBA")
    w, h = base.size

    tissue = grid[mask]
    vmax = float(np.percentile(tissue, 99.5)) if tissue.size else 1.0
    vmax = max(vmax, 1e-3)

    smoothed = np.where(mask, grid, 0.0)
    if sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=sigma)

    normalised = np.clip(smoothed / vmax, 0.0, 1.0)
    jet = cm.get_cmap("jet")
    rgba = (jet(normalised) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)

    heat = Image.fromarray(rgba, mode="RGBA").resize(
        (w, h), Image.NEAREST)
    return Image.alpha_composite(base, heat).convert("RGB")
