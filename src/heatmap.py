"""WSI sliding-window tumor probability heatmap + thumbnail overlay.

Resolution matching
-------------------
PCam patches are 96×96 at ~0.972 µm/px (~10×). TCGA-BRCA SVS files are
typically 0.25 µm/px (40×) at level 0 with a 3-level pyramid (×1, ×4, ×16),
which puts **level 1 at ~0.994 µm/px** — essentially the PCam scale. We
pick whichever pyramid level is closest to PCam's mpp and read 96×96
patches there directly, so no resampling is required in the common case.

Background filtering uses the mean-intensity heuristic suggested in the
handoff (``patch.mean() > bg_threshold`` → skip).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.inference import PCAM_PATCH, _device, load_model, predict_batch


PCAM_MPP = 0.972  # µm/px, approximate scale of Kaggle Histopath patches


def pick_level(slide: openslide.OpenSlide,
               target_mpp: float = PCAM_MPP) -> tuple[int, float, float]:
    """Choose the pyramid level whose mpp best matches ``target_mpp``.

    Returns ``(level, level_mpp, scale)`` where ``scale`` is the factor to
    resize a patch read at this level so that it matches ``target_mpp``.
    A scale close to 1.0 means no resampling is needed.
    """
    mpp0 = float(slide.properties.get("openslide.mpp-x")
                 or slide.properties.get("openslide.mpp-y")
                 or 0.0)
    if mpp0 <= 0:
        raise RuntimeError("Slide does not report openslide.mpp-x/mpp-y; "
                           "cannot match PCam magnification.")

    best_level, best_err = 0, float("inf")
    for lvl in range(slide.level_count):
        lvl_mpp = mpp0 * slide.level_downsamples[lvl]
        err = abs(np.log2(lvl_mpp / target_mpp))
        if err < best_err:
            best_err, best_level = err, lvl
    level_mpp = mpp0 * slide.level_downsamples[best_level]
    scale = level_mpp / target_mpp
    return best_level, level_mpp, scale


def is_tissue(patch_rgb: np.ndarray, bg_threshold: float = 220.0) -> bool:
    """Simple background filter: skip near-white patches."""
    return float(patch_rgb.mean()) < bg_threshold


def iter_patches(slide: openslide.OpenSlide,
                 level: int,
                 patch_size: int,
                 stride: int) -> Iterator[tuple[int, int, Image.Image]]:
    """Yield ``(x0, y0, patch)`` where ``(x0, y0)`` is the top-left in
    **level-0** coordinates (what ``read_region`` expects)."""
    w, h = slide.level_dimensions[level]
    downsample = int(round(slide.level_downsamples[level]))
    stride_px0 = stride * downsample
    patch_px0 = patch_size * downsample

    x_lvl = 0
    while x_lvl + patch_size <= w:
        y_lvl = 0
        while y_lvl + patch_size <= h:
            x0 = x_lvl * downsample
            y0 = y_lvl * downsample
            tile = slide.read_region((x0, y0), level,
                                     (patch_size, patch_size)).convert("RGB")
            yield x0, y0, tile
            y_lvl += stride
        x_lvl += stride


def build_heatmap(slide_path: str | Path,
                  weights_path: str | Path,
                  patch_size: int = PCAM_PATCH,
                  stride: int | None = None,
                  batch_size: int = 128,
                  bg_threshold: float = 220.0) -> dict:
    """Score every tissue-rich patch on the best-matching pyramid level."""
    stride = stride or patch_size  # non-overlapping by default

    slide = openslide.OpenSlide(str(slide_path))
    try:
        level, level_mpp, scale = pick_level(slide)
        w_lvl, h_lvl = slide.level_dimensions[level]
        downsample = int(round(slide.level_downsamples[level]))

        n_cols = w_lvl // stride
        n_rows = h_lvl // stride

        device = _device()
        model = load_model(weights_path, device=device)
        print(f"[heatmap] level={level}  level_mpp={level_mpp:.4f} "
              f"scale={scale:.3f}  grid={n_cols}×{n_rows}  device={device}")

        grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        mask = np.zeros((n_rows, n_cols), dtype=bool)

        buf_imgs: list[Image.Image] = []
        buf_idx: list[tuple[int, int]] = []

        def _flush():
            if not buf_imgs:
                return
            imgs = buf_imgs
            if abs(scale - 1.0) > 0.05:
                target = patch_size  # always score at PCam native size
                imgs = [im.resize((target, target), Image.LANCZOS)
                        for im in buf_imgs]
            probs = predict_batch(model, imgs, device=device,
                                  batch_size=batch_size)
            for (r, c), p in zip(buf_idx, probs):
                grid[r, c] = p
                mask[r, c] = True
            buf_imgs.clear()
            buf_idx.clear()

        total = n_rows * n_cols
        pbar = tqdm(total=total, desc="patches")
        for x_lvl in range(0, n_cols * stride, stride):
            for y_lvl in range(0, n_rows * stride, stride):
                x0, y0 = x_lvl * downsample, y_lvl * downsample
                tile = slide.read_region((x0, y0), level,
                                         (patch_size, patch_size)).convert("RGB")
                arr = np.asarray(tile)
                pbar.update(1)
                if not is_tissue(arr, bg_threshold=bg_threshold):
                    continue
                buf_imgs.append(tile)
                buf_idx.append((y_lvl // stride, x_lvl // stride))
                if len(buf_imgs) >= batch_size:
                    _flush()
        _flush()
        pbar.close()

        return {
            "grid": grid,
            "mask": mask,
            "level": level,
            "level_mpp": level_mpp,
            "downsample": downsample,
            "stride": stride,
            "patch_size": patch_size,
            "slide_dims_l0": slide.dimensions,
            "slide_dims_lvl": (w_lvl, h_lvl),
        }
    finally:
        slide.close()


def render_overlay(slide_path: str | Path,
                   result: dict,
                   out_path: str | Path,
                   thumbnail_max: int = 2048,
                   alpha: float = 0.45,
                   cmap: str = "jet",
                   sigma: float = 1.5,
                   vmax: float | None = None) -> Path:
    """Render the heatmap on top of a slide thumbnail.

    The PCam classifier was trained on lymph-node metastases, so on a
    primary-tumour TCGA-BRCA slide its absolute probabilities are heavily
    compressed near 0. ``vmax=None`` auto-scales to the 99.5th percentile
    of scored patches so that spatial structure remains visible.
    """
    slide = openslide.OpenSlide(str(slide_path))
    try:
        w0, h0 = slide.dimensions
        sc = thumbnail_max / max(w0, h0)
        tw, th = int(w0 * sc), int(h0 * sc)
        thumb = slide.get_thumbnail((tw, th))
    finally:
        slide.close()

    grid = result["grid"]
    mask = result["mask"]

    smoothed = np.where(mask, grid, 0.0)
    if sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=sigma)

    tissue_probs = grid[mask]
    if vmax is None:
        vmax = max(float(np.percentile(tissue_probs, 99.5))
                   if tissue_probs.size else 1.0,
                   1e-3)

    fig, ax = plt.subplots(figsize=(10, 10 * th / tw))
    ax.imshow(thumb)
    heat = np.ma.array(smoothed, mask=~mask)
    ax.imshow(heat,
              extent=(0, tw, th, 0),
              cmap=cmap, vmin=0.0, vmax=vmax, alpha=alpha)
    ax.set_axis_off()

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, label=f"P(tumor) (vmax={vmax:.3f})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="WSI tumor probability heatmap.")
    p.add_argument("--slide", required=True)
    p.add_argument("--weights", default="models/resnet18_pcam.pth")
    p.add_argument("--out", default="outputs/")
    p.add_argument("--patch-size", type=int, default=PCAM_PATCH)
    p.add_argument("--stride", type=int, default=PCAM_PATCH)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bg-threshold", type=float, default=220.0)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--sigma", type=float, default=1.5,
                   help="Gaussian smoothing sigma for overlay (0 to disable).")
    p.add_argument("--vmax", type=float, default=None,
                   help="Heatmap upper colour bound; default = p99.5 (auto).")
    args = p.parse_args()

    result = build_heatmap(args.slide, args.weights,
                           patch_size=args.patch_size,
                           stride=args.stride,
                           batch_size=args.batch_size,
                           bg_threshold=args.bg_threshold)

    out_dir = Path(args.out)
    slide_stem = Path(args.slide).stem
    fig_path = out_dir / f"{slide_stem}_heatmap.png"
    render_overlay(args.slide, result, fig_path,
                   alpha=args.alpha, sigma=args.sigma, vmax=args.vmax)

    grid_path = out_dir / f"{slide_stem}_heatmap_grid.npy"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grid_path, result["grid"])

    tissue_probs = result["grid"][result["mask"]]
    print(f"[heatmap] scored {tissue_probs.size} tissue patches")
    if tissue_probs.size:
        print(f"[heatmap] P(tumor): mean={tissue_probs.mean():.3f} "
              f"max={tissue_probs.max():.3f} "
              f">=0.5 ratio={(tissue_probs >= 0.5).mean():.3f}")
    print(f"[heatmap] figure -> {fig_path}")
    print(f"[heatmap] grid   -> {grid_path}")


if __name__ == "__main__":
    main()
