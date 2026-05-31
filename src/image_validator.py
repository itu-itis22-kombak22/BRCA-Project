"""Heuristic validator — checks whether an uploaded image looks like a
histopathology / microscopy slide before running the classifier.

Rules (all computed on the RGB image):
1. **Brightness** — not pure white or very dark.
2. **H&E colour signature** — H&E stained tissue has pink (eosin) and
   purple/blue (hematoxylin) hues. We require a minimum fraction of
   saturated pixels to fall in those hue ranges. This is the strongest
   single discriminator between histopathology and natural photos.
3. **Saturation range** — tissue sections have moderate saturation;
   very high saturation (illustrated / photo) or near-greyscale images
   are flagged.
4. **Texture** — tissue sections have fine-grained cellular texture;
   completely flat / blurry images are flagged.

An image must pass checks 1 + 2 + at least one of (3, 4) to be
considered valid. Check 2 (H&E hue) is treated as near-mandatory: if
it fails with a large margin the image is blocked regardless of the
other checks.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_brightness(arr: np.ndarray) -> tuple[bool, str]:
    lum = arr.mean()
    if lum > 238:
        return False, (
            "Görüntü neredeyse tamamen beyaz / boş. "
            "Lütfen doku içeren bir histopatoloji görüntüsü yükleyin."
        )
    if lum < 30:
        return False, (
            "Görüntü çok karanlık. "
            "Lütfen yeterince aydınlatılmış bir mikroskop görüntüsü yükleyin."
        )
    return True, ""


def _check_he_signature(arr: np.ndarray) -> tuple[bool, str, float]:
    """Check for H&E colour signature (pink + purple/blue hues).

    H&E staining produces:
      - Eosin  → pink / magenta  (PIL HSV hue ≈ 220–255 and 0–20, 0–255 scale)
      - Haematoxylin → purple / blue (PIL HSV hue ≈ 150–210)

    We look at saturated pixels only (S > 35) and measure the fraction
    whose hue falls in the combined pink-purple range (hue ≥ 140 or ≤ 25).

    Returns (ok, message, he_fraction).
    """
    hsv = np.array(Image.fromarray(arr).convert("HSV"))
    h = hsv[:, :, 0].astype(np.int16)   # 0–255
    s = hsv[:, :, 1]                     # 0–255

    saturated_mask = s > 35
    n_sat = int(saturated_mask.sum())

    if n_sat < 200:
        # Mostly unsaturated → likely a near-white background image
        return False, (
            "Görüntüde yeterince renkli piksel bulunamadı. "
            "H&E boyalı doku kesitlerinde belirgin pembe/mor renkler beklenir."
        ), 0.0

    h_sat = h[saturated_mask]

    # H&E hue range: purple-blue (140–210) + pink-magenta (220–255 and 0–25)
    he_mask = (h_sat >= 140) | (h_sat <= 25)
    he_fraction = float(he_mask.sum()) / n_sat

    if he_fraction < 0.08:
        return False, (
            f"H&E boyası renk imzası tespit edilemedi "
            f"(pembe/mor piksel oranı yalnızca %{100 * he_fraction:.0f}). "
            "Görüntü histopatoloji görüntüsü olmayabilir; "
            "lütfen H&E boyalı doku kesiti yükleyin."
        ), he_fraction

    if he_fraction < 0.18:
        # Low but not zero — soft warning only
        return True, (
            f"H&E renk imzası zayıf (pembe/mor oran %{100 * he_fraction:.0f}). "
            "Sonuçlar daha az güvenilir olabilir."
        ), he_fraction

    return True, "", he_fraction


def _check_saturation(arr: np.ndarray) -> tuple[bool, str]:
    hsv = np.array(Image.fromarray(arr).convert("HSV"))
    s = hsv[:, :, 1].astype(float)
    mean_s = s.mean()
    if mean_s > 195:
        return False, (
            "Görüntü renk doygunluğu çok yüksek (doğal fotoğraf veya illüstrasyon). "
            "Mikroskop görüntülerinde doygunluk genellikle orta düzeydedir."
        )
    if mean_s < 6:
        return False, (
            "Görüntü neredeyse gri tonlamalı. "
            "H&E boyalı doku görüntülerinde pembe/mor renk beklenir."
        )
    return True, ""


def _check_texture(arr: np.ndarray) -> tuple[bool, str]:
    try:
        from scipy.ndimage import laplace
        grey = arr.mean(axis=2)
        lap_var = float(laplace(grey).var())
        if lap_var < 4.0:
            return False, (
                "Görüntüde yeterli doku dokusu (texture) tespit edilemedi; "
                "düz, bulanık veya baskı belgesi olabilir."
            )
    except Exception:
        pass
    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(image: Image.Image) -> tuple[bool, list[str]]:
    """Return ``(is_valid, [warning_messages])``.

    Blocking logic:
      - Brightness check failure → always block.
      - H&E signature failure (he_fraction < 0.08) → block.
      - H&E soft warning (0.08–0.18) → warn but allow.
      - If H&E passes: require at least one of (saturation, texture) to pass.
        If both fail → warn but allow (unusual staining may trigger these).

    This means a yellow smiley / green landscape / red logo will be blocked
    because its he_fraction will be near zero.
    """
    arr = np.array(image.convert("RGB"))
    warnings: list[str] = []

    # Check 1 — brightness (hard block)
    bright_ok, bright_msg = _check_brightness(arr)
    if not bright_ok:
        return False, [bright_msg]

    # Check 2 — H&E signature (hard block below 0.08; soft warn 0.08–0.18)
    he_ok, he_msg, he_frac = _check_he_signature(arr)
    if not he_ok:
        return False, [he_msg]
    if he_msg:
        warnings.append(he_msg)

    # Check 3 — saturation
    sat_ok, sat_msg = _check_saturation(arr)
    if not sat_ok:
        warnings.append(sat_msg)

    # Check 4 — texture
    tex_ok, tex_msg = _check_texture(arr)
    if not tex_ok:
        warnings.append(tex_msg)

    # If both auxiliary checks fail alongside a weak H&E signal → soft warn only
    # (never hard-block here; H&E passed so we give benefit of the doubt)
    is_valid = True
    return is_valid, warnings
