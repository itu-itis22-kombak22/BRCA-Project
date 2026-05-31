"""Streamlit UI — dark, minimal, professional.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Streamlit launches app.py directly, so the project root isn't on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.inference import _device, load_model
from src.image_validator import validate as validate_image
from src.report import generate as generate_report
from src.upload_pipeline import extract_images, score_image


MODELS_DIR = Path("models")
ACCEPTED_SUFFIXES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip", "pdf"]

MODEL_META = {
    "resnet18_pcam.pth": {
        "label": "ResNet-18 / PCam",
        "domain": "lymph-node metastasis",
        "val_auc": "0.978",
    },
    "resnet18_breakhis.pth": {
        "label": "ResNet-18 / BreaKHis",
        "domain": "primary breast tissue",
        "val_auc": "0.793",
    },
}


def _list_weights() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.pth"))


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: -0.01em;
}

code, pre, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
}

/* ---------- Layout ---------- */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 5rem;
    max-width: 1280px;
}

/* ---------- Typography ---------- */
h1.page-title {
    font-weight: 700;
    letter-spacing: -0.04em;
    font-size: 2rem;
    margin-bottom: 0.2rem;
    line-height: 1.15;
    color: #E6EDF3;
}

.page-tagline {
    color: #6E7681;
    font-size: 0.88rem;
    font-weight: 400;
    margin-bottom: 0;
}

.page-header-rule {
    border: none;
    border-top: 1px solid #21262D;
    margin: 1rem 0 1.75rem 0;
}

/* ---------- Badges ---------- */
.badge-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    background: #161B22;
    border: 1px solid #30363D;
    color: #8B949E;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    white-space: nowrap;
}
.badge.blue {
    background: rgba(79,157,255,0.10);
    border-color: rgba(79,157,255,0.30);
    color: #58A6FF;
}
.badge.green {
    background: rgba(63,185,80,0.10);
    border-color: rgba(63,185,80,0.30);
    color: #3FB950;
}
.badge.amber {
    background: rgba(210,153,34,0.12);
    border-color: rgba(210,153,34,0.30);
    color: #D2933A;
}

/* ---------- Upload zone ---------- */
[data-testid="stFileUploader"] section {
    border: 1px dashed #30363D;
    background: #0D1117;
    border-radius: 10px;
    transition: border-color 0.15s, background 0.15s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4F9DFF;
    background: rgba(79,157,255,0.03);
}

/* ---------- Divider between results ---------- */
.result-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 2.5rem 0;
}

/* ---------- Result section header ---------- */
.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---------- Legend strip ---------- */
.legend-strip {
    font-size: 0.78rem;
    color: #8B949E;
    margin-top: -0.25rem;
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}
.legend-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    transition: border-color 0.15s;
}
[data-testid="stMetric"]:hover {
    border-color: #30363D;
}
[data-testid="stMetricLabel"] {
    color: #6E7681 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #E6EDF3 !important;
}

/* ---------- Warning / validation banner ---------- */
.validation-warn {
    background: rgba(210,153,34,0.10);
    border: 1px solid rgba(210,153,34,0.35);
    border-left: 3px solid #D2933A;
    border-radius: 7px;
    padding: 0.85rem 1.1rem;
    color: #E3B566;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 1.25rem;
}
.validation-warn .warn-title {
    font-weight: 600;
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 0.35rem;
}
.validation-block {
    background: rgba(248,81,73,0.09);
    border: 1px solid rgba(248,81,73,0.30);
    border-left: 3px solid #F85149;
    border-radius: 7px;
    padding: 0.85rem 1.1rem;
    color: #FF7B72;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 1.25rem;
}
.validation-block .warn-title {
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 0.35rem;
}

/* ---------- Report box ---------- */
.report-box {
    background: linear-gradient(180deg, rgba(79,157,255,0.055), rgba(79,157,255,0.015));
    border: 1px solid rgba(79,157,255,0.20);
    border-left: 3px solid #4F9DFF;
    border-radius: 7px;
    padding: 1rem 1.3rem;
    line-height: 1.65;
    font-size: 0.94rem;
    color: #C9D1D9;
    margin: 0.5rem 0 1rem 0;
}

/* ---------- (reserved) ---------- */

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #080B10;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6E7681 !important;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}
[data-testid="stSidebar"] hr {
    border-color: #21262D;
    margin: 1rem 0;
}

/* ---------- Home / brand block ---------- */
.home-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0 1rem 0;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid #21262D;
    cursor: default;
}
.home-brand-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #1A3A5C 0%, #0D1F33 100%);
    border: 1px solid rgba(79,157,255,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
}
.home-brand-text {
    line-height: 1.25;
}
.home-brand-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #E6EDF3;
    letter-spacing: -0.01em;
}
.home-brand-sub {
    font-size: 0.68rem;
    color: #6E7681;
    letter-spacing: 0.02em;
}
.home-btn {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 100%;
    padding: 0.45rem 0.75rem;
    border-radius: 6px;
    background: transparent;
    border: 1px solid transparent;
    color: #8B949E;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
    margin-bottom: 0.5rem;
    text-align: left;
}
.home-btn:hover {
    background: #161B22;
    border-color: #30363D;
    color: #E6EDF3;
}

/* ---------- Captions ---------- */
[data-testid="stCaptionContainer"] p {
    font-size: 0.75rem !important;
    color: #6E7681 !important;
    letter-spacing: 0.02em;
}

/* ---------- Empty state ---------- */
.empty-state {
    border: 1px dashed #21262D;
    border-radius: 10px;
    padding: 3rem 2rem;
    text-align: center;
    color: #6E7681;
}
.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}
.empty-state p {
    font-size: 0.9rem;
    margin: 0.2rem 0;
}

/* ---------- Footer note ---------- */
.footnote {
    color: #6E7681;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #21262D;
    line-height: 1.6;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header [data-testid="stToolbar"] { visibility: hidden; }
</style>
"""


@st.cache_resource(show_spinner="Loading model weights…")
def _load_cached_model(weights_path: str):
    device = _device()
    model = load_model(weights_path, device=device)
    return model, device


def _render_header(device, weights_name: str) -> None:
    meta = MODEL_META.get(weights_name, {})
    model_label = meta.get("label", weights_name)
    domain = meta.get("domain", "—")
    auc = meta.get("val_auc", "—")

    st.markdown(
        "<h1 class='page-title'>BRCA Tumor Probability Analysis</h1>"
        "<p class='page-tagline'>"
        "Patch-level histopathology classifier · ResNet-18 fine-tuned for "
        "breast cancer detection · decision support only"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='page-header-rule'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='badge-row'>"
        f"<span class='badge blue'>⚡ {device}</span>"
        f"<span class='badge green'>◎ {model_label}</span>"
        f"<span class='badge'>domain · {domain}</span>"
        f"<span class='badge amber'>AUC · {auc}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_validation_warning(warnings: list[str], blocked: bool) -> None:
    """Show a styled warning or blocking banner for suspicious images."""
    msgs_html = "".join(f"<li>{w}</li>" for w in warnings)
    if blocked:
        st.markdown(
            "<div class='validation-block'>"
            "<div class='warn-title'>⛔ Histopatoloji Görüntüsü Değil</div>"
            "Yüklenen dosya bir histopatoloji / mikroskop görüntüsüne "
            "benzemiyor. Lütfen H&amp;E boyalı doku kesiti veya mikroskop "
            "görüntüsü yükleyin."
            f"<ul style='margin:0.5rem 0 0 0;padding-left:1.2rem;'>{msgs_html}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='validation-warn'>"
            "<div class='warn-title'>⚠️ Görüntü Kalitesi Şüpheli</div>"
            "Görüntü bazı histopatoloji kriterlerini karşılamıyor olabilir. "
            "Sonuçlar daha az güvenilir olabilir."
            f"<ul style='margin:0.5rem 0 0 0;padding-left:1.2rem;'>{msgs_html}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_result(label: str, stats: dict, image, dt: float) -> None:
    st.markdown(
        f"<p class='result-title'>📄 {label}</p>",
        unsafe_allow_html=True,
    )

    col_img, col_overlay = st.columns(2, gap="medium")
    with col_img:
        st.caption("source image")
        st.image(image, use_container_width=True)
    with col_overlay:
        st.caption("patch-level predictions")
        st.image(stats["overlay"], use_container_width=True)
        st.markdown(
            "<div class='legend-strip'>"
            "<span><span class='legend-dot' style='background:#E64040'></span>"
            "tumor-predicted</span>"
            "<span><span class='legend-dot' style='background:#4182EB'></span>"
            "non-tumor tissue</span>"
            "<span><span class='legend-dot' "
            "style='background:#30363D;border:1px solid #6E7681'></span>"
            "background (skipped)</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Metrics row ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Tissue patches",
        f"{stats['n_tissue']}",
        help=f"{stats['n_total']} total tiles; background excluded.",
    )
    m2.metric("Mean P(tumor)", f"{stats['mean']:.3f}")
    m3.metric("Max P(tumor)",  f"{stats['max']:.3f}")
    m4.metric(
        "Absolute ≥ 0.5",
        f"%{100 * stats['suspicious_ratio']:.1f}".replace(".", ","),
        help="Fraction of tissue patches above the 0.50 absolute threshold.",
    )
    m5.metric(
        "Relative hotspots",
        f"%{100 * stats.get('relative_ratio', 0.0):.1f}".replace(".", ","),
        help=(
            "Patches notably above this image's own median "
            f"(threshold {stats.get('relative_threshold', 0.0):.2f})."
        ),
    )

    st.write("")

    # ── Clinical report ───────────────────────────────────────────────────────
    st.markdown(
        f"<div class='report-box'>{generate_report(stats, image_name=label)}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"analysis time · {dt:.2f} s  ·  tiles scored · {stats['n_tissue']}")


def main() -> None:
    st.set_page_config(
        page_title="BRCA · Tumor Analysis",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Brand / home block
        st.markdown(
            "<div class='home-brand'>"
            "<div class='home-brand-icon'>🔬</div>"
            "<div class='home-brand-text'>"
            "<div class='home-brand-title'>BRCA Analysis</div>"
            "<div class='home-brand-sub'>Tumor Probability · v1.0</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("⌂  New Analysis", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("### Model")

        tissue_choice = st.radio(
            "Tissue type",
            options=["Primary breast", "Lymph node"],
            index=0,
            help=(
                "Primary breast → BreaKHis checkpoint.  "
                "Lymph node → PCam checkpoint.  "
                "Match this to the tissue type in your image."
            ),
        )
        tissue_to_weights = {
            "Primary breast": "resnet18_breakhis.pth",
            "Lymph node":    "resnet18_pcam.pth",
        }
        weights_path = str(MODELS_DIR / tissue_to_weights[tissue_choice])
        meta = MODEL_META.get(Path(weights_path).name, {})
        if meta:
            st.caption(
                f"domain · {meta.get('domain', '—')}  "
                f"·  val AUC · {meta.get('val_auc', '—')}"
            )

        st.markdown("### Analysis")

        tumor_threshold = st.slider(
            "Tumor threshold",
            min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            help=(
                "P(tumor) threshold for red/blue patch colouring. "
                "For the lymph-node model on primary breast tissue, "
                "try 0.10–0.20."
            ),
        )
        bg_threshold = st.slider(
            "Background threshold",
            min_value=180, max_value=240, value=220, step=5,
            help=(
                "Pixels above this mean brightness are treated as background "
                "and skipped. Lower = include more tiles."
            ),
        )
        stride = st.slider(
            "Tile stride (px)",
            min_value=32, max_value=96, value=96, step=16,
            help="96 = non-overlapping (fastest). 32 = 3× denser scan (slower).",
        )

        st.markdown("### Validation")
        skip_validation = st.checkbox(
            "Skip image validation",
            value=False,
            help=(
                "Disable the histopathology image check. "
                "Use only if the validator is incorrectly flagging your image."
            ),
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#6E7681;line-height:1.55;'>"
            "For research and demonstration purposes only.<br>"
            "Not a medical device. Not for clinical use."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Check model weights ───────────────────────────────────────────────────
    if not Path(weights_path).exists():
        st.error(f"Model weights not found: `{weights_path}`")
        st.stop()

    model, device = _load_cached_model(weights_path)

    # ── Page header ───────────────────────────────────────────────────────────
    _render_header(device, Path(weights_path).name)

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload image",
        type=ACCEPTED_SUFFIXES,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='empty-state-icon'>🔬</div>"
            "<p><strong>Drop a histopathology image here</strong></p>"
            "<p>PNG · JPG · TIF · BMP · ZIP of images · multi-page PDF</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='footnote'>"
            "<strong>Model notes —</strong> "
            "The PCam checkpoint was trained on lymph-node metastasis patches; "
            "on primary breast tissue its absolute probabilities are compressed "
            "near zero. "
            "The BreaKHis checkpoint is domain-matched to primary breast and "
            "returns probabilities on the expected 0–1 scale. "
            "Pick the tissue type that matches your image."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Parse file ────────────────────────────────────────────────────────────
    raw = uploaded.read()
    try:
        items = extract_images(uploaded.name, raw)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    if not items:
        st.warning("No analysable images found in the upload.")
        return

    n = len(items)
    st.markdown(
        f"<div style='color:#6E7681;font-size:0.82rem;margin:0.35rem 0 1.5rem;'>"
        f"{n} image{'s' if n != 1 else ''} detected</div>",
        unsafe_allow_html=True,
    )

    # ── Process each image ────────────────────────────────────────────────────
    for i, (label, img) in enumerate(items):
        if i > 0:
            st.markdown("<hr class='result-divider'>", unsafe_allow_html=True)

        # Validation
        if not skip_validation:
            is_valid, warnings = validate_image(img)
            if warnings:
                _render_validation_warning(warnings, blocked=not is_valid)
            if not is_valid:
                st.markdown(
                    "<div style='color:#6E7681;font-size:0.83rem;"
                    "padding:0.5rem 0 1rem;'>"
                    "Analiz durduruldu. Lütfen H&amp;E boyalı doku kesiti "
                    "veya mikroskop görüntüsü yükleyin. "
                    "Doğrulamayı devre dışı bırakmak için sol panelden "
                    "<em>Skip image validation</em> seçeneğini işaretleyin."
                    "</div>",
                    unsafe_allow_html=True,
                )
                continue  # skip inference for this image

        with st.spinner(f"Analyzing {label}…"):
            t0 = time.time()
            stats = score_image(
                model, img, device=device,
                stride=stride,
                bg_threshold=float(bg_threshold),
                tumor_threshold=float(tumor_threshold),
            )
            dt = time.time() - t0

        _render_result(label, stats, img, dt)


if __name__ == "__main__":
    main()
