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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: -0.01em;
}

code, pre, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.03em;
    font-size: 2.2rem;
    margin-bottom: 0.25rem;
    line-height: 1.15;
}

.subtitle {
    color: #8B949E;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #21262D;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    background: #161B22;
    border: 1px solid #30363D;
    color: #8B949E;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
}
.badge.primary {
    background: rgba(79, 157, 255, 0.10);
    border-color: rgba(79, 157, 255, 0.35);
    color: #4F9DFF;
}

.result-card {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}

.report-box {
    background: linear-gradient(180deg, rgba(79,157,255,0.06), rgba(79,157,255,0.02));
    border: 1px solid rgba(79,157,255,0.25);
    border-left: 3px solid #4F9DFF;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    line-height: 1.6;
    font-size: 0.98rem;
    color: #E6EDF3;
    margin: 0.5rem 0 1.25rem 0;
}

[data-testid="stMetric"] {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 6px;
    padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #E6EDF3 !important;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed #30363D;
    background: #0D1117;
    border-radius: 8px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4F9DFF;
    background: rgba(79, 157, 255, 0.03);
}

[data-testid="stSidebar"] {
    background: #0A0D12;
    border-right: 1px solid #21262D;
}

.footnote {
    color: #6E7681;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #21262D;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header [data-testid="stToolbar"] { visibility: hidden; }
</style>
"""


@st.cache_resource(show_spinner="Loading model weights")
def _load_cached_model(weights_path: str):
    device = _device()
    model = load_model(weights_path, device=device)
    return model, device


def _header(device, weights_name: str) -> None:
    st.markdown(
        "<h1>BRCA Tumor Probability Analysis</h1>"
        "<div class='subtitle'>"
        "Patch-level inference on histopathology images using a ResNet-18 "
        "fine-tuned for breast cancer classification. Upload a single "
        "image, a ZIP of images, or a PDF; receive per-patch tumor "
        "probabilities, a heatmap overlay and a concise clinical-style "
        "summary."
        "</div>",
        unsafe_allow_html=True,
    )
    meta = MODEL_META.get(weights_name, {})
    model_label = meta.get("label", weights_name)
    auc = meta.get("val_auc", "—")
    st.markdown(
        f"<span class='badge primary'>device · {device}</span>"
        f"<span class='badge'>model · {model_label}</span>"
        f"<span class='badge'>val AUC · {auc}</span>",
        unsafe_allow_html=True,
    )
    st.write("")


def _render_result(label: str, stats: dict, image, dt: float) -> None:
    st.markdown(f"<h3 style='margin-top:2rem;margin-bottom:0.75rem;"
                f"font-weight:600;color:#E6EDF3;'>{label}</h3>",
                unsafe_allow_html=True)

    col_img, col_overlay = st.columns(2, gap="medium")
    with col_img:
        st.caption("Source image")
        st.image(image, use_container_width=True)
    with col_overlay:
        st.caption("P(tumor) heatmap")
        st.image(stats["overlay"], use_container_width=True)

    st.write("")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tissue patches",
              f"{stats['n_tissue']}",
              help=f"{stats['n_total']} toplam parça, arkaplan hariç")
    m2.metric("Mean P(tumor)", f"{stats['mean']:.3f}")
    m3.metric("Max P(tumor)",  f"{stats['max']:.3f}")
    m4.metric("Absolute ≥ 0.5",
              f"%{100 * stats['suspicious_ratio']:.1f}".replace(".", ","),
              help="0,50 mutlak eşiğini aşan parça oranı.")
    m5.metric("Relative hotspots",
              f"%{100 * stats.get('relative_ratio', 0.0):.1f}".replace(".", ","),
              help=(f"Bu görüntünün kendi medyanının belirgin üstündeki "
                    f"parçalar (eşik "
                    f"{stats.get('relative_threshold', 0.0):.2f})."))

    st.markdown(
        f"<div class='report-box'>{generate_report(stats, image_name=label)}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"analysis time · {dt:.2f} s")


def main() -> None:
    st.set_page_config(
        page_title="BRCA Tumor Analysis",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration")

        available = _list_weights()
        if available:
            names = [p.name for p in available]
            default_idx = next(
                (i for i, n in enumerate(names) if "breakhis" in n.lower()),
                0,
            )
            choice = st.selectbox(
                "Model",
                names,
                index=default_idx,
                format_func=lambda n: MODEL_META.get(n, {}).get("label", n),
            )
            weights_path = str(MODELS_DIR / choice)
            meta = MODEL_META.get(choice, {})
            if meta:
                st.caption(
                    f"domain · {meta.get('domain', '—')}  "
                    f"·  val AUC · {meta.get('val_auc', '—')}"
                )
        else:
            weights_path = st.text_input(
                "Model weights (models/ klasörü boş)",
                value="models/resnet18_pcam.pth")

        bg_threshold = st.slider(
            "Background threshold",
            min_value=180, max_value=240, value=220, step=5,
            help="Doku kabul eşiği. Düşürürsen daha fazla parça doku "
                 "sayılır — arkaplan içeriğe dahil olabilir.")
        stride = st.slider(
            "Tile stride (px)",
            min_value=32, max_value=96, value=96, step=16,
            help="Pencere kayması. 96 = örtüşmesiz (en hızlı). "
                 "32 = 3× daha yoğun tarama (yavaş).")

    if not Path(weights_path).exists():
        st.error(f"Model weights not found: `{weights_path}`")
        st.stop()

    model, device = _load_cached_model(weights_path)

    _header(device, Path(weights_path).name)

    uploaded = st.file_uploader(
        "Upload image",
        type=ACCEPTED_SUFFIXES,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            "<div class='result-card' style='text-align:center;color:#8B949E;'>"
            "Drag and drop an image here — PNG, JPG, TIF, ZIP of images, "
            "or a multi-page PDF.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='footnote'>"
            "The PCam checkpoint was trained on lymph-node metastasis "
            "patches; on primary breast tissue its absolute probabilities "
            "are compressed near zero. The BreaKHis checkpoint is domain-"
            "matched to primary breast and returns probabilities on the "
            "expected 0–1 scale. Pick whichever fits your input."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    raw = uploaded.read()
    try:
        items = extract_images(uploaded.name, raw)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    if not items:
        st.warning("No analysable images found in the upload.")
        return

    st.markdown(
        f"<div style='color:#8B949E;font-size:0.9rem;margin:0.5rem 0 0 0;'>"
        f"{len(items)} image{'s' if len(items) != 1 else ''} detected"
        f"</div>",
        unsafe_allow_html=True,
    )

    for label, img in items:
        with st.spinner(f"Analyzing {label}"):
            t0 = time.time()
            stats = score_image(model, img, device=device,
                                stride=stride,
                                bg_threshold=float(bg_threshold))
            dt = time.time() - t0
        _render_result(label, stats, img, dt)


if __name__ == "__main__":
    main()
