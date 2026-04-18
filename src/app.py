"""Streamlit UI: upload → model inference → short Turkish report.

Run with:

    streamlit run src/app.py
"""

from __future__ import annotations

import io
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


DEFAULT_WEIGHTS = "models/resnet18_pcam.pth"
ACCEPTED_SUFFIXES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "zip", "pdf"]


@st.cache_resource(show_spinner="Model yükleniyor…")
def _load_cached_model(weights_path: str):
    device = _device()
    model = load_model(weights_path, device=device)
    return model, device


def _page_config() -> None:
    st.set_page_config(
        page_title="BRCA Patch Classifier",
        page_icon="🔬",
        layout="wide",
    )


def main() -> None:
    _page_config()

    st.title("🔬 Meme Patolojisi — Tümör Olasılığı Analizi")
    st.caption(
        "Histopatoloji görüntüsü yükleyin (PNG/JPG/TIF/ZIP/PDF). Model her "
        "görseli 96×96 piksellik parçalara ayırır, her parçayı PCam üzerinde "
        "eğitilmiş ResNet-18 ile skorlar ve kısa bir rapor üretir."
    )

    with st.sidebar:
        st.header("Ayarlar")
        weights_path = st.text_input("Model ağırlık dosyası",
                                     value=DEFAULT_WEIGHTS)
        bg_threshold = st.slider(
            "Arkaplan eşiği (düşürürsen daha çok patch doku sayılır)",
            min_value=180, max_value=240, value=220, step=5)
        stride = st.slider("Kayma (stride, piksel)",
                           min_value=32, max_value=96, value=96, step=16)
        st.markdown("---")
        st.markdown(
            "**Not:** Model, PCam (lenf nodu metastazı) verisinde eğitildi. "
            "Primer meme dokusunda skorlar sistematik olarak düşük kalabilir."
        )

    if not Path(weights_path).exists():
        st.error(f"Model dosyası bulunamadı: `{weights_path}`")
        st.stop()

    model, device = _load_cached_model(weights_path)
    st.caption(f"Cihaz: `{device}`")

    uploaded = st.file_uploader(
        "Görüntü yükle (PNG, JPG, TIF, ZIP içinde görseller, veya PDF)",
        type=ACCEPTED_SUFFIXES,
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("Başlamak için bir dosya sürükleyip bırakın.")
        return

    raw = uploaded.read()
    try:
        items = extract_images(uploaded.name, raw)
    except Exception as exc:
        st.error(f"Dosya okunamadı: {exc}")
        return

    if not items:
        st.warning(
            "Yüklenen dosyada analiz edilebilir görsel bulunamadı "
            "(ZIP/PDF içinde görüntü yok mu?)."
        )
        return

    st.success(f"{len(items)} görsel bulundu. Analiz başlatılıyor…")

    for label, img in items:
        with st.spinner(f"{label} analiz ediliyor…"):
            t0 = time.time()
            stats = score_image(model, img, device=device,
                                stride=stride, bg_threshold=float(bg_threshold))
            dt = time.time() - t0

        st.markdown("---")
        st.subheader(label)

        col_img, col_overlay = st.columns(2)
        with col_img:
            st.markdown("**Orijinal**")
            st.image(img, use_container_width=True)
        with col_overlay:
            st.markdown("**Tümör olasılığı haritası**")
            st.image(stats["overlay"], use_container_width=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Doku parçası", f"{stats['n_tissue']} / {stats['n_total']}")
        col_b.metric("Ortalama P(tümör)", f"{stats['mean']:.3f}")
        col_c.metric("En yüksek P(tümör)", f"{stats['max']:.3f}")
        col_d.metric("Şüpheli oran (≥0.5)",
                     f"%{100 * stats['suspicious_ratio']:.1f}")

        st.info(generate_report(stats, image_name=label))
        st.caption(f"Süre: {dt:.2f} sn")


if __name__ == "__main__":
    main()
