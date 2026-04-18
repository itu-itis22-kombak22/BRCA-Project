"""Rule-based 2-3 sentence Turkish report from a scored image."""

from __future__ import annotations


def _fmt_pct(p: float) -> str:
    return f"%{100 * p:.1f}".replace(".", ",")


def generate(stats: dict, image_name: str | None = None) -> str:
    """Return a short Turkish report from ``score_image`` output."""
    mode = stats.get("mode", "grid")
    n_tissue = int(stats.get("n_tissue", 0))
    mean_p = float(stats.get("mean", 0.0))
    max_p = float(stats.get("max", 0.0))
    ratio = float(stats.get("suspicious_ratio", 0.0))

    if mode == "single":
        if max_p >= 0.5:
            verdict = "**şüpheli (kanser lehine)** olarak işaretlendi"
        elif max_p >= 0.3:
            verdict = "**sınırda** bir sonuç verdi"
        else:
            verdict = "**normale yakın** olarak sınıflandırıldı"
        return (
            f"Yüklenen görsel tek parça olarak değerlendirildi ve {verdict}. "
            f"Modelin tümör olasılığı: {_fmt_pct(max_p)}. "
            "Bu çıktı tanı değil karar-destek amaçlıdır; kesin değerlendirme "
            "için bir patolog görüşü gereklidir."
        )

    if n_tissue == 0:
        return (
            "Görselde doku içerdiği değerlendirilen bir bölge bulunamadı; "
            "büyük ihtimalle boş/arkaplan ağırlıklı bir görüntü yüklendi. "
            "Lütfen doku içeren bir görsel ile tekrar deneyin."
        )

    if ratio >= 0.3:
        severity = (
            f"Doku parçalarının {_fmt_pct(ratio)}'i şüpheli sınıflandırıldı — "
            f"yaygın şüpheli alanlar gözleniyor"
        )
        recommendation = "uzman patolog değerlendirmesi güçlü biçimde önerilir."
    elif ratio >= 0.1:
        severity = (
            f"Doku parçalarının {_fmt_pct(ratio)}'i şüpheli işaretlendi — "
            f"odaksal (bölgesel) şüpheli alanlar mevcut"
        )
        recommendation = "detaylı inceleme önerilir."
    elif ratio > 0.0 or max_p >= 0.4:
        severity = (
            f"Şüpheli sinyal sınırlı ({_fmt_pct(ratio)} şüpheli parça, "
            f"en yüksek güven {_fmt_pct(max_p)})"
        )
        recommendation = (
            "bulgu küçük ve izole; ancak sinyalin gerçekliğinden emin olmak "
            "için kontrol edilmesi iyi olur."
        )
    else:
        severity = (
            f"Belirgin tümör sinyali tespit edilmedi "
            f"(ortalama {_fmt_pct(mean_p)}, en yüksek {_fmt_pct(max_p)})"
        )
        recommendation = (
            "görüntü büyük oranda normal görünüyor; yine de klinik bağlam "
            "mutlaka patolog tarafından değerlendirilmelidir."
        )

    header = (
        f"Görsel analizi tamamlandı: toplam {n_tissue} doku parçası "
        f"değerlendirildi."
    )
    return f"{header} {severity}; {recommendation}"
