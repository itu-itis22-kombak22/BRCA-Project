"""Rule-based 2-3 sentence Turkish report from a scored image.

Two-tier detection:
  * **Absolute**: P(tumor) >= 0.5 over tissue patches → "şüpheli".
  * **Relative**: patches notably above the image's own median → even if
    absolute values are compressed (e.g. domain-shifted PCam model on
    primary breast tissue), we flag the top-tier regions so the user
    does not conclude "no tumor at all" when spatial heterogeneity is
    clearly present.
"""

from __future__ import annotations


def _fmt_pct(p: float) -> str:
    return f"%{100 * p:.1f}".replace(".", ",")


def generate(stats: dict, image_name: str | None = None) -> str:
    mode = stats.get("mode", "grid")
    n_tissue = int(stats.get("n_tissue", 0))
    mean_p = float(stats.get("mean", 0.0))
    max_p = float(stats.get("max", 0.0))
    abs_ratio = float(stats.get("suspicious_ratio", 0.0))
    rel_ratio = float(stats.get("relative_ratio", 0.0))
    rel_thresh = float(stats.get("relative_threshold", 0.0))

    if mode == "single":
        if max_p >= 0.5:
            verdict = "**şüpheli (tümör lehine)** olarak işaretlendi"
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

    header = (
        f"Toplam {n_tissue} doku parçası değerlendirildi "
        f"(ortalama {_fmt_pct(mean_p)}, en yüksek {_fmt_pct(max_p)})."
    )

    # Tier 1: absolute confidence is high somewhere
    if abs_ratio >= 0.3:
        body = (
            f"Doku parçalarının {_fmt_pct(abs_ratio)}'i 0,50 eşiğinin "
            f"üstünde şüpheli sınıflandırıldı; yaygın şüpheli alanlar "
            f"mevcut, uzman patolog değerlendirmesi güçlü biçimde önerilir."
        )
    elif abs_ratio >= 0.1:
        body = (
            f"Doku parçalarının {_fmt_pct(abs_ratio)}'i 0,50 eşiğinin "
            f"üstünde şüpheli işaretlendi; odaksal (bölgesel) şüpheli "
            f"alanlar mevcut, detaylı inceleme önerilir."
        )
    elif abs_ratio > 0.0:
        body = (
            f"Sınırlı düzeyde şüpheli sinyal var ({_fmt_pct(abs_ratio)} "
            f"parça 0,50 eşiği üstünde); bulgu küçük ve izole, "
            f"doğrulanması için patolog kontrolü yararlı olabilir."
        )
    # Tier 2: absolute values all low but relative hotspots exist
    elif (rel_ratio >= 0.02 and max_p >= 0.08) or max_p >= 0.15:
        body = (
            f"Mutlak olasılıklar düşük kaldı, ancak görsel içinde görece "
            f"öne çıkan {_fmt_pct(rel_ratio)} oranında bölge tespit edildi "
            f"(maks {_fmt_pct(max_p)}); ısı haritasındaki sıcak noktalar "
            f"patolog tarafından doğrulanmaya değer."
        )
    else:
        body = (
            "Bu görüntüde modelin eşik üstü bir tümör sinyali bulmadığı "
            "görülmektedir; ancak sınıflandırıcı PCam (lenf nodu metastazı) "
            "dağılımında eğitildiği için primer meme dokusunda skorlar "
            "sistematik olarak düşük kalabilir ve negatif sonuç tek başına "
            "tümör yokluğu anlamına gelmez."
        )

    disclaimer = (
        "Bu çıktı tanı değil karar-destek amaçlıdır; "
        "kesin değerlendirme patoloji uzmanı tarafından yapılmalıdır."
    )
    return f"{header} {body} {disclaimer}"
