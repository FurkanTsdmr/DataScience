"""Feature Engineering modülü.

ÖNEMLİ TASARIM KARARI: Özellik üretimi bir fonksiyon (`add_features`) olarak yazıldı ve
modelin Pipeline'ının İLK adımı olarak kaydedilir. Böylece:
  * eğitimde de, API'de de, Streamlit'te de AYNI kod çalışır (training-serving skew olmaz),
  * kayıtlı model ham şemayı (14 sütun) doğrudan kabul eder.

Eksik değer kuralı: eksik, "0" demek değildir. Eksik girdiden türeyen özellik de eksik kalır
ve Pipeline içindeki imputer eğitim verisinden öğrendiği değerle doldurur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]

ENGINEERED_NUMERIC = [
    "has_late_payment", "has_support_contact", "has_complaint", "total_issues_6m",
]
ENGINEERED_CATEGORICAL = ["tenure_group"]

# monthly_fee_tl bilerek hiçbir sette YOK: plan x (1 - indirim) ile birebir belirlendiği için
# yeni bilgi taşımaz (çoklu doğrusallık). Notebook 06'da bunu gösteriyoruz.
BASE_NUMERIC = [
    "age", "tenure_months", "monthly_usage_hours", "discount_rate",
    "late_payments_6m", "support_calls_6m", "complaints_6m", "satisfaction_score",
]
MODEL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = C.CATEGORICAL_COLS + ENGINEERED_CATEGORICAL

# Karşılaştırdığımız üç özellik seti: (sayısal sütunlar, kategorik sütunlar)
FEATURE_SETS = {
    "base": (BASE_NUMERIC, C.CATEGORICAL_COLS),
    "engineered": (MODEL_NUMERIC, MODEL_CATEGORICAL),
    # Eğitim verisindeki istatistiksel taramada (BH düzeltmeli) anlamlı çıkan 5 değişken (Notebook 05 §5.7)
    "core": (
        ["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m"],
        ["contract_type"],
    ),
}
SELECTED_FEATURE_SET = "core"   # Notebook 06'daki karşılaştırma sonucuna göre (bkz. docs)


def _flag(series: pd.Series) -> pd.Series:
    """>0 ise 1, değilse 0; eksikse NaN (eksik 'yok' demek değildir)."""
    return (series > 0).astype("float64").where(series.notna(), np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Şemadaki tüm ham sütunlar yoksa NaN olarak ekle: API yalnızca modelin ihtiyaç
    # duyduğu alanları gönderse bile fonksiyon hata vermez.
    out = df.reindex(columns=list(dict.fromkeys(list(df.columns) + C.RAW_FEATURE_COLS))).copy()

    # Kategorik sütunları sklearn'ün rahat işlediği 'object' + NaN biçimine getir
    for col in C.CATEGORICAL_COLS:
        out[col] = out[col].astype("object").where(out[col].notna(), np.nan)

    # Sayısal sütunları float'a çevir (API'den int/str gelse bile aynı davranış)
    for col in C.NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["tenure_group"] = (
        pd.cut(out["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True)
        .astype("object")
        .where(out["tenure_months"].notna(), np.nan)
    )
    out["has_late_payment"] = _flag(out["late_payments_6m"])
    out["has_support_contact"] = _flag(out["support_calls_6m"])
    out["has_complaint"] = _flag(out["complaints_6m"])

    # Toplam sorun sayısı: üç sayaç da biliniyorsa topla, biri eksikse NaN bırak
    counts = out[C.COUNT_COLS]
    out["total_issues_6m"] = counts.sum(axis=1, skipna=False)
    return out
