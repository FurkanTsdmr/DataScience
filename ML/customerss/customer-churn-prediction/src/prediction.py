"""Tahmin servisi çekirdeği: API ve Streamlit aynı fonksiyonları kullanır.

Model dosyası ham şemayı kabul eder; özellik üretimi Pipeline'ın içindedir (src/features.py).
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config as C
from src import evaluation as E


@lru_cache(maxsize=1)
def load_artifacts():
    if not C.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {C.MODEL_PATH}. Önce `python -m src.train` çalıştırın.")
    model = joblib.load(C.MODEL_PATH)
    meta = json.loads(C.MODEL_META_PATH.read_text(encoding="utf-8"))
    return model, meta


def validate_frame(df: pd.DataFrame) -> None:
    """Zorunlu sütunlar, kategori değerleri ve sayısal aralıklar. Hata varsa ValueError."""
    _, meta = load_artifacts()
    missing = [c for c in meta["required_inputs"] if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik zorunlu sütun(lar): {missing}")
    for c in meta["required_inputs"]:
        # satisfaction_score eğitimde de eksik olabiliyordu (imputer doldurur); diğerleri zorunlu
        if c != "satisfaction_score" and df[c].isna().any():
            raise ValueError(f"'{c}' sütununda boş değer var; bu alan zorunlu.")
    errors = []
    for col, levels in C.CATEGORY_LEVELS.items():
        if col in df.columns:
            bad = set(df[col].dropna().unique()) - set(levels)
            if bad:
                errors.append(f"{col}: geçersiz değer {sorted(bad)}; geçerli: {levels}")
    for col, (lo, hi) in C.RANGE_RULES.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad_n = (df[col].notna() & ~vals.between(lo, hi)).sum()
            if bad_n:
                errors.append(f"{col}: {bad_n} değer [{lo}, {hi}] aralığı dışında")
    for col in C.COUNT_COLS:
        if col in df.columns and (pd.to_numeric(df[col], errors="coerce") < 0).any():
            errors.append(f"{col}: negatif değer olamaz")
    if errors:
        raise ValueError("; ".join(errors))


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model, meta = load_artifacts()
    validate_frame(df)
    score = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["risk_score"] = score
    out["risk_tier"] = E.risk_tier(score, meta["threshold"])
    out["flagged"] = (score >= meta["threshold"]).astype(int)
    return out


def predict_one(record: dict) -> dict:
    _, meta = load_artifacts()
    row = predict_dataframe(pd.DataFrame([record])).iloc[0]
    return {
        "risk_score": float(row["risk_score"]),
        "risk_tier": str(row["risk_tier"]),
        "flagged": bool(row["flagged"]),
        "threshold": float(meta["threshold"]),
    }
