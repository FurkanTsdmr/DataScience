"""Model değerlendirme yardımcıları: metrikler, eşik analizi, iş kârı, bootstrap güven aralığı."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)

from src import config as C


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "flagged": int(pred.sum()),
        "flagged_pct": float(pred.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ranking_metrics(y_true, proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def net_value(y_true, flagged, fee, assumptions: dict | None = None) -> float:
    """Kampanyanın beklenen net değeri (TL).

    Aranan her müşteri için teklif maliyeti ödenir. Ayrılacak müşteri aranırsa, teklifin
    başarı olasılığı kadar müşterinin 12 aylık marjı kurtarılır. Varsayımlar config.py'de
    ve GERÇEK şirket verisiyle değiştirilmelidir.
    """
    a = {**C.ASSUMPTIONS, **(assumptions or {})}
    y = np.asarray(y_true); f = np.asarray(flagged).astype(bool); fee = np.asarray(fee, dtype=float)
    value = fee * a["customer_value_months"] * a["margin"]
    saved = a["offer_success_rate"] * value * (y == 1) * f
    cost = a["offer_cost_tl"] * f
    return float(saved.sum() - cost.sum())


def threshold_table(y_true, proba, fee, thresholds=None, assumptions=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.02, 0.62, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_true, proba, t)
        m["net_value_tl"] = net_value(y_true, np.asarray(proba) >= t, fee, assumptions)
        rows.append(m)
    return pd.DataFrame(rows)


def gain_table(y_true, proba, bins=10) -> pd.DataFrame:
    """Risk skoruna göre en yüksek %10, %20... müşteri içinde ayrılanların yüzdesi (kümülatif kazanç) ve lift."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    n, pos = len(d), d["y"].sum()
    rows = []
    for k in range(1, bins + 1):
        top = d.iloc[: int(round(n * k / bins))]
        rows.append({"top_pct": k * 100 // bins, "customers": len(top),
                     "churners_caught": int(top["y"].sum()),
                     "capture_rate": top["y"].sum() / pos,
                     "lift": (top["y"].mean() / d["y"].mean())})
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, proba, threshold, n_boot=2000, seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Küçük test kümesinde metriklerin ne kadar oynak olduğunu göstermek için %95 bootstrap aralığı."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); p = np.asarray(proba); n = len(y)
    stats = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        pred = (p[idx] >= threshold).astype(int)
        stats["roc_auc"].append(roc_auc_score(y[idx], p[idx]))
        stats["pr_auc"].append(average_precision_score(y[idx], p[idx]))
        stats["precision"].append(precision_score(y[idx], pred, zero_division=0))
        stats["recall"].append(recall_score(y[idx], pred, zero_division=0))
    out = pd.DataFrame({k: {"low": np.percentile(v, 2.5), "median": np.percentile(v, 50),
                            "high": np.percentile(v, 97.5)} for k, v in stats.items()}).T
    return out


def risk_tier(score, threshold: float):
    """High: eşik ve üstü | Medium: eşiğin yarısı ve üstü | Low: geri kalan."""
    s = np.asarray(score, dtype=float)
    return np.where(s >= threshold, "High", np.where(s >= threshold / 2, "Medium", "Low"))
