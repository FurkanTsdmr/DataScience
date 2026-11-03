"""Uçtan uca eğitim: python -m src.train

Sıra (her adımın nedeni docs/ klasöründe anlatılıyor):
  1. Temiz veriyi oku, stratified train/test böl. Test kümesi SON ADIMA kadar kilitli kalır.
  2. Özellik setlerini (base / engineered / core) eğitim verisinde CV ile karşılaştır.
  3. Modelleri ayarla + karşılaştır, 1-SE kuralıyla seç.
  4. Eşiği SADECE eğitim verisinin out-of-fold tahminlerinde, iş kârına göre seç.
  5. Seçilen modeli eğit; testte BİR KEZ değerlendir (+ bootstrap güven aralığı).
  6. Önem analizi, kayıt, grafikler, her müşteri için dürüst skor dosyası.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay)

from src import config as C
from src import evaluation as E
from src import modeling as M
from src.features import FEATURE_SETS, SELECTED_FEATURE_SET
from src.preprocessing import build_pipeline

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def smooth_best_threshold(table: pd.DataFrame, window: int = 5) -> float:
    """Net değer eğrisi gürültülüdür; tek bir tepe noktasına aşırı uymamak için kayan ortalama alınır."""
    smooth = table["net_value_tl"].rolling(window, center=True, min_periods=1).mean()
    return float(table.loc[smooth.idxmax(), "threshold"])


def main() -> dict:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1) Veri ve bölme ------------------------------------------------------------------------
    df = M.load_clean()
    train, test = M.split_data(df)
    X_tr, y_tr = M.xy(train)
    X_te, y_te = M.xy(test)
    print(f"Train {len(train)} (churn {y_tr.sum()}), Test {len(test)} (churn {y_te.sum()})")

    # 2) Özellik seti karşılaştırması (Logistic Regression ile) -------------------------------
    fs_rows = []
    for fs in FEATURE_SETS:
        r, _ = M.tune_and_compare(X_tr, y_tr, fs, names=["Logistic Regression"])
        fs_rows.append(r)
    fs_df = pd.concat(fs_rows, ignore_index=True)
    fs_df.to_csv(C.ROOT / "reports" / "feature_set_comparison.csv", index=False)
    print(fs_df[["feature_set", "pr_auc", "pr_auc_std", "roc_auc", "brier"]])

    # 3) Model karşılaştırması + seçim ----------------------------------------------------------
    results, tuned = M.tune_and_compare(X_tr, y_tr, SELECTED_FEATURE_SET)
    results.to_csv(C.ROOT / "reports" / "model_comparison.csv", index=False)
    chosen_name = M.select_by_one_se(results)
    chosen_params = results.loc[results["model"] == chosen_name, "best_params"].iloc[0]
    print(results[["model", "pr_auc", "pr_auc_std", "roc_auc", "brier"]])
    print("Seçilen model:", chosen_name, chosen_params)
    pipeline = tuned[chosen_name]

    # 4) Eşik seçimi: yalnızca eğitim verisinin out-of-fold tahminleri ------------------------
    oof = M.oof_probabilities(pipeline, X_tr, y_tr)
    th_table = E.threshold_table(y_tr, oof, train["monthly_fee_tl"])
    th_table.to_csv(C.ROOT / "reports" / "threshold_table_train_oof.csv", index=False)
    threshold = smooth_best_threshold(th_table)
    oof_at_thr = E.threshold_metrics(y_tr, oof, threshold)
    print("Seçilen eşik:", threshold, oof_at_thr)

    # 5) Final model + TEK SEFERLİK test değerlendirmesi ---------------------------------------
    final = clone(pipeline).fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    test_rank = E.ranking_metrics(y_te, p_te)
    test_thr = E.threshold_metrics(y_te, p_te, threshold)
    test_thr["net_value_tl"] = E.net_value(y_te, p_te >= threshold, test["monthly_fee_tl"])
    ci = E.bootstrap_ci(y_te, p_te, threshold)
    ci.to_csv(C.ROOT / "reports" / "test_bootstrap_ci.csv")
    gain = E.gain_table(y_te, p_te)
    gain.to_csv(C.ROOT / "reports" / "test_gain_table.csv", index=False)

    dummy = build_pipeline(DummyClassifier(strategy="prior"), False, *FEATURE_SETS[SELECTED_FEATURE_SET]).fit(X_tr, y_tr)
    p_dummy = dummy.predict_proba(X_te)[:, 1]
    baseline = {**E.ranking_metrics(y_te, p_dummy),
                "accuracy_predict_all_zero": float(1 - y_te.mean())}
    print("Test:", test_rank, test_thr)

    # 6) Önem analizi ---------------------------------------------------------------------------
    used_raw = list(dict.fromkeys(
        c for c in C.RAW_FEATURE_COLS if c in FEATURE_SETS[SELECTED_FEATURE_SET][0] + FEATURE_SETS[SELECTED_FEATURE_SET][1]))
    perm_df = M.cv_permutation_importance(pipeline, X_tr, y_tr, used_raw)
    perm_df.to_csv(C.ROOT / "reports" / "permutation_importance.csv", index=False)

    # Kampanya maliyeti duyarlılık analizi (eşik ve değer varsayıma çok bağlıdır!)
    sens_rows = []
    for cost in [20, 50, 100, 150]:
        a = {"offer_cost_tl": cost}
        tt = E.threshold_table(y_tr, oof, train["monthly_fee_tl"], assumptions=a)
        th = smooth_best_threshold(tt)
        m = E.threshold_metrics(y_te, p_te, th)
        sens_rows.append({"offer_cost_tl": cost, "threshold": th,
                          "train_oof_flagged_pct": float(tt.loc[tt["threshold"] == th, "flagged_pct"].iloc[0]),
                          "test_flagged_pct": m["flagged_pct"], "test_recall": m["recall"], "test_precision": m["precision"],
                          "test_net_value_tl": E.net_value(y_te, p_te >= th, test["monthly_fee_tl"], a),
                          "test_net_value_if_contact_everyone_tl": E.net_value(y_te, np.ones(len(y_te)), test["monthly_fee_tl"], a)})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(C.ROOT / "reports" / "cost_sensitivity.csv", index=False)

    coef_df = None
    if chosen_name.startswith("Logistic"):
        names = final.named_steps["preprocessor"].get_feature_names_out()
        coef = final.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)}) \
            .sort_values("coef", ascending=False)
        coef_df.to_csv(C.ROOT / "reports" / "model_coefficients.csv", index=False)

    # 7) Dürüst skorlar: eğitim satırları için OOF, test satırları için test tahmini ----------
    scored = pd.concat([
        train.assign(risk_score=oof, score_source="out-of-fold (train)"),
        test.assign(risk_score=p_te, score_source="hold-out (test)"),
    ]).sort_index()
    scored["risk_tier"] = E.risk_tier(scored["risk_score"].to_numpy(), threshold)
    scored["flagged"] = (scored["risk_score"] >= threshold).astype(int)
    scored.to_csv(C.SCORED_PATH, index=False)

    # 8) Kayıt -----------------------------------------------------------------------------------
    joblib.dump(final, C.MODEL_PATH)
    required = [c for c in FEATURE_SETS[SELECTED_FEATURE_SET][0] + FEATURE_SETS[SELECTED_FEATURE_SET][1]
                if c in C.RAW_FEATURE_COLS]
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": chosen_name, "hyperparameters": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in chosen_params.items()},
        "feature_set": SELECTED_FEATURE_SET,
        "required_inputs": required,
        "optional_inputs": [c for c in C.RAW_FEATURE_COLS if c not in required],
        "threshold": threshold, "medium_threshold": threshold / 2,
        "assumptions": C.ASSUMPTIONS,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "categorical_levels": C.CATEGORY_LEVELS,
    }
    C.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "n_total": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
        "churn_rate_total": float(df[C.TARGET].mean()), "churn_train": int(y_tr.sum()), "churn_test": int(y_te.sum()),
        "selected_model": chosen_name, "feature_set": SELECTED_FEATURE_SET,
        "cv_selected": results.loc[results["model"] == chosen_name,
                                   ["pr_auc", "pr_auc_std", "roc_auc", "roc_auc_std", "brier"]].iloc[0].to_dict(),
        "threshold": threshold, "oof_at_threshold": oof_at_thr,
        "test_ranking": test_rank, "test_at_threshold": test_thr,
        "test_baseline_dummy": baseline,
        "test_bootstrap_ci_95": ci.round(4).to_dict(orient="index"),
        "assumptions": C.ASSUMPTIONS,
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    make_figures(y_tr, oof, y_te, p_te, th_table, threshold, gain, perm_df, coef_df, final, X_te)
    return metrics


def make_figures(y_tr, oof, y_te, p_te, th_table, threshold, gain, perm_df, coef_df, final, X_te):
    F = C.FIGURES_DIR
    # ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_te, p_te, ax=ax[0], name="Model")
    ax[0].plot([0, 1], [0, 1], "k--", label="Rastgele"); ax[0].legend(); ax[0].set_title("ROC (test)")
    PrecisionRecallDisplay.from_predictions(y_te, p_te, ax=ax[1], name="Model")
    ax[1].axhline(y_te.mean(), color="k", ls="--", label=f"Taban oran {y_te.mean():.3f}")
    ax[1].legend(); ax[1].set_title("Precision-Recall (test)")
    fig.tight_layout(); fig.savefig(F / "roc_pr_test.png"); plt.close(fig)

    # Confusion matrix
    pred = (p_te >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Kalır", "Ayrılır"], cmap="Blues", ax=ax, colorbar=False)
    ax.grid(False); ax.set_title(f"Karışıklık matrisi (test, eşik={threshold:.2f})")
    fig.tight_layout(); fig.savefig(F / "confusion_matrix_test.png"); plt.close(fig)

    # Eşik / net değer (eğitim OOF)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(th_table["threshold"], th_table["precision"], label="Precision")
    ax[0].plot(th_table["threshold"], th_table["recall"], label="Recall")
    ax[0].plot(th_table["threshold"], th_table["f2"], label="F2")
    ax[0].axvline(threshold, color="k", ls="--", label=f"Seçilen eşik {threshold:.2f}")
    ax[0].set_xlabel("Eşik"); ax[0].legend(); ax[0].set_title("Eşik vs metrikler (train OOF)")
    ax[1].plot(th_table["threshold"], th_table["net_value_tl"] / 1000)
    ax[1].axvline(threshold, color="k", ls="--"); ax[1].axhline(0, color="gray")
    ax[1].set_xlabel("Eşik"); ax[1].set_ylabel("Net değer (bin TL)"); ax[1].set_title("Kampanya net değeri (train OOF, varsayımlarla)")
    fig.tight_layout(); fig.savefig(F / "threshold_analysis.png"); plt.close(fig)

    # Kümülatif kazanç
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = [0] + list(gain["top_pct"]); yv = [0] + list(gain["capture_rate"] * 100)
    ax.plot(x, yv, marker="o", label="Model"); ax.plot([0, 100], [0, 100], "k--", label="Rastgele")
    ax.set_xlabel("Risk skoruna göre aranan müşteri (%)"); ax.set_ylabel("Yakalanan ayrılan (%)")
    ax.set_title("Kümülatif kazanç (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(F / "cumulative_gain_test.png"); plt.close(fig)

    # Kalibrasyon
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    frac, mean_pred = calibration_curve(y_tr, oof, n_bins=6, strategy="quantile")
    ax.plot(mean_pred, frac, marker="o", label="Model (train OOF)"); ax.plot([0, 0.5], [0, 0.5], "k--", label="Mükemmel")
    ax.set_xlabel("Tahmin edilen olasılık"); ax.set_ylabel("Gerçekleşen ayrılma oranı"); ax.legend()
    ax.set_title("Kalibrasyon")
    fig.tight_layout(); fig.savefig(F / "calibration.png"); plt.close(fig)

    # Permütasyon önemi
    fig, ax = plt.subplots(figsize=(7, 4))
    d = perm_df.sort_values("importance")
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color="#4c78a8")
    ax.set_xlabel("PR-AUC düşüşü (permütasyon)"); ax.set_title("Permütasyon önemi (train, 5-fold CV)")
    fig.tight_layout(); fig.savefig(F / "permutation_importance.png"); plt.close(fig)

    if coef_df is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = coef_df.sort_values("coef")
        ax.barh(d["feature"], d["coef"], color=np.where(d["coef"] > 0, "#e45756", "#54a24b"))
        ax.axvline(0, color="k"); ax.set_xlabel("Katsayı (log-odds)  | + ayrılma riskini artırır")
        ax.set_title("Lojistik regresyon katsayıları")
        fig.tight_layout(); fig.savefig(F / "model_coefficients.png"); plt.close(fig)


if __name__ == "__main__":
    main()
