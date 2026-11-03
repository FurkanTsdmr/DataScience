"""Model adayları, çapraz doğrulama karşılaştırması, model seçimi ve eşik seçimi.

Notebook'lar ve train.py aynı fonksiyonları kullanır; böylece sayılar her yerde aynıdır.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src.features import FEATURE_SETS
from src.preprocessing import build_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS, N_REPEATS = 5, 3


# ----------------------------------------------------------------------------- veri
def load_clean() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_PATH)


def split_data(df: pd.DataFrame, test_size=0.20):
    """Stratified bölme. Dönen indeksler ID takibi ve fee erişimi için df üzerindedir."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=C.RANDOM_STATE, stratify=df[C.TARGET])
    return df.loc[train_idx], df.loc[test_idx]


def xy(df: pd.DataFrame):
    return df[C.RAW_FEATURE_COLS], df[C.TARGET]


def make_cv(repeated=True, seed=C.RANDOM_STATE):
    if repeated:
        return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


# ----------------------------------------------------------------------------- adaylar
# (açıklama, estimator, ölçekleme gerekir mi, sadelik sırası [küçük = daha sade], param grid)
def candidates():
    rs = C.RANDOM_STATE
    return {
        "Dummy (prior)": dict(
            estimator=DummyClassifier(strategy="prior"), scale=False, simplicity=0, grid={}),
        "Logistic Regression": dict(
            estimator=LogisticRegression(max_iter=5000, random_state=rs), scale=True, simplicity=1,
            grid={"model__C": [0.03, 0.1, 0.3, 1, 3]}),
        "Logistic Regression (balanced)": dict(
            estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=rs),
            scale=True, simplicity=1, grid={"model__C": [0.03, 0.1, 0.3, 1, 3]}),
        "Decision Tree": dict(
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=rs), scale=False,
            simplicity=2, grid={"model__max_depth": [2, 3, 4, 5], "model__min_samples_leaf": [10, 20, 40]}),
        "Random Forest": dict(
            estimator=RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                             random_state=rs, n_jobs=1),
            scale=False, simplicity=3,
            grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [10, 20, 40]}),
        "Hist Gradient Boosting": dict(
            estimator=HistGradientBoostingClassifier(class_weight="balanced", random_state=rs,
                                                     max_iter=100, learning_rate=0.05),
            scale=False, simplicity=4,
            grid={"model__max_depth": [2, 3], "model__min_samples_leaf": [20, 40]}),
    }


SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "neg_brier": "neg_brier_score"}


def tune_and_compare(X_train, y_train, feature_set: str, names=None, n_jobs=-1):
    """Her aday için küçük bir grid araması (hiperparametre) + tekrarlı CV skorları.

    Seçim ölçütü PR-AUC (average_precision): dengesiz veride en bilgilendirici sıralama metriği.
    Dönüş: (sonuç tablosu, {isim: en iyi pipeline (henüz fit edilmemiş, parametreleri ayarlı)})
    """
    num, cat = FEATURE_SETS[feature_set]
    cv = make_cv()
    rows, best = [], {}
    for name, spec in candidates().items():
        if names and name not in names:
            continue
        pipe = build_pipeline(spec["estimator"], spec["scale"], num, cat)
        if spec["grid"]:
            gs = GridSearchCV(pipe, spec["grid"], scoring=SCORING, refit="pr_auc", cv=cv, n_jobs=n_jobs)
            gs.fit(X_train, y_train)
            i = gs.best_index_
            r = gs.cv_results_
            row = {"model": name, "feature_set": feature_set,
                   "pr_auc": r["mean_test_pr_auc"][i], "pr_auc_std": r["std_test_pr_auc"][i],
                   "roc_auc": r["mean_test_roc_auc"][i], "roc_auc_std": r["std_test_roc_auc"][i],
                   "brier": -r["mean_test_neg_brier"][i],
                   "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}}
            best[name] = gs.best_estimator_
        else:
            from sklearn.model_selection import cross_validate
            cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=n_jobs)
            row = {"model": name, "feature_set": feature_set,
                   "pr_auc": cvres["test_pr_auc"].mean(), "pr_auc_std": cvres["test_pr_auc"].std(),
                   "roc_auc": cvres["test_roc_auc"].mean(), "roc_auc_std": cvres["test_roc_auc"].std(),
                   "brier": -cvres["test_neg_brier"].mean(), "best_params": {}}
            best[name] = pipe
        row["simplicity"] = spec["simplicity"]
        rows.append(row)
    return pd.DataFrame(rows), best


def select_by_one_se(results: pd.DataFrame) -> str:
    """1-standart-hata kuralı: en iyi PR-AUC'nin 1 SE içindeki modeller arasından EN SADE olanı seç.

    Eşit sadelikte düşük Brier skoru (daha iyi kalibrasyon) tercih edilir.
    SE = std / sqrt(n_splits). (Tekrarlı CV fold'ları bağımsız olmadığı için sqrt(toplam fold)
    kullanmak aralığı olduğundan dar gösterirdi.)
    """
    best = results.loc[results["pr_auc"].idxmax()]
    se = best["pr_auc_std"] / np.sqrt(N_SPLITS)
    eligible = results[results["pr_auc"] >= best["pr_auc"] - se]
    eligible = eligible[eligible["model"] != "Dummy (prior)"]
    # Eşit sadelikte: olasılıkları daha iyi kalibre olan (düşük Brier) model kazanır
    return eligible.sort_values(["simplicity", "brier"], ascending=[True, True]).iloc[0]["model"]


# ----------------------------------------------------------------------------- tahmin
def oof_probabilities(pipeline, X, y, repeats=5, seed=C.RANDOM_STATE) -> np.ndarray:
    """Out-of-fold olasılıklar: her müşterinin skoru, o müşteriyi GÖRMEMİŞ bir modelden gelir.
    Tekrarlı (farklı fold bölmeleri) ortalama alınarak gürültü azaltılır."""
    from sklearn.base import clone
    acc = np.zeros(len(y))
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + r)
        acc += cross_val_predict(clone(pipeline), X, y, cv=cv, method="predict_proba")[:, 1]
    return acc / repeats


def cv_permutation_importance(pipeline, X, y, columns, n_repeats=10, seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Çapraz doğrulamalı permütasyon önemi.

    Küçük test kümesinde (30 ayrılan) permütasyon önemi çok gürültülüdür. Burada her fold'da modeli
    eğitip, doğrulama fold'unda bir sütunu karıştırıp PR-AUC'nin ne kadar düştüğünü ölçeriz ve
    tüm fold/tekrarlar üzerinden ortalarız (eğitim verisinin tamamı kullanılır; test'e dokunulmaz).
    """
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    drops = {c: [] for c in columns}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        Xv, yv = X.iloc[va_idx].copy(), y.iloc[va_idx]
        base = average_precision_score(yv, model.predict_proba(Xv)[:, 1])
        for c in columns:
            for _ in range(n_repeats):
                Xp = Xv.copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                drops[c].append(base - average_precision_score(yv, model.predict_proba(Xp)[:, 1]))
    return (pd.DataFrame({"feature": list(drops), "importance": [np.mean(v) for v in drops.values()],
                          "std": [np.std(v) for v in drops.values()]})
            .sort_values("importance", ascending=False).reset_index(drop=True))
