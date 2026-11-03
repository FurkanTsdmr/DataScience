"""Ön işleme + Pipeline kurucusu.

Sıra: add_features -> (imputer + scaler / imputer + one-hot) -> model
Tüm 'öğrenen' adımlar (medyan, ortalama, kategori listesi) SADECE eğitim verisinden öğrenilir;
bu yüzden cross-validation içinde de veri sızıntısı (data leakage) olmaz.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import MODEL_CATEGORICAL, MODEL_NUMERIC, add_features


def build_preprocessor(scale_numeric: bool = True, numeric_cols=None, categorical_cols=None) -> ColumnTransformer:
    numeric_cols = list(MODEL_NUMERIC if numeric_cols is None else numeric_cols)
    categorical_cols = list(MODEL_CATEGORICAL if categorical_cols is None else categorical_cols)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_cols),
            ("categorical", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",   # customer_id, monthly_fee_tl, churn gibi kullanmadıklarımız düşer
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale_numeric: bool = True, numeric_cols=None, categorical_cols=None) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocessor", build_preprocessor(scale_numeric, numeric_cols, categorical_cols)),
            ("model", estimator),
        ]
    )
