import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.features import FEATURE_SETS, add_features
from src.preprocessing import build_pipeline


def _row(**kw):
    base = dict(age=30, region="Marmara", device_type="Mobile", plan="Basic", contract_type="Monthly",
                tenure_months=6, monthly_usage_hours=40.0, monthly_fee_tl=249.0, discount_rate=0.0,
                payment_method="Cash", late_payments_6m=0, support_calls_6m=0, complaints_6m=0, satisfaction_score=6.0)
    base.update(kw)
    return base


def test_tenure_group_edges():
    df = pd.DataFrame([_row(tenure_months=m) for m in [1, 6, 7, 12, 13, 24, 25, 48, 49, 72]])
    assert list(add_features(df)["tenure_group"]) == ["0-6", "0-6", "7-12", "7-12", "13-24", "13-24", "25-48", "25-48", "49-72", "49-72"]


def test_flags_and_missing_semantics():
    df = pd.DataFrame([_row(late_payments_6m=0, complaints_6m=2), _row(late_payments_6m=np.nan, support_calls_6m=3)])
    out = add_features(df)
    assert list(out["has_late_payment"].iloc[:1]) == [0.0] and np.isnan(out["has_late_payment"].iloc[1])
    assert out["has_complaint"].iloc[0] == 1.0
    assert out["total_issues_6m"].iloc[0] == 2.0 and np.isnan(out["total_issues_6m"].iloc[1])


def test_add_features_does_not_mutate_input():
    df = pd.DataFrame([_row()]); before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_columns_are_tolerated():
    out = add_features(pd.DataFrame([{"tenure_months": 5, "contract_type": "Annual", "plan": "Basic",
                                      "late_payments_6m": 0, "support_calls_6m": 0, "complaints_6m": 0}]))
    assert set(C.RAW_FEATURE_COLS) <= set(out.columns)


@pytest.mark.parametrize("fs", list(FEATURE_SETS))
def test_pipeline_fits_predicts_all_feature_sets(fs):
    df = pd.read_csv(C.CLEAN_PATH)
    X, y = df[C.RAW_FEATURE_COLS], df[C.TARGET]
    num, cat = FEATURE_SETS[fs]
    pipe = build_pipeline(LogisticRegression(max_iter=2000), True, num, cat).fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_no_identifier_or_target_or_fee_in_model_inputs():
    for fs, (num, cat) in FEATURE_SETS.items():
        cols = set(num) | set(cat)
        assert not ({"customer_id", "churn", "monthly_fee_tl"} & cols), fs


def test_pipeline_handles_unseen_category_and_all_missing_optional():
    df = pd.read_csv(C.CLEAN_PATH)
    num, cat = FEATURE_SETS["engineered"]
    pipe = build_pipeline(LogisticRegression(max_iter=2000), True, num, cat).fit(df[C.RAW_FEATURE_COLS], df[C.TARGET])
    new = pd.DataFrame([_row(region="Atlantis", age=np.nan, satisfaction_score=np.nan)])
    assert np.isfinite(pipe.predict_proba(new)[:, 1]).all()
