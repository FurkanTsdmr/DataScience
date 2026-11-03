import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_cleaning import clean_data, read_raw


def test_row_counts(cleaning):
    log = cleaning.log.set_index("adim")["etkilenen"]
    assert log["baslangic"] == 1820
    assert log["birebir_tekrar_silindi"] == 12
    assert log["cakisan_id_ikinci_kayit_silindi"] == 8
    assert len(cleaning.clean) == 1800 == log["son"]


def test_raw_is_not_mutated():
    raw = read_raw()
    snapshot = raw.copy()
    clean_data(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_ids_unique_and_match_reference(cleaning, reference):
    ids = cleaning.clean["customer_id"]
    assert ids.is_unique
    assert set(ids) == set(reference.index)


def test_conflicts_file_content(cleaning):
    c = cleaning.conflicts
    assert c["customer_id"].nunique() == 8
    assert len(c) == 16
    assert c.groupby("customer_id")["kept"].sum().eq(1).all()


def test_categories_are_canonical(cleaning):
    for col, levels in C.CATEGORY_LEVELS.items():
        assert set(cleaning.clean[col].dropna()) <= set(levels), col


def test_ranges_hold(cleaning):
    df = cleaning.clean
    for col, (lo, hi) in C.RANGE_RULES.items():
        assert df[col].dropna().between(lo, hi).all(), col
    assert df["monthly_fee_tl"].between(0.01, C.FEE_MAX).all()
    for col in C.COUNT_COLS:
        assert (df[col] >= 0).all()


def test_fee_is_fully_recovered_and_follows_rule(cleaning, reference):
    df = cleaning.clean.set_index("customer_id")
    assert df["monthly_fee_tl"].notna().all()
    expected = (df["plan"].map(C.PLAN_BASE_PRICE) * (1 - df["discount_rate"])).round(2)
    assert np.allclose(df["monthly_fee_tl"], expected)
    assert np.allclose(df["monthly_fee_tl"], reference.loc[df.index, "monthly_fee_tl"])


def test_plan_matches_reference_exactly(cleaning, reference):
    df = cleaning.clean.set_index("customer_id")
    assert (df["plan"] == reference.loc[df.index, "plan"]).all()


@pytest.mark.parametrize("col", ["age", "monthly_usage_hours", "satisfaction_score", "tenure_months",
                                 "discount_rate", "late_payments_6m", "support_calls_6m",
                                 "complaints_6m", "churn", "device_type"])
def test_non_missing_values_agree_with_reference(cleaning, reference, col):
    df = cleaning.clean.set_index("customer_id")
    ok = df[col].notna()
    a, b = df.loc[ok, col], reference.loc[df.index[ok], col]
    if b.dtype.kind in "if":
        assert np.allclose(a.astype(float), b.astype(float)), col
    else:
        assert (a.astype(str) == b.astype(str)).all(), col


def test_target_untouched(cleaning, reference):
    df = cleaning.clean.set_index("customer_id")
    assert (df["churn"] == reference.loc[df.index, "churn"]).all()
    assert df["churn"].sum() == 148


def test_known_remaining_missing_counts(cleaning):
    miss = cleaning.clean.isna().sum()
    assert miss["age"] == 50 and miss["monthly_usage_hours"] == 67
    assert miss["satisfaction_score"] == 82 and miss["region"] == 32 and miss["payment_method"] == 27
    assert miss[["plan", "monthly_fee_tl", "contract_type", "tenure_months", "churn"]].sum() == 0


def test_idempotent(cleaning):
    again = clean_data(cleaning.clean)
    pd.testing.assert_frame_equal(
        again.clean.reset_index(drop=True), cleaning.clean.reset_index(drop=True), check_dtype=False)
