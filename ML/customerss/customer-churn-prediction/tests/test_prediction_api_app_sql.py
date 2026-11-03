import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config as C
from src.prediction import predict_dataframe, predict_one
from src.sql_utils import build_connection, load_queries, run_all

HIGH = dict(tenure_months=3, satisfaction_score=2.5, complaints_6m=2, support_calls_6m=4, late_payments_6m=2,
            contract_type="Monthly", plan="Basic")
LOW = dict(tenure_months=65, satisfaction_score=9.0, complaints_6m=0, support_calls_6m=0, late_payments_6m=0,
           contract_type="Two-Year", plan="Standard")


def test_model_artifacts_exist():
    assert C.MODEL_PATH.exists() and C.MODEL_META_PATH.exists() and C.SCORED_PATH.exists()


def test_risk_ordering_is_sensible():
    hi, lo = predict_one(HIGH), predict_one(LOW)
    assert hi["risk_score"] > lo["risk_score"]
    assert hi["risk_tier"] == "High" and lo["risk_tier"] == "Low"
    assert 0 <= lo["risk_score"] <= hi["risk_score"] <= 1


def test_missing_satisfaction_is_allowed():
    r = predict_one({**HIGH, "satisfaction_score": None})
    assert 0 <= r["risk_score"] <= 1


def test_validation_errors():
    with pytest.raises(ValueError):
        predict_one({**HIGH, "contract_type": "Weekly"})
    with pytest.raises(ValueError):
        predict_one({**HIGH, "tenure_months": 0})
    with pytest.raises(ValueError):
        predict_dataframe(pd.DataFrame([{"plan": "Basic"}]))
    with pytest.raises(ValueError):
        predict_one({**HIGH, "complaints_6m": None})


def test_batch_matches_single():
    df = pd.DataFrame([HIGH, LOW])
    out = predict_dataframe(df)
    assert abs(out["risk_score"].iloc[0] - predict_one(HIGH)["risk_score"]) < 1e-12
    assert list(out["risk_tier"]) == ["High", "Low"]


def test_saved_scores_are_consistent_with_threshold():
    s = pd.read_csv(C.SCORED_PATH)
    assert len(s) == 1800 and s["customer_id"].is_unique
    from src.prediction import load_artifacts
    thr = load_artifacts()[1]["threshold"]
    assert (s["flagged"] == (s["risk_score"] >= thr).astype(int)).all()
    tiers = s.groupby("risk_tier")["churn"].mean()
    assert tiers["High"] > tiers["Medium"] > tiers["Low"]


client = TestClient(__import__("api").app)


def test_api_health_and_info():
    assert client.get("/health").json()["status"] == "ok"
    assert "required_inputs" in client.get("/model-info").json()


def test_api_predict_ok_and_matches_core():
    r = client.post("/predict", json=HIGH)
    assert r.status_code == 200
    assert abs(r.json()["risk_score"] - predict_one(HIGH)["risk_score"]) < 1e-12


@pytest.mark.parametrize("bad", [{**HIGH, "tenure_months": 400}, {**HIGH, "contract_type": "Weekly"},
                                 {k: v for k, v in HIGH.items() if k != "contract_type"}, {**HIGH, "complaints_6m": -1}])
def test_api_rejects_invalid(bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_api_batch():
    r = client.post("/predict/batch", json=[HIGH, LOW])
    assert r.status_code == 200 and [x["risk_tier"] for x in r.json()] == ["High", "Low"]
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_streamlit_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception and len(at.metric) >= 3


def test_sql_queries_all_run_and_reconcile():
    res = run_all(build_connection())
    assert set(res) == set(load_queries())
    assert res["01_genel_ozet"]["toplam_musteri"].iloc[0] == 1800
    assert res["02_sozlesme_bazinda_churn_ve_lift"]["musteri"].sum() == 1800
    assert res["04_abonelik_suresi_dilimleri"]["musteri"].sum() == 1800
    assert res["08_risk_kademesi_basarimi"]["musteri"].sum() == 1800
    assert res["11_kumulatif_yakalama_pencere"]["kumulatif_yakalanan_yuzde"].iloc[-1] == 100.0
