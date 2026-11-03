"""FastAPI servisi.  Çalıştır:  uvicorn api:app --reload   |   Dokümantasyon: /docs"""
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config as C
from src.prediction import load_artifacts, predict_dataframe, predict_one

import pandas as pd

Contract = Literal["Monthly", "Annual", "Two-Year"]
Plan = Literal["Basic", "Standard", "Premium"]


class Customer(BaseModel):
    """Modelin zorunlu girdileri + (opsiyonel) diğer ham alanlar. Opsiyonel alanlar skoru etkilemez."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "tenure_months": 4, "satisfaction_score": 3.5, "complaints_6m": 1, "support_calls_6m": 4,
        "contract_type": "Monthly"}})

    tenure_months: int = Field(ge=1, le=72, description="Abonelik süresi (ay)")
    satisfaction_score: Optional[float] = Field(None, ge=1, le=10, description="1-10; bilinmiyorsa boş bırakılabilir")
    complaints_6m: int = Field(ge=0, le=50)
    support_calls_6m: int = Field(ge=0, le=100)
    contract_type: Contract
    # Opsiyonel ham alanlar (şema uyumu için kabul edilir; mevcut model bunları KULLANMAZ)
    plan: Optional[Plan] = None
    late_payments_6m: Optional[int] = Field(None, ge=0, le=50)
    age: Optional[float] = Field(None, ge=18, le=100)
    region: Optional[Literal[tuple(C.CATEGORY_LEVELS["region"])]] = None
    device_type: Optional[Literal[tuple(C.CATEGORY_LEVELS["device_type"])]] = None
    payment_method: Optional[Literal[tuple(C.CATEGORY_LEVELS["payment_method"])]] = None
    monthly_usage_hours: Optional[float] = Field(None, ge=0, le=100)
    monthly_fee_tl: Optional[float] = Field(None, gt=0, le=1000)
    discount_rate: Optional[float] = Field(None, ge=0, le=0.20)


class Prediction(BaseModel):
    risk_score: float
    risk_tier: Literal["Low", "Medium", "High"]
    flagged: bool
    threshold: float


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0",
              description="Müşteri ayrılma risk skoru. Skor, kalibre olasılığa yakındır; bir 'karar' değil, öncelik sıralamasıdır.")


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        _, meta = load_artifacts()
        return {"status": "ok", "model": meta["model"], "trained_at_utc": meta["trained_at_utc"]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model-info")
def model_info():
    _, meta = load_artifacts()
    return {k: meta[k] for k in ["model", "feature_set", "required_inputs", "optional_inputs", "threshold", "assumptions", "trained_at_utc"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        return predict_one(customer.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    if not customers:
        raise HTTPException(status_code=422, detail="Boş liste gönderildi.")
    if len(customers) > 1000:
        raise HTTPException(status_code=413, detail="En fazla 1000 kayıt gönderebilirsiniz.")
    try:
        _, meta = load_artifacts()
        out = predict_dataframe(pd.DataFrame([c.model_dump() for c in customers]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return [{"risk_score": float(r.risk_score), "risk_tier": str(r.risk_tier), "flagged": bool(r.flagged),
             "threshold": float(meta["threshold"])} for r in out.itertuples()]
