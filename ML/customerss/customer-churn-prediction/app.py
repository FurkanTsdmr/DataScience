"""Streamlit arayüzü.  Çalıştır:  streamlit run app.py"""
import json
import io

import pandas as pd
import streamlit as st

from src import config as C
from src.prediction import load_artifacts, predict_dataframe, predict_one

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Müşterinin ayrılma riskini tahmin eder. Skor bir 'karar' değil, kampanya önceliği sıralamasıdır.")

try:
    model, meta = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["Tek müşteri", "Toplu tahmin (CSV)", "Model hakkında"])

TIER_TEXT = {
    "High": ("🔴 Yüksek risk", "Elde tutma kampanyası için öncelikli aday."),
    "Medium": ("🟠 Orta risk", "İzlenmeli; düşük maliyetli aksiyon düşünülebilir."),
    "Low": ("🟢 Düşük risk", "Şu an için aksiyon gerekmiyor."),
}

with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        contract = st.selectbox("Sözleşme türü", C.CATEGORY_LEVELS["contract_type"], index=1)
        tenure = st.number_input("Abonelik süresi (ay)", 1, 72, 12)
    with c2:
        satisfaction = st.slider("Memnuniyet puanı (1-10)", 1.0, 10.0, 6.0, 0.1)
        support = st.number_input("Son 6 ay destek araması", 0, 50, 1)
        complaints = st.number_input("Son 6 ay şikâyet", 0, 20, 0)

    if st.button("Riski hesapla", type="primary"):
        result = predict_one({
            "contract_type": contract, "tenure_months": int(tenure),
            "satisfaction_score": float(satisfaction),
            "support_calls_6m": int(support), "complaints_6m": int(complaints)})
        label, advice = TIER_TEXT[result["risk_tier"]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk skoru", f"{result['risk_score']:.1%}")
        m2.metric("Risk kademesi", label)
        m3.metric("Aksiyon eşiği", f"{result['threshold']:.0%}")
        st.progress(min(result["risk_score"] / 0.5, 1.0))
        st.info(advice)
        st.caption("Taban ayrılma oranı yaklaşık %8'dir; 'Orta' ve 'Yüksek' kademeler bu orana göre göreli risktir.")

with tab_batch:
    st.write("Zorunlu sütunlar:", ", ".join(f"`{c}`" for c in meta["required_inputs"]),
             "— diğer sütunlar (plan, yaş, bölge …) kabul edilir ama skoru etkilemez.")
    sample = pd.DataFrame([{"contract_type": "Monthly", "tenure_months": 4, "satisfaction_score": 3.5,
                            "support_calls_6m": 4, "complaints_6m": 1}])
    st.download_button("Örnek CSV indir", sample.to_csv(index=False), "sample_customers.csv", "text/csv")
    uploaded = st.file_uploader("CSV yükle", type="csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            scored = predict_dataframe(data)
            st.dataframe(scored.sort_values("risk_score", ascending=False), width="stretch")
            st.download_button("Sonuçları indir", scored.to_csv(index=False), "scored_customers.csv", "text/csv")
        except ValueError as exc:
            st.error(f"Dosya geçersiz: {exc}")

with tab_about:
    st.subheader("Model")
    st.write(f"**{meta['model']}** — özellik seti: `{meta['feature_set']}` — eşik: **{meta['threshold']:.2f}**")
    if C.METRICS_PATH.exists():
        m = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
        t = m["test_ranking"]; ci = m["test_bootstrap_ci_95"]
        st.write(f"Test ROC-AUC **{t['roc_auc']:.2f}** (%95 aralık {ci['roc_auc']['low']:.2f}–{ci['roc_auc']['high']:.2f}), "
                 f"PR-AUC **{t['pr_auc']:.2f}** (taban oran {m['churn_rate_total']:.2f}).")
    st.warning("Veri sentetiktir; bu uygulama bir öğrenme/portföy projesidir. Eşik, varsayımsal kampanya maliyetlerine dayanır "
               "ve gerçek bir şirkette kendi maliyetleriyle yeniden belirlenmelidir.")
    for name in ["roc_pr_test.png", "cumulative_gain_test.png", "model_coefficients.png"]:
        path = C.FIGURES_DIR / name
        if path.exists():
            st.image(str(path))
