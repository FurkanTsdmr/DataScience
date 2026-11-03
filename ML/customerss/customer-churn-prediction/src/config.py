"""Proje genelinde kullanılan sabitler.

Tek bir yerde tutuyoruz ki notebook, script, API ve Streamlit aynı kuralları kullansın.
(Kural değişirse tek dosyayı güncellemek yeter.)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Dosya yolları -----------------------------------------------------------
RAW_PATH = ROOT / "data" / "raw" / "customer_churn_raw.csv"
REFERENCE_PATH = ROOT / "data" / "reference" / "customer_churn_clean_reference.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "customer_churn_clean.csv"
CONFLICTS_PATH = ROOT / "data" / "processed" / "conflicting_customers.csv"
CLEANING_LOG_PATH = ROOT / "reports" / "cleaning_log.csv"
SCORED_PATH = ROOT / "data" / "processed" / "scored_customers.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
MODEL_META_PATH = ROOT / "models" / "model_metadata.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Şema --------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "churn"
ID_COL = "customer_id"

CATEGORICAL_COLS = ["region", "device_type", "plan", "contract_type", "payment_method"]
COUNT_COLS = ["late_payments_6m", "support_calls_6m", "complaints_6m"]
NUMERIC_COLS = [
    "age", "tenure_months", "monthly_usage_hours", "monthly_fee_tl",
    "discount_rate", "late_payments_6m", "support_calls_6m",
    "complaints_6m", "satisfaction_score",
]
# Modelin ham girdi şeması (customer_id ve churn hariç 14 sütun)
RAW_FEATURE_COLS = [
    "age", "region", "device_type", "plan", "contract_type", "tenure_months",
    "monthly_usage_hours", "monthly_fee_tl", "discount_rate", "payment_method",
    "late_payments_6m", "support_calls_6m", "complaints_6m", "satisfaction_score",
]

# --- İş kuralları (data_dictionary.csv ile uyumlu) ------------------------------
RANGE_RULES = {
    "age": (18, 100),
    "tenure_months": (1, 72),
    "monthly_usage_hours": (0, 100),
    "discount_rate": (0.0, 0.20),
    "satisfaction_score": (1, 10),
}
FEE_MAX = 1000.0  # ücret: 0 < ücret <= 1000

# Veride keşfettiğimiz kural: ücret = plan liste fiyatı * (1 - indirim)
PLAN_BASE_PRICE = {"Basic": 249.0, "Standard": 399.0, "Premium": 599.0}

# Kategori standardizasyonu: küçük harf -> kanonik yazım
CATEGORY_MAP = {
    "region": {
        "marmara": "Marmara", "aegean": "Aegean",
        "central anatolia": "Central Anatolia", "black sea": "Black Sea",
        "mediterranean": "Mediterranean", "eastern anatolia": "Eastern Anatolia",
    },
    "device_type": {"mobile": "Mobile", "desktop": "Desktop", "tablet": "Tablet"},
    "plan": {"basic": "Basic", "standard": "Standard", "premium": "Premium"},
    "contract_type": {"monthly": "Monthly", "annual": "Annual", "two-year": "Two-Year"},
    "payment_method": {
        "credit card": "Credit Card", "bank transfer": "Bank Transfer",
        "e-wallet": "E-Wallet", "cash": "Cash",
    },
}
CATEGORY_LEVELS = {col: sorted(m.values()) for col, m in CATEGORY_MAP.items()}

# --- İş varsayımları (eşik seçimi için; GERÇEK şirket değerleriyle değiştirilmeli) ---
ASSUMPTIONS = {
    "offer_cost_tl": 50.0,          # aranan/ulaşılan her müşteri için kampanya maliyeti (SMS/e-posta + küçük bonus)
    "offer_success_rate": 0.30,     # ayrılacak müşteriyi teklifle tutma olasılığı
    "customer_value_months": 12,    # kurtarılan müşterinin getireceği ay sayısı
    "margin": 0.40,                 # aylık ücretin kâra dönen oranı
}
