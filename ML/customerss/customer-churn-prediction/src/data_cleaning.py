"""Veri temizleme modülü.

Her adım ne yaptığını `log` tablosuna yazar; böylece "veriye ne yaptık?" sorusunun
cevabı her zaman elimizde olur (denetlenebilirlik).

Kullanım:
    python -m src.data_cleaning
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C


@dataclass
class CleaningResult:
    clean: pd.DataFrame        # temiz veri (eksikler NaN olarak duruyor)
    conflicts: pd.DataFrame    # aynı ID'ye ait çelişkili kayıtlar (denetim için)
    log: pd.DataFrame          # hangi adımda kaç hücre/satır etkilendi


def read_raw(path=C.RAW_PATH) -> pd.DataFrame:
    """Ham veriyi okur. Ücret sütununu metin olarak okuyoruz ki '399TL' gibi
    değerleri kaybetmeden inceleyebilelim."""
    return pd.read_csv(path, dtype={"monthly_fee_tl": "object"})


def _standardize_categories(df: pd.DataFrame, log: list) -> pd.DataFrame:
    for col, mapping in C.CATEGORY_MAP.items():
        original = df[col]
        normalized = original.astype("object").where(original.notna(), np.nan)
        normalized = normalized.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
        mapped = normalized.map(lambda v: mapping.get(v, np.nan) if isinstance(v, str) else np.nan)

        # Sadece yazım biçimi (büyük/küçük harf, boşluk) düzeltilen hücreler
        changed = (original.notna() & mapped.notna() & (original.astype("object") != mapped)).sum()
        # Sözlükte olmayan (bilinmeyen) kategori -> NaN olur; bunu sessizce geçmeyiz
        unknown = (original.notna() & mapped.isna()).sum()
        if changed:
            log.append(("kategori_standardizasyonu", col, int(changed), "büyük/küçük harf-boşluk düzeltildi"))
        if unknown:
            log.append(("bilinmeyen_kategori", col, int(unknown), "NaN yapıldı"))
        df[col] = mapped.astype("object")
    return df


def _apply_range_rules(df: pd.DataFrame, log: list) -> pd.DataFrame:
    for col, (lo, hi) in C.RANGE_RULES.items():
        valid = df[col].between(lo, hi)
        invalid = df[col].notna() & ~valid
        if invalid.any():
            log.append(("aralik_disi_deger", col, int(invalid.sum()), f"geçerli aralık [{lo}, {hi}] dışı -> NaN"))
        df[col] = df[col].where(valid | df[col].isna(), np.nan)

    for col in C.COUNT_COLS:
        invalid = df[col].notna() & (df[col] < 0)
        if invalid.any():
            log.append(("negatif_sayim", col, int(invalid.sum()), "NaN yapıldı"))
        df[col] = df[col].where(~invalid, np.nan)
    return df


_FEE_WITH_UNIT = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:TL|₺)\s*$", flags=re.IGNORECASE)


def _clean_fee(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Ücret sütunu: metin -> sayı, sonra iş kuralıyla geri kazanım.

    1) '399TL' gibi birimli değerleri sayıya çeviririz (bilgi kaybetmemek için).
    2) Çevrilemeyen metinler ('free', 'error', '?', ...) NaN olur.
    3) (0, 1000] dışı değerler NaN olur.
    4) Ücret = plan liste fiyatı x (1 - indirim) kuralı veride tutarlı biçimde geçerli.
       Bu yüzden eksik/geçersiz ücretleri tahmin etmek yerine kuraldan GERİ KAZANIRIZ;
       kuraldan sapan değerler de kuralla değiştirilir ve loglanır.
    """
    raw = df["monthly_fee_tl"]

    def parse(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip()
        match = _FEE_WITH_UNIT.match(text)
        if match:
            return float(match.group(1).replace(",", "."))
        try:
            return float(text)
        except ValueError:
            return np.nan

    parsed = raw.map(parse).astype("float64")

    recovered_units = raw.map(lambda v: isinstance(v, str) and bool(_FEE_WITH_UNIT.match(v)))
    if recovered_units.any():
        log.append(("ucret_birim_temizleme", "monthly_fee_tl", int(recovered_units.sum()), "'399TL' -> 399.0"))

    unparseable = raw.notna() & parsed.isna()
    if unparseable.any():
        log.append(("ucret_metin_hatasi", "monthly_fee_tl", int(unparseable.sum()),
                    "sayıya çevrilemeyen metin -> NaN: " + ", ".join(sorted(map(str, raw[unparseable].unique())))))

    out_of_range = parsed.notna() & ~((parsed > 0) & (parsed <= C.FEE_MAX))
    if out_of_range.any():
        log.append(("aralik_disi_deger", "monthly_fee_tl", int(out_of_range.sum()),
                    f"ücret (0, {C.FEE_MAX:.0f}] dışı -> NaN"))
    parsed = parsed.where(~out_of_range, np.nan)

    # --- Alanlar arası tutarlılık: ücret / (1 - indirim) tam bir liste fiyatına denk geliyorsa
    # ama bu fiyat plan etiketiyle çelişiyorsa, etiketin yanlış olduğunu varsayarız.
    # Gerekçe: rastgele bir yanlış ücretin başka bir planın liste fiyatına KURUŞUNA kadar denk
    # gelmesi neredeyse imkânsızdır; oysa plan etiketinde yalnızca 3 olası değer vardır.
    price_to_plan = {price: plan for plan, price in C.PLAN_BASE_PRICE.items()}
    implied_price = (parsed / (1 - df["discount_rate"])).round(2)
    implied_plan = implied_price.map(price_to_plan)
    plan_conflict = implied_plan.notna() & df["plan"].notna() & (implied_plan != df["plan"])
    if plan_conflict.any():
        log.append(("plan_ucret_celiskisi", "plan", int(plan_conflict.sum()),
                    "plan etiketi ücret/indirimle çelişiyor -> plan ücretten türetildi (varsayım)"))
        df.loc[plan_conflict, "plan"] = implied_plan[plan_conflict]

    expected = (df["plan"].map(C.PLAN_BASE_PRICE) * (1 - df["discount_rate"])).round(2)

    # Kural tutarlılık kontrolü: geçerli görünen ücret kuraldan sapıyor mu?
    # (Örn. '399TL' sayıya çevrilebiliyor ama Premium + %20 indirim için doğru ücret 479.2'dir.)
    comparable = parsed.notna() & expected.notna()
    mismatch = comparable & ((parsed - expected).abs() > 0.01)
    if mismatch.any():
        log.append(("ucret_kuraldan_sapma_duzeltildi", "monthly_fee_tl", int(mismatch.sum()),
                    f"{int(comparable.sum())} geçerli ücretten kuraldan sapanlar kuralla değiştirildi"))

    to_fill = parsed.isna() & expected.notna()
    if to_fill.any():
        log.append(("ucret_kuralla_geri_kazanim", "monthly_fee_tl", int(to_fill.sum()),
                    "eksik/geçersiz ücret = plan fiyatı x (1-indirim) ile dolduruldu"))
    df["monthly_fee_tl"] = expected.where(expected.notna(), parsed)
    return df


def _resolve_duplicates(df: pd.DataFrame, log: list):
    """Önce birebir kopyalar, sonra aynı ID'ye ait çelişkili kayıtlar.

    Çelişkili kayıtlarda ZAMAN DAMGASI olmadığı için hangisinin doğru olduğunu
    kanıtlayamayız. Varsayım: dosyada ilk görünen kayıt asıldır; dosya sonuna
    eklenen ikinci kayıt geç gelen/yinelenen bir satırdır. Bu bir VARSAYIMDIR ve
    çelişen kayıtlar conflicting_customers.csv dosyasına yazılır.
    """
    exact = df.duplicated(subset=[c for c in df.columns if c != "source_row"], keep="first")
    if exact.any():
        log.append(("birebir_tekrar_silindi", "tum_satir", int(exact.sum()), "ilk kayıt tutuldu"))
    df = df.loc[~exact].copy()

    conflict_mask = df.duplicated(subset=C.ID_COL, keep=False)
    conflicts = df.loc[conflict_mask].sort_values([C.ID_COL, "source_row"]).copy()
    conflicts["kept"] = ~conflicts.duplicated(subset=C.ID_COL, keep="first")

    drop_mask = df.duplicated(subset=C.ID_COL, keep="first")
    if drop_mask.any():
        log.append(("cakisan_id_ikinci_kayit_silindi", C.ID_COL, int(drop_mask.sum()),
                    f"{int(conflicts[C.ID_COL].nunique())} müşteri için ilk kayıt tutuldu (varsayım)"))
    df = df.loc[~drop_mask].copy()
    return df, conflicts


def clean_data(raw: pd.DataFrame) -> CleaningResult:
    log: list = []
    df = raw.copy()                       # ham veriyi asla değiştirmiyoruz
    df["source_row"] = np.arange(len(df))  # dosyadaki orijinal sıra (çakışma çözümü için)
    log.append(("baslangic", "-", int(len(df)), "ham satır sayısı"))

    df = _standardize_categories(df, log)
    df = _apply_range_rules(df, log)
    df = _clean_fee(df, log)
    df, conflicts = _resolve_duplicates(df, log)

    df = df.drop(columns="source_row").reset_index(drop=True)
    conflicts = conflicts.rename(columns={"source_row": "source_row"})

    for col in C.COUNT_COLS + ["tenure_months", C.TARGET]:
        if df[col].notna().all():
            df[col] = df[col].astype("int64")

    log.append(("son", "-", int(len(df)), "temiz satır sayısı"))
    log_df = pd.DataFrame(log, columns=["adim", "sutun", "etkilenen", "aciklama"])
    return CleaningResult(clean=df, conflicts=conflicts, log=log_df)


def run() -> CleaningResult:
    raw = read_raw()
    result = clean_data(raw)
    C.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.CLEANING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.clean.to_csv(C.CLEAN_PATH, index=False)
    result.conflicts.to_csv(C.CONFLICTS_PATH, index=False)
    result.log.to_csv(C.CLEANING_LOG_PATH, index=False)
    return result


if __name__ == "__main__":
    res = run()
    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(res.log.to_string(index=False))
    print("\nTemiz veri:", res.clean.shape, "->", C.CLEAN_PATH)
    print("Eksik değerler:\n", res.clean.isna().sum()[res.clean.isna().sum() > 0])
