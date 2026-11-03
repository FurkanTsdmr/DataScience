from tools.nbbuild import md, code

NB01 = [
md("""
# 01 — İş Problemini Anlama (Business Understanding)

**Bu notebook'ta ne yapıyoruz?** Hiç kod yazmadan önce şu soruyu cevaplıyoruz: *Bu model kime, hangi karar için lazım?*
Veri bilimi projelerinin en sık başarısız olma nedeni kötü model değil, **yanlış sorudur**.

## Problem
Bir abonelik şirketi (telekom/dijital hizmet) her ay müşteri kaybediyor. Yeni müşteri kazanmak, mevcut müşteriyi tutmaktan genellikle daha pahalıdır.
Amaç: **ayrılma ihtimali yüksek müşterileri önceden bulup**, sınırlı bütçeyle elde tutma kampanyası yapmak.

## Bu bir ne tür problem?
* Hedef değişken `churn` (0/1) → **ikili sınıflandırma**.
* Çıktı olarak sadece 0/1 değil, **risk skoru (olasılık)** istiyoruz; çünkü kampanya bütçesi kadar müşteriyi riske göre sıralayıp arayacağız.

## Başarıyı nasıl ölçeceğiz?
| Katman | Metrik | Neden |
|---|---|---|
| Model (teknik) | **PR-AUC**, ROC-AUC, Brier | Dengesiz sınıfta sıralama kalitesi + olasılıkların doğruluğu |
| Karar (operasyonel) | Recall, Precision, aranan müşteri yüzdesi | "Kaç ayrılanı, kaç aramayla yakalıyoruz?" |
| İş (finansal) | **Kampanya net değeri (TL)** | Sonuçta para kazandırmayan model işe yaramaz |

> **Accuracy'yi bilerek ana metrik yapmıyoruz.** Aşağıda göreceğin gibi, herkese "kalır" diyen bir model bile ~%92 Accuracy alır.
""")
,
code("""
data_dictionary = pd.read_csv(C.ROOT / "data" / "raw" / "data_dictionary.csv", encoding="utf-8-sig")
display(data_dictionary)
"""),
md("""
## Sınıf dengesizliği neden önemli? (Küçük bir deney)
Aşağıda hiçbir şey öğrenmeyen bir "model" kuruyoruz: **herkese churn=0 diyor.**
"""),
code("""
raw = pd.read_csv(C.RAW_PATH, dtype={"monthly_fee_tl": "object"})
churn_rate = raw["churn"].mean()
display(Markdown(f"Ham veride churn oranı: **{churn_rate:.2%}**  →  herkese 'kalır' diyen modelin Accuracy'si: **{1-churn_rate:.2%}**, "
                 f"yakaladığı ayrılan müşteri: **0**."))
"""),
md("""
Yani **%91'in üstünde Accuracy'ye sahip, iş değeri sıfır** bir model mümkün. Bu yüzden Precision, Recall, PR-AUC ve para üzerinden değerlendireceğiz.

## İş varsayımları (kampanya ekonomisi)
Elimizde gerçek kampanya maliyeti yok; bu yüzden **açıkça yazılmış varsayımlar** kullanıyoruz ve sonucun bu varsayımlara ne kadar duyarlı olduğunu da (Notebook 08) göstereceğiz.
"""),
code("""
display(pd.Series(C.ASSUMPTIONS, name="varsayım").to_frame())
fee = 351.86
value = fee * C.ASSUMPTIONS["customer_value_months"] * C.ASSUMPTIONS["margin"]
gain_if_success = C.ASSUMPTIONS["offer_success_rate"] * value
breakeven = C.ASSUMPTIONS["offer_cost_tl"] / gain_if_success
display(Markdown(f"Ortalama müşteri için: kurtarılan müşterinin değeri ≈ **{value:,.0f} TL**, beklenen kazanç (başarı olasılığıyla) ≈ **{gain_if_success:,.0f} TL**. "
                 f"Aranan müşteri başına maliyet **{C.ASSUMPTIONS['offer_cost_tl']:.0f} TL** ise, kampanyanın **başabaş precision'ı ≈ %{breakeven:.1%}**. "
                 f"Yani aradığımız grubun en az %{breakeven:.0%}'i gerçekten ayrılacaksa kampanya kâr eder. Taban oran %{churn_rate:.1%} olduğu için model bu eşiği aşan bir *kesim* bulabilmeli."))
"""),
md("""
## Proje riskleri / sınırlılıklar (baştan yazalım)
1. **Veri sentetik.** Gerçek müşteri davranışını temsil ettiği garanti değil.
2. **Az sayıda pozitif örnek** (~148 ayrılan). Test kümesinde sadece ~30 ayrılan olacak → metrikler oynak; bunu güven aralığıyla göstereceğiz.
3. **Korelasyon ≠ nedensellik.** "Memnuniyeti düşük müşteri ayrılıyor" demek, memnuniyeti artırınca ayrılmayacağı anlamına gelmez. Kampanya etkisi A/B testiyle ölçülmeli.
4. **Eşik kararı iş maliyetine bağlı.** Bizim maliyet rakamlarımız varsayımdır.
""")
]

NB02 = [
md("""
# 02 — Veriyi Anlama (Data Understanding)

**Kural: Bu notebook'ta veriyi DEĞİŞTİRMİYORUZ.** Sadece bakıyor, soruyor ve sorunları listeliyoruz.
Ham veri `data/raw/` altında dokunulmaz kalır; temizleme Notebook 03'te yeni bir kopya üzerinde yapılır.

## 2.1 Veriyi okuma
`monthly_fee_tl` sütununu `dtype="object"` (metin) olarak okuyoruz. **Neden?** Pandas bu sütunda `399TL`, `free` gibi değerler görürse zaten metin yapar; bilinçli olarak metin okuyarak bu bozuk değerleri *görebiliriz*.
"""),
code("""
df_raw = pd.read_csv(C.RAW_PATH, dtype={"monthly_fee_tl": "object"})
display(df_raw.head())
print("Satır, sütun:", df_raw.shape)
"""),
md("## 2.2 Veri tipleri ve eksiklik özeti — `info()`"),
code("df_raw.info()"),
md("""
**Okuma:** `monthly_fee_tl` metin (object) — sayısal olmalıydı. `age`, `region`, `monthly_usage_hours`, `payment_method`, `satisfaction_score` sütunlarında boş değerler var.
"""),
md("## 2.3 Sayısal özet — `describe()`"),
code("display(df_raw.describe().T)"),
md("""
**Okuma (kırmızı bayraklar):**
* `age`: min **-3**, max **130** → imkânsız yaşlar.
* `monthly_usage_hours`: min **-5**, max **180** → negatif değerler imkânsız, 100 saat üstü de şüpheli (veri sözlüğüne göre makul aralık 0–100).
* `satisfaction_score`: min **0**, max **11** → ölçek 1–10 olmalı.
* Diğerleri (tenure, discount, sayaçlar) ilk bakışta makul.
"""),
md("## 2.4 Kategorik sütunlar"),
code("""
cat_cols = ["region", "device_type", "plan", "contract_type", "payment_method"]
for col in cat_cols:
    print("\\n==", col)
    display(df_raw[col].value_counts(dropna=False).to_frame("adet").T)
"""),
md("""
**Okuma:** `plan` içinde `Basic` ve `basic`, `contract_type` içinde `Monthly` ve `monthly` ... aynı kategori iki farklı yazılmış. Bunu makineye bırakırsak ayrı kategori sayar.
"""),
md("## 2.5 Eksik değer raporu"),
code("""
missing = pd.DataFrame({"eksik_adet": df_raw.isna().sum(), "eksik_yuzde": df_raw.isna().mean() * 100})
missing = missing[missing.eksik_adet > 0].sort_values("eksik_yuzde", ascending=False)
display(missing)
fig, ax = plt.subplots(figsize=(8, 3.5))
sns.barplot(x=missing.index, y=missing["eksik_yuzde"], color="#4c78a8", ax=ax)
ax.set_ylabel("Eksik (%)"); ax.set_title("Sütunlara göre eksik değer oranı"); plt.xticks(rotation=30); plt.tight_layout(); plt.show()
"""),
md("""
**Not:** Bu tabloda `monthly_fee_tl` yok; çünkü içinde `"free"`, `"?"` gibi *metin* var ve bunlar Pandas için "boş" değil, geçerli bir string. Bu, **eksik değerin her zaman NaN olarak gelmediğinin** klasik örneği. Şimdi bozuk ücret değerlerine bakalım.
"""),
code("""
fee_numeric = pd.to_numeric(df_raw["monthly_fee_tl"], errors="coerce")
text_fee = df_raw.loc[fee_numeric.isna() & df_raw["monthly_fee_tl"].notna(), "monthly_fee_tl"]
display(Markdown(f"Sayıya çevrilemeyen ücret değerleri (**{len(text_fee)}** adet): " + ", ".join(f"`{v}`" for v in text_fee)))
display(fee_numeric.describe().to_frame("sayıya çevrilebilen ücretler").T)
display(df_raw.loc[(fee_numeric <= 0) | (fee_numeric > 1000), ["customer_id", "plan", "discount_rate", "monthly_fee_tl"]])
"""),
md("""
## 2.6 Tekrarlayan kayıtlar — iki ayrı problem!
1. **Birebir aynı satır** (tüm sütunlar aynı).
2. **Aynı `customer_id`, farklı bilgiler** (çelişki). Bu çok daha sinsi: hangisi doğru?
"""),
code("""
exact = df_raw.duplicated(keep=False)
display(Markdown(f"Birebir tekrar eden satır (ilk kopya hariç): **{df_raw.duplicated().sum()}**"))
id_dups = df_raw[df_raw.duplicated(subset='customer_id', keep=False) & ~df_raw.duplicated(keep=False)]
display(Markdown(f"Birebir kopya olmayıp aynı `customer_id`'yi paylaşan satır: **{len(id_dups)}** ({id_dups.customer_id.nunique()} müşteri)"))
display(id_dups.sort_values("customer_id")[["customer_id", "monthly_usage_hours", "support_calls_6m", "monthly_fee_tl", "satisfaction_score"]].assign(satir_no=lambda d: d.index))
"""),
md("""
**Okuma:** Çelişkili çiftlerde fark genelde `monthly_usage_hours` (+birkaç saat) ve `support_calls_6m` (+1). Çelişen satırlar dosyanın **sonuna** eklenmiş (satır no ≥ 1812). Zaman damgamız olmadığı için hangisinin doğru olduğunu *kanıtlayamayız*; Notebook 03'te bunu bir **varsayım** olarak ele alacağız.
"""),
md("""
## 2.7 İş kurallarına göre geçerlilik kontrolü
Veri sözlüğündeki aralıkları (`config.RANGE_RULES`) ham veriye uyguluyoruz — **hiçbir şeyi silmeden sayıyoruz.**
"""),
code("""
rows = []
for col, (lo, hi) in C.RANGE_RULES.items():
    bad = df_raw[col].notna() & ~df_raw[col].between(lo, hi)
    rows.append({"sütun": col, "geçerli_aralık": f"[{lo}, {hi}]", "geçersiz_adet": int(bad.sum()),
                 "geçersiz_değerler": sorted(df_raw.loc[bad, col].unique().tolist())})
display(pd.DataFrame(rows))
"""),
md("""
## 2.8 Alanlar arası tutarlılık: ücret nasıl oluşuyor?
Tek tek sütunlara bakmak yetmez; **sütunlar birbirini açıklayabilir.** `monthly_fee_tl` ile `plan` ve `discount_rate` arasında bir ilişki var mı?
"""),
code("""
plan_clean = df_raw["plan"].str.strip().str.title()
implied_list_price = (fee_numeric / (1 - df_raw["discount_rate"])).round(2)
display(pd.crosstab(plan_clean, implied_list_price).loc[:, lambda d: d.sum() > 3])
"""),
md("""
**Büyük keşif:** `ücret / (1 − indirim)` hesapladığımızda değerler neredeyse hep **249 (Basic), 399 (Standard), 599 (Premium)** çıkıyor. Yani:

$$\\text{ücret} = \\text{plan liste fiyatı} \\times (1 - \\text{indirim})$$

Bu kural iki işe yarayacak (Notebook 03): eksik/bozuk ücreti **tahmin etmek yerine kesin olarak geri hesaplamak** ve plan ile ücret *çeliştiğinde* hatalı alanı yakalamak. Yukarıdaki tabloda yalnızca en az 4 gözlemi olan fiyatları gösterdik; kuralı bozan nadir hücreleri hemen aşağıda ayrıca listeliyoruz.
"""),
code("""
cross = pd.DataFrame({"plan": plan_clean, "implied": implied_list_price})
price_to_plan = {v: k for k, v in C.PLAN_BASE_PRICE.items()}
cross["implied_plan"] = cross["implied"].map(price_to_plan)
conflict = cross[cross.implied_plan.notna() & (cross.implied_plan != cross.plan)]
display(Markdown(f"**Plan etiketi ile ücretin işaret ettiği plan çelişen satır sayısı: {len(conflict)}**"))
display(df_raw.loc[conflict.index, ["customer_id", "plan", "discount_rate", "monthly_fee_tl"]].assign(ucretin_isaret_ettigi_plan=conflict.implied_plan))
"""),
md("""
## 2.9 Denetim özeti (veri kalitesi sorun listesi)
| # | Sorun | Nerede | Çözüm (Notebook 03) |
|---|---|---|---|
| 1 | Ücret metin içeriyor (`free`, `?`, `399TL` …) | `monthly_fee_tl` | Sayıya çevir; formülle doğrula/geri kazan |
| 2 | Aralık dışı değerler | `age`, `monthly_usage_hours`, `satisfaction_score`, ücret | NaN yap, eğitimde imputer doldursun |
| 3 | Büyük/küçük harf tutarsızlığı | `region`, `plan`, `contract_type`, `payment_method` | Standartlaştır |
| 4 | Plan ↔ ücret çelişkisi | `plan` | Ücret/indirimden plan türet |
| 5 | Birebir tekrar satırlar | tüm satır | Sil |
| 6 | Aynı ID, farklı bilgi | `customer_id` | İlk kaydı tut (varsayım) + denetim dosyası |
| 7 | Gerçek eksik değerler | 5 sütun | **Silme, doldurma Pipeline içinde** (leakage olmasın) |
""")
]
