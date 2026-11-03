from tools.nbbuild import md, code

NB03 = [
md("""
# 03 — Veri Temizleme (Data Cleaning)

**Altın kural:** ham veriyi değiştirmiyoruz → `df = df_raw.copy()`. Temizleme mantığının tamamı `src/data_cleaning.py` içinde; bu notebook her adımı **gösteriyor ve neden yaptığını açıklıyor**, sonunda modülün sonucunu referans temiz veriyle doğruluyor.

> **Neden fonksiyon/modül?** Aynı temizliği hem burada, hem testlerde, hem yeni gelen veriyle tekrarlayabilmek için. Notebook'ta elle yazılan temizlik tekrar üretilemez.
"""),
code("""
from src.data_cleaning import read_raw, clean_data
raw = read_raw()
df = raw.copy()   # Ham veriyi korumak için kopya. Bundan sonra df üzerinde çalışırız.
print("Ham:", raw.shape)
"""),
md("""
## Adım 1 — Ücret sütunu: metin → sayı (`pd.to_numeric(errors="coerce")`)
`errors="coerce"` sayıya çevrilemeyen değeri hata vermek yerine `NaN` yapar. Ama dikkat: **kör `coerce` bilgi kaybettirir.** Önce ne kaybedeceğimize bakalım:
"""),
code("""
naive = pd.to_numeric(df["monthly_fee_tl"], errors="coerce")
lost = df.loc[naive.isna() & df["monthly_fee_tl"].notna(), ["customer_id", "plan", "discount_rate", "monthly_fee_tl"]]
display(lost)
"""),
md("""
`399TL` sayıya çevrilebilir görünüyor (birim eki). Peki 399 doğru mu? Kurala göre bakalım: bu müşteri **Premium + %20 indirim** → doğru ücret `599 × 0.80 = 479.2`. Yani `399TL` bir birim yazım hatası değil, **tamamen yanlış değer**.
"""),
code("""
row = df[df.monthly_fee_tl == "399TL"][["customer_id", "plan", "discount_rate", "monthly_fee_tl"]]
row["kuralın_beklediği_ücret"] = row["plan"].map(C.PLAN_BASE_PRICE) * (1 - row["discount_rate"])
display(row)
"""),
md("""
> **Ders:** "Düzeltilebilir görünen" bir değeri, doğrulayabildiğin bir kuralla çapraz kontrol etmeden güvenme. Temizleme modülümüz bu yüzden: (1) metni sayıya çevirir, (2) aralık dışını eler, (3) plan×indirim kuralıyla **tutarlılık kontrolü** yapar ve sapanları kuralla değiştirir (hepsi loglanır).

## Adım 2 — Kategorileri standartlaştırma
`strip()` baştaki/sondaki boşluğu, `lower()` harf farkını giderir; sonra sözlükle kanonik yazıma çeviririz. **Sözlükte olmayan değer sessizce NaN olmasın diye** modül bunu loga yazar.
"""),
code("""
before = {c: raw[c].nunique() for c in C.CATEGORICAL_COLS}
res = clean_data(raw)
after = {c: res.clean[c].nunique() for c in C.CATEGORICAL_COLS}
display(pd.DataFrame({"önce_farklı_değer": before, "sonra_farklı_değer": after}))
"""),
md("""
## Adım 3 — Aralık dışı değerler → `NaN`
Yaş = 130 gibi imkânsız değerleri **silmek** yerine *eksik* yaparız (`NaN`). Neden?
* Satırı silmek, o müşterinin diğer 14 doğru bilgisini de çöpe atar.
* Aralık dışı değeri medyanla **burada** doldurmak ise *veri sızıntısı* olur (aşağıda açıklıyoruz). Doldurmayı Pipeline'a bırakıyoruz.

## Adım 4 — Plan↔ücret çelişkisi ve ücretin geri kazanımı
Notebook 02'de ücret = plan fiyatı × (1 − indirim) kuralını keşfettik. Uygulama sırası önemli:
1. Geçerli ücret + indirim, başka bir planın liste fiyatına *kuruşu kuruşuna* denk geliyorsa **plan etiketi yanlıştır** (3 olası değerli etiketin hatalı olma ihtimali, rastgele bir ücretin başka bir planla tam eşleşme ihtimalinden çok daha yüksek).
2. Sonra ücret, düzeltilmiş plan ve indirimden **hesaplanır**; eksik/bozuk/sapan ücretler bununla değiştirilir.
"""),
code("""
display(res.log)
"""),
md("""
## Adım 5 — Tekrarlar
1. **Birebir tekrar** (12 satır): ilk kopyayı tut.
2. **Çelişkili ID** (8 müşteri, 16 satır): hangisinin doğru olduğunu zaman damgası olmadan bilemeyiz. İki seçenek var:

| Seçenek | Sonuç | Sorun |
|---|---|---|
| Çelişkili ID'lerin **hepsini sil** | 1820 − 12 − 16 = **1792** satır | 8 gerçek müşteriyi tamamen kaybederiz |
| **İlk kaydı tut** (dosya sırasına göre) | **1800** satır | Bir *varsayım* yapmış oluruz |

Varsayımımız: dosya sonuna eklenen ikinci kayıt geç gelen/yinelenen satırdır. Tüm çelişkiler `conflicting_customers.csv`'ye yazılır ki sonradan kontrol edilebilsin.
"""),
code("""
display(res.conflicts[["customer_id", "source_row", "monthly_usage_hours", "support_calls_6m", "kept"]].head(8))
"""),
md("""
> ⚠️ **Dürüst not:** Bu bir varsayımdır. Aşağıda "referans temiz veri" ile karşılaştırınca ilk kaydın doğru olduğunu göreceğiz; **gerçek bir projede referansın olmaz** — CRM'den güncelleme zamanını almalı ya da iş birimine sormalısın.

## Adım 6 — Doğrulama: referans temiz veriyle karşılaştırma
Elimizde (ödev/sentetik veri olduğu için) gerçek değerleri içeren bir `clean_reference` dosyası var. Bunu **model eğitiminde ASLA kullanmıyoruz**; sadece temizleme kodumuzun doğruluğunu ölçmek için kullanıyoruz.
"""),
code("""
ref = pd.read_csv(C.REFERENCE_PATH).set_index("customer_id")
c = res.clean.set_index("customer_id").loc[ref.index]
rows = []
for col in ref.columns:
    ok = c[col].notna()
    if ref[col].dtype.kind in "if":
        diff = (~np.isclose(c.loc[ok, col].astype(float), ref.loc[ok.index[ok], col].astype(float))).sum()
    else:
        diff = (c.loc[ok, col].astype(str) != ref.loc[ok.index[ok], col].astype(str)).sum()
    rows.append({"sütun": col, "karşılaştırılan": int(ok.sum()), "referanstan_farklı": int(diff), "hâlâ_eksik(NaN)": int((~ok).sum())})
report = pd.DataFrame(rows); display(report)
"""),
md("""
**Okuma:**
* `plan`, `monthly_fee_tl`, `age`, `monthly_usage_hours`, `satisfaction_score`, `churn` vb. → **0 fark**.
* `region`, `contract_type`, `payment_method` sütunlarında küçük harfle yazılmış hücrelerin (ör. `marmara`, `monthly`, `credit card`) bir kısmı referanstan farklı: bu hücrelerin **değeri de yanlışmış** (sadece yazımı değil). Bunu hiçbir kural yakalayamaz çünkü `marmara` geçerli bir bölge adıdır. Toplam etkilenen hücre sayısı çok küçük (aşağıda).
"""),
code("""
n_diff = int(report.loc[report.sütun.isin(["region", "contract_type", "payment_method"]), "referanstan_farklı"].sum())
total_cells = res.clean.shape[0] * (res.clean.shape[1] - 1)
display(Markdown(f"Referanstan farklı kalan hücre: **{n_diff}** / {total_cells:,} (**%{n_diff/total_cells:.3%}**). Bu, veri kalitesinde 'bilinmeyen bilinmeyen' sınıfıdır: tespit edemediğin hata."))
"""),
md("""
## Adım 7 — Eksik değerler: doldurmalı mıyız?
Burada **doldurmuyoruz.** Çünkü medyan/mod gibi değerleri *tüm veriden* hesaplayıp sonra train/test ayırırsak, test kümesinin istatistiği eğitime sızar. Buna **Data Leakage** denir. Doldurma, eğitim verisinden öğrenilen değerlerle **Pipeline içinde** yapılacak (Notebook 07).

Kalan eksikler:
"""),
code("""
miss = res.clean.isna().sum(); display(miss[miss > 0].to_frame("eksik_adet"))
"""),
md("## Adım 8 — Kaydet"),
code("""
res.clean.to_csv(C.CLEAN_PATH, index=False)
res.conflicts.to_csv(C.CONFLICTS_PATH, index=False)
res.log.to_csv(C.CLEANING_LOG_PATH, index=False)
print("Kaydedildi:", C.CLEAN_PATH.relative_to(C.ROOT), res.clean.shape)
"""),
]

NB04 = [
md("""
# 04 — Keşifsel Veri Analizi (EDA)

**Amaç:** "Kim ayrılıyor?" sorusuna grafiklerle ve oranlarla cevap aramak. Model kurmadan önce **hangi değişkenlerin işe yarayabileceğine dair hipotez üretiriz.**

⚠️ EDA tamamen *betimleyicidir*; burada gördüğümüz ilişkiler nedensellik kanıtı değildir.
"""),
code("""
df = pd.read_csv(C.CLEAN_PATH)
print(df.shape)
display(df.head())
"""),
md("## 4.1 Hedef değişken: churn dağılımı"),
code("""
n, k = len(df), int(df.churn.sum())
display(Markdown(f"Müşteri: **{n}**, ayrılan: **{k}**, churn oranı: **{k/n:.2%}**, elde tutma oranı: **{1-k/n:.2%}**"))
fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.countplot(data=df, x="churn", palette=["#4c78a8", "#e45756"], ax=ax); ax.set_xticklabels(["Kaldı (0)", "Ayrıldı (1)"]); ax.set_title("Churn dağılımı")
plt.tight_layout(); plt.show()
"""),
md("""
## 4.2 Sayısal değişkenlerin dağılımı (histogram)
Histogram bir değişkenin hangi aralıkta yoğunlaştığını gösterir. Çarpıklık/aykırı değer var mı diye bakarız.
"""),
code("""
num_cols = C.NUMERIC_COLS
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col].dropna(), bins=25, ax=ax, color="#4c78a8"); ax.set_title(col)
plt.tight_layout(); plt.show()
"""),
md("""
**Okuma:** `monthly_fee_tl` üç plan fiyatından türediği için *kümeli* (ayrık tepeler). Sayaç değişkenleri (gecikme, destek, şikâyet) sağa çarpık: çoğu müşteri 0-1.
"""),
md("""
## 4.3 Ayrılan vs ayrılmayan: sayısal değişkenler (boxplot)
Boxplot: medyan, çeyrekler ve IQR dışındaki noktalar. **Aykırı görünen her değer hata değildir** — 6 destek araması nadir ama gerçek olabilir.
"""),
code("""
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, col in zip(axes.ravel(), [c for c in num_cols if c != "monthly_fee_tl"]):
    sns.boxplot(data=df, x="churn", y=col, ax=ax, palette=["#4c78a8", "#e45756"]); ax.set_title(col); ax.set_xlabel("churn")
plt.tight_layout(); plt.show()
"""),
code("""
display(df.groupby("churn")[["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m", "late_payments_6m", "monthly_usage_hours", "age"]].agg(["mean", "median"]).round(2).T)
"""),
md("""
**Okuma:** Ayrılanlar daha **kısa süreli aboneler**, memnuniyetleri **daha düşük**, şikâyet/destek sayıları **daha yüksek**. Kullanım saati ve yaş neredeyse ayırt edici değil.

## 4.4 Kategorik değişkenlerde churn oranı — **örneklem büyüklüğüyle birlikte!**
Bir segmentte %20 churn görmek, o segmentte 10 müşteri varsa anlamsızdır. Bu yüzden her çubuğun yanına **%95 Wilson güven aralığı** koyuyoruz.
"""),
code("""
from statsmodels.stats.proportion import proportion_confint

def segment_table(df, col):
    g = df.groupby(col, dropna=False)["churn"].agg(["size", "sum"]).rename(columns={"size": "musteri", "sum": "ayrilan"})
    g["churn_orani"] = g["ayrilan"] / g["musteri"]
    ci = [proportion_confint(a, n, method="wilson") for a, n in zip(g["ayrilan"], g["musteri"])]
    g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return g.reset_index().rename(columns={col: "segment"}).assign(degisken=col)

def plot_segments(col, ax):
    t = segment_table(df.assign(**{col: df[col].fillna("(eksik)")}), col).sort_values("churn_orani", ascending=False)
    ax.bar(t["segment"].astype(str), t["churn_orani"], color="#4c78a8")
    ax.errorbar(t["segment"].astype(str), t["churn_orani"], yerr=[t["churn_orani"] - t["ci_low"], t["ci_high"] - t["churn_orani"]], fmt="none", color="k", capsize=3)
    for i, (r, n) in enumerate(zip(t["churn_orani"], t["musteri"])): ax.text(i, 0.002, f"n={n}", ha="center", va="bottom", color="white", fontsize=8)
    ax.axhline(df.churn.mean(), color="red", ls="--", lw=1); ax.set_title(f"{col}"); ax.tick_params(axis="x", rotation=30)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), ["contract_type", "plan", "region", "payment_method", "device_type"]): plot_segments(col, ax)
axes.ravel()[-1].axis("off"); plt.tight_layout(); plt.show()
"""),
md("""
**Okuma:**
* **Sözleşme türü en güçlü ayırıcı:** Aylık sözleşmelerde churn çok yüksek, yıllık/iki yıllıkta çok düşük (güven aralıkları ayrışıyor).
* **Plan:** Standard daha düşük; Basic/Premium yüksek (aralıklar kısmen örtüşüyor → dikkatli ol).
* **Bölge:** Marmara düşük görünüyor ama aralıklar geniş; ayrıca bölge etkisi *sözleşme dağılımından* kaynaklanıyor olabilir (aşağıda kontrol).
* **Ödeme yöntemi ve cihaz:** belirgin bir fark yok.
* Kırmızı kesikli çizgi genel churn oranıdır.

## 4.5 Abonelik süresi ve memnuniyet: eğilim
"""),
code("""
df["tenure_group"] = pd.cut(df.tenure_months, [0, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
df["sat_band"] = pd.cut(df.satisfaction_score, [0, 3, 4, 5, 6, 7, 8, 10], labels=["≤3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10"])
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["tenure_group", "sat_band", "complaints_6m"]):
    t = segment_table(df.assign(**{col: df[col].astype(str)}), col)
    t = t.sort_values("segment") if col != "sat_band" else t.set_index("segment").loc[["≤3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10"]].reset_index()
    ax.bar(t["segment"], t["churn_orani"], color="#4c78a8")
    ax.errorbar(t["segment"], t["churn_orani"], yerr=[t["churn_orani"] - t["ci_low"], t["ci_high"] - t["churn_orani"]], fmt="none", color="k", capsize=3)
    ax.axhline(df.churn.mean(), color="red", ls="--", lw=1); ax.set_title(col)
plt.tight_layout(); plt.show()
"""),
md("""
**Okuma:** Abonelik süresi arttıkça churn düşüyor (yeni müşteri riski). Memnuniyet düştükçe churn artıyor. Şikâyet sayısı arttıkça artıyor (3+ şikâyette örnek az, aralıklar geniş).

## 4.6 Karıştırıcı (confounding) kontrolü: bölge etkisi gerçek mi?
Marmara'da churn düşük. Ama belki Marmara'da sadece daha fazla yıllık sözleşmeli müşteri var? Sözleşme türünü sabitleyip bölgeye bakalım.
"""),
code("""
display(pd.crosstab(df.region, df.contract_type, normalize="index").round(3))
display(df.pivot_table(index="region", columns="contract_type", values="churn", aggfunc="mean").round(3))
"""),
md("""
**Okuma:** Sözleşme dağılımı bölgeler arasında benzer (aylık %53-59); ayrıca Marmara aylık sözleşmelilerde de diğer bölgelerden düşük (%7 civarı). Yani Marmara'nın düşük oranı sözleşme dağılımıyla açıklanmıyor görünüyor. Ama her hücrede müşteri sayısı küçük olduğundan, bu farkın **şans olup olmadığını** Notebook 05'te istatistiksel testle sorgulayacağız.

## 4.7 Korelasyon matrisi
Korelasyon −1 ile +1 arasındadır ve yalnızca **doğrusal** ilişkiyi ölçer. `churn` ikili olduğu için bu "point-biserial" korelasyondur.
"""),
code("""
corr = df[C.NUMERIC_COLS + ["churn"]].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, mask=np.triu(np.ones_like(corr, dtype=bool), 1)); ax.set_title("Korelasyon (churn dahil)")
plt.tight_layout(); plt.show()
display(corr["churn"].drop("churn").sort_values().to_frame("churn ile korelasyon"))
"""),
md("""
**Okuma:** Tüm korelasyonlar küçük (|r| ≲ 0,14). Bu, **churn'ün tek bir değişkenle açıklanamayacağını** ve modelden mucize beklememek gerektiğini söyler. `monthly_fee_tl` ile `discount_rate` arasındaki ilişki plan/indirim yapısından gelir.

## 4.8 Eksik olmak bilgi taşıyor mu? (missing-not-at-random kontrolü)
Bir değer *neden* eksik? Eksik olan müşteriler farklı mı davranıyor?
"""),
code("""
rows = []
for col in ["age", "region", "monthly_usage_hours", "payment_method", "satisfaction_score"]:
    m = df[col].isna()
    rows.append({"sütun": col, "eksik_adet": int(m.sum()), "churn_eksik": df.loc[m, "churn"].mean(), "churn_dolu": df.loc[~m, "churn"].mean()})
display(pd.DataFrame(rows))
"""),
md("""
**Okuma:** `region` eksik olanlarda churn %18,8 görünüyor (n=32, yani ~6 kişi): dikkat çekici ama örnek çok küçük. İstatistiksel olarak anlamlı mı? → Notebook 05.

## 4.9 EDA sonuç özeti
1. Dengesiz hedef (~%8). Accuracy kullanılmayacak.
2. En güçlü sinyaller: **sözleşme türü, abonelik süresi, memnuniyet, şikâyet/destek**.
3. Zayıf/ayırt edici olmayan: cihaz, ödeme yöntemi, yaş, kullanım saati.
4. Tek tek ilişkiler zayıf → makul model başarısı **orta düzeyde** olacak (ROC-AUC ≈ 0,7-0,75). Çok yüksek bir skor görürsek şüphelenmeliyiz (sızıntı?).
""")
]
