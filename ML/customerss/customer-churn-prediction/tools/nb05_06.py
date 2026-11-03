from tools.nbbuild import md, code

NB05 = [
md("""
# 05 — İstatistiksel Analiz

EDA bize örüntüleri **gösterdi**. İstatistik bize şunu söyler: *"Bu fark şans eseri olabilir mi?"*

| Soru | Araç |
|---|---|
| Churn oranı ne kadar kesin ölçülmüş? | **Güven aralığı** (Wilson) |
| Kategorik değişken ↔ churn ilişkisi var mı? | **Ki-kare testi** + etki büyüklüğü **Cramér's V** |
| Sayısal değişkenin dağılımı iki grupta farklı mı? | **Mann-Whitney U** + etki büyüklüğü |
| Çok test yaptık; "tesadüfen anlamlı" çıkanları nasıl ayıklarız? | **Çoklu test düzeltmesi** (Benjamini-Hochberg FDR) |
| Diğer değişkenler sabitken etki nedir? | **Lojistik regresyon (statsmodels)** → odds ratio |
"""),
code("""
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

df = pd.read_csv(C.CLEAN_PATH)
from src.features import add_features
df = add_features(df)
"""),
md("""
## 5.1 Churn oranı için %95 güven aralığı
Elimizdeki 1800 müşteri bir **örneklem**. Gerçek (popülasyon) churn oranı tam olarak %8,22 olmayabilir. **Wilson aralığı** oranlar için kullanılan, küçük oranlarda normal yaklaşıma göre daha güvenilir bir yöntemdir.
"""),
code("""
k, n = int(df.churn.sum()), len(df)
lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
display(Markdown(f"Churn oranı **{k/n:.2%}** ({k}/{n}); %95 güven aralığı: **[{lo:.2%}, {hi:.2%}]**"))
"""),
md("""
## 5.2 Kategorik değişkenler: Ki-kare + Cramér's V

* **H0:** Değişken ile churn bağımsızdır. **H1:** İlişki vardır.
* p < 0,05 ise H0 reddedilir — ama **p-değeri ilişkinin gücünü söylemez** (büyük örneklemde küçük fark da "anlamlı" çıkar). Bu yüzden **Cramér's V** (0 = ilişki yok, 1 = mükemmel) ile gücü de raporlarız.
* Ki-kare varsayımı: beklenen hücre sayıları ≥ 5. Aşağıda en küçük beklenen değeri de gösteriyoruz.
"""),
code("""
def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum(); k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k))

rows = []
for col in ["contract_type", "plan", "region", "payment_method", "device_type", "tenure_group"]:
    sub = df.dropna(subset=[col])                      # eksik kategori testten çıkarılır (adet raporlanır)
    table = pd.crosstab(sub[col], sub["churn"])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    rows.append({"değişken": col, "n_kullanılan": len(sub), "chi2": chi2, "dof": dof, "p_değeri": p,
                 "cramers_V": cramers_v(table), "min_beklenen_hücre": expected.min()})
cat_tests = pd.DataFrame(rows)
cat_tests["p_BH_düzeltmeli"] = multipletests(cat_tests["p_değeri"], method="fdr_bh")[1]
display(cat_tests.sort_values("p_değeri"))
"""),
md("""
**Okuma:** `contract_type` ve `tenure_group` çok güçlü kanıt (düzeltmeden sonra da). `plan` ham p-değerinde sınırda anlamlı (≈0,04) ama **çoklu test düzeltmesinden sonra anlamlılığını kaybediyor** (≈0,08). `region`, `device_type`, `payment_method` anlamlı değil. **Cramér's V değerleri küçük** (en büyüğü ≈ 0,15): ilişkiler gerçek ama zayıf-orta büyüklükte. *EDA'da Marmara düşük görünmüştü; ama bu fark çoklu test düzeltmesinden sonra şansla açıklanabilir bir düzeyde.*

> **Çoklu test sorunu:** 6 test yaparsan, hepsi gerçekte ilişkisiz olsa bile ~%26 ihtimalle en az biri p<0,05 çıkar. Benjamini-Hochberg düzeltmesi bunu kontrol eder.

## 5.3 Sayısal değişkenler: Mann-Whitney U
Neden t-testi değil? Sayaç değişkenleri (0,1,2…) ve puanlar normal dağılmıyor; Mann-Whitney **dağılım varsayımı yapmaz** (sıra tabanlıdır).
**Etki büyüklüğü:** "rank-biserial" korelasyon `r = 2U/(n₁n₂) − 1` — ayrılan grubun değerlerinin ayrılmayandan büyük olma eğilimini ölçer (−1…+1).
"""),
code("""
rows = []
for col in ["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m", "late_payments_6m", "monthly_usage_hours", "age", "discount_rate"]:
    g0 = df.loc[df.churn == 0, col].dropna(); g1 = df.loc[df.churn == 1, col].dropna()
    U, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    rows.append({"değişken": col, "medyan_ayrılan": g1.median(), "medyan_kalan": g0.median(),
                 "U": U, "p_değeri": p, "rank_biserial_r": 2 * U / (len(g1) * len(g0)) - 1})
num_tests = pd.DataFrame(rows)
num_tests["p_BH_düzeltmeli"] = multipletests(num_tests["p_değeri"], method="fdr_bh")[1]
display(num_tests.sort_values("p_değeri"))
"""),
md("""
**Okuma:** `tenure_months`, `satisfaction_score`, `complaints_6m`, `support_calls_6m` anlamlı farklı; `monthly_usage_hours`, `age`, `late_payments_6m`, `discount_rate` anlamlı değil. Etki büyüklükleri (|r| ≈ 0,15–0,27) **küçük-orta**. Dikkat: `late_payments_6m` (ödeme gecikmesi) sezgisel olarak önemli görünse de bu veride anlamlı bir fark yok.

## 5.4 Eksiklik bilgi taşıyor mu? (Fisher exact testi)
`region` eksik olan 32 müşteride churn çok yüksek görünüyordu. Küçük örneklemde ki-kare yerine **Fisher exact** kullanılır.
"""),
code("""
rows = []
for col in ["age", "region", "monthly_usage_hours", "payment_method", "satisfaction_score"]:
    m = df[col].isna()
    table = [[(m & (df.churn == 1)).sum(), (m & (df.churn == 0)).sum()], [(~m & (df.churn == 1)).sum(), (~m & (df.churn == 0)).sum()]]
    odds, p = stats.fisher_exact(table)
    rows.append({"sütun": col, "eksik_adet": int(m.sum()), "churn_eksik": df.loc[m, "churn"].mean(), "churn_dolu": df.loc[~m, "churn"].mean(), "p_değeri": p})
miss_tests = pd.DataFrame(rows); miss_tests["p_BH_düzeltmeli"] = multipletests(miss_tests["p_değeri"], method="fdr_bh")[1]
display(miss_tests)
"""),
md("""
**Okuma:** Düzeltmeden sonra eksiklik anlamlı bilgi taşıdığına dair **güçlü kanıt yok** (`region` en yakın aday ama n küçük). Bu yüzden basit imputasyon (medyan/mod) kullanıyoruz; ayrı "eksiklik göstergesi" sütunu eklemiyoruz. *Kanıt yokluğu, yokluğun kanıtı değildir:* örnek büyüdükçe bu karar yeniden gözden geçirilmeli.

## 5.5 Çoklu doğrusallık: `monthly_fee_tl` yeni bilgi taşıyor mu?
`ücret = plan fiyatı × (1 − indirim)` olduğunu biliyoruz. Ücreti plan ve indirimden regresyonla tahmin edersek R² kaç çıkar?
"""),
code("""
expected = df["plan"].map(C.PLAN_BASE_PRICE) * (1 - df["discount_rate"])
r2 = np.corrcoef(expected, df["monthly_fee_tl"])[0, 1] ** 2
display(Markdown(f"`plan fiyatı × (1 − indirim)` ile gerçek ücret arasındaki R² = **{r2:.6f}**, en büyük mutlak fark = **{(expected - df.monthly_fee_tl).abs().max():.4f} TL**"))
"""),
md("""
**Sonuç:** Ücret, plan ve indirimden **tam olarak türüyor** (R² = 1). Modele eklemek yeni bilgi vermez, sadece çoklu doğrusallık yaratır. Bu yüzden `monthly_fee_tl` model girdilerinde yok; ama **iş analizinde** (gelir, riskteki gelir) kullanılıyor.

## 5.6 Çok değişkenli bakış: Lojistik regresyon (statsmodels) ve Odds Ratio
Tek tek testler, değişkenler birbirine bağlıyken yanıltabilir (ör. kısa süreli müşteriler aynı zamanda daha çok aylık sözleşmeli olabilir). **Diğerleri sabitken** her değişkenin etkisini görmek için lojistik regresyon kullanıyoruz.

* **Odds ratio (OR) = exp(katsayı).** OR > 1: risk artar; < 1: azalır. Sayısal değişkenleri standartlaştırdık → OR, "1 standart sapma artışın" etkisidir.
* Tam gözlemli satırlar kullanılır (eksik satisfaction nedeniyle ~82 satır dışarıda). Bu **çıkarım** analizidir; tahmin modelini Notebook 07'de ayrıca kuruyoruz.
"""),
code("""
cols = ["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m", "late_payments_6m"]
d = df.dropna(subset=cols + ["contract_type", "plan"]).copy()
Z = (d[cols] - d[cols].mean()) / d[cols].std()
X = pd.concat([Z, pd.get_dummies(d[["contract_type", "plan"]], drop_first=True).astype(float)], axis=1)
X = sm.add_constant(X)
logit = sm.Logit(d["churn"], X).fit(disp=False)
ci = logit.conf_int()
res = pd.DataFrame({"OR": np.exp(logit.params), "OR_low": np.exp(ci[0]), "OR_high": np.exp(ci[1]), "p": logit.pvalues}).drop(index="const")
display(res.round(3))
display(Markdown(f"Kullanılan satır: **{len(d)}**, pseudo R² = **{logit.prsquared:.3f}** (düşük: churn çoğunlukla bu değişkenlerle açıklanamıyor)."))
"""),
code("""
fig, ax = plt.subplots(figsize=(7, 4))
r = res.sort_values("OR")
ax.errorbar(r["OR"], r.index, xerr=[r["OR"] - r["OR_low"], r["OR_high"] - r["OR"]], fmt="o", color="#4c78a8", capsize=3)
ax.axvline(1, color="red", ls="--"); ax.set_xscale("log"); ax.set_xlabel("Odds ratio (log ölçek), %95 GA"); ax.set_title("Diğerleri sabitken churn odds oranları")
plt.tight_layout(); plt.show()
"""),
md("""
**Okuma (diğer değişkenler sabitken):**
* **Referans kategoriler:** sözleşme için `Annual`, plan için `Basic` (`drop_first=True` alfabetik ilk kategoriyi düşürür).
* **Aylık sözleşme**, yıllığa göre odds'u yaklaşık **2,9 kat** artırıyor (en güçlü kategorik etki). İki yıllık, yıllıktan ayırt edilebilir değil (aralık 1'i kapsıyor; yalnızca ~264 müşteri).
* **Memnuniyet** ve **abonelik süresi** arttıkça risk **azalıyor** (OR < 1); **şikâyet** arttıkça artıyor.
* `support_calls_6m` sınırda (aralık 1'e çok yakın), `late_payments_6m` etkisi diğerleri sabit tutulunca **yok** — tek başına anlamlı görünmeleri, şikâyet/memnuniyetle ilişkili olmalarından kaynaklanıyor olabilir.
* `plan_Standard` Basic'e göre daha düşük risk gösteriyor ama bu, çoklu karşılaştırma bağlamında zayıf kanıttır.
* Bu sonuçlar **nedensel değildir**; yalnızca gözlemsel ilişkilerdir.

## 5.7 Özellik tarama — yalnızca EĞİTİM verisinde (sızıntıyı önlemek için)
Modelde hangi değişkenleri kullanacağımıza karar verirken test kümesine bakmak, test sonucunu iyimserleştirir. Bu yüzden *aynı taramayı yalnızca eğitim verisinde* tekrarlıyoruz ve genel sonuçla tutarlı mı diye bakıyoruz.
"""),
code("""
from src.modeling import load_clean, split_data
tr, te = split_data(load_clean())
tr = add_features(tr)
screen = []
for col in ["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m", "late_payments_6m", "monthly_usage_hours", "age", "discount_rate"]:
    g0, g1 = tr.loc[tr.churn == 0, col].dropna(), tr.loc[tr.churn == 1, col].dropna()
    U, p = stats.mannwhitneyu(g1, g0); screen.append({"değişken": col, "tür": "sayısal", "etki": abs(2 * U / (len(g1) * len(g0)) - 1), "p": p})
for col in ["contract_type", "plan", "region", "payment_method", "device_type"]:
    t = pd.crosstab(tr[col], tr["churn"]); chi2, p, *_ = stats.chi2_contingency(t); screen.append({"değişken": col, "tür": "kategorik", "etki": cramers_v(t), "p": p})
screen = pd.DataFrame(screen); screen["p_BH"] = multipletests(screen["p"], method="fdr_bh")[1]
display(screen.sort_values("p").round(4))
"""),
md("""
**Sonuç:** Eğitim verisinde, çoklu test düzeltmesinden sonra anlamlı kalan **beş** değişken var: `satisfaction_score`, `tenure_months`, `contract_type`, `complaints_6m`, `support_calls_6m`. Bu, **çekirdek özellik setimizin (core)** dayanağıdır; seçim test verisine bakılarak yapılmadı. `late_payments_6m`, `plan`, `region` gibi "sezgisel olarak önemli" görünen değişkenler kanıt vermediği için çekirdeğe alınmadı (Notebook 06'da CV ile ayrıca sınıyoruz).
""")
]

NB06 = [
md("""
# 06 — Özellik Mühendisliği (Feature Engineering)

**Feature engineering = modelin işine yarayacak yeni sütunlar türetmek.** Ama bir uyarı: *özellik eklemek her zaman modeli iyileştirmez.* Her türetilen özelliği **ölçmek** zorundayız. Bu notebook'ta hem özellikleri üretiyor hem de gerçekten işe yarayıp yaramadıklarını **çapraz doğrulamayla sınıyoruz**.

## 6.1 Üretilen özellikler (`src/features.py → add_features`)
| Özellik | Tanım | Neden? |
|---|---|---|
| `tenure_group` | Abonelik süresi dilimleri (0-6, 7-12, 13-24, 25-48, 49-72) | Churn süreyle doğrusal olmayabilir; yeni müşteri riskini yakalar |
| `has_late_payment` | Son 6 ayda gecikme var mı (0/1) | "Hiç yok" ile "var" arasındaki fark, 1→2 farkından daha önemli olabilir |
| `has_support_contact` | Destek araması var mı | aynı |
| `has_complaint` | Şikâyet var mı | aynı |
| `total_issues_6m` | gecikme + destek + şikâyet toplamı | Genel "sürtünme" ölçüsü |

**Eksik değer kuralı:** Eksik, "0" demek değildir. Girdisi eksik olan özellik de eksik kalır (NaN); Pipeline'daki imputer eğitim verisinden öğrendiği değerle doldurur.
"""),
code("""
from src.features import add_features, FEATURE_SETS
raw = pd.read_csv(C.CLEAN_PATH)
fe = add_features(raw)
display(fe[["tenure_months", "tenure_group", "late_payments_6m", "has_late_payment", "support_calls_6m", "has_support_contact",
            "complaints_6m", "has_complaint", "total_issues_6m"]].head(8))
"""),
md("## 6.2 Yeni özellikler churn ile ilişkili mi?"),
code("""
display(fe.groupby("tenure_group", observed=True)["churn"].agg(["size", "mean"]).round(3).T)
display(fe.groupby("total_issues_6m")["churn"].agg(["size", "mean"]).round(3).T)
"""),
md("""
## 6.3 En önemli bölüm: Bu özellikler modeli gerçekten iyileştiriyor mu?
Üç özellik seti tanımlıyoruz ve **aynı model (Lojistik Regresyon)** ile, **yalnızca eğitim verisinde**, tekrarlı çapraz doğrulamayla (5 katlı × 3 tekrar) karşılaştırıyoruz:

| Set | İçerik |
|---|---|
| `base` | Tüm ham değişkenler (ücret hariç): 8 sayısal + 5 kategorik |
| `engineered` | `base` + üretilen tüm özellikler |
| `core` | Eğitim verisindeki istatistiksel taramada (Notebook 05 §5.7) anlamlı kalan 5 değişken: `tenure_months`, `satisfaction_score`, `complaints_6m`, `support_calls_6m`, `contract_type` |

Metrik: **PR-AUC** (dengesiz veride ana ölçüt) ± standart sapma (fold'lar arası).
"""),
code("""
from src.modeling import split_data, xy, tune_and_compare
tr, te = split_data(raw); Xtr, ytr = xy(tr)
parts = []
for name in FEATURE_SETS:
    r, _ = tune_and_compare(Xtr, ytr, name, names=["Logistic Regression"]); parts.append(r)
fs = pd.concat(parts, ignore_index=True)
display(fs[["feature_set", "pr_auc", "pr_auc_std", "roc_auc", "roc_auc_std", "brier"]])
"""),
code("""
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(fs["feature_set"], fs["pr_auc"], yerr=fs["pr_auc_std"], capsize=6, color=["#9ecae9", "#9ecae9", "#4c78a8"])
ax.axhline(ytr.mean(), color="red", ls="--", label=f"Rastgele (taban oran {ytr.mean():.3f})"); ax.legend()
ax.set_ylabel("PR-AUC (CV)"); ax.set_title("Özellik seti karşılaştırması"); plt.tight_layout(); plt.show()
"""),
md("""
### Yorum — dürüst sonuç
1. **Üç set de aynı bantta** (farklar fold standart sapmasının çok altında). *Üretilen özellikler, Lojistik Regresyon'a ölçülebilir bir katkı sağlamadı.* Bunun makul nedeni: lojistik regresyon zaten `satisfaction`, `complaints` vb. sayaçları doğrusal kullanabiliyor; bayraklar ve toplamlar aynı bilgiyi tekrar ediyor.
2. **Sade `core` set en az diğerleri kadar iyi** (hatta sayısal olarak biraz daha iyi, ama bu fark da gürültü içinde).
3. **Karar kuralımız (parsimony / 1-SE):** Başarı fark etmiyorsa **en sade seti** seç. Daha az girdi = daha az hata riski, daha kolay API, daha kolay açıklama, daha az eksik veri sorunu. → `SELECTED_FEATURE_SET = "core"`.

> Bu, portföyünde gösterebileceğin değerli bir olgunluk işaretidir: *"Özellik ürettim, test ettim, işe yaramayanı modele koymadım."* Mülakatta "feature engineering ile skor ne kadar arttı?" diye sorulursa dürüst cevap: **bu veri setinde artmadı; sebebi şu.**

`add_features` kodda duruyor çünkü (a) farklı modeller/veri setlerinde değerli olabilir, (b) sonucu yeniden test edebilmen için.
""")
]
