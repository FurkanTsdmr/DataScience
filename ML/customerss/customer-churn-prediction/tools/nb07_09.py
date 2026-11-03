from tools.nbbuild import md, code

NB07 = [
md("""
# 07 — Makine Öğrenmesi: Bölme, Pipeline, Baseline ve Model Karşılaştırması

**Bu notebook'un kuralı:** TEST KÜMESİNE BURADA HİÇ DOKUNMUYORUZ. Tüm karşılaştırmalar yalnızca eğitim verisinde, çapraz doğrulamayla yapılır. Test kümesi Notebook 08'de **bir kez** açılır.

## 7.1 Train / Test bölme
* `stratify=y`: churn oranı (~%8) hem eğitimde hem testte korunur. Yoksa teste hiç ayrılan müşteri düşebilir.
* `random_state=42`: tekrarlanabilirlik.
* `customer_id` ve `churn` girdi (X) değildir: ID anlamlı bir davranış değil; churn zaten hedef (X'e koyarsak *target leakage*).
"""),
code("""
from src.modeling import load_clean, split_data, xy
df = load_clean()
train, test = split_data(df)
X_tr, y_tr = xy(train); X_te, y_te = xy(test)
display(pd.DataFrame({"satır": [len(train), len(test)], "ayrılan": [y_tr.sum(), y_te.sum()], "churn_oranı": [y_tr.mean(), y_te.mean()]}, index=["train", "test"]))
print("Girdi sütunları:", list(X_tr.columns))
"""),
md("""
## 7.2 Ön işleme ve Pipeline — neden şart?
Model sayısal matris ister; elimizde eksikler ve metin kategoriler var. Gerekenler:

| Adım | Sınıf | Ne yapar | Neden |
|---|---|---|---|
| Özellik üretimi | `FunctionTransformer(add_features)` | Notebook 06'daki fonksiyon | API'de de aynı kod çalışsın |
| Eksik doldurma (sayısal) | `SimpleImputer("median")` | Eksiği **eğitimin** medyanıyla doldurur | Uç değerlere dayanıklı |
| Eksik doldurma (kategorik) | `SimpleImputer("most_frequent")` | En sık kategori | Basit; eksiklik bilgi taşımıyordu (NB05) |
| Ölçekleme | `StandardScaler` | $z=(x-\\mu)/\\sigma$ | Lojistik regresyon ölçeğe duyarlı; ağaçlar değil |
| Kodlama | `OneHotEncoder(handle_unknown="ignore")` | Kategoriyi 0/1 sütunlara açar | Eğitimde görülmeyen kategori hata vermesin |
| Birleştirme | `ColumnTransformer` | Sütun gruplarına farklı işlem | |
| Zincir | `Pipeline` | Hepsini + modeli tek nesne yapar | **Sızıntıyı önler**: CV'de her fold'da imputer/scaler yalnızca o fold'un eğitim kısmından öğrenir |

### Data Leakage (veri sızıntısı) küçük bir gösterim
Yanlış yol: medyanı **tüm veriden** hesaplayıp doldurmak; doğru yol: yalnızca eğitimden öğrenip teste uygulamak. Fark küçük görünür ama test bilgisi eğitime karışmış olur:
"""),
code("""
wrong = df["satisfaction_score"].median()          # tüm veri (test dahil!)
right = X_tr["satisfaction_score"].median()        # sadece eğitim
display(Markdown(f"satisfaction_score medyanı — tüm veri: **{wrong:.3f}**, yalnızca eğitim: **{right:.3f}**. Küçük fark, ama prensip: test'ten hiçbir istatistik öğrenilmemeli."))
"""),
code("""
from src.preprocessing import build_pipeline, build_preprocessor
from src.features import FEATURE_SETS, SELECTED_FEATURE_SET, add_features
from sklearn.linear_model import LogisticRegression
num, cat = FEATURE_SETS[SELECTED_FEATURE_SET]
pre = build_preprocessor(True, num, cat)
Xt = pre.fit_transform(add_features(X_tr))
names = pre.get_feature_names_out()
display(Markdown(f"Dönüşüm sonrası matris: **{Xt.shape}** — sütunlar: `{list(names)}`"))
display(pd.DataFrame(Xt.toarray() if hasattr(Xt, 'toarray') else Xt, columns=names).head(4))
display(Markdown("Eğitimden öğrenilen medyanlar: " + ", ".join(f"`{c}`={v:.2f}" for c, v in zip(num, pre.named_transformers_['numeric'].named_steps['imputer'].statistics_))))
"""),
md("""
## 7.3 Baseline: "en sık sınıfı söyle"
Her model **bir referansla** kıyaslanmalı. Dummy model her müşteri için eğitimdeki oranı (~%8) tahmin eder.
"""),
code("""
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from src.modeling import make_cv
dummy_major = build_pipeline(DummyClassifier(strategy="most_frequent"), False, num, cat)
r = cross_validate(dummy_major, X_tr, y_tr, cv=make_cv(), scoring=["accuracy", "recall", "precision", "roc_auc", "average_precision"], n_jobs=1)
display(pd.Series({k.replace("test_", ""): np.mean(v) for k, v in r.items() if k.startswith("test_")}, name="Dummy (most_frequent), CV").to_frame())
"""),
md("""
**Okuma:** Accuracy ≈ 0,92 ama **Recall = 0**: tek bir ayrılanı bile bulamıyor. PR-AUC taban oranı (≈0,08) bize **"rastgele"nin skorunun ne olduğunu** söylüyor. Gerçek modellerimiz bunu açıkça aşmalı.

## 7.4 Model adayları
| Model | Fikir | Neden aday? | Sadelik sırası |
|---|---|---|---|
| Logistic Regression | Doğrusal sınır, olasılık verir | Hızlı, yorumlanabilir, küçük veride güçlü | 1 |
| Logistic Regression (balanced) | `class_weight="balanced"` ile azınlığa ağırlık | Dengesizliğe klasik çare | 1 |
| Decision Tree | Kurallarla dallanma | Açıklanabilir | 2 |
| Random Forest | Çok sayıda ağacın ortalaması | Doğrusal olmayan etkileşimleri yakalar | 3 |
| Hist Gradient Boosting | Ardışık ağaçlar | Genelde tablo verisinde güçlü | 4 |

Her model için **küçük bir hiperparametre ızgarası** (`GridSearchCV`) ve **5 katlı × 3 tekrarlı** stratified CV. Seçim ölçütü **PR-AUC (average precision)**.

> Hiperparametre seçimini ve model karşılaştırmasını aynı CV'de yapmak hafif iyimserdir; ama tüm modellere eşit uygulandığı için karşılaştırma adildir ve **gerçek, bağımsız not testte** gelir.
"""),
code("""
from src.modeling import tune_and_compare, select_by_one_se
results, tuned = tune_and_compare(X_tr, y_tr, SELECTED_FEATURE_SET)
show = results[["model", "pr_auc", "pr_auc_std", "roc_auc", "roc_auc_std", "brier", "best_params"]].copy()
display(show)
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
r = results.sort_values("pr_auc")
ax[0].barh(r["model"], r["pr_auc"], xerr=r["pr_auc_std"], color="#4c78a8", capsize=3); ax[0].axvline(y_tr.mean(), color="red", ls="--"); ax[0].set_title("PR-AUC (CV ± std)")
ax[1].barh(r["model"], r["roc_auc"], xerr=r["roc_auc_std"], color="#72b7b2", capsize=3); ax[1].axvline(0.5, color="red", ls="--"); ax[1].set_title("ROC-AUC (CV ± std)")
plt.tight_layout(); plt.show()
"""),
md("""
## 7.5 Seçim kuralı: 1-standart-hata
Skorlar birbirine çok yakın ve fold gürültüsü büyük. "En yüksek sayıyı seç" kuralı **gürültüyü ödüllendirir**. Daha sağlam kural:
1. En iyi PR-AUC'yi bul.
2. Ondan **1 standart hata** geride kalmayan tüm modelleri "eşdeğer" say.
3. Bunlar arasında **en sade** olanı seç. Sadelik eşitse **olasılıkları daha iyi kalibre olanı** (düşük Brier) seç.
"""),
code("""
best = results.loc[results.pr_auc.idxmax()]
se = best.pr_auc_std / np.sqrt(5)
elig = results[(results.pr_auc >= best.pr_auc - se) & (results.model != "Dummy (prior)")]
display(Markdown(f"En iyi PR-AUC: **{best.model} ({best.pr_auc:.3f})**, SE ≈ {se:.3f} → alt sınır **{best.pr_auc - se:.3f}**"))
display(elig[["model", "pr_auc", "simplicity", "brier"]])
chosen = select_by_one_se(results)
display(Markdown(f"### Seçilen model: **{chosen}**  (parametreler: `{results.loc[results.model == chosen, 'best_params'].iloc[0]}`)"))
"""),
md("""
## 7.6 `class_weight="balanced"` ne yapar? (Kalibrasyon deneyi)
Dengesizliğe karşı en bilinen çare ağırlıklandırma. Ama bedeli var: **olasılıklar şişer.** Gerçek churn oranı %8 iken model "ortalama risk %40" demeye başlar. Sıralama aynı kalır, ama "%35 risk" sözü artık yanlıştır. Aşağıda iki lojistik modelin out-of-fold olasılıklarını karşılaştırıyoruz.
"""),
code("""
from src.modeling import oof_probabilities
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr
p_plain = oof_probabilities(tuned["Logistic Regression"], X_tr, y_tr, repeats=3)
p_bal = oof_probabilities(tuned["Logistic Regression (balanced)"], X_tr, y_tr, repeats=3)
display(pd.DataFrame({"ort. tahmin": [p_plain.mean(), p_bal.mean()], "gerçek churn oranı": [y_tr.mean()] * 2},
                     index=["ağırlıksız", "balanced"]))
display(Markdown(f"İki skorun sıralama korelasyonu (Spearman): **{spearmanr(p_plain, p_bal)[0]:.3f}** → sıralama neredeyse aynı, **ölçek farklı**."))
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for p, lab in [(p_plain, "ağırlıksız"), (p_bal, "balanced")]:
    f, m = calibration_curve(y_tr, p, n_bins=6, strategy="quantile"); ax.plot(m, f, marker="o", label=lab)
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("Tahmin edilen olasılık"); ax.set_ylabel("Gerçekleşen oran"); ax.legend(); ax.set_title("Kalibrasyon")
plt.tight_layout(); plt.show()
"""),
md("""
**Sonuç:** Ağırlıksız model olasılıkları gerçeğe yakın (kalibre). Dengesizliği **eşik seçimiyle** (Notebook 08) çözmek, ağırlıkla çözmekten daha temiz: hem sıralama aynı hem olasılıklar anlamlı. Karar ağacı örneği aşağıda: sığ bir ağacın kurallarını *okuyabilmek* öğretici (ama ağaç bu veride zayıf).
"""),
code("""
from sklearn.tree import DecisionTreeClassifier, export_text
tree = build_pipeline(DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, class_weight="balanced", random_state=C.RANDOM_STATE), False, num, cat).fit(X_tr, y_tr)
print(export_text(tree.named_steps["model"], feature_names=list(tree.named_steps["preprocessor"].get_feature_names_out()), max_depth=3))
"""),
md("""
## 7.7 Özet — bu notebook'ta neler karar verildi?
1. Özellik seti: **core** (Notebook 06).
2. Model: **Lojistik Regresyon** (yukarıdaki kuralla; karmaşık modeller anlamlı bir kazanç sağlamadı).
3. Dengesizlik: ağırlıkla değil **eşikle** ele alınacak (kalibre olasılıklar için).
4. Test kümesi hâlâ açılmadı.
""")
]

NB08 = [
md("""
# 08 — Model Değerlendirme, Eşik, Önem ve Kayıt

Bu notebook'ta sırayla: **(1)** eşiği eğitim verisinde seç, **(2)** test kümesini **bir kez** aç, **(3)** metrikleri belirsizlikle raporla, **(4)** hataları ve önemleri incele, **(5)** modeli kaydet/yükle/doğrula.

## 8.1 Model, bölme ve out-of-fold (OOF) olasılıklar
Eşik, eğitim verisinin **out-of-fold** tahminleriyle seçilir: her müşterinin skoru, o müşteriyi görmemiş bir modelden gelir → eşik seçimi dürüst olur.
"""),
code("""
import json, joblib
from src import evaluation as E
from src import modeling as M
from src.train import smooth_best_threshold

df = M.load_clean(); train, test = M.split_data(df)
X_tr, y_tr = M.xy(train); X_te, y_te = M.xy(test)
model = joblib.load(C.MODEL_PATH); meta = json.loads(C.MODEL_META_PATH.read_text(encoding="utf-8")); saved = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
print("Kayıtlı model:", meta["model"], meta["hyperparameters"], "| özellik seti:", meta["feature_set"])

oof = M.oof_probabilities(model, X_tr, y_tr)    # model unfitted kopyalanır, fold'larda eğitilir
"""),
md("""
## 8.2 Eşik analizi — neden 0,50 değil?
Sınıflandırıcı olasılık üretir; "ayrılacak" demek için bir **eşik** seçilir. Varsayılan 0,50 burada işe yaramaz: hiçbir müşteri 0,5'e yakın skor almıyor, en yüksek skor ≈ 0,6 ve müşterilerin çoğu 0,15'in altında (taban oran %8!). Eşiği **iş değerine** göre seçiyoruz:

$$\\text{net değer} = \\sum_{\\text{aranan, ayrılacak}} (\\text{başarı oranı} \\times \\text{12 ay} \\times \\text{ücret} \\times \\text{marj}) - \\text{maliyet} \\times \\text{aranan sayısı}$$
"""),
code("""
tt = E.threshold_table(y_tr, oof, train["monthly_fee_tl"])
thr = smooth_best_threshold(tt)
display(Markdown(f"Seçilen eşik: **{thr:.2f}** (kayıtlı model eşiği: {meta['threshold']:.2f})"))
assert abs(thr - meta["threshold"]) < 1e-9
display(tt.loc[tt.threshold.isin([0.05, 0.07, 0.09, 0.11, 0.15, 0.20, 0.30]), ["threshold", "flagged_pct", "precision", "recall", "f1", "f2", "net_value_tl"]])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(tt.threshold, tt.precision, label="Precision"); ax[0].plot(tt.threshold, tt.recall, label="Recall"); ax[0].plot(tt.threshold, tt.f2, label="F2")
ax[0].axvline(thr, color="k", ls="--"); ax[0].legend(); ax[0].set_xlabel("Eşik"); ax[0].set_title("Eşik trade-off (train OOF)")
ax[1].plot(tt.threshold, tt.net_value_tl / 1000); ax[1].axvline(thr, color="k", ls="--"); ax[1].axhline(0, color="gray")
ax[1].set_xlabel("Eşik"); ax[1].set_ylabel("Net değer (bin TL)"); ax[1].set_title("Kampanya net değeri (varsayımlarla)")
plt.tight_layout(); plt.show()
"""),
md("""
**Precision–Recall takası:** Eşik düştükçe daha çok ayrılanı yakalarız (Recall ↑) ama daha çok *gereksiz* arama yaparız (Precision ↓). **F2**, Recall'a Precision'ın 2 katı ağırlık verir (kaçırmak daha pahalıysa). Hangi noktanın doğru olduğu **iş maliyetine** bağlı → net değer eğrisi.

### Duyarlılık analizi: sonuç maliyet varsayımına çok bağlı!
"""),
code("""
sens = pd.read_csv(C.ROOT / "reports" / "cost_sensitivity.csv")
display(sens.rename(columns={"offer_cost_tl": "kampanya_maliyeti_TL", "test_net_value_if_contact_everyone_tl": "herkesi_ararsak_net_TL"}).round(3))
"""),
md("""
**Okuma:**
* **Düşük maliyet (20–50 TL):** Model değer katıyor; 50 TL'de test'te net değer pozitif, **herkesi aramak ise zararlı**. 20 TL'de herkesi aramak neredeyse aynı kadar iyi → model neredeyse gereksiz.
* **Yüksek maliyet (100–150 TL):** Model bu kampanyayı kârlı yapacak kadar keskin değil; çok az müşteri eşiği geçiyor ve net değer sıfıra yakın/negatif.
* **Ders:** "Model iyi mi?" sorusunun cevabı, *"hangi aksiyon için?"* sorusuna bağlıdır. Bu yüzden gerçek maliyetler iş birimiyle konuşulmalı. Varsayılan senaryomuz 50 TL.

## 8.3 Final: test kümesini AÇIYORUZ (tek seferlik)
"""),
code("""
p_te = model.predict_proba(X_te)[:, 1]
rank = E.ranking_metrics(y_te, p_te); at = E.threshold_metrics(y_te, p_te, thr)
at["net_value_tl"] = E.net_value(y_te, p_te >= thr, test["monthly_fee_tl"])
display(pd.Series({**rank, **{k: v for k, v in at.items() if k in ["flagged", "flagged_pct", "precision", "recall", "f1", "f2", "net_value_tl"]}}, name="TEST").to_frame())
# Kayıtlı metrics.json ile birebir aynı mı?
assert abs(rank["pr_auc"] - saved["test_ranking"]["pr_auc"]) < 1e-12 and abs(at["recall"] - saved["test_at_threshold"]["recall"]) < 1e-12
print("✔ Notebook sonuçları kayıtlı metrics.json ile birebir aynı.")
"""),
code("""
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay
pred = (p_te >= thr).astype(int)
print(classification_report(y_te, pred, target_names=["Kalır", "Ayrılır"], zero_division=0))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Kalır", "Ayrılır"], cmap="Blues", ax=ax[0], colorbar=False); ax[0].grid(False); ax[0].set_title(f"Karışıklık matrisi (eşik {thr:.2f})")
RocCurveDisplay.from_predictions(y_te, p_te, ax=ax[1]); ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("ROC")
PrecisionRecallDisplay.from_predictions(y_te, p_te, ax=ax[2]); ax[2].axhline(y_te.mean(), color="k", ls="--"); ax[2].set_title("Precision-Recall")
plt.tight_layout(); plt.show()
"""),
md("""
### Karışıklık matrisinin iş anlamı
* **TN:** Kalacak müşteriyi doğru bildik. **FP:** Kalacak müşteriyi "riskli" dedik (gereksiz arama maliyeti).
* **FN:** Ayrılacak müşteriyi kaçırdık (en pahalı hata olabilir). **TP:** Ayrılacak müşteriyi yakaladık.

## 8.4 Küçük test kümesi: sonuçlar ne kadar güvenilir? (Bootstrap %95 aralık)
Testte yalnızca **~30 ayrılan** var; tek bir müşteri bile metrikleri oynatır. Bu yüzden nokta tahminin yanına belirsizlik koymak zorundayız.
"""),
code("""
ci = E.bootstrap_ci(y_te, p_te, thr); display(ci.round(3))
base = saved["test_baseline_dummy"]
display(Markdown(f"Baseline (dummy) test ROC-AUC = {base['roc_auc']:.2f}, PR-AUC = {base['pr_auc']:.3f} (taban oran). Model PR-AUC **{rank['pr_auc']:.3f}** → taban orana göre ≈ **{rank['pr_auc']/y_te.mean():.1f}×** kazanç."))
"""),
md("""
**Okuma:** Aralıklar **geniş**. ROC-AUC için aralık 0,5'in belirgin üstünde → model gerçekten bir şey öğreniyor. Ama PR-AUC ve precision aralıkları geniş: *"test'te 0,16 çıktı"* ile *"CV'de 0,27 çıktı"* arasındaki fark bu belirsizlik içinde. Tek bir sayıya fazla güvenmeyin.

## 8.5 İş açısından: Kümülatif kazanç (Gain) ve Lift
"Müşterileri risk skoruna göre sıralayıp en riskli %10'unu ararsak ayrılanların yüzde kaçını yakalarız?"
"""),
code("""
gain = E.gain_table(y_te, p_te); display(gain.round(3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([0] + list(gain.top_pct), [0] + list(gain.capture_rate * 100), marker="o", label="Model"); ax[0].plot([0, 100], [0, 100], "k--", label="Rastgele"); ax[0].legend()
ax[0].set_xlabel("Aranan müşteri (%)"); ax[0].set_ylabel("Yakalanan ayrılan (%)"); ax[0].set_title("Kümülatif kazanç (test)")
ax[1].bar(gain.top_pct.astype(str), gain.lift, color="#4c78a8"); ax[1].axhline(1, color="k", ls="--"); ax[1].set_xlabel("En riskli ilk %"); ax[1].set_ylabel("Lift"); ax[1].set_title("Kümülatif lift")
plt.tight_layout(); plt.show()
"""),
md("""
**Okuma:** Lift ≈ 2,3 demek: en riskli %10'luk dilim, rastgele seçime göre **yaklaşık 2,3 kat** fazla ayrılan içeriyor (n küçük: 36 müşteri, 7 ayrılan). Model mucize değil ama **sıralamayı rastgeleden belirgin şekilde iyileştiriyor.**

## 8.6 Hata analizi: kimleri kaçırıyoruz?
"""),
code("""
te = test.assign(score=p_te, pred=pred)
fn = te[(te.churn == 1) & (te.pred == 0)]; tp = te[(te.churn == 1) & (te.pred == 1)]
cols = ["tenure_months", "satisfaction_score", "complaints_6m", "support_calls_6m", "late_payments_6m"]
cmp = pd.DataFrame({"Yakalanan (TP)": tp[cols].mean(), "Kaçırılan (FN)": fn[cols].mean(), "Kalan müşteriler": te[te.churn == 0][cols].mean()})
display(cmp); display(pd.crosstab(te.contract_type, [te.churn, te.pred], rownames=["sözleşme"], colnames=["churn", "tahmin"]))
"""),
md("""
**Okuma:** Kaçırılan 12 ayrılanın ortalama abonelik süresi (~39 ay) yakalananlardan (~19 ay) çok daha uzun, şikâyetleri daha az ve memnuniyetleri daha yüksek; 5'i yıllık/iki yıllık sözleşmeli. Yani model, "klasik riskli profil"e uymayan ayrılanları kaçırıyor. Bu veride gözlenmeyen nedenler (rakip teklifi, taşınma vb.) olabilir. Dikkat: n = 12, bu bir hipotezdir, kanıt değil.

## 8.7 Hangi değişkenler önemli?
İki bakış: **(a) lojistik regresyon katsayıları**, **(b) çapraz doğrulamalı permütasyon önemi** (bir sütunu karıştırınca PR-AUC ne kadar düşüyor?). Küçük testte permütasyon önemi çok gürültülü olduğu için eğitim verisinde CV ile hesaplanır.
"""),
code("""
coef = pd.read_csv(C.ROOT / "reports" / "model_coefficients.csv"); perm = pd.read_csv(C.ROOT / "reports" / "permutation_importance.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
c = coef.sort_values("coef"); ax[0].barh(c.feature, c.coef, color=np.where(c.coef > 0, "#e45756", "#54a24b")); ax[0].axvline(0, color="k"); ax[0].set_title("Katsayılar (+ riski artırır)")
p = perm.sort_values("importance"); ax[1].barh(p.feature, p.importance, xerr=p["std"], color="#4c78a8"); ax[1].set_title("Permütasyon önemi (train CV)")
plt.tight_layout(); plt.show()
display(coef.round(3))
"""),
md("""
**Okuma:** Risk artıranlar: aylık sözleşme, şikâyet, destek araması. Risk azaltanlar: uzun abonelik süresi, yüksek memnuniyet, iki yıllık sözleşme (sayısal değişkenler standartlaştırıldığı için katsayılar "1 standart sapma" etkisidir). Permütasyon önemi: memnuniyet, sözleşme ve süre en önde; şikâyet daha küçük; `support_calls_6m` ≈ 0 (katkısı belirsiz). Std değerleri büyük olduğundan sıralamaya fazla güvenme. (Önem ≠ nedensellik.)

## 8.8 Modeli kaydetme ve yeniden yükleme
`joblib.dump(pipeline)` — **tüm Pipeline** (özellik üretimi + imputer + scaler + encoder + model) tek dosyada kaydedilir. Böylece API/Streamlit ham veriyi doğrudan verir; dönüşümleri yeniden yazmak gerekmez.
"""),
code("""
loaded = joblib.load(C.MODEL_PATH)
new_customer = pd.DataFrame([{"tenure_months": 4, "satisfaction_score": 3.5, "complaints_6m": 1, "support_calls_6m": 4,
                              "late_payments_6m": 2, "contract_type": "Monthly", "plan": "Basic"}])
risk = loaded.predict_proba(new_customer)[:, 1][0]
display(Markdown(f"Yeni müşteri (4 aylık, düşük memnuniyet, aylık sözleşme, 4 destek araması) risk skoru: **{risk:.1%}** → kademe: **{E.risk_tier(risk, thr)}**  (taban oran ≈ %8; eşik {thr:.0%})"))
assert np.allclose(loaded.predict_proba(X_te)[:, 1], p_te)
print("✔ Yüklenen model, kayıtlı modelle aynı tahminleri üretiyor.")
"""),
md("""
## 8.9 Sonuç özeti
| | |
|---|---|
| Model | Lojistik Regresyon, 5 girdi |
| CV PR-AUC / ROC-AUC (train) | ≈ 0,27 / 0,75 |
| Test ROC-AUC / PR-AUC | ≈ 0,71 / 0,16 (geniş güven aralığı) |
| Eşik (50 TL maliyet varsayımı) | 0,10 → müşterilerin ≈ %28-31'i |
| Test'te yakalama / precision | ≈ %60 / ≈ %16 |
| Lift (en riskli %10) | ≈ 2,3× |

**Bu modeli savunmanın dürüst yolu:** "Zayıf-orta güçlü bir sinyal var; model rastgeleden belirgin şekilde iyi, düşük maliyetli kampanyalarda değer üretiyor, pahalı kampanyalarda yetersiz. Gerçek maliyetler ve bir A/B testi olmadan canlıya alınmamalı."
""")
]

NB09 = [
md("""
# 09 — SQL Analizi

Veri analistliği/bilimciliği mülakatlarında SQL neredeyse her zaman sorulur. Temiz veriyi ve model skorlarını **SQLite** veritabanına aktarıp `sql/churn_analysis.sql` içindeki sorguları çalıştırıyoruz.

* `customers`: temiz veri
* `scored_customers`: temiz veri + dürüst risk skoru (eğitim satırları için out-of-fold, test satırları için hold-out)

Sorgular dosyada isimlendirilmiş; kullanılan SQL kavramları: `GROUP BY`, `HAVING`, `CASE WHEN`, `CTE (WITH)`, `RANK()`, `ROW_NUMBER()`, `NTILE()`, kümülatif `SUM() OVER`.
"""),
code("""
from src.sql_utils import build_connection, load_queries, run_all
con = build_connection()
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con).name.tolist())
queries = load_queries(); results = run_all(con)
def show(name, note=""):
    print(f"--- {name} ---"); print(queries[name]); display(results[name]); 
    if note: display(Markdown(note))
"""),
md("## Genel özet ve sözleşme bazında churn + lift (CTE)"),
code("show('01_genel_ozet'); show('02_sozlesme_bazinda_churn_ve_lift', '**Lift**: segmentin churn oranı / genel churn oranı. Aylık sözleşme ≈ 1,4× riskli; iki yıllık ≈ 0,3×.')"),
md("## Plan × sözleşme matrisi (HAVING ile küçük grupları ele)"),
code("show('03_plan_sozlesme_matrisi')"),
md("## Abonelik süresi dilimleri (CASE WHEN)"),
code("show('04_abonelik_suresi_dilimleri')"),
md("## Ödeme gecikmesi — dikkat: ilişki monoton değil"),
code("show('05_odeme_gecikmesi', '3 gecikmede %18, 4 gecikmede %0 görünüyor ama 4 gecikmeli sadece **9 müşteri** var: küçük örneklemde oranlara güvenme.')"),
md("## Bölge sıralaması (RANK pencere fonksiyonu)"),
code("show('06_bolge_siralamasi_pencere_fonksiyonu')"),
md("## Memnuniyet × şikâyet"),
code("show('07_memnuniyet_dilimi_ve_sikayet')"),
md("## Model skorlarıyla iş soruları"),
code("show('08_risk_kademesi_basarimi', 'Kademeler **tek yönlü** ayrışıyor: High kademesi müşterilerin ≈ %28\\'i ama ayrılanların ≈ %60\\'ı.'); show('09_riskteki_gelir')"),
code("show('10_en_riskli_20_musteri')"),
md("## Kümülatif yakalama (NTILE + pencere toplamı)"),
code("show('11_kumulatif_yakalama_pencere')"),
code("con.close()")
]
