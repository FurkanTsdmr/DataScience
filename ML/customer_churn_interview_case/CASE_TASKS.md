# Case görevleri — işi bir Data Scientist gibi çöz

## 1. Business Understanding
- Churn nedir? Şirket için maliyeti ne olabilir?
- Yönetim hangi kararları almak istiyor?
- KPI: churn rate, retention rate, segment bazlı churn. Veri setinin desteklemediği KPI'ları uydurma.

## 2. Data Understanding / EDA
- `shape`, `info`, `describe`, `nunique`, `value_counts`.
- Eksik değer sayısı/oranı, tam tekrarlar ve müşteri ID tekrarları.
- Sayısal aralık ve kategori tutarlılığı kontrolleri.
- Churn sınıf dağılımı.
- En az 8 grafik: hedef dağılımı, histogramlar, boxplot, sözleşmeye göre churn, plana göre churn, bölgeye göre churn, korelasyon ısı haritası, ücret-memnuniyet ilişkisi.
- Her grafik için bulgu + iş yorumu + sınırlılık yaz.

## 3. Data Cleaning
- Eksik değerleri sütun türü ve dağılımına göre ele al; medyan/mod seçimini gerekçelendir.
- Geçersiz yaş, kullanım, ücret ve memnuniyet değerlerini tespit et.
- Kategorileri normalize et: boşluk, büyük/küçük harf, yazım eşleştirme.
- Tam kopya satırları ve aynı ID'ye ait çelişkili kayıtları ayrı incele.
- Metin bozulmuş ücretleri sayısala çevir; dönüşmeyenleri NaN yap ve say.
- Her temizlik adımından önce/sonra satır sayısını raporla.
- Temiz veri çıktısı: `data/processed/customer_churn_clean.csv`.

## 4. Feature Engineering
- tenure grupları, yaş grupları, destek teması var/yok, gecikmiş ödeme var/yok gibi alanlar üret.
- ID ve target leakage kontrolü yap.
- Her yeni özelliğin iş gerekçesini açıkla.

## 5. İstatistik
- Churn oranı için güven aralığı hesapla.
- Kategorik alanlarda ki-kare; sayısal alanlarda uygun grup karşılaştırması dene.
- Korelasyonu nedensellik olarak sunma.

## 6. Machine Learning
- Baseline: DummyClassifier.
- LogisticRegression, DecisionTree, RandomForest; istersen Gradient Boosting.
- Stratified train/test split.
- SimpleImputer + OneHotEncoder + StandardScaler + ColumnTransformer + Pipeline.
- Cross-validation ile karşılaştır.
- Confusion matrix, Precision, Recall, F1, ROC-AUC, PR-AUC.
- 0.30/0.40/0.50/0.60/0.70 eşiklerini validation üzerinde karşılaştır; iş maliyetini düşün.
- Feature importance / permutation importance yorumla.

## 7. İş önerisi
- En riskli segmentleri ve olası nedenleri bul.
- Her segmente ölçülebilir aksiyon öner.
- Kampanya etkisini kontrol grubu/A-B testi ile ölçme planı yaz.
- Yönetici özeti: 3 bulgu, 3 aksiyon, model sınırları.

## Mülakat soruları
1. Eksik veriyi neden medyan/mod ile doldurdun?
2. Aykırı değerleri neden sildin veya korudun?
3. Churn dengesizse hangi metriklere bakarsın?
4. Accuracy yüksekken Recall düşükse ne anlama gelir?
5. Logistic Regression neden baseline olabilir?
6. OneHotEncoder neyi çözer?
7. StandardScaler hangi modellerde önemlidir?
8. Data leakage nedir, nasıl önledin?
9. Threshold neden her zaman 0.50 değildir?
10. Model tahminlerine göre kampanya gönderince başarıyı nasıl ölçersin?
11. Korelasyon nedensellik midir?
12. Gerçek müşteri verisinde KVKK ve veri minimizasyonu açısından ne yaparsın?
