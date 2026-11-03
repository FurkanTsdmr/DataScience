# Customer Churn — End-to-End Data Science Interview Case

## İş senaryosu
Bir abonelik/telekom şirketi müşteri kaybını azaltmak istiyor. Yönetim; churn oranını, riskli segmentleri, ayrılma ihtimali yüksek müşterileri ve uygulanabilir elde tutma aksiyonlarını görmek istiyor.

## Vaka kaynağı ve veri dürüstlüğü
Vaka kurgusu, IBM'in yayımladığı kurgusal telekom şirketi churn örneği ve kamuya açık churn mülakat/case çalışmalarındaki iş sorularından esinlenmiştir. Bu depodaki CSV ise **özgün olarak sentetik üretilmiştir**; gerçek IBM/Kaggle verisi değildir ve gerçek müşteri kayıtları içermez. IBM örneğinde müşteri profili, hizmetler, sözleşme ve ücret alanları ile son ayda ayrılma etiketi bulunur. Kaynaklar: IBM Cognos Telco Customer Churn örneği ve Kaggle Telco Customer Churn açıklaması (aşağıdaki bağlantılar).

- IBM örnek açıklaması: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
- Kaggle vaka/dataset açıklaması: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- SQL churn case örneği: https://datadank.com/interview/sql/churn-rate-sql-case-study/

## Dosyalar
- `data/raw/customer_churn_raw.csv`: Bilerek sorunlu ham veri; buradan başla.
- `data/reference/customer_churn_clean_reference.csv`: Temiz üretim referansı. Çözümünü bitirmeden açma; gerçek dünyadaki tek doğru temizlik kararı değildir.
- `data/data_dictionary.csv`: Alan açıklamaları.
- `CASE_TASKS.md`: Uçtan uca görevler ve mülakat soruları.
- `DATA_QUALITY_GUIDE.md`: Ham veride bulunan sorun kategorileri.
- `requirements.txt`: Paketler.

## Kurulum
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Altın kurallar
1. Ham CSV'yi değiştirme; temiz veriyi `data/processed/` altına kaydet.
2. `customer_id` kimliktir; modele sokma.
3. `churn` hedef değişkendir: `X = df.drop("churn", axis=1)` ve `y = df["churn"]`.
4. Imputer, encoder ve scaler'ı test verisine fit etme. Pipeline + ColumnTransformer kullan.
5. Accuracy tek başına yeterli değildir. Precision, Recall, F1, ROC-AUC ve PR-AUC raporla.
6. Test setini eşik/model seçmek için tekrar tekrar kullanma.
7. Bu veri sentetiktir; bulguları gerçek telekom müşterilerine genelleme.
