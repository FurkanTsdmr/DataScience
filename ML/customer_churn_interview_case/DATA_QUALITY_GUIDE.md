# Veri kalitesi rehberi (cevap anahtarı değil)
Ham CSV'de özellikle şu sorun türleri bulunur:
- Eksik değerler: age, region, monthly_usage_hours, monthly_fee_tl, satisfaction_score, payment_method
- Tam satır kopyaları
- Aynı customer_id için çelişkili kayıtlar
- Yaş/kullanım/ücret/memnuniyet aralık hataları
- Kategori yazım ve harf biçimi tutarsızlıkları
- monthly_fee_tl içinde metinsel bozulmalar
- Churn hedefinde sınıf dengesizliği olasılığı

Her sorun için şu raporu üret: kaç kayıt etkilendi, uygulanan kural, değişen/silinen/doldurulan kayıt sayısı, varsayım ve gerekçe. ID çakışmalarında otomatik silme; önce incele.
