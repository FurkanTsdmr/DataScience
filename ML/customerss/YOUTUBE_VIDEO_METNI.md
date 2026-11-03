# 🎬 YouTube Video Metni — "Sıfırdan Uçtan Uca Veri Bilimi Projesi: Customer Churn"

**Tahmini süre:** 55–70 dakika (istersen 3 parçaya böl: Bölüm 1–5 / 6–10 / 11–16)
**Ton:** Sakin, öğretmen. Kamera/ekran: sağda kod, solda kısa notlar.
**Kural:** Metindeki rakamlar repodaki gerçek çıktılardır. Videoda notebook'u çalıştırıp çıktıyı **ekranda göster**; sayı farklı çıkarsa dur ve repoyu kontrol et.
`[EKRAN]` = ekranda ne göstereceğin. `[VURGU]` = ses tonunu değiştir.

---
## BÖLÜM 0 — Açılış (0:00 – 3:00)
Merhaba arkadaşlar. Bugün bir veri bilimi projesini **baştan sona**, hiçbir adımı atlamadan yapacağız. Veri okumadan başlayıp, API ve web uygulamasına kadar gideceğiz.

Ama baştan söyleyeyim: bu videoda "şu kadar doğruluk elde ettik, mükemmel!" demeyeceğim. Çünkü gerçek hayatta mükemmel model yoktur. **Doğru süreç** vardır. Bu projede size modelin kendisinden çok, *nasıl düşündüğümüzü* göstermek istiyorum: veriye nasıl güvenmeyeceğimizi, kendi hatalarımızı nasıl yakalayacağımızı, sonucu nasıl dürüstçe anlatacağımızı.

`[EKRAN]` GitHub repo ana sayfası, klasör yapısı.

Bu videoda göreceğimiz her şey şu repoda var, kod, notebook'lar, testler. Videoyu izlerken kendi bilgisayarınızda da çalıştırmanızı öneriyorum.

Başlayalım.

---
## BÖLÜM 1 — İş problemi (3:00 – 9:00) · Notebook 01
Bir şirket düşünün: aylık abonelikle hizmet satıyor. Her ay bazı müşteriler gidiyor. Buna **churn** diyoruz. Yeni müşteri bulmak pahalı; mevcut müşteriyi tutmak genelde daha ucuz. Soru şu: **Kimin gideceğini önceden tahmin edip, ona özel bir teklif yapabilir miyiz?**

Bu bir **sınıflandırma** problemi: churn, sıfır ya da bir. Ama bizim asıl istediğimiz sıfır-bir cevabı değil, bir **risk skoru**. Çünkü şirketin bütçesi sınırlı; en riskli müşterilerden başlayıp aşağı doğru ineceğiz.

`[EKRAN]` Notebook 01, "Sınıf dengesizliği" hücresi.

Şimdi çok önemli bir deney yapacağım. Hiçbir şey öğrenmeyen bir model düşünün: herkese "kalır" diyor. Bu veride müşterilerin yüzde 8'i ayrılıyor. Yani bu aptal model **yüzde 91,7 doğruluk** alıyor. Yüzde doksan bir virgül yedi! Ve tek bir ayrılan müşteriyi bile bulamıyor.

`[VURGU]` İşte bu yüzden **Accuracy'ye güvenmeyeceğiz.** Mülakatta "Neden Accuracy kullanmadın?" diye sorarlarsa cevabınız bu.

Bunun yerine şunlara bakacağız: **PR-AUC**, **Recall**, **Precision** ve en önemlisi, **para**. Çünkü sonunda şirket için önemli olan, kampanyanın kâr getirip getirmediği.

`[EKRAN]` "Başabaş precision" hücresi.

Burada bir hesap yaptık: kampanya 50 liraya mal oluyor, müşteri kurtarma ihtimali yüzde 30, kurtarılan müşterinin 12 aylık marjı yaklaşık bin yedi yüz lira. Başabaş noktası yaklaşık yüzde 10: aradığımız grubun en az yüzde 10'u gerçekten ayrılacaksa kampanya kâr eder. Taban oran yüzde 8, yani model yüzde 8'in üstünde bir **kesim** bulabilmeli.

Bunların hepsi **varsayım**. Ekranda da yazıyor. Gerçekte bu rakamları şirketle konuşursunuz. Biz sonucun bu varsayımlara ne kadar bağlı olduğunu da sonda göstereceğiz.

---
## BÖLÜM 2 — Proje yapısı ve ham veri kuralı (9:00 – 11:00)
`[EKRAN]` Klasör ağacı.

`data/raw` içinde ham veri var. **Buna asla dokunmuyoruz.** Neden? Temizlerken hata yaparsanız geri dönebilmeniz lazım. Temizlenmiş veri `data/processed`'a gidiyor.

`src` içinde fonksiyonlarımız var, `notebooks` içinde anlatımlarımız, `tests` içinde testlerimiz. Neden test? Çünkü "kodum doğru çalışıyor" cümlesini *söylemek* kolay, *kanıtlamak* zor. 45 testimiz var; sonunda çalıştıracağız.

Bir de `config.py` var: iş kuralları, yani "yaş 18 ile 100 arasında olmalı" gibi kurallar tek bir yerde. Aynı kuralı beş yere kopyalarsanız biri mutlaka unutulur.

---
## BÖLÜM 3 — Veriyi anlama (11:00 – 18:00) · Notebook 02
`[EKRAN]` Notebook 02.

Kural: bu notebook'ta **hiçbir şeyi değiştirmiyoruz.** Sadece bakıyoruz.

`read_csv` ile okuyorum. Ama dikkat, ücret sütununu **metin olarak** okuyorum: `dtype object`. Neden? Çünkü bu sütunda bozuk değerler var ve ben onları *görmek* istiyorum.

`shape`: 1.820 satır, 16 sütun. `info`: ücret sütunu `object`, yani metin. Sayısal olması gerekirdi. İlk kırmızı bayrak.

`describe`: yaş minimum **eksi 3**, maksimum **130**. Aylık kullanım **eksi 5** ve **180**. Memnuniyet puanı **0 ile 11**, oysa ölçek 1'den 10'a. Veri sözlüğümüz bunu söylüyor.

`[VURGU]` Burada bir parantez açayım: **eksik değer her zaman NaN değildir.** Ücret sütununda "free", "unknown", soru işareti yazıyor. Pandas bunları eksik saymaz, çünkü geçerli birer metin. Yedi tane bozuk ücret var.

Şimdi tekrarlar. İki ayrı problem var, karıştırmayın:
- Birebir aynı satır: 12 tane.
- Aynı müşteri numarası ama **farklı bilgiler**: 8 müşteri.

İkincisi çok daha sinsi. Aynı müşterinin iki kaydı var, kullanım saati birkaç saat farklı, destek araması bir fazla. Hangisi doğru? Zaman damgamız yok, bilemeyiz. Bunu birazdan konuşacağız.

Şimdi bu bölümün en güzel keşfine geliyorum. `[EKRAN]` 2.8 hücresi.

Ücreti şuna bölüyorum: `ücret / (1 − indirim)`. Ve çıkan sayılar... neredeyse hep **249, 399 ve 599**. Yani:

> ücret = plan liste fiyatı × (1 − indirim)

Bu kural hem eksik ücretleri **tahmin etmeden kesin hesaplamamızı** sağlayacak, hem de başka bir şeyi yakalayacak. Plan etiketi ile ücretin işaret ettiği plan **çelişen 5 satır** var. Mesela plan "basic" yazıyor ama ücret 599 ve indirim sıfır. Bu, Premium'un fiyatı. Hangisi yalan söylüyor? Plan etiketinde 3 olası değer var; rastgele bir ücretin başka bir planın fiyatına kuruşuna kadar denk gelmesi ise neredeyse imkânsız. Demek ki yanlış olan **etiket**.

Bunu tek tek sütunlara bakarak asla bulamazdık. **Sütunlar arası ilişkilere bakın.**

---
## BÖLÜM 4 — Temizleme (18:00 – 27:00) · Notebook 03 + `data_cleaning.py`
`[EKRAN]` Notebook 03.

`df = raw.copy()`. Ham veri korunuyor.

**Ücret.** `pd.to_numeric(errors="coerce")`. Bu parametre sayıya çevrilemeyen değeri hata vermek yerine `NaN` yapar. Pratik. Ama bir tuzağı var. `[EKRAN]` "399TL" hücresi.

Bir ücret "399TL" yazıyor. Birim eki. Herkes diyecek ki "TL'yi at, 399 olsun". Ben de ilk önce öyle yaptım. Sonra kurala baktım: bu müşteri **Premium ve yüzde 20 indirimli**. Doğru ücret 599 çarpı 0,8, yani **479,2**. "399TL" bir yazım hatası değil, tamamen yanlış bir değermiş.

`[VURGU]` Ders: **Doğrulayamadığınız bir düzeltmeye güvenmeyin.** Temizleme kodumuz bu yüzden her ücreti kuralla çapraz kontrol ediyor; kuraldan sapanı kuralla değiştiriyor ve **loga yazıyor.**

**Kategoriler.** `strip`, `lower`, sonra sözlükle standart yazım. "basic" ile "Basic" iki ayrı kategori olmasın. Ve sözlükte olmayan bir değer sessizce `NaN` olmasın diye bilinmeyen kategori de loglanıyor.

**Aralık dışı değerler.** Yaş 130'u **silmiyoruz**, `NaN` yapıyoruz. Neden? Satırı silerseniz müşterinin diğer 14 doğru bilgisini de çöpe atarsınız. Peki `NaN`'ı neden hemen medyanla doldurmuyoruz? Birazdan, "leakage" bölümünde.

**Tekrarlar.** Birebir tekrarlar: ilk kopyayı tut.

Çakışan ID'ler. `[EKRAN]` karşılaştırma tablosu. İki seçenek var:
- Hepsini sil: 1.820 eksi 12 eksi 16 satır, yani **1.792**. Sekiz gerçek müşteriyi kaybettik.
- İlk kaydı tut: **1.800**.

Ben ikincisini seçtim: dosyada ilk görünen kayıt asıl, sonuna eklenen ikincisi yinelenen satır. Ama dürüst olalım: bu bir **varsayım**. Gerçek bir işte CRM'den güncelleme zamanını alırsınız ya da iş birimine sorarsınız. Ben bu çelişen kayıtları ayrı bir dosyaya kaydediyorum ki sonradan kontrol edilebilsin.

`[EKRAN]` Doğrulama tablosu.

Şimdi bu ödev veri seti olduğu için elimde bir "doğru cevap" dosyası var, referans temiz veri. Bunu **model eğitiminde kullanmıyorum**, sadece temizleme kodumu doğrulamak için. Sonuç: plan, ücret, yaş, kullanım, memnuniyet ve hedef değişken **sıfır farkla** eşleşiyor. Geriye 15 hücre fark kalıyor, 27 bin hücrenin yüzde 0,056'sı. Bunlar küçük harfle yazılmış "marmara", "monthly" gibi hücreler; değerleri de yanlışmış ve hiçbir kural bunu yakalayamaz. **Bu, veri kalitesinin kaçınılmaz gerçeği:** her hatayı bulamazsınız.

Eksik değerleri doldurmadan bıraktım: yaş 50, bölge 32, kullanım 67, ödeme yöntemi 27, memnuniyet 82 eksik. Neden doldurmadım? Bölüm 9'da.

---
## BÖLÜM 5 — EDA (27:00 – 34:00) · Notebook 04
Şimdi sorumuz: **kim ayrılıyor?**

`[EKRAN]` Segment grafikleri.

Her çubuğun yanında **güven aralığı** var, siyah çizgiler. Bunu neden koydum? Çünkü bir segmentte yüzde 20 churn görmek, orada 10 müşteri varsa hiçbir şey ifade etmez. Çubuğun içinde `n` değerini de yazdım.

Bulgular:
- **Sözleşme türü** en güçlü: aylık sözleşmelerde churn yüzde 11,9; yıllıkta 4,3; iki yıllıkta 2,7.
- **Abonelik süresi:** 0–6 ay yüzde 15,4; 49–72 ay yüzde 4,2.
- Memnuniyet düştükçe ve şikâyet arttıkça churn artıyor.
- Cihaz tipi, ödeme yöntemi, yaş, kullanım saati: belirgin fark yok.

Korelasyon matrisi: hepsi küçük, en büyüğü yaklaşık 0,13. `[VURGU]` Bu bize şunu söylüyor: tek bir değişkenle churn tahmin edilemez, ve **modelden mucize beklememeliyiz.** Çok yüksek bir skor görürsek şüphelenmeliyiz, muhtemelen bir yerde sızıntı vardır.

Bölge grafiğinde Marmara düşük görünüyor. "Marmara müşterileri sadık" demek cazip. Ama sözleşme dağılımını kontrol ettim; yine de bunun gerçek mi şans mı olduğunu bir sonraki bölümde **test edeceğiz.** EDA hipotez üretir, istatistik hipotezi sınar.

---
## BÖLÜM 6 — İstatistiksel analiz (34:00 – 41:00) · Notebook 05
Dört araç:

**Güven aralığı.** Churn oranı yüzde 8,22. Ama bu bir örneklem. Wilson yöntemiyle gerçek oran yüzde 7,04 ile 9,58 arasında.

**Ki-kare ve Cramér's V.** Kategorik değişken ile churn arasında ilişki var mı? Ki-kare "var mı" sorusunu cevaplar, ama **ne kadar güçlü** olduğunu söylemez. Büyük örneklemde küçük fark bile "anlamlı" çıkar. O yüzden Cramér's V'ye de bakıyoruz. Sözleşme türü: V yaklaşık 0,15, çok anlamlı.

**Mann-Whitney.** Sayısal değişken iki grupta farklı mı? Neden t-testi değil? Sayaç değişkenleri normal dağılmıyor, Mann-Whitney dağılım varsayımı istemez.

**Çoklu test düzeltmesi.** `[VURGU]` Çok önemli ve çok atlanıyor. Altı test yaparsanız, hiçbir gerçek ilişki olmasa bile yüzde 26 ihtimalle en az biri tesadüfen anlamlı çıkar. **Benjamini-Hochberg** düzeltmesini uyguladım. Sonuç: `plan` ham p-değerinde sınırda anlamlıydı, 0,038. Düzeltince 0,075 oldu. Anlamlılığını kaybetti. **Bölge de anlamlı değil.** Marmara'nın düşük görünmesi şansla açıklanabilir.

**Lojistik regresyon ve odds ratio.** Diğer değişkenler sabitken etkiye bakıyoruz. Aylık sözleşme, yıllığa göre churn odds'unu yaklaşık **2,9 kat** artırıyor, güven aralığı 1,8 ile 4,6. Aralık 1'i içermiyor, güçlü kanıt. İki yıllık sözleşme ise yıllıktan ayırt edilemiyor; çünkü sadece 264 müşteri var.

`[VURGU]` Hepsi **gözlemsel**. "Memnuniyet artarsa kalırlar" demiyoruz. Bunu ancak deneyle, yani A/B testiyle söyleyebiliriz.

Son olarak eğitim verisinde aynı taramayı tekrarladım. Çünkü modele hangi değişkenleri koyacağıma **test kümesine bakarak** karar verirsem, test sonucunu iyimserleştirmiş olurum. Eğitim verisinde anlamlı kalan beş değişken: memnuniyet, sözleşme türü, abonelik süresi, şikâyet, destek araması.

---
## BÖLÜM 7 — Feature engineering (41:00 – 45:00) · Notebook 06
Yeni sütunlar ürettik: abonelik süresi grupları, "gecikme var mı", "şikâyet var mı", "toplam sorun sayısı". Eksik değerleri sıfır saymadık: eksik girdiden türeyen özellik de eksik kalıyor.

Ve şimdi en önemli soru: **işe yaradı mı?**

`[EKRAN]` Özellik seti grafiği.

Üç seti aynı modelle, tekrarlı çapraz doğrulamayla karşılaştırdım:
- Ham değişkenler: PR-AUC 0,244.
- Ham + üretilenler: 0,240.
- Sade beş değişkenli set: 0,267.

Üretilen özellikler **işe yaramadı.** Fark standart sapmanın (yaklaşık 0,07) çok altında. Neden? Lojistik regresyon zaten bu sayaçları doğrusal kullanabiliyor, bayraklar aynı bilgiyi tekrarlıyor.

`[VURGU]` Bu bir başarısızlık değil, **bir bulgu**. Mülakatta "feature engineering ile skor ne kadar arttı?" diye sorulursa dürüst cevap: "Bu veri setinde artmadı, sebebi şu, ve ben bunu ölçtüm." Bu, "yaptım, çok iyi oldu" diyenden çok daha güvenilir.

Kural: **Başarı aynıysa en sade olanı seç.** Daha az girdi, daha az hata, daha kolay API, daha kolay açıklama.

Bir de `monthly_fee_tl`'yi modelden çıkardım. Neden? Ücret, plan ve indirimden **tam türüyor**, R-kare tam bir. Yeni bilgi taşımıyor.

---
## BÖLÜM 8 — Bölme, Pipeline ve sızıntı (45:00 – 51:00) · Notebook 07
**Train-test bölme.** Yüzde 80 eğitim, yüzde 20 test. `stratify`: churn oranı iki tarafta da yüzde 8 kalsın. Test'te 30 ayrılan müşteri var. Aklınızda tutun: **30.** Bu sayıyı birazdan tekrar anacağız.

`customer_id` girdi değil: kimlik numarasından davranış öğrenemez. `churn` zaten hedef; X'e koyarsanız **target leakage**.

**Data leakage.** `[EKRAN]` Medyan karşılaştırma hücresi. Memnuniyet medyanı tüm veride 6,20, sadece eğitimde 6,10. Fark küçük, ama prensip büyük: test'in istatistiği eğitime karışmamalı. İşte `Pipeline` bunu çözüyor. Eksik doldurma, ölçekleme, kodlama, hepsi Pipeline'ın içinde. Çapraz doğrulamada her fold'da bu adımlar **sadece o fold'un eğitim kısmından** öğreniyor.

Pipeline'ın ilk adımı bizim özellik üretme fonksiyonumuz. Neden? Çünkü API'de de aynı kodun çalışmasını istiyoruz. İki yerde ayrı ayrı yazarsanız biri farklı çıkar; buna *training-serving skew* denir.

---
## BÖLÜM 9 — Modeller ve seçim (51:00 – 57:00) · Notebook 07
**Baseline.** Dummy model: doğruluk 0,918, Recall **sıfır**. Bu bizim referansımız. PR-AUC 0,082, yani taban oran.

Adaylar: lojistik regresyon, karar ağacı, random forest, gradient boosting. Hepsinde küçük bir hiperparametre araması, hepsi 5 katlı **üç kere tekrarlanan** çapraz doğrulama. Neden tekrar? Eğitimde sadece 118 ayrılan var; tek bölmeye güvenilmez.

`[EKRAN]` Sonuç tablosu.

| Model | PR-AUC |
|---|---|
| Lojistik regresyon | 0,267 |
| Random forest | 0,275 |
| Gradient boosting | 0,245 |
| Karar ağacı | 0,208 |

Random forest en yüksek. Ama fark, standart hatanın içinde. `[VURGU]` "En yüksek sayıyı seç" kuralı **gürültüyü ödüllendirir.** Ben **1-standart-hata kuralını** kullanıyorum: en iyiden bir standart hata içinde kalan modeller eşdeğerdir; bunların **en sadesini** seçerim. O da lojistik regresyon.

Son bir deney: `class_weight="balanced"`. Dengesizliğe klasik çare. Ama bakın: ortalama tahmin yüzde 41! Gerçek oran yüzde 8. Sıralama aynı, korelasyon 0,997, ama **olasılıklar şişmiş.** "Bu müşterinin ayrılma riski yüzde 40" demek artık yalan. Brier skorunda da görülüyor: 0,069'a karşı 0,201. O yüzden ağırlıksız modeli seçtim ve dengesizliği **eşikle** çözdüm.

---
## BÖLÜM 10 — Eşik ve iş değeri (57:00 – 62:00) · Notebook 08
`[EKRAN]` Eşik grafiği.

Sınıflandırıcı olasılık verir. "Ayrılacak" demek için bir eşik lazım. Varsayılan 0,5 burada anlamsız: 1.800 müşteriden sadece 4'ü 0,5'in üstünde.

E�iği şöyle seçtim:
1. **Eğitim verisinin out-of-fold tahminlerinde.** Her müşterinin skoru, onu hiç görmemiş bir modelden geliyor.
2. **Kampanya net değerini** maksimize eden eşik. Eşik 0,10 çıktı: müşterilerin yaklaşık dörtte biri ile üçte biri arası aranıyor.

Precision-Recall takası: eşik düştükçe daha çok ayrılanı yakalıyoruz ama daha çok gereksiz arama yapıyoruz. Hangisinin doğru olduğu **iş maliyetine** bağlı.

Ve şimdi tabloya bakın. `[EKRAN]` Duyarlılık analizi.

| Kampanya maliyeti | Test net değer | Herkesi arasak |
|---|---|---|
| 20 TL | +7.731 | +7.008 |
| **50 TL** | **+2.962** | **−3.792** |
| 100 TL | −141 | −21.792 |
| 150 TL | −900 | −39.792 |

`[VURGU]` Bu tablo projenin en önemli dersi. 50 liralık kampanyada model **değer üretiyor**, herkesi aramak zarar. 20 liralık çok ucuz kampanyada herkesi aramak neredeyse aynı kadar iyi, yani model gereksiz. 100 liranın üstünde ise model kampanyayı kârlı yapacak kadar **keskin değil**. "Model iyi mi?" sorusunun cevabı: **"Hangi aksiyon için?"**

---
## BÖLÜM 11 — Test: tek seferlik (62:00 – 66:00) · Notebook 08
Şimdi test kümesini açıyorum. **Bir kez.** Eğer bakıp "hmm, düşük" deyip modeli değiştirirsem, test kümesi artık test olmaktan çıkar.

Sonuçlar:
- ROC-AUC: **0,715**
- PR-AUC: **0,164**, taban oran 0,083, yani yaklaşık iki kat.
- Eşik 0,10: Recall **0,60**, Precision **0,16**.

Karışıklık matrisinin iş anlamı: yakalanan 18 ayrılan, kaçırılan 12 ayrılan, 92 gereksiz arama.

`[VURGU]` Ve şimdi söz verdiğim sayıya dönüyoruz: **30.** Testte sadece 30 ayrılan var. Bir müşteri değişse sonuç oynar. O yüzden **bootstrap güven aralığı** hesapladım. ROC-AUC için aralık 0,62 ile 0,80. PR-AUC için 0,11 ile 0,27. Çapraz doğrulamada 0,27 görmüştük, testte 0,16. Bu fark, aralığın içinde. "Model testte kötüleşti" demiyorum, "belirsizlik geniş" diyorum.

İş açısından: müşterileri skora göre sıraladığınızda **en riskli yüzde 10'luk dilimde** ayrılma oranı rastgeleye göre yaklaşık **2,3 kat**. Tüm veride "High" kademesi müşterilerin yüzde 28'i ama ayrılanların yaklaşık **yüzde 60'ı**; gerçek churn oranı yüzde 17,3, "Low" kademesinde yüzde 2,7. Kademeler tek yönlü ayrışıyor. Model mucize değil, ama **işe yarıyor.**

Hata analizi: kaçırılan 12 ayrılan ortalama 39 aylık müşteri; yakalananlar 19 aylık. Kaçırdıklarımız "klasik riskli profile" uymayanlar. Veride olmayan nedenler olabilir: rakip teklif, taşınma. Ama n on iki; bu bir hipotez, kanıt değil.

---
## BÖLÜM 12 — Modeli kaydetme, API ve Streamlit (66:00 – 70:00)
`joblib.dump`: **tüm Pipeline'ı** kaydediyorum, özellik üretimi dahil. API'ye ham veri gönderiyorum, model her şeyi kendi yapıyor.

`[EKRAN]` FastAPI `/docs`.

Girdi doğrulaması: tenure 400 gönderirseniz **422** dönüyor. Kullanıcıya "geçersiz" demeden modelin saçma bir sayı üretmesine izin vermiyoruz. Bunu testle de doğruladım.

`[EKRAN]` Streamlit. Kısa süreli, düşük memnuniyetli, aylık sözleşmeli bir müşteri girin: risk yaklaşık yüzde 48, "High". Uzun süreli, memnun, iki yıllık sözleşmeli müşteri: yüzde 2'nin altı, "Low".

Gerçek hayatta bu eşik bir **öneri**; kararı insan verir ve kampanyanın etkisi A/B testiyle ölçülür.

---
## BÖLÜM 13 — SQL ve Power BI (70:00 – 73:00)
SQL: aynı analizi bir de SQL'de yaptım. `GROUP BY`, `HAVING`, `CASE WHEN`, **CTE** ile lift hesabı, `RANK`, `NTILE` ve **kümülatif pencere toplamı**: "en riskli yüzde 10, 20, 30'u ararsak ayrılanların yüzde kaçı yakalanır?" Mülakatta bu soruyu çok duyarsınız.

Power BI: repoda veri dosyası, DAX ölçüleri ve sayfa düzeni rehberi var. Kartlar, sözleşme-churn grafiği, risk kademesi grafiği, riskli müşteri listesi. Önemli bir detay: skor dosyasında eğitim satırları için **out-of-fold**, test satırları için **hold-out** skor var. Yani dashboard'daki her skor, o müşteriyi görmemiş bir modelden geliyor. Dürüst bir dashboard.

---
## BÖLÜM 14 — Sınırlılıklar ve kapanış (73:00 – 76:00)
Kapanışta kendi projemi eleştireyim, çünkü iyi bir veri bilimcinin işi bu:
1. **Veri sentetik.**
2. Sadece 148 ayrılan müşteri.
3. Çakışan ID kuralı bir varsayım.
4. Maliyetler varsayım.
5. **Korelasyon nedensellik değil**; kampanya etkisi bilinmiyor.
6. Tek kesit veri; zamansal doğrulama yok.

Bu projeyi canlıya alır mıydım? **Hayır.** Önce gerçek maliyetleri doğrulatır, küçük bir A/B testi yapardım: model hedefli kampanya mı daha iyi, rastgele mi?

Bu projeden almanızı istediğim üç şey:
1. **Veriye güvenme, denetle.** Sütunlar arası ilişkilere bak.
2. **Test kümesini koru, sızıntıyı önle.** Pipeline kullan.
3. **Belirsizliği ve varsayımları gizleme.** Güven aralığı, duyarlılık analizi.

### Ödev
1. Repoyu klonla, `python -m src.train` çalıştır, 45 testin geçtiğini gör.
2. `config.py`'de kampanya maliyetini değiştir, eşiğin nasıl hareket ettiğini gözle.
3. Kendi bir özelliğini ekle ve CV'de ölçmeden modele koyma.
4. Bu projede bulamadığım bir veri hatası bulabilirsen yorumlara yaz.

Teşekkürler, bir sonraki videoda görüşürüz.

---
## EK — Sık yapılan hatalar (kısa video/shorts için)
* Accuracy'ye güvenmek (%91,7'lik aptal model).
* Eksik değeri tüm veriden doldurmak (leakage).
* `coerce` sonrası NaN'lara bakmamak.
* Doğrulayamadığın düzeltmeye güvenmek (`399TL`).
* "En yüksek skor = en iyi model" (gürültü).
* Küçük test setinde tek sayı raporlamak.
* `balanced` sonrası olasılığa "yüzde 40 risk" demek.
* Eşiği 0,5'te bırakmak.
