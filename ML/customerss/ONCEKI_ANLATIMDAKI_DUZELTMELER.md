# Başka bir yapay zekâdan aldığın anlatımdaki hatalar ve eksikler

Yüklediğin metni, **gerçek veri üzerinde kodları çalıştırarak** denetledim. Aşağıdakiler *ölçülmüş* bulgulardır (Python 3.12, pandas 3.0.2, scikit-learn 1.8.0).

## A. Kesin hatalar
| # | Önceki anlatım | Gerçek | Etki |
|---|---|---|---|
| 1 | "1.820 − 12 tekrar − 8 çakışan ID = **1.800**" | 8 çakışan ID'nin **hepsi 2 satır** içeriyor; hepsini silmek 1.820−12−16 = **1.792** satır bırakır | Metindeki rakam kodla tutmaz; ayrıca 8 gerçek müşteri kaybedilir |
| 2 | `astype("string")` + `SimpleImputer(most_frequent)` | Bu ortamda `TypeError: boolean value of NA is ambiguous` ile **çöküyor** (`string` dtype'ı `pd.NA` kullanır) | Pipeline eğitilemez. Çözüm: kategorikleri `object` + `np.nan` tut |
| 3 | `to_numeric(errors="coerce")` ile ücret temizliği, kalanı medyanla doldurma | 30 ücret `NaN` olur ve **medyan (359,1 TL)** ile doldurulur; oysa ücret **plan × (1 − indirim)** ile **kesin** hesaplanabilir | Yanlış doldurma + kaçırılan bilgi |
| 4 | Plan ve ücret ayrı ayrı kontrol ediliyor | **5 satırda plan etiketi ücretle çelişiyor** (plan `basic`, ücret Premium/Standard fiyatı) | Sessiz etiket hatası modele girer |
| 5 | Kodlar (Bölüm 15, 16, 22, 23) yarım/kesik | `cross_validate` sonuç tablosu, eşik döngüsü, Streamlit ve API gövdeleri tamamlanmamış | Kopyalayınca çalışmaz. Tamamlanmışları bu repoda |

## B. Metodolojik zayıflıklar
| # | Konu | Sorun | Bu projedeki çözüm |
|---|---|---|---|
| 6 | `class_weight="balanced"` + eşik 0,40 | Ağırlık olasılıkları şişirir (ort. %41 vs gerçek %8); 0,40 anlamsızlaşır | Ağırlıksız model + **iş değerine göre** eşik; Brier ile kalibrasyon ölçümü |
| 7 | Eşik "örnek" olarak 0,40 | Gerekçesiz sayı | Train OOF + net değer eğrisi + maliyet **duyarlılık analizi** |
| 8 | Test setinde tek nokta metrik | 30 ayrılanlık testte aşırı oynak | **Bootstrap %95 aralıkları** |
| 9 | Permütasyon önemi test'te | Test'te çok gürültülü (negatif önemler çıkabilir) | Eğitim verisinde **CV'li** permütasyon |
| 10 | Tek 5-fold CV | 118 pozitifte kararsız | **5×3 tekrarlı** CV |
| 11 | "En yüksek skor = seçilen model" | Gürültüyü ödüllendirir | **1-SE kuralı** |
| 12 | `monthly_fee_tl` modelde | Plan×indirimin tam türevi → çoklu doğrusallık | Modelden çıkarıldı, kanıtlandı (R²=1) |
| 13 | Çoklu test düzeltmesi yok | 6–8 testte yanlış pozitif riski | **Benjamini-Hochberg** |
| 14 | Feature engineering "yaptık" denmiş | İşe yarayıp yaramadığı ölçülmemiş | CV ile ablation: katkı **yok** → sade `core` set |
| 15 | Eşik/özellik seçimi için tek veri | Seçim sızıntısı riski | Özellik taraması **yalnızca train**'de |
| 16 | Model kaydı: ham `X` + elle özellikler | API'de özellikler yeniden elle üretilmeli → *training-serving skew* | `add_features` **Pipeline'ın içinde** |
| 17 | Streamlit'te 0,50 eşiği | Kalibre bir modelde 1.800 müşteriden yalnızca **4'ü** 0,5'in üstünde skor alır; 0,5 eşiği neredeyse kimseyi işaretlemez | Metadata'daki seçilmiş eşik (0,10) |
| 18 | Test yok | "Sıfır hata" kanıtlanamaz | 45 otomatik test |
| 19 | Power BI "yaptık" | `.pbix` dosyası üretilemez (masaüstü uygulaması) | Veri + ölçüler + adım adım rehber (`04_POWERBI_REHBERI.md`) |

## C. Doğru olan ve korunanlar
Ham/işlenmiş klasör ayrımı, `copy()`, `errors="coerce"` açıklaması, stratified split, leakage uyarısı, Accuracy'nin yanıltıcılığı, Pipeline fikri — hepsi doğruydu ve burada da var.

## D. "0 hata" iddiası hakkında dürüst not
Yaptığım: tüm kod bu ortamda çalıştırıldı, **45 test geçiyor**, 9 notebook hatasız çalıştı, API ve Streamlit otomatik test edildi, notebook sonuçları kayıtlı metriklerle birebir eşleşiyor (assert).
Yapamadığım/yapamayacağım: Power BI `.pbix` üretmek, GitHub'a senin hesabınla push etmek, Streamlit'i gerçek tarayıcıda görsel olarak denemek, farklı işletim sistemi/sürümlerde çalıştırmak. Bunlar için README'deki sürüm uyarılarına bak.
