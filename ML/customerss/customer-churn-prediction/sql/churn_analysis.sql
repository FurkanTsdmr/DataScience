-- Customer Churn: SQL analizleri (SQLite söz dizimi; PostgreSQL/SQL Server için küçük değişikliklerle çalışır)
-- Tablolar: customers (temiz veri), scored_customers (temiz veri + risk_score, risk_tier, flagged, score_source)
-- Her sorgu "-- name:" satırıyla başlar; src/sql_utils.py bunları isimleriyle çalıştırır.

-- name: 01_genel_ozet
SELECT COUNT(*)                          AS toplam_musteri,
       SUM(churn)                        AS ayrilan,
       ROUND(100.0 * AVG(churn), 2)      AS churn_orani_yuzde,
       ROUND(AVG(monthly_fee_tl), 2)     AS ort_aylik_ucret_tl
FROM customers;

-- name: 02_sozlesme_bazinda_churn_ve_lift
-- CTE ile genel oranı bir kez hesaplayıp her segmentin genele göre kaç kat riskli olduğunu (lift) buluruz.
WITH genel AS (SELECT AVG(churn) AS oran FROM customers)
SELECT c.contract_type,
       COUNT(*)                                   AS musteri,
       SUM(c.churn)                               AS ayrilan,
       ROUND(100.0 * AVG(c.churn), 2)             AS churn_yuzde,
       ROUND(AVG(c.churn) / g.oran, 2)            AS lift
FROM customers c CROSS JOIN genel g
GROUP BY c.contract_type, g.oran
ORDER BY churn_yuzde DESC;

-- name: 03_plan_sozlesme_matrisi
SELECT plan, contract_type,
       COUNT(*)                       AS musteri,
       ROUND(100.0 * AVG(churn), 2)   AS churn_yuzde
FROM customers
GROUP BY plan, contract_type
HAVING COUNT(*) >= 30          -- küçük gruplarda oranlar yanıltıcıdır
ORDER BY churn_yuzde DESC;

-- name: 04_abonelik_suresi_dilimleri
SELECT CASE WHEN tenure_months <= 6  THEN '1) 0-6 ay'
            WHEN tenure_months <= 12 THEN '2) 7-12 ay'
            WHEN tenure_months <= 24 THEN '3) 13-24 ay'
            WHEN tenure_months <= 48 THEN '4) 25-48 ay'
            ELSE '5) 49-72 ay' END     AS sure_dilimi,
       COUNT(*)                        AS musteri,
       SUM(churn)                      AS ayrilan,
       ROUND(100.0 * AVG(churn), 2)    AS churn_yuzde
FROM customers
GROUP BY sure_dilimi
ORDER BY sure_dilimi;

-- name: 05_odeme_gecikmesi
SELECT late_payments_6m,
       COUNT(*)                       AS musteri,
       ROUND(100.0 * AVG(churn), 2)   AS churn_yuzde
FROM customers
GROUP BY late_payments_6m
ORDER BY late_payments_6m;

-- name: 06_bolge_siralamasi_pencere_fonksiyonu
-- RANK() pencere fonksiyonu: bölgeleri churn oranına göre sıralar (eksik bölge ayrı raporlanır).
SELECT RANK() OVER (ORDER BY AVG(churn) DESC)  AS sira,
       COALESCE(region, '(bilinmiyor)')         AS bolge,
       COUNT(*)                                 AS musteri,
       ROUND(100.0 * AVG(churn), 2)             AS churn_yuzde
FROM customers
GROUP BY region
ORDER BY sira;

-- name: 07_memnuniyet_dilimi_ve_sikayet
SELECT CASE WHEN satisfaction_score IS NULL THEN 'bilinmiyor'
            WHEN satisfaction_score < 4      THEN 'dusuk (<4)'
            WHEN satisfaction_score < 7      THEN 'orta (4-7)'
            ELSE 'yuksek (7+)' END           AS memnuniyet,
       CASE WHEN complaints_6m > 0 THEN 'sikayet var' ELSE 'sikayet yok' END AS sikayet,
       COUNT(*)                              AS musteri,
       ROUND(100.0 * AVG(churn), 2)          AS churn_yuzde
FROM customers
GROUP BY memnuniyet, sikayet
ORDER BY memnuniyet, sikayet;

-- name: 08_risk_kademesi_basarimi
-- Model skorları dürüst (out-of-fold / hold-out) olduğu için bu tablo gerçekçi bir başarı özetidir.
SELECT risk_tier,
       COUNT(*)                                          AS musteri,
       SUM(churn)                                        AS ayrilan,
       ROUND(100.0 * AVG(churn), 2)                      AS gercek_churn_yuzde,
       ROUND(100.0 * SUM(churn) / (SELECT SUM(churn) FROM scored_customers), 1) AS tum_ayrilanlarin_yuzdesi
FROM scored_customers
GROUP BY risk_tier
ORDER BY CASE risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;

-- name: 09_riskteki_gelir
-- Aksiyon eşiğinin üstündeki (flagged) müşterilerin aylık gelir payı.
SELECT SUM(CASE WHEN flagged = 1 THEN monthly_fee_tl ELSE 0 END)                       AS riskli_aylik_gelir_tl,
       SUM(monthly_fee_tl)                                                              AS toplam_aylik_gelir_tl,
       ROUND(100.0 * SUM(CASE WHEN flagged = 1 THEN monthly_fee_tl ELSE 0 END) / SUM(monthly_fee_tl), 1) AS riskli_gelir_yuzde
FROM scored_customers;

-- name: 10_en_riskli_20_musteri
SELECT ROW_NUMBER() OVER (ORDER BY risk_score DESC) AS sira,
       customer_id, contract_type, plan, tenure_months, satisfaction_score,
       ROUND(risk_score, 3) AS risk_score, churn AS gercek_churn
FROM scored_customers
ORDER BY risk_score DESC
LIMIT 20;

-- name: 11_kumulatif_yakalama_pencere
-- Müşterileri skora göre 10 eşit dilime böl (NTILE) ve ayrılanların kümülatif yakalanma oranını göster.
WITH dilim AS (
    SELECT NTILE(10) OVER (ORDER BY risk_score DESC) AS risk_dilimi, churn
    FROM scored_customers
), ozet AS (
    SELECT risk_dilimi, COUNT(*) AS musteri, SUM(churn) AS ayrilan
    FROM dilim GROUP BY risk_dilimi
)
SELECT risk_dilimi, musteri, ayrilan,
       ROUND(100.0 * SUM(ayrilan) OVER (ORDER BY risk_dilimi) / SUM(ayrilan) OVER (), 1) AS kumulatif_yakalanan_yuzde
FROM ozet
ORDER BY risk_dilimi;
