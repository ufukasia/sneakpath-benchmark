# Çerçeve duyarlılığı ve paralel GE — sonuçların özeti (8 Eylül 2026)

Veri: 145 örnek (99 VLSI, 46 TSPLIB; n = 100 … 104 814). Üç dev sanat/dünya kümesi
(n = 238k, 498k, 745k) kapsam dışı (kesin NN O(n²)). Maliyet daima orijinal
koordinatlarda TSPLIB yuvarlamasıyla; gap = BKS'ye göre %. Ham tablolar:
`out/report/tables/T*.md|tex`, `out_parallel/report/tables/T7*.md|tex`; figürler
`out/report/figures/F1–F6`, `out_parallel/report/figures/F7, F7b`.

## 1. Çerçeve duyarlılığı: iki sınıf (T2, F1, F2)

| Kurucu | Sınıf | Açı aralığı medyan (%) | En kötü açı kaybı (puan) | En iyi açı kazancı (puan) |
|---|---|---|---|---|
| Greedy-Edge k=8 | değişmez | 4.65 | 3.2 | 2.8 |
| Greedy-Edge k=15 | değişmez | 4.62 | 2.8 | 2.8 |
| NN (kesin) | değişmez | 5.76 | 4.4 | 3.6 |
| NN (ızgara) | değişmez + yaklaşıklık | 9.26 | 4.6 | 11.4 |
| Farthest Insertion (n≤3000) | değişmez | 1.35 | 0.9 | 0.8 |
| Strip | bağımlı | 21.6 | 25.3 | 22.3 |
| Hilbert | bağımlı | 9.7 | 9.0 | 13.0 |
| Morton | bağımlı | 17.8 | 20.7 | 31.4 |
| 2-bantlı serpantin GE | bağımlı | 9.6 | 9.1 | 4.9 |

Okuma: bağımlı sınıfın aralığı değişmez sınıfın 2–5 katı; ama değişmez sınıfın aralığı
sıfır değil. Bunun **tamamı** beraberlik kırma + yaklaşık aday listesidir (Bölüm 2).

## 2. Greedy-Edge'de açı = beraberlik zarı (T3, T3b, T3c, F3, F4)

- Rotasyon aralığı ↔ jitter aralığı: Spearman 0.95; KS testi 145 örneğin %81'inde ayırt
  edemiyor; eşit bütçeli "en iyi açı" (2.82 puan) ↔ "en iyi jitter" (2.71 puan), işaret
  testi 65/65, p=1.0.
- Ortalamaya gerileme: Spearman(θ=0 gap, en iyi açı kazancı) = 0.70. "Rotasyon kazandırdı"
  görülen yerler GE'nin θ=0'da şanssız olduğu yerler.
- Beraberliksiz koordinat (1e-4 × medyan-NN gürültü): aralık medyanı 0.000; %23 örnekte
  kalan artık **kesin k-NN ile tam 0.000** (10 örnek, 12 açı, tek maliyet:
  `report/exact_knn_check.json`). Yani GE tam olarak dönme-değişmez; görünen tüm yayılım
  (i) tam-eşit uzunluklu kenarların float sırası, (ii) `grid_knn`'in eksen-hizalı hücre
  yaklaşıklığıdır.
- Boyut: aralık medyanı n<500'de %7.6, 500–2000'de %7.7, 2000–10000'de %4.6, n≥10000'de
  %1.4 (göreli etki; tek beraberlik kararı küçük turu daha çok oynatır). Strip aralığı n ile
  küçülmez (%20 → %27).
- Beraberliksiz TSPLIB'de (kroA100, ch150, kroB200, rd400, dsj1000) aralık = 0.000: tez
  sınır durumunda da tutuyor.

**Makale cümlesi:** Greedy-Edge için "en iyi açı" aramak, k yeniden başlatmanın en iyisini
almaktan farksızdır; oracle açı yoktur. RGGE = GE + beraberlik zarı; RSGE(b=1) = GE.

## 3. Strip'in en iyi çerçevesi kafes ekseni DEĞİL (T2b, T5)

- Strip'in en iyi tarama açısı bir kafes eksenine (0/±90 ± 5°) yalnız VLSI'da %30,
  TSPLIB'de %33 oranında düşüyor. θ=0'da ortalama gap %68.4, en iyi açıda %44.5.
- Hizasızlaştırma (φ ∈ {7.5, 22.5, 37.5, rastgele}): VLSI'da strip **iyileşiyor** (68.4 →
  55.8, p<1e-4), GE değişmiyor (17.04 → 17.22, p=0.10). |fark| strip 12.6 puan, GE 0.3.
- Sonuç: "hizalama strip'i kurtarır" **yanlış**. Doğru cümle: strip çok duyarlıdır ve en iyi
  çerçevesi veriye özgüdür; geometrik dedektör değil maliyet taraması (`rotation_strip`)
  gerekir. Kafes hizası ise **tam geri kazanım** sorusudur (Bölüm 4).

## 4. Hizasız kart hizalanabilir — birebir (T4, T4b, F5)

- Dedektörler (VLSI, 396 deneme): comb medyan hata 0.000° ama p90 27.9° (büyük kartlarda
  1° kaba adım yetmiyor: açısal çözünürlük 1/yayılım); NN-kenar-yönü medyan 0.23°, %96'sı
  <1°; PCA medyan 1.9°.
- Snap (en iyi kaba aday + hiyerarşik kafes inceltme + tamsayıya yuvarlama): **396/396**
  VLSI ve 72/72 tamsayı-kafesli TSPLIB denemesinde kafes birebir geri geldi, GE turu
  **bit-aynı**, maliyet aynı. Kalıntı 0.129 → 0.0007 birim.
- Kafessiz TSPLIB'de (dsj1000, kroA100 vb.) dedektör güveni düşer, snap yapılmaz; bu
  beklenen ve raporlanan davranıştır.

## 5. Dikişli bantlama kaybeder (T6, F6)

θ=0'da b bant, tek tura dikiş: b=2 +3.8 puan (18/145 kazanç), b=4 +12.3, b=8 +29.4,
b=16 +60.9; kayıp monoton, p<1e-4 her b için. RSGE'nin kaynak/bütçe kurallarının
n<20000'de b=1 vermesi bu gerçeğin kuralın diliyle söylenmesidir.

## 6. Paralel GE: k drone (T7, T7b, T7c, F7, F7b)

Ölçüt: makespan / (L_GE/k) — 1.0 ideal (k araç tek aracın 1/k süresinde biter).

| k | bant (θ★) | bant (22.5°) | k-means | global tur kesme |
|---|---|---|---|---|
| 2 | 1.067 | 1.064 | 1.066 | 1.116 |
| 4 | 1.199 | 1.193 | 1.182 | 1.291 |
| 8 | 1.487 | 1.441 | 1.303 | 1.621 |
| 16 | 2.056 | 1.902 | 1.425 | 2.192 |

- **k ≤ 4:** bant ve k-means eşdeğer (kazanan sayıları 36/46 vs 39/39), bant çok daha
  hızlı kurulur (hız kazancı k'ya yakın: k=8'de 8.6×, k=16'da 15.6×; k-means 2–3×).
- **k ≥ 8:** k-means açık ara (VLSI k=16: 73/99 kazanç, p<1e-4; TSPLIB k=16: 40/43).
  1-B kesim (bant) ince şeritlerde uzun turlar üretir; 2-B kompakt kümeler makespan'ı korur
  ama dengesizdir (max/ort 1.38).
- **Toplam yol:** k-means k=16'da +3.7 %, bant +58 %, kesme +18 %.
- **Global tur kesme** her k'da en kötü ve paralel kurma kazancı yok: paralel GE için
  bölümleme önce yapılmalı.
- **Bant çerçevesi:** bantlar aralık medyanı %15 (VLSI) / %23 (TSPLIB) ile çerçeveye
  bağımlı, ama kafes-hizalı çerçeve **en iyi değil**: VLSI'da hizasız (22.5°) bant hizalı
  banttan k≥12'de anlamlı olarak iyi (k=16: −0.23, p<1e-4). Mekanizma: tam hizalı çerçevede
  kafes sütunları aynı x'i paylaşır, eşit-sayılı kesim bir sütunu iki banda rastgele böler
  (θ=0'da 7/7 sınır x paylaşıyor, 0.3°'de 0/7) — yine bir beraberlik olgusu.

**Makale cümlesi:** paralel GE'de parça-içi kurucu (GE) çerçeveden bağımsızdır, bölümleme
bağımlıdır; az drone için hizalanmış-olmayan eşit-sayılı bant en ucuz ve yeterli, çok drone
için 2-B kümeleme şart. Kafes hizası bölümleme kalitesi için gerekli değildir.

## 7. Makale iskeleti (önerilen)

1. Giriş: Öklid TSP kurucuları ve koordinat çerçevesi; RGGE/RSGE'nin motivasyonu; soru.
2. Tanımlar: çerçeveye bağımlı / dönme-değişmez kurucu; RSGE(θ,1)=RGGE(θ)=GE özdeşliği.
3. Deney düzeneği: 145 örnek, açı taraması, kontroller (jitter, beraberliksiz, kesin k-NN),
   hizasızlaştırma/yeniden hizalama, bantlama, paralel GE; panelden yeniden üretilebilirlik.
4. Sonuç A — sınıf ayrımı (T2, F1, F2).
5. Sonuç B — GE'de açı = beraberlik (T3, T3c, F3, F4, kesin k-NN kontrolü).
6. Sonuç C — strip'in en iyi çerçevesi ve yeniden hizalama (T2b, T4, T4b, T5, F5).
7. Sonuç D — dikişli bantlama (T6, F6) ve paralel GE (T7, T7b, T7c, F7, F7b).
8. Tartışma: dedektör çözünürlüğü (comb 1/yayılım, nndir sağlam), pratik öneri tablosu
   ("hangi kurucuya ne yapılır"), sınırlamalar (saf Python süreleri, 145 örnek, GE k=8).
9. Sonuç.
