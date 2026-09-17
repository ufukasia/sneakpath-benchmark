# Çerçeve (açı) duyarlılığı deneyi — makale koşumu

Tek koşum, makalenin üç iddiasının tüm ölçümleri:

1. **Çerçeveye bağımlı kurucular** (strip, Hilbert, Morton, 2-bantlı serpantin GE)
   açıya büyük ölçüde duyarlıdır.
2. **Dönme-değişmez kurucular** (Greedy-Edge k=8/15, kesin NN, ızgara-NN, FI) için
   açı yalnızca bir beraberlik-kırma zarıdır: rotasyon taraması ile tie-jitter
   dağılımları ayırt edilemez, beraberlikler kaldırılınca etki sıfırlanır.
3. **Hizasız VLSI kartı** bir açı dedektörüyle 0 hizasına geri getirilebilir;
   kafes birebir geri gelir, GE turu bit-aynı çıkar. Bu strip'i kurtarır, GE'ye
   hiçbir şey yapmaz.
4. (Ek) Bantlama (b ekseni) hizalı çerçevede bile GE'ye maliyet ekler.

## Koşum

```
python paper_experiments/frame_experiment.py --workers 8      # tam koşum (resume'lu)
python paper_experiments/frame_experiment.py --list           # örnek listesi
python paper_experiments/frame_experiment.py --quick --only xqf131,kroA100   # duman testi
python paper_experiments/make_report.py                       # tablolar + figürler
```

Çıktı: `out/per_instance/<ad>.json` (örnek başına ham veri),
`out/report/summary.md`, `out/report/tables/T*.{md,tex}`,
`out/report/figures/F*.png`, `out/report/per_instance.csv`.

## Deney tasarımı

| Bileşen | Ne yapılır |
|---|---|
| Açı taraması | [-90, 90) adım 5/10/15/30° (n'e göre); tüm kurucular aynı açılarda; maliyet daima orijinal koordinatlarda TSPLIB yuvarlamasıyla |
| GE kontrolleri | rot (θ taraması), jitter (θ=0, tie_eps=1e-9 tohumlu, aynı örnek sayısı), detied (koordinat + 1e-4×medyan-NN gürültü, sonra θ taraması) |
| Dedektörler | comb (grid_theta.grid_angle), nndir (NN kenar yönü dairesel ortalaması), pca (ana eksen); + kafes-kalıntısı ince ayarı (snap) |
| Yeniden hizalama | φ ∈ {7.5, 22.5, 37.5, U(-45,45)}; hizasız ve geri hizalanmış kartta GE/strip/Hilbert/Morton; kafes birebir geri geldi mi, GE turu bit-aynı mı |
| Bantlama | θ=0'da b ∈ {1,2,3,4,6,8,12,16}, bant içi GE k=8 |

Örnekler: `results/*.json` içindeki 102 VLSI + EUC TSPLIB kümeleri ve
`data_tsplib/` içinden ek TSPLIB kontrolleri (toplam 148; makale 145'ini kullanır — n>120k üç sanat/dünya kümesi kapsam dışı). GEO kümeler atlanır.

Paralel GE deneyi: `parallel_experiment.py` → `out_parallel/`, rapor `make_parallel_report.py` (T7/T7b/T7c, F7/F7b). Kesin k-NN kontrolü: `check_exact_knn.py`.
