# -*- coding: utf-8 -*-
"""Snake-Grid alternatives: GREEDY/FARTHEST-STRIP HYBRIDS.

An earlier round of "cheap formula" alternatives (aspect-corrected strip
counts, equal-count bands, proxy-scored variant scans) plateaued around a
38-49% gap -- see SNAKE_GRID_SINIF_VE_GPU_HAT_AKADEMIK_NOT.md Sec.9 for that
full, still-honest historical record (all of it was removed from THIS file
at the user's request; the mechanisms, the two real bugs found, and the one
dropped idea remain documented there for transparency).

The insight that broke through: the weak link isn't WHICH strip is picked,
it's HOW POINTS ARE ORDERED WITHIN a strip -- a naive axis-sort ignores the
strip's own 2-D point scatter. Replacing that inner ordering with a REAL
construction heuristic (reusing tsplib_engine's own proven implementations)
while keeping the strip/serpentine macro-structure ("still goes row by
row") works dramatically better. A first hybrid generation tried both
Greedy-Edge and grid-NN per band; per the user's explicit follow-up
request, the NN-based variants (snake_nn_hybrid, snake_best_band,
snake_greedy_seeded, snake_nn_stitch) were REMOVED, and Farthest-Insertion
-- empirically the single best CONSTRUCTION baseline in this project
(9.75% gap on kroA100 vs Greedy-Edge's 13.69%) -- was tried as the
per-band ordering instead. It won decisively: `snake_farthest_band` beats
`snake_greedy_band` at every scale tested.

  snake_greedy_band_tour    -- Strip/aspect-corrected macro-bands, each
                               band's INTERNAL order from
                               `tsplib_engine.greedy_edge_tour` (k-NN
                               candidate greedy-edge, O(m log m) per band ->
                               O(n log n) TOTAL, measured n^1.10 -- the same
                               complexity class as the plain greedy_edge
                               rival, NOT a higher one),
                               rotated so the point nearest the previous
                               band's exit becomes the entry point.

  snake_farthest_band_tour  -- Same macro-banding, but each band's internal
                               order comes from
                               `tsplib_engine.farthest_insertion_tour` --
                               UNACCELERATED O(m^2) per band (no k-NN
                               structure exists for it), so band size is
                               capped MUCH more aggressively than
                               snake_greedy_band's (mult scales down as n
                               grows -- see _farthest_mult) to keep total
                               cost from exploding. Flagship of this file:
                               beats snake_greedy_band's quality at every
                               scale tested (10-set mean 17.4% vs 22.7%;
                               n=47608: 14.5% vs ~21-25%), at a genuinely
                               higher but still bounded cost.

(A third variant, snake_best_insertion_band_tour, ran BOTH constructions per
band and kept whichever was locally shorter. Because it started from the
SAME two candidates as its siblings above, it could only ever match or beat
them -- it "won" every same-family benchmark statistic by construction, not
because it was a genuinely better method. Removed at the user's explicit
request (2026-07-12); see runner.py's REMOVED_METHODS note.)

Only `snake_stratified_subsample_tour` survives from the very first
("formula") generation -- it is not a formula method, it reuses Snake-
Grid's own real search machinery on a density-proportional sample instead
of uniform random.

All take raw (xs, ys) and return a tour as list[int], same contract as
tsplib_engine's hilbert/morton/strip/nn, EXCEPT snake_stratified_subsample_
tour which (like the Snake-Grid machinery it reuses) needs coords+ewt to
build a real TSPInstance.
"""
from __future__ import annotations

import itertools
import math
import random

import tsplib_engine as E


def _raw_length(tour, xs, ys):
    """Cheap O(n) raw-Euclidean proxy length, used only to RANK a handful of
    candidates (no TSPInstance/rounding needed)."""
    s = 0.0
    for i in range(1, len(tour)):
        a, b = tour[i - 1], tour[i]
        s += math.hypot(xs[a] - xs[b], ys[a] - ys[b])
    return s


#: Ic taramanin deneyecegi BANT SAYILARI (2026-07-31, olculdu).
#: Ayrinti ve olcum kaydi `_band_hybrid_tour` icindeki nottadir. Ozet:
#: gercek kosumlarda kazanan bant sayisi yalnizca 1 (%69) ya da 2 (%31)
#: cikti, bant >= 3 hicbir kez kazanmadi; {1,2}'ye budama 26 kumede
#: BIT-AYNI sonuc verip buyuk n'de 1.3-1.6x hizlandirdi.
BAND_COUNT_CANDIDATES = (1, 2)

#: Havuzlarin DUZLESTIRILMIS genisligi (2026-07-31, kullanici karari).
#:
#: SORUN: havuz fonksiyonlari her (theta, k) hucresi icin _band_hybrid_tour'u
#: cagiriyor; o da 2 bant sayisi x 2 eksen = 4 tur kurup SADECE EN IYIYI
#: donduruyordu. Yani 12 adayli bir v2 havuzu arka planda ~48 tur kurup
#: 36'sini ATIYORDU. Esit duvar-saati kiyasinda Greedy-Edge'in kaybettirdigi
#: fark tam olarak buradan geliyordu: GE ayni surede kurdugu 34-52 turun
#: HEPSINI havuzda tutuyor, Snake dortte birini tutuyordu (olculdu, 6 kume:
#: GE 5/6 kazaniyor, +1.06 puan).
#:
#: COZUM: atilan turlar da havuza aday olarak girer (bkz. `collect`). Turlar
#: ZATEN hesaplanmistir, ek INSA maliyeti YOKTUR; tek maliyet onarim
#: butcesinin daha cok adaya bolunmesidir. Bu yuzden havuz `cap` ile
#: sinirlanir: en dusuk gercek maliyetli `cap` aday tutulur.
POOL_FLAT_CAP = 2

#: ================= ADAPTIF-HIBRIT HAVUZ TASARIMI (2026-08-01) =============
#: Havuzlar EN FAZLA 2 ADAY kurar. Gerekce KIYAS ADALETIDIR: rakip
#: `ge_pool2` de 2 adaylidir; 12-24 adayli bir havuzla 2 adayli bir rakibi
#: yenmek yontemin degil BUTCENIN sonucudur ve hakem bunu hakli olarak
#: reddeder. Kisit RAPORLANAN aday sayisina degil KURULAN tur sayisina
#: uygulanir -- 12 kurup 2 raporlamak da ayni itiraza acikti.
#:
#: Yapilandirma = (aci, bant-ici k, bant sayisi, x-ekseni mi).
#: aci None ise DEDEKTOR acisi (theta_proxy) kullanilir; sayi ise mutlak
#: derece, "+d" biciminde string ise dedektor acisina gore kaydirma.
#:
#: SECIM VERIDEN GELDI: 41 ornekte 60 yapilandirmanin TAMAMI olculdu
#: (grid taramasi), sonra "en fazla 2 aday" kisiti altinda butun ciftler
#: tarandi. Rakip ge_pool2 = 17.04%. Olculen:
#:     tek aday (havuz YOK)            17.35%   GE'ye +0.31  -> KAYBEDIYOR
#:     2 aday, v2 kimligi (th* kullan) 16.29%   GE'ye -0.75  20/41
#:     2 aday, v4 kimligi (sabit sonda)16.40%   GE'ye -0.64  20/41
#:     2 aday, kisitsiz en iyi         16.16%   GE'ye -0.88  22/41
#: Yani havuzun 1'den 2'ye cikmasi yontemin rakibi GECMESINI sagliyor;
#: 2'den fazlasina GEREK YOK ve zaten adil degil.
#: --- ADAPTIF SECIM: KAPLAMA olcusu -------------------------------------
#: kaplama = (medyan en-yakin-komsu mesafesi)^2 * n / alan
#: Noktalarin ne kadar DUZENLI dagildigini olcer: duzgun bir izgarada ~1,
#: kumeli/bosluklu dagilimlarda <<1. O(n) ornekleme ile hesaplanir (varsayilan
#: 400 nokta), yani insa maliyetinin yaninda ihmal edilebilir.
#:
#: NEDEN BU OLCU, "n" DEGIL: 44 ornekte LOOCV (leave-one-out) ile bes aday
#: ozellik sinandi. Rakip ge_pool2 = 16.83%:
#:     ozellik      LOOCV gap   GE farki   kazanma   isaret testi p
#:     (sabit 2'li)   16.30%      -0.53      19/44        0.644
#:     n              16.54%      -0.29      22/44        0.324
#:     aspect         16.68%      -0.15      16/44        0.268
#:     nn_cv          16.70%      -0.13      20/44        0.871
#:     KAPLAMA        15.91%      -0.92      28/44        0.017   <-- ANLAMLI
#: n ile esiklemek LOOCV'de COKTU (esik gurultuye uyuyor, folddan folda
#: 258..2119 arasi ziplıyor). Kaplama esigi ise KARARLI ve p<0.05.
#:
#: Esigin ayirdigi sey fiziksel olarak anlamli: dusuk taraf (10/44) kumeli /
#: bosluklu ornekler (p654, pr144, d198, fl1400 -- nn_cv 1.2-2.0), yuksek
#: taraf duzenli izgara benzeri ornekler. Iki grup FARKLI aci-k ciftini
#: tercih ediyor, ve fark olculdu.
#:
#: !! ASIRI UYUM UYARISI !! Esik ve ciftler 44 ornekten SECILDI. Rapor edilen
#: 15.91% LOOCV degeridir (secim egitim katlarinda yapilir), yani secim
#: sapmasindan arindirilmistir -- ama koleksiyon degisirse yeniden olculmeli.
#: ============ HIBRIT SOZLESMESI (2026-08-01, kullanici karari) ==========
#: Her havuzun IKI adayindan BIRI SERPANTIN (bant>=2), digeri BANTSIZ
#: (bant=1, saf greedy-edge). Yani yontem, makalenin anlattigi seyin
#: TA KENDISI: strip ayrisimi ile greedy-edge'in HIBRIDI, ve hangisinin
#: kullanilacagina ornek bazinda GERCEK MALIYET karar veriyor.
#:
#: NEDEN IKISI BIRDEN, "ikisi de bantli" DEGIL (44 kumede olculdu):
#:     ikisi de BANTLI      18.46%   GE'ye +1.64  11/44   <- KAYBEDIYOR
#:     ikisi de BANTSIZ     16.16%   GE'ye -0.67  22/44
#:     HIBRIT (1+1) sabit   16.54%   GE'ye -0.28  18/44
#:     HIBRIT (1+1) adaptif 15.74%   GE'ye -1.09  23/44   <- EN IYI
#: Yani serpantin adayini havuzda tutmak hem kimligi korur hem SONUCU
#: IYILESTIRIR (15.74 < 16.16): bantli aday 44 kumenin 10'unda kazaniyor
#: ve kazandiginda buyuk kazaniyor (pr144 -6.63, xqf131 -5.14 puan).
#: Buna karsilik IKI adayi birden banda harcamak, bantin kaybettigi 34
#: kumede yedeksiz birakir -- olculen bedel +2.72 puan.
#:
#: DURUSTLUK NOTU: isaret testi p=0.337 (23/44 kazanma). Ortalama kazanc
#: buyuk ama kume basina tutarli DEGIL; birkac buyuk kazancin surukledigi
#: bir ortalamadir ve makalede boyle raporlanmalidir.
KAPLAMA_ESIK = 0.842

#: AILENIN SERPANTIN CEKIRDEGI -- greedy_snake_v1'in TA KENDISI.
#: Her iki kolun da ilk adayi BUDUR; boylece merdiven garantisi
#:      maliyet(v1) >= maliyet(v2)
#: YAPISAL kalir (v1'in turu v2 havuzunda AYNEN vardir ve v2 gercek
#: maliyetle secer). Kollarin strip adayini ayristirmak 0.26 puan daha iyi
#: olurdu (15.74 vs 16.00) ama merdiveni KIRARDI -- makalenin v1 ⊂ v2
#: iddiasi bu garantiye dayandigi icin birlesik hali tercih edildi.
#: Gorsellestirme sekmesi de bu yapilandirmayi cizer: 2 bantli serpantin.
#:
#: ACI ALANI = None -> "DEDEKTOR ACISI" (theta_proxy). 2026-08-02'de
#: -90.0'dan None'a cevrildi (kullanici karari). GEREKCE, olculmus bir
#: TUTARSIZLIK:
#:   * snake_v1_tour(theta_deg=None)      SABIT -90 kullaniyordu
#:   * far_snake_v1_tour(theta_deg=None)  theta_proxy kullaniyordu
#: Yani iki aile "tek degisken bant-ici kurucu" AYNASI DEGILDI -- aci
#: makinesi de farkliydi ve hakem itirazina verilen cevabin dayanagi bu
#: simetriydi. Ayrica panel greedy_snake_v1 icin "Izgara acisi (aile
#: varsayilani)" yaziyordu ama o aci HIC CAGRILMIYORDU.
#: 8 kumede olculdu: v1(varsayilan) sekizinde de v1(th=-90) ile BIT-AYNI,
#: yedisinde v1(th=proxy)'den FARKLI (ornek att532: 102913 vs 114272).
#: EK SONUC: hizalama ablasyonu (@zero) eskiden "-90 vs 0" -- ax=False
#: oldugu icin pratikte "bantlar x ekseninde mi y ekseninde mi" -- olcuyordu;
#: artik gercekten "veriden kestirilen aci <-> hic kestirmemek" olcuyor.
V1_CONFIG = (None, 4, 2, False)
V1_CONFIG_YERTUTUCU = V1_CONFIG

#: (aci, bant-ici k, bant sayisi, x-ekseni mi) -- ilk aday SERPANTIN
V2_CONFIGS_KUMELI = (V1_CONFIG_YERTUTUCU,     # strip: 2 bant (= v1)
                     (-90.0, 4, 1, False))    # bantsiz: saf greedy-edge
V2_CONFIGS_DUZENLI = (V1_CONFIG_YERTUTUCU,    # strip: 2 bant (= v1)
                      (-90.0, 16, 1, False))  # bantsiz: saf greedy-edge
V2_CONFIGS = V2_CONFIGS_DUZENLI      # geriye donuk varsayilan
#: ===========================================================================
#: v4 = BANT SAYISI TARAMASI (2026-08-01, kullanici karari) -- TANI SATIRI
#: ===========================================================================
#: ONCEKI HALI (hibrit sozlesmesi, 2 aday):
#:     ((-90.0, 8, 2, False), (-90.0, 4, 1, False))  -> 16.54%
#: Kaldirildi cunku amaci degisti: artik REKABET degil TESHIS satiri.
#:
#: AMAC. "Hangi ornek boyutunda kac bant mantikli?" sorusunu ORNEK ORNEK
#: olcmek. Iki sabit aci (-90 ve 0) x bant sayisi 1..50 = 100 aday. Kazanan
#: adayin etiketi (`th=.../k=8/b=<bant>y/tarama`) o ornek icin en iyi bant
#: sayisini DOGRUDAN verir; sonuc dosyalarindan n'e karsi bant egrisi
#: cikarilabilir ve bant sayisi boyuta gore ONCEDEN secilebilir.
#:
#: NEDEN b=1 DE VAR (kullanici "2'den 50'ye" dedi): b=1 KONTROLdur. Bugune
#: kadarki butun olcumlerde en iyi bant sayisi 1 ya da 2 cikti; kontrolu
#: disarida birakirsak tarama "en iyi bant 2" der ama aslinda 1 daha iyiyse
#: bunu goremeyiz. 1..50 ayrica tam 100 aday verir (2..50 = 98).
#:
#: !! ADALET UYARISI !! Bu satir 100 tur kurar; rakip (ge_pool2) 2 tur kurar.
#: BU SATIR REKABET TABLOSUNDA RAPORLANAMAZ -- oturum boyunca kurdugumuz
#: "esit insa sayisi" sozlesmesini bilerek ihlal eder. Etikete eklenen
#: `/tarama` eki ve yontem adindaki "TANI" damgasi bunu her ciktida gorunur
#: kilar; ayrica eski (2 adayli) v4 satirlarini `runner._STALE_SIGNATURES`
#: eskimis sayar, boylece iki uretim ayni etiket altinda KARISMAZ.
V4_TARAMA_ACILARI = (-90.0, 0.0)
V4_TARAMA_BANTLARI = range(1, 51)     # 1..50 -> kontrol dahil
V4_TARAMA_K = 8
V4_CONFIGS = tuple((_a, V4_TARAMA_K, _b, False)
                   for _a in V4_TARAMA_ACILARI
                   for _b in V4_TARAMA_BANTLARI)   # 2 x 50 = 100 aday
V4_TARAMA_ETIKET_EKI = "/tarama"

#: ===========================================================================
#: CESITLENDIRME EKSENI ADAYLARI (2026-08-02)
#: ===========================================================================
#: SORU: iki adayli bir havuzda adaylari BIRBIRINDEN ne ayirmali?
#: Dort cevap var ve dordu de AYNI insa sayisini (2 tur) kurar, yani
#: kiyas tek degiskenlidir:
#:
#:     genislik   k = 8, 15                       -> mevcut `ge_pool2`
#:     jitter     k = 8, tie_seed = 1, 2          -> `ge_pool2_jitter`
#:     yon        theta = 0, -90 (k sabit 8)      -> `ge_pool2_yon`
#:     yon x k    asagidaki ciftler               -> `knn_pool2(_uyarlanir)`
#:
#: Bu sabitler YALNIZCA yapilandirma tanimlaridir; hangisinin kazandigina
#: KOSUM karar verir. Hicbirine "onerilen" damgasi vurulmamistir -- panelde
#: dordu de ayni oebekte, ayni rolde durur.
#:
#: BANT YOKTUR (hepsinde b=1). Bant ekseni ayri ve olculmus bir negatiftir
#: (measurements_jpdc/bant_ici_deneyi); burada degiskeni acik tutmak
#: kiyasi tek degiskenli olmaktan cikarirdi.
#:
#: Aci alanindaki "+10" `_cfg_pool` sozdizimidir: dedektor acisina gore
#: +10 derece kaydirma (bkz. `_cfg_pool` docstring'i).
#: SERPANTIN iki-acili havuz (2026-08-02, kullanici talebi). Yukaridaki
#: YON_CONFIGS'ten TEK farki BANT SAYISIDIR (b=2, yani gercek serpantin);
#: ikisi yan yana okununca "yon ekseni bantla mi bantsiz mi daha iyi
#: calisiyor?" sorusu tek degiskenli olur.
#: Bu, v4'un TANI TARAMASINA cevrilmeden ONCEKI 2 adayli halidir
#: (kullanici karari 2026-08-01: "(-90, k=8, b=2) kalsin, digerini
#: (0, k=8, b=2) yap") -- tarama o soruyu yanitladi, rekabet satiri olarak
#: geri geliyor. TEK KOSUMDUR: 2 tur kurar, tarama 100 tur kuruyordu.
SERPANTIN_ACI2_CONFIGS = ((-90.0, 8, 2, False), (0.0, 8, 2, False))

YON_CONFIGS = ((0.0, 8, 1, False), (-90.0, 8, 1, False))
KNN_KUMELI_CONFIGS = ((-90.0, 4, 1, False), ("+10", 8, 1, False))
KNN_DUZENLI_CONFIGS = ((None, 4, 1, False), (-90.0, 16, 1, False))

#: Jitter havuzunun parametreleri. `E.greedy_edge_tour(tie_seed=, tie_eps=)`
#: es-uzunluklu aday kenarlari tohuma bagli olarak kucuk bir epsilon ile
#: ayristirir. Bu havuz TOHUMA BAGLIDIR -- digerlerinin aksine bit-ayni
#: DEGILDIR; tam da bu yuzden tabloda tutulur (determinizm farki olculebilir
#: kalsin).
JITTER_K = 8
JITTER_TOHUMLARI = (1, 2)
JITTER_EPS = 0.01


def knn_configs_for(xs, ys):
    """`knn_pool2_uyarlanir` icin kaplama olcusune gore cift secimi.
    `v2_configs_for` ile AYNI esigi (KAPLAMA_ESIK) kullanir; boylece
    "uyarlama isse yariyor mu?" sorusu iki ailede de ayni bicimde okunur."""
    return (KNN_KUMELI_CONFIGS if kaplama_olcusu(xs, ys) < KAPLAMA_ESIK
            else KNN_DUZENLI_CONFIGS)


def _band_indices(px, k):
    """Equal-WIDTH bucketing along `px` into `k` bands -- same convention as
    `tsplib_engine.snake_order`, just returning the raw index buckets
    instead of an already-serpentined order (callers here pick their OWN
    internal ordering per band). NOTE: forcing a large, n-proportional `k`
    with equal-WIDTH buckets on non-uniform (VLSI) point clouds can starve
    most bands to near-empty while a few overload -- this is why callers
    below derive `k` from an ASPECT-CORRECTED sqrt(n) formula (bounded band
    COUNT growth), never from a fixed target band SIZE; a fixed-size-target
    version was tried and produced a catastrophic 220-580% gap at
    n=47608 (see academic note Sec.9 post-mortem)."""
    n = len(px)
    minx, maxx = min(px), max(px)
    w = (maxx - minx) or 1.0
    inv = k / (w * (1.0 + 1e-9))
    buckets: list[list[int]] = [[] for _ in range(k)]
    for i in range(n):
        b = int((px[i] - minx) * inv)
        if b >= k:
            b = k - 1
        elif b < 0:
            b = 0
        buckets[b].append(i)
    return buckets


def _aspect_k0(n, px, py, mult):
    width = max(px) - min(px) or 1.0
    height = max(py) - min(py) or 1.0
    aspect = width / height
    return max(1, round(math.sqrt(n * aspect / 2.0) * mult))


# ---------------------------------------------------------------------------
# ILERIYE BAKISLI (look-ahead) BANT YONELIMI -- 2026-07-26, olculdu ve uygulandi
# ---------------------------------------------------------------------------
# ESKI KURAL: bant ici kurucu bir CEVRIM uretir; cevrim, ONCEKI bandin cikis
# noktasina en yakin noktadan baslayacak sekilde dondurulurdu ve HEP ILERI
# okunurdu. Bu tek adimli/geriye bakisli bir secimdi: bandin NEREDE BITTIGINE
# hic bakmiyordu, dolayisiyla bant bir sonraki banda taban tabana zit uctan
# "teslim" edebiliyordu -> serpantin gecislerinde kor atlamalar.
#
# YENI KURAL: cevrimin TUM baslangiclari x IKI OKUMA YONU uzerinde
#     skor = d(onceki_cikis, giris) + LAMBDA * d(cikis, sonraki_bant_merkezi)
# en kucuk olan yonelim secilir. Yani bant hem gecmise (nereden geldim) hem
# gelecege (bir sonraki banda nereden teslim edecegim) bakar.
#
# MALIYET: bant basina O(m) ek is (2m yonelim, her biri O(1) skor). Bant ici
# kurucu zaten O(m log m) (greedy-edge) veya O(m^2) (farthest-insertion)
# oldugu icin bu ASIMPTOTIK OLARAK BEDAVA; olcumde toplam insa suresi
# pcb3038'de 0.61s -> 0.88s, fnl4461'de 1.04s -> 1.47s.
#
# OLCUM (13 enstans, tabana gore, negatif = iyi; 2026-07-26):
#   lambda=0.5 : ort -0.93%   lambda=1.0 : ort -1.14%  (6/13 kazanc, 1 kayip)
#   en buyuk kazanclar pma343 -10.1%, dcb2086 -3.1%, fnb1615 -3.2%
#   tek kayip xqf131 (n=131) +3.7% -- kucuk enstansta bant sayisi az, proxy
#   secimi gurultulu; v2/v3 GERCEK maliyetle sectigi icin orada yutulur.
#
# lambda TARAMAYA KATILMADI (kasitli): {0, 0.5, 1.0} taramasi aday sayisini
# 3x buyutup insa suresini 3x'e cikariyor ama sabit lambda=1.0'i GECMIYOR
# (ort -1.08% vs -1.14%) -- cunku _band_hybrid_tour ic taramayi GERCEK
# maliyetle degil _raw_length PROXY'siyle seciyor ve proxy bazen yanlis
# adayi tutuyor (lim963: tarama 3425, sabit lambda=1.0 3371). Pahali ve
# faydasiz oldugu icin elendi.
_LOOKAHEAD_LAMBDA = 1.0


def _orient_cycle(seg, xs, ys, entry_pt, next_c):
    """Bant cevrimini yonlendirir: tum baslangiclar x iki okuma yonu icinde
    d(giris) + LAMBDA*d(cikis, sonraki_merkez) en kucuk olani dondurur.
    next_c None ise (SON bant) yalniz girise gore secer -- bu, eski kuralin
    yon secimi eklenmis halidir."""
    m = len(seg)
    if m <= 2:
        return list(seg)
    hypot = math.hypot
    ex, ey = entry_pt
    # Skorlama O(1)/yonelim: aday listeyi KURMADAN yalniz uc noktalari bak.
    #   ileri  (seg[si:]+seg[:si])            -> giris seg[si], cikis seg[si-1]
    #   geri   ([fwd[0]] + fwd[1:][::-1])     -> giris seg[si], cikis seg[si+1]
    # (Naif hal her yonelim icin listeyi kopyaliyordu -> bant basina O(m^2);
    #  bu haliyle O(m) ve yalniz KAZANAN yonelim maddilestirilir.)
    best_s, best_si, best_fwd = None, 0, True
    for si in range(m):
        a = seg[si]
        da = hypot(xs[a] - ex, ys[a] - ey)
        if next_c is None:
            # ileriye bakilacak bir sey yok: eski kural (girise en yakin
            # baslangic, ileri okuma) ile birebir ayni secim.
            if best_s is None or da < best_s:
                best_s, best_si, best_fwd = da, si, True
            continue
        ncx, ncy = next_c
        for fwd in (True, False):
            b = seg[si - 1] if fwd else seg[(si + 1) % m]
            s = da + _LOOKAHEAD_LAMBDA * hypot(xs[b] - ncx, ys[b] - ncy)
            if best_s is None or s < best_s:
                best_s, best_si, best_fwd = s, si, fwd
    rot = seg[best_si:] + seg[:best_si]
    return rot if best_fwd else [rot[0]] + rot[1:][::-1]


# ---------------------------------------------------------------------------
# HAVUZ ADAY-LISTESI TAVANI (2026-07-28, kullanici karari)
# ---------------------------------------------------------------------------
# "Varyantli havuzlu yapilarda hic Greedy Snake'te ya da Greedy Edge'de k
# degeri en fazla 8 olsun -- esit olcme, adil olcum."
#
# Greedy Snake'in bant-ici greedy-edge'i her zaman bu tavanda kurulur; GE
# havuzu (runner._ge_pool_tours) da AYNI tavanin altinda tarar. Boylece iki
# ailenin havuz cesitliligi kendi mekanizmasindan gelir (bizde theta x bant,
# GE'de jitter), aday-listesi genisligi FARKINDAN degil.
#
# Bu tavan YALNIZ havuzlar icindir. Tek-turlu `greedy_edge` rakip satiri
# literatur degeri k=15'te BIRAKILDI: onu 8'e cekmek rakibi zayiflatirdi,
# yani adaleti ters yone bozardi. Duyarlilik varyanti greedy_edge@knn8_greedy
# zaten panelden secilebiliyor ve kanonik satirin YANINA duser.
#
# Olculdu (9 kume, GE havuzu 10..22 taramasindan bu tavana gecerken):
#   ge_pool4 +0.096%   ge_pool -0.544%   -- yani rakip zayiflamiyor, 10'luk
#   havuzda IYILESIYOR. Bedeli tek bir yapisal garanti: k=15 artik havuzun
#   icinde olmadigi icin "havuz tek-tur greedy_edge'den kotu olamaz"
#   esitsizligi DUSTU (olculen ihlal: 9 kumenin 2'sinde).
POOL_KNN_CAP = 8


def _greedy_band_order(band, xs, ys, entry_pt, next_c=None,
                       tie_seed=None, tie_eps=0.0):
    """Orders one band's points via `E.greedy_edge_tour` (k-NN candidate
    greedy-edge, O(m log m)), then ORIENTS the resulting cycle (start point +
    reading direction) via `_orient_cycle` -- greedy-edge naturally forms a
    good CYCLE with an arbitrary break point, so this orientation is what
    makes it usable as a continuity-respecting PATH segment.

    next_c (2026-07-26): bir SONRAKI bandin agirlik merkezi; verilirse
    yonelim ileriye bakisli secilir (bkz. `_orient_cycle`). None = son bant.

    tie_seed / tie_eps (2026-07-24, bant-ici cesitlendirme deneyi):
    E.greedy_edge_tour'un tohumlu beraberlik-bozma jitter'i bant ici
    siraya da gecirilir -- GE-havuzunun (ge_pool_repair) kanitlanmis
    "gercek havza cesitliligi" silahinin Greedy Snake bant ici kurucusuna
    birebir aynisi. tie_seed=None eski davranisla bit-ayni.

    k (2026-07-28): aday listesi genisligi `POOL_KNN_CAP` ile SINIRLI ve bu
    deger GE havuzuyla ORTAK -- bkz. o sabitin notu. Sayi degismedi (zaten
    8'di), degisen sey artik adinin olmasi: iki ailenin tavani tek yerden
    okunuyor, sessizce ayrisamiyor."""
    m = len(band)
    if m <= 3:
        return sorted(band, key=lambda i: math.hypot(xs[i] - entry_pt[0], ys[i] - entry_pt[1]))
    lxs = [xs[i] for i in band]
    lys = [ys[i] for i in band]
    local = E.greedy_edge_tour(lxs, lys, k=min(POOL_KNN_CAP, m - 1),
                               tie_seed=tie_seed, tie_eps=tie_eps)
    seg = [band[j] for j in local]
    return _orient_cycle(seg, xs, ys, entry_pt, next_c)


def _greedy_band_order_k(kv=None):
    """`_greedy_band_order`in aday-listesi genisligi SABITLENMIS hali.

    kv=None -> modul varsayilani (POOL_KNN_CAP). Greedy Snake v4'un k ekseni
    bunu kullanir; POOL_KNN_CAP'i gecici olarak degistirmek yerine acikca
    parametre gecmek tercih edildi -- global mutasyon coklu-is-parcacikli
    kullanimda sessizce yanlis tur uretirdi ve olcum betikleriyle uretim
    yolunu ayristirirdi."""
    if kv is None:
        return _greedy_band_order

    def f(band, xs, ys, entry_pt, next_c=None, tie_seed=None, tie_eps=0.0):
        m = len(band)
        if m <= 3:
            return sorted(band, key=lambda i: math.hypot(
                xs[i] - entry_pt[0], ys[i] - entry_pt[1]))
        local = E.greedy_edge_tour([xs[i] for i in band], [ys[i] for i in band],
                                   k=min(kv, m - 1),
                                   tie_seed=tie_seed, tie_eps=tie_eps)
        return _orient_cycle([band[j] for j in local], xs, ys, entry_pt, next_c)
    return f


def v4_label_theta(label):
    """v4 havuz etiketinden ACIYI okur. Etiket bicimleri:
        "th=-90*"          (yalniz aci ekseni -- eski/Far Snake yolu)
        "th=-90*/k=4"      (aci x k capraz havuzu -- varsayilan)
    Cozulemezse None doner (cagiran alani BOS birakir, patlamaz)."""
    try:
        return float(str(label).split("=", 1)[1].split("/", 1)[0].rstrip("*"))
    except (IndexError, ValueError, AttributeError):
        return None


def v4_label_knn(label):
    """v4 havuz etiketinden bant-ici k'yi okur; etikette yoksa None.

    Etiket "th=-90*/k=4" ya da duz havuzda "th=-90*/k=4/b=1x" biciminde
    olabilir -- ilk "/"e kadar okunur (2026-07-31 duzeltmesi: onceki hal
    "4/b=1x"i int()'e verip sessizce None donuyordu)."""
    s = str(label)
    if "/k=" not in s:
        return None
    try:
        return int(s.split("/k=", 1)[1].split("/", 1)[0])
    except ValueError:
        return None


#: FAR v1 = GREEDY v1'IN BIREBIR AYNASI (2026-09-07 kullanici karari:
#: "havuz olmamali, tek tur tek aday olmali").
#:
#: Bant sayisi ve kesme ekseni V1_CONFIG'TEN TURETILIR -- elle kopyalanmaz.
#: Sebep: iki satirin AYNI olmasi gereken tek sey bunlar; kopyalansa zamanla
#: sessizce ayrisir ve "tek fark bant-ici kurucu" iddiasi gecersizlesirdi.
#: (V1_CONFIG = (aci, bant-ici k, BANT SAYISI, x-ekseni mi))
FAR_V1_BAND = V1_CONFIG[2]
FAR_V1_AXIS = V1_CONFIG[3]

#: FAR SNAKE'IN KENDI CESITLILIK EKSENI (2026-07-28, olculdu).
#:
#: farthest-insertion'in bant-ici BASLANGIC DUGUMU. Kesir olarak verilir ve
#: bant boyutuna gore olceklenir (start = int(f*m) % m), boylece her bant
#: boyutunda AYNI goreli dagilim kullanilir ve secim bant sayisindan bagimsiz
#: tekrarlanabilir kalir.
#:
#: NEDEN GEREKLI -- olculen kok neden: Greedy Snake'in havuz ekseni theta'dir
#: ve greedy-edge'de calisir, cunku rotasyon k-NN aday listesinin
#: beraberlik-bozmasini oynatir (tamsayi kafeslerde esit kenar boldur).
#: farthest-insertion ise DONMEYE DUYARSIZDIR: max-min uzaklik secimi ve en
#: ucuz ekleme, Oklidyen uzaklik rotasyonda degismedigi icin ayni turu verir.
#: Bant sayisi 1'e dustugunde (kucuk/orta n'de sik) theta ekseni bu yuzden
#: SIFIR cesitlilik uretir -- ali535'te olculdu: 4 aday, 1 benzersiz kenar
#: kumesi, yayilim %0.00.
#:
#: OLCULDU (6 kume, esit genislik 4 aday, saf theta ekseni vs saf start
#: ekseni; en iyi ham tur maliyeti):
#:     xqf131   629 -> 596   (-5.2%)     ali535  227224 -> 221021  (-2.7%)
#:     bcl380  1842 -> 1800  (-2.3%)     rat783    9770 ->   9744  (-0.3%)
#:     pma343  1720 -> 1717  (-0.2%)     bck2217   8048 ->   8048  (esit)
#: 6/6 kumede COKME YOK (4/4 benzersiz kenar kumesi), 5/6'da daha iyi, hic
#: kayip yok.
#:
#: TASARIM SOZLESMESI (kullanici karari 2026-07-28): iki havuz ailesi ESIT
#: GENISLIKTE ama KENDI eksenlerinde calisir -- ge_pool'un saf jitter'a
#: gecmesiyle BIREBIR ayni gerekce (bkz. HAVUZ_ADALETI_...NOT.md §5b:
#: "her aile kendi EN IYI eksenini kullanir"). Ayrinti:
#: FAR_SNAKE_AILESI_AKADEMIK_NOT.md §9.
def far_start_fracs(count):
    """`count` adet baslangic KESRI -- van der Corput (taban 2) dizisi.

    Iki ozelligi birden saglamak zorunda oldugu icin duz `v/count` degil:

      1) IC-ICE GECME (nested prefix): ilk k eleman, count'tan BAGIMSIZ
         olarak hep aynidir. `fi_pool4` bu sayede `fi_pool`'un ONEK ALT
         KUMESI olur ve maliyet(fi_pool4) >= maliyet(fi_pool) esitsizligi
         YAPISAL kalir -- ge_pool4 ⊂ ge_pool ile birebir ayni sozlesme.
         (`v/count` bunu SAGLAMAZ: count degisince tum degerler kayar.)
      2) HER ONEKTE IYI YAYILIM: 0, 1/2, 1/4, 3/4, 1/8, 5/8, ... -- ilk 4
         eleman {0, 1/4, 1/2, 3/4} kumesini verir, yani dar havuz da tum
         araligi tarar.

    Bkz. `FAR_STARTS` notu (eksenin NEDEN start oldugu, olcumler)."""
    out = []
    for v in range(count):
        f, den, i = 0.0, 0.5, v
        while i:
            f += (i & 1) * den
            i >>= 1
            den *= 0.5
        out.append(f)
    return out


#: Far Snake v2'nin 4 adayi. `far_start_fracs(4)` = {0, 1/2, 1/4, 3/4} --
#: kume olarak (0, 0.25, 0.5, 0.75) ile AYNI, sirasi ic-ice gecme kuralindan
#: gelir (bkz. `far_start_fracs`).
#: 2026-08-01 (kullanici karari -- KIYAS ADALETI): 4 -> 2 aday.
#: Rakip ge_pool2 de 2 adayli; 4 adayla 2 adayi yenmek butcenin sonucu olur.
#: OLCULDU (27 kume <=3000, tek baslangic ortalamalari):
#:   0=11.33  0.125=11.26  0.25=11.47  0.375=11.72  0.5=10.58  0.625=11.18
#:   0.75=11.78  0.875=11.56
#: En iyi 2'li {0.375, 0.5} = 9.91%; mevcut 4'lu {0,0.5,0.25,0.75} = 9.80%.
#: Yani 4'ten 2'ye inmenin bedeli yalnizca +0.11 puan -- adalet bu fiyata
#: alinir. van der Corput ONEKI ({0,0.5} = 10.13%) DEGIL, olculen en iyi
#: cift kullanilir: artik bir 4'lu havuz kalmadigi icin "onek alt kumesi"
#: garantisinin korunacagi bir ust kume de YOKTUR.
FAR_STARTS = (0.375, 0.5)

#: FAR AILESININ MERDIVEN CEKIRDEGI -- far_snake_v1'in baslangic kesiri.
#: 2026-08-02'de EKLENDI (olculmus merdiven kacagi). Sorun: 4->2 aday
#: indirimi `start=0.0`'i havuzdan dusurdu, ama `far_snake_v1_tour` hala
#: start=0 kuruyordu. Yani v1'in turu ARTIK HAVUZUN ICINDE DEGILDI ve
#: docstring'in "MERDIVEN KORUNUR ... YAPISALDIR" iddiasi YANLISTI.
#: OLCULDU (10 kume): maliyet(far v1) >= maliyet(far v2) yalniz 7/10'da
#: tutuyordu -- d1291, d657, fl1400'de v1 v2'yi GECIYORDU.
#: COZUM: v1'i havuzun ILK adayina bagla (Greedy tarafinda V1_CONFIG'in
#: V2_CONFIGS[0] olmasiyla BIREBIR ayni desen). Boylece garanti tekrar
#: YAPISAL olur ve KALITE BEDELI YOKTUR -- alternatif (havuza 0.0'i geri
#: koymak) olculen en iyi cifti {0.375,0.5}=9.91'den {0,0.5}=10.13'e
#: dusururdu, yani +0.22 puan oderdi. Bu yol 0 puan oder.
FAR_V1_START = FAR_STARTS[0]


def _farthest_band_order(band, xs, ys, entry_pt, next_c=None, start_frac=0.0):
    """Orders one band's points via `E.farthest_insertion_tour`
    (UNACCELERATED O(m^2) -- no k-NN structure backs it), then rotates the
    resulting cycle-reading to start at the point nearest `entry_pt` (same
    trick as `_greedy_band_order`). Callers MUST keep `m` bounded (see
    `_farthest_mult`) since this has no sub-quadratic fallback.

    start_frac (2026-07-28): farthest-insertion'in baslangic dugumu, bant
    boyutuna gore kesir olarak. `FAR_STARTS` notuna bakiniz -- bu, Far
    Snake ailesinin KENDI cesitlilik eksenidir ve `_greedy_band_order`'in
    tie_seed/tie_eps jitter'inin tam karsiligidir. start_frac=0.0 eski
    davranisla BIT-AYNIdir."""
    m = len(band)
    if m <= 3:
        return sorted(band, key=lambda i: math.hypot(xs[i] - entry_pt[0], ys[i] - entry_pt[1]))
    lxs = [xs[i] for i in band]
    lys = [ys[i] for i in band]
    local = E.farthest_insertion_tour(lxs, lys,
                                      start=int(start_frac * m) % m)
    seg = [band[j] for j in local]
    return _orient_cycle(seg, xs, ys, entry_pt, next_c)


# ---------------------------------------------------------------------------
# BANT SINIRI DIKISI (boundary stitching) -- 2026-07-26, olculdu ve uygulandi
# ---------------------------------------------------------------------------
# Bant A biter, bant B baslar; gecis kenari bant ici kurucularin HICBIRININ
# gormedigi tek kenardir (her kurucu yalniz kendi bandini optimize eder).
# Cozum: gecisin iki yanindaki m+m noktayi, PENCERE DISINDAKI iki komsuyu
# (P ve S) SABIT tutarak en kisa siralamayla yeniden bagla.
#
# Neden guvenli: pencere ici brute-force, mevcut siralamayi da aday olarak
# icerir -> secilen siralama HER ZAMAN <= mevcut uzunluk. Yani bu adim turu
# ASLA KOTULESTIREMEZ (olcumde de 12 enstansta 0 kayip).
#
# Bedeli: gecis basina (2m)! permutasyon x O(m) uzunluk hesabi. m=3 -> 720
# permutasyon; bant sayisi ~O(sqrt(n)) oldugu icin toplam ihmal edilebilir.
# YALNIZ KAZANAN adaya uygulanir (tarama icindeki her adaya degil), boylece
# v3'un 10-16 adayli havuzunda maliyet 1x kalir.
#
# OLCUM (12 enstans, look-ahead UZERINE, 2026-07-26):
#   m=2 : ort -0.022%  (6 kazanc, 0 kayip)
#   m=3 : ort -0.064%  (7 kazanc, 0 kayip)
# Kazanc kucuk -- cunku ileriye bakisli yonelim gecislerin buyuk kismini
# zaten duzeltiyor; dikis geriye kalan yerel kirilmalari topluyor.
_STITCH_M = 3


def _boundary_stitch(tour, bounds, xs, ys, m=_STITCH_M):
    """Bant gecislerindeki 2m'lik pencereleri uclari sabit tutarak yeniden
    siralar. `bounds`: tur icindeki gecis konumlari (tour[b-1] ile tour[b]
    farkli bantlarin son/ilk noktasi). Tur yerinde DEGISTIRILMEZ; yeni liste
    doner."""
    if not bounds or m < 2:
        return tour
    t = list(tour)
    n = len(t)
    hypot = math.hypot
    for cut in bounds:
        lo, hi = cut - m, cut + m
        if lo - 1 < 0 or hi >= n:
            continue                      # pencere tur uclarina tasiyor
        P, S = t[lo - 1], t[hi]
        win = t[lo:hi]
        if len(win) < 3:
            continue
        best, best_w = None, None
        for perm in itertools.permutations(win):
            d = hypot(xs[P] - xs[perm[0]], ys[P] - ys[perm[0]])
            for a, b in zip(perm, perm[1:]):
                d += hypot(xs[a] - xs[b], ys[a] - ys[b])
            d += hypot(xs[perm[-1]] - xs[S], ys[perm[-1]] - ys[S])
            if best is None or d < best:
                best, best_w = d, perm
        t[lo:hi] = list(best_w)
    return t


#: `_band_hybrid_tour`'in EN SON kurdugu kazanan adayin ic yapisi. Yalniz
#: tanilama/gorsellestirme icindir (dashboard "Gorsellestirme" sekmesi
#: Greedy Snake v1'i adim adim cizerken kullanir); kurucu davranisini ETKILEMEZ.
LAST_BUILD: dict = {}


def _band_hybrid_tour(xs, ys, order_fn, mult, dks=(-1, 0, 1),
                      axes=(True, False), k0=None, collect=None):
    """Shared driver for the band-hybrid family: aspect-corrected macro-
    bands (bounded sweep: len(dks) counts x len(axes) axes, proxy-scored),
    each band's internal order delegated to `order_fn(band, px, py,
    entry_pt)`, entry_pt threaded from the previous band's exit point for
    continuity. lite1 icin ic tarama tamamen kapatilabilir:
    dks=(0,), axes=(True,).

    collect (2026-07-31): bir liste verilirse IC TARAMANIN HER ADAYI
    (bant sayisi x eksen) dikisi yapilmis haliyle
        (bant_sayisi, x_ekseni_mi, tur)
    ucluleri olarak o listeye EKLENIR. Donus degeri DEGISMEZ (yine yalniz
    kazanan) -- yani mevcut cagiranlar BIT-AYNI kalir.
    Gerekce: ic tarama zaten 2 bant sayisi x 2 eksen = 4 tur uretip 3'unu
    ATIYOR. Bu turlar HESAPLANMIS maliyettir; havuza aday olarak verilince
    onarim havzasi bedelsiz genisler (olculdu: pcb3038 -0.27, bgb4355 -0.10).

    k0 (2026-07-30): bant sayisini DOGRUDAN dayatir; verilirse `mult`
    yalnizca kayit icin tasinir (LAST_BUILD). Greedy Snake v4'un satir
    yukseklik havuzu bunu kullanir -- aksi halde istenen bant sayisini
    elde etmek icin `_aspect_k0` formulunu ters cevirip IKINCI bir yerde
    kopyalamak gerekirdi ve iki formul zamanla sessizce ayrisirdi."""
    n = len(xs)
    if n <= 3:
        return list(range(n))
    k0 = _aspect_k0(n, xs, ys, mult) if k0 is None else max(1, int(k0))

    def build(px, py, k):
        # Bos bantlar ONCE elenir: ileriye bakisli yonelim (2026-07-26) "bir
        # SONRAKI bant" kavramina dayanir, o yuzden bantlar once somut bir
        # listeye alinir ve her birinin agirlik merkezi hesaplanir.
        bands = [b for b in _band_indices(px, k) if b]
        centers = [(sum(px[i] for i in b) / len(b),
                    sum(py[i] for i in b) / len(b)) for b in bands]
        tour: list[int] = []
        bounds: list[int] = []            # bant gecislerinin tur icindeki yeri
        entry = None
        for bi, band in enumerate(bands):
            E.check_construction_deadline()   # 300s insa kapagi (runner kurar)
            if entry is None:
                entry = (px[band[0]], min(py[i] for i in band))
            # son bantta ileriye bakilacak bir sey yok -> next_c None
            next_c = centers[bi + 1] if bi + 1 < len(bands) else None
            seg = order_fn(band, px, py, entry, next_c)
            if tour:
                bounds.append(len(tour))
            tour.extend(seg)
            entry = (px[seg[-1]], py[seg[-1]])
        return tour, bounds

    best_tour, best_proxy, best_bounds = None, None, None
    best_k, best_axis = k0, True
    # ---- ADAY BANT SAYILARI (2026-07-31, olculdu) -------------------------
    # IC TARAMA BUDANDI: aday bant sayilari {1, 2} (BAND_COUNT_CANDIDATES).
    #
    # OLCUM 1 -- kazanan bant sayisi: gercek havuz kosumlari izlendi (4 kume,
    # 42 ic insa). Kazanan %69 bant=1, %31 bant=2; BANT >= 3 HICBIR KEZ
    # kazanmadi. Ayrica bant sayisi TAM SABITLENEREK yapilan ayri taramada
    # (28 kume, theta = -90/0/45) en iyi bant sayisi 25 kumede 1, 2 kumede 2,
    # 1 kumede 3'tu -- yani optimum n'den BAGIMSIZ olarak 1-2'de kaliyor.
    #
    # OLCUM 2 -- budamanin bedeli: 26 kumede MEVCUT pencere ile {1,2}
    # karsilastirildi -> 26/26 BIT-AYNI sonuc (0 iyilesme, 0 kotulesme).
    # {1,2,k0} varyanti da hicbir sey eklemedi. Kazanc SURE: buyuk n'de
    # 1.3-1.6x hizlanma (frh19289 65.7s -> 40.0s, ido21215 75.3s -> 49.3s).
    #
    # NEDEN ONEMLI: eski pencere {k0-1, k0, k0+1} + {1} idi ve k0 formulu n
    # ile buyudugu icin (n=19289'da k0=6) her aday BOSA giden 6 insa
    # yapiyordu. Havuz mimarisinde sure = aday sayisi demektir; burada
    # kazanilan sure ACI x k havuzuna aktarilir, yani olcumun ODEDIGI eksene.
    #
    # !! GENISLETME NOTU !! Bu kume 26 TSPLIB/VLSI kumesinde olculdu. Baska
    # yapida bir koleksiyon (orn. DIMACS kumeli ornekler) eklenirse once
    # yeniden olculmeli; genisletmek tek satirlik degisikliktir.
    #
    # dks TEK elemanliysa budama UYGULANMAZ -- o cagri "tam olarak bu bant
    # sayisi" sozlesmesidir (olcum betikleri ve k0 dayatmasi buna dayanir).
    #
    _kcands = (sorted({max(1, k0 + dk) for dk in dks}) if len(dks) == 1
               else list(BAND_COUNT_CANDIDATES))
    for k in _kcands:
        for axis_xy in axes:
            t, bnd = build(xs, ys, k) if axis_xy else build(ys, xs, k)
            proxy = _raw_length(t, xs, ys)
            if collect is not None:
                # Aday olarak DISARI verilecekse dikisi de yapilmali: kazanan
                # dikisli, digerleri dikissiz olsaydi kiyas adaletsiz olurdu.
                collect.append((k, bool(axis_xy),
                                _boundary_stitch(t, bnd, xs, ys)))
            if best_proxy is None or proxy < best_proxy:
                best_proxy, best_tour, best_bounds = proxy, t, bnd
                best_k, best_axis = k, axis_xy
    # Bant siniri dikisi YALNIZ kazanan adaya (bkz. _boundary_stitch notu):
    # asla kotulestiremez, o yuzden proxy secimini bozmaz.
    final = _boundary_stitch(best_tour, best_bounds, xs, ys)
    # ---- ic yapiyi disariya ac (2026-07-27) --------------------------------
    # Gorsellestirme sekmesi Greedy Snake v1'i ADIM ADIM anlatabilsin diye kazanan
    # adayin ic yapisi burada saklanir. Hesaplama YAPILMAZ, yalnizca zaten
    # uretilmis degerler kaydedilir -- sicak yol etkilenmez. Tek-is-parcacikli
    # kullanim varsayilir (runner ve sunucu boyle cagiriyor); yaris kosulu
    # halinde yalniz bu tani sozlugu tutarsiz olur, TUR ETKILENMEZ.
    LAST_BUILD.clear()
    LAST_BUILD.update({
        "k0": k0, "k": best_k, "axis_x": bool(best_axis), "mult": mult,
        "bounds": list(best_bounds or []),
        "tour_prestitch": list(best_tour),
        "tour": list(final),
        "proxy": best_proxy,
        "n_bands": len(best_bounds or []) + 1,
        "sweep": len(_kcands) * len(axes),
        "k_cands": list(_kcands),
    })
    return final


def snake_greedy_band_pool(xs, ys, mult=0.15, cost_fn=None, thetas=None,
                           extra_thetas=()):
    """snake_greedy_band'in ADAY HAVUZU -- `snake_v2_pool`'un TA KENDISI, tek
    fark mult=0.15 (v2'de 0.05). Bkz. snake_v2_pool ortak-govde notu."""
    return snake_v2_pool(xs, ys, mult=mult, cost_fn=cost_fn,
                         order_fn=_greedy_band_order, thetas=thetas,
                         extra_thetas=extra_thetas)


def snake_greedy_band_tour(xs, ys, mult=0.15, cost_fn=None, thetas=None,
                           extra_thetas=()):
    """Strip macro-bands + greedy-edge WITHIN each band (see module
    docstring).

    KARMASIKLIK (2026-07-27 DUZELTMESI): burada "total ~O(n^1.5 log n)"
    yaziyordu -- YANLIS ve kendi icinde tutarsizdi. Bant basina O(m log m) ise
    k bant uzerinde toplam O(n log(n/k)) = O(n log n)'dir. Olculdu (10 kume,
    sure-n log-log egimi): snake_greedy_band n^1.10, greedy_snake_v1 n^1.08,
    tek-parca greedy_edge n^1.09 -- ucu de AYNI SINIF. O(n^1.5) iddiasi
    snake_farthest_band'e aittir (bant ici farthest-insertion HIZLANDIRILMAMIS
    O(m^2); olculdu n^1.51). Eski rakam muhtemelen _orient_cycle'in O(m^2)
    oldugu donemden kalmaydi; o hata 2026-07-26'da O(m)'e indirildi.

    2026-07-27: artik Greedy Snake v2 ile AYNI aci havuzundan gecer
    ({0, θ★, θ★±10}, θ★=_strip_oracle_theta, secim gercek maliyetle) --
    onceden yalnizca θ=0'da kuruluyor, aci taramasini runner FARKLI bir
    kestiriciyle yapiyordu. Gerekce ve olcumler: snake_v2_pool docstring'i."""
    pool = snake_greedy_band_pool(xs, ys, mult=mult, cost_fn=cost_fn,
                                  thetas=thetas, extra_thetas=extra_thetas)
    return pool[0][1] if pool else list(range(len(xs)))


def _farthest_mult(n):
    """Farthest-insertion is UNACCELERATED O(m^2) per band (unlike greedy-
    edge's O(m log m)) -- left at snake_greedy_band's mult=0.15, band size
    grows as sqrt(n) and n=47608 alone takes ~114s. Scaling mult UP as n
    grows shrinks the average band size faster than sqrt(n), keeping the
    O(bands x band_size^2) = O(n x band_size) total bounded in practice
    (n=47608 at mult=0.3: 61s, better QUALITY too -- more, smaller bands
    let farthest-insertion's O(m^2) global-coverage search stay precise
    without over-diluting across a huge point set)."""
    if n <= 3000:
        return 0.15
    if n <= 15000:
        return 0.22
    return 0.30


def _far_dks(n):
    """Far Snake ailesinin ic dk taramasi -- TEK KAYNAK.

    Farthest-insertion bant-ici HIZLANDIRILMAMIS O(m^2): buyuk n'de ic dk
    taramasi tek-k'ya daraltilir (aci havuzu zaten 4 insa yapiyor, ikisi
    carpilmamali). far_snake_v1/v2/v3'un UCU DE burayi cagirmak ZORUNDA --
    aksi halde aday kumeleri ic-ice gecmez ve v1 >= v2 >= v3 merdiveni
    (bkz. `far_snake_v1_tour` aile sozlesmesi) kirilir."""
    return (-1, 0, 1) if n <= 5000 else (0,)


# ===========================================================================
# FAR SNAKE AILESI (2026-07-28, hakem talebi)
# ===========================================================================
# Hakem itirazi: "Greedy Snake icin kurdugunuz butun merdiveni (v1/v2/v3, ham
# + havuzlu + onarimli) farthest-insertion kurucusu icin de kurun; yoksa
# 'serpantin makro-yapisi + guclu bant-ici kurucu' iddianiz TEK bir bant-ici
# kurucuda olculmus olur."
#
# Cevap: Far Snake, Greedy Snake'in BIREBIR AYNASIdir. Tek degisken bant-ici
# kurucudur:
#
#     Greedy Snake : _greedy_band_order   (greedy-edge, k<=POOL_KNN_CAP)
#     Far Snake    : _farthest_band_order (farthest-insertion, O(m^2))
#
# Makro yapinin GERI KALANI (aspekt duzeltmeli serit bantlari, ileriye bakisli
# yonelim, bant siniri dikisi, aci vekili/aci havuzu, gercek-maliyetle secim,
# havuz genisligi, onarim katmani ve butcesi) IKI AILEDE DE ayni kodu cagirir.
#
# IKI KASITLI ASIMETRI VAR VE IKISI DE FAR SNAKE'IN ALEYHINEDIR (yani
# olcumu bizim lehimize saptirmaz):
#   1) mult: Greedy Snake sabit 0.05 kullanir; Far Snake `_farthest_mult(n)`
#      ile n buyudukce bantlari KUCULTUR. Sebep karmasiklik: O(m^2) kurucu
#      sabit bant boyutunda n=47608'de tek basina ~114s aliyor.
#   2) ic dk taramasi: Greedy Snake her zaman (-1,0,1); Far Snake n>5000'de
#      (0,)'a daralir -- yani BUYUK n'DE DAHA AZ ADAY gorur.
# Ikisi de kaliteyi degil SUREYI sinirlamak icindir ve dogrudan
# farthest-insertion'in kendi karmasikligindan dogar; ikisi de Far Snake'e
# daha DAR bir arama verir. Ayrinti: FAR_SNAKE_AILESI_AKADEMIK_NOT.md.
# ===========================================================================

def far_snake_v1_tour(xs, ys, mult=None, theta_deg=None):
    """Far Snake v1: TEK aci, TEK tur, TEK ADAY -- ic havuz YOK.

    `snake_v1_tour`'un BIREBIR AYNASI: ayni aci vekili, ayni bant sayisi ve
    ayni kesme ekseni (ikisi de V1_CONFIG'ten gelir -- bkz. FAR_V1_BAND),
    ayni dikis. TEK degisken bant-ici kurucudur:
        greedy_snake_v1 -> greedy-edge         (k=4)
        far_snake_v1    -> farthest-insertion  (start_frac=FAR_V1_START)
    `mult` yalnizca kayda gecer (bant sayisi k0 ile dayatildigi icin
    _aspect_k0 formulu kullanilmaz).

    2026-09-07: IC HAVUZ KALDIRILDI (kullanici karari "havuz olmamali,
    havuz olursa adil olmaz"). Eskiden n<=5000'de b in {1,2} x 2 eksen = 4
    tur kurulup ham-uzunluk VEKILIYLE en iyisi seciliyordu; n>5000'de
    b=_aspect_k0 (pla7397'de 14). Yani satir rakiple esit butcede DEGILDI
    ve "tek tur" etiketi yanlisti.

    ================== AILE SOZLESMESI (KIYAS GECERLILIGI) ==================
    Greedy Snake'te oldugu gibi v1/v2/v3 ayni insa cekirdeginin giderek
    genisleyen uc basamagidir; ucu de ayni aci vekilini (`theta_proxy`), ayni
    mult'u (`_farthest_mult(n)`) ve ayni ic taramayi (`_far_dks(n)`) kullanir.
    Tek fark ADAY KUMESININ genisligidir:

        far v1 : {theta*}                x {fm}          ->  1 aday
        far v2 : {0, theta*, theta*+-10} x {fm}          ->  4 aday
        far v3 : {0, theta*+-10/20/30}   x {fm, 1.6*fm}  -> 26+ aday (en iyi 10)

    Aday kumeleri IC-ICE GECIKTIR ve v2/v3 adaylar arasindan GERCEK TSPLIB
    maliyetiyle secer, dolayisiyla

        maliyet(far v1) >= maliyet(far v2) >= maliyet(far v3_ham)

    YAPISAL bir esitsizliktir (deneysel degil) -- Greedy Snake merdiveniyle
    birebir ayni gerekce.

    !!! theta_deg'i DISARIDAN GECMEYIN !!! Sebep `snake_v1_tour`'daki ile
    AYNIdir: baska bir aci kestiricisi v1'i v2'nin aday kumesinin disina
    cikarir ve merdiveni bozar. `theta_deg` yalnizca aci-duyarliligi
    deneyleri icindir; o durumda sonuc ailenin v1 basamagi DEGILDIR."""
    n = len(xs)
    if mult is None:
        mult = _farthest_mult(n)
    th = theta_proxy(xs, ys) if theta_deg is None else float(theta_deg)
    th = ((th + 90.0) % 180.0) - 90.0
    rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    # 2026-08-02: baslangic kesiri ARTIK havuzun ilk adayindan gelir
    # (FAR_V1_START = FAR_STARTS[0]), sabit 0.0'dan DEGIL. Merdiven
    # garantisinin tek dayanagi budur; gerekce FAR_V1_START notunda.
    import functools          # modul govdesinde yok; havuz da boyle aliyor
    _of = (_farthest_band_order if not FAR_V1_START else
           functools.partial(_farthest_band_order, start_frac=FAR_V1_START))
    # TEK TUR, TEK ADAY (2026-09-07). dks=(0,) + axes=(True,) + k0 verilmesi
    # ic taramayi TAMAMEN kapatir: _band_hybrid_tour tam 1 tur kurar ve
    # secim yapmaz. Bant sayisi/eksen V1_CONFIG'ten gelir, yani satir
    # greedy_snake_v1 ile AYNI makro-yapiyi kullanir; tek degisken bant-ici
    # kurucudur. Eski hal 4 tur kurup ham-uzunluk VEKILIYLE seciyordu --
    # "tek tur" etiketi yanlisti ve rakiple esit butcede degildi.
    px, py = (rx, ry) if FAR_V1_AXIS else (ry, rx)
    return _band_hybrid_tour(px, py, _of, mult, dks=(0,), axes=(True,),
                             k0=FAR_V1_BAND), th


def far_snake_v2_pool(xs, ys, mult=None, cost_fn=None, starts=None,
                      theta_deg=None):
    """Far Snake v2'nin ADAY HAVUZU -- 4 aday, greedy_snake_v2 / ge_pool4 ile
    ESIT GENISLIKTE.

    ==================== EKSEN SECIMI (2026-07-28) =======================
    Havuz TEK acida (θ★ = `theta_proxy`, ailenin ortak vekili) kurulur ve
    cesitliliginin TAMAMINI `FAR_STARTS` ekseninden -- farthest-insertion'in
    bant-ici BASLANGIC DUGUMUNDEN -- alir:

        f=0.00  -> baslangic dugumu 0    (== far_snake_v1'in turu)
        f=0.25 / 0.50 / 0.75             (bant boyutuna gore olceklenir)

    Bu, `ge_pool4`'un saf-jitter tasarimiyla BIREBIR ayni sozlesmedir:
    **esit GENISLIK, kendi EKSENI**. Uc havuz ailesi de 4 aday uretir, ama
    her biri kendi kurucusunun gercekten cesitlendigi eksende:

        greedy_snake_v2 : θ ekseni     ({0, θ★, θ★±10})
        ge_pool4        : jitter ekseni (tie_seed/tie_eps, k=8 sabit)
        far_snake_v2    : start ekseni  (FAR_STARTS, θ★ sabit)

    NEDEN θ DEGIL (olculdu, kok neden): farthest-insertion DONMEYE
    DUYARSIZDIR -- Oklidyen uzakliklar rotasyonda degismedigi icin max-min
    secimi ve en ucuz ekleme ayni turu verir. Bant sayisi 1'e dustugunde
    (kucuk/orta n'de sik) θ ekseni bu yuzden SIFIR cesitlilik uretir;
    ali535'te olculdu: 4 aday, 1 benzersiz kenar kumesi, yayilim %0.00 ve
    havuz+onarim satiri bu yuzden geriliyordu. Tam olcum: `FAR_STARTS` notu.

    MERDIVEN KORUNUR: far_snake_v1 (θ★, start=0) bu kumenin ICINDEdir ve
    secim GERCEK maliyetle yapilir -> maliyet(far v1) >= maliyet(far v2)
    YAPISALDIR.

    2026-07-28 ADI DEGISTI: `snake_farthest_band_pool` -> `far_snake_v2_pool`.
    Sebep salt adlandirma degil KIYAS GECERLILIGIdir: bu satir zaten
    greedy_snake_v2'nin tam muadiliydi ama benchmark'ta tek basina duran bir
    "ablasyon bileseni" gibi gorunuyordu. Artik ailenin v2 basamagi ve RAKIP
    satirdir.

    Donus: [(etiket, tur, skor), ...] skora gore ARTAN sirali."""
    import functools
    n = len(xs)
    if mult is None:
        mult = _farthest_mult(n)
    if starts is None:
        starts = FAR_STARTS
    th = theta_proxy(xs, ys) if theta_deg is None else float(theta_deg)
    th = ((th + 90.0) % 180.0) - 90.0
    rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    dks = _far_dks(n)
    pool = []
    for f in starts:
        order_fn = (_farthest_band_order if not f else
                    functools.partial(_farthest_band_order, start_frac=f))
        t = _band_hybrid_tour(rx, ry, order_fn, mult, dks=dks)
        score = cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)
        pool.append((f"th={th:g}/s={f:g}", t, score))
    # Kararli siralama: esit skorlu adaylarda uretim sirasi korunur, boylece
    # "kazanan varyant" etiketi kosumdan kosuma degismez.
    pool.sort(key=lambda r: r[2])
    return pool


def far_snake_v2_tour(xs, ys, mult=None, cost_fn=None, starts=None,
                      theta_deg=None):
    """Strip macro-bands + farthest-insertion WITHIN each band (see module
    docstring); `far_snake_v2_pool`'un kazanan adayi.

    Complexity is genuinely worse than Greedy Snake's (O(m^2) per band, no
    k-NN acceleration exists for farthest-insertion) -- `mult` (and therefore
    band count/size) is scaled with n via `_farthest_mult`, and the dk-sweep
    is narrowed for large n (`_far_dks`), specifically to keep this bounded;
    see `_farthest_mult`'s docstring for the measured numbers."""
    pool = far_snake_v2_pool(xs, ys, mult=mult, cost_fn=cost_fn,
                             starts=starts, theta_deg=theta_deg)
    return pool[0][1] if pool else list(range(len(xs)))


#: Geriye donuk takma adlar. Eski cagri yerleri (tanilama betikleri, GPU
#: kardesler) bozulmasin diye duruyorlar; DAVRANIS BIT-AYNIdir.
snake_farthest_band_pool = far_snake_v2_pool
snake_farthest_band_tour = far_snake_v2_tour


def _strip_oracle_theta(xs, ys, step=10):
    """Coarse scan of the plain strip tour's raw length over [-90, 90] --
    cheap O((180/step) x n log n) angle PROXY used by snake_v2 to seed its
    candidate thetas. (E1 ablation in results/SNAKE_TESHIS.md: the band
    hybrid's own oracle angle correlates strongly with the strip's --
    fl3795: -90/-90, rl5915: 70/80 -- while PCA does not.)"""
    best_th, best_len = 0.0, None
    for th in range(-90, 91, step):
        rx, ry = (xs, ys) if th == 0 else E.rotate_coords(xs, ys, float(th))
        t = E.snake_order(rx, ry, 0)
        L = _raw_length(t, xs, ys)
        if best_len is None or L < best_len:
            best_len, best_th = L, float(th)
    return best_th


# ---------------------------------------------------------------------------
# ACI VEKILI ANAHTARI (2026-07-28)
# ---------------------------------------------------------------------------
# v1/v2/v3 ve bant hibritlerinin HEPSI aciyi buradan alir. Tek bir anahtar
# olmasinin sebebi yapisaldir: aday kumelerinin ic-ice gecmesi (v1 ⊂ v2 ⊂ v3)
# ve dolayisiyla maliyet(v1) >= maliyet(v2) >= maliyet(v3) garantisi, ancak
# ucu de AYNI vekili kullanirsa gecerlidir. Vekili yalniz v1 icin degistirmek
# merdiveni kirar (26 ornekte olculdu: 22'sinde v1'in acisi v2'nin aday
# kumesinin disinda kaldi, 2'sinde v1 v2'yi gecti).
#
#   "grid"         -- grid_theta.theta_for: izgara acisi; guven esigin altinda
#                     kalirsa KENDILIGINDEN strip-oracle'a duser.
#   "strip_oracle" -- eski davranis, birebir.
#
# NEDEN VARSAYILAN "grid" (bkz. IZGARA_ACISI_VE_SERIT_SAYISI_AKADEMIK_NOT.md):
#   * Kanitlanmis optimum LKH turlarina karsi 6/6 sette 0.05-0.43 derece hata;
#     strip-oracle 3/6 sette 10-30 derece sapiyor.
#   * 8 buyuk kanitlanmis-optimum VLSI kumesinde v1 cekirdegi %18.23 -> %16.92
#     ve DAHA UCUZ (O(180n) numpy taramasi vs 19 boustrophedon insasi).
#   * v2/v3 icin risk YOK: theta=0 aday kumesinde her zaman var ve secim
#     GERCEK maliyetle yapiliyor, yani yeni vekil kotu bir aci onerse bile
#     sonuc theta=0'dan kotu OLAMAZ. Risk yalniz v1'dedir (tek aci) ve orada
#     da olcum lehte.
ANGLE_PROXY = "grid"


def theta_proxy(xs, ys):
    """Ailenin ORTAK aci vekili. `ANGLE_PROXY` anahtarina gore dagitir."""
    if ANGLE_PROXY == "strip_oracle":
        return _strip_oracle_theta(xs, ys)
    # Gec import: grid_theta bu modulu import ediyor, dairesel bagimliligi
    # modul yukleme aninda degil cagri aninda cozeriz.
    import grid_theta as _GT
    return _GT.theta_for(xs, ys, fallback=None)


def snake_v1_tour(xs, ys, mult=0.05, theta_deg=None, bands=None):
    """Greedy Snake v1: ailenin EN TEMEL uyesi -- TEK aci, TEK tur.

    ================== AILE SOZLESMESI (KIYAS GECERLILIGI) ==================
    v1/v2/v3 AYNI yontemin giderek genisleyen uc basamagidir; ucu de ayni
    insa cekirdegini (`_band_hybrid_tour` + `_greedy_band_order`, ayni ic
    dk/eksen taramasi) ve ayni aci vekilini (`_strip_oracle_theta`) kullanir.
    Tek fark ADAY KUMESININ genisligidir:

        v1 : {theta*}                     x {mult=0.05}    ->  1 aday
        v2 : {0, theta*, theta*+-10}      x {mult=0.05}    ->  4 aday
        v3 : {0, theta*+-10/20/30}        x {0.05, 0.08}   -> 16 aday (en iyi 10)

    Aday kumeleri IC-ICE GECIKTIR (v1 ⊂ v2 ⊂ v3) ve v2/v3 adaylar arasindan
    GERCEK TSPLIB maliyetiyle secer. Bundan su GARANTI dogar:

        maliyet(v1) >= maliyet(v2) >= maliyet(v3_ham)

    Bu esitsizlik yapisaldir (deneysel degil): v1'in turu v2'nin aday
    listesinde AYNEN vardir, v2'ninki de v3'unkinde.

    !!! theta_deg'i DISARIDAN GECMEYIN !!!
    2026-07-24 dogrulamasi: runner'in `theta_star`'i (rotation_strip/hist/
    mean dedektorlerinin en iyisi, 5 derece adim + seyrek ornekleme) FARKLI
    bir kestiricidir. 26 ornekte olculdu: v1'in acisi 22'sinde v2'nin aday
    kumesinin DISINDA kaldi ve 2 ornekte (rbu737, lu980) v1 v2'yi GECTI --
    yani merdiven bozuldu, "v2 = iyilestirilmis v1" iddiasi gecersizlesti.
    Bu yuzden varsayilan (theta_deg=None) v2/v3 ile AYNI vekili kullanir.
    `theta_deg` yalnizca aci-duyarliligi deneyleri icin vardir; o durumda
    sonuc artik ailenin v1 basamagi DEGILDIR ve oyle raporlanmamalidir."""
    # 2026-08-01: v1 ARTIK SERPANTIN CEKIRDEGIDIR (V1_CONFIG). Onceden ic
    # tarama {1,2} arasindan secip cogu ornekte 1 BANT (= duz greedy-edge)
    # donuyordu; olculdu, 7 kumenin 5'inde uretilen tur greedy_edge ile
    # BIT-AYNIYDI. O halde v1 "strip ailesinin taban basamagi" olmaktan
    # cikmisti. Artik bant sayisi ve bant-ici k SABITTIR: yontem her zaman
    # gercek bir serpantin uretir, gorsellestirme sekmesi de onu cizer.
    # `bands` KESIF icindir (gorsellestirme sekmesindeki bant kaydiricisi):
    # verilirse sonuc ARTIK v1 DEGILDIR ve cagiran bunu boyle etiketlemelidir.
    _th0, _kv, _band, _ax = V1_CONFIG
    if bands is not None:
        _band = max(1, int(bands))
    # 2026-08-02: aci cozumu ARTIK _cfg_pool ile BIREBIR AYNI kuraldir --
    # V1_CONFIG'in aci alani None ise dedektor (theta_proxy), sayi ise mutlak
    # derece. Ayni kural olmasi SART: v2 havuzunun ILK adayi V1_CONFIG'dir ve
    # merdiven garantisi (maliyet(v1) >= maliyet(v2)) yalniz v1'in turu
    # havuzda AYNEN varsa yapisaldir. Iki yer farkli cozerse v1 havuzun
    # disina duser ve garanti sessizce kaybolur.
    _raw = theta_deg if theta_deg is not None else _th0
    if _raw is None:
        _raw = theta_proxy(xs, ys)
    th = (float(_raw) + 90.0) % 180.0 - 90.0
    rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    px, py = (rx, ry) if _ax else (ry, rx)
    t = _band_hybrid_tour(px, py, _greedy_band_order_k(_kv), mult,
                          dks=(0,), axes=(True,), k0=_band)
    return t, th


def snake_v2_tour(xs, ys, mult=0.05, cost_fn=None):
    """snake_v2: kok-neden teshisi (results/SNAKE_TESHIS.md) sonrasi yeni
    serit-bant hibrit insasi. Iki duzeltme:

      1) mult=0.05 (eski 0.15): `_aspect_k0` formulu buyuk n'de bantlari
         asiri inceltiyordu (rl5915'te 12 bant, her biri ~740 nokta) ve
         bant ici greedy-edge kisa goruslu kaliyordu. 2-3 makro-bant +
         dogru aci ham gap'i greedy_edge duzeyine indiriyor
         (fl3795: 12.96 vs 12.87; fnl4461/bgb4355: greedy_edge'i GECIYOR).
         Kucuk n'de ic dk/eksen taramasi zaten k=1..2 arasindan en iyisini
         sectigi icin gerileme yok (ali535: 19.51 aynen korunuyor).
      2) theta adaylari {0, strip_oracle, strip_oracle+-10}: E1 gosterdi ki
         rotasyonlu bulutlarda aci secimi kritik (rl5915: 18.21 -> 13.06);
         strip-oracle ucuz ve guvenilir bir vekil.

    Her adayda `_band_hybrid_tour`'un kendi dk/eksen taramasi + proxy secimi
    aynen korunur. Adaylar ARASI secim: `cost_fn` verilirse (runner
    `inst.tour_cost` geciyor) GERCEK maliyetle yapilir -- boylece theta=0
    adayi her zaman yaristigindan "asla theta=0'dan kotu olamaz" garantisi
    (EUC_2D yuvarlamasi proxy'yi saptirdiginda bile, ali535 vakasi) birebir
    korunur; yoksa raw-length proxy'sine duser.
    """
    pool = snake_v2_pool(xs, ys, mult=mult, cost_fn=cost_fn)
    return pool[0][1] if pool else None


def theta_of_label(label):
    """`snake_v2_pool` etiketinden ("th=<aci>/mult=<m>") aciyi cikarir."""
    try:
        return float(label.split("/", 1)[0].split("=", 1)[1])
    except Exception:
        return 0.0


#: greedy_snake_v2'nin IKINCI havuz ekseni: bant-ici greedy-edge aday
#: genisligi (2026-07-31, kullanici karari, olculdu).
#:
#: NEDEN {4, 8, 16} ve neden 8 LISTEDE KALIYOR: 8 eski TEK degerdi; listede
#: tutulunca yeni havuz eskisinin UST KUMESI olur ve secim gercek maliyetle
#: yapildigi icin
#:     maliyet(yeni v2) <= maliyet(eski v2)
#: YAPISAL bir esitsizlik haline gelir -- olculdu, 23 kumenin HICBIRINDE
#: kotulesme yok (18'inde iyilesme). {4,16} ikilisi biraz daha ucuz ama 4
#: kumede kotulesiyordu; kullanici garantili olani sectii.
#:
#: OLCULDU (23 kume, BKS'ye ort. gap, greedy_snake_v2 satiri):
#:     k=8 (eski)      16.31%
#:     {4,16}          15.22%   (18 iyi / 4 kotu)
#:     {4,8,16}        15.04%   (18 iyi / 0 kotu)   <- secilen
#: Bedel: havuz 4 -> 12 aday, insa ~3x (k=4 0.90x, k=8 1.00x, k=16 1.34x).
V2_KNN_VALS = (4, 8, 16)


def kaplama_olcusu(xs, ys, ornek=400, tohum=20260801):
    """Ornegin DUZENLILIK olcusu -- bkz. KAPLAMA_ESIK notu.
    Deterministiktir (sabit tohum): ayni ornek her zaman ayni degeri verir,
    yani yapilandirma secimi kosumdan kosuma DEGISMEZ."""
    n = len(xs)
    if n < 4:
        return 1.0
    w = (max(xs) - min(xs)) or 1.0
    h = (max(ys) - min(ys)) or 1.0
    rng = random.Random(tohum)
    idx = rng.sample(range(n), min(n, ornek))
    nn = []
    for i in idx:
        best = None
        for j in rng.sample(range(n), min(n, 60)):
            if j == i:
                continue
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if best is None or d < best:
                best = d
        if best:
            nn.append(best)
    if not nn:
        return 1.0
    nn.sort()
    med = nn[len(nn) // 2]
    return (med * med * n) / (w * h)


def v2_configs_for(xs, ys):
    """ADAPTIF secim: ornegin kaplama olcusune gore 2 yapilandirma."""
    return (V2_CONFIGS_KUMELI if kaplama_olcusu(xs, ys) < KAPLAMA_ESIK
            else V2_CONFIGS_DUZENLI)


def _cfg_pool(xs, ys, cfgs, cost_fn=None, mult=0.05, order_fn=None):
    """ACIK YAPILANDIRMA LISTESINDEN havuz kurar -- her yapilandirma icin
    TAM BIR tur (ic tarama YOK, atilan aday YOK).

    `cfgs`: ((aci, k, bant, x_ekseni_mi), ...)
        aci None      -> dedektor acisi theta_proxy(xs, ys)
        aci "+d"/"-d" -> dedektor acisina gore kaydirma (derece)
        aci sayi      -> mutlak derece
    `order_fn` verilirse k YOK SAYILIR (Far Snake boyle cagirir).

    Kurulan tur sayisi = len(cfgs). Bu, kiyas adaletinin OLCULEBILIR
    tanimidir: rakip kac tur kuruyorsa biz de o kadar kuruyoruz."""
    so = None
    pool = []
    for th, kv, band, ax in cfgs:
        if th is None or isinstance(th, str):
            if so is None:
                so = theta_proxy(xs, ys)
            d = 0.0 if th is None else float(th)
            ang = ((so + d + 90.0) % 180.0) - 90.0
        else:
            ang = float(th)
        rx, ry = ((xs, ys) if abs(ang) < 1e-9
                  else E.rotate_coords(xs, ys, ang))
        px, py = (rx, ry) if ax else (ry, rx)
        _of = order_fn if order_fn is not None else _greedy_band_order_k(kv)
        t = _band_hybrid_tour(px, py, _of, mult, dks=(0,), axes=(True,),
                              k0=band)
        sc = cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)
        lab = f"th={ang:g}/k={kv}/b={band}{'x' if ax else 'y'}"
        pool.append((lab, t, sc))
    return _finish_pool(pool)


def _finish_pool(pool, cap=None):
    """Havuz sozlesmesinin TEK uygulama yeri: gercek maliyete gore artan
    siralar, AYNI turu iki kez saymamak icin tur hash'iyle tekillestirir ve
    `cap` verilmisse en iyi `cap` adayi tutar.

    Kararli siralama: esit skorlu adaylarda uretim sirasi korunur, boylece
    "kazanan varyant" etiketi kosumdan kosuma degismez."""
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if cap is not None and len(out) >= cap:
            break
    return out


def snake_v2_pool(xs, ys, mult=0.05, cost_fn=None,
                  order_fn=None, dks=(-1, 0, 1), thetas=None,
                  extra_thetas=(), knn_vals=V2_KNN_VALS,
                  flat=True, cap=None, configs=None):
    """snake_v2'nin ADAY HAVUZU: `snake_v2_tour`'un ic dongusunun aynisi, ama
    yalniz kazanani degil TUM adaylari dondurur -- [(etiket, tur, skor), ...],
    skora gore ARTAN sirali (pool[0] = en iyi, pool[-1] = en kotu).

    ==================== ORTAK GOVDE (2026-07-27) =========================
    `order_fn` / `dks` / `mult` parametreleri eklendi. Artik snake_greedy_band
    ve snake_farthest_band da TAM OLARAK BU GOVDEYI cagirir; aralarindaki tek
    fark su uclu:

        greedy_snake_v2         : (_greedy_band_order,   mult=0.05, dks=(-1,0,1))
        snake_greedy_band    : (_greedy_band_order,   mult=0.15, dks=(-1,0,1))
        snake_farthest_band  : (_farthest_band_order, mult=_farthest_mult(n),
                                dks=(-1,0,1) veya (0,) buyuk n'de)

    ACI MAKINESI UCUNDE DE BIREBIR AYNIdir: {0, θ★, θ★±10}, θ★ =
    `_strip_oracle_theta`, secim GERCEK maliyetle (cost_fn).

    NEDEN (kullanici tespiti 2026-07-27): hibritler daha once HIC dondurulmuyor,
    aci taramasini runner yapiyordu ve o da FARKLI bir kestirici kullaniyordu
    ({0, rotation_strip dedektorunun acisi}, hem de yalnizca o dedektor satiri
    kosulduysa). Sonuc: tabloda hibritlerin acisi cogu kosumda 0 gorunuyordu ve
    "ayni serpantin cekirdegi" iddiasi aslinda dogru degildi -- v-ailesi bir
    aci vekilini, hibritler baskasini kullaniyordu.

    OLCULEN KAZANC (14 kume, tek degisken aci makinesi):
        snake_greedy_band    ort -1.586%, 9 kazanc / 0 kayip (en iyi -4.74%)
        snake_farthest_band  ort -1.596%, 9 kazanc / 0 kayip (en iyi -15.41%)
    KAYIP SAYISININ 0 OLMASI YAPISALDIR: θ=0 aday kumesinde her zaman vardir ve
    secim gercek maliyetle yapilir -> yeni sonuc eski (θ=0) sonuctan kotu
    OLAMAZ. Bedeli tek-insaya gore ~3.6-3.8x; runner zaten {0, θ_strip} iki
    insa yaptigindan tam benchmark'ta artis ~2x.

    `thetas`: acikca bir aci listesi verilirse tarama O kume uzerinde yapilir
    (tanilama betikleri kendi acilarini gecirir; thetas=(0.0,) eski davranis).

    Ayrilma sebebi (2026-07-25, kullanici talebi): v2 tek tur uretmiyor, 4
    aday uretip en iyisini seciyor. Raporda yalniz en iyiyi gostermek havuzun
    ne kadar ise yaradigini gizliyordu; en kotu aday da yazilinca "aci secimi
    gercekten ne kazandiriyor?" sorusu dogrudan okunur hale geliyor (en iyi
    ile en kotu arasindaki fark = havuzun katkisinin ust siniri).

    `snake_v2_tour` artik bu fonksiyonun ince bir sarmalayicisidir; boylece
    havuz uretimi TEK yerde tanimli kalir ve rapor ile kosum ayrisamaz."""
    _caller_order = order_fn is not None
    if order_fn is None:
        order_fn = _greedy_band_order
    if thetas is None:
        so = theta_proxy(xs, ys)
        cands = {0.0, so}
        for d in (-10.0, 10.0):
            cands.add(((so + d + 90.0) % 180.0) - 90.0)
        thetas = sorted(cands)
    if extra_thetas:
        # v2 kumesinin UST KUMESI: aday eklemek maliyeti asla artiramaz
        # (secim gercek maliyetle, mevcut adaylarin hepsi yarisiyor).
        _t = list(thetas)
        for e_ in extra_thetas:
            e_ = ((float(e_) + 90.0) % 180.0) - 90.0
            if all(abs(e_ - u) > 1e-6 for u in _t):
                _t.append(e_)
        thetas = sorted(_t)
    # ---- IKINCI EKSEN: bant-ici aday genisligi k (2026-07-31, olculdu) ----
    # knn_vals=None -> eski davranis (tek k, `order_fn`in kendi genisligi):
    # BIT-AYNI. Liste verilirse havuz aci x k caprazi olur.
    # `order_fn` cagiran tarafindan ACIKCA verilmisse k ekseni DEVRE DISI
    # kalir -- o cagiran kendi bant-ici kurucusunu dayatmistir (Far Snake
    # boyle cagirir ve farthest-insertion'in zaten k'si YOKTUR).
    # ---- 2 ADAYLI ADAPTIF-HIBRIT TASARIM (2026-08-01) --------------------
    # Varsayilan yol: V2_CONFIGS'teki 2 yapilandirma, ic tarama YOK.
    # Eski genis-havuz yolu `configs=None` verilerek hala erisilebilir
    # (olcum betikleri ve tarihsel kiyaslar icin).
    if configs is not False and not _caller_order:
        return _cfg_pool(xs, ys, configs or v2_configs_for(xs, ys),
                         cost_fn=cost_fn, mult=mult)
    _kv = (None,) if (_caller_order or not knn_vals) else tuple(knn_vals)
    pool = []
    for th in thetas:
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for kv in _kv:
            _of = order_fn if kv is None else _greedy_band_order_k(kv)
            acc = [] if flat else None
            t = _band_hybrid_tour(rx, ry, _of, mult, dks=dks, collect=acc)
            lab = f"th={th:g}/mult={mult:g}"
            if kv is not None:
                lab += f"/k={kv}"
            if acc is None:
                score = (cost_fn(t) if cost_fn is not None
                         else _raw_length(t, xs, ys))
                pool.append((lab, t, score))
            else:
                # DUZ HAVUZ: ic taramanin TUM adaylari (bkz. POOL_FLAT_CAP).
                # Etiket bant sayisini ve ekseni de tasir ("b=2y"), boylece
                # kazanan satirdan "kac bant, hangi eksen" okunabilir.
                for bk, bax, bt in acc:
                    sc = (cost_fn(bt) if cost_fn is not None
                          else _raw_length(bt, xs, ys))
                    pool.append((f"{lab}/b={bk}{'x' if bax else 'y'}", bt, sc))
    return _finish_pool(pool, cap)


def snake_v2_variant_pool(xs, ys, mults=(0.05, 0.08), cost_fn=None, k=6,
                          deltas=(-10.0, 10.0), theta_mode="auto",
                          inner_dks=(-1, 0, 1), inner_axes=(True, False)):
    """snake_v2'nin theta x bant-konfigurasyonu aday HAVUZU (gpu_snake_v2 /
    gpu_hybrid_v2 ve lite-merdiven varyantlari icin). SNAKE_V2_ONARIMLI.md'nin
    ana bulgusu "ham siralama != onarim havzasi siralamasi" oldugundan, tek
    en-iyi ham turu secmek yerine en iyi-k ham varyant uretilip GPU paralel
    onarima BIRLIKTE beslenir; havuz, havza belirsizligini satin alir.

    theta_mode (SNAKE_V2_LITE.md merdiveni):
      "auto"           -> {0, oracle} U {oracle+d : d in deltas}
      "oracle_only"    -> {oracle}            (lite1: tek insa)
      "zero_and_oracle"-> {0, oracle}         (lite2: 2 insa)
      "narrow"         -> {0, oracle, oracle+-10}  (mid: 4 insa)
    inner_dks/inner_axes: _band_hybrid_tour'un ic taramasi; lite1
    (0,)/(True,) ile ic taramayi tamamen kapatir.

    Donus: [(etiket, tur, ham_maliyet), ...] -- gercek maliyete (cost_fn;
    yoksa raw-length proxy) gore artan sirali, yinelenen turlar ayiklanmis,
    en cok k eleman."""
    so = theta_proxy(xs, ys)
    if theta_mode == "oracle_only":
        thetas = {so}
    elif theta_mode == "zero_and_oracle":
        thetas = {0.0, so}
    elif theta_mode == "narrow":
        thetas = {0.0, so}
        for d in (-10.0, 10.0):
            thetas.add(((so + d + 90.0) % 180.0) - 90.0)
    else:  # "auto"
        thetas = {0.0, so}
        for d in deltas:
            thetas.add(((so + d + 90.0) % 180.0) - 90.0)
    pool = []
    for th in sorted(thetas):
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for m in mults:
            t = _band_hybrid_tour(rx, ry, _greedy_band_order, m,
                                  dks=inner_dks, axes=inner_axes)
            c = cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)
            pool.append((f"th={th:g}/mult={m:g}", t, c))
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if len(out) >= k:
            break
    return out


#: farthest-insertion bant-ici kurucu HIZLANDIRILMAMIS O(m^2)'dir; havuza
#: eklenen her theta varyanti bu maliyeti bir kez daha oder. Bu esigin
#: uzerinde farthest bacagi havuza ALINMAZ (ve cagirana bildirilir) --
#: sessizce yavaslamak yerine acikca kapsam disi kalir.
V3_FARTHEST_MAX_N = 30000


def snake_v3_hybrid_pool(xs, ys, cost_fn=None, k=10,
                         mults=(0.05, 0.08),
                         deltas=(-30.0, -20.0, -10.0, 10.0, 20.0, 30.0),
                         farthest_thetas=2, extra_tours=None,
                         farthest_max_n=V3_FARTHEST_MAX_N):
    """Greedy Snake v3 HIBRIT havuzu (2026-07-24, kullanici karari).

    Eski v3 (`snake_v2_variant_pool`) havuzu yalniz TEK bir bant-ici
    kurucudan (greedy-edge) uretiyordu: theta x mult ekseninde 16 aday, ama
    hepsi ayni yerel kuruculu oldugu icin turlar birbirine cok benziyordu.
    Olculen sonuc: etkin havuz genisligi 8.9/10, kroA100'de yalnizca 3
    benzersiz tur -- yani "10 varyant" nominal, gercekte degil. Bu, GE'nin
    jitter'li havuzunun (ge_pool_repair) v3'u gecmesinin dogrudan sebebi:
    onarim havzasi cesitliligi tohumlarin YAPISAL farkindan gelir, ayni
    kurucunun aci varyantlarindan degil.

    Hibrit havuz ayni theta eksenini KORUR ve uzerine bant-ici kurucu
    eksenini ekler:
      * greedy-band  : theta in {0, so, so+-10/20/30} x mult in mults
      * farthest-band: theta in {0, so} (ilk `farthest_thetas` aci) x
                       `_farthest_mult(n)` -- YAPISAL olarak farkli turlar
      * extra_tours  : cagiranin ZATEN hesapladigi kardes turlar
                       (or. snake_stratified_subsample) -- yeniden
                       hesaplanmaz, bedava cesitlilik

    Aday kumesi eski v3'un UST KUMESI oldugundan v1 ⊂ v2 ⊂ v3 merdiveni
    korunur (maliyet(v3) yalnizca dusebilir veya ayni kalir).

    Donus: [(etiket, tur, ham_maliyet), ...] gercek maliyete gore artan,
    tekillestirilmis, en cok k eleman."""
    n = len(xs)
    so = theta_proxy(xs, ys)
    # --- IKI TOHUMLU ACI KUMESI (2026-07-28) --------------------------------
    # v3 ailenin GENIS havuz uyesidir; aci tohumu olarak HER IKI kestiriciyi
    # birden alir (izgara acisi + strip-oracle). Sebep olculdu: vekil tek
    # basina izgara acisina cevrildiginde v3 16 kumede ort. %13.47 -> %14.69
    # geriledi ve pma343'te %6.36 -> %22.37 (+16 puan) koptu -- cunku o
    # kumede kazanan aci strip-oracle'in cevresindeydi (theta*=60) ve izgara
    # tohumunun {0, +-10/20/30} penceresine hic girmiyordu.
    #
    # UST KUME OLDUGU ICIN GERILEME YAPISAL OLARAK IMKANSIZ: her iki eski
    # aday kumesi de bu kumenin ICINDE ve secim GERCEK maliyetle yapiliyor.
    # (Ayni gerekce bant hibritlerinin `extra_thetas` parametresinde de var.)
    # Bedeli: iki tohum FARKLI oldugunda greedy ekseni ~2x aday uretir; ayni
    # olduklarinda (izgara guveni dusup strip-oracle'a duşulen kumeler dahil)
    # maliyet DEGISMEZ.
    #
    # v1 ⊂ v2 ⊂ v3 KORUNUR: v2'nin kumesi {0, so, so+-10}, bu kumenin alt
    # kumesidir; v1'in tek acisi (so) de icindedir.
    seeds = [so]
    _alt = _strip_oracle_theta(xs, ys) if ANGLE_PROXY == "grid" else None
    if _alt is not None and abs(((_alt - so + 90.0) % 180.0) - 90.0) > 1e-9:
        seeds.append(_alt)
    thetas = [0.0]
    for _s in seeds:
        for d in (0.0,) + tuple(deltas):
            th = ((_s + d + 90.0) % 180.0) - 90.0
            if all(abs(th - t) > 1e-9 for t in thetas):
                thetas.append(th)
    thetas.sort()
    pool = []
    # --- eksen 1: greedy-band (eski v3 ile BIREBIR ayni uretim) ---
    for th in thetas:
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for m in mults:
            t = _band_hybrid_tour(rx, ry, _greedy_band_order, m)
            pool.append((f"ge/th={th:g}/mult={m:g}", t,
                         cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    # --- eksen 2: farthest-band (yapisal olarak farkli bant-ici kurucu) ---
    if n <= farthest_max_n and farthest_thetas > 0:
        fm = _farthest_mult(n)
        fdks = (-1, 0, 1) if n <= 5000 else (0,)
        # Farthest ekseni de AYNI tohum kumesini gormek ZORUNDA. Onceden
        # yalniz `so` kullaniyordu; vekil izgara acisina cevrilince pma343'te
        # theta=60'taki farthest adayi dusuyor ve v3 %6.36'dan %22.37'ye
        # kopuyordu -- yani "ust kume oldugu icin gerileyemez" garantisi
        # SADECE greedy ekseninde saglaniyordu. Kapasite tohum sayisi kadar
        # buyutulur; tek tohumda eski davranis birebir korunur.
        _fseeds = [0.0]
        for _s in seeds:
            if all(abs(_s - t) > 1e-9 for t in _fseeds):
                _fseeds.append(_s)
        for th in _fseeds[:farthest_thetas + len(seeds) - 1]:
            rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
            t = _band_hybrid_tour(rx, ry, _farthest_band_order, fm, dks=fdks)
            pool.append((f"fi/th={th:g}/mult={fm:g}", t,
                         cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    # --- eksen 3: cagiranin hazir turlari (bedava cesitlilik) ---
    for lab, t in (extra_tours or []):
        if t is None or len(t) != n:
            continue
        pool.append((f"ext/{lab}", list(t),
                     cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if len(out) >= k:
            break
    return out


#: Far Snake v3'un IKINCI mult'u, ailenin taban mult'unun katsayisi olarak.
#: Greedy Snake v3 (0.05, 0.08) kullanir -> oran 1.6. Far Snake'in tabani n'e
#: bagli oldugu icin (`_farthest_mult`) ikinci mult SABIT bir sayi olamaz;
#: AYNI ORAN uygulanir ki "iki mult'lu ikinci eksen" iki ailede de ayni
#: GENISLIKTE olsun (13 aci x 2 mult), sadece taban degeri kaysin.
_FAR_MULT_RATIO = 1.6


def far_snake_v3_hybrid_pool(xs, ys, cost_fn=None, k=10, mults=None,
                             deltas=(-30.0, -20.0, -10.0, 10.0, 20.0, 30.0),
                             greedy_thetas=2, extra_tours=None,
                             starts=None, start_thetas=2):
    """Far Snake v3 HIBRIT havuzu -- `snake_v3_hybrid_pool`'un yapisal
    AYNASI, iki bant-ici kurucu ekseni YER DEGISTIRMIS halde, USTUNE ailenin
    kendi cesitlilik ekseni eklenmis:

        greedy_snake_v3 : eksen1  = greedy-band (13 aci x 2 mult)   <- ANA
                          eksen2  = farthest-band (2-3 aci)          <- YAN
        far_snake_v3    : eksen1  = farthest-band (13 aci x 2 mult)  <- ANA
                          eksen1b = START ekseni (2 aci x 3 start)   <- KENDI
                          eksen2  = greedy-band (2-3 aci)            <- YAN

    Aci makinesi (iki tohumlu kume: izgara acisi + strip-oracle, {0} U
    {tohum + 0/±10/±20/±30}), tekillestirme, gercek-maliyetle siralama ve
    k=10 kapagi IKI FONKSIYONDA DA AYNI koddur.

    EKSEN 1b NEDEN VAR (2026-07-28, olculdu): theta ekseni greedy-edge'in
    cesitlilik silahidir; farthest-insertion DONMEYE DUYARSIZ oldugu icin
    onda calismaz (bant sayisi 1'e dustugunde SIFIR cesitlilik uretir --
    ali535'te 4 aday, 1 benzersiz kenar kumesi). Ailenin kendi ekseni
    bant-ici BASLANGIC DUGUMUDUR; bkz. `FAR_STARTS` notu ve
    `far_snake_v2_pool` docstring'i.

    MERDIVEN KORUNUR: far_snake_v2 = {θ★} x `FAR_STARTS` x mults[0]
    kumesidir; bunun start=0 uyesi eksen 1'de (th=θ★, mult=mults[0]), geri
    kalani eksen 1b'de AYNEN uretilir ve ic tarama `_far_dks(n)` ile
    AYNIdir -> far v1 ⊂ far v2 ⊂ far v3, dolayisiyla
    maliyet(far v2) >= maliyet(far v3_ham) YAPISALDIR.

    `extra_tours` (eksen3) parametresi ARAYUZDE VARDIR ama runner onu BOS
    gecer (`_far_v3_extra_tours`). Simetrik doldurmak (greedy_snake_v2'nin
    turunu vermek) denendi ve olculdu: 8/8 kumede kazanc 0.000% -- eksen 2
    (ic greedy-band bacagi) onu zaten kapsiyor. Bos birakmak ayrica
    greedy_snake_v2'yi tam yetkili rakip satir olarak korur (aksi halde
    maliyet(far_snake_v3) <= greedy_snake_v2 YAPISAL olur ve o satirin
    Sira/Kazanma% havuzundan cikarilmasi gerekirdi). Gerekce ve olcumler:
    runner._far_v3_extra_tours.

    KARMASIKLIK UYARISI (durustluk): eksen1 burada O(m^2) kurucudur, yani
    bu havuz greedy_snake_v3'ten belirgin olarak PAHALIdir. `_farthest_mult`
    ve `_far_dks` bunu sinirlar, geri kalanini runner'in insa butcesi
    (`E.check_construction_deadline`) keser -- butce asilirsa satir
    "unavailable" olarak DUSER, sessizce kirpilmis bir havuzla RAPORLANMAZ."""
    n = len(xs)
    fm = _farthest_mult(n)
    if mults is None:
        mults = (fm, round(fm * _FAR_MULT_RATIO, 4))
    fdks = _far_dks(n)
    so = theta_proxy(xs, ys)
    # Iki tohumlu aci kumesi -- gerekce ve olcumler snake_v3_hybrid_pool'da.
    seeds = [so]
    _alt = _strip_oracle_theta(xs, ys) if ANGLE_PROXY == "grid" else None
    if _alt is not None and abs(((_alt - so + 90.0) % 180.0) - 90.0) > 1e-9:
        seeds.append(_alt)
    thetas = [0.0]
    for _s in seeds:
        for d in (0.0,) + tuple(deltas):
            th = ((_s + d + 90.0) % 180.0) - 90.0
            if all(abs(th - t) > 1e-9 for t in thetas):
                thetas.append(th)
    thetas.sort()
    import functools

    def _fi(f):
        return (_farthest_band_order if not f else
                functools.partial(_farthest_band_order, start_frac=f))

    pool = []
    # --- eksen 1: farthest-band, theta x mult (yapinin ekseni) ---
    for th in thetas:
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for m in mults:
            t = _band_hybrid_tour(rx, ry, _fi(0.0), m, dks=fdks)
            pool.append((f"fi/th={th:g}/mult={m:g}", t,
                         cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    # --- eksen 1b: START ekseni -- ailenin KENDI cesitlilik silahi ---
    # v2 ile MERDIVEN SARTI: far_snake_v2 = {θ★} x FAR_STARTS x fm oldugundan
    # o kume burada AYNEN uretilir (θ★ satiri) -> far v2 ⊂ far v3, dolayisiyla
    # maliyet(far v2) >= maliyet(far v3_ham) YAPISAL kalir.
    # Ikinci aci olarak θ=0 eklenir: start ekseni acidan bagimsiz calisir, iki
    # acida da denemek ucuzdur (aday basina 1 insa) ve havuzun yapisal
    # cesitliligini olcumde en cok artiran ekti (bkz. FAR_STARTS notu).
    if starts is None:
        starts = FAR_STARTS
    _sth = [so]
    if all(abs(0.0 - t) > 1e-9 for t in _sth):
        _sth.append(0.0)
    for th in _sth[:start_thetas]:
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for f in starts:
            if not f:
                continue          # start=0 zaten eksen 1'de uretildi
            t = _band_hybrid_tour(rx, ry, _fi(f), mults[0], dks=fdks)
            pool.append((f"fi/th={th:g}/s={f:g}", t,
                         cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    # --- eksen 2: greedy-band bacagi (YAPISAL olarak farkli bant-ici kurucu).
    #     greedy_snake_v3'un farthest bacaginin AYNADAKI karsiligi: ayni aci
    #     sayisi (tohum sayisina gore buyur), karsi ailenin KENDI mult'u. ---
    if greedy_thetas > 0:
        _gseeds = [0.0]
        for _s in seeds:
            if all(abs(_s - t) > 1e-9 for t in _gseeds):
                _gseeds.append(_s)
        for th in _gseeds[:greedy_thetas + len(seeds) - 1]:
            rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
            t = _band_hybrid_tour(rx, ry, _greedy_band_order, 0.05)
            pool.append((f"ge/th={th:g}/mult=0.05", t,
                         cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    # --- eksen 3: cagiranin hazir turlari (bedava cesitlilik) ---
    for lab, t in (extra_tours or []):
        if t is None or len(t) != n:
            continue
        pool.append((f"ext/{lab}", list(t),
                     cost_fn(t) if cost_fn else _raw_length(t, xs, ys)))
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# GREEDY SNAKE v4 -- ACI INCE TARAMASI (2026-07-31, kullanici karari)
# ---------------------------------------------------------------------------
# v4'un ILK tasarimi (2026-07-30) satir yuksekligini cesitlendiriyordu;
# olculdu ve TUTMADI (ham +%10.81, onarimli +%2.12, 0/8 kazanc -- ayrinti
# IZGARA_ACISI_VE_SERIT_SAYISI_AKADEMIK_NOT.md §14). Kullanici karariyla v4
# yuvasi ACI eksenine devredildi; eski havuz uretici `snake_v4_height_pool`
# olarak ASAGIDA DURUYOR ama artik hicbir benchmark satirina bagli DEGILDIR
# (ayni sozlesme: snake_greedy_band_tour / snake_stratified_subsample_tour).
#
# YENI TASARIM (kullanici, birebir): "-90, 0 bunlardan iyi olan hangisi ise
# onun yakinlarinda 5 derece saga sola oynasin ... 10lu varyantta ise daha
# ince olsun 2'ser derecelik adimlar ile arasin."
#
# IKI ASAMALI: once IKI SONDA acisi (-90 ve 0) gercek maliyetle yarisir,
# sonra KAZANANIN etrafi taranir:
#
#     4'lu havuz : {-90, 0} + {w-5, w+5}                        -> 4 aday
#     10'lu havuz: {-90, 0} + {w+-2, w+-4, w+-6, w+-8}          -> 10 aday
#
# (w = sonda kazanani.) Aci normalize EDILMEZ: kullanici "-90 gelirse saga
# sola 5" dedi, satirda -95/-85 gorunmeli. Serit yonu 180 derece periyodik
# oldugu icin -95 ≡ 85'tir ama gezinme yonu terslenir, yani ikisi ayni tur
# DEGILDIR; yazilan deger neyse o kurulur.
#
# !! MERDIVEN UYARISI !! 4'lu havuz 10'lunun ALT KUMESI DEGILDIR (w+-5, 2'ser
# adimli izgarada YOKTUR). Yani maliyet(v4_4) >= maliyet(v4) esitsizligi bu
# tasarimda YAPISAL DEGILDIR ve v4_4 bazi ornekte v4'u GECEBILIR. Bu bilincli
# bir kabuldur (kullanici adimlari acikca 5 ve 2 olarak belirledi); nesting
# istenirse tek degisiklik V4_COARSE_STEP'i 2'nin katina cekmektir.
#
# ORTAK TASARIM (iki tasarimda da ayni gerekce):
#  1. `axes=(True,)` -- bant HER ZAMAN dondurulmus x ekseninde acilir.
#     axes=(True,False) birakilsaydi -90 ve 0 sondalari AYNI turu verirdi
#     (-90'da y ekseninde bantlamak, 0'da x ekseninde bantlamakla AYNIdir)
#     ve deneyin tamami anlamsizlasirdi.
#  2. `dks=(-1,0,1)` -- bant SAYISI artik degisken degil, sikici bir
#     parametre; her aci adayinin kendi en iyi bant sayisini bulmasina izin
#     verilir (v1/v2/v3 ile ayni ic tarama). Boylece aciler adil kiyaslanir.
#  3. Havuza KARDES TUR ALINMAZ (`extra_tours` yok): baska bir kaynaktan
#     gelen tur havuzu kazanirsa satir artik "aci ince taramasi" olmaz.
#
# --- ESKI TASARIMIN NOTU (satir yuksekligi; artik bagli degil) --------------
# v2 ve v3'un havuz ekseni ACIdir (theta x mult). Eski v4 havuzunun ekseni
# BASKAYDI: aci SABIT (-90 derece, kullanici hipotezi -- bkz.
# IZGARA_ACISI_..._NOT.md §13) ve degisen tek sey BANT YUKSEKLIGIYDI.
#
# FIKIR (kullanici, birebir): "varyantlar rastlantisal olarak da olsa dogru
# band ayrimlarini yakalayip o buyuk bosluklarla ayrilmis parcalarin icinde
# greedy algoritmasini kosturabilirse cok iyi sonuclar elde edebiliriz."
#
# Mekanizma: `_band_indices` ESIT GENISLIKTE kova acar, yani bant sinirlarinin
# YERI yalniz bant SAYISINA (k) baglidir. k degistiginde butun sinirlar birden
# kayar. Farkli k'larda kurulan turlardan biri, sinirlarini levhadaki gercek
# bosluklara denk getirirse o varyantin bant-ici greedy-edge'i "dogru
# parcalarin" icinde calisir. Havuz bu SANSI 10 kez dener ve secimi GERCEK
# TSPLIB maliyetiyle yapar.
#
# NEDEN BU EKSEN BOSTA DEGIL -- olculmus bir bosluk: `far_start_fracs`
# notunda kayitli oldugu gibi, bant sayisi 1'e dustugunde ACI EKSENI SIFIR
# cesitlilik uretir (ali535: 4 aday, 1 benzersiz kenar kumesi, yayilim %0.00).
# Yukseklik ekseni tam da orada calisir: k=1'i k=2,3,4,...'e acar.
#
# TASARIM KARARLARI (uc tanesi de bilerek ve olcume aciktir):
#
#  1. `dks=(0,)` ve `axes=(True,)` -- her varyant TEK bir (k, eksen) insasi.
#     Sebep: `_band_hybrid_tour`'un kendi ic taramasi (dks=(-1,0,1),
#     axes=(True,False)) 6 adayi PROXY maliyetle eleyip TEK tur dondururdu;
#     havuz ise adaylari GERCEK maliyetle karsilastirir. Ic taramayi acik
#     birakmak (a) 10 varyantin cogunu ayni tura cokerdir (k0+1 ile k0'in
#     komsu carpanlari cakisir), (b) secimi proxy'ye devrederdi.
#     axes=(True,False) ayrica "aci sabit -90" iddiasini SESSIZCE bozardi:
#     -90'da y ekseninde bantlamak, 0 derecede x ekseninde bantlamakla AYNI
#     seydir -- yani havuz gizlice theta=0 insasini icerirdi.
#  2. Havuza KARDES TUR ALINMAZ (`extra_tours` yok). v3 kardes insalari
#     bedava cesitlilik olarak alir; burada bir baska acidaki tur havuzu
#     kazanirsa satir artik "-90'da yukseklik taramasi" olmaz.
#  3. TEK BANT (k=1) HAVUZA GIRMEZ. Bu, olcumle bulunan bir tuzaktir: ilk
#     tasarimda carpan merdiveni k=1'e kadar iniyordu ve 10 kumenin 8'inde
#     havuzu KAZANAN aday k=1 oldu. k=1 demek hic bant olmamasi, yani bant-ici
#     kurucunun butun ornek uzerinde tek basina kosmasi demektir -- bu zaten
#     `greedy_edge` RAKIP SATIRIDIR. Havuzda birakilsaydi v4 adi altinda
#     aslinda greedy_edge raporlanirdi ve "satir yuksekligi ise yariyor"
#     sonucu tamamen sahte cikardi (ayni kok neden snake_best_insertion_band'i
#     kaldirtmisti; bkz. runner.REMOVED_METHODS). Bu yuzden V4_MIN_BANDS = 2.
#
# MERDIVEN: `V4_HEIGHT_FACTORS_4`, `V4_HEIGHT_FACTORS`'in ALT KUMESIDIR, yani
#     maliyet(v4_4) >= maliyet(v4)
# yapisaldir (v2 ⊂ v3 ile birebir ayni gerekce: secim gercek maliyetle).

#: SONDA acilari: once bu ikisi gercek maliyetle yarisir, ince tarama
#: KAZANANIN etrafinda yapilir. Ikisi de sabittir -- panelden aci degistirmek
#: v4'un ne oldugunu degistirmemeli, yoksa iki kosum arasinda ayni ada sahip
#: satirlar kiyaslanamaz olur.
V4_PROBE_ANGLES = (-90.0, 0.0)
#: 4'lu havuzun adimi (kullanici: "5 derece saga sola").
V4_COARSE_STEP = 5.0
#: 10'lu havuzun adimi (kullanici: "daha ince olsun, 2'ser derecelik").
V4_FINE_STEP = 2.0
#: Ailenin taban bant carpani (v1/v2 ile ayni). Aci havuzunda her aday BUNUNLA
#: kurulur (bant sayisi artik degisken degil); eski yukseklik havuzunda
#: merdivenin tabaniydi.
V4_BASE_MULT = 0.05
#: Havuzun NOMINAL genisligi -- her iki tasarimda da 10 (v3 ile esit genislik).
V4_POOL_WIDTH = 10

# --- ASAGIDAKILER YALNIZ ESKI (satir yuksekligi) TASARIMINA AITTIR ----------
# `snake_v4_height_pool` disinda kullanan yoktur; hicbir benchmark satiri
# bagli degildir (bkz. akademik not §14).
#: Eski tasariminin SABIT acisi.
V4_THETA_DEG = -90.0
#: EN AZ bant sayisi -- k=1 (bantsiz) havuza alinmazdi.
V4_MIN_BANDS = 2
#: Bant BASINA en az nokta; ust ucta bantlarin bosalmasini engeller. Esit
#: GENISLIKTE kova acildigi icin (bkz. `_band_indices`) k'yi n ile orantili
#: buyutmek bantlarin cogunu bosaltip birkacini asiri yukler -- n=47608'de
#: %220-580 gap veren tam da buydu.
V4_MIN_PTS_PER_BAND = 8
#: Merdivenin UST UCU, ailenin taban bant sayisinin (k_base) kati olarak.
#: Ust uc k_base ile olceklendigi icin bant sayisi sqrt(n) buyur -- sabit bant
#: BOYUTU hedeflemek n=47608'de %220-580 gap veriyordu (bkz. `_band_indices`).
V4_TOP_FACTOR = 8.0
#: 4-adayli DAR havuzun secimi: 10'luk merdivenin SIRALI KONUMLARI, iki uc
#: dahil esit log adimla. Bant sayilarindan YENIDEN HESAPLANMAZ; konumdan
#: secmek "alt kume" olmayi TANIMI GEREGI saglar, yani
#:     maliyet(v4_4) >= maliyet(v4)
#: yapisal kalir. (Ilk tasarim dar havuzu kendi carpan listesinden kuruyordu
#: ve yuvarlama yuzunden genis havuzda BULUNMAYAN bant sayilari uretiyordu --
#: k_base=1'de olculdu: dar havuzda k=5 vardi, genis havuzda yoktu.)
V4_POOL4_PICKS = (0, 3, 6, 9)


def _geom_ints(lo, hi, count):
    """[lo, hi] araliginda `count` adet ARTAN ve FARKLI tamsayi, geometrik
    adimla. Yuvarlama iki adimi ayni sayiya dusurursi bir sonraki tamsayiya
    kayilir; bu yuzden aralik dar oldugunda dizi ardisik tamsayilara dogru
    yumusakca bozunur (2,3,4,...) ve HICBIR ZAMAN tekrar uretmez."""
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    if count <= 1:
        return [lo]
    r = (hi / lo) ** (1.0 / (count - 1)) if hi > lo else 1.0
    out, prev = [], lo - 1
    for i in range(count):
        v = max(prev + 1, int(round(lo * (r ** i))))
        out.append(v)
        prev = v
    return out


def v4_band_counts(n, px, py, width=V4_POOL_WIDTH, base_mult=V4_BASE_MULT,
                   min_bands=V4_MIN_BANDS, min_pts=V4_MIN_PTS_PER_BAND,
                   top_factor=V4_TOP_FACTOR, picks=V4_POOL4_PICKS,
                   full_width=V4_POOL_WIDTH):
    """v4 havuzunun BANT SAYILARI -- farkli satir yuksekliklerinin listesi.

    Merdiven `min_bands`'ten (=2, bantsiz insa havuza girmez) baslar ve
    ailenin taban bant sayisinin `top_factor` katina kadar GEOMETRIK adimla
    cikar. Iki ucu da anlamli: alt uc en kaba ayrisim, ust uc ince satirlar.
    `min_pts` tavani (bant basina en az nokta) ust ucu kirpar -- bosalan bant
    uretmektense merdiven KISALIR ve havuz dar raporlanir.

    NEDEN k_base'in KATLARI DEGIL DE 2'DEN BASLAYAN BIR MERDIVEN: k_base
    kucuk n'de 1-3 arasindadir; carpan merdiveni oraya oturtuldugunda
    carpanlarin yarisi ayni tamsayiya yuvarlaniyor, kalan yer de +1'lerle
    (17, 18, 19 gibi) dolduruluyordu -- yani havuzun yarisi birbirinin
    kopyasi ya da bilgi tasimayan ince varyantlardi. Olculdu ve degistirildi.

    `width` tam merdivenden kucukse `picks` konumlari secilir (alt kume
    garantisi -- bkz. V4_POOL4_PICKS)."""
    k_base = _aspect_k0(n, px, py, base_mult)
    k_cap = max(min_bands, n // max(1, min_pts))
    k_top = min(k_cap, max(min_bands + 1,
                           int(round(k_base * float(top_factor)))))
    ks = [k for k in _geom_ints(min_bands, k_top, full_width) if k <= k_cap]
    if not ks:
        ks = [min_bands]
    if width >= len(ks):
        return ks
    sel = [ks[i] for i in picks if i < len(ks)]
    return sel[:width] if sel else ks[:width]


def v4_angle_ladder(width, winner, coarse_step=V4_COARSE_STEP,
                    fine_step=V4_FINE_STEP, probes=V4_PROBE_ANGLES):
    """Sonda kazanani `winner` etrafindaki INCE TARAMA acilarini dondurur.

    Adim genisligi havuz genisligine baglidir (kullanici karari):
        width <= 4  -> 5 derece  ->  [w-5, w+5]
        width  = 10 -> 2 derece  ->  [w-2, w+2, w-4, w+4, w-6, w+6, w-8, w+8]

    Sonda acilari (probes) HAVUZDA ZATEN VARDIR, bu yuzden geriye kalan
    genislik kadar aday uretilir. Simetrik cift halinde ilerlenir ki tarama
    kazananin iki yaninda DENGELI olsun; tek sayi kalirsa once negatif yon
    (kullanicinin "saga sola" sirasi) alinir."""
    need = max(0, int(width) - len(probes))
    step = coarse_step if int(width) <= 4 else fine_step
    out, i = [], 1
    while len(out) < need:
        for sgn in (-1.0, 1.0):
            if len(out) >= need:
                break
            out.append(float(winner) + sgn * i * step)
        i += 1
    return out


#: Bant-ici greedy-edge ADAY GENISLIKLERI -- v4'un IKINCI havuz ekseni
#: (2026-07-31, olculdu ve kullanici karariyla eklendi).
#:
#: NEDEN k BIR EKSEN, "n'e gore secilecek bir sabit" DEGIL:
#: 32 kumede 9 k degeri tarandi (theta=-90, mult=0.05 sabit). Kazanan k n ile
#: DUZENLI DEGISMIYOR -- n bandi basina en iyi ortalama sirasiyla 24 / 4 / 4 /
#: 16 / 24 cikti, yani zikzak. Bant basina 5-8 ornek var; bu yapi degil
#: GURULTU. Dolayisiyla "n su araliktaysa k su olsun" kurali veriye DAYANMAZ.
#:
#: Buna karsilik k, HAVUZ EKSENI olarak cok degerli (ayni 32 kume, en iyiye
#: gore ortalama fark):
#:     en iyi TEK k (=4)      %1.82
#:     {k=4, k=16} ikilisi    %0.86     <- yarisindan az
#:     {k=4, k=5, k=24}       %0.46
#: Yani dogru tasarim k'yi TAHMIN etmek degil, iki k'yi havuza koyup GERCEK
#: maliyetle sectirmektir -- ornege gore "akilli davranma" tam olarak budur
#: ve zaten havuz mimarisinin yaptigi seydir, yalnizca yanlis eksende
#: (yalniz aci) kullaniliyordu.
#:
#: Mevcut varsayilan k=8'in kotu bir secim oldugu da ayni taramada olculdu:
#: ortalama %2.62 ile 5. sirada ve 32 kumenin yalniz 2'sinde kazaniyor;
#: k=4 hem daha iyi (%1.82, 8/32) hem ~%10 daha HIZLI.
V4_KNN_VALS = (4, 16)


def snake_v4_angle_pool(xs, ys, cost_fn=None, k=V4_POOL_WIDTH, width=None,
                        base_mult=V4_BASE_MULT, order_fn=None,
                        probes=V4_PROBE_ANGLES, knn_vals=V4_KNN_VALS,
                        flat=True, configs=None):
    """Greedy Snake v4: ACI x BANT-ICI ADAY GENISLIGI capraz havuzu.

    Havuz iki BAGIMSIZ eksenin capraz carpimidir:
        aci ekseni : `probes`      -- varsayilan (-90, 0)
        k   ekseni : `knn_vals`    -- varsayilan (4, 16)
    yani varsayilan havuz 2 x 2 = 4 adaydir ve hepsi GERCEK maliyetle
    yarisir. Aci ARANMAZ (dedektor yok, ince tarama yok): iki sabit hipotez
    ile iki aday genisligi yarisir.

    ORDER_FN UYARISI: `order_fn` acikca verilirse k ekseni DEVRE DISI kalir
    (cagiran kendi bant-ici kurucusunu dayatmistir; Far Snake boyle cagirir).
    O durumda havuz yalniz aci ekseninde kurulur.

    ESKI ACI-INCE-TARAMA YOLU (v4_angle_ladder) KORUNDU: capraz carpim
    `width`i doldurmazsa kalan genislik, kazanan acinin etrafini tarayarak
    tamamlanir. Varsayilan yapilandirmada bu yol HIC calismaz (2x2 = 4 = w).

    Donus: [(etiket, tur, ham_maliyet), ...] -- gercek maliyete (cost_fn;
    yoksa raw-length proxy) gore artan sirali, yinelenen turlar ayiklanmis,
    en cok `width` eleman. Etiket "th=<aci>*/k=<k>" biciminde; sonda acilari
    yildizli, ince tarama acilari yildizsizdir. Boylece kazananin HANGI aci
    ve HANGI aday genisligi oldugu sonuc satirindan dogrudan okunur."""
    if configs is not False and order_fn is None:
        _cfgs = configs or V4_CONFIGS
        _pool = _cfg_pool(xs, ys, _cfgs, cost_fn=cost_fn, mult=base_mult)
        if _cfgs is V4_CONFIGS:
            # Tarama satirini etiketten AYIRT EDILEBILIR yap. Iki sebep:
            #   1) 100 adayli tani havuzu ile 2 adayli eski rekabet havuzu
            #      ayni etiket bicimini uretir (`th=-90/k=8/b=2y`) -- ek
            #      olmadan diskteki eski satir yeni sanilir ve kiyas
            #      sessizce bozulur (bkz. runner._STALE_SIGNATURES).
            #   2) Ciktida "bu satir 100 tur kurdu" bilgisi kaybolmasin.
            _pool = [(lab + V4_TARAMA_ETIKET_EKI, t, c) for lab, t, c in _pool]
        return _pool
    w = int(k if width is None else width)
    _fixed_order = order_fn is not None
    kvals = (None,) if _fixed_order else tuple(knn_vals or (POOL_KNN_CAP,))

    def _score(t):
        return cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)

    def build(th, kv):
        """(en_iyi_tur, en_iyi_skor, [(bant_etiketi, tur, skor), ...])"""
        rx, ry = ((xs, ys) if abs(th) < 1e-9
                  else E.rotate_coords(xs, ys, th))
        _of = order_fn if _fixed_order else _greedy_band_order_k(kv)
        acc = [] if flat else None
        t = _band_hybrid_tour(rx, ry, _of, base_mult, axes=(True,),
                              collect=acc)
        extra = ([] if acc is None else
                 [(f"/b={bk}{'x' if bax else 'y'}", bt, _score(bt))
                  for bk, bax, bt in acc])
        return t, _score(t), extra

    def _lab(th, kv, probe=True):
        s = f"th={th:g}" + ("*" if probe else "")
        return s if kv is None else f"{s}/k={kv}"

    pool = []
    best_th, best_c = None, None
    for th in probes:
        for kv in kvals:
            t, c, extra = build(th, kv)
            base = _lab(th, kv)
            pool.append((base, t, c))
            for suf, bt, bc in extra:
                pool.append((base + suf, bt, bc))
            if best_c is None or c < best_c:
                best_th, best_c = th, c
    # Capraz carpim `w`yi doldurmadiysa eski aci-ince-tarama yolu devreye
    # girer (geriye donuk uyum; varsayilan yapilandirmada calismaz).
    _need = w if not flat else 0
    for th in (v4_angle_ladder(_need, best_th) if len(pool) < _need else ()):
        for kv in kvals:
            t, c, extra = build(th, kv)
            base = _lab(th, kv, probe=False)
            pool.append((base, t, c))
            for suf, bt, bc in extra:
                pool.append((base + suf, bt, bc))
    return _finish_pool(pool, POOL_FLAT_CAP if flat else w)


def snake_v4_height_pool(xs, ys, cost_fn=None, k=V4_POOL_WIDTH,
                         theta_deg=V4_THETA_DEG, width=None,
                         base_mult=V4_BASE_MULT, order_fn=None):
    """ESKI v4 tasarimi -- SABIT acida (-90) SATIR YUKSEKLIGI havuzu.

    ARTIK HICBIR BENCHMARK SATIRINA BAGLI DEGIL (2026-07-31). Olculdu ve
    tutmadi: ham +%10.81 / onarimli +%2.12, 0/8 kazanc; ayrinti akademik
    notun §14'unde. Fonksiyon o olcumun tekrarlanabilmesi icin DURUYOR --
    `snake_greedy_band_tour` ve `snake_stratified_subsample_tour` ile ayni
    sozlesme (kaldirilan sey BENCHMARK SATIRI, fonksiyon degil).

    Blok basindaki nota bakin -- eksen, gerekce ve tasarim kararlari orada.
    Havuzun TEK degiskeni bant sayisi k'dir (yukseklik = genislik / k);
    aci, bant-ici kurucu, ic tarama ve secim olcutu her adayda AYNIdir.

    order_fn: bant-ici kurucu; varsayilan `_greedy_band_order` (greedy-edge).
    `_farthest_band_order` verilirse ayni havuz Far Snake tarafinda kosar.

    Donus: [(etiket, tur, ham_maliyet), ...] -- gercek maliyete (cost_fn;
    yoksa raw-length proxy) gore artan sirali, yinelenen turlar ayiklanmis,
    en cok k eleman. Etiket "h/k=<bant sayisi>" biciminde, yani kazanan
    varyantin KAC BANT kullandigi sonuc satirindan dogrudan okunur."""
    n = len(xs)
    if order_fn is None:
        order_fn = _greedy_band_order
    th = float(theta_deg)
    rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    pool = []
    for kb in v4_band_counts(n, rx, ry,
                             width=(k if width is None else width),
                             base_mult=base_mult):
        t = _band_hybrid_tour(rx, ry, order_fn, base_mult, dks=(0,),
                              axes=(True,), k0=kb)
        c = cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)
        pool.append((f"h/k={kb}", t, c))
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if len(out) >= k:
            break
    return out


def snake_v2_jitter_pool(xs, ys, cost_fn=None, k=10,
                         thetas_mode="zero_oracle", mults=(0.05, 0.08),
                         jitters=((None, 0.0), (1, 0.01), (2, 0.02)),
                         inner_dks=(-1, 0, 1), inner_axes=(True, False)):
    """snake_v2'nin BANT-ICI CESITLENDIRILMIS havuzu (2026-07-24, bant-ici
    cesitlendirme deneyi -- BIRLESIK_HAVUZ.md bulgusunun dogrudan takibi:
    20 tohumlu birlesik havuzda kazanan 13/14 GE cikti; snake tohumlari
    theta x bant cesitliliginin GERCEK havza cesitliligi uretmedigi
    goruldu; GE'nin silahi greedy_edge_tour(tie_seed, tie_eps) jitter'iydi).
    Bu havuz ayni silahi Greedy Snake'e verir: theta x mult x bant-ici-jitter
    kombinasyonlari -- bant ici greedy_edge_tour cagrilari tie_seed/tie_eps
    ile calisir, boylece esit/near-esit uzunluklu kenar secimleri (izgara-
    tipi VLSI levhalarinda cok yaygin) tohumdan tohuma degisir -> ayni
    makro-bant yapisi icinde bile FARKLI onarim havzalari.

    Varsayilan uzay: 2 theta ({0, strip-oracle}) x 2 mult x 3 jitter
    (saf + eps rampali iki tohum) = 12 aday; tekillestirilip gercek maliyete
    (cost_fn; yoksa raw-length proxy) gore artan siralanir, ilk k doner.

    Donus: [(etiket, tur, ham_maliyet), ...] -- etiket
    "th=.../mult=...[+jS/eE.EE]" biciminde (jitter'li adaylar ayrisik)."""
    so = theta_proxy(xs, ys)
    if thetas_mode == "zero_oracle":
        thetas = sorted({0.0, so})
    else:
        thetas = sorted({0.0})
    import functools
    pool = []
    for th in thetas:
        rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        for m in mults:
            for js, je in jitters:
                order_fn = (_greedy_band_order if js is None else
                            functools.partial(_greedy_band_order,
                                              tie_seed=js, tie_eps=je))
                t = _band_hybrid_tour(rx, ry, order_fn, m,
                                      dks=inner_dks, axes=inner_axes)
                c = cost_fn(t) if cost_fn is not None else _raw_length(t, xs, ys)
                lab = f"th={th:g}/mult={m:g}"
                if js is not None:
                    lab += f"+j{js}/e{je:.2f}"
                pool.append((lab, t, c))
    pool.sort(key=lambda r: r[2])
    out, seen = [], set()
    for lab, t, c in pool:
        h = hash(tuple(t))
        if h in seen:
            continue
        seen.add(h)
        out.append((lab, t, c))
        if len(out) >= k:
            break
    return out


_SUBSAMPLE_SIZE = 1200   # same constant as runner.py's theta-scan subset trick
_SUBSAMPLE_SEED = 42


def _stratified_sample_idx(coords, subset_size, seed, grid_n=None):
    """Coarse-grid stratified sample: buckets points on a grid_n x grid_n
    background grid and draws from each cell in proportion to that cell's
    own point count, instead of uniform random sampling -- preserves the
    full point cloud's local density signal (which Snake-Grid's candidate
    generator relies on) even when subset_size << n."""
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    if grid_n is None:
        grid_n = max(4, round(math.sqrt(subset_size / 4.0)))
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        bx = min(grid_n - 1, int((xs[i] - minx) / w * grid_n))
        by = min(grid_n - 1, int((ys[i] - miny) / h * grid_n))
        buckets.setdefault((bx, by), []).append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for pts in buckets.values():
        take = min(len(pts), max(1, round(len(pts) / n * subset_size)))
        picked.extend(rng.sample(pts, take))
    if len(picked) > subset_size:
        picked = rng.sample(picked, subset_size)
    return sorted(picked)


def snake_stratified_subsample_tour(coords, ewt, subset_size=_SUBSAMPLE_SIZE,
                                     seed=_SUBSAMPLE_SEED, theta_deg=0.0):
    """Runs Snake-Grid's REAL, UNMODIFIED search (same candidate generator,
    same 8 variants) on a bounded STRATIFIED sample (density-proportional,
    see _stratified_sample_idx -- not uniform random) to pick (rows, cols,
    axis, reverse_primary, reverse_secondary), then builds the full tour
    ONCE with that winning configuration. `theta_deg`, if nonzero, rotates
    the CONSTRUCTION layout only (true distances/cost always computed on
    the ORIGINAL, unrotated coordinates)."""
    from core import SnakeGridCandidate, SnakeGridPathInitializer

    n = len(coords)
    if n <= subset_size:
        inst, _ = E.make_instance(coords, ewt, theta_deg=theta_deg)
        return SnakeGridPathInitializer(inst).build().tour

    idx = _stratified_sample_idx(coords, subset_size, seed)
    sub_inst, _ = E.make_instance([coords[i] for i in idx], ewt, theta_deg=theta_deg)
    sub_result = SnakeGridPathInitializer(sub_inst).build()

    full_inst, _ = E.make_instance(coords, ewt, theta_deg=theta_deg)
    helper = SnakeGridPathInitializer(full_inst)
    min_x, max_x, min_y, max_y = helper._bounds(helper.points)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    rows, cols = sub_result.rows, sub_result.cols
    cell_size = max(width / cols, height / rows, 1e-9)
    board_w, board_h = cols * cell_size, rows * cell_size
    origin_x = min_x - (board_w - width) / 2.0
    origin_y = min_y - (board_h - height) / 2.0
    full_candidate = SnakeGridCandidate(rows=rows, cols=cols, cell_size=cell_size,
                                         origin_x=origin_x, origin_y=origin_y)
    cells = helper._assign_to_cells(full_candidate)
    return helper._build_variant_tour(
        candidate=full_candidate, cells=cells,
        axis=sub_result.axis, reverse_primary=sub_result.reverse_primary,
        reverse_secondary=sub_result.reverse_secondary)


# ===========================================================================
#  RSGE -- Rotated Serpentine Greedy Edge
#          (Dondurulmus Serpantin Greedy-Edge)
# ===========================================================================
#
#  AILE TANIMI
#  -----------
#  RSGE(theta, b): koordinatlar theta acisinda dondurulur, CIZGILERE DIK
#  eksen boyunca b esit-GENISLIKLI bant acilir, her bandin ici BAGIMSIZ bir
#  greedy-edge cevrimiyle kurulur (k = RSGE_KNN), bantlar serpantin
#  sirasinda birbirine baglanir (ileriye bakisli yonelim + bant siniri
#  dikisi).
#
#  b = 1 OZDESLIGI (makalenin eksenini tasiyan saglama)
#  ----------------------------------------------------
#      RSGE(theta, 1)  ==  theta'da dondurulmus koordinatlarda GLOBAL
#                          greedy-edge (k = RSGE_KNN = 8)
#      RSGE(0,      1)  ==  runner'in `greedy_edge@knn8_greedy` satiri
#                          -- CEVRIM olarak BIT-AYNI
#
#  Cunku b=1'de tek bant butun noktalardir, `_orient_cycle` cevrimi yalnizca
#  bir baslangica dondurup yonunu secer (kapali tur olarak maliyeti
#  DEGISTIRMEZ) ve `_boundary_stitch` bant siniri olmadigi icin islemsizdir.
#  Bu ozdeslik `verification/verify_rsge.py` icinde sinanir.
#
#  NEDEN k = 8 VE NEDEN HAVUZ YOK
#  ------------------------------
#  k=8 projenin aday-listesi TAVANIDIR (bkz. RSGE_KNN notu) ve RGGE ile
#  aynidir. Kiyas eslesi `greedy_edge@knn8_greedy` satiridir: bant-ici aday
#  genisligi ONUNLA AYNI olmak ZORUNDA, aksi halde "bant mi zarar verdi,
#  aday listesi mi daraldi?" ayrisamaz. Ayni sebeple RSGE TEK tur kurar --
#  havuz kurup en iyisini gercek maliyetle secmek bant sayisi kadraninin
#  uzerine ikinci bir kadran koyardi.
#
#  NOT: literatur rakibi `greedy_edge` (knn15_greedy) tabloda AYRICA durur ve
#  k=8'e cekilmez -- rakibi zayiflatmak adaleti ters yone bozardi. RSGE'nin
#  capasi onun k=8 duyarlilik varyantidir.
#
#  DORT BANT SAYISI KURALI -- makalenin bagimsiz degiskeni
#  -------------------------------------------------------
#  Dort satirin FARKLI OLDUGU TEK SEY `b` tamsayisidir. Bolme semasi
#  (esit-genislik), aci kaynagi, kesme ekseni, bant-ici kurucu, k, dikis --
#  hepsi BIREBIR aynidir. Boylece "bant sayisi nasil secilirse secilsin
#  bantlama kaybediyor" iddiasi tek degiskenli olarak okunur.
#
#      resource : b = ceil(n / m*)                        -- kaynak tavani
#      corridor : b = 1 + #{bosluk > C x medyan bosluk}   -- veriden
#      budget   : b = ceil(c n^2 / T)                     -- maliyet modeli
#      fixed    : b = RSGE_FIXED_B (=2)   -- en kucuk asikar-olmayan bolme
#
#  ESIT-GENISLIK vs ESIT-SAYI (olculdu, bilerek KARISTIRILMADI)
#  ------------------------------------------------------------
#  Paralel isciler icin dengeli (esit-SAYI) bant gerekir; kalite bedeli
#  olculdu ve kucuktur (IZGARA_ACISI_VE_SERIT_SAYISI_AKADEMIK_NOT.md
#  Bolum 6: k=8'de 39.0 <-> 39.6, yani +0.6 puan). Yine de DORT SATIRIN
#  HEPSI esit-GENISLIK kullanir: satirlar arasinda iki degisken (b VE bolme
#  semasi) degistirmek kiyasi okunamaz hale getirirdi. Esit-sayi semasi
#  ayrica `grid_theta.snap_bands` icinde durur ve orada olculmustur.
# ---------------------------------------------------------------------------

#: Bant-ici aday genisligi. 2026-09-07'de 15 -> 8 (kullanici karari).
#:
#: NEDEN 8: projenin KENDI kurali. Bkz. yukaridaki "HAVUZ ADAY-LISTESI
#: TAVANI" notu -- "hic Greedy Snake'te ya da Greedy Edge'de k degeri en
#: fazla 8 olsun, esit olcme adil olcum". k=15 bu kurali ihlal ediyordu.
#:
#: SONUCU: b=1 capasinin REFERANSI degisir (capa yapisal olarak KALIR):
#:     RSGE(theta=0, b=1) == greedy_edge@knn8_greedy   [cevrim olarak bit-ayni]
#: Panelde `greedy_edge`in `knn8_greedy` varyanti tam bu satirdir, yani capa
#: hala gercekten kosulabilen bir satira denk gelir.
#:
#: YAN KAZANC: RGGE de k=8 kullanir (rgge.RGGE_KNN). Iki aile artik AYNI aday
#: genisliginde, dolayisiyla dogrudan kiyaslanabilir -- tek degisken BANT mi
#: CERCEVE mi. k farki kaldigi surece bu kiyas kurulamiyordu.
#:
#: Degistirilirse capa da degisir: hangi `greedy_edge` varyantina esit
#: oldugunu `verification/verify_rsge.py` her kosumda sinar.
RSGE_KNN = 8

#: Kaynak kurali: bir iscinin / tek bir aday listesinin tuttugu EN BUYUK
#: bant boyu. b = ceil(n / m*), yani "kisitin izin verdigi EN KUCUK b".
#: 20000 secildi cunku akademik notun Bolum 8 olcumu n <= 105k araliginda
#: global greedy-edge'in hem daha hizli hem daha iyi oldugunu gosteriyor;
#: kural bu yuzden ancak gercekten bir tavan isirdiginda b > 1 uretir.
RSGE_M_STAR = 20000

#: Butce kurali: `fi_bant_deneyi`nde OLCULEN maliyet modeli
#: (b = ceil(c n^2 / T), c = 3.93e-7 s/nokta^2, p = +0.87 ... +1.01).
#: Model KARESEL bir kurucu icin kalibre edildi; greedy-edge O(n log n)
#: oldugundan bu satirin BEKLENEN davranisi "bantlamayi REDDETMEK"tir
#: (T = 300 s'de n <= 27600 icin b = 1). Bu bir kusur degil, kuralin
#: kendisinin verdigi cevaptir ve makalede oyle raporlanir.
RSGE_BUDGET_C = 3.93e-7
RSGE_BUDGET_T = 300.0

#: Koridor kurali: esik = C x (medyan ardisik bosluk). C = 32, akademik
#: notun Bolum 9b taramasindan (C = 2/8/32/64 arasinda 32, yontem k=1'e
#: yakinsamadan once gercekten blok olcekli koridor bulan en kucuk esik).
RSGE_CORRIDOR_C = 32.0

#: Sabit kural: en kucuk asikar-olmayan bolme.
RSGE_FIXED_B = 2

#: Kural anahtarlari -- runner satirlariyla TEK KAYNAK.
RSGE_RULES = ("resource", "corridor", "budget", "fixed")


def _rsge_reduce(deg):
    """Aciyi [-90, 90) araligina indirger. Serit yonu 180 derece periyodik
    oldugundan bu indirgeme bant AILESINI degistirmez."""
    return (float(deg) + 90.0) % 180.0 - 90.0


def _rsge_median(vals):
    """Medyan (numpy'siz; snake_alt numpy import ETMEZ)."""
    s = sorted(vals)
    m = len(s)
    if not m:
        return 0.0
    h = m // 2
    return float(s[h]) if m % 2 else (float(s[h - 1]) + float(s[h])) / 2.0


def rsge_corridor_count(px, c=None):
    """Koridor kurali: kesme ekseni izdusumunde ardisik bosluklardan
    `c x medyan` esigini GECENLERIN sayisi + 1.

    Yalnizca bir TAMSAYI dondurur, kesik KONUMLARINI degil -- bantlar sonra
    esit-GENISLIK ile acilir (bkz. ustteki not: satirlar arasinda tek
    degisken `b` olmali). Kesik konumlarinin kendisiyle yapilan ayristirma
    ayri olculdu (`grid_theta.void_bands` / `snap_bands`).

    Medyan bosluk 0 ise (tamsayi kafeslerde tekrarli izdusum degeri boldur)
    POZITIF bosluklarin medyanina duser; hic pozitif bosluk yoksa (butun
    noktalar kesme eksenine dik tek bir dogru uzerinde) b = 1 doner."""
    c = RSGE_CORRIDOR_C if c is None else float(c)
    v = sorted(px)
    if len(v) < 2:
        return 1, {"rsge_corridor_median_gap": 0.0, "rsge_corridor_thr": 0.0}
    gaps = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    med = _rsge_median(gaps)
    if med <= 0.0:
        pos = [g for g in gaps if g > 0.0]
        if not pos:
            return 1, {"rsge_corridor_median_gap": 0.0,
                       "rsge_corridor_thr": 0.0}
        med = _rsge_median(pos)
    thr = c * med
    b = 1 + sum(1 for g in gaps if g > thr)
    return b, {"rsge_corridor_median_gap": round(med, 6),
               "rsge_corridor_thr": round(thr, 6)}


def rsge_band_count(rule, n, px=None, *, m_star=None, budget_c=None,
                    budget_t=None, corridor_c=None, fixed_b=None):
    """RSGE'nin DORT bant sayisi kurali -- hepsi tek bir tamsayi `b` uretir.

    Doner: (b, detay_sozlugu). Detay satira yazilir (`rsge_*` alanlari) ki
    tabloda "bu kural bu ornekte KAC bant istedi" gorunur olsun; iki kuralin
    ayni b'yi verdigi ornekler (kucuk n'de resource == budget == 1
    beklenir) boylece rastlanti olarak degil OLCUM olarak okunur."""
    if rule not in RSGE_RULES:
        raise ValueError(f"bilinmeyen RSGE bant kurali: {rule!r}")
    d = {"rsge_rule": rule}
    if rule == "resource":
        ms = RSGE_M_STAR if m_star is None else max(1, int(m_star))
        b = -(-int(n) // ms)                      # ceil(n / m*)
        d["rsge_m_star"] = ms
    elif rule == "budget":
        cc = RSGE_BUDGET_C if budget_c is None else float(budget_c)
        tt = RSGE_BUDGET_T if budget_t is None else float(budget_t)
        b = math.ceil(cc * float(n) * float(n) / tt) if tt > 0 else 1
        d["rsge_budget_c"] = cc
        d["rsge_budget_t"] = tt
        d["rsge_budget_pred_s"] = round(cc * float(n) * float(n), 3)
    elif rule == "corridor":
        if px is None:
            raise ValueError("corridor kurali izdusumu (px) ister")
        b, extra = rsge_corridor_count(px, corridor_c)
        d.update(extra)
        d["rsge_corridor_c"] = (RSGE_CORRIDOR_C if corridor_c is None
                                else float(corridor_c))
    else:                                         # fixed
        b = RSGE_FIXED_B if fixed_b is None else int(fixed_b)
    b = max(1, min(int(b), max(1, int(n))))
    d["rsge_b_requested"] = b
    return b, d


def rsge_tour(xs, ys, rule="resource", theta_deg=None, **kw):
    """RSGE(theta, b) -- TEK tur. Doner: (tur, theta, b_istenen, detay).

    `theta_deg=None` -> ailenin ORTAK aci vekili (`theta_proxy`, yani
    varsayilan `grid_theta`). Sayi verilirse hicbir dedektor kosmaz; runner
    aci-duyarliligi satirlarini (ornegin `@zero`, `@manual:-90`) boyle kurar
    ve `@zero` tam olarak "hizalama YOK" ablasyonudur.

    CERCEVE TEK BIR SAYIDIR (theta'). Bantlar HER ZAMAN dondurulmus x
    ekseninde acilir; "y ekseninde kes" durumu theta'ya 90 EKLENEREK
    temsil edilir (`grid_theta.cut_axis` yalnizca bu karari verir). Sebep
    olculdu: eksen takasi bir YANSIMADIR ve tamsayi kafeslerde greedy-edge'in
    k-NN beraberlik bozmasini %3.6'ya kadar oynatir -- yani b=1 ozdesligini
    ve "b ekseni boyunca cerceve sabit" sozlesmesini kirardi.

    KATLAMA YALNIZ DEDEKTOR YOLUNDA yapilir (theta_deg=None). Elle verilen
    acida (`zero`, `manual:<derece>`) aci HARFIYEN kosulur: `@zero`
    ablasyonunun anlami "hicbir hizalama karari yok"tur ve oraya 90 eklemek
    gizli bir hizalama karari olurdu."""
    n = len(xs)
    if n <= 3:
        return (list(range(n)), 0.0, 1,
                {"rsge_rule": rule, "rsge_b_requested": 1, "rsge_b": 1,
                 "rsge_knn": RSGE_KNN})
    _auto = theta_deg is None
    _raw = theta_proxy(xs, ys) if _auto else float(theta_deg)
    th = _rsge_reduce(_raw)
    th_det, folded = th, False
    if _auto:
        # KESME EKSENI ACININ ICINDE (bkz. ustteki "CERCEVE" notu): dedektor
        # "cizgiler y'de tarakli" derse theta+90'a doneriz ve her zaman x'te
        # keseriz. Eksen takasi (px, py -> py, px) bir YANSIMADIR ve olculdu
        # ki tamsayi kafeslerde greedy-edge'i %3.6'ya kadar oynatir
        # (beraberlik bozmasi uzerinden); saf dondurme bu belirsizligi
        # ortadan kaldirir ve cerceveyi TEK sayiya indirir.
        rx0, ry0 = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
        try:
            import grid_theta as _GT
            folded = (_GT.cut_axis(rx0, ry0, theta=0.0) == 1)
        except Exception:
            folded = False
        if folded:
            th = _rsge_reduce(th + 90.0)
    px, py = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    b, d = rsge_band_count(rule, n, px, **kw)
    t = _band_hybrid_tour(px, py, _greedy_band_order_k(RSGE_KNN), 0.0,
                          dks=(0,), axes=(True,), k0=b)
    d.update({"rsge_b": int(LAST_BUILD.get("n_bands", b)),
              "rsge_knn": RSGE_KNN,
              "rsge_theta": round(th, 4),
              "rsge_theta_detected": round(th_det, 4),
              "rsge_axis_folded": bool(folded)})
    return t, th, b, d
