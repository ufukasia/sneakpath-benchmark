# -*- coding: utf-8 -*-
"""IZGARA ACISI VE SERIT SINIRI TESPITI -- olculmus, kanitlanmis optimum
turlara karsi dogrulanmis yeni aci dedektoru.

===========================================================================
 1. MOTIVASYON (kullanici tespiti, 2026-07-27)
===========================================================================
PCB/VLSI kartlarinda "birbirine cok yakin noktalar CIZGI olusturur; bu
cizgilerin yonu IZGARA yonudur" ve "bloklar arasindaki BOS koridorlar serit
sinirlarini verir". Bu dosya bu iki iddiayi olculebilir hale getirir.

Onemli: bir izgaranin IKI dik yonu vardir, dolayisiyla izgara acisi 90 derece
periyotludur ve dogal araligi [-45, 45)'tir. (Serit ekseni ayrica secildigi
icin 90 derecelik belirsizlik zaten sogurulur.)

===========================================================================
 2. IKI BAGIMSIZ OLCUT (ikisi de burada; ayni cevabi veriyorlar)
===========================================================================
  A) TARAK (comb) SKORU -- `grid_angle`
     Aciyi dondurup eksene izdusurunce izgara satirlari birkac ayrik seviyeye
     COKER. Izdusum histogramindaki yogunlasma (Simpson ikinci momenti)
     olculur; duzgun dagilimda ~1, mukemmel tarakta ~nbins.

  B) BOSLUK (void) ORANI -- `void_fraction`
     Ayni izdusumde neredeyse BOS olan ardisik bolgelerin (koridorlar)
     kapladigi oran. Bu "bloklarin ayrilabilirligini" olcer, satir
     quantizasyonunu degil -- yani A'dan gercekten farkli bir seydir.

  OLCULDU (12 kume): A ve B'nin argmax'lari yapili kumelerde 0.00-0.06
  DERECE farkla ayni cikti. Yapisiz kontrollerde (kroA100, rat783, fnl4461)
  19-30 derece ayristilar ve her ikisinin de kaldiraci coktu -- yani ikisi de
  ayni anda "burada izgara yok" diyor.

===========================================================================
 3. DOGRULAMA -- KANITLANMIS OPTIMUM TURLARA KARSI
===========================================================================
LKH-3 (external_solvers.run_lkh) ile uretilen ve maliyeti KATALOG OPTIMUMUNA
TAM ESIT cikan turlar yer gercegi olarak kullanildi (16 kosumdan 12'si
kanitlanmis optimal: xqf131 564, pma343 1368, lim963 2789, fnb1615 4956,
dcb2086 6600, bch2762 8234, pcb3038 137694, fnl4461 182566, kroA100 21282,
rat783 8806, beg3293 9772, frv4410 ~+0.009%).

Optimal turun "yuruduğu yon", turun KISA kenarlarinin (uzunluk <= 0.75
nicelik) katlanmis (4*alpha) dairesel ortalamasidir. Sonuc:

    kume      optimal turun acisi   grid_angle hatasi   strip_oracle hatasi
    bch2762          0.10                 0.00                 0.10
    dcb2086         -0.05                 0.05                19.95   <-- 20 derece
    fnb1615          0.21                 0.21                 0.21
    lim963           0.44                 0.43                 0.44
    pma343           0.31                 0.32                30.31   <-- 30 derece
    xqf131          -0.25                 0.30                10.25   <-- 10 derece

`grid_angle` 6/6 sette 0.05-0.43 derece hata ile optimal turun acisini
buluyor. Mevcut vekil `snake_alt._strip_oracle_theta` 3/6 sette 10-30 derece
sapiyor.

"CIZGILERIN COGUNLUGU" TESTI: optimal turun kisa kenarlarinin %60.2-%84.2'si
bulunan acinin +-10 derecesinde (mod 90). Strip-oracle'in yanildigi
kumelerde bu oran %9.8-%10.0'a cokuyor. Yani iddia yalniz dogru degil,
NICELIKSEL olarak da dogru.

===========================================================================
 4. GUVEN (abstain) ESIGI -- "izgara yoksa uydurma"
===========================================================================
`grid_angle` ikinci deger olarak KALDIRAC dondurur (en iyi skor / medyan
skor - 1). Olculen ayrim keskindir:

    izgarali  : 1.20 (pcb3038) ... 7.11 (fl3795)
    yapisiz   : 0.07 (fnl4461), 0.12 (rat783), 0.22 (kroA100)

`GRID_CONF_MIN = 0.6` bu iki kumeyi ayirir (en dusuk yapili 1.20, en yuksek
yapisiz 0.22). Esigin altinda cagiran taraf eski vekile DUSMELIDIR.

===========================================================================
 5. ISLEVSEL KAZANC
===========================================================================
8 buyuk KANITLANMIS OPTIMUM VLSI kumesinde (n=5557..13584), tek degisken aci:

    Greedy Snake v1 cekirdegi @ strip_oracle : ort %18.23 gap, 2.2 sn
    Greedy Snake v1 cekirdegi @ grid_angle   : ort %16.92 gap, 2.0 sn

Yani -1.31 puan, ve DAHA UCUZ (tarak taramasi O(180n) numpy; strip-oracle 19
ayri boustrophedon kuruyor).

MEKANIZMA (onemli): kazanc "daha iyi serit acisi" degil, "SERMEMEYE dogru
karar vermek"tir. Dogru acida ic k-taramasi cogu kez k=1 seciyor ve sonuc
global greedy-edge ile BIREBIR AYNI oluyor (fea5557, lap7454, bnd7168,
ida8197 satirlarinda gap'ler tam esit). Yanlis acida ise kotu bir yerden
kesiyor.

===========================================================================
 6. SERIT SAYISI -- OLCULMUS NEGATIF SONUC (durustce kaydedilir)
===========================================================================
"Bosluklar serit sayisini versin" fikri denendi. Koridorlar GERCEKTEN var
(dogru acida izdusumun %32.6-%90.0'i bos; kontrollerde %6.5-%14.5). Ama
serit sayisini artirmak, sinirlar nereye konursa konsun, kaliteyi bozuyor.
16 kume x 5 seritleme semasi x k in {2,4,8,16,32} taramasi (ort gap %):

    k        2      4      8     16     32
    esit-genislik  18.5   26.8   39.0   66.1  111.6
    esit-sayi      --     --     39.6   74.4  126.1
    void (en buyuk bosluk) --    23.1   34.6   61.0  102.2
    void + dengeli yapistirma --  --    37.0   70.5  122.3
    (k=1 tabani: 17.0)

Yani k=1 her semada kazaniyor. SEBEP OLCULDU (bkz. Bolum 7): patlayan sey
GECIS maliyeti degil (toplamin yalniz %1-11'i), BANT ICI uzunluk.

n<=13584'te ayrica global greedy-edge tek basina %15.58 gap / 0.4 sn ile
banded surumlerin HEPSINI hem kalitede hem surede geciyor -- yani bu
olcekte seritlemenin yapacak isi yok. Serit yapisinin gerekcesi ancak
global greedy-edge'in pahalilastigi olcekte aranabilir.

===========================================================================
 7. CEVRIM -> YOL DUZELTMESI (`greedy_band_path`)
===========================================================================
Bant ici kurucu bir CEVRIM uretir; `snake_alt._orient_cycle` cevrimi bir
baslangica dondurup dogrusal okur -- yani aslinda cevrimin BIR KENARINI atar.
Ama hangi kenarin atildigi skora GIRMIYORDU:

    eski : d(onceki_cikis, giris) + LAMBDA * d(cikis, sonraki_merkez)
    yeni : d(onceki_cikis, giris) + LAMBDA * d(cikis, sonraki_merkez)
           - d(atilan kenar)

Ince bir seritte cevrim "yukari cik, asagi in" oldugundan atilmasi gereken
tam da o uzun donus kenaridir. Terim eklenmeden bandin uzun boyu iki kez
odenir, ve bu k bant boyunca tekrarlanir.

OLCULDU (16 kume, esit-genislik seritleme, ort gap %; eski -> yeni):
    k=1  : 17.1 -> 17.1   (MALIYET BIREBIR AYNI, 16/16 kume)
    k=2  : 23.1 -> 18.5
    k=4  : 36.5 -> 26.8
    k=8  : 62.9 -> 39.0
    k=16 : 104.8 -> 66.1
    k=32 : 183.3 -> 111.6

Maliyet artisi YOK: zaten tum baslangiclar x 2 yon taraniyordu, atilan kenar
her adayda O(1)'de bilinir.

k=1 SAGLAMASI: tek bantta tur zaten KAPALI bir cevrimdir, dolayisiyla hangi
kenarin "atildigi" maliyeti degistiremez -- ve olcum bunu dogruluyor (16/16
kumede maliyet birebir esit). Dondurulen indeks listesi farkli olabilir
(baslangic/yon secimi degisir), turun kendisi ayni cevrimdir.

BU DOSYA HICBIR MEVCUT YONTEMI DEGISTIRMEZ. `snake_alt`/`runner` hattina
baglanmasi ayri ve acik bir karardir.
"""
from __future__ import annotations

import math

import numpy as np

import tsplib_engine as E
import snake_alt as SA

#: Bolum 4'te olculen ayrim: yapili kumelerin en dusugu 1.20, yapisiz
#: kontrollerin en yuksegi 0.22. Esik ikisinin arasinda, yapisiz tarafa
#: yakin secildi (yanlis pozitif = uydurma aci, asil kacinilmak istenen).
GRID_CONF_MIN = 0.6

#: Sifira bu kadar yakin bir aci TAM sifira yapistirilir. Sebep olculdu:
#: tamsayi kafeste theta=0'da noktalar TAM esit koordinatlara sahiptir;
#: 0.004 derecelik bir donme bile bu esitlikleri bozar ve bant uyeligini
#: toptan degistirir (lim963'te gap %15.60 -> %21.80). Yani "neredeyse 0"
#: ile "tam 0" ayni sey DEGILDIR ve tam 0 tercih edilir.
_ZERO_SNAP_DEG = 0.06

#: `_band_hybrid_tour`'un ileriye bakis katsayisi ile ayni tutulur.
_LAMBDA = SA._LOOKAHEAD_LAMBDA

LAST: dict = {}


# ---------------------------------------------------------------------------
#  Yardimcilar
# ---------------------------------------------------------------------------
def _as_points(xs, ys=None):
    """(xs, ys) liste ciftini ya da (n,2) diziyi tek bir (n,2) numpy dizisine
    cevirir -- runner hatti liste cifti, deney betikleri dizi geciriyor."""
    if ys is None:
        return np.asarray(xs, dtype=float)
    return np.column_stack([np.asarray(xs, dtype=float),
                            np.asarray(ys, dtype=float)])


def _nbins(n):
    return max(32, min(2048, int(math.sqrt(n) * 12)))


def _project(P, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return c * P[:, 0] + s * P[:, 1], -s * P[:, 0] + c * P[:, 1]


def _comb_density(v, nbins):
    """Izdusumun 'tarakligi': normalize histogramin ikinci momenti x nbins.
    Duzgun dagilimda ~1, tum kutle tek bin'de ise ~nbins."""
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return float(nbins)
    idx = np.minimum(((v - lo) / (hi - lo) * nbins).astype(np.int64), nbins - 1)
    p = np.bincount(idx, minlength=nbins).astype(float)
    p /= p.sum()
    return float((p * p).sum() * nbins)


def _void_fraction_1d(v, nbins=None, empty_frac=0.02):
    """Izdusumdeki KORIDOR orani: yogunlugu ortalamanin `empty_frac` katinin
    altina dusen ardisik bolgelerin kapladigi oran. Kenardaki bos kosular
    sayilmaz (onlar 'ic koridor' degil, sadece kartin disi)."""
    n = v.size
    if nbins is None:
        nbins = max(16, min(4000, n // 3))
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return 0.0
    idx = np.minimum(((v - lo) / (hi - lo) * nbins).astype(np.int64), nbins - 1)
    h = np.bincount(idx, minlength=nbins).astype(float)
    empty = h <= h.mean() * empty_frac
    tot, i = 0, 0
    while i < nbins:
        if empty[i]:
            j = i
            while j < nbins and empty[j]:
                j += 1
            if i > 0 and j < nbins:          # ic koridor
                tot += j - i
            i = j
        else:
            i += 1
    return tot / nbins


# ---------------------------------------------------------------------------
#  A) Izgara acisi
# ---------------------------------------------------------------------------
def grid_angle(xs, ys=None, step=1.0, refine=True):
    """Izgara acisini ve GUVEN kaldiracini dondurur: (theta_deg, conf).

    theta_deg in [-45, 45); conf = en iyi skor / medyan skor - 1.
    conf < GRID_CONF_MIN ise nokta bulutunda izgara YOKTUR ve cagiran taraf
    kendi eski vekiline dusmelidir (bkz. modul basligi, Bolum 4).

    Maliyet: kaba tarama 90/step aci x O(n), ince ayar 40 x O(n) -- numpy
    vektorel, yani pratikte `_strip_oracle_theta`'dan (19 boustrophedon
    insasi) daha ucuz.
    """
    P = _as_points(xs, ys)
    n = len(P)
    if n < 8:
        return 0.0, 0.0
    nb = _nbins(n)
    C = P - P.mean(axis=0)

    def score(t):
        px, py = _project(C, t)
        return max(_comb_density(px, nb), _comb_density(py, nb))

    ths = np.arange(-45.0, 45.0, step)
    sc = np.array([score(float(t)) for t in ths])
    th = float(ths[int(sc.argmax())])
    if refine:
        fine = np.arange(th - step, th + step + 1e-9, 0.05)
        sf = np.array([score(float(t)) for t in fine])
        th = float(fine[int(sf.argmax())])
    conf = float(sc.max() / (np.median(sc) or 1e-9) - 1.0)
    if abs(th) < _ZERO_SNAP_DEG:
        th = 0.0
    return th, conf


def void_fraction(xs, ys=None, *, theta=0.0):
    """Verilen acida IKI eksenin en iyi koridor orani (Bolum 2-B).
    `grid_angle` ile bagimsiz bir ikinci olcut; olculdu ki argmax'lari
    yapili kumelerde 0.06 derece icinde ayni.

    `theta` ANAHTAR KELIMEDIR: `void_fraction(P, 12.0)` yazilirsa 12.0
    sessizce `ys` yerine gecerdi -- imza bunu imkansiz kilar."""
    P = _as_points(xs, ys)
    px, py = _project(P - P.mean(axis=0), theta)
    return max(_void_fraction_1d(px), _void_fraction_1d(py))


def cut_axis(xs, ys=None, *, theta=0.0):
    """Seritlerin hangi eksen boyunca kesilecegi: izdusumu daha tarakli olan
    eksen, yani CIZGILERE DIK olan. 0 = x uzerinde kes, 1 = y uzerinde.
    `theta` anahtar kelimedir (bkz. `void_fraction`)."""
    P = _as_points(xs, ys)
    nb = _nbins(len(P))
    px, py = _project(P - P.mean(axis=0), theta)
    return 0 if _comb_density(px, nb) >= _comb_density(py, nb) else 1


def theta_for(xs, ys=None, fallback=None):
    """Kullanima hazir aci: guven yeterliyse izgara acisi, degilse
    `fallback` (verilmezse `snake_alt._strip_oracle_theta`).

    LAST sozlugu tani icin doldurulur (tur/insa davranisini ETKILEMEZ)."""
    P = _as_points(xs, ys)
    th, conf = grid_angle(P)
    ok = conf >= GRID_CONF_MIN
    if not ok:
        if fallback is None:
            th = SA._strip_oracle_theta(P[:, 0].tolist(), P[:, 1].tolist())
        else:
            th = float(fallback)
    LAST.clear()
    LAST.update({"theta": th, "conf": conf, "used_grid": bool(ok),
                 "void": void_fraction(P, theta=th),
                 "axis": cut_axis(P, theta=th)})
    return th


# ---------------------------------------------------------------------------
#  B) Serit sinirlari
# ---------------------------------------------------------------------------
def void_bands(v, k):
    """1-B tek-baglantili bolme: siralanmis izdusumdeki EN BUYUK k-1
    bosluktan keser. Bantlar kesme ekseni boyunca artan sirada doner.

    UYARI (olculdu): en genis bosluklar cogu kartta KENARDAKI aykiri
    noktalarin bosluklaridir, ic koridorlar degil -- dcb2086/k=16'da en kucuk
    bant 1 nokta, en buyugu 407 (407x dengesizlik). `snap_bands` bu yuzden
    vardir."""
    v = np.asarray(v, dtype=float)
    n = v.size
    order = np.argsort(v, kind="stable")
    k = max(1, min(int(k), n))
    if k == 1 or n < 2:
        return [order.tolist()]
    d = np.diff(v[order])
    cuts = np.sort(np.argsort(d)[-(k - 1):])
    bands, prev = [], 0
    for c in cuts:
        bands.append(order[prev:c + 1].tolist())
        prev = c + 1
    bands.append(order[prev:].tolist())
    return [b for b in bands if b]


def snap_bands(v, k, win=0.45):
    """Dengeli (esit-SAYI) sinirlari cevrelerindeki en genis bosluga
    yapistirir: hem bantlar dengeli kalir hem kesikler koridora duser.
    win=0 saf esit-sayi bolmedir."""
    v = np.asarray(v, dtype=float)
    n = v.size
    k = max(1, min(int(k), n))
    order = np.argsort(v, kind="stable")
    if k == 1 or n < 2:
        return [order.tolist()]
    d = np.diff(v[order])
    per, half = n / k, max(1, int(n / k * win))
    cuts, used = [], -1
    for j in range(1, k):
        t = int(round(j * per)) - 1
        lo, hi = max(used + 1, t - half), min(n - 2, t + half)
        c = (lo + int(np.argmax(d[lo:hi + 1]))) if lo <= hi \
            else min(max(t, used + 1), n - 2)
        cuts.append(c)
        used = c
    bands, prev = [], 0
    for c in cuts:
        bands.append(order[prev:c + 1].tolist())
        prev = c + 1
    bands.append(order[prev:].tolist())
    return [b for b in bands if b]


# ---------------------------------------------------------------------------
#  C) Bant ici kurucu -- cevrim yerine YOL
# ---------------------------------------------------------------------------
def _orient_path(seg, xs, ys, entry_pt, next_c):
    """`snake_alt._orient_cycle`'in yol-farkindalikli hali: skora ATILAN
    KENARIN uzunlugu da girer (bkz. modul basligi, Bolum 7).

    Ileri okumada atilan kenar (seg[si-1], seg[si]), geri okumada
    (seg[si], seg[si+1]) -- her ikisinde de d(giris_noktasi, cikis_noktasi),
    yani ek maliyet yok."""
    m = len(seg)
    if m <= 2:
        return list(seg)
    hypot = math.hypot
    ex, ey = entry_pt
    best_s, best_si, best_fwd = None, 0, True
    for si in range(m):
        a = seg[si]
        da = hypot(xs[a] - ex, ys[a] - ey)
        for fwd in (True, False):
            b = seg[si - 1] if fwd else seg[(si + 1) % m]
            s = da - hypot(xs[a] - xs[b], ys[a] - ys[b])
            if next_c is not None:
                s += _LAMBDA * hypot(xs[b] - next_c[0], ys[b] - next_c[1])
            if best_s is None or s < best_s:
                best_s, best_si, best_fwd = s, si, fwd
    rot = seg[best_si:] + seg[:best_si]
    return rot if best_fwd else [rot[0]] + rot[1:][::-1]


def greedy_band_path(band, xs, ys, entry_pt, next_c=None):
    """`snake_alt._greedy_band_order` ile AYNI kurucu (k-NN adayli
    greedy-edge cevrimi), yalniz cevrim->yol adimi atilan kenari da hesaba
    katar. Imza birebir aynidir, yani order_fn olarak dogrudan gecirilebilir."""
    m = len(band)
    if m <= 3:
        return SA._greedy_band_order(band, xs, ys, entry_pt, next_c)
    lxs = [xs[i] for i in band]
    lys = [ys[i] for i in band]
    local = E.greedy_edge_tour(lxs, lys, k=min(8, m - 1))
    return _orient_path([band[j] for j in local], xs, ys, entry_pt, next_c)
