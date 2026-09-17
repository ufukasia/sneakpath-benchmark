# -*- coding: utf-8 -*-
"""RGGE -- Rotated-Grid Greedy Edge / Dondurulmus Izgara Greedy-Edge.

TEK TUR. HAVUZ YOK.
===================
Bu modul, kardes projedeki `knn_pool.py`nin (bkz. o projenin
`ADAY_LISTESI_PERTURBASYONU_AKADEMIK_NOT.md`) yalniz TEK ADAY kurucusunu ve
MEKANIZMA OLCUSUNU tasir. Havuz makinesi (aday kumesi kurup en iyisini
gercek maliyetle secme) BILEREK TASINMAMISTIR -- 2026-09-07 kullanici
karari: "havuz istemiyorum sadece tek tur, havuzlar deneme sayisindan
basarili gozukuyor". Havuzun kendisi zaten olculmustu ve iddiasi zayifti
(o notun Bolum 3'u: yon x genislik havuzu, tie-jitter havuzuyla
ISTATISTIKSEL OLARAK AYIRT EDILEMIYOR -- 38/39, p = 1.0).

YONTEM
------
Koordinatlar theta acisinda dondurulur, istege bagli olarak eksen sirasi
DEVRIK alinir, ve duz `greedy_edge` kosulur (k = RGGE_KNN). Kurucu
MEKANIGI hic degismez -- degisen tek sey greedy-edge'in gordugu ADAY
LISTESININ ic yapisidir.

    RGGE(theta, devrik) = greedy_edge( donusturulmus koordinatlar, k )

MEKANIZMA -- neden ise yariyor (olculdu, kardes projede)
--------------------------------------------------------
Oklid mesafesi rotasyon degismezidir; o halde tur neden degisiyor? Iki
aday aciklama vardi:

  (a) `grid_knn` eksen-hizali bir izgara kullanir -> dondurunce YAKLASIK
      kNN'in hatasi degisir.
  (b) BERABERLIK: tamsayi kafeste es-uzaklikli komsular/kenarlar
      rotasyondan sonra float'ta ~1e-13 ayrisir.

Ayni mekanik KESIN (O(n^2)) kNN ile kosuldu ve etki DURDU -> aciklama (a)
DEGIL (b). 87 ornekte dogrulandi: beraberligin iki kanali da kapatildiginda
(kirpma + siralama) yon ekseni 23 ornegin 22'sinde ETKISIZ; tek gercek
istisna gr229. Spearman sira korelasyonu: sinir beraberligi 0.63, tekrarli
uzunluk 0.65 (kontrol olarak n: 0.40). Ortalama yon kazanci beraberlikli
orneklerde %1.81, beraberliksizlerde %0.06.

  > Rotasyon geometrik bir arama DEGIL, YAPILI BIR BERABERLIK KIRMA
  > operatorudur; etkisi beraberlik bollugu ile olculebilir bicimde
  > sinirlidir.

`tie_measure` tam olarak bu kosulu olcer ve YALNIZ rgge satirlarina yazilir
(kullanici karari: "diger yontemlere kesinlikle bulastirilmamali").

EKSEN DEVRIGI -- rotasyondan AYRI bir kanal
-------------------------------------------
`devrik=True` koordinat sirasini (px, py) -> (py, px) yapar. Bu GEOMETRIK
BIR DONUSUM DEGILDIR: mesafeler ve dolayisiyla beraberlik-DISI her sey
aynidir. Degisen sey `grid_knn`in hucre numaralandirmasidir, yani
ES-UZAKLIKLI adaylar arasinda yapilan kirpma. Olculdu: 8 kumenin 7'sinde
etkisi YOK; u159 gibi beraberligi bol bir kafeste 48156 vs 49589.

Bu yuzden iki satir vardir (`rgge` devrik, `rgge_x` devriksiz): aralarindaki
TEK degisken bu kanaldir, dolayisiyla "kazanc rotasyondan mi izgara
yeniden-numaralandirmasindan mi?" sorusu tabloda tek degiskenli okunur.

YAPISAL CIPA
------------
    RGGE(theta=0, devrik=False)  ==  greedy_edge@knn8_greedy
                                     [CEVRIM OLARAK BIT-AYNI]

Cunku theta=0'da dondurme yok, devrik=False'ta takas yok, ve geri kalan tek
islem `_baslangica_dondur`dur -- kapali bir turun maliyetini DEGISTIRMEZ.
Yani rakip, bu ailenin (theta=0, devriksiz) noktasidir; kiyas iki farkli
yontem arasinda degil TEK ailenin icinde yapilir. Saglama:
`verification/verify_rgge.py`.

k = 8 NEDEN
-----------
Kardes projedeki yapilandirmalarin bant-ici/aday genisligi 8'dir ve
`greedy_edge`in `knn8_greedy` varyanti bunun BIREBIR karsiligidir. Boylece
capa satiri panelde gercekten kosulabilir bir satira denk gelir. k ekseni
zaten `greedy_edge`in kendi varyant boyutudur (knn3/knn8/knn15/full);
burada k SABIT tutulur ki tek degisken CERCEVE olsun.
"""
from __future__ import annotations

import math
import random

import tsplib_engine as E

#: Aday listesi genisligi. SABIT -- k ekseni `greedy_edge`in kendi varyant
#: boyutudur (GE_VARIANT_SELECTABLE); burada degistirilirse tek-degisken
#: sozlesmesi ve `knn8_greedy` capasi kirilir.
RGGE_KNN = 8

#: `tie_measure` ornekleme parametreleri. Deterministik (sabit tohum):
#: ayni ornek her zaman ayni sayiyi verir, yoksa "mekanizma olcusu"
#: kosumdan kosuma oynardi ve iddia dogrulanamazdi.
TIE_SAMPLE = 400
TIE_SEED = 20260802


def tie_measure(xs, ys, k=None, ornek=TIE_SAMPLE, tohum=TIE_SEED):
    """BERABERLIK BOLLUGU -- yontemin calisabilmesinin YAPISAL KOSULU.

    Doner: (sinir_beraberlik, tekrarli_uzunluk)

      sinir_beraberlik  -- ornekte, k'inci ve (k+1)'inci komsu mesafesi
                           BIREBIR esit olan noktalarin orani. Aday
                           listesinin KIRPMA sinirinda gercek bir
                           belirsizlik var mi?
      tekrarli_uzunluk  -- aday kenar uzunluklarinin tekrarlilik orani
                           (1 - benzersiz/toplam). Kenar SIRALAMASINDA
                           beraberlik ne kadar bol?

    Aday listesi `grid_knn` ile kurulur, yani kurucunun GERCEKTEN gordugu
    yapiyla ayni; olcu O(n*k) ve n=100k'da da kosulabilir.

    NEDEN ONEMLI: tez "rotasyon = yapili beraberlik kirma"dir. Tez dogruysa
    bu olcu SIFIRA yaklastiginda cerceve degisiklikleri ETKISIZ kalmalidir.
    Olculdu ve oyle: kroA100 / ch150 / rd400 (olcu = 0.000) uc kumede de
    cerceve degisikligi tek turla birebir ayni sonucu verdi.

    (Kardes projedeki `knn_pool.beraberlik_olcusu`den BIREBIR tasinmistir:
    ayni tohum, ayni ornekleme, ayni sayi.)

    !! `tekrarli_uzunluk`UN TABANI VAR -- MUTLAK DEGERI OKUMAYIN !!
    2026-09-07'de olculdu: ornek n'e yaklastiginda (ornek >= n ise TUM
    noktalar secilir) KARSILIKLI komsu ciftleri iki kez sayilir -- (i,j)
    mesafesi bir kez i'nin listesinde, bir kez j'nin listesinde. Bu, hicbir
    gercek beraberlik olmasa bile ~0.4 taban uretir: duzgun (float) bir
    bulutta olcu 0.41 cikiyor, oysa o bulutta es-uzaklikli kenar YOK
    (sinir_beraberlik = 0.000 ayni ornekte).

    Yani olcu KARSILASTIRMALI okunmalidir (kafes 0.99 <-> duzgun 0.41), mutlak
    bir "kenarlarin %41'i beraberlikli" ifadesi olarak DEGIL. Tanim BILEREK
    duzeltilmedi: kardes projenin yayimlanmis Spearman katsayilari (0.65) ve
    esik olcumleri bu tanimla hesaplandi; degistirmek onlari karsilastirilamaz
    kilardi. Temiz sinyal `sinir_beraberlik`tir (tabani yoktur)."""
    k = RGGE_KNN if k is None else int(k)
    n = len(xs)
    if n < k + 3:
        return (0.0, 0.0)
    knn = E.grid_knn(xs, ys, min(k + 1, n - 1))
    rng = random.Random(tohum)
    idx = rng.sample(range(n), min(n, ornek))
    hypot = math.hypot
    sinir = 0
    uzunluklar = []
    for i in idx:
        d = [hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in knn[i]]
        if len(d) > k and abs(d[k - 1] - d[k]) < 1e-9:
            sinir += 1
        uzunluklar.extend(round(v, 6) for v in d[:k])
    if not uzunluklar:
        return (0.0, 0.0)
    return (sinir / len(idx), 1.0 - len(set(uzunluklar)) / len(uzunluklar))


def _baslangica_dondur(tur, px, py, giris):
    """Cevrimi `giris`e EN YAKIN dugumden baslayacak sekilde dondurur.

    KOZMETIKTIR: kapali bir turun maliyetini DEGISTIRMEZ, yalnizca liste
    hangi indeksten basliyor onu belirler. Korunmasinin sebebi tarihsel
    BIT-PARITEdir -- onarim katmani turu bir LISTE olarak alir ve kardes
    projedeki uretim yolu tam olarak bunu yapiyordu; kaldirilirsa maliyetler
    ayni kalir ama onarim SONRASI sayilar kayabilir."""
    m = len(tur)
    if m <= 2:
        return list(tur)
    hypot = math.hypot
    ex, ey = giris
    best_s, best_si = None, 0
    for si in range(m):
        a = tur[si]
        da = hypot(px[a] - ex, py[a] - ey)
        if best_s is None or da < best_s:
            best_s, best_si = da, si
    return tur[best_si:] + tur[:best_si]


def theta_proxy(xs, ys):
    """Ailenin ORTAK aci vekili. `snake_alt.theta_proxy`e devreder.

    NEDEN ORADA: vekil iki kaynakli bir anahtardir -- izgara acisi
    (`grid_theta.theta_for`) ve guven esigi altinda ona dusulen eski
    strip-oracle. Tek anahtar sozlesmesi (snake_alt.ANGLE_PROXY) korunmali,
    yoksa iki kopya sessizce ayrisir ve "ayni aci makinesi" iddiasi coker."""
    import snake_alt as _SA
    return _SA.theta_proxy(xs, ys)


def rgge_tour(xs, ys, theta_deg=None, devrik=True):
    """RGGE -- TEK tur. Doner: (tur, theta, detay).

    `theta_deg=None` -> ailenin ORTAK aci vekili (`theta_proxy`, varsayilan
    `grid_theta`). Sayi verilirse hicbir dedektor kosmaz.

    ACI SECIMI (2026-09-07, kullanici karari): runner bu satiri YALNIZ
    panelde secili aci kaynaginda kosar -- tek satir, tek aci
    (runner.ANGLE_PANEL_CANONICAL). Eskiden her kosumda grid_theta + theta=0
    + panel secimi olmak uzere UC satir uretiliyordu; olculdu ki grid_theta
    106 kumenin 98'inde tam 0.0 donuyor (eksen-hizali VLSI kafesleri) ve
    kanonik satir 99/106 kumede greedy_edge@knn8 ile BIREBIR ayni cikiyordu.
    "Hizalama YOK" ablasyonu (theta=0, ailenin capasi) icin panelden
    "theta = 0" secilir.

    `devrik` -> k-NN izgarasinin hucre numaralandirmasini degistiren eksen
    takasi (bkz. modul basligi "EKSEN DEVRIGI"). Geometrik degildir.

    Aci [-90, 90) araligina indirgenir: cerceve yonu 180 derece periyodiktir.
    """
    n = len(xs)
    d = {"rgge_knn": RGGE_KNN, "rgge_devrik": bool(devrik)}
    if n <= 3:
        d.update(rgge_theta=0.0, rgge_tie_boundary=0.0, rgge_tie_repeat=0.0)
        return list(range(n)), 0.0, d
    _raw = theta_proxy(xs, ys) if theta_deg is None else float(theta_deg)
    th = (float(_raw) + 90.0) % 180.0 - 90.0
    rx, ry = (xs, ys) if abs(th) < 1e-9 else E.rotate_coords(xs, ys, th)
    px, py = (ry, rx) if devrik else (rx, ry)
    tur = E.greedy_edge_tour(px, py, k=min(RGGE_KNN, n - 1))
    tur = _baslangica_dondur(tur, px, py, (px[0], min(py)))
    d["rgge_theta"] = round(th, 4)
    # MEKANIZMA OLCUSU -- yalniz bu satira yazilir. Asla patlatmamali:
    # bir tani alani, turu uretmeyen bir yan urundur.
    try:
        sinir, tekrar = tie_measure(xs, ys)
        d["rgge_tie_boundary"] = round(sinir, 4)
        d["rgge_tie_repeat"] = round(tekrar, 4)
    except Exception:
        pass
    return tur, th, d
