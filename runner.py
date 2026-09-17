# -*- coding: utf-8 -*-
"""Benchmark runner: runs the user's method pipeline on a TSPLIB instance and
emits a complete academic result JSON (cost, gap%% vs BKS, time, iterations,
improvements, convergence history, ablation deltas, multi-seed stochastic stats).

Usage:
  python runner.py --dataset kroA100                 # one dataset, all methods
  python runner.py --dataset pcb3038 --methods greedy_edge,repair_vnd,repair_window,ils
  # insa-disi yontemlerin GIRDISI admin panelinden secilir
  # (dashboard/methods_config.json -> "seeds"); bir ablasyon satiri baska bir
  # ablasyonun ciktisini de girdi alabilir (zincir), dongu otomatik kirilir.
  python runner.py --all                              # batch over every dataset
  python runner.py --dataset rat783 --seeds 8 --time-budget 1.5
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import external_solvers as ES
import rgge as RG
import frame_methods as FM
from parallel_ge import concat_tour as PG_concat
import snake_alt as SA
import tsplib_engine as E
from core import (
    build_window_repair_result,
    build_adaptive_window_repair_result,
)
import line_reassign_optimizer as lro
import repair as REP

# Batch benchmarking never replays the per-move tour snapshots; dropping them
# avoids O(n) copies per accepted move (gigabytes on large instances). It also
# silences the optimizers' per-iteration progress output so hot loops don't
# waste time formatting strings (the runner emits its own concise progress).
lro.RECORD_FRAME_TOURS = False
lro.VERBOSE = False

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA_VLSI = HERE / "data_vlsi"   # Waterloo VLSI koleksiyonu (indirilenler)
DATA_TSPLIB = HERE / "data_tsplib"  # orijinal TSPLIB95 (Heidelberg/Reinelt)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def _atomic_write_json(path: Path, payload) -> None:
    """Önce .tmp'e yaz + os.replace: koşu tam yazım sırasında öldürülse bile
    sonuç dosyası yarım/bozuk JSON olarak kalmaz (ara-kayıt güvenliği).
    OneDrive eşitlemesi yeni dosyayı kısa süre kilitlediğinde (WinError 5)
    birkaç kez yeniden dener; son çare olarak doğrudan yazar."""
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, separators=(",", ":"))
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.4 * (attempt + 1))
    # kilit çözülmediyse atomiklikten vazgeç, içeriği doğrudan yaz
    path.write_text(text, encoding="utf-8")

import vlsi_datasets as VLSI  # noqa: E402  (katalog + BKS; dosyalar data_vlsi/)
import tsplib_datasets as TSPLIB  # noqa: E402  (orijinal TSPLIB95; data_tsplib/)

ALL_DATASETS = ["kroA100", "rat783", "ali535", "gr666", "pcb3038",
                "fl3795", "fnl4461", "rl5915", "usa13509"]


def dataset_path(name: str) -> Path:
    """Bir veri kümesi adını .tsp dosyasına çözer: önce klasik data/, sonra
    indirilebilir koleksiyon dizinleri (data_vlsi/ data_tsplib/).
    (Ülke ve Sanat koleksiyonları 2026-09-08'de kaldırıldı: makale kapsamı
    dışı, n>100k.)
    Yoksa açıklayıcı hata verir."""
    p = DATA / f"{name}.tsp"
    if p.exists():
        return p
    for cat, cdir, label in ((VLSI, DATA_VLSI, "VLSI"),
                             (TSPLIB, DATA_TSPLIB, "TSPLIB95")):
        pv = cdir / f"{name}.tsp"
        if pv.exists():
            return pv
        if name in cat.CATALOG:
            raise FileNotFoundError(
                f"{name} {label} kataloğunda var ama henüz indirilmemiş "
                f"(dashboard'daki İndir düğmesi veya {cat.url_for(name)})")
    raise FileNotFoundError(f"bilinmeyen veri kümesi: {name}")

# methods presented as an ablation ladder + alternatives.
# The theta-star detectors (rotation_*) run FIRST: they find the best
# strip-sweep angle over the STRIP tour's true cost; the best detector's
# theta*-rotated strip tour becomes the pipeline seed (window repair ->
# line reassignment -> metaheuristics) if it truly beats the theta=0 strip.
# (The old self-invented Snake-Grid construction, its snake-cost sparse-scan
# detector ("rotation") and the pure GPU snake method ("gpu_snake") were
# REMOVED at the user's request -- too slow; the angle is now found FOR the
# strip, ON the strip tour.)
# ---------------------------------------------------------------------------
# YAPI (2026-07-24 yeniden duzenleme, kullanici karari):
#   1) GPU YOK. Bu calisma bir GPU calismasi degil; onarim katmani CPU'da
#      calisir (gpu_snake.repair_tours device="cpu" ile cagrilir -- ayni
#      hamle semantigi, toplu/vektorize degerlendirme). Yontem adlarinda
#      "GPU" gecmez: "Toplu (batched) Onarim".
#   2) TEK CATI: ayni motoru ayni parametrelerle calistirip yalnizca
#      BASLANGIC TURU (tohum) degisen satirlar TEK yonteme katlandi.
#      ge_repair/nn_repair/fi_repair/greedy_snake_v2_repair/adaptive_lr ->
#      hepsi "repair_vnd <- <tohum>"; line_reassign -> "repair_relocate";
#      dense_adaptive_lr -> "repair_vnd_dense"; ils_sss/ils_hyb/ils_cls ->
#      "ils <- <tohum>". Tohum artik admin panelinden SECILIR (SEEDABLE).
#   3) Insa-disi her yontemin girdisi kullanici tarafindan secilir ve
#      sonuclarda "Yontem <- Tohum" olarak GORUNUR (bkz. seed_key/seed_name).
# ---------------------------------------------------------------------------
# YUVA (slot) MEKANIZMASI KALDIRILDI (2026-09-07, kullanici karari):
# "bu yontem havuzunu tamamen sil artik". Bir yontemin bes kopyasini ayri
# anahtarlar olarak tasiyip her birine panelden farkli bir girdi sectirmek
# icin vardi. Yerini COGALTMA (panel "⧉ cogalt", dashboard/server.py
# `_clones`) aldi: kopyalar artik yontem anahtari uretmiyor, tek koşumda
# `--choices` ile tabanin farkli girdileriyle kosuluyor. On anahtar
# REMOVED_METHODS'a girdi ki eski results/*.json satirlari hayalet olarak
# gorunmesin.

METHOD_ORDER = [
    # ---------------------------------------------------------------------
    # 2026-09-07 TEMIZLIK (kullanici karari): havuzlu satirlar, Greedy/Far
    # Snake aileleri, tekil onarim komsuluklari, GLOP ve Quick-Boruvka KODDAN
    # kaldirildi (anahtarlar REMOVED_METHODS'ta; eski results/*.json
    # satirlari ayiklanir). Kalan yuzey = makalenin yontemleri:
    #   klasik kurucular + theta dedektorleri + RGGE/RSGE + CERCEVE DENEYI
    #   (frame_methods.py) + tek onarim satiri + ILS + exact referanslar.
    # ---------------------------------------------------------------------
    # --- INSA (klasik kurucular; tohum OLURLAR) ---
    "nn", "nn_exact", "strip", "hilbert", "morton",
    "farthest_insertion", "greedy_edge",
    # --- theta dedektoru (strip uzerinden aci uretir; RGGE/RSGE aci kaynagi) ---
    "rotation_strip",
    # --- RGGE: dondurulmus izgarada tek-tur greedy-edge (rgge.py) ---
    #     RGGE(theta=0, duz) == greedy_edge@knn8_greedy (bit-ayni capa).
    "rgge_y", "rgge_x",
    # --- RSGE: dondurulmus serpantin greedy-edge, dort bant kurali
    #     (snake_alt.rsge_tour). RSGE(0, b=1) == greedy_edge@knn8_greedy.
    "rsge_resource", "rsge_corridor", "rsge_budget", "rsge_fixed2",
    # --- CERCEVE DENEYI (makale; frame_methods.py) -----------------------
    #     fs_*      : aci taramasi (TANI) -- satir theta=0 turu, savings =
    #                 en kotu - en iyi aci maliyeti
    #     fs_ge8_*  : beraberlik kontrolleri (jitter / beraberliksiz /
    #                 beraberliksiz+kesin k-NN -> yayilim TAM 0)
    #     fs_band_* : bant sayisi taramasi
    #     realign_* : hizasiz kart -> dedektor -> snap -> kurucu; "aci"
    #                 kutusu = kartin HIZASIZLASTIRILDIGI aci (varsayilan 22.5)
    "fs_ge8", "fs_ge15", "fs_nn", "fs_nn_grid", "fs_fi",
    "fs_strip", "fs_hilbert", "fs_morton", "fs_band2ge",
    "fs_ge8_jitter", "fs_ge8_detied", "fs_ge8_exactknn", "fs_band_ge8",
    # 2026-09-09 (hakem A3/E9): beraberliksiz kontrol diger "degismez" kuruculara da
    *FM.DETIED_KEYS,
    "realign_ge8", "realign_strip", "realign_hilbert", "realign_morton",
    # 2026-09-09 (hakem E8): rastgele phi + konum gurultusu + nokta silme
    FM.REALIGN_NOISE_KEY,
    # --- PARALEL GE (k drone; parallel_ge.py): bolumleme stratejisi x k ----
    #     band (cerceveye bagimli, aci panelden) / kmeans / split (donme-
    #     degismez). Satir maliyeti = k kapali turun TOPLAMI; makespan, denge,
    #     paralel sure alanlarda. Amac: k drone ile tarlanin en kisa surede
    #     bitmesi -- tek dikisli tur (RSGE) DEGIL.
    *FM.PGE_KEYS, FM.PGE_SWEEP_KEY, FM.PGE_KMSEED_KEY,
    # --- CERCEVE YARISI: paralel GE + bant basina VND onarimi (pgr_*) --------
    #     Hipotez (kullanici): kafes-hizali bant insada seyrek taramaya kaybetse de
    #     onarim bittiginde kazanir. Aci kutusu: kafes (varsayilan) / seyrek / 0 / 22.5.
    *FM.PGR_KEYS,
    # --- ONARIM ABLASYONU (tohum panelden secilir): uc tekil komsuluk + tam VND ---
    #     Kullanici karari (2026-09-07, 2. tur): "bizim yontemlerde ILS/onarim daha
    #     derine duzenleme yapabiliyor mu?" sorusu icin geri getirildi.
    "repair_two_opt", "repair_or_opt", "repair_relocate", "repair_vnd",
    # --- METASEZGISEL (tohum secilir) ---
    "ils",
    # --- METASEZGISEL (tohum secilir) ---
    # --- EXACT / ALTIN STANDART (kiyas TABANI, rakip DEGIL) ---
    "lkh3", "concorde",
]
# "Exact yontem suresi" kutusunun (panel sol blok) etkiledigi yontemler.
# Kuruculardaki "Insa ust sure siniri" ile ayni role sahiptir: duvar-saati
# ust siniri, varsayilan EXACT_TIME_DEFAULT saniye.
EXACT_METHODS = ["lkh3", "concorde"]
EXACT_TIME_DEFAULT = 300.0
# Insa-disi, girdisi (tohumu) kullanici tarafindan secilebilen yontemler.
# Panelde her birinin yaninda bir tohum kutusu cikar; secilen tohum
# sonuc satirinda "Yontem <- Tohum" olarak gorunur.
SEEDABLE = {"repair_two_opt", "repair_or_opt", "repair_relocate", "repair_vnd", "ils"}
# Tohum olarak secilebilecek yontemler (SEEDABLE yontemlerin girdi havuzu).
# YALNIZ yapicilar degil: bir ablasyon satiri BASKA bir ablasyon satirinin
# ciktisini girdi alabilir (2026-07-24, kullanici karari) -- boylece eski
# sabit zincir (pencere -> LR -> adaptif LR -> yogun LR) artik panelden
# ISTENILEN sirayla, acikca kurulabilir. Dongu (a<-b<-a) calisma aninda
# tespit edilir ve ilgili satir DEFAULT_SEED'e duser (sessiz kalmaz).
SEED_CONSTRUCTIONS = [
    "greedy_edge", "nn", "nn_exact",
    "farthest_insertion", "strip", "rotation_strip", "hilbert", "morton",
    # RSGE/RGGE: onarim katmani bu satirlardan da beslenebilmeli (bant
    # sayisi kurali kazandirir mi sorusu onarim SONRASINDA da sorulmali).
    # Tohum olarak KANONIK satirlarini verirler (bkz. `_row_of`).
    "rsge_resource", "rsge_corridor", "rsge_budget", "rsge_fixed2",
    "rgge_y", "rgge_x",
    # yeniden hizalanmis kartta kurulan turlar da tohum olabilir
    "realign_ge8", "realign_strip",
]
SEED_ABLATIONS = ["repair_two_opt", "repair_or_opt", "repair_relocate", "repair_vnd"]
SEED_POOLS = []
SEED_CHOICES = SEED_CONSTRUCTIONS + SEED_POOLS + SEED_ABLATIONS
# Panel bir tohum belirtmediginde kullanilan varsayilan (ana kiyas kurucusu).
DEFAULT_SEED = "greedy_edge"
# Havuzlu yontemler: tohumlarini TANIMI GEREGI kendileri uretir (varyant
# havuzu), bu yuzden SEEDABLE degildir; havuz kaynagi burada belgelenir.
POOL_SOURCE = {}

# ---------------------------------------------------------------------------
# HAVUZ ONARIMI SECIMI (2026-07-25, kullanici karari)
# ---------------------------------------------------------------------------
# Havuzlu iki satirin ONARIM KATMANI artik sabit degil, panelden SECILIR --
# ablasyonlardaki tohum kutusunun onarim karsiligi. Sebep bir tutarsizlikti:
# havuzlar `gpu_snake.repair_tours` kullaniyordu, yani komsuluk kumesi
# {2-opt, Or-opt<=3}; RELOCATE YOK. Ablasyon tablosunda bunun BIREBIR
# karsiligi YOKTU -- en yakini "repair_vnd" ama o relocate'i de iceriyor.
# Yani makalenin iki amiral satiri, hicbir ablasyon satirinin olcmedigi bir
# onarim gucuyle kosuyordu. Artik onarim acikca secilir ve satirda gorunur.
#
# ADLANDIRMA KURALI (kullanici talebi): literaturde adi olan komsuluklar
# KENDI adiyla ve kaynagiyla yazilir; bu projeye ozgu olanlar acikca
# "BIZIM YONTEM" diye isaretlenir. Toplu (batched) uygulama literatur
# komsuluklarinin bizim es-zamanli kosumumuzdur -- komsuluk literatur,
# kosum bicimi bizim; etiket bunu ayirir.
POOL_REPAIR_SELECTABLE = set()
DEFAULT_POOL_REPAIR = "batch_2opt_oropt"   # mevcut/tarihsel davranis
POOL_REPAIR_CHOICES = [
    "batch_2opt_oropt",   # gpu_snake.repair_tours -- havuzun TAMAMI tek batch
    "two_opt", "or_opt", "relocate", "vnd",     # repair.run_single_neighborhood
    "vnd_dense",                                # lro.run_neighborhood_vnd (genis)
    "window", "window_adaptive",                # core.*WindowRepair
]
POOL_REPAIR_LABELS = {
    "batch_2opt_oropt":
        "2-opt + Or-opt(≤3), havuz eş-zamanlı (bizim toplu uygulamamız)",
    "two_opt":         "yalnız 2-opt (Croes 1958)",
    "or_opt":          "yalnız Or-opt, segment ≤3 (Or 1976)",
    "relocate":        "yalnız Relocate / tek-nokta taşıma (klasik komşuluk)",
    "vnd":             "tam VND: 2-opt + Or-opt + Relocate (Mladenović & Hansen 1997)",
    "vnd_dense":       "genişletilmiş VND (Or-opt≤6 + yoğun toplu) — BİZİM YÖNTEM",
    "window":          "sabit-uçlu geometrik pencere onarımı (pencere=10) — BİZİM YÖNTEM",
    "window_adaptive": "uyarlanır döngüsel pencere onarımı (10/12 + yeniden kuyruk) — BİZİM YÖNTEM",
}
# Secilen onarimin ABLASYON TABLOSUNDAKI muadili (varsa). Panel bunu
# gosterir ki okuyucu "bu havuz hangi ablasyon satiriyla ayni onarim
# gucunde kosuyor?" sorusunu tek bakista cevaplasin. None = muadili YOK.
POOL_REPAIR_ABLATION = {
    "batch_2opt_oropt": None,   # {2-opt, Or-opt} -- relocate'siz VND; ablasyonda YOK
    "two_opt": "repair_two_opt",
    "or_opt": "repair_or_opt",
    "relocate": "repair_relocate",
    "vnd": "repair_vnd",
    "vnd_dense": "repair_vnd_dense",
    "window": "repair_window",
    "window_adaptive": "repair_window_adaptive",
}

# ---------------------------------------------------------------------------
# ACI (theta) SECIMI -- yalniz greedy_snake_v1 icin.
# v1 tek acida tek tur kurar; o acinin NEREDEN geldigi panelden secilir
# (ablasyonlardaki tohum kutusunun aci karsiligi). Secim sonuc satirina
# angle_key/angle_name olarak yazilir ve gosterim adina "← <aci kaynagi>"
# olarak eklenir.
#
# !! KIYAS UYARISI !! Varsayilan secim, v2/v3'un kullandigi vekilin BIREBIR
# AYNISIdir (ikisi de `snake_alt.theta_proxy`den gecer); v1 ⊂ v2 ⊂ v3 ic-ice
# gecmesini ve dolayisiyla maliyet(v1) >= maliyet(v2) >= maliyet(v3)
# garantisini SAGLAYAN tek secenektir. Baska bir kaynak secilirse satir bir
# ACI-DUYARLILIGI deneyidir, ailenin v1 basamagi DEGILDIR (26 ornekte
# olculdu: dedektor acisiyla 22 ornekte aci v2'nin aday kumesi disinda kaldi,
# 2 ornekte v1 v2'yi gecti). Panel bunu uyari olarak gosterir.
#
# 2026-07-28: VARSAYILAN "strip_oracle" -> "grid_theta" olarak degisti.
# Gerekce IZGARA_ACISI_VE_SERIT_SAYISI_AKADEMIK_NOT.md'de olculmustur:
# kanitlanmis optimum LKH turlarina karsi grid_theta 6/6 sette 0.05-0.43
# derece hata verirken strip-oracle 3/6 sette 10-30 derece sapiyor; 8 buyuk
# kanitlanmis-optimum VLSI kumesinde v1 cekirdegi %18.23 -> %16.92 ve daha
# ucuz. "strip_oracle" secenegi KALDI -- artik yeni varsayilana karsi
# aci-duyarliligi satiri olarak kosuluyor, yani eski davranis olculebilir
# kaliyor.
# far_snake_v1 de buradadir: aci makinesi greedy_snake_v1 ile BIREBIR ayni
# oldugundan aci-duyarliligi deneyi iki ailede de AYNI bicimde kosulabilmeli
# (aksi halde "aci secimi ne kadar onemli?" sorusu yalniz bir aile icin
# cevaplanmis olur ve merdiven uyarisi Far Snake tarafinda gorunmez).
#: RSGE satirlari -> bant sayisi kurali (snake_alt.RSGE_RULES). TEK KAYNAK:
#: kosum blogu, panel etiketi ve olcum betikleri buradan okur.
RSGE_RULE_OF = {"rsge_resource": "resource", "rsge_corridor": "corridor",
                "rsge_budget": "budget", "rsge_fixed2": "fixed"}
RSGE_METHODS = tuple(RSGE_RULE_OF)

#: RGGE satirlari -> k-NN izgarasinin eksen sirasi DEVRIK mi? TEK KAYNAK.
RGGE_DEVRIK_OF = {"rgge_y": True, "rgge_x": False}
RGGE_METHODS = tuple(RGGE_DEVRIK_OF)

# RSGE dortlusu ve RGGE ikilisi de aci-secilebilir.
#
# 2026-09-07 (KULLANICI KARARI -- ONCEKI DAVRANIS DEGISTI):
# Eskiden bu iki aile her kosumda UC satir uretiyordu -- kanonik (grid_theta)
# + @zero + panelde secili aci. Yani panelde "Strip Seyrek Tarama" secili
# olsa bile tabloda gorunen ana satir grid_theta ile kosuluyordu ve
# SECILMEYEN acilarda da tarama yapiliyordu. Olculdu (108 kume): grid_theta
# 106 kumenin 98'inde tam 0.0 donuyor (eksen-hizali VLSI kafesleri; guven
# ~5.0 ile DOGRU cevap), dolayisiyla kanonik RGGE satiri 99/106 kumede
# greedy_edge@knn8 ile BIREBIR AYNI maliyeti veriyordu -- yani panel secimi
# tabloda hic gorunmuyordu.
#
# ARTIK: her iki aile de YALNIZ panelde secili acida kosar (tek satir, tek
# aci). Bunun icin "varsayilan secim" bu ailelerde sabit DEGIL, PANELDE
# SECILI olandir (bkz. ANGLE_PANEL_CANONICAL + default_choice_of) -- boylece
# tek satir daima temiz anahtarini (`rgge_x`) tasir, onarim tohumu zinciri
# (`_row_of`) kopmaz ve sunucunun "bu kume hazir mi" kontrolu (row_id ile)
# ayni satiri bulur.
#
# BUNUN BEDELI: `@zero` capa satiri (RGGE(0,duz) == greedy_edge@knn8_greedy)
# ARTIK OTOMATIK URETILMEZ. Capayi gormek icin panelden "θ = 0° (hizalamasiz
# taban cizgisi)" secilmelidir. greedy_snake_v1 / far_snake_v1 ESKI
# davranisi korur (merdiven ablasyonlari onlara bagli).
#: Yeniden-hizalama satirlari (frame_methods.REALIGN_KEYS). Bu satirlarda
#: panelin "aci" kutusu KARTIN HIZASIZLASTIRILDIGI aciyi (phi) secer;
#: varsayilan elle girilen 22.5 derece.
REALIGN_METHODS = tuple(FM.REALIGN_KEYS)
DEFAULT_ANGLE_OF = {k: f"manual:{FM.DEFAULT_MISALIGN_DEG:g}" for k in REALIGN_METHODS}
DEFAULT_ANGLE_OF[FM.REALIGN_NOISE_KEY] = "random"     # gurultu ablasyonu: phi rastgele
#: RGGE/RSGE (2026-09-07, 2. tur): kanonik aci = rotation_strip (strip-oracle).
#: Eski "kanonik = panelde secili" kurali (ANGLE_PANEL_CANONICAL) COGALTMAYI
#: bozuyordu: `--choices rgge_x=zero` ile kosulan satir da temiz anahtari
#: aliyor ve kanonik satirin USTUNE yaziyordu. Artik diger aci-secilebilir
#: yontemlerle ayni kural: varsayilan disi secim `@<secim>` sonekiyle YANINA duser.
# RGGE (donme-degismez GE): kanonik aci = strip-oracle (rotation_strip) -- SIFIRDAN
# FARKLI gercek bir aci; @zero kopyasiyla farki "rotasyon = beraberlik gurultusu"
# olcumudur (grid_theta VLSI'da hep 0 verir, satir GE'nin kopyasi olurdu).
# RSGE (bantli, cerceveye BAGIMLI): kanonik aci = KAFES yonu (grid_theta); strip
# oracle acisi ve theta=0 kopya olarak yaninda durur.
DEFAULT_ANGLE_OF.update({k: "rotation_strip" for k in RGGE_METHODS})
DEFAULT_ANGLE_OF.update({k: "grid_theta" for k in RSGE_METHODS})
#: Paralel GE bant bolumlemesi de aci-secilebilir (bantlarin yonu = cerceve).
PGE_BAND_METHODS = tuple(FM.PGE_BAND_KEYS)
# Bant hizasi KAFES yonunden gelir (grid_theta: tarak dedektoru, guven altinda
# strip-oracle'a duser). rotation_strip strip turunun en iyi acisidir, kafes
# hizasi DEGIL: xqf131'de 16 derece verip bantlari kafesle hizasiz kesti
# (toplam %34.6 <-> theta=0'da %24.1).
PGR_BAND_METHODS = tuple(FM.PGR_BAND_KEYS)
DEFAULT_ANGLE_OF.update({k: "grid_theta" for k in PGE_BAND_METHODS + PGR_BAND_METHODS})
ANGLE_SELECTABLE = (set(RSGE_METHODS) | set(RGGE_METHODS) | set(REALIGN_METHODS)
                    | set(PGE_BAND_METHODS) | set(PGR_BAND_METHODS) | {FM.REALIGN_NOISE_KEY})

#: Aci "varsayilani" PANELDEN gelen aileler -- yalniz secili acida kosarlar.
ANGLE_PANEL_CANONICAL: set = set()   # 2026-09-07: kaldirildi (bkz. DEFAULT_ANGLE_OF notu)
DEFAULT_ANGLE = "grid_theta"
ANGLE_CHOICES = [
    "grid_theta",       # grid_theta.theta_for -- AILE VARSAYILANI (2026-07-28)
    "strip_oracle",     # snake_alt._strip_oracle_theta -- eski aile varsayilani
    "theta_star",       # en iyi skorlu dedektor (runner mirasi)
    "rotation_strip",   # theta* Rotasyon — Strip Seyrek Tarama
    "rotation_hist",    # theta* Rotasyon — Yon Histogrami
    "rotation_mean",    # theta* Rotasyon — Acisal Ortalama
    "rotation_pca",     # theta* Rotasyon — PCA Ozbekseni
    "manual",           # ELLE girilen sabit aci (varsayilan -90 derece)
    "zero",             # theta = 0 (hizalamasiz taban cizgisi)
    "random",           # 2026-09-09 (hakem E8): U(-45,45), ornek basina deterministik (tohum n)
]
ANGLE_LABELS = {
    "grid_theta": "Izgara açısı (tarak taraması — aile varsayılanı)",
    "strip_oracle": "Strip-oracle (kaba 10° tarama — eski varsayılan)",
    "theta_star": "θ★ (en iyi skorlu dedektör)",
    "rotation_strip": "θ★ Rotasyon — Strip Seyrek Tarama",
    "rotation_hist": "θ★ Rotasyon — Yön Histogramı",
    "rotation_mean": "θ★ Rotasyon — Açısal Ortalama",
    "rotation_pca": "θ★ Rotasyon — PCA Özekseni",
    "manual": "Elle girilen açı (θ panelden yazılır)",
    "random": "Rastgele hizasızlık φ ~ U(−45°, 45°) (örnek başına sabit tohum)",
    "zero": "θ = 0° (hizalamasız taban çizgisi)",
}

# ---------------------------------------------------------------------------
# ELLE GIRILEN ACI (2026-07-30, kullanici talebi)
# ---------------------------------------------------------------------------
# Yukaridaki seceneklerin HEPSI aciyi VERIDEN kestirir (izgara taramasi, strip
# oracle, rotasyon dedektorleri). Kullanicinin sorusu bunlarin hicbirinin
# cevaplamadigi bir soruydu: "acinin kendisi hipotezimse ne olacak?" -- yani
# dedektore hic sormadan SABIT bir aciyla kosmak. Bu secenek tam olarak onu
# yapar: `theta_deg` dogrudan v1 kurucusuna gecer, hicbir kestirici kosmaz.
#
# VARSAYILAN -90 derecedir (kullanici hipotezi). Aci [-90, 90) araligina
# indirgenMEZ: serit yonu 180 derece periyodik oldugundan -90 ile +90 ayni
# serit ailesini verir, ama boustrophedon GEZINME yonu ters doner; kullanici
# "-90" yazdiysa satirda -90 gorunmeli, sessizce +90'a cevrilmemeli.
#
# SECIM ANAHTARI aciyi KENDI ICINDE tasir: "manual:-90", "manual:37.5" ...
# Sebep row_id sistemidir (bkz. `row_id`): satir kimligi secim anahtarindan
# turedigi icin, aci anahtarin disinda tutulsaydi -45 ile kosulan satir -90
# ile kosulanin USTUNE yazardi. Boylece farkli acilar YAN YANA durur:
#     greedy_snake_v1@manual:-90   ve   greedy_snake_v1@manual:45
# Cıplak "manual" anahtari varsayilanin (=-90) takma adidir ve
# `resolve_angle_choice` onu kanonik "manual:-90" bicimine cevirir.
#
# !! KIYAS UYARISI !! Elle girilen aci da DEFAULT_ANGLE degildir, yani bu
# satir bir ACI-DUYARLILIGI deneyidir: v1 ⊂ v2 ⊂ v3 merdiveni GARANTI DEGIL
# (kanonik v1 satiri her kosuda ayrica uretilmeye devam eder).
MANUAL_ANGLE_PREFIX = "manual"
DEFAULT_MANUAL_ANGLE = -90.0


def manual_angle_deg(choice: "str | None") -> "float | None":
    """Elle-aci secim anahtarindaki dereceyi dondurur, degilse None.

        "manual"        -> DEFAULT_MANUAL_ANGLE (-90.0)
        "manual:-90"    -> -90.0
        "manual:37.5"   ->  37.5
        "grid_theta"    -> None   (elle-aci secimi degil)
    """
    if not isinstance(choice, str):
        return None
    if choice == MANUAL_ANGLE_PREFIX:
        return DEFAULT_MANUAL_ANGLE
    pre, sep, val = choice.partition(":")
    if pre != MANUAL_ANGLE_PREFIX or not sep:
        return None
    try:
        deg = float(val)
    except (TypeError, ValueError):
        return None
    return deg if math.isfinite(deg) else None


def manual_angle_choice(deg: float) -> str:
    """Dereceden KANONIK secim anahtari: -90.0 -> "manual:-90"."""
    return f"{MANUAL_ANGLE_PREFIX}:{float(deg):g}"


def is_angle_choice(choice: "str | None") -> bool:
    """Panelden gelen bir degerin gecerli aci secimi olup olmadigi.
    ANGLE_CHOICES uyeligi TEK BASINA yetmez: elle girilen aci anahtarlari
    ("manual:-90") listede sabit olarak duramaz, cunku sonsuz coklukta."""
    return choice in ANGLE_CHOICES or manual_angle_deg(choice) is not None


def angle_label(choice: "str | None") -> str:
    """Secimin gosterim adi. Elle girilen acida DERECE etikete yazilir --
    tabloda "← Elle girilen açı (θ = -90°)" gorunur, yani hangi aciyla
    kosuldugu satirin adindan okunur."""
    deg = manual_angle_deg(choice)
    if deg is not None:
        return f"Elle girilen açı (θ = {deg:g}°)"
    return ANGLE_LABELS.get(choice, choice)

# ---------------------------------------------------------------------------
# GREEDY-EDGE DUYARLILIK ABLASYONU (2026-07-26) -- yalniz greedy_edge icin.
# ---------------------------------------------------------------------------
# Itiraz: "rakip Greedy-Edge satiri haksiz guclu -- k-NN budamasi ona uzamsal
# zeka veriyor, aday kenarlar bitince yaptigi akilli uc-eslestirme ikinci bir
# silah, ve arkasinda derlenmis C kodu var."
#
# Uc iddia da olculdu (bkz. tsplib_engine.greedy_edge_ablation_tour docstring
# ustundeki olcum tablosu) ve DOGRULANMADI. Rakibi zayiflatmak yerine itirazi
# VERIYLE kapatiyoruz: her karsi-olgusal varyant panelden secilerek ASIL
# SATIRIN YANINA kosar (row_id sistemi: greedy_edge@full_greedy vb.), boylece
# makalede "baseline sisirilmedi" iddiasi olcumle desteklenir.
#
# !! KIYAS UYARISI !! Varsayilan DISI her secim BILEREK ZAYIFLATILMIS ya da
# BILEREK YAVASLATILMIS bir Greedy-Edge'dir; RAKIP SATIRI DEGILDIR, bir
# duyarlilik deneyidir. Ozellikle knn15_random, Johnson & McGeoch'un yayinlanan
# Greedy degerlerinin ~5x disina cikar -- rakip olarak raporlanamaz.
GE_VARIANT_SELECTABLE = {"greedy_edge"}
DEFAULT_GE_VARIANT = "knn15_greedy"          # mevcut/tarihsel rakip satiri
GE_VARIANT_CHOICES = [
    "knn15_greedy",   # k=15 + kanonik dikis -- RAKIP SATIRIN KENDISI
    "knn8_greedy",    # k=8: Greedy Snake bant kurucusuyla ayni aday genisligi
    "knn3_greedy",    # k=3: "aday listesini bant boyutuna cek" itirazinin ucu
    "full_greedy",    # k-NN YOK, tum n(n-1)/2 kenar (n<=6000)
    "knn15_random",   # akilli dikis KAPALI -> rastgele uc baglama
]
GE_VARIANT_LABELS = {
    "knn15_greedy":
        "k=15 aday listesi + kanonik dikiş (Johnson & McGeoch 1997 — RAKİP SATIR)",
    "knn8_greedy":
        "k=8 aday listesi — Greedy Snake bant kurucusuyla eşitlenmiş (duyarlılık)",
    "knn3_greedy":
        "k=3 dar aday listesi — kısıtlanmış Greedy (duyarlılık)",
    "full_greedy":
        "k-NN YOK: budanmamış O(n²) tüm kenarlar — saf Greedy (n≤6000, duyarlılık)",
    "knn15_random":
        "k=15 + AKILLI DİKİŞ KAPALI (rastgele uç bağlama) — kukla taban çizgisi",
}
# Varsayilan disi secimlerin satira dusen acik uyarisi (panel + sonuc satiri).
GE_VARIANT_WARNING = {
    "knn15_greedy": None,
    "knn8_greedy": "Aday genişliği Greedy Snake bandıyla eşitlendi — duyarlılık deneyi.",
    "knn3_greedy": "Bilerek kısıtlanmış Greedy — literatür rakibi DEĞİLDİR.",
    "full_greedy": "Budanmamış O(n²) Greedy — kalite ~aynı, süre ~30× (ölçüldü).",
    "knn15_random": "Dikiş rastgeleleştirildi — literatür Greedy'sinin ~5× dışında, "
                    "RAKİP OLARAK RAPORLANAMAZ.",
}
# Bu koddan KALDIRILAN eski yontem anahtarlari: diskte onceki kosumlardan
# kalan satirlar merge sirasinda ve sunucu tarafinda ayiklanir, boylece
# dashboard/istatistik hicbir zaman artik var olmayan bir yontemi gostermez.
# snake_best_insertion_band / gpu_snake_best_insertion_band: iki insaati
# (greedy + farthest) her bantta calistirip yerel-kisa olani tutan "en iyisini
# sec" hibritleriydi -- yapisi geregi kendi kardesleri snake_greedy_band /
# snake_farthest_band'i HICBIR ZAMAN kaybetmiyordu, bu da istatistiklerde
# (Sinif-Ici Karsilastirma, Kazanma%) her kosuda "1. cikan" sahte bir kazanan
# yaratiyordu (bkz. LEE'nin ayni sebeple ayri gruba tasindigi onceki duzeltme
# -- burada kok neden ayni oldugu icin yontem dogrudan KALDIRILDI, ayri
# gruba almak yeterli olmazdi cunku bant-hibrit ailesinin ICINDE karsilastirma
# yapiliyor). Kullanicinin acik talebiyle kaldirildi (2026-07-12).
REMOVED_METHODS = {"snake", "rotation", "gpu_snake",
                    # --- 2026-09-07 (2. tur): makalede kullanilmayanlar ---
                    "rotation_hist",
                    "rotation_mean",
                    "nearest_insertion",
                    # --- 2026-09-07 TEMIZLIK (kullanici karari): makale
                    #     yuzeyi disindaki her sey koddan kaldirildi ---
                    "quick_boruvka",
                    "rotation_pca",
                    "greedy_snake_v1",
                    "greedy_snake_v2",
                    "greedy_snake_v4_2x2",
                    "far_snake_v1",
                    "far_snake_v2",
                    "ge_pool2",
                    "ge_pool2_jitter",
                    "ge_pool2_yon",
                    "knn_pool2",
                    "knn_pool2_uyarlanir",
                    "serpantin_aci2",
                    "strip_theta_ge",
                    "fi_pool2",
                    "fi_band_auto",
                    "repair_vnd_dense",
                    "repair_window",
                    "repair_window_adaptive",
                    "greedy_snake_v2_repair",
                    "greedy_snake_v4_2x2_repair",
                    "far_snake_v2_repair",
                    "ge_pool2_repair",
                    "fi_pool2_repair",
                    "glop_like",
                    # --- 2026-09-07 kullanici karari: YUVA mekanizmasi
                    #     tamamen kaldirildi ("bu yontem havuzunu tamamen
                    #     sil artik"). Anahtarlar buraya girer ki eski
                    #     results/*.json dosyalarindaki yuva satirlari
                    #     tabloda HAYALET olarak gorunmesin. Yerini panel
                    #     COGALTMA'si aldi (bkz. server.py `_clones`).
                    "greedy_snake_v1_y1", "greedy_snake_v1_y2",
                    "greedy_snake_v1_y3", "greedy_snake_v1_y4",
                    "greedy_snake_v1_y5",
                    "repair_vnd_y1", "repair_vnd_y2", "repair_vnd_y3",
                    "repair_vnd_y4", "repair_vnd_y5",
                    "snake_best_insertion_band", "gpu_snake_best_insertion_band",
                    # makale kapsami disina cikarilanlar (2026-07-22, kullanici
                    # karari): GNN/NGLS ailesi ve LEE polish
                    "gnn", "gnn_trained", "ngls", "gnn_gpu", "gnn_free_gpu",
                    "ngls_gpu", "lee",
                    # NOT (2026-07-25): "lkh3"/"concorde" ARTIK KALDIRILMIS
                    # DEGIL -- external_solvers.py koprusu eklendi ve ikisi de
                    # METHOD_ORDER'in sonuna EXACT grubu olarak girdi. Eski
                    # kosumlardan kalan lkh3 satirlari onceki merge'lerde zaten
                    # temizlendi (diskte 0 satir kaldigi dogrulandi), bu yuzden
                    # ayrica bir eskime filtresine gerek yok.
                    # --- 2026-07-24 yeniden duzenleme (kullanici karari) ---
                    # (a) GPU satirlari: bu calisma GPU calismasi degil
                    "gpu_snake_greedy_band", "gpu_snake_farthest_band",
                    "gpu_snake_stratified_subsample",
                    # (c) 2026-07-27 kullanici karari: snake_stratified_
                    #     subsample benchmark YUZEYINDEN cikarildi. Ilkel bir
                    #     model (yogunluk-orantili orneklem + Snake-Grid
                    #     aramasi); makalede yer almayacak. greedy_snake_v3'un
                    #     havuzuna ZARAR VERMEZ: 17 kumede olculdu, havuzu
                    #     HIC kazanmiyor (en iyi sirasi 2., cogunlukla 9-10.
                    #     ya da ilk 10'a hic giremiyor) ve cikarilinca v3
                    #     ort. %+0.000 degisiyor -- 0 kumede kotulesme.
                    #     NOT: snake_alt.snake_stratified_subsample_tour
                    #     fonksiyonu DURUYOR (measurements_jpdc/ ve
                    #     verification/ altindaki 10 tarihsel betik onu
                    #     import ediyor); kaldirilan sey BENCHMARK SATIRI.
                    "snake_stratified_subsample",
                    # (d) 2026-07-28 kullanici karari: snake_greedy_band
                    #     benchmark YUZEYINDEN cikarildi -- "buna ihtiyacimiz
                    #     yok". Zaten Greedy Snake v2'nin TA KENDISIYDI, tek
                    #     fark mult=0.15 (v2'de 0.05); ortak govde
                    #     snake_alt.snake_v2_pool. Ayri bir satir olarak
                    #     tasidigi bilgi, v2'nin mult ekseninde zaten var.
                    #     greedy_snake_v3'un havuzuna maliyeti OLCULDU
                    #     (_v3_extra_tours'tan cikarma testi, 9 kume:
                    #     xqf131/pma343/bcl380/dcb2086/beg3293/pcb3038/
                    #     fl3795/bgb4355/xqe3891): 9/9'da fark 0.000%,
                    #     toplamda 0 -- yani mult=0.15 turu havuzun EN IYISI
                    #     hicbir kumede olmuyordu. Kaldirma BEDELSIZ.
                    #     NOT: snake_alt.snake_greedy_band_tour / _pool
                    #     fonksiyonlari DURUYOR (benchmark/diag_snake.py,
                    #     verify_snake_alt2.py ve measurements_jpdc/ altindaki
                    #     tarihsel betikler onlari import ediyor); kaldirilan
                    #     sey BENCHMARK SATIRI -- snake_stratified_subsample
                    #     ile ayni sozlesme.
                    "snake_greedy_band",
                    # (d2) 2026-07-28, hakem talebi: "snake_farthest_band"
                    #     FAR SNAKE ailesinin v2 basamagi oldu ve
                    #     `far_snake_v2` adiyla METHOD_ORDER'da duruyor --
                    #     yani yontem KALDIRILMADI, YENIDEN ADLANDIRILDI.
                    #     Eski ANAHTAR yine de buraya girer (takma-ad
                    #     DEGIL): tek bir davranis farki var, yeni satir
                    #     `extra_thetas` almiyor (5 aday -> 4 aday,
                    #     greedy_snake_v2 ile esit genislik). Takma-ad
                    #     devri diskteki 5-adayli eski sonucu 4-aday
                    #     etiketli yeni satira miras birakirdi; bu yanlis
                    #     olurdu. Boylece eski satirlar gorunumden
                    #     ayiklanir ve far_snake_v2 her kosumda KENDI
                    #     ayarlariyla yeniden uretilir.
                    #     snake_alt.snake_farthest_band_tour / _pool
                    #     FONKSIYONLARI DURUYOR (GPU kardesler ve dogrulama
                    #     betikleri onlari import ediyor; far_snake_v2_*'in
                    #     takma adi, bit-ayni).
                    "snake_farthest_band",
                    "gpu_hybrid", "gpu_hybrid_v2", "ge_gpu",
                    # (b) TEK CATIYA katlananlar: ayni motor + ayni butce,
                    #     tek fark tohum -> "repair_vnd <- <tohum>" oldu
                    "ge_repair", "nn_repair", "fi_repair", "adaptive_lr",
                    # NOT: "greedy_snake_v2_repair" 2026-07-27'de GERI GELDI ama
                    # BASKA bir anlamla: eskisi tek v2 turunu onaran satirdi
                    # (tek cati altina katlandi); yenisi v2'nin 4-ADAYLI
                    # HAVUZUNU ge_pool2_repair ile ayni onarim katmanindan
                    # geciren havuz satiridir. Diskte eski anlamda satir
                    # KALMADIGI dogrulandi (0 kume), cakisma yok.
                    #     tek-nokta relocate -> "repair_relocate" ile OZDES
                    "line_reassign",
                    #     genisletilmis VND -> "repair_vnd_dense" olarak
                    #     yeniden adlandirildi
                    "dense_adaptive_lr",
                    #     ILS ailesi -> tek "ils <- <tohum>" oldu
                    "ils_sss", "ils_hyb", "ils_cls",
                    # (c) pencere-tabanli onarim ONARIM CATISINA TASINDI
                    #     (2026-07-24, kullanici karari): eski anahtarlar
                    #     "repair_window" / "repair_window_adaptive" oldu,
                    #     tohumlari artik panelden secilir. Eski satirlar
                    #     ayiklanir ki yeni anahtarlarla karismasin.
                    "window_repair", "adaptive_repair",
                    # (d) birlesik havuz: min(GE-havuz, SP-v3) OZDESLIGI
                    #     oldugu 100 sette dogrulandi -- bagimsiz bir deney
                    #     degil, hesaplanabilir bir sutun
                    "greedy_snake_v3_birlesik", "greedy_snake_v3j_birlesik",
                    "greedy_snake_v3j_repair",
                    # (e) lite merdiveni: havuz genisligi ablasyonu kapsam disi
                    "snake_v2_lite1", "snake_v2_lite2", "snake_v2_mid",
                    # --- (f) 2026-07-31, kullanici karari (TRUBA kosumu
                    #     oncesi): 10 ADAYLI HAVUZ SATIRLARININ TAMAMI. Kalan
                    #     yuzey tek havuz genisligindedir (nominal 4 aday);
                    #     bkz. METHOD_ORDER'daki matris notu.
                    #
                    #     NEDEN BEDELSIZ (olculdu): kaldirma anindaki 177
                    #     sonuc dosyasinda bu 10 anahtarin TOPLAM satir sayisi
                    #     0'di -- panelden zaten gizlenmislerdi
                    #     (dashboard/methods_config.json) ve haftalardir
                    #     kosulmuyorlardi. Yani hicbir olcum kaybedilmedi.
                    #
                    #     NEDEN REMOVED, LEGACY_ALIASES DEGIL: 4'lu satirlar
                    #     10'lunun ONEK ALT KUMESIdir (ge_pool2/fi_pool2) ya
                    #     da HIC alt kumesi degildir (v4_4, bkz. ±5° notu) --
                    #     iki durumda da 10-adayla olculmus bir sonucu 4-aday
                    #     etiketli satira miras birakmak YANLIS olurdu.
                    #
                    #     KAPSAM DISI KALAN: "havuzu 4'ten 10'a genisletmek ne
                    #     kazandiriyor?" ablasyonu. Bu bilincli bir kabuldur;
                    #     geri istenirse git b90409a'da tam govdesiyle durur.
                    #     snake_alt tarafindaki URETICI fonksiyonlar
                    #     (snake_v3_pool, far_snake_v3_pool, ge_pool_tours,
                    #     snake_v4_angle_pool(k=10)) DURUYOR -- kaldirilan sey
                    #     BENCHMARK SATIRI, snake_greedy_band ile ayni
                    #     sozlesme.
                    "greedy_snake_v3", "greedy_snake_v3_repair",
                    "greedy_snake_v4", "greedy_snake_v4_repair",
                    "far_snake_v3", "far_snake_v3_repair",
                    "ge_pool", "ge_pool_repair",
                    "fi_pool", "fi_pool_repair",
                    # --- (g) 2026-07-31, kullanici karari: IKI HAVUZ 2 ADAYA
                    #     INDIRILDI ve anahtarlari YENIDEN ADLANDIRILDI:
                    #         greedy_snake_v4_4   -> greedy_snake_v4_2x2
                    #                                (yalniz {-90°, 0°} sondalari;
                    #                                 ±5° ince tarama KALDIRILDI)
                    #         ge_pool4            -> ge_pool2
                    #                                (saf jitter -> k taramasi:
                    #                                 k=8 ve k=15)
                    #     ...ve onarimli kardesleri.
                    #
                    #     NEDEN TAKMA-AD DEGIL, REMOVED: diskte bu anahtarlarin
                    #     167-172 satirlik OLCUMU VARDI ve o satirlar 4 ADAYLA
                    #     uretilmisti. Takma-ad devri onlari "2 aday" etiketli
                    #     yeni satirin yerine koyardi -- tabloda 4-adayla
                    #     olculmus ama 2-aday diye raporlanan satirlar dururdu.
                    #     Ustelik ge_pool tarafinda uretici de degisti (jitter
                    #     -> k taramasi), yani eski satirlar ayni deneyin dar
                    #     hali bile degil, BASKA bir deneydir.
                    #
                    #     ADIN "_4"/"4" olarak kalmasi da yanlis olurdu: bu
                    #     projede havuz adindaki sayi NOMINAL GENISLIKTIR ve
                    #     butce payi (_share = butce/genislik) dogrudan ondan
                    #     hesaplanir.
                    "greedy_snake_v4_4", "greedy_snake_v4_4_repair",
                    "ge_pool4", "ge_pool4_repair",
                    # --- (h) 2026-07-31, kullanici karari (olcume dayali):
                    #     v4 havuzuna IKINCI eksen eklendi -- bant-ici aday
                    #     genisligi k in {4, 16}. Havuz artik 2 aci x 2 k = 4
                    #     adaydir ve anahtar greedy_snake_v4_2x2 oldu.
                    #     Eski greedy_snake_v4_2 anahtari diskte 94 satirla
                    #     duruyordu ve o satirlar 2 ADAYLA (yalniz aci
                    #     ekseni, sabit k=8) uretilmisti -- ayni anahtarda
                    #     birakmak 2-adayla olculmus satirlari 4-aday etiketi
                    #     altinda raporlamak olurdu.
                    #     OLCULDU (23 kume, BKS'ye ort. gap): 17.29 -> 15.30,
                    #     18 kumede iyilesme / 4 kumede kotulesme.
                    "greedy_snake_v4_2", "greedy_snake_v4_2_repair",
                    # --- (i) 2026-08-01, kullanici karari (KIYAS ADALETI):
                    #     TUM havuzlar en fazla 2 ADAY kurar. fi_pool4 ->
                    #     fi_pool2 (4 -> 2 baslangic dugumu). Diskteki
                    #     satirlar 4 ADAYLA uretilmisti; ayni anahtarda
                    #     birakmak 4-adayli sonucu 2-aday etiketiyle
                    #     raporlamak olurdu.
                    "fi_pool4", "fi_pool4_repair",
                    # gpu_snake_v2: eskiden greedy_snake_v3_repair'in takma
                    # adiydi (LEGACY_ALIASES). Hedef anahtar (f)'de
                    # kaldirildigi icin takma-ad devri ARTIK ANLAMSIZ: eski
                    # satiri kaldirilmis bir anahtara miras birakirdi ve
                    # ayiklama filtresinden SONRA calistigi icin diskte hayalet
                    # bir satir birakirdi. Takma-ad kaldirildi, anahtar buraya
                    # alindi.
                    "gpu_snake_v2"}
# Resmi isimlendirme (2026-07-23, kullanici karari): eski anahtarlar
# REMOVED'a GIRMEZ -- ayiklama degil, takma-ad. Diskteki eski satir korunur
# VE koşu başı snapshot'ında yeni resmi anahtara devredilir (yeni anahtar
# eski sonucu miras alir; her iki satir da dosyada kalir).
LEGACY_ALIASES = {"snake_v2": "greedy_snake_v2",
                  "snake_v2_repair": "greedy_snake_v2_repair"}
#
# NEDEN "snake_farthest_band" BURADA DEGIL, REMOVED_METHODS'TA (2026-07-28):
# O satir far_snake_v2 adini aldi ve uretim govdesi ayni kaldi -- ama TEK bir
# davranis farki var: artik `extra_thetas` ALMIYOR, yani aday sayisi 5'ten
# 4'e indi (greedy_snake_v2 ile esitlendi, bkz. insa dagitimi).
#
# Takma-ad devri "eski satirin sonucunu yeni anahtara MIRAS BIRAK" demektir;
# burada bu YANLIS olurdu: diskteki eski satir 5 adayla uretilmis bir sonuctur
# ve resume onu "far_snake_v2 hazir" sayip 4-adayli satirin yerine koyardi --
# yani tabloda 4-aday etiketli ama 5-adayla olculmus bir satir dururdu.
# Bu yuzden eski anahtar REMOVED_METHODS'a girer: eski satir gorunumden
# ayiklanir, far_snake_v2 her kosumda KENDI ayarlariyla yeniden uretilir.
STAGE = {
    # --- insa ---
    "nn": "construction", "strip": "construction",
    "hilbert": "construction", "morton": "construction",
    "farthest_insertion": "construction",
    "greedy_edge": "construction",
    "rotation_strip": "construction",
    "rsge_resource": "construction", "rsge_corridor": "construction",
    "rsge_budget": "construction", "rsge_fixed2": "construction",
    "rgge_y": "construction", "rgge_x": "construction",
    "nn_exact": "construction",
    "fs_ge8": "construction",
    "fs_ge15": "construction",
    "fs_nn": "construction",
    "fs_nn_grid": "construction",
    "fs_fi": "construction",
    "fs_strip": "construction",
    "fs_hilbert": "construction",
    "fs_morton": "construction",
    "fs_band2ge": "construction",
    "fs_ge8_jitter": "construction",
    "fs_ge8_detied": "construction",
    "fs_ge8_exactknn": "construction",
    "fs_band_ge8": "construction",
    "realign_ge8": "construction",
    "realign_strip": "construction",
    "realign_hilbert": "construction",
    "realign_morton": "construction",
    **{k: "construction" for k in list(FM.PGE_KEYS) + [FM.PGE_SWEEP_KEY, FM.PGE_KMSEED_KEY]},
    **{k: "construction" for k in FM.DETIED_KEYS},
    FM.REALIGN_NOISE_KEY: "construction",
    **{k: "repair" for k in FM.PGR_KEYS},
    # --- onarim katmani (tek cati): tek degisken komsuluk kumesi,
    #     tohum panelden secilir ve satirda gorunur ---
    "repair_two_opt": "repair", "repair_or_opt": "repair",
    "repair_relocate": "repair", "repair_vnd": "repair",
    # --- varyant havuzu + toplu onarim (tohumunu kendi uretir) ---
    # --- metasezgisel (tohum secilir) ---
    # --- rakip: GLOP'un ogrenmesiz yeniden-uyarlamasi ---
    # --- exact / altin standart: referans cizgisi, rakip DEGIL ---
    "ils": "metaheuristic",
    "lkh3": "reference", "concorde": "reference",
}
# ---------------------------------------------------------------------------
# GORUNEN ADLAR (2026-08-02, kullanici karari: "isimden yontem anlasilir olsun,
# v2/v3/v4 olmasin")
# ---------------------------------------------------------------------------
# ADLANDIRMA DILBILGISI -- bizim ailelerimizin her adi su kaliba uyar:
#
#     <makro-yapi> + <bant-ici kurucu> — <cesitlilik ekseni> (<aday sayisi>)
#
# Ornek: "Serpantin + Greedy-Edge — açı havuzu (2 aday)". Boylece iki satir
# YAN YANA okundugunda aralarindaki TEK DEGISKEN adin kendisinden gorulur:
#   ... Greedy-Edge — tek tur      vs  ... Greedy-Edge — açı havuzu  -> havuz
#   Serpantin + Greedy-Edge        vs  Serpantin + Farthest-Ins.     -> kurucu
#   Serpantin + Greedy-Edge        vs  Greedy-Edge (serpantinsiz)    -> makro-yapi
# "v1/v2/v4" surum numaralari KALDIRILDI: bir okuyucu icin sira numarasi
# yontemi anlatmiyordu, kalip anlatiyor.
#
# !! IC ANAHTARLAR DEGISMEDI !! ("greedy_snake_v2" vb.) Anahtarlar diskteki
# results/*.json satirlarinda, _STALE_SIGNATURES imzalarinda, LEGACY_ALIASES
# ve REMOVED_METHODS kumelerinde gomulu; degistirilmeleri butun gecmis
# olcumleri gecersiz kilardi. Degisen YALNIZ gosterim katmanidir (bu sozluk +
# dashboard METHODS dizisi + benchmark/method_table.py).
#
# KLASIK yontemlerin adlarina DOKUNULMADI: "Greedy-Edge", "Farthest Insertion",
# "Nearest Neighbor" zaten literaturun kanonik adlaridir -- Turkcelestirmek
# makalede takip edilebilirligi bozardi.
PRETTY = {
    # --- CERCEVE DENEYI (frame_methods.py; adlar oradan, tek kaynak) ---
    **FM.PRETTY,
    "nn": "Nearest Neighbor",
    "strip": "Strip (Space-Filling Curve)",
    "hilbert": "Hilbert-Egrisi (Platzman & Bartholdi 1989)",
    "morton": "Morton/Z-Order Egrisi",
    "farthest_insertion": "Farthest Insertion",
    "greedy_edge": "Greedy-Edge",
    # --- BIZIM AILE: serpantin makro-yapisi + bant-ici kurucu ---
    # --- KONTROL: ayni kurucu, serpantin YOK (makro-yapinin ablasyonu) ---
    # AYNI kurucu, AYNI 2 tur; degisen tek sey adaylari ne AYIRDIGI.
    # --- RGGE: Döndürülmüş Izgara Greedy-Edge (tek tur, havuz YOK) ---
    "rgge_y":
        "RGGE — Greedy-Edge döndürülmüş ızgarada, ızgara DEVRİK "
        "(θ açı kutusundan, k=8, tek tur, bant yok, havuz yok)",
    "rgge_x":
        "RGGE — Greedy-Edge döndürülmüş ızgarada, ızgara DÜZ "
        "(θ açı kutusundan, k=8, tek tur, bant yok, havuz yok)",
    # --- RSGE: Döndürülmüş Serpantin Greedy-Edge (bant sayısı = tek değişken) ---
    "rsge_resource":
        "RSGE — bant = kaynak tavanı (b=⌈n/m*⌉, m*=20000; θ açı kutusundan, k=8, tek tur)",
    "rsge_corridor":
        "RSGE — bant = koridor (b=1+#{boşluk>32×medyan}; θ açı kutusundan, k=8, tek tur)",
    "rsge_budget":
        "RSGE — bant = bütçe modeli (b=⌈c·n²/T⌉, c=3.93e-7, T=300 s; "
        "θ açı kutusundan, k=8, tek tur)",
    "rsge_fixed2":
        "RSGE — bant = sabit 2 (en küçük aşikâr-olmayan bölme; θ açı kutusundan, "
        "k=8, tek tur)",
    "rotation_strip": "θ★ Rotasyon — Strip Seyrek Tarama",
    # --- onarim katmani: tek cati, tek degisken = komsuluk kumesi.
    #     Panelde secilen tohum, gosterim adina "← <tohum>" olarak eklenir
    #     (bkz. display_name()). ---
    "repair_two_opt": "Onarım — yalnız 2-opt",
    "repair_or_opt": "Onarım — yalnız Or-opt (segment ≤3)",
    "repair_relocate": "Onarım — yalnız Relocate (tek nokta)",
    "repair_vnd": "Onarım — tam VND (2-opt + Or-opt + relocate)",
    "ils": "Iterated Local Search (ILS)",
    # ADLANDIRMA (2026-07-25, kullanici duzeltmesi): eski "+ Toplu Onarım"
    # ifadesi onarimin NE oldugunu soylemiyordu. Onarim ARTIK SECILEBILIR
    # (POOL_REPAIR_CHOICES), bu yuzden taban ad onu icermez -- ablasyonlarda
    # oldugu gibi display_name() secilen onarimi "← <onarim>" olarak ekler,
    # boylece satirin hangi onarimla kosuldugu tabloda her zaman okunur.
    "lkh3": "LKH-3 (Gold Standard)",
    "concorde": "Concorde (Exact/Optimal)",
}


def display_name(key: str, seed_key: "str | None" = None,
                 angle_key: "str | None" = None,
                 repair_key: "str | None" = None,
                 ge_variant_key: "str | None" = None) -> str:
    """Sonuc/panel gosterim adi. Girdisi secilebilen yontemler o girdiyle
    birlikte gosterilir -- "Onarım — tam VND ← Greedy-Edge",
    "Greedy Snake v1 ← Strip-oracle ...", "Greedy Snake v3 havuzu ← tam VND ..."
    gibi; boylece satirin hangi girdiden/hangi onarimla kosuldugu tabloda ve
    dashboard'da her zaman okunur."""
    base = PRETTY.get(key, key)
    if key in SEEDABLE and seed_key:
        return f"{base} ← {PRETTY.get(seed_key, seed_key)}"
    if key in ANGLE_SELECTABLE and angle_key:
        return f"{base} ← {angle_label(angle_key)}"
    if key in POOL_REPAIR_SELECTABLE and repair_key:
        return f"{base} ← {POOL_REPAIR_LABELS.get(repair_key, repair_key)}"
    if key in GE_VARIANT_SELECTABLE and ge_variant_key:
        return f"{base} ← {GE_VARIANT_LABELS.get(ge_variant_key, ge_variant_key)}"
    return base


# ---------------------------------------------------------------------------
# VARYANT SATIRLARI (2026-07-25, kullanici talebi)
# ---------------------------------------------------------------------------
# "Bir yontemi FARKLI girdilerle kosturup ikisini de gormek istiyorum."
# Eskiden sonuclar yontem anahtariyla saklaniyordu, dolayisiyla ayni yontemi
# ikinci kez baska bir tohum/aci/onarim ile kosmak oncekinin USTUNE yaziyordu.
# Artik her satirin bir ROW_ID'si var:
#
#     <yontem>              -- VARSAYILAN secimle kosulmus satir (degismedi)
#     <yontem>@<secim>      -- varsayilan DISI bir secimle kosulmus varyant
#
# Ornek: repair_vnd (girdi=greedy_edge, varsayilan)  ve
#        repair_vnd@greedy_snake_v3 (girdi=Greedy Snake v3)  AYNI dosyada YAN YANA.
#
# Neden varsayilan satira sonek YOK: diskteki tum eski sonuclar gecerli kalir
# (goc/migration gerekmez) ve "kanonik" satir temiz anahtarini korur. `key`
# alani HER ZAMAN yontem anahtarini tasir -> STAGE/PRETTY/FAMILY/renk/grup
# aramalari ve REMOVED_METHODS ayiklamasi oldugu gibi calisir.
def default_choice_of(method_key: str) -> str:
    """Yontemin secilebilir boyutundaki VARSAYILAN deger ("" = boyut yok)."""
    if method_key in SEEDABLE:
        return DEFAULT_SEED
    if method_key in ANGLE_PANEL_CANONICAL:
        # RGGE/RSGE: "varsayilan" = PANELDE SECILI aci (bkz.
        # ANGLE_PANEL_CANONICAL). Tek satir uretildigi icin o satirin temiz
        # anahtari tasimasi gerekir; aksi halde secim grid_theta disindayken
        # `rgge_x` satiri hic olusmaz ve _row_of/tohum zinciri kopar.
        return resolve_angle_choice(method_key, _panel_angle_config())
    if method_key in ANGLE_SELECTABLE:
        return DEFAULT_ANGLE_OF.get(method_key, DEFAULT_ANGLE)
    if method_key in POOL_REPAIR_SELECTABLE:
        return DEFAULT_POOL_REPAIR
    if method_key in GE_VARIANT_SELECTABLE:
        return DEFAULT_GE_VARIANT
    return ""


def row_id(method_key: str, choice: "str | None" = None) -> str:
    """Satir kimligi: varsayilan secimde yontem anahtarinin KENDISI, aksi
    halde "<yontem>@<secim>". Secilebilir boyutu olmayan yontemlerde her
    zaman yontem anahtari."""
    dflt = default_choice_of(method_key)
    if not dflt or not choice or choice == dflt:
        return method_key
    return f"{method_key}@{choice}"


def split_row_id(rid: str) -> "tuple[str, str]":
    """row_id -> (yontem_anahtari, secim). Sonek yoksa secim "" doner."""
    base, sep, choice = rid.partition("@")
    return (base, choice) if sep else (rid, "")


def _order_rows(rows_by_id: dict) -> list:
    """Satirlari raporlama sirasina dizer: once METHOD_ORDER sirasi, sonra
    ayni yontemin varyantlari (VARSAYILAN once, digerleri alfabetik). Boylece
    "repair_vnd" ile "repair_vnd@greedy_snake_v3" tabloda YAN YANA durur."""
    def _k(rid):
        base, choice = split_row_id(rid)
        try:
            i = METHOD_ORDER.index(base)
        except ValueError:
            i = len(METHOD_ORDER)
        return (i, choice != "", choice)
    return [rows_by_id[r] for r in sorted(rows_by_id, key=_k)]


def resolve_pool_repair_choice(method_key: str, repairs_cfg: "dict | None") -> str:
    """Havuzlu bir yontem icin panelden secilen ONARIM katmanini dondurur;
    gecersiz/eksik secimde DEFAULT_POOL_REPAIR'a duser."""
    if method_key not in POOL_REPAIR_SELECTABLE:
        return ""
    choice = (repairs_cfg or {}).get(method_key)
    return choice if choice in POOL_REPAIR_CHOICES else DEFAULT_POOL_REPAIR


def resolve_angle_choice(method_key: str, angles_cfg: "dict | None") -> str:
    """ANGLE_SELECTABLE bir yontem icin panelden secilen aci kaynagini
    dondurur; gecersiz/eksik secimde DEFAULT_ANGLE'a duser.

    Elle girilen aci KANONIKLESTIRILIR ("manual" -> "manual:-90"): aksi
    halde cıplak anahtar ile derece yazili anahtar AYNI aciyi iki ayri
    row_id'ye yazardi."""
    if method_key not in ANGLE_SELECTABLE:
        return ""
    choice = (angles_cfg or {}).get(method_key)
    deg = manual_angle_deg(choice)
    if deg is not None:
        return manual_angle_choice(deg)
    return choice if choice in ANGLE_CHOICES else DEFAULT_ANGLE_OF.get(method_key, DEFAULT_ANGLE)


def resolve_ge_variant_choice(method_key: str, ge_cfg: "dict | None") -> str:
    """GE_VARIANT_SELECTABLE bir yontem icin panelden secilen Greedy-Edge
    duyarlilik varyantini dondurur; gecersiz/eksik secimde
    DEFAULT_GE_VARIANT'a (= rakip satirin kendisi) duser."""
    if method_key not in GE_VARIANT_SELECTABLE:
        return ""
    choice = (ge_cfg or {}).get(method_key)
    return choice if choice in GE_VARIANT_CHOICES else DEFAULT_GE_VARIANT
# ---------------------------------------------------------------------------
# FAMILY: same-class grouping for a fair "who wins within its own complexity
# class" comparison (see EGITIMLI_GNN_AKADEMIK_NOT.md-style notes and the
# Snake-Grid academic note). Do NOT compare "curve" against "insertion" or
# "edge_greedy" and claim a win -- those are different asymptotic/quality
# classes; see hyperparameters()["family_note"]. Used purely for reporting/
# chart color-grouping, has no effect on which methods run.
# ---------------------------------------------------------------------------
FAMILY = {
    "nn": "curve",              # grid-accelerated NN: near-linear, same budget
    "nn_exact": "curve",        # kesin NN: ayni kurucu, hizlandirma yok
    "strip": "curve",
    "hilbert": "curve",
    "morton": "curve",
    # v4_4 de "curve"dedir, v2 ile AYNI gerekceyle: bant + bant-ici
    # greedy-edge, O(n log n) sinifi. (ge_pool2/fi_pool2'un aksine -- onlar
    # BASKA bir kurucu ailesinin havuzu; buradaki uc satir ayni kurucunun
    # farkli EKSENLERDE havuzlanmis halidir ve ayni rafta okunmalidir.)
    "rotation_strip": "curve",
    "farthest_insertion": "insertion",
    # Far Snake "insertion" ailesindedir, "curve"da DEGIL (2026-07-28
    # duzeltmesi). Eski `snake_farthest_band` satiri "curve" icinde
    # duruyordu; bu YANLISTI ve olculmustu: bant-ici farthest-insertion
    # HIZLANDIRILMAMIS O(m^2) oldugu icin ailenin sure-n egimi n^1.51,
    # curve ailesininki ise n^1.08-1.10. Yani "kendi karmasiklik sinifinda
    # kim kazaniyor?" tablosunda daha ucuz kuruculara karsi haksiz bir
    # avantajla goruluyordu. Dogru komsusu farthest_insertion'dir (O(n^2)):
    # Far Snake tam olarak ONUN maliyetini serpantin ayrisimiyla bant
    # boyutuna indirme denemesidir, o yuzden ayni rafta okunmalidir.
    "greedy_edge": "edge_greedy",
    # strip_theta_ge TEK tur kurar (havuz degil), yani greedy_edge ile
    # AYNI butce sinifindadir ve FAMILY'ye girer -- havuzlu satirlardan
    # farki tam olarak budur.
    # RSGE dortlusu de "edge_greedy"dedir ve bu ZORUNLU: dordu de TEK tur
    # kurar, bant-ici kurucusu greedy-edge, aday genisligi k=15 -- yani
    # `greedy_edge` ile AYNI butce sinifi. RSGE(theta=0,b=1) == greedy_edge
    # ozdesligi zaten ayni rafta olmayi zorunlu kilar.
    "rsge_resource": "edge_greedy", "rsge_corridor": "edge_greedy",
    "rsge_budget": "edge_greedy", "rsge_fixed2": "edge_greedy",
    # RGGE: TEK tur, kurucu greedy-edge, k=8 -> `greedy_edge` ile AYNI
    # butce sinifi. Havuzlu satirlarin aksine FAMILY'ye girer, cunku
    # esit-tur-sayisi sozlesmesini gercekten sagliyor (1 tur <-> 1 tur).
    "rgge_y": "edge_greedy", "rgge_x": "edge_greedy",
    # ge_pool2 / fi_pool2 -- ve 2026-08-02'de eklenen ge_pool2_jitter,
    # ge_pool2_yon, knn_pool2, knn_pool2_uyarlanir -- FAMILY'ye GIRMEZ (bilerek):
    # hepsi cok varyantli bir HAVUZ
    # kurup en iyisini gercek maliyetle secerler, yani tek-turlu saf
    # kuruculardan farkli bir butce sinifindadirlar. Dogru muadilleri
    # greedy_snake_v2 / far_snake_v2'dir (ham, 4 aday) -- hepsi havuzlu,
    # hepsi onarimsiz.
}
FAMILY_LABEL = {
    "curve": "Uzay-bolmeli / egri-tabanli kurucular (O(n log n), aynı sınıf)",
    "insertion": "Artimli ekleme ailesi (insertion, O(n^2))",
    "edge_greedy": "Kenar-secim ailesi (greedy-edge / quick-boruvka, aday listeli O(n log n) ama farkli is sabiti/kalite)",
}

# NOTE: glop_like BILEREK STOCHASTIC degil -- varsayilan cagri deterministik
# bolumleme + sabit-tohumlu ic kurucular kullanir (adil, tekrarlanabilir rakip
# satiri); tohumlu varyant glop_like.solve(seed=...) ile ayrica istenebilir.
STOCHASTIC = {"ils"}

# ---------------------------------------------------------------------------
# TOHUM SECIMI (2026-07-24): "baslangic turu -> yerel arama hangi havzaya
# iner" deneyi artik AYRI yontem satirlariyla degil, TEK yontem + SECILEN
# TOHUM ile yapilir. Kullanici admin panelinden her SEEDABLE yontem icin bir
# insa secer; secim methods_config.json'daki "seeds" alaninda tutulur ve
# sonuc satirina seed_key/seed_name olarak yazilir. Ayni yontemi farkli
# tohumlarla kosmak isteyen, panelden tohumu degistirip yeniden kosar.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# KOSUM BASINA SECIM GECERSIZ KILMA (2026-09-07) -- panel "cogaltma"nin ayagi
# ---------------------------------------------------------------------------
# Panelde bir yontemin TEK secimi (tohum/aci/onarim/varyant) kayitlidir. Ayni
# yontemi ayni kosumda IKI farkli secimle gormek isteyen kullanici, panelden
# yontemi COGALTIR; her kopya kendi secimini tasir ve sunucu her kopya icin
# runner'i `--choices <yontem>=<secim>` ile cagirir.
#
# NEDEN GLOBAL BIR DEPO (fonksiyon parametresi degil): secim, run_dataset'in
# icinde TEK yerden okunmuyor -- aci cozumlemesi (v1 hattinda) ve
# hyperparameters() raporu _panel_*_config()'i BAGIMSIZ olarak yeniden
# cagiriyor. Parametre gecirseydik bu ikinci/ucuncu okuma panel dosyasindaki
# ESKI secimi gorur, satir "manual:0" adiyla kaydedilip gercekte "manual:-90"
# ile kosardi (sessiz yanlis etiket -- row_id mekanizmasinin tam da onlemek
# icin var oldugu hata). Deponun okundugu tek nokta panel okuyuculari oldugu
# icin butun cagri yollari otomatik olarak tutarli kalir.
#
# Surec basina TEK kosum yapilandirmasi vardir (her kopya ayri bir alt-surec),
# bu yuzden global durum burada guvenlidir.
_CHOICE_OVERRIDE: dict = {}


def set_choice_overrides(overrides: "dict | None") -> dict:
    """Bu SURECTEKI koşum için panel seçimlerini geçersiz kılar.
    {yontem: secim}; gecersiz anahtar/deger sessizce atilmaz -- dogrulama
    secimi cozen resolve_* fonksiyonlarinda zaten var (gecersiz deger
    varsayilana duser), burada yalniz yontem anahtari suzulur."""
    global _CHOICE_OVERRIDE
    _CHOICE_OVERRIDE = {m: c for m, c in (overrides or {}).items()
                        if m in METHOD_ORDER}
    return dict(_CHOICE_OVERRIDE)


def choice_overrides() -> dict:
    return dict(_CHOICE_OVERRIDE)


def _apply_override(cfg: dict, dimension: set) -> dict:
    """Panel sozlugunun uzerine, o BOYUTA ait gecersiz kilmalari yazar."""
    if not _CHOICE_OVERRIDE:
        return cfg
    out = dict(cfg)
    out.update({m: c for m, c in _CHOICE_OVERRIDE.items() if m in dimension})
    return out


def parse_choice_overrides(spec: "str | None") -> dict:
    """`--choices` degerini cozer: "yontem=secim,yontem2=secim2".
    Secim degerinde ':' bulunabilir (ornegin "manual:-90"); ayirici yalniz
    ILK '=' isaretidir."""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"--choices ögesi 'yöntem=seçim' biçiminde olmalı: {part!r}")
        out[k.strip()] = v.strip()
    return out


def _panel_seed_config() -> dict:
    """Admin panelinden secilen tohumlar (dashboard/methods_config.json ->
    "seeds": {yontem: insa}). Dosya yoksa/bozuksa bos sozluk doner ve her
    SEEDABLE yontem DEFAULT_SEED ile kosar."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "methods_config.json")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
        s = cfg.get("seeds")
        s = s if isinstance(s, dict) else {}
    except Exception:
        s = {}
    return _apply_override(s, SEEDABLE)


#: `_panel_angle_config` onbellegi. default_choice_of -> row_id yolu artik
#: bu yapilandirmaya baglidir ve row_id siralama/atlama dongulerinde binlerce
#: kez cagrilir; her cagrida dosya okumak kosumu gereksiz yavaslatirdi.
#: Anahtar dosyanin mtime+boyutu -- panelden kayit yapilinca kendiliginden
#: tazelenir.
_ANGLE_CFG_CACHE: dict = {"stamp": None, "raw": {}}


def _panel_angle_config() -> dict:
    """Admin panelinden secilen aci kaynaklari
    (dashboard/methods_config.json -> "angles": {yontem: aci_kaynagi})."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "methods_config.json")
    try:
        st = os.stat(p)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    if stamp is None or stamp != _ANGLE_CFG_CACHE["stamp"]:
        try:
            with open(p, encoding="utf-8") as fh:
                cfg = json.load(fh)
            a = cfg.get("angles")
            a = a if isinstance(a, dict) else {}
        except Exception:
            a = {}
        _ANGLE_CFG_CACHE["stamp"] = stamp
        _ANGLE_CFG_CACHE["raw"] = a
    return _apply_override(dict(_ANGLE_CFG_CACHE["raw"]), ANGLE_SELECTABLE)


def _panel_pool_repair_config() -> dict:
    """Admin panelinden secilen HAVUZ ONARIMI katmanlari
    (dashboard/methods_config.json -> "pool_repairs": {yontem: onarim})."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "methods_config.json")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
        r = cfg.get("pool_repairs")
        r = r if isinstance(r, dict) else {}
    except Exception:
        r = {}
    return _apply_override(r, POOL_REPAIR_SELECTABLE)


def _panel_ge_variant_config() -> dict:
    """Admin panelinden secilen GREEDY-EDGE DUYARLILIK varyantlari
    (dashboard/methods_config.json -> "ge_variants": {yontem: varyant})."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "methods_config.json")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
        g = cfg.get("ge_variants")
        g = g if isinstance(g, dict) else {}
    except Exception:
        g = {}
    return _apply_override(g, GE_VARIANT_SELECTABLE)


# ---------------------------------------------------------------------------
# ADMIN PANELINDEN KALICI SILME (2026-09-07, kullanici talebi)
# ---------------------------------------------------------------------------
# REMOVED_METHODS bu KODDAN cikarilan yontemlerin sabit listesidir (kaynak
# kodu degistirmek gerekir). Panelden silme ise VERI ile calisir: silinen
# anahtarlar dashboard/methods_config.json -> "deleted_methods" listesinde
# tutulur ve KALICIDIR -- sunucu yeniden baslasa, tarayici onbellegi
# temizlense de geri gelmez.
#
# "Gizli" ile "silinmis" ARASINDAKI FARK (ikisini karistirmak eski bir hata
# kaynagiydi):
#   gizli    -> yontem koda ve diske ait, sadece panelde/rapor da gosterilmez;
#               tiki geri isaretleyince satirlari OLDUGU GIBI geri gelir.
#   silinmis -> yontem hic yokmus gibi davranilir: kosulmaz, tohum/aci/onarim
#               kutularinda SECENEK OLARAK CIKMAZ, admin listesinde yer almaz
#               ve results/*.json icindeki satirlari diskten TEMIZLENIR.
#
# Yukleme mtime ile onbelleklenir: sunucu sureci runner'i BIR KEZ import eder
# ama panelden silme sirasinda dosya degisir -- modul duzeyinde bir kez okusak
# sunucu eski listeyi tasirdi (ayni desen "admin tiki geri kaliyor" hayalet
# hatasini uretmisti, bkz. server.py allow_reuse_address notu).
_DELETED_CACHE = {"mtime": None, "set": frozenset()}


def _panel_deleted_config() -> frozenset:
    """Panelden KALICI olarak silinen yontem anahtarlari
    (dashboard/methods_config.json -> "deleted_methods")."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "methods_config.json")
    try:
        mt = os.path.getmtime(p)
    except OSError:
        _DELETED_CACHE.update(mtime=None, set=frozenset())
        return _DELETED_CACHE["set"]
    if _DELETED_CACHE["mtime"] == mt:
        return _DELETED_CACHE["set"]
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh).get("deleted_methods")
        got = frozenset(d) if isinstance(d, (list, tuple, set)) else frozenset()
    except Exception:
        got = frozenset()
    _DELETED_CACHE.update(mtime=mt, set=got)
    return got


def deleted_methods() -> frozenset:
    """Panelden silinen yontemler (mtime onbellekli okuma)."""
    return _panel_deleted_config()


def is_dropped_method(key: str) -> bool:
    """Bu anahtar artik RAPORLANMAZ mi? Iki kaynak: koddan kaldirilanlar
    (REMOVED_METHODS) ve panelden kalici silinenler (deleted_methods()).
    Diskteki eski satirlarin ayiklandigi HER yerde bu tek kapi kullanilir."""
    return key in REMOVED_METHODS or key in _panel_deleted_config()


def live_method_order() -> list:
    """METHOD_ORDER'in silinmemis hali -- 'kosulabilir/gosterilebilir yontem'
    listesi isteyen her yer (panel, varsayilan kosu kumesi) bunu kullanir."""
    dead = _panel_deleted_config()
    return [k for k in METHOD_ORDER if k not in dead]


def resolve_seed_choice(method_key: str, seeds_cfg: "dict | None") -> str:
    """SEEDABLE bir yontem icin panelden secilen tohum anahtarini dondurur.
    Gecersiz/eksik secimde DEFAULT_SEED'e duser (sessiz kalmaz: cagiran
    taraf logladigi icin secim her zaman sonuc dosyasinda gorunur).
    Kendi kendini tohum secmek yasaktir."""
    if method_key not in SEEDABLE:
        return POOL_SOURCE.get(method_key, "")
    choice = (seeds_cfg or {}).get(method_key)
    if choice in SEED_CHOICES and choice != method_key:
        return choice
    return DEFAULT_SEED


def order_seedable(seed_of: dict) -> "tuple[list, dict]":
    """SEEDABLE yontemleri BAGIMLILIK sirasina dizer: bir satir, tohumu olan
    satirdan SONRA kosar. Dongu (a<-b<-a) bulunursa dongudeki satirlar
    DEFAULT_SEED'e dusurulur.

    Doner: (sirali_yontem_listesi, duzeltilmis_seed_of).
    Yalniz SEEDABLE->SEEDABLE kenarlari bagimliliktir; yapici/havuz tohumlari
    zaten bu asamadan once kosar."""
    seed_of = dict(seed_of)
    nodes = [m for m in METHOD_ORDER if m in seed_of]
    # dongu tespiti: renklendirmeli DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {m: WHITE for m in nodes}
    order: list = []
    broken: list = []

    def visit(m, stack):
        if color[m] == BLACK:
            return
        if color[m] == GRAY:                     # dongu: bu kenari kes
            broken.append(m)
            seed_of[m] = DEFAULT_SEED
            return
        color[m] = GRAY
        dep = seed_of.get(m)
        if dep in seed_of and dep != m:
            visit(dep, stack + [m])
        color[m] = BLACK
        order.append(m)

    for m in nodes:
        visit(m, [])
    return order, seed_of, broken


def _strip_tour_at_angle(inst, xs, ys, deg):
    """Builds the STRIP (boustrophedon) tour with the coordinates rotated by
    `deg` degrees and returns (tour, TRUE_TSPLIB_cost). Rotation only changes
    the strip ORDERING; the cost is always scored on the original instance,
    so results at different angles are directly comparable. O(n log n)."""
    if abs(deg) < 1e-9:
        tour = E.snake_order(xs, ys, 0)
    else:
        rx, ry = E.rotate_coords(xs, ys, deg)
        tour = E.snake_order(rx, ry, 0)
    return tour, inst.tour_cost(tour)


_ROTATION_FINE_STEP = 0.5    # always refine the best region down to this (degrees)
_ROTATION_SUBSET_MIN = 500   # n<=500: theta taramasi HER ZAMAN tam kumede kosar.
                              # n>500: kullanicinin belirledigi ORAN (rotation_sparsity,
                              # dashboard'da "Rotasyon kaba adim (theta)" altindaki
                              # seyreklik orani kaydiricisi) kadar rastgele alt-kume
                              # kullanilir -- eskiden sabit 1200 nokta idi (fl3795
                              # deneyinde dogrulanmisti), artik n'e ORANTILI ve
                              # kullanici kontrolunde (bkz _ROTATION_DEFAULT_SPARSITY).
_ROTATION_DEFAULT_SPARSITY = 0.15  # kaydirici hic dokunulmamissa kullanilan varsayilan oran
_ROTATION_SUBSET_SEED = 42   # deterministik: ayni kosum hep ayni theta*'yi bulur

# ---------------------------------------------------------------------------
# BOYUT ESIGI -- "bu n'de zaten bitmez, HIC denemeyelim" (2026-07-30)
#
# SORUN. Insa butcesi kapagi (asagidaki _CONSTRUCTION_MAX_S) satiri kurtarir
# ama BEDELINI ODEYEREK: yontem 300 s calisir, kapaga takilir, not yazilir.
# Bu bedel bir kez odenip hafizaya girse sorun degil -- ama hicbir zaman
# denenmemis 27 sette (n=3795..28924) fi_pool ailesinin DORT satiri da
# bostaydi, yani ilk kosum bu setlerin her birinde 4 x 300 s = 20 dk yakmak
# zorundaydi. Halbuki n'e bakip ONCEDEN bilebiliriz.
#
# ESIKLER UYDURULMADI, OLCULDU. Diskteki 128 sonuc dosyasindan yontem basina
# (n, sure) ornekleri toplanip log-log en kucuk karelerle t(n)=a*n^p uyduruldu:
#
#   yontem              ornek   us p    R^2    en buyuk BASARI   en kucuk BASARISIZ
#   nearest_insertion     113   2.11   0.999        24104              22775
#   farthest_insertion    113   2.10   0.998        24104              22775
#   fi_pool                83   2.06   0.997         6117               7146
#   fi_pool2               92   2.07   0.997        10639              14185
#   fi_pool_repair         84     --     --          7146               7663
#   fi_pool2_repair        93     --     --         13509              14185
#
# p ~ 2.1 ve R^2 >= 0.997: bu satirlar GERCEKTEN Theta(n^2) ve n'den
# ongorulebilir. Esikler her yontemin en buyuk GOZLENEN BASARISININ USTUNDE,
# en kucuk gozlenen basarisizliginin ALTINDA secildi -- yani basarabilecegini
# BILDIGIMIZ hicbir kosumu engellemiyoruz.
#
# ESIK BUTCEYLE BUYUR. Sabit sayi yazsaydik TRUBA'da --construction-max-s 1800
# verdigimizde n<=7000 siniri hala fi_pool'u keserdi ve gecerli veriyi
# kaybederdik. max_n(butce) = n_ref * (butce/300)^(1/p) -- 1800 s'te fi_pool
# siniri ~17000'e cikar. Ayni sebeple butce KUCULTULURSE sinir daralir.
#
# ESIK KOYMADIGIMIZ YONTEMLER (olcum destekLEMEDIGI icin):
#   far_snake_v1/v2/v3 + onarimlari -- R^2 0.84..0.94 ve basari/basarisizlik
#     araliklari CAKISIYOR (v3: n=4663'te basarisiz, n=10639'da basarili).
#     Maliyet n'in tek basina fonksiyonu degil (bant carpani, aci havuzu);
#     esik koymak calisan satirlari keserdi. Bunlar not hafizasina birakildi.
#   lkh3 -- n=21215'te BILE tur uretiyor (109 olcum). Yavas ama sonuc veriyor;
#     n esigi koymak gecerli sonuclari atmak olurdu.
#   concorde -- R^2=0.66, 28 ornek, ve basarisizlik araligi basari araligini
#     KAPSIYOR (n=813'te basarisiz, n=2481'de basarili). Kesin cozucunun
#     suresi ornek boyutunun degil ornek YAPISININ fonksiyonu; n esigi
#     ilkesel olarak yanlis olurdu. Not hafizasina birakildi.
#
# Geriye donuk uyum: nearest/farthest_insertion esigi 25000'de KALDI (eski
# sabit). Olcum daraltmayi desteklemiyor -- basari (24104) ile basarisizlik
# (22775) araliklari orada da cakisiyor. Degisen tek sey artik butceyle
# olceklenmesi.
# ---------------------------------------------------------------------------
_SIZE_LIMIT_REF_S = 300.0        # esiklerin olculdugu referans insa butcesi
_SIZE_LIMITS = {
    #  yontem                (n_ref @ 300 s,  olculen us p)
    "nearest_insertion":     (25000, 2.11),
    "farthest_insertion":    (25000, 2.10),
    "fi_pool2":              (11000, 2.07),
    # Onarimli satir KENDI havuzunun ureticisini (_fi_pool_tours) cagirir:
    # havuz kurulamazsa onarim satiri da olusamaz. Esik kendi gozlenen
    # basarisinin (13509) hemen ustune kondu.
    "fi_pool2_repair":       (13600, 2.07),
    # NOT (2026-07-31): fi_pool / fi_pool_repair esikleri (7000 / 7500)
    # satirlarla birlikte kaldirildi -- bkz. REMOVED_METHODS (f).
}

# Eski ad: disaridan referans verilmis olabilir.
_INSERTION_MAX_N = _SIZE_LIMITS["nearest_insertion"][0]


def method_max_n(key, budget_s):
    """`key` yonteminin `budget_s` insa butcesine sigan en buyuk n'i.

    Esigi olmayan yontemde None doner (= sinir yok, her zaman denenir).
    """
    ent = _SIZE_LIMITS.get(key)
    if ent is None:
        return None
    n_ref, p = ent
    scale = max(1e-9, float(budget_s)) / _SIZE_LIMIT_REF_S
    return int(n_ref * (scale ** (1.0 / p)))

# Insa (construction) yontemi basina duvar-saati ust siniri (kullanici karari
# 2026-07-16, sra104815 vakasi: greedy_edge 4094s, gpu_snake_greedy_band 6326s
# surdu -- "300s'i gecerse islem yapmasin"). Kapak KOOPERATIFTIR: kuruculardaki
# check_construction_deadline() cagrilari E.ConstructionTimeout firlatir, yontem
# "denendi, sonuc yok" kutusuna duser (note_unavailable), kismi sonuc ATILIR.
# Yalniz insa asamasina uygulanir; ILS/metasezgisel butceleri ayridir.
_CONSTRUCTION_MAX_S = 300.0

# ---------------------------------------------------------------------------
# SURE ASIMI HAFIZASI -- IKI BUTCE TIPI
#
# "Bir kere kosuldu, verilen surede bitiremedi" diyen her not hafizaya girer
# ve AYNI (ya da daha dar) butceyle BIR DAHA DENENMEZ. Iki ayri butce var ve
# ikisi ayri bayraklarla ayarlanir; bu yuzden `reason` alani hangisinin
# gevsetilmesi gerektigini de soyler:
#
#   reason="construction_budget"  butce: --construction-max-s  (insa asamasi)
#   reason="exact_budget"         butce: --exact-time          (LKH-3/Concorde)
#
# 2026-07-29 olcumu, diskteki 128 sonuc dosyasi:
#     90 Concorde sure asimi  27.009 s   (7,5 saat)
#     13 LKH-3 sure asimi      7.801 s   (2,2 saat)
#     79 insa butcesi asimi   23.796 s   (6,6 saat)
# Toplam ~16 saat, HER tam kosumda yeniden yaniyordu.
#
# NEDEN GUVENLI. Insa deterministiktir; ayni koordinat + ayni butce = ayni
# sonuc. Kesin cozuculer icin argüman biraz farkli: Concorde/LKH daha HIZLI
# bir makinede (ornegin TRUBA) ayni 300 s icinde bitirebilir. Bu yuzden
# hafiza "sonsuza kadar yasak" DEGIL, "ayni butceyle tekrar deneme"dir --
# butceyi buyutmek (--exact-time 1800) ya da --force vermek hafizayi cozer.
# TRUBA'ya gecerken zaten butceyi buyutecegiz, satirlar kendiliginden geri
# gelir.
# ---------------------------------------------------------------------------

# Eski (2026-07-29 oncesi) notlari METINDEN tanima desenleri: o notlarda
# `reason`/`budget_s` alanlari yok.
_BUDGET_NOTE_RE = re.compile(r"inşa\s+([\d.]+)\s*s\s+bütçesini\s+aştı")
_CONCORDE_TIMEOUT_RE = re.compile(r"did not finish within\s+([\d.]+)\s*s")
_LKH_TIMEOUT_RE = re.compile(r"hard-killed after\s+([\d.]+)\s*s")

# `reason` -> hangi bayrak butceyi ayarliyor (kullaniciya gosterilen metin)
BUDGET_FLAG = {
    "construction_budget": "--construction-max-s (panelde “İnşa üst süre sınırı”)",
    "exact_budget": "--exact-time (kesin çözücü süre sınırı)",
    # size_limit de INSA butcesine baglidir: esik butceyle olceklenir
    # (bkz. `method_max_n`), dolayisiyla ayni bayrak onu da cozer.
    "size_limit": "--construction-max-s (panelde “İnşa üst süre sınırı”)",
}

# Panelde notun sonuna eklenen aciklama. Kullanici kutuyu gorunce satirin
# NEDEN bir daha kosulmadigini ve NASIL zorlayacagini bilmeli; yoksa
# "sistem bunu atliyor ama neden" sorusu kaliyor.
_RETRY_HINT = "bu bütçeyle bir daha denenmez; denemek için “{flag}” değerini büyütün"


def budget_note_suffix(reason):
    return " — " + _RETRY_HINT.format(flag=BUDGET_FLAG[reason])


def exact_timeout_budget(status):
    """Metin bir KESIN COZUCU sure asimi bildiriyorsa yapilandirilmis butceyi
    (saniye) dondurur; baska bir basarisizliksa None.

    AYIRIM ONEMLI: "binary not found" ya da "produced no tour file (rc=...)"
    sure asimi DEGILDIR. Ikincisi bir cokme; ikili kurulunca ya da hata
    duzelince satir kosmali. Onlari hafizaya alirsak kullanici Concorde'u
    kurdugunda satir sessizce atlanmaya devam ederdi.
    """
    if not isinstance(status, str):
        return None
    m = _CONCORDE_TIMEOUT_RE.search(status)
    if m:
        # Concorde metne dogrudan yapilandirilmis time_limit'i yazar.
        return float(m.group(1))
    m = _LKH_TIMEOUT_RE.search(status)
    if m:
        # LKH metne DIS KAPAGI yazar; yapilandirilmis butceye cevir.
        return ES.lkh_time_limit_from_hard_cap(float(m.group(1)))
    return None


def migrate_solver_notes(notes):
    """Eski biçimli notlara `reason`/`budget_s` alanlarını ekler.

    NEDEN GEREKLI: bütçe hafızası (run_dataset içindeki `_budget_blocked`) bir
    notu MAKINE-OKUNUR alanlarindan tanir. Alanlar yoksa hafiza notu goremez
    ve o satir her kosumda butce kadar zaman yakar. Butce degeri metnin
    ICINDE zaten sayi olarak duruyor; onu okuyup alanlari dolduruyoruz. Desen
    tutmazsa not oldugu gibi birakilir -- yanlis pozitif yok.

    TEK KAYNAK: hem runner'in devam mantigi hem de HPC dagiticisinin is
    kestirimi (hpc/schedule.py) bu fonksiyonu cagirir. Iki kopya olsaydi biri
    guncellenip digeri unutuldugunda dagitici "bu satir kosacak" deyip butce
    ayirir, runner ise atlardi -- sira tahmini sessizce sisirdi.

    Listeyi YERINDE degistirir; degisen not sayisini dondurur.
    """
    changed = 0
    for nt in notes or []:
        if not isinstance(nt, dict):
            continue
        if nt.get("reason") or not isinstance(nt.get("status"), str):
            continue
        m = _BUDGET_NOTE_RE.search(nt["status"])
        if m:
            reason, budget = "construction_budget", float(m.group(1))
        else:
            b = exact_timeout_budget(nt["status"])
            if not b:
                continue
            reason, budget = "exact_budget", float(b)
        nt["reason"] = reason
        nt["budget_s"] = budget
        # Metni de tamamla: eski notlar yalnizca "yetismedi" diyordu, satirin
        # bir daha KOSULMAYACAGINI ve nasil zorlanacagini soylemiyordu.
        if "bir daha denenmez" not in nt["status"]:
            nt["status"] += budget_note_suffix(reason)
        changed += 1
    return changed


# Eski ad: hpc/schedule.py ve olasi disaridan cagrilar icin korunuyor.
migrate_construction_notes = migrate_solver_notes


# ---------------------------------------------------------------------------
# ESKI v4 SATIRLARININ AYIKLANMASI (2026-07-31)
# ---------------------------------------------------------------------------
# greedy_snake_v4 ailesi 2026-07-30'da SATIR YUKSEKLIGI havuzuydu; 2026-07-31'de
# kullanici karariyla ACI INCE TARAMASINA cevrildi. ANAHTARLAR AYNI KALDI
# (kullanicinin adlandirmasi), fakat diskte 103 kumede eski tasarimin satirlari
# duruyordu ve resume mantigi "turu var -> hazir" dedigi icin bunlar BIR DAHA
# HIC yeniden uretilmezdi: tabloda yeni yontemin adiyla ESKI yontemin sayilari
# kalirdi. Sessiz ve kalici bir yanlis etiketleme olurdu.
#
# REMOVED_METHODS ise kullanilamaz: anahtar hem aktif hem "kaldirilmis"
# olamaz. Cozum IMZA TABANLI ayiklama -- iki tasarim etiketinden ayirt edilir:
#     eski (yukseklik): winner = "h/k=3"
#     yeni (aci)      : winner = "th=-4" / "th=0*"
# Boylece yalnizca eski uretimden gelen satirlar dusurulur, yeni satirlar ve
# diger butun yontemler dokunulmadan kalir.
_V4_KEYS = {"greedy_snake_v4", "greedy_snake_v4_2x2",
            "greedy_snake_v4_repair", "greedy_snake_v4_2x2_repair"}
# NOT: 10'lu anahtarlar (greedy_snake_v4 / _repair) 2026-07-31'de
# REMOVED_METHODS'a girdi ve snapshot filtresi onlari ZATEN ayikliyor; yine de
# listede BIRAKILDILAR. Iki filtre farkli sey yapar ve sirasi onemlidir:
# REMOVED yalniz _merge_snapshot'i suzer, drop_stale_v4_rows ise DISKTEKI
# payload'i da temizler. 10'lu anahtarlari burada tutmak, eski dosyalardaki
# satir-yuksekligi uretimlerinin diskten de silinmesini garanti eder.
_V4_STALE_WINNER = "h/k="

# ---------------------------------------------------------------------------
# IMZA TABLOSU: (anahtar kumesi, "bu winner ESKI mi?" testi)
# ---------------------------------------------------------------------------
# Bir yontemin ANAHTARI ayni kalip URETIMI degistiginde REMOVED_METHODS
# kullanilamaz (anahtar hem aktif hem kaldirilmis olamaz). O durumda satirlar
# `winner` etiketinin BICIMINDEN ayirt edilir. Yeni bir tasarim degisikligi
# yapildiginda buraya bir satir eklenmesi ZORUNLUDUR, yoksa eski uretim
# diskte kalir ve yeni yontemin adiyla ESKI yontemin sayilari raporlanir.
_STALE_SIGNATURES = [
    # v4: satir yuksekligi havuzu (2026-07-30) -> aci havuzu (2026-07-31).
    #     eski winner "h/k=3", yeni "th=-90*/k=4".
    # v4 (2026-08-01): 2 adayli REKABET havuzu -> 100 adayli BANT TARAMASI
    #     (tani satiri, snake_alt.V4_CONFIGS). Iki uretim AYNI etiket
    #     bicimini uretir (`th=-90/k=8/b=2y`), o yuzden yeni uretim etikete
    #     `/tarama` ekler ve eskiligin olcutu o ektir. Bu kural onceki
    #     "h/k=" (satir yuksekligi havuzu) kuralini da KAPSAR -- o etiketler
    #     de `/tarama` icermez.
    (_V4_KEYS,
     lambda w: SA.V4_TARAMA_ETIKET_EKI not in w),
    # v2: tek k (=8) -> aci x k caprazi {4,8,16} (2026-07-31, kullanici
    #     karari). Eski winner "th=-10/mult=0.05", yeni ayni etiketin sonuna
    #     "/k=<k>" ekler. Yeni havuz eskisinin UST KUMESI oldugu icin sonuc
    #     asla kotulesemez, ama satir 4 aday yerine 12 adayla uretilmistir --
    #     ikisini ayni etiket altinda karistirmak yanlis raporlama olurdu.
    ({"greedy_snake_v2", "greedy_snake_v2_repair"},
     lambda w: "/k=" not in w),
]


def is_stale_angle_row(row: "dict | None") -> bool:
    """Eski UC SATIRLI RGGE/RSGE sozlesmesinden kalma bir satir mi?

    2026-09-07'den once bu aileler her kosumda `<yontem>` (grid_theta),
    `<yontem>@zero` ve panel secimi olmak uzere UC satir uretiyordu. Artik
    YALNIZ panelde secili acida TEK satir uretilir ve o satir daima temiz
    anahtardadir; dolayisiyla "@" sonekli her RGGE/RSGE satiri eski
    uretimdendir. Diskteki dosyalar bir sonraki kosumda kalici olarak
    temizlenir (`drop_stale_v4_rows`); bu yardimci, HENUZ yeniden kosulmamis
    dosyalarda panelin ayni yontemi 2-3 kez gostermemesi icin sunucu
    tarafinda da kullanilir."""
    if not isinstance(row, dict):
        return False
    key = row.get("key")
    return (key in ANGLE_PANEL_CANONICAL
            and "@" in str(row.get("row_id") or key))


def drop_stale_v4_rows(snapshot: dict, payload: "dict | None" = None) -> int:
    """Uretimi degismis ama ANAHTARI ayni kalmis satirlari snapshot'tan VE
    payload'dan siler; silinen satir sayisini dondurur.

    Imzasi olmayan (winner alani bulunmayan) satirlar da ESKI sayilir -- bu
    tasarimlarin hepsi `winner`i HER ZAMAN yazar, dolayisiyla eksik alan
    eski uretimin isaretidir.

    Ad tarihseldir (once yalniz v4 icindi); artik _STALE_SIGNATURES
    tablosundaki her yontem ailesini kapsar."""
    def _stale(row):
        if not isinstance(row, dict):
            return False
        key = row.get("key")
        # 2026-09-07: RGGE/RSGE artik TEK satir uretir ve o satir daima temiz
        # anahtardadir. Diskte kalan "@grid_theta / @zero / @rotation_strip"
        # satirlari ESKI uc-satirli sozlesmedendir; birakilirsa tabloda ayni
        # yontem 2-3 kez, farkli acilarda gorunur.
        if is_stale_angle_row(row):
            return True
        w = str(row.get("winner") or "")
        for keys, is_old in _STALE_SIGNATURES:
            if key in keys:
                return (not w) or is_old(w)
        return False

    gone = [rid for rid, row in snapshot.items() if _stale(row)]
    for rid in gone:
        snapshot.pop(rid, None)
    if payload is not None and gone:
        payload["methods"] = [m for m in payload.get("methods", [])
                              if not _stale(m)]
    return len(gone)


@contextlib.contextmanager
def _construction_budget(seconds: float = _CONSTRUCTION_MAX_S):
    """Blok suresince E.CONSTRUCTION_DEADLINE'i kurar (mutlak son an) ve cikista
    her kosulda temizler -- deadline None'ken motor kontrolleri no-op oldugundan
    metasezgisel asamalarin icindeki kurucu cagrilari kapaktan ETKILENMEZ."""
    E.CONSTRUCTION_DEADLINE = time.perf_counter() + seconds
    try:
        yield
    finally:
        E.CONSTRUCTION_DEADLINE = None


def _rotation_subset_size(n: int, sparsity: float | None) -> int:
    """n<=_ROTATION_SUBSET_MIN icin n'in kendisini (tam kume), uzerinde ise
    round(n*sparsity) dondurur (en az 1 nokta). `sparsity` None ise
    _ROTATION_DEFAULT_SPARSITY kullanilir."""
    if n <= _ROTATION_SUBSET_MIN:
        return n
    ratio = _ROTATION_DEFAULT_SPARSITY if sparsity is None else float(sparsity)
    ratio = min(1.0, max(0.0, ratio))
    return max(1, min(n, round(n * ratio)))


def _theta_hist_candidates(coords, n, sample_size=2000,
                           seed=_ROTATION_SUBSET_SEED):
    """Komsu-yon histogrami: rastgele bir orneklemde her noktanin en yakin
    komsusuna giden vektorun acisi (mod 180) histogramlanir; tepe(ler) baskin
    satir/dizi dogrultusunu verir. Iki aday uretilir: satirlari serit-ici
    supurme yonune hizalayan (90-phi) ve seritlere dik birakan (-phi) --
    hangisinin kazandigina gercek snake maliyeti karar verir."""
    rng = random.Random(seed + 1)
    s = min(sample_size, n)
    sample = rng.sample(range(n), s)
    sx = [coords[i][0] for i in sample]
    sy = [coords[i][1] for i in sample]
    g = max(1, int(math.sqrt(s)))
    minx, maxx = min(sx), max(sx)
    miny, maxy = min(sy), max(sy)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    cells = {}
    def cell(i):
        return (int((sx[i] - minx) / w * g * 0.999),
                int((sy[i] - miny) / h * g * 0.999))
    for i in range(s):
        cells.setdefault(cell(i), []).append(i)
    hist = [0.0] * 180
    for i in range(s):
        ci, cj = cell(i)
        best_d, best_j = None, -1
        for r_ in range(0, 4):
            found = False
            for di in range(-r_, r_ + 1):
                for dj in range(-r_, r_ + 1):
                    if max(abs(di), abs(dj)) != r_:
                        continue
                    for j in cells.get((ci + di, cj + dj), ()):
                        if j == i:
                            continue
                        d = (sx[j] - sx[i]) ** 2 + (sy[j] - sy[i]) ** 2
                        if best_d is None or d < best_d:
                            best_d, best_j = d, j
                            found = True
            if best_d is not None and r_ >= 1 and not found:
                break
        if best_j < 0:
            continue
        ang = math.degrees(math.atan2(sy[best_j] - sy[i],
                                      sx[best_j] - sx[i])) % 180.0
        hist[int(ang) % 180] += 1.0
    sm = [sum(hist[(k + d) % 180] for d in range(-2, 3)) for k in range(180)]
    peaks = sorted(range(180), key=lambda k: -sm[k])
    top = []
    for p in peaks:
        if all(min(abs(p - q), 180 - abs(p - q)) > 8 for q in top):
            top.append(p)
        if len(top) >= 3:
            break
    def norm(a):
        while a > 90:
            a -= 180
        while a < -90:
            a += 180
        return a
    cands = []
    for phi in top:
        cands.append(norm(90 - phi))
        cands.append(norm(-phi))
    return cands


def _theta_axial_mean(coords, n, sample_size=2000, k=3,
                      seed=_ROTATION_SUBSET_SEED):
    """Komsu yonlerinin DOGRU ortalamasi: yonsel istatistik, iki harmonik.

    Rastgele orneklemdeki her noktanin k en yakin komsusuna giden vektorlerin
    yonu phi icin 1/d agirlikli birim vektorler toplanir; aci histogram
    kutusuna yuvarlanmadigi icin sonuc SUREKLIDIR (kesirli derece).

    Iki harmonik birden hesaplanir, cunku duz aritmetik/tek-harmonik ortalama
    izgara verisinde COKER:
      mod-180 (cift aci, cos2phi/sin2phi): tek baskin dogrultu (satirli PCB).
      mod-90 (dortlu aci, cos4phi/sin4phi): KARE IZGARA — satir ve sutun
        modlari diktir, cift-aci uzayinda birbirini goturur (R2~0); dortlu
        aci ikisini ayni yone katlar, izgara yonelimi mod 90 olarak cikar.

    Returns (phi180_deg, R2, phi90_deg, R4); R'ler (0..1) bileske uzunlugu =
    o harmonikteki yogunlasma. Kullanici hangisi buyukse onu almali."""
    rng = random.Random(seed + 2)
    s = min(sample_size, n)
    sample = rng.sample(range(n), s)
    sx = [coords[i][0] for i in sample]
    sy = [coords[i][1] for i in sample]
    g = max(1, int(math.sqrt(s)))
    minx, maxx = min(sx), max(sx)
    miny, maxy = min(sy), max(sy)
    w = (maxx - minx) or 1.0
    h = (maxy - miny) or 1.0
    cells = {}
    def cell(i):
        return (int((sx[i] - minx) / w * g * 0.999),
                int((sy[i] - miny) / h * g * 0.999))
    for i in range(s):
        cells.setdefault(cell(i), []).append(i)
    C2 = S2 = C4 = S4 = W = 0.0
    for i in range(s):
        ci, cj = cell(i)
        neigh = []   # (d2, j) — k en yakin
        for r_ in range(0, 4):
            for di in range(-r_, r_ + 1):
                for dj in range(-r_, r_ + 1):
                    if max(abs(di), abs(dj)) != r_:
                        continue
                    for j in cells.get((ci + di, cj + dj), ()):
                        if j == i:
                            continue
                        d2 = (sx[j] - sx[i]) ** 2 + (sy[j] - sy[i]) ** 2
                        neigh.append((d2, j))
            if len(neigh) >= k and r_ >= 1:
                break
        neigh.sort()
        for d2, j in neigh[:k]:
            d = math.sqrt(d2) or 1e-12
            phi = math.atan2(sy[j] - sy[i], sx[j] - sx[i])
            wgt = 1.0 / d   # yakin komsu (satir arkadasi) daha cok soz sahibi
            C2 += wgt * math.cos(2.0 * phi)
            S2 += wgt * math.sin(2.0 * phi)
            C4 += wgt * math.cos(4.0 * phi)
            S4 += wgt * math.sin(4.0 * phi)
            W += wgt
    if W <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    R2 = math.hypot(C2, S2) / W
    R4 = math.hypot(C4, S4) / W
    phi180 = math.degrees(0.5 * math.atan2(S2, C2)) % 180.0
    phi90 = math.degrees(0.25 * math.atan2(S4, C4)) % 90.0
    return phi180, R2, phi90, R4


def _theta_pca(coords, n, sample_size=2000, seed=_ROTATION_SUBSET_SEED):
    """P1-7 PCA taban cizgisi: nokta bulutunun kovaryans matrisinin baskin
    OZVEKTORUNUN yonu ile theta kestirimi (rotation_hist/rotation_mean ile
    ayni ornekleme sozlesmesi; dogrulama yine strip turunun gercek maliyetiyle
    yapilir -- dedektor yalniz aday uretir).

    2x2 kovaryansin baskin ozvektorunun acisi kapali formda:
        phi = 0.5 * atan2(2*cov_xy, var_x - var_y)   (mod 180, eksenel)
    Strip tarama yonu baskin eksene DIK degil ona PARALEL secildiginde serit
    sayisi en aza iner; mevcut dedektorlerle ayni donusum kullanilir:
    aday = 90 - phi (ve mod-180 eslenigi). Kare-izgara rejiminde PCA eksenleri
    esit agirlikli oldugundan ikinci ozvektor de aday olarak verilir.

    Returns (theta_deg, R) — R = lambda1/(lambda1+lambda2) baskinlik orani
    (0.5: yon yok, 1.0: tek dogrultu). Bilgi amacli; aday secimi etkilenmez."""
    rng = random.Random(seed + 3)
    s = min(sample_size, n)
    sample = rng.sample(range(n), s)
    mx = sum(coords[i][0] for i in sample) / s
    my = sum(coords[i][1] for i in sample) / s
    vxx = vyy = vxy = 0.0
    for i in sample:
        dx = coords[i][0] - mx
        dy = coords[i][1] - my
        vxx += dx * dx
        vyy += dy * dy
        vxy += dx * dy
    vxx /= s; vyy /= s; vxy /= s
    tr = vxx + vyy
    det = vxx * vyy - vxy * vxy
    disc = max(0.0, tr * tr / 4.0 - det)
    lam1 = tr / 2.0 + math.sqrt(disc)
    R = (lam1 / tr) if tr > 1e-12 else 0.5
    phi = math.degrees(0.5 * math.atan2(2.0 * vxy, vxx - vyy)) % 180.0
    return phi, R


def _best_strip_angle_geometric(inst, n, coords, ewt, mode="hist", sparsity=None):
    """theta* dedektorleri "rotation_hist"/"rotation_mean": aday acilari SALT
    GEOMETRIDEN uretir (tur insa etmeden), sonra her adayi STRIP turunun
    GERCEK maliyetiyle (inst.tour_cost, TSPLIB-exact) dar pencerede dogrular.
    Eski snake-tabanli dogrulama KALDIRILDI: aci artik strip ICIN bulunur ve
    strip turuna uygulanir ("aci bulunuyorsa strip icin bulunuyor").

      "hist"  komsu-yon histogrami adaylari (_theta_hist_candidates), her
              aday +-1 derece pencerede 0.5'lik adimla incelenir.
      "mean"  eksenel cift-aci dairesel ORTALAMA (_theta_axial_mean): baskin
              dogrultu surekli (kesirli derece) hesaplanir; adaylar +-2
              derece pencerede 0.25'lik adimla incelenir. En hassas tahmin.

    `sparsity` yalniz aday uretiminin IC orneklem boyutunu etkiler
    (_theta_hist_candidates/_theta_axial_mean); strip dogrulamasi O(n log n)/
    aci kadar ucuz oldugundan HER ZAMAN tam kumede, gercek maliyetle yapilir
    (eski snake dogrulamasinin alt-kume mekanizmasina gerek kalmadi).

    GUARANTEE: theta=0 her zaman degerlendirilir ve dogrulama zaten tam-kume
    strip'in gercek maliyetiyle yapildigindan secilen aci theta=0 strip'ten
    HICBIR ZAMAN kotu olamaz (ayri bir "full build" karsilastirma adimina
    gerek yok -- tarama karsilastirmanin kendisidir).

    Returns (theta_deg, tour, cost, diag); diag = {"search_time", "best_c",
    "worst_c", "worst_a"}. search_time fonksiyonun TUM suresi = arama eforu
    (aday uretimi + strip dogrulamalari; son tur zaten dogrulama sirasinda
    kurulan turdur, ekstra insa maliyeti yoktur). worst_c/worst_a yalniz DAR
    aday pencerelerini kapsar -- TEMSILI DEGILDIR; caller (run_dataset) bu
    yuzden hist/mean icin savings/worst_cost alanlarini DOLDURMAZ (yapisal
    olarak musait degil; temsili "en kotu" yalniz tam [-90,90] tarayan
    rotation_strip'te raporlanir)."""
    t_search0 = time.perf_counter()
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    scan_sample_size = _rotation_subset_size(n, sparsity)

    cache = {}

    def strip_cost(deg):
        deg = round(deg, 2)
        if deg in cache:
            return cache[deg]
        c = _strip_tour_at_angle(inst, xs, ys, deg)[1]
        cache[deg] = c
        return c

    best_a, best_c = 0.0, strip_cost(0.0)

    def sweep(lo, hi, step):
        nonlocal best_a, best_c
        a = lo
        while a <= hi + 1e-9:
            if -90.0 <= a <= 90.0:
                c = strip_cost(a)
                if c < best_c - 1e-9:
                    best_a, best_c = a, c
            a += step

    def _norm90(a):
        while a > 90:
            a -= 180
        while a < -90:
            a += 180
        return a

    if mode == "pca":
        # P1-7 taban cizgisi: kovaryans baskin ozvektorunden aday uretimi;
        # dogrulama/dar-pencere sozlesmesi "mean" ile ayni (+-2 derece, 0.25
        # adim, kesirli adayin kendisi de denenir). Ikinci ozvektor (dik eksen)
        # kare-izgara rejimini karsilamak icin ek adaydir.
        phi_pca, _R = _theta_pca(coords, n, sample_size=scan_sample_size)
        cands = []
        for base_cand in (_norm90(90.0 - phi_pca), _norm90(-phi_pca)):
            if base_cand not in cands:
                cands.append(base_cand)
        for cand in cands:
            if -90.0 <= cand <= 90.0:
                c = strip_cost(cand)     # kesirli adayin kendisi
                if c < best_c - 1e-9:
                    best_a, best_c = round(cand, 2), c
            for base in (cand - 180.0, cand, cand + 180.0):
                if -92.0 <= base <= 92.0:
                    sweep(base - 2.0, base + 2.0, 0.25)
    elif mode == "mean":
        # surekli aci: hangi harmonik daha yogunsa (R) onun ortalamasi
        # kullanilir; adaylar dar pencerede 0.25'lik adimla incelenir ve
        # kesirli acinin kendisi de aynen denenir
        cands = []
        phi180, R2, phi90, R4 = _theta_axial_mean(coords, n, sample_size=scan_sample_size)
        if R4 >= R2 and R4 > 1e-9:
            # kare-izgara rejimi: yonelim mod 90 — iki temsilci aci
            cands = [_norm90(-phi90), _norm90(90.0 - phi90)]
        elif R2 > 1e-9:
            # tek baskin dogrultu (satirli PCB): mod 180
            cands = [_norm90(90.0 - phi180), _norm90(-phi180)]
        for cand in cands:
            if -90.0 <= cand <= 90.0:
                c = strip_cost(cand)     # kesirli adayin kendisi
                if c < best_c - 1e-9:
                    best_a, best_c = round(cand, 2), c
            # +-2 derece pencere: ortalama kestirimin ~1-2 derecelik olasi
            # sapmasini karsilar; 0.25'lik adim tarama-tabanli varyantlardan
            # daha hassas kalir. Aci uzayi mod-180 DONGUSELDIR: aday -89'daysa
            # optimum +89.5'te olabilir (aralarinda mod-180'de 1.3 derece var)
            # -- pencere 180 kaydirilmis kopyalariyla birlikte taranir, sweep
            # zaten [-90,90] disini atlar.
            for base in (cand - 180.0, cand, cand + 180.0):
                if -92.0 <= base <= 92.0:
                    sweep(base - 2.0, base + 2.0, 0.25)
    else:  # "hist"
        cands = _theta_hist_candidates(coords, n, sample_size=scan_sample_size)
        for cand in cands:                # histogram aday pencereleri (0.5'lik)
            sweep(cand - 1.0, cand + 1.0, _ROTATION_FINE_STEP)

    search_time = time.perf_counter() - t_search0
    cache_vals = list(cache.values())
    diag = {
        "search_time": search_time,
        "best_c": min(cache_vals) if cache_vals else None,
        "worst_c": max(cache_vals) if cache_vals else None,
        # cache anahtarlari zaten (yuvarlanmis) taranan acilar -- en kotu
        # maliyetin HANGI acida bulundugunu ayrica raporlamak icin
        "worst_a": max(cache, key=cache.get) if cache else None,
    }
    if abs(best_a) < 1e-6:
        best_a = 0.0
    tour, cost = _strip_tour_at_angle(inst, xs, ys, best_a)
    return best_a, tour, cost, diag


def _scan_theta_proxy(inst, xs, ys, coarse_step=5.0):
    """Full-range theta-star scan on the STRIP tour: rotate the points, lay a
    simple boustrophedon strip, score it with the TRUE TSPLIB cost (via
    inst.tour_cost), and return the angle that minimises it. O(n) per angle,
    so fast even on sparse/large instances. The [-90,90] sweep with x-strips
    already covers both axis orientations. Using the true cost (not raw
    Euclidean) makes it faithful for GEO too. `coarse_step` (degrees, UI
    "Rotasyon kaba adim" slider) sets the primary sweep resolution; the best
    region is always refined at +-4 deg / 1-deg steps afterwards.
    Returns (theta_deg in [-90, 90], best_cost, worst_cost, worst_theta_deg,
    elapsed_s) -- best/worst are the min/max STRIP cost seen across every
    candidate angle evaluated, so "worst - best" is a genuine, representative
    improvement figure (this scan covers the FULL [-90,90] range, unlike
    hist/mean's narrow candidate-window checks -- see caller);
    worst_theta_deg is the (un-normalized, raw sweep) angle where that worst
    cost occurred, so the UI can show "which angle would have been the bad
    choice", not just how bad it was. `elapsed_s` is this function's OWN
    wall time -- pure search effort. theta=0 is always on the sweep grid
    (the sweep starts at -90 with a step that divides 90 for the default 5),
    and the caller compares against the theta=0 strip anyway, so the chosen
    angle can never lose to no-rotation."""
    t0 = time.perf_counter()
    best_a, best_c = 0.0, float("inf")
    worst_c, worst_a = float("-inf"), 0.0

    def ev(deg):
        nonlocal worst_c, worst_a
        rx, ry = E.rotate_coords(xs, ys, deg)
        c = inst.tour_cost(E.snake_order(rx, ry, 0))
        if c > worst_c:
            worst_c, worst_a = c, deg
        return c

    # Primary sweep: -90 deg .. +90 deg in coarse_step-degree steps (find the
    # best strip-sweep angle), then a fine +/-4 deg / 1-deg refinement, and
    # theta=0 explicitly (in case coarse_step doesn't land on 0).
    step = max(1.0, float(coarse_step))
    best_c, best_a = ev(0.0), 0.0
    a = -90.0
    while a <= 90.0 + 1e-9:
        c = ev(a)
        if c < best_c:
            best_c, best_a = c, a
        a += step
    a = best_a - 4.0
    while a <= best_a + 4.0 + 1e-9:
        c = ev(a)
        if c < best_c:
            best_c, best_a = c, a
        a += 1.0
    norm = best_a
    while norm > 90.0:
        norm -= 180.0
    while norm < -90.0:
        norm += 180.0
    return norm, best_c, worst_c, worst_a, time.perf_counter() - t0


def _best_strip_angle_from_scan(inst, xs, ys, coarse_step=5.0):
    """theta* dedektoru "rotation_strip": [-90,90] TAM taramayi STRIP'in
    gercek maliyetiyle yapar (_scan_theta_proxy -- her aday acida yalniz O(n)
    bir boustrophedon kurar; kaba adim + en iyi bolge cevresinde 1-derece
    ince tarama). Aci artik dogrudan STRIP ICIN bulunur ve strip turuna
    uygulanir ("aci bulunuyorsa strip icin bulunuyor") -- eski snake-tabanli
    dogrulama/insa tamamen KALDIRILDI. theta=0 taramada her zaman
    degerlendirildiginden secilen aci theta=0 strip'ten asla kotu olamaz.

    Returns (theta_deg, tour, cost, diag) -- diag = {"search_time", "best_c",
    "worst_c", "worst_a"}: search_time yalniz taramanin kendi suresi (saf
    arama eforu; son tur zaten taramada kurulmus olan aciyla O(n) yeniden
    kurulur, ihmal edilebilir); best_c/worst_c TAM [-90,90] taramadaki en
    ucuz/en pahali strip maliyeti -- TEMSILI oldugu icin caller savings/
    worst_cost alanlarini bunlardan doldurur; worst_a en kotu maliyetin
    bulundugu aci ("hangi aci kotuydu" sorusuna cevap)."""
    th, best_c, worst_c, worst_a, search_time = _scan_theta_proxy(
        inst, xs, ys, coarse_step=coarse_step)
    diag = {"search_time": search_time, "best_c": best_c,
            "worst_c": worst_c, "worst_a": worst_a}
    if abs(th) < 1e-6:
        th = 0.0
    tour, cost = _strip_tour_at_angle(inst, xs, ys, th)
    return th, tour, cost, diag


# ---------------------------------------------------------------------------
# SURECLER-ARASI GPU KILIDI: dashboard'un toplu (paralel) kosusu her veri
# kumesini AYRI bir runner.py sureci olarak baslatir. Tek bir 8 GB GPU'ya
# ayni anda birden fazla surec yuklenirse VRAM cakismasi/OOM/rastgele CUDA
# hatalari olusur VE es zamanli isler birbirinin duvar-saati suresini sisirir.
# Bu kilit, GPU kullanan HER yontem blogunu makine capinda seri hale getirir
# (dosya kilidi -- ayni anda tek surec); CPU yontemleri paralel kalir.
# ONEMLI: kilidi BEKLEME suresi yontemin raporlanan suresine KATILMAZ --
# zamanlayici kilit alindiktan SONRA baslatilir (bkz _gpu_turn), boylece
# paralel toplu kosudaki sure olcumleri tek-basina kosuyla karsilastirilabilir
# kalir.
# ---------------------------------------------------------------------------
# COK GPU'LU DUGUM (TRUBA): kilit TEK dosya oldugu icin varsayilanda TUM GPU
# isleri makine capinda seri koşar -- 4 GPU'lu bir dugumde 3 GPU bos durur.
# Dagitici (hpc/dispatcher.py) her isciye kendi CUDA_VISIBLE_DEVICES'ini ve
# ONA AIT ayri bir kilit dosyasini SNEAKPATH_GPU_LOCK ile verir; boylece
# serilestirme GPU BASINA yapilir. Degisken yoksa davranis eskisiyle AYNI.
_GPU_LOCK_PATH = Path(os.environ.get("SNEAKPATH_GPU_LOCK")
                      or (HERE / ".gpu.lock"))

@contextlib.contextmanager
def _gpu_lock():
    f = open(_GPU_LOCK_PATH, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    # LK_LOCK ~10 s bekleyip OSError atar; kilit uzun sure
                    # doluysa dongude beklemeye devam ederiz (bloklayici).
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


@contextlib.contextmanager
def _gpu_turn(key):
    """GPU sirasi: kilidi al, beklenen sureyi logla (yontem suresine dahil
    edilmez -- blok icindeki zamanlayicilar kilit alindiktan sonra baslar)."""
    t_wait = time.perf_counter()
    with _gpu_lock():
        waited = time.perf_counter() - t_wait
        if waited > 0.1:
            _log(f"   [gpu-kilit] {key}: sirada {waited:.1f}s bekledi "
                 f"(es zamanli baska bir GPU isi vardi; bu bekleme yontem "
                 f"suresine dahil DEGIL)")
        yield


@contextlib.contextmanager
def _quiet():
    """Silences the optimizers' verbose progress prints."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _log(msg):
    print(msg, file=sys.__stderr__, flush=True)


def _scale_config(n: int, time_budget: float):
    """Scale iteration budgets / time caps / dense levels by instance size.

    Metaheuristic keys explained:
      ils_iter   : perturb+search cycles for ILS (NOT the deterministic max_iter)
      ils_rounds : local-search rounds per descent (max_ls_rounds). Each ILS
                   descent (incl. the uninterruptible Phase-0 init LS) runs this
                   many full passes — the single biggest lever on wall time at
                   large n, since the time cap only fires *between* iterations.
      nb2/nbo/rk/rnb : 2-opt / or-opt / relocate-k / relocate neighbour limits.
      gpu_time   : batched GPU onarım katmanının (repair_tours) ortak duvar-saati
                   bütçesi — ge_gpu, gpu_hybrid, GPU bant/stratified hibritleri ve
                   P0-1 klasik+onarım satırları (ge_repair/nn_repair/fi_repair)
                   hep AYNI bütçeyi görür (adil kıyas).
    """
    tb = time_budget
    # Seeds default to >=10 on small/medium instances so the multi-seed stats
    # (mean +/- CI, Wilcoxon) are meaningful; reduced on large ones for runtime.
    if n <= 1000:
        cfg = dict(max_iter=1000, ils_iter=500, ils_time=30 * tb,
                   seeds=10, ils_rounds=10, nb2=20, nbo=20, rk=30, rnb=120,
                   gpu_time=20 * tb)
    elif n <= 2500:
        cfg = dict(max_iter=800, ils_iter=400, ils_time=45 * tb,
                   seeds=10, ils_rounds=8, nb2=16, nbo=16, rk=24, rnb=100,
                   gpu_time=30 * tb)
    elif n <= 5000:
        cfg = dict(max_iter=500, ils_iter=180, ils_time=70 * tb,
                   seeds=5, ils_rounds=6, nb2=12, nbo=12, rk=18, rnb=60,
                   gpu_time=45 * tb)
    elif n <= 9000:
        cfg = dict(max_iter=300, ils_iter=80, ils_time=110 * tb,
                   seeds=2, ils_rounds=4, nb2=10, nbo=10, rk=14, rnb=40,
                   gpu_time=60 * tb)
    else:  # > 9000 (usa13509): tightest budget
        cfg = dict(max_iter=180, ils_iter=30, ils_time=120 * tb,
                   seeds=1, ils_rounds=3, nb2=8, nbo=8, rk=10, rnb=24,
                   gpu_time=90 * tb)
    # dense levels: scale down for large n
    if n <= 2500:
        cfg["dense"] = E.lro.DEFAULT_DENSE_ADAPTIVE_LEVELS
    elif n <= 6000:
        cfg["dense"] = (lro.DenseAdaptiveLevel(18, 64, 0.40, 2),
                        lro.DenseAdaptiveLevel(36, 120, 0.65, 3),
                        lro.DenseAdaptiveLevel(72, 200, 0.85, 4))
    else:
        cfg["dense"] = (lro.DenseAdaptiveLevel(18, 50, 0.35, 2),
                        lro.DenseAdaptiveLevel(36, 90, 0.60, 3))
    return cfg


def _downsample_history(history, max_points=200):
    """history: list of HistoryPoint -> [[elapsed_time, cost], ...] downsampled."""
    pts = [[round(h.elapsed_time, 4), round(h.cost, 2)] for h in history]
    if len(pts) <= max_points:
        return pts
    step = len(pts) / max_points
    out = [pts[int(i * step)] for i in range(max_points)]
    out[-1] = pts[-1]
    return out


def _gap(cost, bks):
    return None if not bks else round(100.0 * (cost - bks) / bks, 4)


# ---- statistics for multi-seed reporting (pure stdlib, no scipy) -------------
# Two-sided t critical values at 95% confidence, indexed by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def _mean_ci95(values):
    """95% confidence interval of the mean (Student-t). Returns (low, high)."""
    n = len(values)
    m = statistics.mean(values)
    if n < 2:
        return (m, m)
    se = statistics.stdev(values) / math.sqrt(n)
    df = n - 1
    t = _T95.get(df) or (2.042 if df > 30 else 1.96)
    return (m - t * se, m + t * se)


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _wilcoxon_signed_rank(x, y):
    """Paired Wilcoxon signed-rank test (normal approximation with tie/continuity
    correction). Returns the statistic W, z, two-sided p, and effective n.
    Pure stdlib; for tiny n treat the p-value as indicative."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return {"W": None, "z": None, "p": None, "n": 0}
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    W = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return {"W": round(W, 2), "z": None, "p": None, "n": n}
    z = (W - mean_w + 0.5) / math.sqrt(var_w)  # continuity correction
    p = 2.0 * _normal_cdf(-abs(z))
    return {"W": round(W, 2), "z": round(z, 3), "p": round(min(1.0, p), 4), "n": n}


def _hyperparameters(cfg):
    """Full, reportable record of every hyperparameter used in a run, so the
    paper can cite exact settings and runs are reproducible (review item P2-12).
    Centralizes the constants that were previously scattered across modules."""
    return {
        "distance": "TSPLIB-exact (EUC_2D nint, GEO formula)",
        "construction": "deterministic theta*-strip: boustrophedon strip tour "
                        "built at the best theta* detector's angle (falls back "
                        "to theta=0 when no detector improves it); this seeds "
                        "the repair/optimization pipeline",
        "strip": {
            "method": "fixed-strip-count boustrophedon (space-filling-curve family); "
                      "strips = round(sqrt(n/2)), no grid/cell-size search — the "
                      "theta=0 BASE construction whose sweep angle the theta* "
                      "detectors (rotation_*) optimize",
            "reference": "Bartholdi & Platzman (1988), 'Heuristics based on "
                         "spacefilling curves for combinatorial problems in "
                         "Euclidean space', Networks 14(1) — classical baseline "
                         "family for large-scale Euclidean TSP construction",
        },
        "hilbert": {
            "method": "Hilbert-curve construction: points are gridded onto a "
                      "2^16 square and sorted by their 1-D Hilbert-curve "
                      "distance (bit-rotation xy2d); same family/complexity as "
                      "strip and Snake-Grid, no local cell-size/axis search",
            "reference": "Platzman & Bartholdi (1989), 'Spacefilling curves and "
                         "the planar travelling salesman problem', J. ACM 36(4) "
                         "— the classical reference point for the space-filling-"
                         "curve family (FAMILY='curve'); same-class comparison "
                         "peer for Snake-Grid",
        },
        "morton": {
            "method": "Morton/Z-order construction: same grid-and-sort recipe "
                      "as the Hilbert-curve tour, cheaper bit-interleave key "
                      "instead of the rotation-based Hilbert distance -- lower "
                      "locality, included as the family's simpler member",
            "reference": "Morton (1966) Z-order encoding, as applied to the "
                         "space-filling-curve TSP family described in Platzman "
                         "& Bartholdi (1989); FAMILY='curve'",
        },
        "nearest_insertion": {
            "method": "classic Nearest-Insertion: repeatedly inserts the unvisited "
                      "city nearest to the current partial tour at its cheapest "
                      "insertion edge; O(n^2), no candidate-list acceleration",
            "reference": "Rosenkrantz, Stearns & Lewis (1977), 'An analysis of "
                         "several heuristics for the traveling salesman problem', "
                         "SIAM J. Computing 6(3) — standard literature baseline",
        },
        "farthest_insertion": {
            "method": "classic Farthest-Insertion: repeatedly inserts the "
                      "unvisited city farthest from the current partial tour "
                      "(maximises the minimum distance to the tour) at its "
                      "cheapest insertion edge; O(n^2), no acceleration",
            "reference": "Rosenkrantz, Stearns & Lewis (1977), as above",
        },
        "greedy_edge": {
            "method": f"Greedy-Edge: sorts k-NN candidate edges (k={15}) by "
                      "length and adds each unless it creates degree>2 or an "
                      "early sub-cycle (union-find); remaining fragment "
                      "endpoints stitched by nearest-pair join",
            "reference": "Johnson & McGeoch (1997), as cited in "
                         "benchmark_methodology_reference above — the 'Greedy' "
                         "construction baseline",
        },
        "frame_experiment": {
            "method": "Cerceve (aci) duyarliligi deneyi -- frame_methods.py; "
                      "ilkeller paper_experiments/frame_experiment.py ile TEK "
                      "KAYNAK. fs_<kurucu>: [-90,90) aci taramasi (adim = "
                      "rotation_step, n buyudukce kabalasir), satir theta=0 "
                      "turu, savings = en kotu - en iyi aci maliyeti; "
                      "fs_ge8_jitter: theta=0, tam-beraberlik jitter (1e-9) "
                      "tohumlari; fs_ge8_detied: 1e-4 x medyan-NN gurultulu "
                      "(beraberliksiz) koordinatta tarama; fs_band_ge8: b in "
                      f"{list(FM.BAND_COUNTS)} bant, bant-ici GE k=8; realign_*: "
                      "kart phi ile dondurulur (panel aci kutusu, varsayilan "
                      f"{FM.DEFAULT_MISALIGN_DEG} derece), comb/nndir/pca "
                      "dedektorleri + hiyerarsik kafes inceltme (snap) ile geri "
                      "hizalanir, kurucu geri hizalanmis kartta kurulur.",
            "thesis": "Donme-degismez kurucular (GE, NN, FI) icin aci yalnizca "
                      "beraberlik-kirma zaridir (rot ~ jitter, detied ~ 0); "
                      "cerceveye bagimli kurucular (strip, Hilbert, Morton, "
                      "bantli GE) aciya buyuk olcude duyarlidir; hizasiz VLSI "
                      "karti kafes birebir geri gelecek sekilde hizalanabilir.",
            "cost": "her tur ORIJINAL koordinatlarda TSPLIB yuvarlamasiyla",
        },
        "rgge": {
            "method": "RGGE — Rotated-Grid Greedy Edge (Döndürülmüş Izgara "
                      "Greedy-Edge): koordinatlar θ açısında döndürülür, "
                      "isteğe bağlı olarak eksen sırası DEVRİK alınır ve DÜZ "
                      f"greedy-edge koşulur (k={RG.RGGE_KNN}). Kurucu mekaniği "
                      "hiç değişmez; değişen tek şey greedy-edge'in gördüğü "
                      "ADAY LİSTESİNİN iç yapısıdır. TEK TUR — havuz YOK.",
            "identity": "RGGE(θ=0, ızgara düz) == greedy_edge@knn8_greedy, "
                        "çevrim olarak BİT-AYNI. Kıyas EŞİT BÜTÇEDEDİR "
                        "(1 tur ↔ 1 tur). Sağlama: "
                        "verification/verify_rgge.py.",
            "rows": {
                "rgge_y": "ızgara DEVRİK — (px,py) → (py,px)",
                "rgge_x": "ızgara DÜZ — döndürme dışında dokunulmaz",
            },
            "axis_channel": "Eksen takası GEOMETRİK BİR DÖNÜŞÜM DEĞİLDİR: "
                            "mesafeler ve beraberlik-dışı her şey aynıdır. "
                            "Değişen şey grid_knn'in hücre "
                            "numaralandırmasıdır, yani EŞ-UZAKLIKLI adaylar "
                            "arasındaki kırpma. İki satırın farkı 'kazanç "
                            "rotasyondan mı ızgara yeniden-"
                            "numaralandırmasından mı?' sorusunu tek "
                            "değişkenli olarak cevaplar.",
            "mechanism": "Rotasyon geometrik bir arama DEĞİL, YAPILI BİR "
                         "BERABERLİK KIRMA operatörüdür. Öklid mesafesi "
                         "rotasyon değişmezidir; tur yine de değişir çünkü "
                         "eş-uzaklıklı komşular/kenarlar döndürmeden sonra "
                         "float'ta ~1e-13 ayrışır. Kesin (O(n²)) kNN ile de "
                         "etki durur, yani 'yaklaşık ızgara' açıklaması "
                         "yanlışlanmıştır. Etki beraberlik bolluğuyla "
                         "sınırlıdır: beraberlikli örneklerde ort. %1.81, "
                         "beraberliksizlerde %0.06 (kardeş proje, 87 örnek).",
            "mechanism_measure": "rgge.tie_measure → (sınır_beraberlik, "
                                 "tekrarlı_uzunluk). YALNIZ rgge satırlarına "
                                 "yazılır (kullanıcı kararı: diğer "
                                 "yöntemlere bulaştırılmaz). sınır_beraberlik "
                                 "= 0 olan bulutlarda çerçeve değişikliğinin "
                                 "etkisi YAPISAL olarak ihmal edilebilir; "
                                 "satır bunu log'a not düşer. UYARI: "
                                 "tekrarlı_uzunluk'un ~0.4'lük bir tabanı "
                                 "vardır (karşılıklı komşu çiftleri iki kez "
                                 "sayılır) — MUTLAK değeri değil, örnekler "
                                 "ARASI karşılaştırması okunmalıdır.",
            "no_pool": "Havuz BİLEREK yok (2026-09-07 kullanıcı kararı). "
                       "Kardeş projede havuzlu hâli ölçülmüştü ve iddiası "
                       "zayıftı: yön×genişlik havuzu, tie-jitter havuzuyla "
                       "istatistiksel olarak AYIRT EDİLEMİYOR (38/39, "
                       "p = 1.0). Tek tur olarak ise rakiple eşit bütçededir.",
            "source": "Çekirdek kardeş projedeki knn_pool.py'nin TEK ADAY "
                      "kurucusundan ve beraberlik ölçüsünden birebir "
                      "taşınmıştır; havuz makinesi taşınmamıştır. Ölçümler: "
                      "o projenin ADAY_LISTESI_PERTURBASYONU_AKADEMIK_NOT.md.",
        },
        "rsge": {
            "method": "RSGE — Rotated Serpentine Greedy Edge (Döndürülmüş "
                      "Serpantin Greedy-Edge): koordinatlar θ' açısında "
                      "döndürülür, döndürülmüş x ekseni boyunca b eşit-"
                      f"GENİŞLİKLİ bant açılır, her bandın içi bağımsız bir "
                      f"greedy-edge çevrimiyle kurulur (k={SA.RSGE_KNN}, "
                      "projenin aday-listesi tavanı; RGGE ile AYNI "
                      "genişlik), bantlar serpantin sırasında ileriye "
                      "bakışlı yönelim + bant sınırı dikişiyle bağlanır. "
                      "TEK tur kurar (havuz YOK).",
            "frame": "θ' TEK sayıdır: dedektör yolunda (varsayılan "
                     "grid_theta) çizgilere dik olma koşulu θ'ya 90 EKLENEREK "
                     "temsil edilir, eksen takası KULLANILMAZ — takas bir "
                     "yansımadır ve tamsayı kafeslerde greedy-edge'in k-NN "
                     "beraberlik bozmasını %3.6'ya kadar oynatır (ölçüldü). "
                     "Elle verilen açıda (zero / manual:<derece>) katlama "
                     "YAPILMAZ: θ=0 ablasyonunun anlamı 'hiçbir hizalama "
                     "kararı yok'tur. AÇI SEÇİMİ (2026-09-07): satır YALNIZ "
                     "panelde seçili açı kaynağında koşar — tek satır, tek "
                     "açı. θ=0 çapasını görmek için panelden 'θ = 0°' "
                     "seçilir.",
            "identity": "RSGE(θ'=0, b=1) == greedy_edge@knn8_greedy, çevrim "
                        "olarak BİT-AYNI. Dört RSGE satırı ile rakip "
                        "`greedy_edge` aynı ailenin b eksenindeki "
                        "noktalarıdır. Sağlama: verification/verify_rsge.py.",
            "band_count_rules": {
                "rsge_resource": f"b = ceil(n / m*), m* = {SA.RSGE_M_STAR} "
                                 "(bir işçinin/aday listesinin tuttuğu en "
                                 "büyük bant) — 'kısıtın izin verdiği EN "
                                 "KÜÇÜK b'",
                "rsge_corridor": f"b = 1 + #{{boşluk > C × medyan boşluk}}, "
                                 f"C = {SA.RSGE_CORRIDOR_C:g} — kesme ekseni "
                                 "izdüşümünden, VERİDEN",
                "rsge_budget": f"b = ceil(c·n²/T), c = {SA.RSGE_BUDGET_C:g} "
                               f"s/nokta², T = {SA.RSGE_BUDGET_T:g} s — "
                               "measurements_jpdc/fi_bant_deneyi'nde ÖLÇÜLEN "
                               "maliyet modeli (p = +0.87…+1.01). Model "
                               "KARESEL bir kurucu için kalibre edildi; "
                               "greedy-edge O(n log n) olduğundan bu satırın "
                               "beklenen davranışı bantlamayı REDDETMEKTİR",
                "rsge_fixed2": f"b = {SA.RSGE_FIXED_B} (sabit) — en küçük "
                               "aşikâr-olmayan bölme",
            },
            "single_variable": "Dört satırın farklı olduğu TEK şey `b` "
                               "tamsayısıdır: bölme şeması (eşit-GENİŞLİK), "
                               "açı kaynağı, çerçeve, bant-içi kurucu, k ve "
                               "dikiş BİREBİR aynıdır. Aynı b → BİREBİR aynı "
                               "tur (runner'ın inşa önbelleği buna dayanır).",
            "equal_width_note": "Paralel işçiler için dengeli (eşit-SAYI) "
                                "bant gerekir; kalite bedeli ölçüldü ve "
                                "küçüktür (k=8'de 39.0 ↔ 39.6, +0.6 puan; "
                                "IZGARA_ACISI_VE_SERIT_SAYISI_AKADEMIK_NOT.md "
                                "Bölüm 6). Yine de dördü de eşit-GENİŞLİK "
                                "kullanır — satırlar arası iki değişken "
                                "(b VE şema) kıyası okunamaz hale getirirdi.",
            "seed_role": "Dört satır ONARIM TOHUMU olarak seçilebilir "
                         "(SEED_CONSTRUCTIONS). Gerekçe doğrudan bulgudur: "
                         "ham inşa avantajının %95'i 2-opt/Or-opt sonrası "
                         "eriyor (measurements_jpdc/bant_ici_deneyi/"
                         "onarim_sonrasi.py), dolayısıyla 'bant kuralı "
                         "kazandırır mı?' sorusu onarım SONRASINDA da "
                         "sorulmalıdır. Tohum olarak KANONİK satırlarını "
                         "verirler; kanonik satır = PANELDE SEÇİLİ açıdaki "
                         "tek satırdır (bkz. _row_of / ANGLE_PANEL_CANONICAL), "
                         "yani tabloda görünen tur ile onarıma giren tur "
                         "AYNIdır. Kendileri tohum ALMAZ (kurucudur).",
            "claim": "ÜSTÜNLÜK İDDİASI TAŞIMAZ. Dört satır bir HİPOTEZİN "
                     "ölçümüdür ('bant sayısı doğru seçilirse bantlama "
                     "kazandırır'); kıyas eşleşi `greedy_edge`dir.",
        },
        "quick_boruvka": {
            "method": "Quick-Boruvka yaklaşımı (DIMACS güçlü kurucusu): çoklu "
                      "parça (fragment) birleştirme — her parça en yakın komşu "
                      "parçasına en kısa kenarla bağlanır (Borůvka turları), "
                      "kalan parçalar en-yakın-çift dikişiyle kapatılır; "
                      "greedy_edge ile aynı 'edge_greedy' sınıfı",
            "reference": "Bentley (1992), 'Fast algorithms for geometric "
                         "traveling salesman problems', ORSA J. Computing 4(4); "
                         "Johnson & McGeoch DIMACS protokolünün standart güçlü "
                         "kurucusu (P1-5 rakip kümesi)",
        },
        "glop_like": {
            "method": "GLOP'un öğrenmesiz yeniden-uyarlaması (adil kıyas rakibi): "
                      "özyinelemeli kd-tree bölümleme (~alt-problem 1000 düğüm) "
                      "-> her yaprakta greedy_edge + 2-opt -> bitişik yaprak "
                      "turları sınır kenarları üzerinden birleştirilir -> kısa "
                      "sınır-ötesi 2-opt geçişi. Öğrenme YOK (ağırlık/eğitim "
                      "verisi kullanılmaz); deterministik + tohumlu varyant",
            "reference": "Ye et al. (2023), 'GLOP: Learning Global Partition and "
                         "Local Construction for Solving Large-Scale Routing "
                         "Problems in Real-Time', NeurIPS 2023 — P1-5 tek güncel "
                         "rakibin öğrenmesiz karşılığı",
        },
        # --- EXACT / ALTIN STANDART referans cizgisi (rakip DEGIL) ---
        # Ikisi de her yontemin okudugu AYNI data/{ad}.tsp dosyasini okur ve
        # donen tur bu projenin KENDI inst.tour_cost'u (TSPLIB-exact) ile
        # puanlanir -> cozucunun kendi raporladigi maliyetle bizim tablodaki
        # maliyet sessizce ayrisamaz. Bir sezgisele TOHUM OLAMAZLAR.
        "lkh3": {
            "method": "harici cozucu koprusu (external_solvers.run_lkh); "
                      "LKH-3 kendi TIME_LIMIT parametresiyle sinirlanir "
                      "(panelden 'Exact yöntem süresi'), donen tur bu projenin "
                      "TSPLIB-exact mesafe fonksiyonlariyla puanlanir",
            "reference": "Helsgaun (2000/2017), 'An Effective Implementation of "
                         "the Lin-Kernighan Traveling Salesman Heuristic' / LKH-3 "
                         "— pratikte TSPLIB olcegindeki Oklid orneklerinde "
                         "optimale cok yakin; literaturun fiili altin standardi",
            "time_limit_s": round(cfg.get("lkh_time", 0.0), 2),
            "status": ("ikili bulundu" if ES.find_lkh() else
                       "ikili bulunamadi (LKH_PATH ayarlayin veya bin/LKH.exe koyun)"),
        },
        "concorde": {
            "method": "harici KESIN cozucu koprusu (external_solvers.run_concorde); "
                      "dal-kesme (branch-and-cut) ile KANITLI optimum. Concorde'un "
                      "kendi duvar-saati kesme mekanizmasi olmadigi icin surec "
                      "disaridan zaman asimina ugratilir; zaman asiminda KISMI tur "
                      "DONDURULMEZ (optimallik garantisi ancak cozum bittiyse "
                      "gecerlidir) -- satir 'denendi, sonuc yok' olarak dusulur",
            "reference": "Applegate, Bixby, Chvatal & Cook (2006), 'The Traveling "
                         "Salesman Problem: A Computational Study' — kesin cozucu "
                         "(kesme duzlemli dal-kesme)",
            "time_limit_s": round(cfg.get("concorde_time", 0.0), 2),
            "status": ("ikili bulundu" if ES.find_concorde() else
                       "ikili bulunamadi (CONCORDE_PATH ayarlayin veya bin/concorde koyun)"),
        },
        "benchmark_methodology_reference": (
            "Johnson & McGeoch, 'The Traveling Salesman Problem: A Case Study in "
            "Local Optimization' (1997) and 'Experimental Analysis of Heuristics "
            "for the STSP' (8th DIMACS Implementation Challenge, in Gutin & "
            "Punnen eds., 2002) — standard experimental protocol for comparing "
            "large-scale Euclidean TSP heuristics (construction baselines: "
            "random / nearest-neighbor / greedy-edge / space-filling-curve; "
            "local search: 2-opt / Or-opt / Lin-Kernighan; metaheuristics), used "
            "here to select the construction-stage baseline family"
        ),
        "rotation_scan": f"theta in [-90,90]; every candidate angle is verified "
                         f"with the STRIP tour's TRUE TSPLIB cost on the FULL "
                         f"point set (the angle is found FOR the strip, ON the "
                         f"strip tour; the old snake-based search/verification "
                         f"was removed). Variants: rotation_strip=full sweep, "
                         f"coarse step {cfg.get('rotation_step', 5.0)} deg "
                         f"(dashboard-controlled) + 1-deg refinement around the "
                         f"best region, O(n) strip per angle; "
                         f"rotation_hist=NN-direction-histogram candidates "
                         f"verified in +-1 deg windows at 0.5 deg steps; "
                         f"rotation_mean=axial double-angle circular mean of "
                         f"1/d-weighted kNN directions (continuous sub-degree "
                         f"estimate), verified in +-2 deg windows at 0.25 deg "
                         f"steps; rotation_pca=PCA taban cizgisi (kovaryans "
                         f"baskin ozvektoru; P1-7), mean ile ayni dogrulama "
                         f"penceresi. hist/mean/pca candidate GENERATION samples "
                         f"m=round(n*rotation_sparsity) points (fixed seed 42, "
                         f"deterministic) when n>500; verification is "
                         f"always full-set. theta=0 is always evaluated, so the "
                         f"chosen angle can never lose to no-rotation. All four "
                         f"compete; the best TRUE strip cost wins and its "
                         f"theta*-strip tour seeds the pipeline (window repair -> "
                         f"line reassignment -> metaheuristics) and joins "
                         f"gpu_hybrid's angle pool. Reported 'time' per detector "
                         f"is SEARCH EFFORT ONLY (angle-finding; the O(n) strip "
                         f"build is part of the scan itself). rotation_strip "
                         f"additionally reports 'savings' = worst-minus-best "
                         f"candidate cost seen across its FULL [-90,90] sweep (a "
                         f"representative measure that the search beats picking a "
                         f"random angle) plus the worst candidate's cost and "
                         f"angle; rotation_hist/rotation_mean leave these empty -- "
                         f"they only verify a few candidate angles in narrow "
                         f"windows, so a 'worst tried' there would not be "
                         f"representative of the whole search space.",
        # ge_gpu / gpu_snake_stratified_subsample / gpu_snake_greedy_band /
        # gpu_snake_farthest_band / gpu_hybrid girdileri KALDIRILDI
        # (2026-07-31): bes anahtarin tamami REMOVED_METHODS'ta, yani bu
        # hiperparametre kayitlari hicbir satirda raporlanamiyordu.
        "ils": {
            "iterations": cfg["ils_iter"], "ls_rounds": cfg["ils_rounds"],
            "neighbor_limit_2opt": cfg["nb2"], "neighbor_limit_oropt": cfg["nbo"],
            "relocate_k": cfg["rk"], "relocate_neighbor_limit": cfg["rnb"],
            "time_budget_s": round(cfg["ils_time"], 2), "seeds": cfg["seeds"],
            "perturbation": "adaptive (double-bridge / segment-reverse)",
            "deadline_safety": lro._LS_DEADLINE_SAFETY,
        },
        # Onarim katmani (tek cati): bes satir da AYNI motoru ve AYNI butce
        # sinirlarini kullanir; tek degisken KOMSULUK KUMESI, baslangic turu
        # ise panelden secilen insadir (seed_key/seed_name satirda raporlu).
        "repair_family": {
            "engine": "repair.run_single_neighborhood / lro.run_neighborhood_vnd",
            "neighbourhoods": {
                "repair_two_opt": "aday-listeli 2-opt",
                "repair_or_opt": "Or-opt (segment <= 3)",
                "repair_relocate": "tek-nokta relocate",
                "repair_vnd": "tam VND (2-opt + Or-opt + relocate, round-robin)",
                "repair_vnd_dense": "genisletilmis VND (Or-opt <= 6 + yogun toplu)",
            },
            "neighbor_limit_2opt": cfg["nb2"], "neighbor_limit_oropt": cfg["nbo"],
            "relocate_k": cfg["rk"], "relocate_neighbor_limit": cfg["rnb"],
            "max_rounds": 40,
            "adaptive_k_schedule": list(lro.DEFAULT_ADAPTIVE_K_SCHEDULE),
            "dense_levels": [
                {"k_nearest_edges": lv.k_nearest_edges,
                 "city_neighbor_limit": lv.city_neighbor_limit,
                 "dense_fraction": lv.dense_fraction,
                 "max_batch_size": lv.max_batch_size}
                for lv in cfg["dense"]],
            "seed_rule": ("admin panelinden secilen insa "
                          "(dashboard/methods_config.json -> seeds); "
                          f"varsayilan: {DEFAULT_SEED}"),
        },
        # --- havuzlu iki satirin ONARIM KATMANI (2026-07-25: secilebilir) ---
        "pool_repair": {
            "selectable_methods": sorted(POOL_REPAIR_SELECTABLE),
            "choices": dict(POOL_REPAIR_LABELS),
            "default": DEFAULT_POOL_REPAIR,
            "selected": {m: resolve_pool_repair_choice(m, _panel_pool_repair_config())
                         for m in sorted(POOL_REPAIR_SELECTABLE)},
            "ablation_equivalent": dict(POOL_REPAIR_ABLATION),
            "budget_rule": (
                "her secimde AYNI toplam duvar-saati butcesi (cfg['gpu_time']); "
                "batch_2opt_oropt tum havuzu tek batch'te onarir, digerleri "
                "havuzdaki her tura butce/|havuz| paylasimiyla TEK TEK "
                "uygulanir -- onarim turunu degistirmek yontemin toplam "
                "zamanini gizlice artirmaz"),
            "note": (
                "batch_2opt_oropt'un komsuluk kumesi {2-opt, Or-opt<=3}; "
                "RELOCATE ICERMEZ, dolayisiyla ablasyon tablosunda BIREBIR "
                "muadili yoktur (repair_vnd relocate'i de icerir). Havuzlu "
                "satiri bir ablasyon satiriyla dogrudan kiyaslamak icin ayni "
                "onarim secilmelidir -- bkz. ablation_equivalent."),
        },
        "family_note": (
            "Construction methods fall into distinct asymptotic/quality "
            "classes (see FAMILY dict in runner.py); a 'best constructor' "
            "claim is only meaningful WITHIN a class. 'curve' = "
            "nn/strip/hilbert/morton/snake/rotation*, all near-linear "
            "grid-or-sort methods with no local search -- this is Snake-"
            "Grid's real peer group. 'insertion' (nearest/farthest "
            "insertion, O(n^2)) and 'edge_greedy' (greedy-edge, higher "
            "quality via a different mechanism) are reported alongside for "
            "context/upper-bound, NOT as same-class competitors."
        ),
        "family": dict(FAMILY),
    }


def run_dataset(name: str, methods=None, time_budget=1.0, seeds_override=None,
                quick=False, method_time=None, rotation_step=5.0,
                rotation_sparsity=None, force=False, construction_max_s=None,
                exact_time=None, choice_overrides=None):
    # Panel seçimlerini bu koşum için geçersiz kıl (panel "çoğaltma"): satır
    # row_id sayesinde kanonik satırın YANINA düşer, üstüne değil.
    if choice_overrides is not None:
        set_choice_overrides(choice_overrides)
    path = dataset_path(name)
    header, coords, ewt = E.parse_tsp(path)
    n = len(coords)
    # BKS: klasik TSPLIB tablosu, yoksa Waterloo katalogları (VLSI/Ülke/Sanat)
    bks = (E.BKS.get(name) or VLSI.BKS.get(name)
           or TSPLIB.BKS.get(name))
    # Insa (construction) yontemleri (greedy_edge, nearest/farthest insertion,
    # snake/GPU bant hibritleri) icin duvar-saati ust siniri -- kullanicidan
    # (dashboard "Insa ust sure siniri" kutusu) None gelirse _CONSTRUCTION_MAX_S
    # (300s) varsayilanina duser.
    constr_budget_s = (float(construction_max_s) if construction_max_s is not None
                       else _CONSTRUCTION_MAX_S)
    cfg = _scale_config(n, time_budget)
    # EXACT/altin standart cozuculerin duvar-saati ust siniri. Kuruculardaki
    # "Insa ust sure siniri" ile AYNI rolde: kullanicidan (panel "Exact yöntem
    # süresi" kutusu) None gelirse EXACT_TIME_DEFAULT (300 s). LKH-3 bunu kendi
    # TIME_LIMIT'i olarak alir (kesildiginde elindeki turu YAZAR), Concorde ise
    # sert bir surec zaman asimi olarak gorur (kesildiginde tur DONMEZ).
    exact_budget_s = (float(exact_time) if exact_time is not None
                      else EXACT_TIME_DEFAULT)
    cfg["lkh_time"] = exact_budget_s
    cfg["concorde_time"] = exact_budget_s
    if seeds_override:
        cfg["seeds"] = seeds_override
    if quick:
        cfg["max_iter"] = min(cfg["max_iter"], 150)
        cfg["ils_iter"] = min(cfg["ils_iter"], 30)
        cfg["ils_rounds"] = min(cfg["ils_rounds"], 3)
        cfg["ils_time"] = min(cfg["ils_time"], 8)
        cfg["gpu_time"] = min(cfg["gpu_time"], 8)
        cfg["seeds"] = min(cfg["seeds"], 2)
        # smoke kosumunda exact cozuculer de kisilir; Concorde bu butcede
        # buyuk orneklerde bitiremez ve "denendi, sonuc yok" dusmesi BEKLENIR.
        cfg["lkh_time"] = min(cfg["lkh_time"], 10.0)
        cfg["concorde_time"] = min(cfg["concorde_time"], 20.0)

    # Common per-method wall-clock budget for a fair anytime comparison: when set,
    # ILS gets this exact wall-clock budget (review item P2-11).
    if method_time is not None:
        cfg["ils_time"] = float(method_time)
    cfg["method_time_equal"] = bool(method_time is not None)
    cfg["rotation_step"] = float(rotation_step)
    cfg["rotation_sparsity"] = (_ROTATION_DEFAULT_SPARSITY if rotation_sparsity is None
                                else float(rotation_sparsity))

    # Panelden KALICI silinen yontemler hicbir kosuda yer almaz -- eski bir
    # --methods listesi (kaydedilmis on ayar, kabuk gecmisi, baska bir betik)
    # silinmis anahtari tasisa bile satir URETILMEZ. Silme "gorunmez yap"
    # degil "yok say" demektir; bkz. is_dropped_method.
    want = set(methods) if methods else set(METHOD_ORDER)
    _dead = deleted_methods()
    if want & _dead:
        _log(f"[{name}] panelden silinmis {len(want & _dead)} yöntem "
             f"istendi, atlanıyor: {', '.join(sorted(want & _dead))}")
        want -= _dead

    # ---- KALDIĞI YERDEN DEVAM (resume, varsayılan AÇIK): results/{name}.json
    # dosyasında istenen yöntemin GEÇERLİ bir satırı (turuyla) zaten varsa o
    # yöntem yeniden KOŞULMAZ; satır diskten `results`e önden yüklenir (ILS
    # ailesi tohumları ve LEE base havuzu onu görebilsin diye) ve final dosyada
    # aynen korunur. Bir yöntemi/örneği zorla yeniden koşturmak için ya ilgili
    # results/{name}.json silinir (tercih edilen akış) ya da --force verilir. ----
    prior_rows = {}
    _out_path = RESULTS / f"{name}.json"
    # _merge_snapshot: koşu BAŞINDAKI dosyanın satırları (REMOVED ayıklanmış).
    # Ara-kayıt (checkpoint) dosyayı koşu ortasında ezdiği için dosya-sonu
    # merge artık diski yeniden okuyamaz -- yoksa --force + --methods alt
    # kümesi koşusu, checkpoint'in yazdığı kısmi dosyayı "önceki sonuç"
    # sanıp GERİ KALAN TÜM YÖNTEMLERİ SİLER (2026-07-23 vakası: 5 setin
    # 36 satırı bu yüzden uçtu). Snapshot force'tan bağımsız HER ZAMAN
    # alınır; yalnız resume-atlama kararı force'a bağlıdır.
    _merge_snapshot = {}
    _merge_payload = None
    _notes_migrated = 0
    _stale_v4 = 0
    _dropped_removed = 0
    if _out_path.exists():
        try:
            _merge_payload = json.loads(_out_path.read_text(encoding="utf-8"))
            # Satirlar ROW_ID ile anahtarlanir (varyantlar yan yana durabilsin);
            # REMOVED ayiklamasi YONTEM anahtari uzerinden yapilir. Eski
            # dosyalarda row_id alani yoktur -> anahtarin kendisine duser.
            _merge_snapshot = {(m.get("row_id") or m["key"]): m
                               for m in _merge_payload.get("methods", [])
                               if not is_dropped_method(m.get("key"))}
            # !! 2026-07-31 DUZELTMESI !! Ayni ayiklamayi PAYLOAD'a da uygula.
            # Once yalniz _merge_snapshot suzuluyordu; payload dokunulmadan
            # kaliyordu. Sonuc: "istenen tum yontemler zaten kosulmus -> ornek
            # atlandi" ERKEN CIKIS yolunda dosya hic yazilmadigi icin
            # kaldirilmis satirlar DISKTE SONSUZA KADAR kaliyordu -- tabloda
            # hayalet satir olarak gorunuyor, ustelik kendi gap'iyle
            # siralamaya giriyorlardi. Olculdu (10'lu havuz kaldirilirken):
            # enjekte edilen sahte greedy_snake_v3_repair satiri (gap=-99)
            # resume'den SAG CIKTI. drop_stale_v4_rows bu tuzagi zaten
            # biliyordu (payload parametresi tam bunun icin var); REMOVED
            # ayiklamasi ondan geri kalmisti.
            if _merge_payload.get("methods"):
                _keep = [m for m in _merge_payload["methods"]
                         if not is_dropped_method(m.get("key"))]
                _dropped_removed = len(_merge_payload["methods"]) - len(_keep)
                if _dropped_removed:
                    _merge_payload["methods"] = _keep
                    _log(f"[{name}] {_dropped_removed} kaldırılmış yöntem "
                         f"satırı diskten ayıklandı")
            # Takma-ad devri: eski anahtarin satiri yeni resmi anahtara
            # kopyalanir (yeni anahtar henuz yoksa) -> resume yeni anahtari
            # "hazir" sayar, eski sonuc kaybolmaz.
            for _old, _new in LEGACY_ALIASES.items():
                if _old in _merge_snapshot and _new not in _merge_snapshot:
                    _row = dict(_merge_snapshot[_old])
                    _row["key"] = _new
                    _row["name"] = PRETTY.get(_new, _new)
                    _merge_snapshot[_new] = _row
            _stale_v4 = drop_stale_v4_rows(_merge_snapshot, _merge_payload)
            if _stale_v4:
                _log(f"[{name}] {_stale_v4} eski v4 satırı ayıklandı "
                     f"(satır yüksekliği havuzu → açı ince taraması); "
                     f"bu koşumda yeniden üretilecek")
            _notes_migrated = migrate_solver_notes(
                _merge_payload.get("solver_notes", []))
        except Exception as exc:
            _log(f"[{name}] devam modu: önceki sonuç okunamadı ({exc}) -> tam koşum")
    if not force:
        prior_rows = _merge_snapshot

    def _skip_done(w):
        # Diskte turu olan satır "hazır" sayılır; tursuz satır yeniden koşulur.
        # ÖNEMLİ (2026-07-25): kontrol YÖNTEM değil ROW_ID üzerinden yapılır --
        # aynı yöntemin BAŞKA bir girdi/açı/onarım seçimiyle koşulmuş satırı
        # diskte olsa bile, YENİ seçim için satır YOKTUR ve yöntem yeniden
        # koşulmalıdır. Eski davranış "repair_vnd zaten var" deyip atlıyordu,
        # dolayısıyla girdiyi değiştirip yeniden koşmak imkânsızdı.
        # RGGE/RSGE ISTISNASI (2026-09-07): bu ailelerde row_id ACIYI
        # TASIMAZ -- tek satir uretilir ve o satir daima temiz anahtardadir
        # (ANGLE_PANEL_CANONICAL). Dolayisiyla row_id tek basina yetmez:
        # panelden aci degistirildiginde ESKI acida kosulmus satir aynen
        # bulunur ve yontem sessizce ATLANIRDI (tabloda yeni acinin adi,
        # icinde eski acinin sayilari). Satirin `angle_key`i de SU ANKI
        # secimle ayni olmali.
        def _hazir(k):
            _row = prior_rows.get(row_id(k, _choice_of(k)), {})
            if not _row.get("tour"):
                return False
            # ACI-SECILEBILIR satirlar: diskteki satirin acisi SU ANKI secimle ayni
            # olmali (2026-09-07: varsayilan aci degistiginde -- orn. RSGE
            # rotation_strip -> grid_theta -- eski kanonik satir sessizce
            # "hazir" sayilmasin).
            if k in ANGLE_SELECTABLE and _row.get("angle_key") != _choice_of(k):
                return False
            return True

        done = {k for k in w if _hazir(k)}
        return w - done, done

    # Panel seçimleri: row_id hesabı için resume'dan ÖNCE okunmalı.
    seeds_cfg = _panel_seed_config()
    _angles_cfg = _panel_angle_config()
    _pool_repairs_cfg = _panel_pool_repair_config()
    _ge_variants_cfg = _panel_ge_variant_config()
    # seed_of TEK DOĞRULUK KAYNAĞI: _choice_of buradan okur. Sebebi kritik --
    # order_seedable() bir tohum DÖNGÜSÜ bulursa ilgili satırı DEFAULT_SEED'e
    # düşürüp seed_of'u yeniden yazar; row_id o düzeltilmiş seçimi görmeli,
    # yoksa satır "repair_x@<döngülü seçim>" adıyla kaydedilir ama gerçekte
    # DEFAULT_SEED'den koşulmuş olur (sessiz yanlış etiket).
    seed_of = {m: resolve_seed_choice(m, seeds_cfg)
               for m in METHOD_ORDER if m in SEEDABLE}

    def _choice_of(method_key):
        """Yöntemin bu koşumdaki PANEL SEÇİMİ (tohum / açı / havuz onarımı);
        seçilebilir boyutu olmayan yöntemlerde ""."""
        if method_key in SEEDABLE:
            _s = seed_of.get(method_key) or DEFAULT_SEED
            # ILS'in tohumu bir KOPYA onarim satiriysa (repair_vnd@strip) satir
            # kimligi o kopyayi tasir -> devam modu "ils@repair_vnd" ile karismaz.
            if method_key == "ils" and _s in SEEDABLE:
                _srid = row_id(_s, _choice_of(_s))
                if "@" in _srid:
                    return _srid
            return _s
        if method_key in ANGLE_SELECTABLE:
            return resolve_angle_choice(method_key, _angles_cfg)
        if method_key in POOL_REPAIR_SELECTABLE:
            return resolve_pool_repair_choice(method_key, _pool_repairs_cfg)
        if method_key in GE_VARIANT_SELECTABLE:
            return resolve_ge_variant_choice(method_key, _ge_variants_cfg)
        return ""

    def _rid(method_key):
        return row_id(method_key, _choice_of(method_key))

    want, _done1 = _skip_done(want)
    # Tohum secilebilen (SEEDABLE) bir yontem KOSULACAKSA, panelden secilen
    # tohum insasini da calistir ki tohum turu uretilsin. Insa zaten ayri
    # satir olarak da raporlanir -> okuyucu baslangic gap'ini (L_init)
    # dogrudan gorur. Insanin diskte hazir turu varsa o da kosulmaz --
    # tohum diskten okunur.
    # tohum zinciri GECISLI kapanisiyla eklenir: repair_window <- repair_vnd
    # <- greedy_edge secildiyse ucu de kosmali (aksi halde ortadaki halka
    # eksik kalir ve satir "tohum uretilemedi" ile duser).
    _stack = list(want)
    while _stack:
        _m = _stack.pop()
        _sk = seed_of.get(_m)
        if _sk and _sk not in want:
            want.add(_sk)
            _stack.append(_sk)
    # ACI bagimliligi: v1 icin bir rotation_* dedektoru secildiyse o dedektor
    # de kosmali (acisini o satirdan okuyoruz).
    # HAVUZ ONARIMI secimi: havuzlu iki satirin onarim katmani panelden
    # secilir (ablasyonlardaki tohum kutusunun onarim karsiligi). Secim
    # sonuc satirina repair_key/repair_name olarak yazilir ve gosterim adi
    # "Havuz ← <onarim>" olur.
    for _m in list(want & ANGLE_SELECTABLE):
        _ak = resolve_angle_choice(_m, _angles_cfg)
        if _ak.startswith("rotation_"):
            want.add(_ak)
    want, _done2 = _skip_done(want)

    # ---- SÜRE AŞIMI HAFIZASI (2026-07-29). "Bir kere koşuldu, verilen
    #      sürede bitiremedi" diyen bir yöntem, AYNI (ya da daha dar)
    #      bütçeyle YENİDEN DENENMEZ. İki bütçe tipi vardır ve her biri
    #      KENDİ bayrağıyla karşılaştırılır -- ayrıntı ve ölçümler için
    #      modül başındaki "SURE ASIMI HAFIZASI" bloğuna bakın:
    #
    #        construction_budget -> constr_budget_s (--construction-max-s)
    #        exact_budget        -> exact_budget_s  (--exact-time)
    #
    #      Not diskte KALIR, yani panelde "denendi, sonuç yok" kutusundaki
    #      açıklama kaybolmaz; sadece bir daha koşulmaz.
    #
    #      YENİDEN denenmesinin İKİ yolu var (ikisi de kullanıcı iradesi):
    #        * ilgili bütçeyi BÜYÜTMEK (yeni bütçe eskisinden büyükse hafıza
    #          bağlamaz -- sonuç değişebilir),
    #        * --force (ya da results/{name}.json'u silmek).
    #
    #      Alanlar 2026-07-29'da eklendiği için daha eski notlarda `reason`
    #      yoktur; onlar yukarıda (merge anlık görüntüsü okunurken)
    #      `migrate_solver_notes` ile metindeki bütçe sayısından göç
    #      ettirilir, böylece hafıza ilk koşumdan itibaren çalışır. ----
    _CURRENT_BUDGET = {"construction_budget": constr_budget_s,
                       "exact_budget": exact_budget_s,
                       # size_limit esigi de insa butcesiyle olceklenir:
                       # butce buyurse sinir buyur, satir yeniden denenir.
                       "size_limit": constr_budget_s}
    _budget_blocked = {}          # yöntem -> (reason, önceki bütçe)
    if _merge_payload and not force:
        for _nt in _merge_payload.get("solver_notes", []):
            _reason = _nt.get("reason")
            if _reason not in _CURRENT_BUDGET:
                continue
            if _nt.get("key") not in want:
                continue
            _pb = _nt.get("budget_s")
            # Bütçe büyütüldüyse hafıza BAĞLAMAZ.
            if _pb is None or _CURRENT_BUDGET[_reason] <= float(_pb) + 1e-9:
                _budget_blocked[_nt["key"]] = (_reason, _pb)
    if _budget_blocked:
        want -= set(_budget_blocked)
        for _reason in sorted({r for r, _ in _budget_blocked.values()}):
            _keys = sorted(k for k, (r, _) in _budget_blocked.items()
                           if r == _reason)
            _log(f"[{name}] süre aşımı hafızası: {len(_keys)} yöntem yeniden "
                 f"DENENMEDİ ({', '.join(_keys)}) -- önceki koşumda "
                 f"{_CURRENT_BUDGET[_reason]:.0f}s bütçeye sığmamışlardı, "
                 f"bütçe hâlâ aynı. Denemek için {BUDGET_FLAG[_reason]} "
                 f"değerini büyütün ya da --force verin")
        if not want:
            # Koşulacak hiçbir şey kalmadı -> normal yazma yoluna hiç
            # girilmiyor. Ama göç ettirilmiş notlar YALNIZCA bellekte kaldıysa
            # panelde sonsuza dek eski, açıklamasız metin görünür ("atlandı"
            # der, NEDEN bir daha koşulmadığını söylemez). Bu tek seferlik
            # yazma onu kalıcı kılar. Yazılan şey diskteki dosyanın AYNISI,
            # sadece not alanları güncel -- veri kaybı riski yok.
            if _notes_migrated or _stale_v4 or _dropped_removed:
                _atomic_write_json(_out_path, _merge_payload)
                if _notes_migrated:
                    _log(f"[{name}] {_notes_migrated} eski not güncel biçime "
                         f"taşındı (açıklama metni + bütçe alanları)")
            _log(f"[{name}] istenen tüm yöntemler bütçe hafızasında -> "
                 f"örnek atlandı\n")
            return

    _skipped = _done1 | _done2
    if _skipped:
        _log(f"[{name}] devam modu: diskte hazır {len(_skipped)} yöntem atlandı "
             f"({', '.join(sorted(_skipped))}) -- yeniden koşturmak için "
             f"results/{name}.json silin veya --force verin")
    if not want:
        # Ayiklama YAPILDIYSA dosya bir kez yazilmali (yukaridaki ayni gerekce):
        # bu yol normal yazma yoluna HIC girmez, dolayisiyla ayiklanan satirlar
        # yalniz BELLEKTE silinir ve diskte kalirdi.
        if _notes_migrated or _stale_v4 or _dropped_removed:
            _atomic_write_json(_out_path, _merge_payload)
        _log(f"[{name}] devam modu: istenen tüm yöntemler zaten koşulmuş -> "
             f"örnek atlandı\n")
        return
    _log(f"[{name}] n={n} ewt={ewt} bks={bks} cfg={ {k:v for k,v in cfg.items() if k!='dense'} }")

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # ---- build the instance in the UNROTATED frame; the rotation step below is a
    #      toggleable improvement that may switch to a rotated frame later. ----
    t_inst = time.perf_counter()
    inst, sparse = E.make_instance(coords, ewt)
    _log(f"[{name}] instance sparse={sparse} built in {time.perf_counter()-t_inst:.2f}s")

    results = {}     # key -> method dict
    # Devam modu: diskteki TÜM eski satırlar önden yüklenir. record() bu
    # koşumda ürettiğini doğrudan üzerine yazar; dokunulmayanlar aynen kalır
    # (dosya-sonu merge ile aynı semantik, sadece erken -- böylece ILS ailesi
    # tohumları ve LEE base havuzu atlanan inşaların turunu diskten görür).
    for _k, _m in prior_rows.items():
        results.setdefault(_k, _m)
    # One timestamp for THIS run, stamped onto every method row record()
    # produces. After the on-disk merge below, rows whose run_at differs from
    # the payload's "generated" are visibly from an OLDER run (possibly with
    # different budgets/settings) -- the dashboard uses this to mark them
    # instead of silently presenting mixed-run rows as one comparison.
    run_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Attempted-but-no-tour method calls (inşa bütçesi aşımı, GPU yokluğu,
    # tohum üretilememesi, ...) -- kept separate from `results`/`methods`
    # (which only ever hold a real cost+tour) so "denendi, sonuç yok"
    # durumu görünür kalır.
    solver_notes = []

    # ---- ARA-KAYIT (checkpoint): uzun koşular (10 tohumlu ILS ~25 dk/set)
    #      300 sn'lik kabuk zaman aşımıyla bölünebilir; sonuç dosyası eskiden
    #      yalnız set SONUNDA yazıldığı için kesinti tüm set ilerlemesini
    #      siliyordu. Artık record() her yöntem satırından sonra mevcut
    #      `results`i ATOMİK olarak results/{ad}.json'a yazar; öldürülen koşu
    #      bir sonraki çağrıda resume ile kaldığı yöntemden devam eder.
    #      Final yazım (dosya sonu) comparison/hyperparameters dolu TAM
    #      payload'ı üretip bunun üzerine yazar; ara yazım aynı şemanın
    #      erken halidir ("partial": True işaretiyle). ----
    theta_star = 0.0        # aşağıda dedektör bölümü günceller
    rotation_applied = False
    seed_cost = None

    def _checkpoint():
        # Koşu başındaki satırları (snapshot) her ara-kayıtta koru: yeni
        # satırlar eskileri EZER (update sırası), koşulmayanlar aynen kalır.
        # Böylece --force + --methods alt kümesi bile dosyayı asla inceltmez.
        merged = dict(_merge_snapshot)
        merged.update(results)
        ordered = _order_rows(merged)
        payload = {
            "dataset": name, "n": n, "ewt": ewt, "is_geo": "GEO" in ewt,
            "bks": bks, "sparse": sparse,
            "theta_star": round(theta_star, 2),
            "rotation_applied": rotation_applied,
            "seed_cost": round(seed_cost, 2) if seed_cost is not None else None,
            "generated": run_stamp,
            "time_budget": time_budget, "construction_max_s": constr_budget_s,
            "exact_max_s": exact_budget_s,
            "config": {k: v for k, v in cfg.items() if k != "dense"},
            "comparison": None, "comparisons": [],
            "hyperparameters": _hyperparameters(cfg),
            "coords": [[round(x, 3), round(y, 3)] for x, y in coords],
            "methods": ordered,
            "best_cost": min((m["cost"] for m in ordered), default=None),
            "solver_notes": solver_notes,
            "partial": True,
        }
        _atomic_write_json(_out_path, payload)

    def note_unavailable(key, status, elapsed, reason=None, budget_s=None):
        # `reason`/`budget_s` MAKINE-OKUNUR alanlardir: bir sonraki kosum
        # notu tekrar denemeye deger mi diye buradan karar verir (bkz.
        # "insa butcesi hafizasi"). Metin (`status`) yalniz insana gosterilir.
        nt = {"key": key, "name": PRETTY[key], "status": status,
              "time": round(elapsed, 1)}
        if reason:
            nt["reason"] = reason
        if budget_s is not None:
            nt["budget_s"] = round(float(budget_s), 1)
        solver_notes.append(nt)
        _log(f"   [skip] {PRETTY[key]}: {status}")
        # ---- NOTU HEMEN DISKE YAZ (2026-07-30). ----
        # ONCEDEN: `_checkpoint()` yalnizca bir yontem BASARILI oldugunda
        # cagriliyordu (record / _record_pool / _record_pool_repair icinden).
        # Notlar yalnizca BELLEKTE birikiyordu ve dosyaya ancak kosumun EN
        # SONUNDA iniyordu. Sonuc: buyuk bir sette fi_pool ailesi + kesin
        # cozuculer arka arkaya butceye takilir (her biri 300 s), sonra
        # baska basarili yontem KALMADIGI icin hicbir ara-kayit olmaz; kosum
        # kesilirse (kullanici kapatir, PC uyur, dagitici oldurur) saatlerce
        # ogrenilen "bu satir bu butceye sigmiyor" bilgisinin TAMAMI ucar ve
        # bir sonraki kosum ayni 300 s'leri bastan yakar.
        #
        # Olculdu: 27 sette fi_pool ailesinin dort satirinda da NE tur NE not
        # vardi (n=3795..28924) -- kullanicinin "PC gunlerce bosuna acik
        # kaliyor" dedigi sey tam olarak buydu.
        #
        # Maliyeti bir dosya yazimi; ogrenilen bilginin bedeli 300 s.
        try:
            _checkpoint()
        except Exception as exc:                     # pragma: no cover
            # Ara-kayit basarisiz olsa bile not bellekte duruyor ve kosum
            # sonunda yazilacak -- eski davranisa duseriz, kosum durmaz.
            _log(f"   [uyarı] not ara-kaydı yazılamadı ({exc}); "
                 f"koşum sonunda yazılacak")

    def size_limited(key):
        """Bu n bu bütçede ölçülmüş sınırın üstündeyse notu yaz ve True dön.

        Yöntem HİÇ çalıştırılmaz -- 300 s'i yakıp "yetişmedi" demenin
        anlamı yok, sonucu zaten biliyoruz (bkz. modül başındaki
        `_SIZE_LIMITS` kalibrasyonu). Not `reason="size_limit"` taşır;
        bütçe büyütülürse hem hafıza çözülür hem sınır büyür, satır
        kendiliğinden yeniden denenir.
        """
        lim = method_max_n(key, constr_budget_s)
        if lim is None or n <= lim:
            return False
        note_unavailable(
            key,
            f"n={n} > {lim}: ölçülen maliyet eğrisine göre bu boyut "
            f"{constr_budget_s:.0f}s inşa bütçesine sığmaz — hiç denenmedi "
            f"(bütçeyi büyütmek sınırı da büyütür: “İnşa üst süre sınırı”)",
            0.0, reason="size_limit", budget_s=constr_budget_s)
        return True

    def note_construction_timeout(key, elapsed, prefix=""):
        """İnşa bütçesi aşımı notu -- TEK KAYNAK.

        Metni burada uretmek sart: bir sonraki kosum bu notu
        `reason="construction_budget"` + `budget_s` alanlarindan tanir ve
        AYNI (ya da daha dar) butceyle satiri YENIDEN DENEMEZ. Cagri
        yerlerinden biri elle metin yazarsa o satir hafizaya girmez ve her
        kosumda butce kadar zaman yakmaya devam eder."""
        note_unavailable(key,
                         f"{prefix}inşa {constr_budget_s:.0f}s bütçesini "
                         f"aştı — atlandı. Bu bütçeyle bir daha denenmez "
                         f"(inşa deterministiktir, sonuç değişmez); denemek "
                         f"için “İnşa üst süre sınırı”nı büyütün", elapsed,
                         reason="construction_budget",
                         budget_s=constr_budget_s)

    def record(key, cost, t, diag=None, tour=None, initial=None, stoch=None,
               savings=None, choice=None):
        # Satir ROW_ID ile saklanir: ayni yontemin farkli panel secimleriyle
        # kosulmus satirlari birbirini EZMEZ (2026-07-25). `key` her zaman
        # yontem anahtaridir -> renk/grup/asama aramalari degismez.
        #
        # `choice` (2026-07-26): satirin secimini ACIKCA belirtir. Normalde
        # panel secimi (_choice_of) kullanilir; ama TEK kosumda AYNI yontemin
        # BIRDEN COK secimi kaydedilebiliyorsa (greedy_edge: kanonik rakip +
        # duyarlilik varyanti ayni kosuda yan yana) her cagrinin kendi
        # secimini bildirmesi gerekir, yoksa ikisi de panel secimine yazilir
        # ve biri digerini ezer.
        _ch = _choice_of(key) if choice is None else choice
        _rid_ = row_id(key, _ch)
        m = {
            "key": key, "row_id": _rid_, "variant": _ch or None,
            "name": PRETTY[key], "stage": STAGE[key],
            "run_at": run_stamp,
            "cost": round(cost, 2), "gap": _gap(cost, bks),
            "time": round(t, 3),
            "initial_cost": round(initial, 2) if initial is not None else None,
            "iters": getattr(diag, "iterations_completed", None) if diag else None,
            "improvements": getattr(diag, "improvements_accepted", None) if diag else None,
            # ILS teşhisi: döngü NEDEN durdu ("time"/"no_improve"/"max_iter"),
            # Phase-0 ilk iniş gerçek yerel optimuma vardı mı (False => eğri hâlâ
            # düşerken kesildi = "platoya ulaşmadan durdu"), ve kaç perturbasyon
            # iterasyonu gerçekten koştu (0 => tek iniş, "iterated" olamadı).
            # ILS dışı yöntemlerde None kalır.
            "stop_reason": getattr(diag, "stop_reason", None) if diag else None,
            "phase0_converged": getattr(diag, "phase0_converged", None) if diag else None,
            "perturbation_iters": getattr(diag, "perturbation_iterations", None) if diag else None,
            # gpu_hybrid: karma havuzda kazanan başlangıç ("snake"/"nn"/"greedy_edge").
            # Kazanan greedy_edge ise gpu_hybrid'in sonucu ge_gpu ile (kırılım
            # eşitliklerine kadar) aynı olur -- tabloda bu görünür olmalı.
            "winner": getattr(diag, "winner", None) if diag else None,
            "candidates": getattr(diag, "total_candidates_evaluated", None) if diag else None,
            # `savings` kwarg (theta dedektorleri: en kötü-en iyi deneme farkı)
            # varsa ONU kullanır, yoksa eski davranışa (diag.total_saving,
            # lro/gpu_hybrid optimizer diag'leri) düşer.
            "savings": (round(savings, 2) if savings is not None
                       else (round(getattr(diag, "total_saving", 0.0), 2) if diag else None)),
            "history": _downsample_history(diag.history) if diag and getattr(diag, "history", None) else [],
            "stochastic": stoch,
            "tour": tour,
        }
        results[_rid_] = m
        # 2026-08-02: konsol satiri ARTIK row_id'yi tasir. Once yalniz PRETTY
        # yazıyordu; ayni yontemin farkli girdilerle (aci/tohum/onarim) kosan
        # satirlari LOGDA BIREBIR AYNI gorunuyordu -- ornegin greedy_snake_v1'in
        # kanonik, "zero" ve elle-aci satirlari uc kez ayni adla akiyordu ve
        # hangi sayinin hangi deneye ait oldugu okunamiyordu. Panelde ayrim
        # zaten vardi (display_name "← <kaynak>" ekler), eksik olan konsoldu.
        _sfx = _rid_[len(key):] if _rid_ != key else ""     # "@zero", "@..." vb.
        _log(f"   {PRETTY[key]}{_sfx}"
             f"  cost={cost:>12.0f} gap={m['gap']}% t={t:.1f}s")
        _checkpoint()

    # ---- TOHUM SOZLESMESI (2026-07-24) -------------------------------------
    # SEEDABLE yontemler, panelden secilen INSA'nin turundan baslar. Secim
    # sonuc satirina seed_key/seed_name olarak yazilir ve gosterim adi
    # "Yontem <- Tohum" olur; boylece hangi girdiden kosuldugu tabloda ve
    # dashboard'da her zaman okunur.
    def _row_of(method_key):
        """Yontemin BU KOSUMDAKI panel secimine karsilik gelen satiri dondurur.
        O secimle kosulmus satir yoksa (ornegin tohum diskte yalniz baska bir
        secimle varsa) VARSAYILAN satira duser -- boylece zincir kopmaz ama
        hangi satirin kullanildigi row_id'den okunabilir kalir.

        ISTISNA (2026-07-26, ANGLE_SELECTABLE icin 2026-07-27'de genisletildi):
        DUYARLILIK boyutu olan yontemler (GE_VARIANT_SELECTABLE aday/dikis
        varyantlari, ANGLE_SELECTABLE aci kaynaklari) TOHUM olarak her zaman
        KANONIK satirlarini verir. greedy_edge DEFAULT_SEED'tir -- panelden
        secilen bir DUYARLILIK varyanti (ozellikle knn15_random, literatur
        Greedy'sinin ~5x disinda) tohum olarak sizarsa ondan beslenen 8
        ablasyon satirinin tamami sessizce bozulur ve tablo "onarim kotu
        calisti" gibi okunur. Ayni sekilde greedy_snake_v1'in aci-duyarliligi
        satiri aile merdiveninin disindadir ve tohum olarak kullanilmamalidir.
        Duyarlilik deneyi yalniz KENDI satirini etkiler; asagi akisi ASLA
        etkilemez. (Iki durumda da KANONIK satir her kosuda uretildigi icin
        bu istisna zinciri koparmaz.)"""
        if method_key in GE_VARIANT_SELECTABLE or method_key in ANGLE_SELECTABLE:
            return results.get(method_key)
        return results.get(_rid(method_key)) or results.get(method_key)

    def _seed_tour_of(seed_key):
        """Secilen insanin bu ornekteki turunu (tour, cost) dondurur; insa
        kosulmadiysa/tur uretmediyse None."""
        m = _row_of(seed_key)
        if m and m.get("tour"):
            return m["tour"], m["cost"]
        return None

    # _v3_extra_tours / _far_v3_extra_tours KALDIRILDI (2026-07-31): tek
    # tuketicileri greedy_snake_v3 ve far_snake_v3 satirlariydi, ikisi de
    # REMOVED_METHODS (f) ile cikarildi.
    #
    # ONEMLI YAN ETKI (kiyas gecerliligi): _v3_extra_tours far_snake_v2'nin
    # turunu v3 havuzuna bedava aday olarak veriyordu ve bu
    #      maliyet(v3) <= far_snake_v2
    # esitsizligini YAPISAL kiliyordu -- bu yuzden dashboard far_snake_v2'yi
    # Sira/Kazanma% havuzundan cikariyordu (index.html V3_POOL_MEMBERS).
    # v3 gidince o kapsama da gitti: ARTIK HICBIR SATIR BASKA BIR SATIRIN
    # TURUNU BEDAVA ALMIYOR, dolayisiyla far_snake_v2 (ve tum digerleri) tam
    # yetkili rakip olarak siralamaya girer. V3_POOL_MEMBERS bosaltildi.
    def _repair_pool(tours, mode, budget_s, width=None):
        """Havuzun TAMAMINI panelden secilen ONARIM katmanindan gecirir.
        Doner: (onarilmis_turlar, diag_veya_None).

        Iki yol var ve BUTCE ikisinde de AYNIdir (adil kiyas sarti):

          * "batch_2opt_oropt" -- gpu_snake.repair_tours: havuzun tum
            varyantlari TEK batch'te es-zamanli onarilir, butce butun havuz
            icin `budget_s`. Komsuluk: aday-listeli (kNN) 2-opt + Or-opt<=3;
            RELOCATE YOK. (Tarihsel/varsayilan davranis.)

          * digerleri -- ablasyon satirlarinin BIREBIR AYNI motoru, havuzdaki
            her tura TEK TEK uygulanir; her tura budget_s/`width` duser.

        ================= `width` NEDEN VAR (2026-07-27 DUZELTMESI) ==========
        Onceden pay `len(tours)` ile hesaplaniyordu -- yani TEKILLESTIRMEDEN
        SONRAKI aday sayisiyla. Bu, az cesitlenen havuzu ODULLENDIRIYORDU:

            pbk411, gpu_time=20s (ESKI davranis)
              greedy_snake_v3_repair  10 benzersiz -> tur basina  2.0 s
              ge_pool_repair        7 benzersiz -> tur basina  2.9 s
              ge_pool2_repair       2 benzersiz -> tur basina 10.0 s

        Greedy-Edge havuzu dogasi geregi cokuyor (10 aday -> 7 benzersiz) ve
        tam bu cokme ona aday basina 5x'e kadar DAHA FAZLA onarim suresi
        kazandiriyordu. Eski yorum bunun tersini iddia ediyordu ("kiyas v3'un
        lehine sapabilir"); sure ekseninde sapma GE lehineydi.

        Artik pay NOMINAL havuz genisligiyle (`width`, cagiranin bildirdigi
        10/10/4) hesaplanir. Sonuc:
          * aday BASINA onarim eforu tum satirlarda AYNI (kontrol edilen
            degisken gercekten havuzun kaynagi olur),
          * cokmus havuz toplam butcenin tamamini kullanamaz -- cesitlenmeme
            artik bir DEZAVANTAJ olarak gorunur, gizli bir avantaj degil,
          * hicbir satir digerinden fazla duvar-saati alamaz.
        `width` verilmezse eski davranisa duser (yalniz geriye uyumluluk).

        NOT: bu etki ancak deadline BAGLAYICI oldugunda ortaya cikar; kucuk n
        ve hizli komsuluklarda motor deadline'dan once yakinsar ve fark
        olusmaz. Buyuk n'de baglayicidir.

        Onarim satirlarinin motorlari tur BASINA calistigi icin burada bir
        diag toplanmaz (per-tur diag'lari tek satirda birlestirmek yanltici
        olurdu); yerine kazanan turun kendi diag'i kullanilir."""
        if mode == "batch_2opt_oropt":
            import gpu_snake as GS
            rep, rep_diag = GS.repair_tours(
                xs, ys, is_geo=(ewt == "GEO"), tours=tours,
                time_limit=budget_s)
            return [[int(c) for c in t] for t in rep], rep_diag
        _share = budget_s / max(1, width if width else len(tours))
        out, best_diag, best_c = [], None, None
        for _t in tours:
            _dl = time.perf_counter() + _share
            with _quiet():
                if mode == "window":
                    _d = build_window_repair_result(
                        work_inst, list(_t), window_size=10,
                        deadline=_dl)[1]
                elif mode == "window_adaptive":
                    _d = build_adaptive_window_repair_result(
                        work_inst, list(_t), window_sizes=(10, 12),
                        requeue_radius=1, deadline=_dl)[1]
                elif mode == "vnd_dense":
                    _d = lro.run_neighborhood_vnd(
                        work_inst, list(_t),
                        neighbor_limit_2opt=cfg["nb2"] * 2,
                        neighbor_limit_oropt=cfg["nbo"] * 2,
                        max_oropt_segment=6, k_relocate=cfg["rk"] + 20,
                        relocate_neighbor_limit=cfg["rnb"] * 2,
                        dense_levels=cfg["dense"], max_rounds=40,
                        deadline=_dl)
                else:
                    _d = REP.run_single_neighborhood(
                        work_inst, list(_t), mode=mode,
                        neighbor_limit_2opt=cfg["nb2"],
                        neighbor_limit_oropt=cfg["nbo"],
                        max_oropt_segment=3, k_relocate=cfg["rk"],
                        relocate_neighbor_limit=cfg["rnb"], max_rounds=40,
                        deadline=_dl)
            out.append(list(_d.tour))
            if best_c is None or _d.cost < best_c:
                best_c, best_diag = _d.cost, _d
        return out, best_diag

    def _tag_pool_repair(method_key, repair_key):
        """Kosulmus havuz satirina secilen ONARIM bilgisini isler: gosterim
        adi "Havuz ← <onarim>" olur ve ablasyon muadili (varsa) yazilir."""
        row = results.get(row_id(method_key, repair_key))
        if not row:
            return
        row["repair_key"] = repair_key
        row["repair_name"] = POOL_REPAIR_LABELS.get(repair_key, repair_key)
        _abl = POOL_REPAIR_ABLATION.get(repair_key)
        row["repair_ablation"] = _abl
        row["repair_ablation_name"] = PRETTY.get(_abl) if _abl else None
        row["name"] = display_name(method_key, repair_key=repair_key)
        _checkpoint()

    def _record_pool(method_key, labels, costs, raw_labels=None, raw_costs=None):
        """PARALEL VARYANT URETEN yontemlerin havuz istatistigi (2026-07-25,
        kullanici talebi). Bu yontemler tek tur uretmez: bir aday HAVUZU
        kurup en iyisini raporlarlar. Yalniz en iyiyi yazmak havuzun ne
        kazandirdigini GIZLER -- en kotu aday da yazilinca

            yayilim = en_kotu - en_iyi

        dogrudan okunur ve "havuz gercekten ise yariyor mu, yoksa adaylar
        birbirinin ayni mi?" sorusu tabloda cevaplanir. (Bu tam olarak
        kroA100'de olculen sorundu: v3 havuzunda 10 adaydan yalniz 3'u
        benzersizdi -> yayilim ~0 -> havuz genisligi bosa gidiyordu.)

        `labels`/`costs`: RAPORLANAN turlarin (onarim varsa onarim SONRASI)
        etiket ve gercek TSPLIB maliyetleri.
        `raw_labels`/`raw_costs`: varsa onarim ONCESI (ham insa) karsiliklari
        -- boylece "havuz zaten ayrisik miydi, yoksa onarim mi ayirdi?"
        ayrimi da gorulur. Saf kurucularda (v2/v3) verilmez."""
        row = _row_of(method_key)
        if not row or not costs:
            return
        _bi = min(range(len(costs)), key=costs.__getitem__)
        _wi = max(range(len(costs)), key=costs.__getitem__)
        info = {
            "size": len(costs),
            "unique": len({round(c, 6) for c in costs}),
            "best": {"label": labels[_bi], "cost": round(costs[_bi], 2),
                     "gap": _gap(costs[_bi], bks)},
            "worst": {"label": labels[_wi], "cost": round(costs[_wi], 2),
                      "gap": _gap(costs[_wi], bks)},
            "spread": round(costs[_wi] - costs[_bi], 2),
            "spread_pct": (round(100.0 * (costs[_wi] - costs[_bi]) / costs[_bi], 3)
                           if costs[_bi] > 0 else None),
        }
        if raw_costs:
            _rb = min(range(len(raw_costs)), key=raw_costs.__getitem__)
            _rw = max(range(len(raw_costs)), key=raw_costs.__getitem__)
            _rl = raw_labels or labels
            info["raw"] = {
                "size": len(raw_costs),
                "unique": len({round(c, 6) for c in raw_costs}),
                "best": {"label": _rl[_rb], "cost": round(raw_costs[_rb], 2),
                         "gap": _gap(raw_costs[_rb], bks)},
                "worst": {"label": _rl[_rw], "cost": round(raw_costs[_rw], 2),
                          "gap": _gap(raw_costs[_rw], bks)},
                "spread": round(raw_costs[_rw] - raw_costs[_rb], 2),
            }
        row["pool"] = info
        row["pool_size"] = info["size"]
        _log(f"      havuz: {info['size']} varyant ({info['unique']} benzersiz) "
             f"en iyi={info['best']['cost']:.0f} (gap {info['best']['gap']}%, "
             f"{info['best']['label']}) | en kötü={info['worst']['cost']:.0f} "
             f"(gap {info['worst']['gap']}%, {info['worst']['label']}) | "
             f"yayılım={info['spread']:.0f}")
        _checkpoint()

    def _tag_seed(method_key, seed_key):
        """Kosulmus satira tohum bilgisini isler (gosterim adi dahil)."""
        row = results.get(row_id(method_key, seed_key))
        if not row:
            return
        row["seed_key"] = seed_key
        row["seed_name"] = PRETTY.get(seed_key, seed_key)
        row["name"] = display_name(method_key, seed_key)

    # ---- pipeline TOHUMU: theta=0 STRIP (boustrophedon) turu. Eski Snake-Grid
    #      tohumu KALDIRILDI ("greedy_snake'ten kurtulduk" -- kullanici istegi):
    #      once theta dedektorleri strip turu uzerinde kosar, EN IYI skorlu
    #      dedektorun acisiyla kurulan strip turu pipeline'a (window repair ->
    #      line reassignment -> metasezgiseller) tohum olarak MIRAS kalir. ----
    # theta* dedektorleri ayrica greedy_snake_v1'in acisini da uretir
    # (v1 = tek theta*, tek kosum -- havuz yok).
    need_seed = bool(want & {"strip", "rotation_strip"})
    t0 = time.perf_counter()
    strip0_tour = None
    strip0_cost = None
    strip0_time = 0.0
    if need_seed:
        strip0_tour = E.snake_order(xs, ys, 0)
        strip0_cost = inst.tour_cost(strip0_tour)
        strip0_time = time.perf_counter() - t0

    # working instance + seed tour fed to window repair; theta inheritance
    # below may switch these to the theta*-rotated frame.
    work_inst = inst
    seed_tour = strip0_tour
    seed_cost = strip0_cost
    theta_star = 0.0
    rotation_applied = False

    if "nn" in want:
        t0 = time.perf_counter()
        nt = E.grid_nn_tour(xs, ys, 0)
        record("nn", inst.tour_cost(nt), time.perf_counter() - t0, tour=nt)

    # ---- strip / space-filling-curve construction (see hyperparameters
    #      ["strip"] for the literature reference). Same input (xs, ys,
    #      unrotated frame) and same output contract (tour + cost via
    #      record()) as "nn", so it is directly comparable in the ablation
    #      chart. Fixed strip count, theta=0: this row is the PIPELINE'S BASE
    #      CONSTRUCTION -- the theta* detectors below optimize exactly this
    #      tour's sweep angle, and the pipeline seeds from the winner. ----
    if "strip" in want and strip0_tour is not None:
        record("strip", strip0_cost, strip0_time, tour=strip0_tour)

    # ---- Hilbert / Morton space-filling curves — same "curve" family as
    #      strip/snake (see FAMILY dict above), literature reference point
    #      (Platzman & Bartholdi 1989) for the same-class comparison table.
    if "hilbert" in want:
        t0 = time.perf_counter()
        ht = E.hilbert_curve_tour(xs, ys)
        record("hilbert", inst.tour_cost(ht), time.perf_counter() - t0, tour=ht)

    if "morton" in want:
        t0 = time.perf_counter()
        mt = E.morton_curve_tour(xs, ys)
        record("morton", inst.tour_cost(mt), time.perf_counter() - t0, tour=mt)

    # ---- classical construction baselines (literature "gold standard" family
    #      the user asked to compare directly against): Nearest Insertion,
    #      Farthest Insertion, Greedy-Edge. Same input contract as "nn"/"strip"
    #      -- raw (xs, ys, unrotated) drives the greedy decisions, same start
    #      city (0) as "nn", TRUE TSPLIB cost via inst.tour_cost -- so all
    #      three land in the exact same ablation table/chart, standalone
    #      (they do not feed into window_repair/line_reassign/etc.). See
    #      hyperparameters["nearest_insertion"/"farthest_insertion"/
    #      "greedy_edge"] for literature references.
    # O(n^2) saf-Python ekleme kuruculari buyuk kumelerde (Ulke/Sanat
    # koleksiyonlari, n=100k-200k) SAATLERCE surer ve "toplu kosunun sonu
    # gelmiyor" sikayetinin ana kaynagidir -- esik ustunde acik bir notla
    # atlanir (dashboard "denendi, sonuc yok" kutusunda gorunur), sessizce
    # kaybolmaz. Esik altinda davranis degismez.
    for ikey, ifn in (("farthest_insertion", E.farthest_insertion_tour),):
        if ikey not in want:
            continue
        if size_limited(ikey):
            continue
        t0 = time.perf_counter()
        try:
            with _construction_budget(constr_budget_s):
                it_ = ifn(xs, ys, 0)
        except E.ConstructionTimeout:
            note_construction_timeout(ikey, time.perf_counter() - t0)
            continue
        record(ikey, inst.tour_cost(it_), time.perf_counter() - t0, tour=it_)

    # ---- Greedy-Edge. Aday listesi / dikiş kuralı PANELDEN seçilir
    #      (GE_VARIANT_SELECTABLE, 2026-07-26). Varsayılan "knn15_greedy"
    #      literatür rakibinin ta kendisidir (Johnson & McGeoch 1997) ve
    #      E.greedy_edge_tour(k=15)'e BİREBİR devreder -- eski satırlar
    #      geçerli kalır. Diğer seçimler karşı-olgusal DUYARLILIK satırlarıdır
    #      ve row_id sayesinde rakibin YANINA düşer (greedy_edge@full_greedy
    #      gibi), üstüne değil. ----
    if "greedy_edge" in want:
        _gv = resolve_ge_variant_choice("greedy_edge", _ge_variants_cfg)

        def _run_ge(_mode):
            """Tek bir Greedy-Edge varyantını koşar ve satırını etiketler."""
            _t0 = time.perf_counter()
            try:
                with _construction_budget(constr_budget_s):
                    _t = E.greedy_edge_ablation_tour(xs, ys, _mode)
            except E.ConstructionTimeout:
                note_construction_timeout("greedy_edge",
                                          time.perf_counter() - _t0)
                return
            except ValueError as _exc:
                # full_greedy'nin n sınırı: budanmamış O(n²) kenar kümesi
                # belleğe sığmıyor. Bu bir HATA değil, ölçülen bulgunun
                # kendisi (saf Greedy ölçeklenmez) -- sessizce kırpılmaz.
                note_unavailable("greedy_edge", str(_exc),
                                 time.perf_counter() - _t0)
                return
            record("greedy_edge", inst.tour_cost(_t),
                   time.perf_counter() - _t0, tour=_t, choice=_mode)
            _r = results[row_id("greedy_edge", _mode)]
            _r["ge_variant_key"] = _mode
            _r["ge_variant_name"] = GE_VARIANT_LABELS.get(_mode, _mode)
            _r["name"] = display_name("greedy_edge", ge_variant_key=_mode)
            # Yalnız varsayılan seçim LİTERATÜR RAKİBİDİR; diğerleri bilerek
            # zayıflatılmış/yavaşlatılmış karşı-olgulardır ve satır bunu
            # açıkça taşır ki tablo ikisini karıştırmasın.
            _r["is_literature_baseline"] = (_mode == DEFAULT_GE_VARIANT)
            _w = GE_VARIANT_WARNING.get(_mode)
            if _w:
                _r["ge_variant_warning"] = _w
                _log(f"   [not] greedy_edge varyantı '{_mode}' — {_w}")

        # KANONİK satır HER ZAMAN koşar. İki sebeple:
        #   1) greedy_edge DEFAULT_SEED'tir -- 8 ablasyon satırı ondan
        #      tohumlanır. Duyarlılık varyantı (özellikle knn15_random)
        #      tohum olarak sızarsa ablasyon tablosunun tamamı sessizce
        #      bozulur. _row_of aşağıda kanonik satıra sabitlenir; o satırın
        #      VAR OLMASINI garanti eden yer burasıdır.
        #   2) Duyarlılık deneyinin okunabilmesi için karşılaştırma tabanının
        #      aynı koşuda, aynı örnekte yanında durması gerekir.
        _run_ge(DEFAULT_GE_VARIANT)
        if _gv != DEFAULT_GE_VARIANT:
            _run_ge(_gv)

    # ---- theta* DEDEKTORLERI (pipeline tohumundan ONCE kosarlar) ----
    # Uc varyant ayni asla-kotu-olamaz garantili cekirdegi (theta=0 her zaman
    # degerlendirilir) farkli arama stratejisiyle kullanir; aci artik STRIP
    # TURU uzerinden bulunur ve strip'e uygulanir ("aci bulunuyorsa strip
    # icin bulunuyor") -- eski snake-tabanli arama ("rotation"/Snake Seyrek
    # Tarama) ve snake insa/dogrulama adimlari KALDIRILDI (cok pahaliydi).
    # EN IYI SKORU ALAN varyantin acisiyla kurulan strip turu asagida
    # pipeline'a tohum olarak (ve gpu_hybrid aci havuzuna) MIRAS kalir.
    #   rotation_strip: [-90,90] TAM tarama, her acida O(n) strip kurup
    #     gercek maliyetle skorlar (en genis arama; savings/worst temsili).
    #   rotation_hist / rotation_mean: salt geometri (komsu-yon histogrami /
    #     acisal ortalama) aday uretir, adaylar dar pencerede strip'in gercek
    #     maliyetiyle dogrulanir (en ucuz arama; tur insasi yalniz dogrulama).
    detector_best = None   # (cost, theta, tour, key)
    theta_strip = 0.0   # rotation_strip's OWN angle -- fed to the greedy/NN
                        # strip hybrids below (NOT the overall detector_best
                        # winner): they share strip's actual sort mechanism,
                        # so the angle optimized for strip's true cost is a
                        # matched-mechanism fit for them, unlike for nn/
                        # hilbert/morton where §6 showed it hurts (capture%
                        # negative) -- see SNAKE_GRID_SINIF_VE_GPU_HAT_
                        # AKADEMIK_NOT.md §9.
    # theta dedektorlerinin "time" alani YALNIZ arama/tespit eforu (rdiag
    # ["search_time"]) -- "yonu bulma" eforu; strip turunun kendisi zaten
    # O(n) oldugundan insa maliyeti ihmal edilebilir ve tarama sirasinda
    # zaten kurulmustur.
    #
    # "savings" (Tasarruf) alani = taramada denenen EN KOTU (worst) aday
    # eksi EN IYI (best) aday: bu dedektorlerin gercekten islevsel bir arama
    # yaptigini (rastgele bir aci secmekten iyi oldugunu) kanitlayan bir
    # sayidir. Yalniz GENIS/temsili tam-araligi ([-90,90]) tarayan
    # rotation_strip icin doldurulur -- rotation_hist/rotation_mean "salt
    # geometri" adaylarini DAR pencerelerde dogrular (bkz.
    # _best_strip_angle_geometric docstring), oralarda "en kotu" temsili
    # olmadigindan (yapisal olarak musait degil) bos birakilir.
    if "rotation_strip" in want and strip0_tour is not None:
        with _quiet():
            th_v, tour_v, cost_v, rdiag = _best_strip_angle_from_scan(
                inst, xs, ys, coarse_step=rotation_step)
        savings = (rdiag["worst_c"] - rdiag["best_c"]
                  if rdiag["best_c"] is not None and rdiag["worst_c"] not in (None, float("-inf"))
                  else None)
        record("rotation_strip", cost_v, rdiag["search_time"],
               tour=tour_v, initial=strip0_cost, savings=savings)
        results["rotation_strip"]["theta"] = round(th_v, 2)
        if savings is not None:
            results["rotation_strip"]["worst_cost"] = round(rdiag["worst_c"], 2)
            results["rotation_strip"]["worst_theta"] = round(rdiag["worst_a"], 2)
        theta_strip = th_v
        _log(f"[{name}] rotation_strip: theta*={th_v:.1f} deg  "
             f"cost={cost_v:.0f} search_t={rdiag['search_time']:.2f}s "
             f"savings={savings}")
        if detector_best is None or cost_v < detector_best[0] - 1e-9:
            detector_best = (cost_v, th_v, tour_v, "rotation_strip")

    # (rotation_hist / rotation_mean 2026-09-07'de kaldirildi; tek dedektor rotation_strip)

    # ---- MIRAS: pipeline tohumu, en iyi skorlu dedektorun acisiyla kurulan
    #      STRIP turu olur (yalniz gercekten theta=0 strip'ten iyiyse) ----
    theta_src = None
    if (detector_best is not None and strip0_cost is not None
            and detector_best[0] < strip0_cost - 1e-6
            and abs(detector_best[1]) >= 1e-6):
        theta_star = detector_best[1]
        theta_src = detector_best[3]
        seed_tour, seed_cost = detector_best[2], detector_best[0]
        # Run the rest of the pipeline in the theta-rotated frame (true costs
        # preserved); the rotated-frame strip ordering == seed_tour.
        work_inst, _ = E.make_instance(coords, ewt, theta_deg=theta_star)
        rotation_applied = True
        _log(f"[{name}] theta* mirasi: {theta_src} -> pipeline tohumu "
             f"theta={theta_star:.1f} deg strip (cost={seed_cost:.0f})")

    # Kosumun en iyi deterministik turu: onarim ailesi ve ILS buradan
    # baslar / bunu gunceller. Tohum = pipeline tohumu (theta* strip).
    run_best_cost, run_best_tour = seed_cost, seed_tour

    # ---- ONARIM KATMANI (TEK CATI, 2026-07-24) -----------------------------
    # Bes satirin tamami AYNI motoru (lro / repair.py) ve AYNI butce
    # sinirlarini kullanir; TEK bagimsiz degisken KOMSULUK KUMESIdir.
    # Baslangic turu artik sabit bir zincirden degil, admin panelinden SECILEN
    # insadan gelir (SEEDABLE) -> "Onarim — tam VND <- Greedy-Edge" gibi.
    #
    # Katlanan eski satirlar (ayni motor, tek fark tohumdu):
    #   line_reassign      == repair_relocate  (tek-nokta relocate)
    #   adaptive_lr        == repair_vnd       (lro.run_neighborhood_vnd)
    #   dense_adaptive_lr  -> repair_vnd_dense (genisletilmis komsuluk)
    #   ge_repair / nn_repair / fi_repair / greedy_snake_v2_repair
    #                      -> repair_vnd <- <ilgili insa>
    _REPAIR_FAMILY = {
        "repair_two_opt": "two_opt",
        "repair_or_opt": "or_opt",
        "repair_relocate": "relocate",
        "repair_vnd": "vnd",
        "repair_vnd_dense": "vnd_dense",
        # geometrik pencere onarimi: komsuluk hamlesi degil AYRI bir
        # mekanizma (tur uzerinde kayan pencere icinde yeniden siralama).
        # Ailede kalir cunku sozlesmesi ayni: tek girdi turu (panelden
        # secilen insa) -> iyilestirilmis tur.
        "repair_window": "window",
        "repair_window_adaptive": "window_adaptive",
    }
    def _run_repair_family():
      """Onarim ailesini BAGIMLILIK sirasinda kosar. Bir ablasyon satiri
      baska bir ablasyonun ciktisini girdi alabildigi icin sira sabit
      degildir: order_seedable() topolojik sirayi verir, dongu bulursa
      ilgili satiri DEFAULT_SEED'e dusurur. Havuzlardan SONRA cagrilir ki
      havuz ciktilari da tohum olarak secilebilsin."""
      nonlocal run_best_cost, run_best_tour, seed_of
      _order, seed_of, _broken = order_seedable(seed_of)
      for _b in _broken:
          _log(f"   [uyari] {_b}: tohum secimi dongu olusturuyordu — "
               f"{DEFAULT_SEED} kullanildi")
      for _rkey in _order:
        _mode = _REPAIR_FAMILY.get(_rkey)
        if _mode is None or _rkey not in want:
            continue
        _sk = seed_of.get(_rkey) or DEFAULT_SEED
        _st = _seed_tour_of(_sk)
        if _st is None:
            note_unavailable(
                _rkey, f"baslangic turu (tohum) uretilemedi — secilen girdi "
                       f"'{_sk}' bu ornekte tur dondurmedi", 0.0)
            continue
        _stour, _scost = _st
        t0 = time.perf_counter()
        # ==================== 2026-07-27 DUZELTMESI =====================
        # ONARIM KATMANININ TEK DUVAR-SAATI BUTCESI. Onceden bu satirlar
        # deadline ALMIYORDU (repair.run_single_neighborhood'da varsayilan
        # None) -- yani yakinsayana kadar SINIRSIZ kosuyorlardi. Havuzlu
        # satirlar ise ayni motoru cfg["gpu_time"] kapagi altinda cagiriyordu.
        # Dolayisiyla POOL_REPAIR_ABLATION'in ("bu havuz satirinin ablasyon
        # muadili sudur") vaadi BUTCE EKSENINDE gecersizdi:
        #     repair_vnd                 : 1 tur, SINIRSIZ sure
        #     greedy_snake_v3_repair@vnd    : 10 tur, her biri gpu_time/10
        # Artik ikisi de AYNI TOPLAM butceyi gorur:
        #     ablasyon  = 1 tur  x  gpu_time
        #     havuz     = W tur  x (gpu_time / W)   -> toplam gpu_time
        # Kontrol edilen degisken boylece gercekten "tek turu derinlemesine mi
        # onarayim, W turu paylastirarak mi" sorusu olur.
        # window_adaptive'in eski ozel kapagi (max(60, ils_time)) da buraya
        # katlandi -- onarim katmaninda artik TEK butce kavrami var.
        _rep_budget_s = cfg["gpu_time"]
        _dl = t0 + _rep_budget_s
        with _quiet():
            if _mode == "window":
                _rdiag = build_window_repair_result(
                    work_inst, _stour, window_size=10,
                    deadline=_dl)[1]
            elif _mode == "window_adaptive":
                _rdiag = build_adaptive_window_repair_result(
                    work_inst, _stour, window_sizes=(10, 12),
                    requeue_radius=1, deadline=_dl)[1]
            elif _mode == "vnd_dense":
                # genisletilmis komsuluk: Or-opt<=6 + daha genis aday
                # listeleri + yogun toplu hamleler (eski dense_adaptive_lr)
                _rdiag = lro.run_neighborhood_vnd(
                    work_inst, _stour,
                    neighbor_limit_2opt=cfg["nb2"] * 2,
                    neighbor_limit_oropt=cfg["nbo"] * 2,
                    max_oropt_segment=6, k_relocate=cfg["rk"] + 20,
                    relocate_neighbor_limit=cfg["rnb"] * 2,
                    dense_levels=cfg["dense"], max_rounds=40,
                    deadline=_dl)
            else:
                _rdiag = REP.run_single_neighborhood(
                    work_inst, _stour, mode=_mode,
                    neighbor_limit_2opt=cfg["nb2"],
                    neighbor_limit_oropt=cfg["nbo"],
                    max_oropt_segment=3, k_relocate=cfg["rk"],
                    relocate_neighbor_limit=cfg["rnb"], max_rounds=40,
                    deadline=_dl)
        # Her onarim satiri KENDI sonucunu raporlar; eski "kosum-en-iyisi ile
        # kirpma" davranisi kaldirildi (o, zincirleme merdivenin artigiydi ve
        # satirlari birbirine bagliyordu -- artik satirlar bagimsiz).
        if run_best_cost is None or _rdiag.cost < run_best_cost - 1e-9:
            run_best_cost, run_best_tour = _rdiag.cost, _rdiag.tour
        record(_rkey, _rdiag.cost, time.perf_counter() - t0, _rdiag,
               _rdiag.tour, initial=_scost)
        _tag_seed(_rkey, _sk)


    # ---- RGGE: Dondurulmus Izgara Greedy-Edge (2026-09-07) --------------
    # TEK TUR, HAVUZ YOK. Iki satir; tek degisken k-NN izgarasinin eksen
    # sirasi (devrik/duz). Aci makinesi RSGE ile BIREBIR ayni (_rsge_theta_in
    # asagida tanimli degil -- bu blok ondan ONCE geldigi icin kendi
    # cozucusunu kullanir; ikisi ayni kurala gore cozer, bkz. _run_snake_v1).
    def _rgge_theta_in(_ak, _mkey):
        """Aci kaynagini `_run_snake_v1` ile AYNI kurala gore cozer."""
        if _ak == "grid_theta":
            return None                     # rgge.theta_proxy (aile vekili)
        if _ak == "strip_oracle":
            return SA._strip_oracle_theta(xs, ys)
        if _ak == "zero":
            return 0.0
        if _ak == "random":                 # 2026-09-09 (hakem E8)
            return round(random.Random(20260909 + n).uniform(-45.0, 45.0), 4)
        if _ak == "theta_star":
            return theta_star if rotation_applied else 0.0
        _man = manual_angle_deg(_ak)
        if _man is not None:
            return _man
        _row = results.get(_ak) or {}
        _th = _row.get("theta")
        if _th is None:
            _log(f"   [uyari] {_mkey}: '{_ak}' bu koşuda açı üretmedi "
                 f"— panelde SEÇİLİ tek açı bu olduğu için satır hiç "
                 f"koşulmadı (dedektörü de seçili tutun ya da başka bir "
                 f"açı kaynağı seçin)")
            return "ATLA"
        return _th

    def _run_rgge(_mkey, _ak):
        _dev = RGGE_DEVRIK_OF[_mkey]
        _th_in = _rgge_theta_in(_ak, _mkey)
        if _th_in == "ATLA":
            return
        _t0 = time.perf_counter()
        try:
            with _construction_budget(constr_budget_s):
                _t, _th, _d = RG.rgge_tour(xs, ys, theta_deg=_th_in,
                                           devrik=_dev)
        except E.ConstructionTimeout:
            note_construction_timeout(_mkey, time.perf_counter() - _t0)
            return
        record(_mkey, inst.tour_cost(_t), time.perf_counter() - _t0,
               tour=_t, choice=_ak)
        _r = results[row_id(_mkey, _ak)]
        # MEKANIZMA OLCUSU yalniz bu satirlara yazilir (kullanici karari:
        # "diger yontemlere kesinlikle bulastirilmamali").
        _r.update({k: v for k, v in _d.items()})
        _r["theta"] = round(_th, 2)
        _r["angle_key"] = _ak
        _r["angle_name"] = angle_label(_ak)
        _r["name"] = display_name(_mkey, angle_key=_ak)
        _r["winner"] = (f"th={_th:g}/k={RG.RGGE_KNN}"
                        f"/eks={'y' if _dev else 'x'}")
        # RGGE bir MERDIVEN ailesi DEGILDIR: iki satir yan yana duran iki
        # cerceve, biri digerinin ust kumesi degil.
        _r["family_ladder"] = False
        _man = manual_angle_deg(_ak)
        if _man is not None:
            _r["manual_angle"] = round(_man, 4)
        if _ak == "grid_theta":
            import grid_theta as _GT
            _gd = dict(_GT.LAST)
            if _gd:
                _r["grid_conf"] = round(_gd.get("conf", 0.0), 3)
                _r["grid_used"] = bool(_gd.get("used_grid"))
        _tb = _d.get("rgge_tie_boundary")
        _tr = _d.get("rgge_tie_repeat")
        _log(f"[{name}] {_mkey}: θ={_th:g}° k={RG.RGGE_KNN} "
             f"ızgara={'DEVRİK' if _dev else 'DÜZ'} — tek tur"
             + (f" · beraberlik sınır={_tb:.3f}" if _tb is not None else ""))
        if _tb is not None and _tb <= 0.0:
            # MEKANIZMA KOSULU SAGLANMIYOR: es-uzaklikli komsu YOK, yani
            # cerceve degisikliginin (aci ya da devrik) etkisi YAPISAL
            # OLARAK sifira yakin olmali. Bu bir hata degil, tezin
            # dogrulanabilir tarafi -- satirda gorunur olmali.
            _log(f"   [not] {_mkey}: sınır beraberliği 0.000 — bu bulutta "
                 f"eş-uzaklıklı komşu YOK, çerçeve değişikliğinin etkisi "
                 f"yapısal olarak ihmal edilebilir (mekanizma tezi)")
        if _ak == "zero" and not _dev:
            _log(f"   [ablasyon] {_mkey} (θ=0 seçili) — θ=0 ve ızgara DÜZ: bu satır "
                 f"greedy_edge@knn8_greedy'nin TA KENDİSİDİR (çevrim olarak "
                 f"bit-aynı). Ailenin çapası budur.")

    # 2026-09-07 (kullanici karari): YALNIZ panelde secili acida kosulur.
    # Eskiden burada grid_theta + zero + secim olmak uzere UC satir vardi;
    # secilmeyen acilarda da tarama yapiliyordu ve tablodaki ana satir
    # (grid_theta) 99/106 kumede greedy_edge'in kopyasi cikiyordu.
    for _gk in RGGE_METHODS:
        if _gk not in want:
            continue
        _run_rgge(_gk, resolve_angle_choice(_gk, _panel_angle_config()))

    # ---- RSGE: Dondurulmus Serpantin Greedy-Edge (2026-09-07) ------------
    # Dort satir, dort BANT SAYISI KURALI, tek degisken `b`. Govde TEK
    # (`_run_rsge`) -- aksi halde dort satir "ayni deneyin dort noktasi"
    # olmaktan cikar ve aralarinda b'den baska farklar birikirdi.
    #
    # ONBELLEK: iki kural ayni b'yi verdiginde (kucuk/orta n'de
    # resource == budget == 1 BEKLENIR) tur BIREBIR aynidir -- ayni aci,
    # ayni bant sayisi, deterministik kurucu. O halde ikinci kez kurmak
    # saf israftir; satirlar yine AYRI kalir (her biri kendi kuralini ve
    # kendi b'sini tasir), yalnizca insa paylasilir.
    _rsge_cache: dict = {}

    def _rsge_theta_in(_ak, _mkey):
        """Aci kaynagini `_run_snake_v1` ile BIREBIR ayni kurala gore cozer.
        None -> snake_alt.theta_proxy (aile vekili). Cozulemezse sentinel."""
        if _ak == "grid_theta":
            return None
        if _ak == "strip_oracle":
            return SA._strip_oracle_theta(xs, ys)
        if _ak == "zero":
            return 0.0
        if _ak == "random":                 # 2026-09-09 (hakem E8)
            return round(random.Random(20260909 + n).uniform(-45.0, 45.0), 4)
        if _ak == "theta_star":
            return theta_star if rotation_applied else 0.0
        _man = manual_angle_deg(_ak)
        if _man is not None:
            return _man
        _row = results.get(_ak) or {}
        _th = _row.get("theta")
        if _th is None:
            _log(f"   [uyari] {_mkey}: '{_ak}' bu koşuda açı üretmedi "
                 f"— panelde SEÇİLİ tek açı bu olduğu için satır hiç "
                 f"koşulmadı (dedektörü de seçili tutun ya da başka bir "
                 f"açı kaynağı seçin)")
            return "ATLA"
        return _th

    def _run_rsge(_mkey, _ak):
        """RSGE satirlarinin TEK govdesi. `_mkey` -> RSGE_RULE_OF ile kurala."""
        _rule = RSGE_RULE_OF[_mkey]
        _th_in = _rsge_theta_in(_ak, _mkey)
        if _th_in == "ATLA":
            return
        _t0 = time.perf_counter()
        try:
            with _construction_budget(constr_budget_s):
                _t, _th, _breq, _d = SA.rsge_tour(xs, ys, rule=_rule,
                                                  theta_deg=_th_in)
                _ck = (round(_th, 6), int(_d.get("rsge_b", _breq)))
                _hit = _ck in _rsge_cache
                if _hit:
                    _t = list(_rsge_cache[_ck])
                else:
                    _rsge_cache[_ck] = list(_t)
        except E.ConstructionTimeout:
            note_construction_timeout(_mkey, time.perf_counter() - _t0)
            return
        record(_mkey, inst.tour_cost(_t), time.perf_counter() - _t0,
               tour=_t, choice=_ak)
        _r = results[row_id(_mkey, _ak)]
        _r.update({k: v for k, v in _d.items()})
        _r["theta"] = round(_th, 2)
        _r["angle_key"] = _ak
        _r["angle_name"] = angle_label(_ak)
        _r["name"] = display_name(_mkey, angle_key=_ak)
        _r["winner"] = f"th={_th:g}/k={SA.RSGE_KNN}/b={_d.get('rsge_b', _breq)}"
        # RSGE bir MERDIVEN ailesi DEGILDIR (v1 ⊂ v2 ⊂ v3 gibi bir
        # ic-ice gecme iddiasi tasimaz): dort satir b ekseninde YAN YANA
        # duran noktalardir, biri digerinin ust kumesi degil.
        _r["family_ladder"] = False
        _man = manual_angle_deg(_ak)
        if _man is not None:
            _r["manual_angle"] = round(_man, 4)
        if _ak == "grid_theta":
            import grid_theta as _GT
            _gd = dict(_GT.LAST)
            if _gd:
                _r["grid_conf"] = round(_gd.get("conf", 0.0), 3)
                _r["grid_used"] = bool(_gd.get("used_grid"))
                _r["grid_void"] = round(_gd.get("void", 0.0), 3)
        _bq, _be = int(_breq), int(_d.get("rsge_b", _breq))
        _extra = "" if _bq == _be else f" (istenen {_bq}, boş bantlar elendi)"
        _log(f"[{name}] {_mkey}: θ={_th:g}° kural={_rule} bant={_be}"
             f"{_extra} k={SA.RSGE_KNN}"
             f"{' · inşa önbellekten' if _hit else ''}")
        if _be == 1:
            # OZDESLIK SATIRI: b=1'de RSGE, dondurulmus koordinatlarda
            # GLOBAL greedy-edge'in TA KENDISIDIR (bkz. snake_alt RSGE
            # blogu). Bu bir kusur degil, KURALIN VERDIGI CEVAPTIR --
            # "bu ornekte bantlamayi reddetti" diye okunur.
            _log(f"   [not] {_mkey}: b=1 — bu kural bu örnekte BANTLAMAYI "
                 f"REDDETTİ; satır, θ={_th:g}°'de global greedy-edge'in "
                 f"kendisidir")

    # 2026-09-07 (kullanici karari): RGGE ile AYNI kural -- yalniz panelde
    # secili acida kosulur. "@zero" ablasyonu icin panelden θ=0 secilmelidir.
    for _rk in RSGE_METHODS:
        if _rk not in want:
            continue
        _run_rsge(_rk, resolve_angle_choice(_rk, _panel_angle_config()))

    # ---- CERCEVE DENEYI (makale, 2026-09-07) -- frame_methods.py ----------
    # Ilkeller paper_experiments/frame_experiment.py'den gelir (tek kaynak);
    # burada yalniz panel satirina cevrilir. Aci kumesi panelin rotation
    # step'inden (n buyudukce kabalasir, bkz. FM.angles_for).
    _fs_angles = FM.angles_for(n, cfg.get("rotation_step", 5.0))
    _fs_cost = inst.tour_cost

    def _fs_guard(_key, _fn, *_a, **_kw):
        _t0 = time.perf_counter()
        try:
            with _construction_budget(constr_budget_s):
                return _fn(*_a, **_kw)
        except E.ConstructionTimeout:
            note_construction_timeout(_key, time.perf_counter() - _t0)
            return None

    if "nn_exact" in want:
        _t0 = time.perf_counter()
        _t = _fs_guard("nn_exact", FM.nn_exact, xs, ys)
        if _t is not None:
            record("nn_exact", _fs_cost(_t), time.perf_counter() - _t0, tour=_t)

    def _fs_record_sweep(_fk, _r, _label):
        record(_fk, _r["cost0"], _r["build_time"], tour=_r["tour0"],
               savings=_r["worst_cost"] - _r["best_cost"])
        _row = results[row_id(_fk, "")]
        _row.update({k: v for k, v in _r.items() if k not in ("tour0", "cost0")})
        _row["cost0"] = round(_r["cost0"], 2)
        _row["sweep_range_pct"] = round(_r["range_pct"], 4)
        _row["winner"] = _label
        _log(f"[{name}] {_fk}: {_label}")

    for _fk, _fm in FM.SWEEP_KEYS.items():
        if _fk not in want:
            continue
        if _fm == "fi" and n > FM.FI_MAX_N:
            note_unavailable(_fk, f"n={n} > {FM.FI_MAX_N}: saf O(n^2) farthest-"
                             f"insertion aci taramasi bu boyutta kosulmaz",
                             0.0, reason="size_limit", budget_s=constr_budget_s)
            continue
        _r = _fs_guard(_fk, FM.sweep, _fm, xs, ys, _fs_cost, _fs_angles)
        if _r is None:
            continue
        _fs_record_sweep(_fk, _r,
                         f"aralık={_r['range_pct']:.2f}% ({_r['n_angles']} açı; "
                         f"en iyi θ={_r['best_theta']:g}°, en kötü θ={_r['worst_theta']:g}°)")

    if "fs_ge8_jitter" in want:
        _r = _fs_guard("fs_ge8_jitter", FM.jitter, xs, ys, _fs_cost, len(_fs_angles))
        if _r is not None:
            _fs_record_sweep("fs_ge8_jitter", _r,
                             f"jitter aralığı={_r['range_pct']:.2f}% ({_r['n_seeds']} tohum, θ=0)")
    if "fs_ge8_detied" in want:
        _r = _fs_guard("fs_ge8_detied", FM.detied, xs, ys, _fs_cost, _fs_angles)
        if _r is not None:
            _fs_record_sweep("fs_ge8_detied", _r,
                             f"beraberliksiz aralık={_r['range_pct']:.3f}% "
                             f"(beraberlik oranı {_r['tie_frac_before']:.3f} → {_r['tie_frac_after']:.3f})")
    if "fs_ge8_exactknn" in want:
        if n > FM.EXACTKNN_MAX_N:
            note_unavailable("fs_ge8_exactknn", f"n={n} > {FM.EXACTKNN_MAX_N}: kesin O(n^2) k-NN "
                             f"taramasi bu boyutta kosulmaz", 0.0, reason="size_limit",
                             budget_s=constr_budget_s)
        else:
            _r = _fs_guard("fs_ge8_exactknn", FM.detied_exactknn, xs, ys, _fs_cost, _fs_angles)
            if _r is not None:
                _fs_record_sweep("fs_ge8_exactknn", _r,
                                 f"beraberliksiz + KESİN k-NN aralık={_r['range_pct']:.3f}% "
                                 f"({_r['n_angles']} açı; benzersiz maliyet "
                                 f"{len(set(_r['sweep'].values()))})")

    # 2026-09-09 (hakem A3/E9): "degismez" etiketli diger kurucularda beraberliksiz kontrol
    for _dk, _dm in FM.DETIED_KEYS.items():
        if _dk not in want:
            continue
        if _dm == "fi" and n > FM.FI_MAX_N:
            note_unavailable(_dk, f"n={n} > {FM.FI_MAX_N}: saf O(n^2) farthest-insertion "
                             f"beraberliksiz taramasi bu boyutta kosulmaz", 0.0,
                             reason="size_limit", budget_s=constr_budget_s)
            continue
        _r = _fs_guard(_dk, FM.detied, xs, ys, _fs_cost, _fs_angles, _dm)
        if _r is not None:
            _fs_record_sweep(_dk, _r,
                             f"beraberliksiz aralık={_r['range_pct']:.3f}% ({_dm}; "
                             f"beraberlik oranı {_r['tie_frac_before']:.3f} → {_r['tie_frac_after']:.3f})")

    if "fs_band_ge8" in want:
        _r = _fs_guard("fs_band_ge8", FM.bands, xs, ys, _fs_cost)
        if _r is not None:
            _c1 = _r["cost_b1"]
            record("fs_band_ge8", _r["cost"], _r["build_time"], tour=_r["tour"],
                   initial=_c1, savings=(_c1 - _r["cost"]) if _c1 is not None else None)
            _row = results[row_id("fs_band_ge8", "")]
            _row.update(bands=_r["bands"], best_b=_r["best_b"], ks=_r.get("ks"),
                        worst_cost=round(_r["worst_cost"], 2))
            _row["winner"] = (f"en iyi b={_r['best_b']}; b=1'e göre "
                              f"{100.0 * (_r['cost'] / _c1 - 1.0):+.2f}%" if _c1 else f"b={_r['best_b']}")
            _log(f"[{name}] fs_band_ge8: {_row['winner']} | "
                 + ", ".join(f"b={b['b']}:{_gap(b['cost'], bks)}%" for b in _r["bands"]))

    _ge8_base = None
    for _rk, _rm in FM.REALIGN_KEYS.items():
        if _rk not in want:
            continue
        _ak = resolve_angle_choice(_rk, _panel_angle_config())
        _phi = _rsge_theta_in(_ak, _rk)
        if _phi == "ATLA":
            continue
        if _phi is None:                    # grid_theta secildiyse: dedektor acisi
            _phi = SA.theta_proxy(xs, ys)
        _r = _fs_guard(_rk, FM.realign, _rm, xs, ys, _fs_cost, float(_phi), _ge8_base)
        if _r is None:
            continue
        record(_rk, _r["cost"], _r["build_time"], tour=_r["tour"],
               initial=_r["mis_cost"], savings=_r["mis_cost"] - _r["cost"], choice=_ak)
        _row = results[row_id(_rk, _ak)]
        _row.update(phi=round(float(_phi), 4), theta=round(_r["theta_hat"], 4),
                    theta_hat=_r["theta_hat"], err_deg=_r["err_deg"], det=_r["det"],
                    snap=_r["snap"], mis_cost=round(_r["mis_cost"], 2),
                    angle_key=_ak, angle_name=angle_label(_ak),
                    name=display_name(_rk, angle_key=_ak), family_ladder=False)
        if "ge8_tour_identical" in _r:
            _row["ge8_tour_identical"] = bool(_r["ge8_tour_identical"])
        _row["winner"] = (f"φ={float(_phi):g}° → θ̂={_r['theta_hat']:.4f}° "
                          f"(hata {_r['err_deg']:.4f}°) kafes="
                          f"{'birebir' if _r['snap'].get('exact_recovery') else 'yok'}"
                          + (" · GE turu bit-aynı" if _r.get("ge8_tour_identical") else ""))
        _log(f"[{name}] {_rk}: {_row['winner']} | hizasız gap "
             f"{_gap(_r['mis_cost'], bks)}% → {_gap(_r['cost'], bks)}%")

    # 2026-09-09 (hakem E8): rastgele phi + konum gurultusu + nokta silme altinda snap
    if FM.REALIGN_NOISE_KEY in want:
        _nk = FM.REALIGN_NOISE_KEY
        _ak = resolve_angle_choice(_nk, _panel_angle_config())
        _phi = _rsge_theta_in(_ak, _nk)
        if _phi is None:
            _phi = SA.theta_proxy(xs, ys)
        if _phi != "ATLA":
            _r = _fs_guard(_nk, FM.realign_noise, "strip", xs, ys, _fs_cost, float(_phi))
            if _r is not None:
                record(_nk, _r["cost"], _r["build_time"], tour=_r["tour"],
                       initial=_r["mis_cost"], savings=_r["mis_cost"] - _r["cost"], choice=_ak)
                _row = results[row_id(_nk, _ak)]
                _row.update(phi=round(float(_phi), 4), theta=round(_r["theta_hat"], 4),
                            theta_hat=_r["theta_hat"], err_deg=_r["err_deg"], det=_r["det"],
                            snap=_r["snap"], sigma=round(_r["sigma"], 4), noise_rel=_r["noise_rel"],
                            mis_cost=round(_r["mis_cost"], 2), oracle_cost=round(_r["oracle_cost"], 2),
                            drop=_r["drop"], angle_key=_ak, angle_name=angle_label(_ak),
                            name=display_name(_nk, angle_key=_ak), family_ladder=False)
                _row["winner"] = (f"φ={float(_phi):g}° σ={_r['sigma']:.3f} → θ̂ hata {_r['err_deg']:.4f}°; "
                                  f"silme: " + ", ".join(f"%{100 * float(k):g}: {v['err_deg']:.4f}°" for k, v in _r["drop"].items()))
                _log(f"[{name}] {_nk}: {_row['winner']} | gap hizasız {_gap(_r['mis_cost'], bks)}% → "
                     f"geri {_gap(_r['cost'], bks)}% (oracle {_gap(_r['oracle_cost'], bks)}%)")

    # ---- PARALEL GE (k drone) -- parallel_ge.py / frame_methods.pge ----------
    # Uc bolumleme stratejisi (band / kmeans / split) x k; bant-ici kurucu hep
    # GE k=8. Satir maliyeti = k kapali turun TOPLAMI (gap BKS'ye gore
    # okunur: k tur >= 1 tur), makespan / denge / paralel sure alanlarda.
    _pge_want = [k for k in list(FM.PGE_KEYS) + [FM.PGE_SWEEP_KEY, FM.PGE_KMSEED_KEY] + list(FM.PGR_KEYS) if k in want]
    if _pge_want:
        _tg0 = time.perf_counter()
        _g_tour = E.greedy_edge_tour(xs, ys, k=8)
        _t_global = time.perf_counter() - _tg0
        _g_cost = inst.tour_cost(_g_tour)

        def _pge_theta(_key):
            if _key not in ANGLE_SELECTABLE:
                return 0.0, ""
            _ak = resolve_angle_choice(_key, _panel_angle_config())
            _th = _rsge_theta_in(_ak, _key)
            if _th == "ATLA":
                return None, _ak
            if _th is None:
                _th = SA.theta_proxy(xs, ys)
            return float(_th), _ak

        def _pge_fill(_row, _r):
            _row.update(total=round(_r["total"], 2), makespan=round(_r["makespan"], 2),
                        imbalance=round(_r["imbalance"], 4), sizes=_r["sizes"], lens=_r["lens"],
                        k=_r["k"], t_part=round(_r["t_part"], 4), t_seq=round(_r["t_seq"], 4),
                        t_par=round(_r["t_par"], 4), part_times=_r["times"],
                        speedup=round(_t_global / max(_r["t_par"], 1e-9), 2),
                        ge_global_cost=round(_g_cost, 2), ge_global_time=round(_t_global, 4),
                        makespan_ratio=round(_r["makespan"] / (_g_cost / max(_r["k"], 1)), 4),
                        strategy=_r["strategy"], theta=round(_r["theta"], 4))

        for _pk in FM.PGE_KEYS:
            if _pk not in want:
                continue
            _s, _k = FM.PGE_KEYS[_pk]
            _th, _ak = _pge_theta(_pk)
            if _th is None:
                continue
            _t0 = time.perf_counter()
            try:
                with _construction_budget(constr_budget_s):
                    _r = FM.pge(_s, _k, xs, ys, ewt, theta=_th, global_tour=_g_tour, t_global=_t_global)
            except E.ConstructionTimeout:
                note_construction_timeout(_pk, time.perf_counter() - _t0)
                continue
            record(_pk, _r["total"], _r["t_seq"], tour=PG_concat(_r["tours"]),
                   initial=_g_cost, choice=(_ak or None))
            _row = results[row_id(_pk, _ak)]
            _pge_fill(_row, _r)
            _row["parts"] = _r["tours"]
            if _ak:
                _row.update(angle_key=_ak, angle_name=angle_label(_ak),
                            name=display_name(_pk, angle_key=_ak), family_ladder=False)
            _row["winner"] = (f"k={_r['k']} makespan={_r['makespan']:.0f} "
                              f"({_row['makespan_ratio']:.2f}× L/k) denge={_r['imbalance']:.2f} "
                              f"t_par={_r['t_par']:.3f}s hız={_row['speedup']}×")
            _log(f"[{name}] {_pk}: toplam gap {_gap(_r['total'], bks)}% | {_row['winner']}")

        if FM.PGE_SWEEP_KEY in want:
            _th, _ = _pge_theta("pge_band_k2")     # bant acisi: bant satirlariyla ayni kaynak
            _th = 0.0 if _th is None else _th
            _t0 = time.perf_counter()
            try:
                with _construction_budget(constr_budget_s):
                    _rows = FM.pge_sweep(xs, ys, ewt, theta=_th, global_tour=_g_tour, t_global=_t_global)
            except E.ConstructionTimeout:
                note_construction_timeout(FM.PGE_SWEEP_KEY, time.perf_counter() - _t0)
                _rows = None
            if _rows:
                record(FM.PGE_SWEEP_KEY, _g_cost, time.perf_counter() - _t0, tour=_g_tour)
                _row = results[row_id(FM.PGE_SWEEP_KEY, "")]
                _row["ksweep"] = [{kk: v for kk, v in r.items() if kk != "tours"} for r in _rows]
                _row["theta"] = round(_th, 4)
                _best = min(_rows, key=lambda r: r["makespan"] * (1.0 + 0.0))
                # k'ya gore en iyi strateji (makespan)
                _bk = {}
                for r in _rows:
                    kk = r["k_requested"]
                    if kk not in _bk or r["makespan"] < _bk[kk]["makespan"]:
                        _bk[kk] = r
                _row["best_by_k"] = {str(kk): dict(strategy=r["strategy"], makespan=round(r["makespan"], 2),
                                                   total=round(r["total"], 2), imbalance=round(r["imbalance"], 4),
                                                   t_par=round(r["t_par"], 4)) for kk, r in _bk.items()}
                _row["winner"] = "; ".join(f"k={kk}:{r['strategy']}" for kk, r in sorted(_bk.items()))
                _log(f"[{name}] pge_ksweep: en iyi strateji (makespan) k'ya göre → {_row['winner']}")

        # 2026-09-09 (hakem E10): k-means++ tohum duyarliligi
        if FM.PGE_KMSEED_KEY in want:
            _t0 = time.perf_counter()
            try:
                with _construction_budget(constr_budget_s):
                    _rows = FM.pge_kmeans_seeds(xs, ys, ewt, global_tour=_g_tour, t_global=_t_global)
            except E.ConstructionTimeout:
                note_construction_timeout(FM.PGE_KMSEED_KEY, time.perf_counter() - _t0)
                _rows = None
            if _rows:
                record(FM.PGE_KMSEED_KEY, _g_cost, time.perf_counter() - _t0, tour=_g_tour)
                _row = results[row_id(FM.PGE_KMSEED_KEY, "")]
                for r in _rows:
                    r["makespan_ratio"] = round(r["makespan"] / (_g_cost / r["k"]), 4)
                _row["kmseed"] = _rows
                _row["ge_global_cost"] = round(_g_cost, 2)
                _sum = []
                for kk in sorted({r["k"] for r in _rows}):
                    v = [r["makespan_ratio"] for r in _rows if r["k"] == kk]
                    _sum.append(f"k={kk}: {min(v):.3f}–{max(v):.3f}")
                _row["winner"] = "tohumlar arası makespan oranı " + "; ".join(_sum)
                _log(f"[{name}] pge_kmseed: {_row['winner']}")

        # ---- pgr_*: paralel GE + parca basina VND (cerceve yarisi) ---------------
        # Onarim butcesi panelin onarim satiriyla ayni (cfg["gpu_time"]), her
        # parcaya ayri; paralel duvar-saati = en uzun parca.
        for _pk, (_s, _k) in FM.PGR_KEYS.items():
            if _pk not in want:
                continue
            _th, _ak = _pge_theta(_pk)
            if _th is None:
                continue
            _t0 = time.perf_counter()
            try:
                _r = FM.pgr(_s, _k, xs, ys, ewt, theta=_th, cfg=cfg, budget_s=cfg["gpu_time"],
                            global_tour=_g_tour, t_global=_t_global)
            except E.ConstructionTimeout:
                note_construction_timeout(_pk, time.perf_counter() - _t0)
                continue
            record(_pk, _r["total_rep"], _r["t_seq"] + _r["t_rep_seq"], tour=PG_concat(_r["tours_rep"]),
                   initial=_r["total"], savings=_r["total"] - _r["total_rep"], choice=(_ak or None))
            _row = results[row_id(_pk, _ak)]
            _pge_fill(_row, _r)
            _row.update(total_before=round(_r["total"], 2), total_rep=round(_r["total_rep"], 2),
                        makespan_before=round(_r["makespan"], 2), makespan_rep=round(_r["makespan_rep"], 2),
                        makespan_ratio_rep=round(_r["makespan_rep"] / (_g_cost / max(_r["k"], 1)), 4),
                        imbalance_rep=round(_r["imbalance_rep"], 4), lens_rep=_r["lens_rep"],
                        t_rep_par=round(_r["t_rep_par"], 3), t_rep_seq=round(_r["t_rep_seq"], 3),
                        t_par_total=round(_r["t_par_total"], 3), parts=_r["tours_rep"])
            if _ak:
                _row.update(angle_key=_ak, angle_name=angle_label(_ak),
                            name=display_name(_pk, angle_key=_ak), family_ladder=False)
            _row["winner"] = (f"k={_r['k']} makespan {_r['makespan']:.0f}→{_r['makespan_rep']:.0f} "
                              f"({_row['makespan_ratio_rep']:.2f}× L/k onarım sonrası) "
                              f"toplam {_r['total']:.0f}→{_r['total_rep']:.0f} t_par={_r['t_par_total']:.2f}s")
            _log(f"[{name}] {_pk}: {_row['winner']}")

    # ---- stochastic metaheuristics start from the best deterministic tour ----
    base = run_best_tour
    base_cost = run_best_cost
    # ---- ONARIM AILESI burada kosar: havuzlardan SONRA, ILS'ten ONCE.
    #      Boylece bir ablasyon satiri hem yapicilarin hem havuzlarin hem de
    #      DIGER ablasyonlarin ciktisini girdi olarak secebilir. ----
    _run_repair_family()

    # ---- ILS (tek satir, tohumu PANELDEN secilir): motor/butce sabit, tek
    #      bagimsiz degisken baslangic turudur. Tohum insasi rotasyonsuz
    #      cercevede (inst) puanlandigi icin ILS de `inst` uzerinde kosar ->
    #      raporlanan initial_cost, tohum insasinin kaydettigi maliyetle
    #      birebir tutar (Delta = init_gap - final_gap durust olculur). ----
    if "ils" in want:
        _sk = seed_of.get("ils") or DEFAULT_SEED
        _st = _seed_tour_of(_sk)
        if _st is None:
            note_unavailable(
                "ils", f"baslangic turu (tohum) uretilemedi — secilen insa "
                       f"'{_sk}' bu ornekte tur dondurmedi", 0.0)
        else:
            _stour, _scost = _st
            # TOHUMUN TOHUMU (2026-09-08, cerceve yarisi): ILS bir KOPYA satirdan
            # (orn. repair_vnd@strip) tohumlaniyorsa satir kimligi o kopyayi tasir
            # -> "ils@repair_vnd@strip"; aksi halde ILS<-VND(GE) ile ILS<-VND(strip)
            # ayni "ils@repair_vnd" kimliginde birbirini ezerdi. Kullanici sozlesmesi:
            # ILS tek basina degil, birlesik onarimin (VND) USTUNE kosulur.
            _srow = _row_of(_sk) or {}
            _seed_rid = _srow.get("row_id") or _sk
            _ch = _seed_rid if ("@" in _seed_rid) else None
            _log(f"   [ils] tohum = {_seed_rid} "
                 f"(cost={_scost:.0f}, gap={_gap(_scost, bks)}%)")
            _run_stochastic("ils", _stour, inst, bks, cfg,
                            (lambda *a, **kw: record(*a, **{**kw, "choice": _ch})) if _ch else record,
                            n, ckpt_file=RESULTS / f".{name}.ils.{_seed_rid.replace('@', '_')}.ckpt.json")
            if _ch:
                _row = results.get(row_id("ils", _ch))
                if _row:
                    _row["seed_key"] = _sk
                    _row["seed_row_id"] = _seed_rid
                    _row["seed_name"] = _srow.get("name", _sk)
                    _row["name"] = f"{PRETTY['ils']} ← {_srow.get('name', _sk)}"
            else:
                _tag_seed("ils", _sk)

    # ---- EXACT / ALTIN STANDART referans cizgisi: LKH-3 ve Concorde.
    #      EN SONDA kosarlar ve BILEREK hicbir seyi tohumlamazlar (bkz.
    #      SEED_CHOICES: ikisi de yok, ve bu blok run_best_tour'u GUNCELLEMEZ)
    #      -- bir sezgiseli kesin cozucunun turundan baslatip sonucu o
    #      sezgiselin basarisi diye raporlamak kiyas degil, bulasmadir.
    #
    #      Ikisi de her yontemin koordinatlarinin geldigi AYNI data/{ad}.tsp
    #      dosyasini okur (E.parse_tsp ile birebir ayni dosya), donen tur bu
    #      projenin KENDI inst.tour_cost'u ile puanlanir -> cozucunun kendi
    #      raporladigi maliyet ile tablodaki maliyet sessizce ayrisamaz.
    #
    #      Ikili kurulu degilse / butce yetmezse satir "denendi, sonuc yok"
    #      notuna duser (note_unavailable) -- kosum durmaz. Diskte hala GECERLI
    #      bir onceki sonuc varsa taze bir basarisizlik onu EZMEZ. ----
    def _note_exact_failure(key, status, elapsed, budget_s):
        """Kesin cozucu basarisizligini not eder; SURE ASIMI ise hafizaya yazar.

        Sure asimi disindaki basarisizliklar (ikili kurulu degil, cokme) NOTA
        duser ama HAFIZAYA GIRMEZ -- kullanici ikiliyi kurdugunda ya da hata
        duzeldiginde satir yeniden kosmali. Sure asiminda `reason`/`budget_s`
        yazilir ve satir ayni butceyle bir daha denenmez.
        """
        b = exact_timeout_budget(status)
        if b is None:
            note_unavailable(key, status, elapsed)
            return
        note_unavailable(
            key,
            f"{status} — bu bütçeyle bir daha denenmez; denemek için "
            f"“{BUDGET_FLAG['exact_budget']}” değerini büyütün",
            elapsed, reason="exact_budget", budget_s=budget_s)

    if "lkh3" in want:
        t0 = time.perf_counter()
        tour, status = ES.run_lkh(path, n, time_limit=cfg["lkh_time"])
        dt = time.perf_counter() - t0
        if tour is None:
            if results.get("lkh3", {}).get("tour"):
                _log(f"   [skip] LKH-3: {status} (önceki sonuç korunuyor)")
            else:
                _note_exact_failure("lkh3", status, dt, cfg["lkh_time"])
        else:
            record("lkh3", inst.tour_cost(tour), dt, tour=tour)
            _log(f"[{name}] LKH-3: {status}")

    if "concorde" in want:
        t0 = time.perf_counter()
        tour, status = ES.run_concorde(path, n, time_limit=cfg["concorde_time"])
        dt = time.perf_counter() - t0
        if tour is None:
            if results.get("concorde", {}).get("tour"):
                _log(f"   [skip] Concorde: {status} (önceki sonuç korunuyor)")
            else:
                _note_exact_failure("concorde", status, dt,
                                    cfg["concorde_time"])
        else:
            record("concorde", inst.tour_cost(tour), dt, tour=tour)
            _log(f"[{name}] Concorde: {status}")

    # ---- merge with any existing on-disk result (CRITICAL: a partial
    #      `--methods` run must never erase methods it didn't run). Koşu
    #      BAŞINDA alınan `_merge_snapshot` kullanılır -- dosya koşu sırasında
    #      checkpoint'lerle zaten ezildiği için diski burada yeniden okumak
    #      kısmi içeriği "önceki sonuç" sanma hatasına yol açar (2026-07-23
    #      düzeltmesi). ----
    out = RESULTS / f"{name}.json"
    prior_payload = _merge_payload
    prior_methods = _merge_snapshot
    for k, m in prior_methods.items():
        # REMOVED_METHODS: bu koddan kaldirilan eski yontemlerin diskte kalan
        # satirlari merge'e ALINMAZ -- boylece bir sonraki kosum eski satirlari
        # kalici olarak temizler.
        if is_dropped_method(k):
            continue
        results.setdefault(k, m)
    if prior_payload and not (want & {"rotation_strip", "rotation_hist",
                                      "rotation_mean"}):
        # no theta detector was recomputed this run; keep the previously-
        # established frame info instead of reporting the fresh defaults.
        theta_star = prior_payload.get("theta_star", theta_star)
        rotation_applied = prior_payload.get("rotation_applied", rotation_applied)
        seed_cost = prior_payload.get("seed_cost", seed_cost)
    # carry forward a prior "attempted, no result" note for a solver this run
    # did NOT retry, as long as it still has no real result (same don't-erase
    # rule as the `results` merge above) -- a note this run DID (re)produce
    # always wins over a stale one.
    if prior_payload:
        seen_note_keys = {nt["key"] for nt in solver_notes}
        for nt in prior_payload.get("solver_notes", []):
            if (nt["key"] not in seen_note_keys
                    and nt["key"] not in {split_row_id(r)[0] for r in results}
                    and nt["key"] not in want
                    and not is_dropped_method(nt["key"])):
                solver_notes.append(nt)

    # ---- statistical comparisons: paired Wilcoxon signed-rank over every
    #      pair of stochastic methods that ran (ILS ailesi) ----
    def _pair_comparison(ka, kb):
        if ka not in results or kb not in results:
            return None
        if not (results[ka].get("stochastic") and results[kb].get("stochastic")):
            return None
        a_s = {s["seed"]: s["cost"] for s in results[ka]["stochastic"]["seeds"]}
        b_s = {s["seed"]: s["cost"] for s in results[kb]["stochastic"]["seeds"]}
        common = sorted(set(a_s) & set(b_s))
        if not common:
            return None
        x = [a_s[s] for s in common]
        y = [b_s[s] for s in common]
        wil = _wilcoxon_signed_rank(x, y)
        mx, my = statistics.mean(x), statistics.mean(y)
        comp = {
            "pair": f"{ka}_vs_{kb}", "a": ka, "b": kb, "n_paired": len(common),
            f"{ka}_mean": round(mx, 2), f"{kb}_mean": round(my, 2),
            "wilcoxon": wil,
            "better": (ka if mx < my else kb if my < mx else "tie"),
        }
        _log(f"[{name}] {ka} vs {kb}: {ka}_mean={comp[f'{ka}_mean']} "
             f"{kb}_mean={comp[f'{kb}_mean']} Wilcoxon p={wil['p']} n={wil['n']}")
        return comp

    # ILS ailesi TEK satira katlandi (tohum panelden secilir); "ayni motor,
    # farkli tohum" karsilastirmasi artik AYRI kosumlarin sonuc dosyalari
    # arasinda yapilir, kosum-ici eslestirmeli test kalmadi.
    comparisons = []
    comparison = None

    # ---- assemble ----
    # "denendi, sonuc yok" notu yalniz GERCEKTEN sonucu olmayan yontemler
    # icin anlamli: bu kosum bir yontemi atlamis/eleyememis olsa bile onceki
    # kosumdan gecerli bir satir merge ile geldiyse not gurultudur -- dusur.
    # "denendi, sonuc yok" notu, o YONTEMIN herhangi bir varyantinin gercek
    # satiri varsa dusurulur (row_id'ler soneklendigi icin base kumeye bak).
    _have_base = {split_row_id(r)[0] for r in results}
    solver_notes = [nt for nt in solver_notes if nt["key"] not in _have_base]
    ordered = _order_rows(results)
    payload = {
        "dataset": name, "n": n, "ewt": ewt, "is_geo": "GEO" in ewt,
        "bks": bks, "sparse": sparse,
        "theta_star": round(theta_star, 2),
        "rotation_applied": rotation_applied,
        "seed_cost": round(seed_cost, 2) if seed_cost is not None else None,
        "generated": run_stamp,
        "time_budget": time_budget, "construction_max_s": constr_budget_s,
        "exact_max_s": exact_budget_s,
        "config": {k: v for k, v in cfg.items() if k != "dense"},
        "comparison": comparison,
        "comparisons": comparisons,
        "hyperparameters": _hyperparameters(cfg),
        "coords": [[round(x, 3), round(y, 3)] for x, y in coords],
        "methods": ordered,
        "best_cost": min((m["cost"] for m in ordered), default=None),
        "solver_notes": solver_notes,
    }
    _atomic_write_json(out, payload)
    _log(f"[{name}] wrote {out} ({out.stat().st_size/1024:.0f} KB)\n")
    return payload


def _run_stochastic(key, base_tour, inst, bks, cfg, record, n, ckpt_file=None):
    seeds = list(range(1, cfg["seeds"] + 1))
    init_cost = inst.tour_cost(base_tour)
    # ---- tohum-bazlı ara-kayıt: 10 tohum x ~150 s'lik ILS koşuları kabuk
    #      zaman aşımıyla bölünürse tamamlanan tohumlar ckpt dosyasından
    #      yüklenir, yalnız eksik tohumlar koşulur. Tohum girdileri (base
    #      turu, cfg hiperparametreleri) aynı koşum içinde değişmediği için
    #      bölünerek üretilen satır tek-parça satırla eşdeğerdir. ----
    done = {}
    if ckpt_file is not None and ckpt_file.exists():
        try:
            for _e in json.loads(ckpt_file.read_text(encoding="utf-8")):
                done[_e["seed"]] = _e
        except Exception:
            done = {}
    per_seed = []
    best_cost = float("inf")
    best_tour = None
    best_diag = None
    total_t = 0.0
    n_seeds = len(seeds)
    ck_entries = [done[sd] for sd in sorted(done)]
    _log(f"   >> {PRETTY[key]} starting ({n_seeds} seed{'s' if n_seeds > 1 else ''}) ...")
    n_loaded = 0
    for sd in seeds:
        e = done.get(sd)
        if e is None:
            continue
        n_loaded += 1
        per_seed.append({"seed": sd, "cost": e["cost"], "gap": e["gap"],
                         "time": e["time"]})
        total_t += e["time"]
        if e["cost"] < best_cost:
            best_cost = e["cost"]
            best_tour = e.get("tour")
            best_diag = None
    if n_loaded:
        _log(f"   >> {PRETTY[key]}: ara-kayıttan {n_loaded}/{n_seeds} tohum "
             f"yüklendi, kalan {n_seeds - n_loaded} koşuluyor")
    for idx, sd in enumerate(seeds, 1):
        if sd in done:
            continue
        t0 = time.perf_counter()
        with _quiet():
            # ILS ailesi (ils_sss/ils_hyb/ils_cls) BİREBİR aynı motoru ve
            # aynı cfg hiperparametrelerini kullanır -> tek değişken tohum.
            _, diag = lro.build_ils_result(
                inst, base_tour, max_iterations=cfg["ils_iter"],
                max_no_improve=max(20, cfg["ils_iter"] // 3),
                max_time_seconds=cfg["ils_time"], seed=sd,
                max_ls_rounds=cfg["ils_rounds"],
                neighbor_limit_2opt=cfg["nb2"], neighbor_limit_oropt=cfg["nbo"],
                k_nearest_edges_relocate=cfg["rk"],
                city_neighbor_limit_relocate=cfg["rnb"])
        dt = time.perf_counter() - t0
        total_t += dt
        # live per-seed line (matches server progress regex: indent + cost=)
        _log(f"   {PRETTY[key]+' [seed '+str(idx)+'/'+str(n_seeds)+']':34s} "
             f"cost={diag.cost:>12.0f} gap={_gap(diag.cost, bks)}% t={dt:.1f}s")
        per_seed.append({"seed": sd, "cost": round(diag.cost, 2),
                         "gap": _gap(diag.cost, bks), "time": round(dt, 2)})
        ck_entries.append({"seed": sd, "cost": round(diag.cost, 2),
                           "gap": _gap(diag.cost, bks), "time": round(dt, 2),
                           "tour": diag.tour})
        if ckpt_file is not None:
            _atomic_write_json(ckpt_file, ck_entries)
        if diag.cost < best_cost:
            best_cost = diag.cost
            best_tour = diag.tour
            best_diag = diag
    per_seed.sort(key=lambda s: s["seed"])
    costs = [s["cost"] for s in per_seed]
    mean = statistics.mean(costs)
    sample_std = statistics.stdev(costs) if len(costs) > 1 else 0.0
    ci_low, ci_high = _mean_ci95(costs)
    stoch = {
        "best": round(min(costs), 2), "mean": round(mean, 2),
        "median": round(statistics.median(costs), 2),
        "worst": round(max(costs), 2),
        # population std kept for backward-compat; "std_sample" is the unbiased
        # estimator to report in the paper, with the 95% CI of the mean.
        "std": round(statistics.pstdev(costs), 2) if len(costs) > 1 else 0.0,
        "std_sample": round(sample_std, 2),
        "sem": round(sample_std / math.sqrt(len(costs)), 2) if len(costs) > 1 else 0.0,
        "ci95_low": round(ci_low, 2), "ci95_high": round(ci_high, 2),
        "mean_gap": _gap(mean, bks), "best_gap": _gap(min(costs), bks),
        "median_gap": _gap(statistics.median(costs), bks),
        "n_seeds": len(seeds),
        "success": sum(1 for c in costs if abs(c - min(costs)) < 1e-6),
        "seeds": per_seed,
    }
    record(key, best_cost, total_t, best_diag, best_tour, initial=init_cost, stoch=stoch)
    if ckpt_file is not None:
        try:
            ckpt_file.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--methods", type=str, default=None,
                    help="comma list; default all")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=1.0,
                    help="multiplier on per-method time caps")
    ap.add_argument("--method-time", type=float, default=None,
                    help="common per-method wall-clock budget (s) for ILS "
                         "(fair anytime comparison); overrides --time-budget for it")
    ap.add_argument("--rotation-step", type=float, default=5.0,
                    help="coarse step (deg) for the strip-sweep angle scan "
                         "(rotation_strip); the best region is always refined "
                         "with a fine local sweep")
    ap.add_argument("--rotation-sparsity", type=float, default=None,
                    help="fraction (0-1) of points sampled for the theta-scan "
                         "when n>500 (n<=500 always scans the full set); "
                         f"defaults to {_ROTATION_DEFAULT_SPARSITY} if unset")
    ap.add_argument("--construction-max-s", type=float, default=None,
                    help="wall-clock cap (s) for construction methods (greedy_edge, "
                         "nearest/farthest insertion, snake/GPU band hybrids); a method "
                         f"exceeding it is skipped ('denendi, sonuç yok'); "
                         f"defaults to {_CONSTRUCTION_MAX_S:.0f}s if unset")
    ap.add_argument("--exact-time", type=float, default=None,
                    help="wall-clock cap (s) for the EXACT/gold-standard "
                         "reference solvers (LKH-3, Concorde); LKH-3 takes it as "
                         "its own TIME_LIMIT (returns its best tour so far), "
                         "Concorde as a hard process timeout (returns NO tour, "
                         "since exactness only holds if the solve finishes); "
                         f"defaults to {EXACT_TIME_DEFAULT:.0f}s if unset")
    ap.add_argument("--choices", type=str, default=None,
                    help="bu koşum için panel seçimlerini geçersiz kıl: "
                         "\"yöntem=seçim,yöntem2=seçim2\" (ör. "
                         "greedy_snake_v1=manual:0). Panelde bir yöntemi "
                         "ÇOĞALTMAK bunu kullanır: satır row_id sayesinde "
                         "kanonik satırın yanına düşer.")
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--force", action="store_true",
                    help="devam modunu kapat: diskte hazır satırı olan yöntemler "
                         "de yeniden koşulur (varsayılan: results/{ad}.json'daki "
                         "hazır yöntemler atlanır, koşum kaldığı yerden sürer)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    try:
        _overrides = parse_choice_overrides(args.choices)
    except ValueError as exc:
        ap.error(str(exc))
    set_choice_overrides(_overrides)
    if args.all:
        targets = ALL_DATASETS
    elif args.dataset:
        targets = [args.dataset]
    else:
        ap.error("provide --dataset NAME or --all")

    failed = []
    for nm in targets:
        try:
            run_dataset(nm, methods, args.time_budget, args.seeds, args.quick,
                        method_time=args.method_time, rotation_step=args.rotation_step,
                        rotation_sparsity=args.rotation_sparsity,
                        force=args.force, construction_max_s=args.construction_max_s,
                        exact_time=args.exact_time)
        except Exception as exc:  # keep batch going
            import traceback
            failed.append(nm)
            _log(f"[{nm}] FAILED: {exc}")
            traceback.print_exc(file=sys.__stderr__)
    if failed:
        # Çökme sessizce "başarılı" görünmesin: dashboard iş durumunu çıkış
        # kodundan okur -- 0 dönersek yarım/boş koşum "✓ tamam" diye görünür
        # (bck2217 vakası). Kısmi başarıda da hata kodu döneriz; başarıyla
        # yazılmış kümelerin dosyaları zaten diskte durur.
        _log(f"BAŞARISIZ kümeler: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
