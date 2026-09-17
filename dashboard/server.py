# -*- coding: utf-8 -*-
"""Greedy Snake benchmark dashboard sunucusu (stdlib only).

Serves the static dashboard + datasets.js + cached results, and runs the
benchmark pipeline (benchmark/run.py) on demand in a background subprocess
with live job status.

Run:  python dashboard/server.py            (then open http://localhost:7100/)
      python dashboard/server.py --port 8080 --no-browser
      npm run dev                           (kök package.json üzerinden aynısı)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DASH = Path(__file__).resolve().parent          # sneakpath_benchmark/dashboard/
ROOT = DASH.parent                              # sneakpath_benchmark/
sys.path.insert(0, str(ROOT))                   # runner/tsplib_engine/... için

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DATA = ROOT / "data"
DATA_VLSI = ROOT / "data_vlsi"
DATA_VLSI.mkdir(exist_ok=True)
DATA_TSPLIB = ROOT / "data_tsplib"
DATA_TSPLIB.mkdir(exist_ok=True)
CONFIG_PATH = DASH / "methods_config.json"
import vlsi_datasets as VD    # Waterloo VLSI kataloğu (name -> n, bks, durum)
import tsplib_datasets as TD  # orijinal TSPLIB95 (Heidelberg/Reinelt)

# İndirilebilir koleksiyon kaydı: anahtar -> (katalog modülü, hedef dizin).
# Tüm koleksiyonlar aynı sözleşmeyi taşır (CATALOG/BKS/url_for), indirme ve
# doğrulama kodu bu kayıt üzerinden geneldir; adlar koleksiyonlar arası
# benzersizdir (indirme uçları addan koleksiyonu kendisi bulur).
DL_COLLECTIONS = {
    "vlsi": (VD, DATA_VLSI),
    # TSPLIB95 kataloğu DÖRT alanlı (n, opt, durum, mesafe_tipi) — diğerleri üç
    # alanlı. Ortak kod `CATALOG[name][0]` (n) ve `[1]` (bks) okuduğu için
    # fazladan alan sorun çıkarmaz; `_collections()` mesafe tipini de yayar.
    "tsplib": (TD, DATA_TSPLIB),
}

ALL_DATASETS = ["kroA100", "rat783", "ali535", "gr666", "pcb3038",
                "fl3795", "fnl4461", "rl5915", "usa13509"]


def _collections():
    """Benchmark sekmesindeki koleksiyon seçicinin veri kaynağı: klasik
    TSPLIB 9'lusu + indirilebilir Waterloo koleksiyonları (VLSI / Ülke /
    Sanat; hangileri indirilmiş, hangi örneğin sonucu var)."""
    classic = []
    for nm in ALL_DATASETS:
        rp = RESULTS / f"{nm}.json"
        classic.append({"name": nm, "has_result": rp.exists()})
    out = {"classic": classic}
    for key, (cat, cdir) in DL_COLLECTIONS.items():
        items = []
        circuit = getattr(cat, "CIRCUIT", frozenset())
        for nm, meta in cat.CATALOG.items():
            n, bks, status = meta[0], meta[1], meta[2]
            row = {"name": nm, "n": n, "bks": bks, "status": status,
                   "downloaded": (cdir / f"{nm}.tsp").exists(),
                   "has_result": (RESULTS / f"{nm}.json").exists()}
            if len(meta) > 3:
                row["ewt"] = meta[3]
            # TSPLIB'in DEVRE alt kümesi (delme problemleri + programlanabilir
            # mantık dizileri): panel bunları ayrı süzebilsin diye işaretlenir.
            if nm in circuit:
                row["circuit"] = True
            # klasik data/ dizininde de duran örnekler: dosya orada, indirme
            # gerekmez (runner.dataset_path önce data/'yı bakar).
            if not row["downloaded"] and (ROOT / "data" / f"{nm}.tsp").exists():
                row["downloaded"] = True
                row["in_classic"] = True
            items.append(row)
        out[key] = items
    return out


# İstatistik paketi önbelleği: results/*.json dosyalarının (dosya sayısı,
# en yeni mtime) imzası değişmedikçe paket yeniden kurulmaz. Paket kurulumu
# 111 dosyanın tam JSON ayrıştırmasını (tur dizileri dahil, yüzlerce MB)
# gerektirdiğinden ~15-20 sn sürer; önbellek olmadan İstatistik sekmesi her
# açılışta "Sonuç yok" yer iminde bekliyor görünüyordu.
_STATS_CACHE = {"sig": None, "payload": None}


def _stats_sig():
    files = list(RESULTS.glob("*.json"))
    newest = max((p.stat().st_mtime_ns for p in files), default=0)
    return (len(files), newest)


def _stats_data():
    """İstatistik sekmesi için TÜM veri kümelerinin sonuçlarını (klasik +
    VLSI + Ülke + Sanat) tek pakette döndürür. 'tour' ve 'coords' alanları
    atılır -- istatistik hesaplarında kullanılmıyorlar ve n=200000'lik sanat
    örneklerinde tek başına megabaytlarca yer kaplayıp yükü gereksiz
    büyütüyorlar (yöntem başına cost/gap/time/history/stochastic yeter)."""
    coll_of = {nm: "classic" for nm in ALL_DATASETS}
    for key, (cat, _cdir) in DL_COLLECTIONS.items():
        for nm in cat.CATALOG:
            coll_of[nm] = key
    out = []
    sig = _stats_sig()
    if _STATS_CACHE["sig"] == sig and _STATS_CACHE["payload"] is not None:
        return _STATS_CACHE["payload"]
    for nm, collection in coll_of.items():
        rp = RESULTS / f"{nm}.json"
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # emniyet filtresi: koddan kaldırılan eski yöntemlerin (snake/rotation/
        # gpu_snake) diskte kalmış satırları istatistiğe hiç girmesin (bir
        # sonraki koşumda merge zaten kalıcı temizler; bu, eski/yedek dosyalara
        # karşı koruma)
        import runner
        # BEYAZ LISTE (2026-09-16): istatistik sekmesi satir basina yalniz
        # key/row_id/cost/gap/time/history kullanir. Eskiden yalniz "tour"
        # atiliyordu; pge/pgr satirlarinin "parts" alani (bant basina tur
        # listeleri) pakette kaliyor ve 111 kumede ~420 MB'lik JSON
        # uretiyordu -- tarayici fetch'i saatlerce suruyor, sayfa "Sonuc yok"
        # yer iminde takili kaliyordu.
        methods = [{k: m[k] for k in ("key", "row_id", "cost", "gap", "time", "history") if k in m}
                   for m in (data.get("methods") or [])
                   if not runner.is_dropped_method(m.get("key"))
                   and not runner.is_stale_angle_row(m)]
        out.append({
            "dataset": nm, "collection": collection,
            "n": data.get("n"), "bks": data.get("bks"),
            "methods": methods,
        })
    payload = {"datasets": out}
    _STATS_CACHE["sig"] = sig
    _STATS_CACHE["payload"] = payload
    return payload


def _fetch_bytes(url: str, timeout: float) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "tsplib-dashboard"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _find_catalog(name: str):
    """Adı içeren indirilebilir koleksiyonu bulur (adlar benzersiz)."""
    for key, (cat, cdir) in DL_COLLECTIONS.items():
        if name in cat.CATALOG:
            return key, cat, cdir
    return None, None, None


def _download_vlsi(name: str):
    """Waterloo'dan {name}.tsp dosyasını ait olduğu koleksiyonun dizinine
    indirir (VLSI -> data_vlsi/, Ülke -> data_world/, Sanat -> data_art/;
    koleksiyon addan bulunur, uç nokta adı tarihsel).

    Waterloo bazı büyük örnekleri (VLSI'de sra104815+, düz .tsp isteği 403
    dönenler) yalnızca .tsp.gz olarak sunuyor. Önce .tsp dener, 403/404
    alırsa .tsp.gz'yi indirip açar; ikisi de .tsp içeriği olarak diske
    yazılır, böylece runner.py hangi kaynaktan geldiğini bilmek zorunda
    kalmaz. Yarım dosya bırakmamak için önce .part'a yazıp sonra adlandırır."""
    import gzip
    import urllib.error
    _key, cat, cdir = _find_catalog(name)
    if cat is None:
        return {"error": f"indirilebilir katalogda yok: {name}", "errkey": "not_in_catalog"}
    dest = cdir / f"{name}.tsp"
    if dest.exists():
        return {"ok": True, "name": name, "already": True,
                "size": dest.stat().st_size}
    n = cat.CATALOG[name][0]
    # büyük dosyalar (yüz binlerce düğüm) indirme+açmada daha uzun sürebilir
    timeout = 300.0 if n < 100_000 else 900.0
    url = cat.url_for(name)
    part = dest.with_suffix(".part")

    def _maybe_gunzip(b: bytes) -> bytes:
        # TSPLIB95 kaynağı doğrudan .tsp.gz verir; Waterloo .tsp verir ve
        # yalnız bazı büyük örneklerde .gz'ye düşer. İkisini de aynı kod
        # yolundan geçirmek için sihirli sayıya bakılır.
        return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b

    try:
        try:
            raw = _maybe_gunzip(_fetch_bytes(url, timeout))
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
            raw = gzip.decompress(_fetch_bytes(url + ".gz", timeout))
        except Exception:
            # Heidelberg zaman zaman erişilemez oluyor -- katalog bir ayna
            # veriyorsa oradan dene (aynı dosyalar, düz .tsp).
            mirror = getattr(cat, "mirror_for", None)
            if mirror is None:
                raise
            raw = _maybe_gunzip(_fetch_bytes(mirror(name), timeout))
        body = raw.decode("utf-8", errors="replace")
        if "NODE_COORD_SECTION" not in body:
            return {"error": f"indirilen dosya .tsp değil (başlangıç: {body[:80]!r})",
                    "errkey": "not_tsp_file"}
        part.write_text(body, encoding="utf-8")
        part.replace(dest)
        return {"ok": True, "name": name, "size": dest.stat().st_size}
    except Exception as exc:
        part.unlink(missing_ok=True)
        return {"error": f"indirme hatası: {exc}", "errkey": "download_failed"}


_DLALL_LOCK = threading.Lock()
_DLALL_JOB = {"status": "idle", "done": 0, "total": 0, "current": None,
              "ok": [], "errors": []}


def _download_all_worker(collection: str = "vlsi"):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cat, _cdir = DL_COLLECTIONS.get(collection, DL_COLLECTIONS["vlsi"])
    names = list(cat.CATALOG)
    with _DLALL_LOCK:
        _DLALL_JOB.update(status="running", done=0, total=len(names),
                           current=None, ok=[], errors=[],
                           collection=collection)
    # kibarca: Waterloo'nun sunucusunu 102 istekle aynı anda boğmamak için
    # ılımlı bir eşzamanlılık sınırı
    def work(nm):
        res = _download_vlsi(nm)
        with _DLALL_LOCK:
            _DLALL_JOB["done"] += 1
            _DLALL_JOB["current"] = nm
            if "error" in res:
                _DLALL_JOB["errors"].append({"name": nm, "error": res["error"]})
            else:
                _DLALL_JOB["ok"].append(nm)
        return res
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(work, nm) for nm in names]
        for _ in as_completed(futs):
            pass
    with _DLALL_LOCK:
        _DLALL_JOB["status"] = "done"


def _download_all_start(collection: str = "vlsi"):
    if collection not in DL_COLLECTIONS:
        return {"error": f"bilinmeyen koleksiyon: {collection}", "errkey": "unknown_collection"}
    with _DLALL_LOCK:
        if _DLALL_JOB["status"] == "running":
            return {"already_running": True}
    threading.Thread(target=_download_all_worker, args=(collection,),
                     daemon=True).start()
    return {"started": True, "collection": collection}


def _download_all_status():
    with _DLALL_LOCK:
        return dict(_DLALL_JOB)


def _clear_results():
    """Bilinen TÜM veri kümelerinin sonuç dosyalarını siler (klasik + VLSI).
    Yalnızca <dataset>.json desenindeki dosyalara dokunur; results/ altındaki
    diğer çıktılar (ör. gpu_repair_ablation.json) korunur."""
    deleted = []
    known = list(ALL_DATASETS)
    for cat, _cdir in DL_COLLECTIONS.values():
        known += list(cat.CATALOG)
    for nm in known:
        rp = RESULTS / f"{nm}.json"
        if rp.exists():
            rp.unlink()
            deleted.append(nm)
    return {"ok": True, "deleted": deleted, "count": len(deleted)}


def _valid_dataset(name: str) -> bool:
    """Koşturulabilir veri kümesi: klasik 9'lu ya da İNDİRİLMİŞ bir
    koleksiyon örneği (VLSI/Ülke/Sanat)."""
    if name in ALL_DATASETS:
        return True
    _key, cat, cdir = _find_catalog(name)
    return cat is not None and (cdir / f"{name}.tsp").exists()


def _read_config_raw() -> dict:
    """methods_config.json'un ham içeriği (yoksa/bozuksa boş sözlük).
    Tek okuma kapısı: her alan için ayrı ayrı dosya açan eski kod, dosya
    yarım yazılmışken alanların bir kısmını görüp bir kısmını görmeyebiliyordu."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _deleted_methods() -> set:
    """Admin panelinden KALICI olarak silinen yöntem anahtarları.
    runner.deleted_methods() ile AYNI dosyayı okur (tek doğruluk kaynağı);
    burada ayrıca okumamızın sebebi sunucunun runner'ı bir kez import etmesi
    ve modül düzeyinde bir kopyanın bayatlayabilecek olmasıdır."""
    d = _read_config_raw().get("deleted_methods")
    return set(d) if isinstance(d, (list, tuple, set)) else set()


def _del_order(k: str) -> int:
    """Silinenleri METHOD_ORDER sırasında göstermek için anahtar; koddan da
    kalkmış (METHOD_ORDER'da olmayan) bir anahtar sona düşer."""
    import runner
    try:
        return runner.METHOD_ORDER.index(k)
    except ValueError:
        return 10 ** 6


def _purge_rows_from_results(row_ids: set) -> dict:
    """Belirli ROW_ID'lere sahip satırları results/*.json'dan siler (kopya
    silmenin disk ayağı). Yöntem bazlı temizlik için
    _purge_methods_from_results'a bakın -- ikisi de aynı yazma kuralını
    (atomik, bozuk dosyaya dokunma, best_cost'u tazele) paylaşır."""
    if not row_ids:
        return {"files": [], "rows": 0}
    return _purge_from_results(lambda m: (m.get("row_id") or m.get("key")) in row_ids,
                               lambda nt: False)


def _purge_methods_from_results(keys: set) -> dict:
    """Silinen yöntemlerin satırlarını results/*.json dosyalarından TEMİZLER.
    Silme "artık gösterme" değil "hiç olmamış gibi" demek olduğu için diskte
    kalan satırlar da gider: aksi halde istatistik sekmesi, tablo üreticisi ve
    bir sonraki koşunun merge'i onları hayalet satır olarak geri getirirdi
    (aynı tuzağa REMOVED_METHODS ayıklaması eklenirken düşülmüştü).

    Bir yöntemin VARYANT satırları (row_id = "yöntem@seçim") da aynı yönteme
    aittir ve birlikte silinir; eşleşme her zaman `key` alanı üzerindendir.
    `solver_notes` içindeki "denendi, sonuç yok" notları da temizlenir."""
    return _purge_from_results(lambda m: m.get("key") in keys,
                               lambda nt: nt.get("key") in keys)


def _purge_from_results(drop_row, drop_note) -> dict:
    """results/*.json üzerinde ortak temizlik: `drop_row(satır)` True dönen
    satırları ve `drop_note(not)` True dönen "denendi, sonuç yok" notlarını
    atar, best_cost'u tazeler, dosyayı ATOMİK yazar. Bozuk/yabancı dosyalara
    dokunmaz (yarım bir dosyayı düzeltmeye çalışmak veriyi kaybettirebilir)."""
    touched, rows = [], 0
    for rp in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue          # bozuk/yabancı dosyaya dokunma
        if not isinstance(data, dict):
            continue
        ms = data.get("methods")
        if not isinstance(ms, list):
            continue
        keep = [m for m in ms if not drop_row(m or {})]
        notes = data.get("solver_notes")
        keep_notes = ([nt for nt in notes if not drop_note(nt or {})]
                      if isinstance(notes, list) else notes)
        if len(keep) == len(ms) and (not isinstance(notes, list)
                                     or len(keep_notes) == len(notes)):
            continue
        rows += len(ms) - len(keep)
        data["methods"] = keep
        if isinstance(notes, list):
            data["solver_notes"] = keep_notes
        # "en iyi maliyet" silinen satırdan geliyor olabilir -> yeniden hesapla
        costs = [m.get("cost") for m in keep if isinstance(m.get("cost"), (int, float))]
        data["best_cost"] = min(costs) if costs else None
        tmp = rp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(rp)       # atomik: yarım dosya bırakma
        touched.append(rp.stem)
    return {"files": touched, "rows": rows}


def _write_config_field(**fields) -> None:
    """methods_config.json'a alan yazar (diğer alanlar korunur, atomik)."""
    cfg = _read_config_raw()
    cfg.update(fields)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


# ---------------------------------------------------------------------------
# YONTEM COGALTMA (2026-09-07, kullanici talebi: "bir yontemi farkli girdiler
# ile yapmak istiyorum")
# ---------------------------------------------------------------------------
# Panelde her yontemin TEK bir secimi (tohum/aci/onarim/varyant) vardir. Ayni
# yontemi ayni tabloda IKI farkli secimle gormek icin yontem COGALTILIR:
# kopya, tabanini (base) ve kendi secimini tasiyan hafif bir kayittir.
#
# Kopya YENI BIR YONTEM DEGILDIR -- runner'da yeni bir anahtar yoktur. Kosum
# sirasinda kopya, `--choices <base>=<secim>` ile taban yontemin bir kosumuna
# cevrilir; sonuc satiri row_id ("<base>@<secim>") sayesinde kanonik satirin
# YANINA duser. Boylece "ayni yontem, farkli girdi" karsilastirmasi tek
# tabloda okunur ve hicbir satir digerinin ustune yazmaz.
#
# ID: "<base>#<sayi>" -- yalniz PANEL ICI kimliktir (hangi kutunun hangi
# secimi tuttugunu bilmek icin). Diskteki satirin kimligi her zaman row_id'dir.
_CLONE_DIMENSIONS = ("seed", "angle", "pool_repair", "ge_variant")


def _dimension_of(base: str) -> str:
    """Yöntemin çoğaltmayı anlamlı kılan seçilebilir boyutu ("" = yok).
    Boyutu olmayan bir yöntemin kopyası taban satırın BİREBİR aynısı olurdu
    (aynı row_id), bu yüzden çoğaltılamaz."""
    import runner
    if base in runner.SEEDABLE:
        return "seed"
    if base in runner.ANGLE_SELECTABLE:
        return "angle"
    if base in runner.POOL_REPAIR_SELECTABLE:
        return "pool_repair"
    if base in runner.GE_VARIANT_SELECTABLE:
        return "ge_variant"
    return ""


def _dimension_choices(dim: str) -> list:
    import runner
    return {"seed": list(runner.SEED_CHOICES),
            "angle": list(runner.ANGLE_CHOICES),
            "pool_repair": list(runner.POOL_REPAIR_CHOICES),
            "ge_variant": list(runner.GE_VARIANT_CHOICES)}.get(dim, [])


def _valid_choice(dim: str, base: str, choice: str) -> bool:
    """Seçim o boyut için geçerli mi? Açı boyutunda elle girilen açı
    ("manual:-90") sabit listede olmadığı hâlde geçerlidir."""
    import runner
    if dim == "angle":
        return runner.is_angle_choice(choice)
    if dim == "seed":
        return choice in runner.SEED_CHOICES and choice != base
    return choice in _dimension_choices(dim)


def _panel_choice_of(base: str, cfg: "dict | None" = None) -> str:
    """Taban yöntemin PANELDE kayıtlı (kanonik) seçimi -- runner ile aynı
    çözümleme, yani geçersiz/eksik değer varsayılana düşer."""
    import runner
    c = cfg if cfg is not None else _read_config_raw()
    dim = _dimension_of(base)
    if dim == "seed":
        return runner.resolve_seed_choice(base, c.get("seeds") or {})
    if dim == "angle":
        return runner.resolve_angle_choice(base, c.get("angles") or {})
    if dim == "pool_repair":
        return runner.resolve_pool_repair_choice(base, c.get("pool_repairs") or {})
    if dim == "ge_variant":
        return runner.resolve_ge_variant_choice(base, c.get("ge_variants") or {})
    return ""


def _clones(cfg: "dict | None" = None) -> list:
    """Geçerli kopyalar. Tabanı silinmiş/bilinmeyen ya da boyutu kalmamış
    kayıtlar sessizce elenir (config elle düzenlenmiş olabilir)."""
    import runner
    c = cfg if cfg is not None else _read_config_raw()
    dead = _deleted_methods()
    out = []
    for it in (c.get("clones") or []):
        if not isinstance(it, dict):
            continue
        base, cid = it.get("base"), it.get("id")
        if not base or not cid or base in dead or base not in runner.METHOD_ORDER:
            continue
        dim = _dimension_of(base)
        if not dim:
            continue
        choice = it.get("choice")
        if not _valid_choice(dim, base, choice or ""):
            choice = _panel_choice_of(base, c)
        out.append({"id": cid, "base": base, "choice": choice, "dimension": dim})
    return out


#: Kopyanın girdi etiketini boyuta göre üreten TEK yer -- runner.display_name
#: ile AYNI kaynakları kullanır ki panel ile sonuç satırı aynı girdiyi aynı
#: sözcüklerle adlandırsın.
def _choice_label(dim: str, choice: str) -> str:
    import runner
    if dim == "seed":
        return runner.PRETTY.get(choice, choice)
    if dim == "angle":
        return runner.angle_label(choice)
    if dim == "pool_repair":
        return runner.POOL_REPAIR_LABELS.get(choice, choice)
    return runner.GE_VARIANT_LABELS.get(choice, choice)


def _clone_view(c: dict) -> dict:
    """Kopyanın panel görünümü: satır kimliği + PANEL gösterim adı.

    AD BİÇİMİ (2026-09-07, kullanıcı geri bildirimi):

        ⧉ <yöntem adı> · kopya N ← girdi: <seçilen girdi>

    Önceden yalnız "<yöntem> ← <girdi>" yazıyordu; girdi etiketi uzun
    olduğunda satır BAŞKA BİR YÖNTEM gibi okunuyordu ve satırın bir kopya
    olduğu adda hiç görünmüyordu. Artık üç şey de açık: aynı yöntem, kaçıncı
    kopya, ve "←"den sonrasının GİRDİ olduğu.

    ⚠ SONUÇ SATIRININ ADI BUNDAN FARKLIDIR ve bu BİLEREKtir. Benchmark
    tablosundaki satır `<taban>@<seçim>` olup adı runner.display_name'den
    gelir ("Onarım — tam VND ← Quick-Boruvka") -- çünkü o satır gerçekten
    budur: tabanın o girdiyle koşulmuş hâli. "kopya N" PANELE ait bir
    kavramdır (hangi admin satırını düzenlediğini söyler), sonuca ait
    değildir; sonuç dosyasında kopya diye ayrı bir nesne yoktur."""
    import runner
    base, ch, dim = c["base"], c["choice"], c["dimension"]
    n = c["id"].rsplit("#", 1)[-1]
    return {**c,
            "row_id": runner.row_id(base, ch),
            "name": (f"⧉ {runner.PRETTY.get(base, base)} · kopya {n} "
                     f"← girdi: {_choice_label(dim, ch)}")}


def _clone_choice_taken(base: str, choice: str, cfg: dict,
                        skip_id: str = "") -> bool:
    """Bu (taban, seçim) çifti zaten bir satır üretiyor mu? Kanonik satır ve
    diğer kopyalar aynı row_id'yi paylaşamaz -- paylaşsalardı biri diğerinin
    ÜSTÜNE yazardı (çoğaltmanın tam olarak önlemek istediği şey)."""
    if choice == _panel_choice_of(base, cfg):
        return True
    return any(c["base"] == base and c["choice"] == choice and c["id"] != skip_id
               for c in _clones(cfg))


def _add_clone(base: str) -> dict:
    """Yöntemi çoğaltır: taban yöntemin HENÜZ KULLANILMAYAN ilk seçimiyle
    yeni bir kopya oluşturur (kopya, tabanın aynısı olarak doğmaz -- öyle
    olsaydı aynı row_id'yi paylaşır, tabloda tek satır görünürdü)."""
    import runner
    if base in _deleted_methods():
        return {"error": "silinmiş bir yöntem çoğaltılamaz", "errkey": "clone_deleted"}
    if base not in runner.METHOD_ORDER:
        return {"error": f"bilinmeyen yöntem: {base}", "errkey": "unknown_method"}
    dim = _dimension_of(base)
    if not dim:
        return {"error": f"'{runner.PRETTY.get(base, base)}' çoğaltılamaz: "
                         f"seçilebilir bir girdisi (tohum/açı/onarım/varyant) yok, "
                         f"kopyası taban satırın birebir aynısı olurdu",
                         "errkey": "not_clonable"}
    cfg = _read_config_raw()
    free = next((c for c in _dimension_choices(dim)
                 if _valid_choice(dim, base, c)
                 and not _clone_choice_taken(base, c, cfg)), "")
    if not free:
        return {"error": "bu yöntemin bütün seçimleri zaten birer satır olarak var",
                "errkey": "all_choices_taken"}
    used = {c["id"] for c in _clones(cfg)}
    i = 2
    while f"{base}#{i}" in used:
        i += 1
    cid = f"{base}#{i}"
    _write_config_field(clones=(cfg.get("clones") or [])
                        + [{"id": cid, "base": base, "choice": free}])
    out = _load_config()
    out["added_clone"] = cid
    return out


def _set_clone_choice(cid: str, choice: str) -> dict:
    """Kopyanın seçimini değiştirir (panelde açılır kutudan)."""
    cfg = _read_config_raw()
    cur = next((c for c in _clones(cfg) if c["id"] == cid), None)
    if cur is None:
        return {"error": "böyle bir kopya yok", "errkey": "no_such_clone"}
    base, dim = cur["base"], cur["dimension"]
    import runner
    if dim == "angle":
        # elle girilen açı kanonikleşir ("manual" -> "manual:-90"), runner ile
        # aynı kural: aksi hâlde aynı açı iki ayrı row_id'ye yazılırdı
        choice = runner.resolve_angle_choice(base, {base: choice})
    if not _valid_choice(dim, base, choice):
        return {"error": f"geçersiz seçim: {choice}", "errkey": "invalid_choice"}
    if _clone_choice_taken(base, choice, cfg, skip_id=cid):
        return {"error": "bu seçim zaten bir satır olarak var (taban ya da "
                         "başka bir kopya) — iki satır aynı kimliği paylaşamaz",
                "errkey": "choice_taken"}
    items = [dict(c) for c in (cfg.get("clones") or []) if isinstance(c, dict)]
    for it in items:
        if it.get("id") == cid:
            it["choice"] = choice
    _write_config_field(clones=items)
    out = _load_config()
    out["updated_clone"] = cid
    return out


def _delete_clones(ids: set) -> dict:
    """Kopyaları siler ve ürettikleri satırları results/*.json'dan temizler.
    Taban yönteme DOKUNULMAZ (kanonik satır yerinde kalır)."""
    cfg = _read_config_raw()
    known = {c["id"]: c for c in _clones(cfg)}
    hit = {i for i in ids if i in known}
    if not hit:
        return {"rows": 0, "files": [], "ids": []}
    rows = {_clone_view(known[i])["row_id"] for i in hit}
    # aynı row_id'yi başka bir (kalan) kopya ya da kanonik satır üretiyorsa
    # diskteki satır SİLİNMEZ -- o satır artık ona aittir
    keep = {_clone_view(c)["row_id"] for c in _clones(cfg) if c["id"] not in hit}
    purged = _purge_rows_from_results(rows - keep)
    items = [c for c in (cfg.get("clones") or [])
             if not (isinstance(c, dict) and c.get("id") in hit)]
    _write_config_field(clones=items)
    return {**purged, "ids": sorted(hit)}


def _delete_methods(keys) -> dict:
    """Yöntemleri KALICI olarak siler: config'e yazar, o yöntemlere ait panel
    seçimlerini (tohum/açı/onarım/varyant) düşürür ve satırlarını
    results/*.json'dan temizler. TEK YÖNLÜDÜR -- geri alma yolu yoktur;
    silinen bir yöntemi geri istemek, methods_config.json'daki
    "deleted_methods" listesini elle düzenlemek demektir.

    Korumalar:
      * bilinmeyen anahtar sessizce atılır;
      * varsayılan tohum (runner.DEFAULT_SEED) SİLİNEMEZ -- silinirse tohum
        seçen her satır kaynaksız kalır, koşu kırılır. Panel bu durumda
        anlaşılır bir hata görür, yarım iş yapılmaz."""
    import runner
    # Kopya kimlikleri ("<base>#2") aynı silme akışıyla gelir: kullanıcı için
    # ikisi de "listedeki bir satırı sil"dir. Kopya silmek TABAN yöntemi
    # etkilemez, yalnız o kopyanın satırını temizler.
    clone_ids = {c["id"] for c in _clones()}
    hit_clones = {k for k in (keys or []) if k in clone_ids}
    valid = set(runner.METHOD_ORDER)
    want = {k for k in (keys or []) if k in valid}
    if not want and not hit_clones:
        return {"error": "silinecek geçerli yöntem yok", "errkey": "nothing_to_delete"}
    if not want:
        cres = _delete_clones(hit_clones)
        out = _load_config()
        out["deleted_now"] = cres["ids"]
        out["purged"] = {"files": cres["files"], "rows": cres["rows"]}
        return out
    if runner.DEFAULT_SEED in want:
        return {"error": f"'{runner.PRETTY.get(runner.DEFAULT_SEED, runner.DEFAULT_SEED)}' "
                         f"silinemez: tohum seçen bütün satırların varsayılan "
                         f"girdisi bu yöntemdir. Önce başka bir varsayılan tohum gerekir.",
                         "errkey": "default_seed_protected"}
    deleted = _deleted_methods() | want
    cfg = _read_config_raw()
    drop = lambda d: {k: v for k, v in (d or {}).items()
                      if k not in want and v not in want}
    _write_config_field(
        deleted_methods=sorted(deleted, key=_del_order),
        # görünürlük listesi de tutarlı kalsın: silinen yöntem "gizli"
        # sayılmaz, LİSTEDE YOKTUR
        hidden_methods=[k for k in (cfg.get("hidden_methods") or [])
                        if k not in want],
        seeds=drop(cfg.get("seeds")),
        angles=drop(cfg.get("angles")),
        pool_repairs=drop(cfg.get("pool_repairs")),
        ge_variants=drop(cfg.get("ge_variants")))
    purged = _purge_methods_from_results(want)
    # tabanı silinen kopyalar da gider (config'te öksüz kayıt kalmasın);
    # satırları zaten yöntem bazlı temizlikte gitti
    orphan = {c["id"] for c in _clones() if c["base"] in want}
    if orphan:
        items = [c for c in (_read_config_raw().get("clones") or [])
                 if not (isinstance(c, dict) and c.get("id") in orphan)]
        _write_config_field(clones=items)
    cres = _delete_clones(hit_clones) if hit_clones else {"files": [], "rows": 0,
                                                          "ids": []}
    out = _load_config()
    out["deleted_now"] = sorted(want, key=_del_order) + cres["ids"]
    out["purged"] = {"files": sorted(set(purged["files"]) | set(cres["files"])),
                     "rows": purged["rows"] + cres["rows"]}
    return out


def _load_config():
    """Admin-controlled method visibility (default: everything visible, i.e.
    OPT-OUT: the config file stores which methods were explicitly HIDDEN, not
    which ones are visible). This matters because the method list grows over
    time -- an opt-IN scheme would make every newly added method invisible
    until someone remembers to flip it on in the Admin tab (the exact bug this
    comment replaced: a method added to runner.METHOD_ORDER after
    methods_config.json was last saved silently never showed up, even though
    nothing had "hidden" it).
    Hidden methods are NOT deleted from results/*.json -- they just aren't
    offered to run or shown in the dashboard until re-enabled.

    Also reports which methods need a library the dashboard can't see for
    itself (GPU family needs PyTorch) so the UI can show WHY a selected-but-
    unavailable method produced no row instead of silently returning nothing.
    (LKH-3/Concorde/GNN/NGLS/LEE temizliği 2026-07-22: o bloklar kaldırıldı,
    kalan tek dış bağımlılık GPU yöntemleri için torch.)"""
    import runner
    from benchmark.method_table import TORCH_METHODS
    # KALICI SILME (2026-09-07): "deleted_methods" listesindeki anahtarlar
    # METHOD_ORDER'da olsalar bile YOK sayilir -- ne kosulur, ne listelenir,
    # ne de baska bir yontemin tohum/aci/onarim kutusunda secenek olur.
    # Gizleme (hidden_methods) ile karistirilmamali: gizli yontem geri
    # acilabilir ve satirlari diskte durur; silinen yontemin satirlari
    # results/*.json'dan da temizlenmistir (bkz. _delete_methods).
    deleted = _deleted_methods()
    all_methods = [k for k in runner.METHOD_ORDER if k not in deleted]
    hidden: set[str] = set()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if "hidden_methods" in saved:
                hidden = set(saved["hidden_methods"])
            elif "visible_methods" in saved:
                # one-time migration from the old opt-IN schema: methods that
                # were on disk at all (i.e. known when that file was saved)
                # and NOT listed as visible were explicitly hidden by the
                # admin; methods the file predates were never mentioned
                # either way and must NOT be inferred as hidden.
                hidden = set(saved.get("_known_methods", saved["visible_methods"])) \
                    - set(saved["visible_methods"])
        except Exception:
            pass
    visible = [k for k in all_methods if k not in hidden]
    unavailable = {}
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        msg = "PyTorch kurulu değil (pip install torch)"
        for k in TORCH_METHODS:
            if k not in deleted:
                unavailable[k] = msg
    # EXACT / altın standart köprüleri: ikili yoksa satır hiç üretilemez —
    # panel bunu ÖNCEDEN göstersin ki kullanıcı boş satırı "yöntem başarısız"
    # sanmasın (torch bloğuyla aynı desen).
    try:
        import external_solvers as _ES
        if _ES.find_lkh() is None:
            unavailable["lkh3"] = ("LKH-3 ikilisi bulunamadı — bin/LKH.exe koyun "
                                   "veya LKH_PATH ayarlayın")
        if _ES.find_concorde() is None:
            unavailable["concorde"] = ("Concorde ikilisi bulunamadı — bin/concorde "
                                       "koyun, CONCORDE_PATH ayarlayın (Windows'ta "
                                       "paketli Linux ikilisi için Docker gerekir)")
    except Exception as _exc:
        unavailable["lkh3"] = unavailable["concorde"] = (
            f"harici çözücü köprüsü yüklenemedi: {_exc}")
    # ---- TOHUM SEÇİMİ (2026-07-24) ----------------------------------------
    # İnşa dışı (SEEDABLE) yöntemlerin girdisi panelden seçilir. Panel her
    # SEEDABLE yöntem için bir kutu gösterir; seçim burada saklanır ve
    # runner.py koşarken okur (runner._panel_seed_config).
    saved_seeds = {}
    if CONFIG_PATH.exists():
        try:
            saved_seeds = json.loads(
                CONFIG_PATH.read_text(encoding="utf-8")).get("seeds") or {}
        except Exception:
            saved_seeds = {}
    seedable = sorted(set(runner.SEEDABLE) - deleted,
                      key=runner.METHOD_ORDER.index)
    # Silinen bir yontem baska bir satirin TOHUMU olarak secilmis olabilir;
    # o secim artik gecersizdir ve DEFAULT_SEED'e duser (yoksa panel var
    # olmayan bir anahtari gosterir, kosu ise sessizce baska sey kosar).
    # Kopyalar da tohum olabilir (2026-09-07): kopya kimligi gecerli bir
    # tohum degeridir ve `_run_plan` onu tabana + secime cevirir.
    _clone_rows = _clones()
    _clone_ids = [c["id"] for c in _clone_rows]
    _seed_ok = set(runner.SEED_CHOICES) | set(_clone_ids)
    seeds = {m: (saved_seeds.get(m)
                 if (saved_seeds.get(m) in _seed_ok
                     and saved_seeds.get(m) not in deleted)
                 else runner.DEFAULT_SEED) for m in seedable}
    # ---- AÇI SEÇİMİ (yalnız greedy_snake_v1) --------------------------------
    saved_angles = {}
    if CONFIG_PATH.exists():
        try:
            saved_angles = json.loads(
                CONFIG_PATH.read_text(encoding="utf-8")).get("angles") or {}
        except Exception:
            saved_angles = {}
    anglable = sorted(set(runner.ANGLE_SELECTABLE) - deleted,
                      key=runner.METHOD_ORDER.index)
    # runner.resolve_angle_choice: geçersiz seçim DEFAULT_ANGLE'a düşer, elle
    # girilen açı kanonikleşir ("manual" -> "manual:-90"). Doğrulamayı burada
    # tekrar yazmak, açı anahtarı biçimi için İKİNCİ bir doğruluk kaynağı
    # olurdu — panel ile koşum farklı şeyler kabul etmeye başlardı.
    angles = {m: runner.resolve_angle_choice(m, saved_angles) for m in anglable}
    # ---- HAVUZ ONARIMI SEÇİMİ (havuzlu iki satır) ------------------------
    saved_reps = {}
    if CONFIG_PATH.exists():
        try:
            saved_reps = json.loads(
                CONFIG_PATH.read_text(encoding="utf-8")).get("pool_repairs") or {}
        except Exception:
            saved_reps = {}
    repairable = sorted(set(runner.POOL_REPAIR_SELECTABLE) - deleted,
                        key=runner.METHOD_ORDER.index)
    pool_repairs = {m: (saved_reps.get(m)
                        if saved_reps.get(m) in runner.POOL_REPAIR_CHOICES
                        else runner.DEFAULT_POOL_REPAIR) for m in repairable}
    # ---- GREEDY-EDGE DUYARLILIK VARYANTI (2026-07-26) --------------------
    # "Rakip Greedy-Edge haksız güçlü" itirazının karşı-olgusal deneyi.
    # Varsayılan dışı seçim rakibi DEĞİŞTİRMEZ: kanonik satır her koşuda
    # yine üretilir, varyant onun YANINA düşer (runner._run_ge).
    saved_gev = {}
    if CONFIG_PATH.exists():
        try:
            saved_gev = json.loads(
                CONFIG_PATH.read_text(encoding="utf-8")).get("ge_variants") or {}
        except Exception:
            saved_gev = {}
    ge_variable = sorted(set(runner.GE_VARIANT_SELECTABLE) - deleted,
                         key=runner.METHOD_ORDER.index)
    ge_variants = {m: (saved_gev.get(m)
                       if saved_gev.get(m) in runner.GE_VARIANT_CHOICES
                       else runner.DEFAULT_GE_VARIANT) for m in ge_variable}
    _live = lambda keys: [k for k in keys if k not in deleted]
    return {"all_methods": all_methods, "visible_methods": visible,
            "unavailable": unavailable,
            # KALICI SILINENLER: panel bunları hiçbir listede çizmez. Alan
            # yalnız istemcinin "bu anahtar artık yok" kararını verebilmesi
            # için yayınlanır (önbellekten gelen eski bir sayfa, silinmiş bir
            # yöntemi tabloda göstermesin diye).
            "deleted_methods": sorted(deleted, key=_del_order),
            # ÇOĞALTMA: taban yöntemin farklı bir seçimle koşulan kopyaları.
            # Panel bunları tabanın altında ayrı satır olarak çizer; koşumda
            # her biri `--choices base=seçim` ile ayrı bir runner çağrısına
            # dönüşür (bkz. _run_plan).
            "clones": [_clone_view(c) for c in _clones()],
            "clonable_methods": [k for k in all_methods if _dimension_of(k)],
            # hangi yöntemler tohum seçer, seçenekler ve mevcut seçim
            "seedable_methods": seedable,
            "seed_choices": _live(runner.SEED_CHOICES) + _clone_ids,
            # açılır kutuda gruplu gösterim: yapıcı / havuz / ablasyon.
            # Ablasyonlar da girdi olabilir (zincir): repair_window ←
            # repair_vnd ← greedy_edge gibi. Döngü runner tarafında kırılır.
            "seed_groups": [
                ["Yapıcı (inşa)", _live(runner.SEED_CONSTRUCTIONS)],
                ["Havuz + toplu onarım", _live(runner.SEED_POOLS)],
                ["Ablasyon (onarım çıktısı)", _live(runner.SEED_ABLATIONS)],
                # Çoğaltılmış satırlar: bir kopyayı girdi seçmek, o kopyanın
                # koşulduğu TURDA beslenmek demektir (bkz. _run_plan).
                ["Kopyalar (çoğaltılmış satırlar)", _clone_ids],
            ],
            "default_seed": runner.DEFAULT_SEED,
            "seeds": seeds,
            # havuzlu yöntemler tohumunu KENDİ üretir (seçilemez) — panel
            # bunları kutu yerine bilgi etiketiyle gösterir
            "pool_source": {k: v for k, v in runner.POOL_SOURCE.items()
                            if k not in deleted},
            # açı seçimi (greedy_snake_v1): kutu sağda, tohum kutusuyla aynı yer.
            # default_angle DIŞINDA bir seçim v1⊂v2⊂v3 merdivenini bozar —
            # panel bunu uyarı olarak gösterir.
            "anglable_methods": anglable,
            "angle_choices": list(runner.ANGLE_CHOICES),
            "angle_labels": dict(runner.ANGLE_LABELS),
            "default_angle": runner.DEFAULT_ANGLE,
            "angles": angles,
            # ELLE GİRİLEN AÇI: "manual" seçilince panel bir derece kutusu
            # açar ve seçimi "manual:<derece>" olarak kaydeder (açı, row_id'yi
            # oluşturan anahtarın İÇİNDEDİR — farklı açılar ayrı satır).
            "manual_angle_prefix": runner.MANUAL_ANGLE_PREFIX,
            "default_manual_angle": runner.DEFAULT_MANUAL_ANGLE,
            # havuzlu satırların ONARIM katmanı: tohum kutusuyla aynı desen.
            # ablation_equivalent, seçilen onarımın ablasyon tablosundaki
            # muadilini söyler (None = muadili yok).
            "pool_repairable_methods": repairable,
            "pool_repair_choices": list(runner.POOL_REPAIR_CHOICES),
            "pool_repair_labels": dict(runner.POOL_REPAIR_LABELS),
            "pool_repair_ablation": dict(runner.POOL_REPAIR_ABLATION),
            "default_pool_repair": runner.DEFAULT_POOL_REPAIR,
            "pool_repairs": pool_repairs,
            # Greedy-Edge duyarlılık varyantı: tohum/açı kutusuyla aynı desen.
            # ge_variant_warning, varsayılan dışı her seçimin neden RAKİP
            # SATIR OLMADIĞINI söyler — panel bunu uyarı olarak gösterir.
            "ge_variant_methods": ge_variable,
            "ge_variant_choices": list(runner.GE_VARIANT_CHOICES),
            "ge_variant_labels": dict(runner.GE_VARIANT_LABELS),
            "ge_variant_warning": dict(runner.GE_VARIANT_WARNING),
            "default_ge_variant": runner.DEFAULT_GE_VARIANT,
            "ge_variants": ge_variants,
            # gösterim adları: "Yöntem ← Tohum"
            "pretty": dict(
                [(k, runner.PRETTY.get(k, k)) for k in all_methods]
                # kopya adlari da haritada olmak ZORUNDA: tohum açılır
                # kutusu adı buradan okur, yoksa ham kimlik ("repair_vnd#2")
                # görünürdü.
                + [(c["id"], _clone_view(c)["name"]) for c in _clone_rows]),
            # toplu koşuda sunucunun aynı anda koşturacağı en fazla iş sayısı
            # (fazlası kuyruğa alınır) — istemci paralel işçi havuzunu buna
            # göre boyutlandırır; TSP_MAX_JOBS ortam değişkeniyle değişir
            "max_parallel_jobs": MAX_PARALLEL_JOBS,
            "threads_per_job": _THREADS_PER_JOB}


def _save_config(visible_methods, seeds=None, angles=None, pool_repairs=None,
                 ge_variants=None):
    """Persists visibility as `hidden_methods` (opt-out) so methods added to
    runner.METHOD_ORDER after this call are visible by default, not hidden.
    `visible_methods` is the caller's desired fully-visible set; anything in
    METHOD_ORDER but NOT in it is recorded as hidden.

    `seeds`: {SEEDABLE yöntem -> inşa anahtarı}. Geçersiz anahtar sessizce
    atılır (DEFAULT_SEED'e düşer); böylece panelden gelen bozuk bir değer
    koşuyu kırmaz. Verilmezse diskteki mevcut seçim KORUNUR."""
    import runner
    deleted = _deleted_methods()
    valid = set(runner.METHOD_ORDER) - deleted
    visible = set(visible_methods) & valid
    # silinen yöntem "gizli" DEĞİLDİR, hiç yoktur: hidden listesine yazılmaz,
    # yoksa geri alındığında sebepsiz yere gizli olarak dönerdi
    hidden = [k for k in runner.METHOD_ORDER
              if k not in visible and k not in deleted]
    prev, prev_a, prev_r, prev_g = {}, {}, {}, {}
    if CONFIG_PATH.exists():
        try:
            _c = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            prev = _c.get("seeds") or {}
            prev_a = _c.get("angles") or {}
            prev_r = _c.get("pool_repairs") or {}
            prev_g = _c.get("ge_variants") or {}
        except Exception:
            prev, prev_a, prev_r, prev_g = {}, {}, {}, {}
    # KOPYA ÇAKIŞMASI: tabanın yeni seçimi, o tabanın bir kopyasının seçimiyle
    # aynıysa iki satır aynı row_id'ye düşerdi -> değişiklik UYGULANMAZ.
    _clone_taken = {}
    for _c in _clones(_read_config_raw()):
        _clone_taken.setdefault(_c["base"], set()).add(_c["choice"])
    rejected = {}

    def _ok(method, value):
        if value in _clone_taken.get(method, ()):
            rejected[method] = value
            return False
        return True

    # TOHUM: kopya kimlikleri de gecerlidir (2026-09-07). Kendi kopyasindan
    # beslenme REDDEDILIR -- kopya, tabanin baska girdiyle kosulmus halidir,
    # yani `repair_vnd <- repair_vnd#2` bir dongudur.
    _cl = {c["id"]: c for c in _clones(_read_config_raw())}
    _seed_values = set(runner.SEED_CHOICES) | set(_cl)
    merged = dict(prev)
    for m, s in (seeds or {}).items():
        if m not in runner.SEEDABLE or s not in _seed_values:
            continue
        if s in _cl and _cl[s]["base"] == m:
            rejected[m] = s          # kendi kopyasindan beslenemez
            continue
        if _ok(m, s):
            merged[m] = s
    merged_a = dict(prev_a)
    for m, a in (angles or {}).items():
        # is_angle_choice, sabit listeye EK OLARAK elle girilen açıyı
        # ("manual:-90", "manual:37.5") kabul eder; kanonikleştirme
        # resolve_angle_choice'ta yapılır (bkz. runner "ELLE GİRİLEN AÇI").
        if m in runner.ANGLE_SELECTABLE and runner.is_angle_choice(a):
            # elle girilen açı kanonikleşsin ki çakışma testi doğru olsun
            a = runner.resolve_angle_choice(m, {m: a})
            if _ok(m, a):
                merged_a[m] = a
    # havuz onarımı seçimi: geçersiz değer sessizce atılır (DEFAULT'a düşer),
    # verilmezse diskteki seçim KORUNUR — seeds/angles ile aynı sözleşme.
    merged_r = dict(prev_r)
    for m, r in (pool_repairs or {}).items():
        if (m in runner.POOL_REPAIR_SELECTABLE
                and r in runner.POOL_REPAIR_CHOICES and _ok(m, r)):
            merged_r[m] = r
    # Greedy-Edge duyarlılık varyantı: aynı sözleşme (geçersiz sessizce atılır,
    # verilmezse diskteki seçim korunur).
    merged_g = dict(prev_g)
    for m, g in (ge_variants or {}).items():
        if (m in runner.GE_VARIANT_SELECTABLE
                and g in runner.GE_VARIANT_CHOICES and _ok(m, g)):
            merged_g[m] = g
    # !! deleted_methods BU YAZIMDA MUTLAKA KORUNUR !! Panelden herhangi bir
    # görünürlük/tohum değişikliği bu fonksiyonu çağırır; alan burada
    # yazılmasaydı ilk tik değişikliği silinen yöntemlerin HEPSİNİ geri
    # getirirdi ("sonsuza kadar silinsin" sözleşmesinin kırılma noktası).
    _write_config_field(hidden_methods=hidden, seeds=merged,
                        angles=merged_a, pool_repairs=merged_r,
                        ge_variants=merged_g,
                        deleted_methods=sorted(deleted, key=_del_order))
    out = _load_config()
    if rejected:
        # panel bunu uyarı olarak gösterir: seçim GERİ ALINMADI, hiç
        # uygulanmadı -- kullanıcı önce çakışan kopyayı değiştirmeli
        out["rejected"] = rejected
    return out

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

_PROGRESS_RE = re.compile(r"^\s{3}(.+?)\s{2,}cost=")

# ---------------------------------------------------------------------------
# TOPLU KOŞU EŞZAMANLILIK SINIRI: her iş bağımsız bir runner.py alt-sürecidir.
# Sınırsız paralellik (eski davranış: istemci çekirdek-1 kadar iş başlatırdı)
# iki somut soruna yol açıyordu:
#   1. Süre ölçümleri bozulur: aynı çekirdekler için yarışan işler birbirinin
#      duvar-saati süresini şişirir -> makaledeki "Süre" sütunu güvenilmez.
#   2. GPU çakışması: birden çok süreç aynı anda CUDA'ya yüklenince VRAM
#      yetmez / rastgele hatalar oluşur (8 GB kart). (Ek koruma olarak
#      runner.py içinde süreçler-arası GPU dosya kilidi de vardır: _gpu_turn.)
# Bu semafor işleri SUNUCUDA kuyruğa alır: fazlası "queued" durumunda bekler
# ve istemci bu durumu görür. TSP_MAX_JOBS ortam değişkeniyle elle
# ayarlanabilir; varsayılan muhafazakâr: çekirdek/4 (en az 1, en çok 4).
# Ayrıca her alt-sürece BLAS/torch iş parçacığı tavanı verilir ki paralel
# işlerin TOPLAM iş parçacığı sayısı makineyi aşmasın (adil ve
# tekrarlanabilir süre ölçümü).
# ---------------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 8
# Varsayılan işçi sayısı: çekirdek/2 (en az 2, en çok 6). Eski çekirdek/4
# formülü 8 çekirdekte 2'ye düşüyordu -- kullanıcı toplu koşuda 4-6 paralel iş
# bekliyor. Süre-adaleti korunur: her işe BLAS/torch iş parçacığı tavanı
# verilir (_job_env) ve GPU bölümleri runner içi kilitle yine SIRAYLA koşar.
MAX_PARALLEL_JOBS = (int(os.environ.get("TSP_MAX_JOBS", "0"))
                     or max(2, min(6, _CPU_COUNT // 2)))
_JOB_SEMA = threading.BoundedSemaphore(MAX_PARALLEL_JOBS)
_THREADS_PER_JOB = max(2, _CPU_COUNT // MAX_PARALLEL_JOBS)


def _job_env():
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(_THREADS_PER_JOB)
    # ALT SURECIN CIKTI KODLAMASI (2026-09-07, olculdu): ebeveyn cocugun
    # stderr'ini UTF-8 olarak okuyor (`encoding="utf-8"`), ama cocuk Windows'ta
    # konsol kod sayfasini (cp1252) kullaniyordu -- Turkce her karakter ve her
    # tire panel canli log'unda U+FFFD (?) olarak goruniyordu. Iki degisken
    # cocugu UTF-8'e sabitler; iki taraf ayni kodlamayi kullanmak ZORUNDA.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _tsp_path(dataset: str):
    """Veri kumesinin .tsp yolunu DORT koleksiyonda birden arar. Onceden yalniz
    `data/` bakiliyordu; VLSI/Ulke/Sanat kumeleri gorsellestirme uclarindan
    ulasilamiyordu (FileNotFoundError). Sira klasik -> vlsi -> ulke -> sanat
    -> TSPLIB95.

    2026-09-07 DUZELTMESI: DATA_TSPLIB listede YOKTU. TSPLIB95 koleksiyonu
    (94 ornek) 2026-07-31'de eklenmis ama bu aramaya islenmemisti, yani
    orijinal TSPLIB ornekleri butun gorsellestirme uclarindan
    ERISILEMEZDI -- tam da makalenin cekirdek ailesi (delme/PCB)."""
    for base in (DATA, DATA_VLSI, DATA_TSPLIB):
        cand = base / f"{dataset}.tsp"
        if cand.exists():
            return cand
    return DATA / f"{dataset}.tsp"      # eski davranis (anlamli hata mesaji)


def _compute_snake(dataset: str, angle: float, do_best: bool, step: float):
    """Computes the ACTUAL benchmark strip tour (boustrophedon sweep, TSPLIB-
    correct distances) at a given angle, or the best-angle scan
    (runner._best_strip_angle_from_scan == the benchmark's rotation_strip
    detector). Used by the visualization tab so its sweep tour / theta-star
    EXACTLY match the benchmark. (Endpoint adı /api/snake tarihsel olarak
    korunmuştur; eski Snake-Grid inşası benchmark'tan kaldırıldı, açı artık
    strip için bulunur ve strip turuna uygulanır.)"""
    import tsplib_engine as E
    import runner
    header, coords, ewt = E.parse_tsp(_tsp_path(dataset))
    inst, _ = E.make_instance(coords, ewt)
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    if do_best:
        theta, tour, cost, _diag = runner._best_strip_angle_from_scan(
            inst, xs, ys, coarse_step=step)
    else:
        tour, cost = runner._strip_tour_at_angle(inst, xs, ys, angle)
        theta = angle
    bks = E.BKS.get(dataset)
    return {"dataset": dataset, "n": n, "theta": round(theta, 2),
            "cost": round(cost, 2), "tour": tour, "bks": bks,
            "gap": (round(100.0 * (cost - bks) / bks, 2) if bks else None)}


def _compute_greedy_snake(dataset: str, mult: float = None):
    """Gorsellestirme sekmesi icin bantli Greedy-Edge'in (makaledeki RSGE
    ailesinin iskeleti) GERCEK insasini ve IC YAPISINI dondurur (istemci
    tarafinda yaklasik bir kopyasi DEGIL).

    Neden sunucuda: sayfanin amaci yontemi ANLATMAK. Istemcide yeniden yazilan
    bir yaklasim, benchmark'ta kosan koddan sapabilir ve sayfa yanlis bir sey
    ogretir. Burada dogrudan snake_alt.snake_v1_tour cagrilir; bant yapisi
    snake_alt.LAST_BUILD'den okunur (kurucunun kendi kaydettigi ic yapi).

    Doner: theta* (seyrek strip taramasi), bant sayisi/eksen, bant sinirlari,
    dikis oncesi/sonrasi tur, ve AYNI ornekte cerceve karsilastirmasi --
    kafes-hizali strip (theta=0) vs seyrek strip (theta*) vs bantli GE vs
    tek-parca Greedy-Edge; yani makalenin insa paradoksu (Adim 4) tekil
    ornekte sayilarla okunur."""
    import time
    import tsplib_engine as E
    import snake_alt as SA
    header, coords, ewt = E.parse_tsp(_tsp_path(dataset))
    inst, _ = E.make_instance(coords, ewt)
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bks = E.BKS.get(dataset)
    gap = (lambda c: round(100.0 * (c - bks) / bks, 2) if bks else None)

    def timed(fn):
        t0 = time.perf_counter()
        tour = fn()
        dt = time.perf_counter() - t0
        return tour, inst.tour_cost(tour), round(dt, 4)

    # --- cerceve taban cizgisi: kafes-hizali duz strip (boustrophedon), theta=0 ---
    strip0, c_strip0, t_strip0 = timed(lambda: E.snake_order(xs, ys, 0))
    # --- bantli GE cekirdek yapisi (aile sozlesmesi: theta* = _strip_oracle_theta) ---
    # mult: istemciden gelen istek. None/0.05 -> cekirdek yapi (V1_CONFIG, b=2).
    # Baska bir deger verilirse BANT SAYISI olarak yorumlanir (tamsayi) ve sonuc
    # artik cekirdek DEGILDIR; bir KESIF/duyarlilik kosumudur ve cikti bunu
    # `is_v1=False` ile acikca tasir (sayfa da oyle etiketler). Makale bulgusu:
    # tek turda b>1 dikisi GE'ye yenilir (Tablo T7); bantlama paralel filo icindir.
    _mult = 0.05 if mult is None else float(mult)
    _bands = None if abs(_mult - 0.05) < 1e-9 else max(1, int(round(_mult)))
    t0 = time.perf_counter()
    v1_tour, theta = SA.snake_v1_tour(xs, ys, bands=_bands)
    t_v1 = round(time.perf_counter() - t0, 4)
    build = dict(SA.LAST_BUILD)
    c_v1 = inst.tour_cost(v1_tour)
    # --- ayni acida DUZ strip: "aci mi kazandirdi, bant-ici kurucu mu?" ---
    rx, ry = ((xs, ys) if abs(theta) < 1e-9 else E.rotate_coords(xs, ys, theta))
    stripT, c_stripT, t_stripT = timed(lambda: E.snake_order(rx, ry, 0))
    # --- literatur rakibi ---
    ge, c_ge, t_ge = timed(lambda: E.greedy_edge_tour(xs, ys, k=15))

    steps = [
        {"key": "strip0", "name": "Strip — kafes-hizalı (θ=0°)", "role": "hizalı çerçeve",
         "cost": round(c_strip0, 2), "gap": gap(c_strip0), "time": t_strip0,
         "complexity": "O(n log n)"},
        {"key": "stripT", "name": "Strip — seyrek tarama (θ★)", "role": "seyrek çerçeve",
         "cost": round(c_stripT, 2), "gap": gap(c_stripT), "time": t_stripT,
         "complexity": "O(n log n) × 37 açı adayı"},
        {"key": "v1",
         "name": ("Bantlı GE (çekirdek, b=2)" if _bands is None
                  else f"Bantlı GE keşif (b={_bands})"),
         "role": "bizim" if _bands is None else "keşif",
         # Bant basina O(m log m), k bant uzerinde toplam = O(n log(n/k)) =
         # O(n log n): bantli GE, tek-parca rakibiyle AYNI karmasiklik
         # sinifindadir (olculdu, log-log egim ~n^1.1). Bant-ici kurucu
         # hizlandirilmamis Farthest-Insertion olsaydi O(n^1.5) cikardi.
"cost": round(c_v1, 2), "gap": gap(c_v1), "time": t_v1,
         "complexity": "O(n log n) — ölçüldü n^1.08"},
        {"key": "greedy_edge", "name": "Greedy-Edge (k=15, tek parça)", "role": "rakip",
         "cost": round(c_ge, 2), "gap": gap(c_ge), "time": t_ge,
         "complexity": "O(n log n) aday listeli — ölçüldü n^1.09"},
    ]
    return {
        "dataset": dataset, "n": n, "bks": bks,
        "theta": round(theta, 2),
        "is_v1": abs(_mult - 0.05) < 1e-9,
        "reqMult": _mult,
        "k": build.get("k"), "k0": build.get("k0"), "mult": build.get("mult"),
        "axis_x": build.get("axis_x"), "n_bands": build.get("n_bands"),
        "sweep": build.get("sweep"),
        "bounds": build.get("bounds", []),
        "tour_prestitch": build.get("tour_prestitch", []),
        "tour": v1_tour,
        "strip0_tour": strip0, "stripT_tour": stripT, "ge_tour": ge,
        "steps": steps,
    }


def _compute_paradoks(dataset: str, k: int = 4, strategy: str = "kd", opt: str = "none"):
    """Gorsellestirme sekmesinin ANA gorunumu: makalenin F0 sekli
    ("Insa paradoksu ve uzamsal bolumleme rotalari") secilen veri kumesinde
    canli uretilir. Uc panel tek cagrida hesaplanir:

      (a) kafes-hizali strip (theta=0)        -- E.snake_order, benchmark'taki
                                                 `strip` satiriyla ayni kod
      (b) seyrek strip taramasi (theta*)      -- runner._best_strip_angle_from_
                                                 scan == `rotation_strip` satiri
      (c) k parcaya uzamsal bolumleme         -- parallel_ge.evaluate; her
                                                 parcada bagimsiz GE (k=8)
                                                 kapali turu, makespan ve denge

    strategy: "kd" (dengeli 2-B k-d medyan kesimi -- makalenin one cikardigi
    secim, Tablo T8) ya da "band" (1-B esit-sayili bant). Tum maliyetler
    TSPLIB kenar maliyetiyle (yuvarlama dahil) ORIJINAL koordinatlarda
    olculur; istemci yalniz cizer, hicbir seyi yaklasik olarak yeniden
    hesaplamaz."""
    import tsplib_engine as E
    import runner
    import parallel_ge as P
    if strategy not in ("kd", "band"):
        strategy = "kd"
    header, coords, ewt = E.parse_tsp(_tsp_path(dataset))
    inst, _ = E.make_instance(coords, ewt)
    n = len(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bks = E.BKS.get(dataset)
    k = max(1, min(int(k), n))

    # (a) kafes-hizali strip
    tour0 = E.snake_order(xs, ys, 0)
    c0 = inst.tour_cost(tour0)
    # (b) seyrek aci taramasi (benchmark rotation_strip ile birebir ayni)
    theta_s, tour_s, c_s, diag = runner._best_strip_angle_from_scan(
        inst, xs, ys, coarse_step=5.0)
    # (c) k parcaya bolumleme + parca-basi GE alt turlari
    res = P.evaluate(strategy, xs, ys, k, ewt)
    # --- istege bagli iyilestirme (gorsellestirme butcesi) ---
    # VND: repair motoru (panelin repair_vnd satiriyla ayni komsuluklar).
    # ILS: makale sozlesmesi geregi VND ustune tohumlanir (runner `ils`
    # satiriyla ayni motor, kucultulmus butce). UC panel de ayni katmandan
    # gecer: (a)/(b)'nin TEK turlari da iyilestirilir (cerceve yarisi,
    # makale Adim 5 -- insada seyrek ondeyken VND/ILS sonrasi hizalinin
    # onde olup olmadigi bu ornekte canli okunur); (c)'de her drone kendi
    # alt turunu bagimsiz iyilestirir, paralel duvar-saati = en uzun parca.
    rep = None
    if opt in ("vnd", "ils"):
        _imp = (lambda tours: P.repair_parts(tours, xs, ys, ewt, mode="vnd")) \
            if opt == "vnd" else \
            (lambda tours: P.ils_parts(tours, xs, ys, ewt, budget_s=1.0))
        t_ab, l_ab, _tm_ab = _imp([tour0, tour_s])
        t2, l2, tm2 = _imp(res["tours"])
        tot2 = float(sum(l2)); mk2 = float(max(l2)); mean2 = tot2 / len(l2)
        rep = {"mode": opt, "tours": t2, "lens": [round(v, 2) for v in l2],
               "makespan": round(mk2, 2), "mean": round(mean2, 2),
               "imbalance": round(mk2 / mean2, 3) if mean2 > 0 else 1.0,
               "gain": round(100.0 * (mk2 - res["makespan"]) / res["makespan"], 2)
                       if res["makespan"] else None,
               "t_seq": round(sum(tm2), 3), "t_par": round(max(tm2), 3),
               # tek-tur panelleri: [0]=(a) kafes-hizali, [1]=(b) seyrek
               "tours_ab": t_ab,
               "costs_ab": [round(float(v), 2) for v in l_ab],
               "gain_ab": [round(100.0 * (l_ab[0] - c0) / c0, 2) if c0 else None,
                           round(100.0 * (l_ab[1] - c_s) / c_s, 2) if c_s else None]}
    gap = (lambda c: round(100.0 * (c - bks) / bks, 2) if bks else None)
    return {
        "dataset": dataset, "n": n, "bks": bks,
        "theta_s": round(float(theta_s), 2),
        "scan_time": round(float(diag.get("search_time", 0.0)), 3),
        "strip0": {"tour": tour0, "cost": round(c0, 2), "gap": gap(c0)},
        "stripS": {"tour": tour_s, "cost": round(c_s, 2), "gap": gap(c_s),
                   "gain": round(100.0 * (c_s - c0) / c0, 2) if c0 else None},
        "part": {"strategy": strategy, "k": res["k"],
                 "tours": res["tours"], "lens": res["lens"],
                 "sizes": res["sizes"],
                 "makespan": round(res["makespan"], 2),
                 "mean": round(res["mean"], 2),
                 "imbalance": round(res["imbalance"], 3),
                 "t_part": round(res["t_part"], 4),
                 "t_seq": round(res["t_seq"], 4),
                 "t_par": round(res["t_par"], 4)},
        "rep": rep,
        # Koordinatlar da sunucudan: tur indeksleri BU diziye goredir.
        # datasets.js yalniz klasik 9'luyu gomuyor ve GEO orneklerini ondan
        # farkli bir forma (ondalik derece, x/y takasli) ceviriyordu --
        # noktalar sunucunun kendi koordinatlarindan cizilince hem VLSI/
        # TSPLIB95 kumeleri secilebilir oldu hem de ali535/gr666 gibi GEO
        # orneklerinde cerceve/bolme geometrisi benchmark ile birebir ortusur.
        "coords": [c for pt in coords for c in pt],
        "ewt": str(ewt),
    }


def _kill_tree(pid: int):
    """Alt-süreci ÇOCUKLARIYLA birlikte öldürür. Koşu süreci kendi alt-süreçlerini
    (BLAS/torch işçileri vb.) açabildiğinden Windows'ta taskkill /T (ağaç)
    şarttır; yalnız üst PID öldürülürse çocuklar yetim kalıp CPU yemeye devam
    eder."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 15)  # SIGTERM
    except Exception:
        pass


def _stop_all_jobs():
    """DURDUR: koşan işlerin süreç ağaçları öldürülür, kuyruktakiler iptal
    işaretlenir (semaforu alınca başlamadan 'cancelled' olurlar). Yarım kalan
    veri kümesi dosyaya YAZILMAZ (runner sonucu koşum sonunda yazar) -> bir
    sonraki koşum devam modunda o kümenin eksiklerini yeniden hesaplar."""
    killed = cancelled = 0
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    for job in jobs:
        if job.get("status") in ("starting", "queued", "running"):
            job["cancel"] = True
            proc = job.get("proc")
            if job.get("status") == "running" and proc is not None and proc.poll() is None:
                _kill_tree(proc.pid)
                killed += 1
            else:
                cancelled += 1
    return {"killed": killed, "cancelled_queued": cancelled}


def _run_plan(selection: list) -> list:
    """Panelden seçilen yöntem/kopya listesini KOŞUM ADIMLARINA böler.

    Dönen değer: [(yöntem_anahtarları, {yöntem: seçim}), ...] -- her adım bir
    runner çağrısıdır.

    Neden birden çok adım gerekebiliyor: runner bir koşumda her yönteme TEK
    seçim uygular (seçim, `--choices` ile geçersiz kılınan panel yapılandırma-
    sından gelir). Aynı yöntemin iki kopyası aynı koşumda farklı seçimlerle
    koşamaz; bu yüzden kopyalar turlara dağıtılır: 1. tur her tabanın ilk
    kopyasını, 2. tur ikincisini... alır. Farklı tabanların kopyaları AYNI
    turda birleşir (geçersiz kılma sözlüğü çok anahtarlıdır), yani "5 yöntemi
    birer kez çoğalt" tek ek koşum eder, beş değil.

    Kanonik (kopyasız) seçimler her zaman ilk adımdadır -- panel seçimleri
    zaten diskte olduğu için o adım geçersiz kılma İÇERMEZ.

    KOPYA-TOHUMLU SATIRLAR (2026-09-07). Tohumu bir KOPYA olan yöntem, o
    kopyanın koşulduğu ADIMDA koşmak zorundadır -- çünkü kopyanın satırı
    ancak `--choices` geçersiz kılması geçerliyken vardır. O adım iki
    geçersiz kılma taşır:

        <taban>  = <kopyanın seçimi>    -> tabanın satırı KOPYA olur
        <yöntem> = <taban>              -> yöntemin tohumu o satır olur

    İkincisi şart: `seeds` dosyasında "repair_vnd#2" yazsa runner onu
    tanımaz (SEED_CHOICES'te yok) ve sessizce DEFAULT_SEED'e düşerdi.
    Böylece kopya kimliği runner'a HİÇ gitmez; çeviri burada yapılır.

    Tohum olarak istenen kopya panelden SEÇİLMEMİŞ olsa bile o adıma
    eklenir -- aksi halde beslenecek satır hiç üretilmezdi."""
    clones = {c["id"]: c for c in _clones()}
    dead = _deleted_methods()
    seed_of = (_read_config_raw().get("seeds") or {})
    plain, picked = [], []
    seeded = {}                      # kopya kimligi -> [o kopyadan beslenenler]
    for k in selection:
        if k in clones:
            picked.append(clones[k])
            continue
        if k in dead:
            continue
        _sd = seed_of.get(k)
        if _sd in clones and clones[_sd]["base"] != k:
            seeded.setdefault(_sd, []).append(k)
        else:
            plain.append(k)
    steps = []
    if plain:
        steps.append((plain, {}))
    by_base = {}
    for c in picked:
        by_base.setdefault(c["base"], []).append(c)
    # Tohum olarak istenen ama SECILMEMIS kopyalari da turlara ekle: onlarin
    # satiri uretilmezse beslenecek bir sey olmaz.
    for cid in seeded:
        c = clones[cid]
        lst = by_base.setdefault(c["base"], [])
        if not any(x["id"] == cid for x in lst):
            lst.append(c)
    while any(by_base.values()):
        methods, ov = [], {}
        for base, lst in by_base.items():
            if not lst:
                continue
            c = lst.pop(0)
            methods.append(base)
            ov[base] = c["choice"]
            for m in seeded.get(c["id"], ()):
                methods.append(m)
                ov[m] = base         # tohum = tabanin (artik kopya olan) satiri
        steps.append((methods, ov))
    return steps


def _run_job(job_id: str, steps: list, common: list):
    """İşin adımlarını SIRAYLA koşar (bkz. _run_plan). Adımlar tek bir iş
    kimliği altındadır: panel için "bu veri kümesini koş" tek bir iştir,
    kaç runner çağrısına bölündüğü bir uygulama ayrıntısıdır."""
    job = _JOBS[job_id]
    cmds = []
    for methods, ov in steps:
        a = list(common) + ["--methods", ",".join(methods)]
        if ov:
            a += ["--choices", ",".join(f"{k}={v}" for k, v in sorted(ov.items()))]
        cmds.append([sys.executable, str(ROOT / "benchmark" / "run.py")] + a)
    job["cmd"] = "  &&  ".join(" ".join(c) for c in cmds)
    job["steps"] = len(cmds)
    # kuyruk: eşzamanlılık sınırı doluysa iş burada "queued" olarak bekler
    # (istemci /api/job üzerinden bu durumu görür ve gösterir)
    job["status"] = "queued"
    with _JOB_SEMA:
        # Durdur'a basıldıysa kuyruktaki iş hiç başlatılmadan iptal edilir
        if job.get("cancel"):
            job["status"] = "cancelled"
            job["finished"] = time.time()
            return
        job["status"] = "running"
        try:
            for i, cmd in enumerate(cmds, 1):
                if job.get("cancel"):
                    break
                job["step"] = i
                proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                    bufsize=1, env=_job_env(),
                )
                job["pid"] = proc.pid
                job["proc"] = proc
                if job.get("cancel"):  # Durdur ile Popen arasındaki yarış penceresi
                    _kill_tree(proc.pid)
                for line in proc.stderr:
                    line = line.rstrip()
                    if not line:
                        continue
                    job["log"].append(line)
                    del job["log"][:-200]  # keep tail
                    m = _PROGRESS_RE.match(line)
                    if m:
                        job["current"] = m.group(1).strip()
                        job["done_methods"].append(m.group(1).strip())
                proc.wait()
                job["returncode"] = proc.returncode
                # bir adım çökerse SONRAKİ adımlar koşmaz: yarım bir koşumu
                # "tamam" diye göstermek, eksik satırı sessizce gizlerdi
                if proc.returncode != 0:
                    break
            if job.get("cancel"):
                job["status"] = "cancelled"
            else:
                job["status"] = "done" if job.get("returncode") == 0 else "error"
        except Exception as exc:
            job["status"] = "error"
            job["log"].append(f"server exception: {exc}")
    job["finished"] = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    # ---- helpers ----
    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        # ---- ONBELLEKLEME KAPALI (2026-09-07, olculdu) --------------------
        # Sunucu hicbir onbellek basligi gondermiyordu: ne Cache-Control, ne
        # ETag, ne Last-Modified. HTTP'de bu "onbelleklenemez" DEMEK DEGILDIR
        # -- tarayici SEZGISEL onbellekleme uygular ve 200 yanitini yeniden
        # dogrulamadan eski haliyle servis edebilir.
        #
        # Somut belirti (kullanici bildirdi): panele yeni eklenen yontemlerde
        # "aci <-" kutusu gorunmuyordu. renderAdmin() asgari bir DOM
        # taklidiyle kosturuldu ve kutularin 8/8 URETILDIGI dogrulandi --
        # yani kod dogruydu, tarayici ESKI index.html'i gosteriyordu.
        #
        # Bu panel bir gelistirme araci: HTML her degisiklikte yeniden
        # yazilir ve API yanitlari (ozellikle /api/config) canli durumu
        # tasir. Bayat bir /api/config, silinmis bir yontemi geri getirmis
        # ya da yeni bir yontemi yok saymis gibi gorunur. O yuzden HICBIR
        # yanit onbelleklenmez.
        self.send_header("Cache-Control",
                         "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")      # HTTP/1.0 vekilleri
        self.send_header("Expires", "0")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype):
        if not path.exists():
            return self._send(404, {"error": "not found"})
        data = path.read_bytes()
        self._send(200, data, ctype)

    # ---- routes ----
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(DASH / "index.html", "text/html; charset=utf-8")
        if p == "/datasets.js":
            return self._file(DASH / "datasets.js", "application/javascript; charset=utf-8")
        if p == "/api/datasets":
            out = []
            for nm in ALL_DATASETS:
                rp = RESULTS / f"{nm}.json"
                out.append({"name": nm, "has_result": rp.exists(),
                            "mtime": rp.stat().st_mtime if rp.exists() else None})
            return self._send(200, out)
        if p == "/api/collections":
            return self._send(200, _collections())
        if p == "/api/stats_data":
            return self._send(200, _stats_data())
        if p == "/api/vlsi/download_all/status":
            return self._send(200, _download_all_status())
        if p.startswith("/api/results/"):
            nm = p.rsplit("/", 1)[-1]
            rp = RESULTS / f"{nm}.json"
            # ESKİ AÇI SATIRLARI (2026-09-07): RGGE/RSGE artık YALNIZ panelde
            # seçili açıda tek satır üretir. Henüz yeniden koşulmamış sonuç
            # dosyalarında eski üç-satırlı sözleşmenin "@grid_theta / @zero /
            # @rotation_strip" satırları duruyor ve tabloda aynı yöntemi 2-3
            # kez, farklı açılarda gösterirdi. Böyle satır VARSA süzülmüş bir
            # kopya gönderilir; yoksa dosya eskisi gibi doğrudan akıtılır
            # (büyük sonuç dosyalarında gereksiz kod-çöz/yeniden-kodla yok).
            # Diskteki dosya bir sonraki koşumda kalıcı olarak temizlenir.
            try:
                import runner
                _d = json.loads(rp.read_text(encoding="utf-8"))
                _ms = _d.get("methods") or []
                _keep = [m for m in _ms if not runner.is_stale_angle_row(m)]
                if len(_keep) != len(_ms):
                    _d["methods"] = _keep
                    return self._send(200, _d,
                                      "application/json; charset=utf-8")
            except Exception:
                pass                     # bozuk/eksik dosya -> eski yol
            return self._file(rp, "application/json; charset=utf-8")
        if p.startswith("/api/have/"):
            # Hafif "neler hazır?" sorgusu: results/{ad}.json içinde TURU olan
            # yöntemler. Toplu koşuda istemci bununla, istenen tüm yöntemleri
            # hazır örnekleri İŞ AÇMADAN (kuyruğa sokmadan) atlar.
            #
            # !! KRİTİK (2026-07-25 düzeltmesi) !!
            # Karşılaştırma YÖNTEM anahtarıyla DEĞİL, ROW_ID ile yapılır.
            # Aynı yöntem farklı panel seçimleriyle (girdi / açı / havuz
            # onarımı) koşulabildiği için, "ge_pool_repair diskte var" demek
            # "SEÇTİĞİN onarımla var" demek değildir. Eski kod yöntem
            # anahtarına bakıyordu, dolayısıyla panelden yeni bir onarım
            # seçip koştuğunda istemci "zaten yapılmış" deyip runner'ı HİÇ
            # ÇAĞIRMIYORDU -- runner'ın row_id farkındalığı devreye bile
            # giremiyordu. Artık burada da mevcut seçimin row_id'si aranır:
            # seçim değiştiyse satır YOKTUR ve yöntem koşulur (üstüne değil,
            # YANINA yeni bir satır olarak).
            nm = p.rsplit("/", 1)[-1]
            rp = RESULTS / f"{nm}.json"
            if not rp.exists():
                return self._send(200, {"dataset": nm, "methods": []})
            try:
                import runner
                d = json.loads(rp.read_text(encoding="utf-8"))
                have_ids = {(m.get("row_id") or m["key"])
                            for m in d.get("methods", [])
                            if m.get("tour")
                            and not runner.is_dropped_method(m.get("key"))
                            and not runner.is_stale_angle_row(m)}
                # panelin ŞU ANKİ seçimleri (runner ile aynı çözümleme)
                _cfg = _load_config()
                _sel = {}
                _sel.update(_cfg.get("seeds") or {})
                _sel.update(_cfg.get("angles") or {})
                _sel.update(_cfg.get("pool_repairs") or {})
                _sel.update(_cfg.get("ge_variants") or {})
                # RGGE/RSGE: row_id açıyı taşımaz (runner.ANGLE_PANEL_CANONICAL),
                # o yüzden satırın `angle_key`i de panelin ŞU ANKİ seçimiyle
                # aynı olmalı; yoksa açı değiştirildiğinde eski açıda koşulmuş
                # satır "hazır" sayılır ve yöntem hiç koşulmaz.
                _ang_of = {(m.get("row_id") or m["key"]): m.get("angle_key")
                           for m in d.get("methods", []) if m.get("tour")}

                def _hazir(k):
                    _rid = runner.row_id(k, _sel.get(k))
                    if _rid not in have_ids:
                        return False
                    if k in getattr(runner, "ANGLE_PANEL_CANONICAL", ()):
                        _want = runner.resolve_angle_choice(k, _cfg.get("angles") or {})
                        if _ang_of.get(_rid) != _want:
                            return False
                    return True

                keys = [k for k in runner.live_method_order() if _hazir(k)]
                # KOPYALAR: kimlikleri panel içidir, diskteki karşılıkları
                # row_id'dir -- toplu koşuda "zaten hazır" atlaması kopyalar
                # için de çalışsın diye burada kendi row_id'leriyle aranırlar.
                keys += [c["id"] for c in _clones()
                         if _clone_view(c)["row_id"] in have_ids]
            except Exception as exc:
                # bozuk/yarım dosya: 'hazır değil' say -> istemci işi normal
                # açar, runner devam modu ne eksikse onu koşar (güvenli taraf)
                return self._send(200, {"dataset": nm, "methods": [],
                                        "error": f"{type(exc).__name__}: {exc}"})
            return self._send(200, {"dataset": nm, "methods": keys})
        if p == "/api/snake":
            q = parse_qs(u.query)
            dataset = (q.get("dataset") or [""])[0]
            if dataset not in ALL_DATASETS:
                return self._send(400, {"error": "invalid dataset"})
            do_best = (q.get("best") or ["0"])[0] in ("1", "true")
            try:
                angle = float((q.get("angle") or ["0"])[0])
                step = float((q.get("step") or ["5"])[0])
            except ValueError:
                angle, step = 0.0, 5.0
            try:
                return self._send(200, _compute_snake(dataset, angle, do_best, step))
            except Exception as exc:
                return self._send(500, {"error": str(exc)})
        if p == "/api/greedy_snake":
            q = parse_qs(u.query)
            dataset = (q.get("dataset") or [""])[0]
            if dataset not in ALL_DATASETS:
                return self._send(400, {"error": "invalid dataset"})
            try:
                _m = (q.get("mult") or [""])[0]
                return self._send(200, _compute_greedy_snake(
                    dataset, float(_m) if _m else None))
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        if p == "/api/paradoks":
            q = parse_qs(u.query)
            dataset = (q.get("dataset") or [""])[0]
            if not _valid_dataset(dataset):
                return self._send(400, {"error": "invalid dataset"})
            try:
                k = int(float((q.get("k") or ["4"])[0]))
            except ValueError:
                k = 4
            strategy = (q.get("strategy") or ["kd"])[0]
            opt = (q.get("opt") or ["none"])[0]
            if opt not in ("none", "vnd", "ils"):
                opt = "none"
            try:
                return self._send(200, _compute_paradoks(dataset, k, strategy, opt))
            except Exception as exc:
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        if p == "/api/config":
            return self._send(200, _load_config())
        if p.startswith("/api/job/"):
            jid = p.rsplit("/", 1)[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, {
                "id": jid, "status": job["status"], "current": job.get("current"),
                "done_methods": job.get("done_methods", []),
                "dataset": job.get("dataset"), "cmd": job.get("cmd"),
                "log": job["log"][-12:], "returncode": job.get("returncode"),
                # çoğaltma koşumları birden çok adıma bölünür (bkz. _run_plan)
                "step": job.get("step"), "steps": job.get("steps"),
            })
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/config":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body or b"{}")
            except Exception:
                req = {}
            return self._send(200, _save_config(req.get("visible_methods") or [],
                                                req.get("seeds"),
                                                req.get("angles"),
                                                req.get("pool_repairs"),
                                                req.get("ge_variants")))
        if u.path == "/api/vlsi/download":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body or b"{}")
            except Exception:
                req = {}
            res = _download_vlsi(str(req.get("name") or ""))
            return self._send(200 if "error" not in res else 400, res)
        if u.path == "/api/vlsi/download_all":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body or b"{}")
            except Exception:
                req = {}
            res = _download_all_start(str(req.get("collection") or "vlsi"))
            return self._send(200 if "error" not in res else 400, res)
        if u.path in ("/api/methods/clone", "/api/methods/clone_choice"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body or b"{}")
            except Exception:
                req = {}
            if u.path.endswith("/clone"):
                res = _add_clone(str(req.get("base") or ""))
            else:
                res = _set_clone_choice(str(req.get("id") or ""),
                                        str(req.get("choice") or ""))
            return self._send(400 if "error" in res else 200, res)
        if u.path == "/api/methods/delete":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                req = json.loads(body or b"{}")
            except Exception:
                req = {}
            ks = req.get("methods") or []
            if not isinstance(ks, list):
                return self._send(400, {"error": "methods bir liste olmalı", "errkey": "methods_not_list"})
            res = _delete_methods(ks)
            return self._send(400 if "error" in res else 200, res)
        if u.path == "/api/clear_results":
            return self._send(200, _clear_results())
        if u.path == "/api/stop":
            return self._send(200, _stop_all_jobs())
        if u.path != "/api/run":
            return self._send(404, {"error": "unknown route"})
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        dataset = req.get("dataset")
        if not _valid_dataset(dataset):
            return self._send(400, {"error": "invalid dataset (koleksiyon örneği ise önce indirilmeli)", "errkey": "invalid_dataset"})
        args = ["--sets", dataset, "--no-tables"]
        methods = req.get("methods")
        if methods:
            # eski/kaydedilmiş bir seçim silinmiş yöntem taşıyabilir; runner
            # de eler ama boş kalan listeyle iş açmanın anlamı yok
            # (kopya kimlikleri METHOD_ORDER'da değildir, silinmiş de
            #  sayılmazlar -- kendi geçerlilikleri _run_plan'da denetlenir)
            _dead = _deleted_methods()
            methods = [m for m in methods if m not in _dead]
            if not methods:
                return self._send(400, {"error": "seçilen yöntemlerin tümü "
                                                 "kalıcı olarak silinmiş", "errkey": "all_methods_deleted"})
        if not methods:
            # istemci seçim göndermediyse (ya da hiç yöntem işaretlenmediyse)
            # Admin panelinde GÖRÜNÜR bırakılan yöntemler koşulur —
            # dashboard/methods_config.json tek doğruluk kaynağıdır.
            methods = _load_config()["visible_methods"]
        steps = _run_plan(methods)
        if not steps:
            return self._send(400, {"error": "koşulacak yöntem yok", "errkey": "no_methods_to_run"})
        if req.get("time_budget"):
            args += ["--budget", str(req["time_budget"])]
        if req.get("seeds"):
            args += ["--seeds", str(int(req["seeds"]))]
        if req.get("construction_max_s"):
            args += ["--construction-max-s", str(float(req["construction_max_s"]))]
        if req.get("exact_time"):
            # exact/altın standart çözücü süre sınırı (LKH-3 / Concorde);
            # kurucuların "İnşa üst süre sınırı" kutusuyla aynı rolde
            args += ["--exact-time", str(float(req["exact_time"]))]
        if req.get("quick"):
            args += ["--quick"]
        if req.get("force"):
            # devam modunu kapat: diskte hazır yöntemler de yeniden koşulsun
            args += ["--force"]

        jid = uuid.uuid4().hex[:10]
        with _JOBS_LOCK:
            _JOBS[jid] = {"status": "starting", "log": [], "current": None,
                          "done_methods": [], "dataset": dataset, "started": time.time()}
        threading.Thread(target=_run_job, args=(jid, steps, args),
                         daemon=True).start()
        return self._send(200, {"id": jid, "dataset": dataset})


class _V6Server(ThreadingHTTPServer):
    """IPv6 ikizi: Windows 11'de tarayici "localhost" adresini once ::1'e
    cozer. Sunucu yalnizca 127.0.0.1'i dinlerse tarayici "Sunucuya
    ulasilamadi" verir. Bu sinifla ::1 icin ikinci bir dinleyici acilir."""
    address_family = socket.AF_INET6

    def server_bind(self):
        try:                                     # yalniz kendi ailesini dinle
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        except (AttributeError, OSError):
            pass
        return super().server_bind()


def _bind_targets(host: str):
    """(sinif, adres) ciftleri. "localhost"/varsayilan icin HEM 127.0.0.1 HEM
    ::1 dinlenir -- ikisi de yalnizca yerel dongu, disariya acilmaz."""
    h = (host or "").strip().lower()
    if h in ("", "localhost", "loopback"):
        return [(ThreadingHTTPServer, "127.0.0.1"), (_V6Server, "::1")]
    if ":" in h:                                 # cikartilmis IPv6 adresi
        return [(_V6Server, h)]
    return [(ThreadingHTTPServer, h)]


def main():
    ap = argparse.ArgumentParser(description="Greedy Snake benchmark dashboard sunucusu")
    ap.add_argument("--host", default="localhost",
                    help="dinlenecek adres (varsayılan localhost = 127.0.0.1 + ::1)")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("TSP_PORT", "7100")),
                    help="dinlenecek port (varsayılan: TSP_PORT ya da 7100)")
    ap.add_argument("--no-browser", action="store_true",
                    help="başlangıçta tarayıcıyı otomatik açma")
    cli = ap.parse_args()

    # Windows'ta SO_REUSEADDR (HTTPServer varsayilani) AYNI porta ikinci bir
    # sunucunun daha baglanmasina izin verir: istekler kopyalardan birine
    # (cogunlukla ESKI kodla calisana) gider ve panel "yeni yontemler yok /
    # admin tiki geri kaliyor" gibi hayalet hatalar uretir (2026-07-13'te
    # yasandi: 4 kopya ayni anda 8765'i dinliyordu). Kapatiyoruz -- port
    # doluysa sessizce ust uste binmek yerine acik bir mesajla cikilir.
    ThreadingHTTPServer.allow_reuse_address = False
    _V6Server.allow_reuse_address = False

    servers = []
    for cls, addr in _bind_targets(cli.host):
        try:
            servers.append(cls((addr, cli.port), Handler))
        except OSError as exc:
            if not servers:
                # ilk (birincil) adres tutulamadi -> port gercekten dolu
                print(f"HATA: {cli.port} portu zaten kullanimda ({addr}) -- "
                      f"baska bir server.py kopyasi acik.")
                print("Ya o kopyayi kapatin ya da tarayicidan mevcut paneli kullanin:")
                print(f"  http://localhost:{cli.port}/")
                print("Kapatmak icin: SUNUCU_RESET.bat")
                return 1
            # ikincil adres (cogunlukla IPv6 yok) -- uyar, devam et
            print(f"UYARI: {addr} dinlenemedi ({exc}); yalniz "
                  f"{servers[0].server_address[0]} uzerinden erisilebilir.")

    url = f"http://localhost:{cli.port}/"
    listening = ", ".join(f"{s.server_address[0]}:{cli.port}" for s in servers)
    print(f"Greedy Snake benchmark dashboard running at {url}  (dinlenen: {listening})")
    print("Press Ctrl+C to stop.")
    if not cli.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # Birincil sunucu bu is parcaciginda, varsa ikizi arka planda kosar.
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
        for s in servers:
            s.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
