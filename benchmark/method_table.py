# -*- coding: utf-8 -*-
"""benchmark/method_table.py -- Makale yöntem tablosu ve sonuç tablosu üreteci.

İki çıktı ailesi üretir (results/tables/ altına):

1. YÖNTEM TABLOSU (makale "Yöntemler" bölümü): runner'daki KAYITLI her yöntem
   için anahtar, görünen ad, aşama (kurucu/onarım/optimizasyon/meta-sezgisel/
   rakip), karmaşıklık sınıfı, GPU evet/hayır, stokastik evet/hayır ve
   makaledeki rol (önerilen/ablasyon/rakip/referans).
     -> yontem_tablosu.md / .csv / .tex

2. SONUÇ TABLOSU (Tablo 1/2 formatı): results/*.json dosyalarından set ×
   yöntem gap% (+ duvar-saati süre) matrisi; stokastik yöntemlerde
   ortalama±std (P1-6); özet satırları: ortalama gap, ortalama süre,
   kazanılan set sayısı.
     -> sonuc_tablosu.md / .csv / .tex

Kullanım:
    python -m benchmark.method_table            # her iki tabloyu da yaz
    python benchmark/method_table.py            # aynı (doğrudan çalıştırma)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import runner as R  # noqa: E402  (METHOD_ORDER / STAGE / PRETTY / FAMILY / STOCHASTIC)

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

# ---------------------------------------------------------------------------
#  Yöntem meta verisi (makale kapsamı; runner kayıtlarıyla birebir)
# ---------------------------------------------------------------------------
# aşama etiketi çevirisi (STAGE -> makaledeki Türkçe aşama adı)
STAGE_LABEL = {
    "construction": "Kurucu",
    "repair": "Onarım",
    "optimization": "Optimizasyon",
    "metaheuristic": "Meta-sezgisel",
    "competitor": "Rakip",
    "reference": "Referans (exact / altın standart)",
}

# "reference" aşamasındaki satırlar (LKH-3, Concorde) KIYAS TABANIDIR, rakip
# değildir: kazanma sayımına GİRMEZLER. Aksi halde altın standart her sette
# 1. çıkar ve "Kazanılan" sütunu anlamını yitirir (aynı kök neden
# snake_best_insertion_band'in kaldırılmasına yol açmıştı). Gap sütunlarında
# görünmeye devam ederler -- okuyucu "kapatılabilir boşluk" ölçüsünü orada okur.
NON_COMPETING_STAGES = {"reference"}

# 2026-07-24 (kullanıcı kararı): bu çalışmada GPU yöntemi YOKTUR. Toplu
# (batched) onarım katmanı CPU'da koşar (gpu_snake.repair_tours device="cpu").
# Havuzlu iki satır torch'u yalnızca vektörize delta hesabı için kullanır —
# CUDA gerektirmez, ama torch paketi kurulu olmalıdır.
GPU_METHODS: set[str] = set()
# 2026-07-31: liste 10 adayli havuz satirlariyla birlikte daraldi
# (greedy_snake_v3_repair / greedy_snake_v4_repair / ge_pool_repair
# kaldirildi, bkz. runner.REMOVED_METHODS (f)). Torch ihtiyaci HAVUZ
# ONARIMINDAN gelir, havuz GENISLIGINDEN degil -- bu yuzden kalan bes
# *_repair satirinin tamami listeye girer.
TORCH_METHODS: set[str] = set()   # 2026-09-07: havuz onarimi satirlari kaldirildi
GPU_WITH_CPU_FALLBACK: set[str] = set()

# karmaşıklık sınıfı (asimptotik / bütçe notu)
COMPLEXITY = {
    "nn": "O(n log n) (izgara k-NN)",
    "strip": "O(n log n)",
    "hilbert": "O(n log n)",
    "morton": "O(n log n)",
    "nearest_insertion": "O(n²)",
    "farthest_insertion": "O(n²)",
    "greedy_edge": "O(n log n) aday-listeli",
    "rotation_strip": "O(A·n log n) tarama",
    "rotation_hist": "O(n log n) + dar pencere",
    "rotation_mean": "O(n log n) + dar pencere",
    # RSGE: bant başına O(m log m), b bant üzerinde toplam O(n log(n/b))
    # = O(n log n) -- yani `greedy_edge` ile AYNI sınıf (ölçüldü: bant-hibrit
    # gövdesinin süre-n log-log eğimi n^1.10, tek parça greedy_edge n^1.09;
    # bkz. snake_alt.snake_greedy_band_tour karmaşıklık düzeltmesi).
    # DÖRDÜ DE aynı ifadeyi taşır: kurallar arasında değişen tek şey b'nin
    # DEĞERİdir, asimptotik sınıf değil.
    # RGGE: dondurme O(n) + DUZ greedy-edge -> `greedy_edge` ile AYNI ifade
    # ve AYNI tur sayisi (1). Fark yalnizca koordinat cercevesidir.
    "rgge_y": "O(n log n) aday-listeli (k=8, tek tur; ızgara devrik)",
    "rgge_x": "O(n log n) aday-listeli (k=8, tek tur; ızgara düz)",
    "rsge_resource": "O(n log n) aday-listeli (bant başına; k=8, tek tur)",
    "rsge_corridor": "O(n log n) aday-listeli (bant başına; k=8, tek tur)",
    "rsge_budget": "O(n log n) aday-listeli (bant başına; k=8, tek tur)",
    "rsge_fixed2": "O(n log n) aday-listeli (bant başına; k=8, tek tur)",
    "repair_vnd": "O(n·k) tur başına (3 komşuluk)",
    # onarım katmanı PANELDEN SEÇİLİR (runner.POOL_REPAIR_CHOICES); varsayılan
    # "batch_2opt_oropt" = aday-listeli (kNN) 2-opt + Or-opt(≤3), relocate YOK,
    # havuzun tümü tek batch'te. Koşulan satırın gerçek seçimi sonuç JSON'undaki
    # repair_key alanında ve hyperparameters()["pool_repair"]["selected"]'da.
    "ils": "bütçe-sınırlı (ils_time)",
    "lkh3": "Lin-Kernighan-Helsgaun (α-yakınlık aday kümesi); süre-sınırlı",
    "concorde": "dal-kesme (branch-and-cut), üstel en kötü durum; süre-sınırlı",
}

# makaledeki rol
ROLE = {
    # ADLANDIRMA (2026-08-02): roller de surum numarasiz yazilir -- makalede
    # okuyucu "v2" diye bir sey aramaz, "serpantin + hangi kurucu, hangi
    # eksende havuz" diye arar. Bkz. runner.PRETTY ustundeki blok.
    # önerilen yöntem çekirdeği (serpantin makro-yapısı + bant-içi kurucu)
    "rotation_strip": "önerilen (θ★ dedektörü)",
    "rotation_hist": "önerilen (θ★ dedektörü)",
    "rotation_mean": "önerilen (θ★ dedektörü)",
    # ablasyon satırları
    "repair_vnd": "ablasyon (P0-4)",
    "ils": "ablasyon (ILS taban çizgisi)",
    # klasik rakip kurucular
    "nn": "rakip (klasik kurucu)",
    "strip": "rakip (klasik kurucu)",
    "hilbert": "rakip (klasik kurucu)",
    "morton": "rakip (klasik kurucu)",
    "nearest_insertion": "rakip (klasik kurucu)",
    "farthest_insertion": "rakip (klasik kurucu)",
    "greedy_edge": "rakip (klasik kurucu)",
    # P0-1 "onarım + klasik kurucu" tam-küme deneyi
    # 2026-08-02 cesitlendirme ekseni -- hicbiri "onerilen" DEGIL,
    # kazanana kosum karar verir.
    # --- RGGE: döndürülmüş ızgarada tek-tur greedy-edge ---------------------
    # Kıyas eşleşi `greedy_edge`dir ve EŞİT BÜTÇEDEDİR (1 tur ↔ 1 tur).
    # Çapa yapısaldır: RGGE(θ=0, ızgara düz) == greedy_edge@knn8_greedy.
    "rgge_y": "önerilen (döndürülmüş ızgara, devrik; eşleş greedy_edge k=8)",
    "rgge_x": "önerilen (döndürülmüş ızgara, düz; eşleş greedy_edge k=8)",
    # --- RSGE: bant sayısı kuralının SINANDIĞI dört satır -------------------
    # Hiçbiri "önerilen" damgası taşımaz: dördü de bir HİPOTEZİN ölçümüdür
    # ("bant sayısı doğru seçilirse bantlama kazandırır"). Kıyas eşleşi her
    # dördü için de `greedy_edge`dir ve bu yapısaldır -- RSGE(θ=0,b=1) o
    # satırın ta kendisidir (çevrim olarak bit-aynı).
    "rsge_resource": "hipotez (RSGE, bant=kaynak tavanı; eşleş greedy_edge k=8)",
    "rsge_corridor": "hipotez (RSGE, bant=koridor/veriden; eşleş greedy_edge k=8)",
    "rsge_budget": "hipotez (RSGE, bant=bütçe modeli; eşleş greedy_edge k=8)",
    "rsge_fixed2": "hipotez (RSGE, bant=sabit 2; eşleş greedy_edge k=8)",
    # DIMACS güçlü kurucusu + öğrenmesiz GLOP
    # exact / altın standart: kıyas TABANI, rakip değil (kazanma sayımı dışı)
    "lkh3": "referans (altın standart, süre-sınırlı)",
    "concorde": "referans (kesin/optimal, süre-sınırlı)",
}

STOCHASTIC_NOTE = {
    # glop_like deterministik varsayılan + isteğe bağlı tohumlu varyant
    "glop_like": "hayır (tohumlu varyant mevcut)",
}


def method_rows() -> list[dict]:
    """runner kayıtlarından yöntem tablosu satırlarını üretir."""
    rows = []
    for key in R.METHOD_ORDER:
        if key in GPU_METHODS:
            gpu = "evet"
        elif key in GPU_WITH_CPU_FALLBACK:
            gpu = "evet (CPU yedekli)"
        else:
            gpu = "hayır"
        stoch = ("evet" if key in R.STOCHASTIC
                 else STOCHASTIC_NOTE.get(key, "hayır"))
        rows.append({
            "anahtar": key,
            "görünen_ad": R.PRETTY[key],
            "aşama": STAGE_LABEL.get(R.STAGE[key], R.STAGE[key]),
            "aile": R.FAMILY.get(key, "—"),
            "karmaşıklık": COMPLEXITY.get(key, "—"),
            "gpu": gpu,
            "stokastik": stoch,
            "rol": ROLE.get(key, "—"),
        })
    return rows


def _tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("_", "\\_").replace("#", "\\#"))


def write_method_table(out_dir: Path = TABLES) -> list[Path]:
    """yontem_tablosu.md / .csv / .tex dosyalarını yazar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = method_rows()
    headers = ["anahtar", "görünen_ad", "aşama", "aile", "karmaşıklık",
               "gpu", "stokastik", "rol"]
    written = []

    md = out_dir / "yontem_tablosu.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Yöntem Tablosu (runner kayıtlarıyla birebir)\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join("---" for _ in headers) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")
    written.append(md)

    csv_path = out_dir / "yontem_tablosu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    written.append(csv_path)

    tex = out_dir / "yontem_tablosu.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("% Otomatik üretildi: benchmark/method_table.py\n")
        f.write("\\begin{tabular}{lllllll}\n\\toprule\n")
        f.write("Anahtar & Yöntem & Aşama & Karmaşıklık & GPU & Stokastik & Rol \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(" & ".join(_tex_escape(str(r[h])) for h in
                               ("anahtar", "görünen_ad", "aşama", "karmaşıklık",
                                "gpu", "stokastik", "rol")) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    written.append(tex)
    return written


# ---------------------------------------------------------------------------
#  Sonuç tablosu (Tablo 1/2 formatı): set × yöntem gap% + süre
# ---------------------------------------------------------------------------
def load_result_matrix(results_dir: Path = RESULTS):
    """results/*.json -> (setler, yöntemler, hücreler).

    hücreler[set][key] = {"gap", "time", "gap_mean", "gap_std", "n_seeds"}
    Stokastik satırlarda gap_mean/gap_std (P1-6 ortalama±std) doldurulur."""
    datasets = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "methods" not in d:
            continue  # eski/farklı şema (liste kök vb.) — tablo dışı
        name = d.get("dataset", p.stem)
        cells = {}
        for m in d.get("methods", []):
            st = m.get("stochastic")
            gap_mean = gap_std = None
            n_seeds = None
            if st:
                n_seeds = st.get("n_seeds")
                if st.get("mean_gap") is not None:
                    gap_mean = st["mean_gap"]
                # std'yi gap uzayında yaklaşık ölçekle (std_cost/bks*100)
                bks = d.get("bks")
                if bks and st.get("std_sample") is not None:
                    gap_std = round(100.0 * st["std_sample"] / bks, 3)
            # Hücreler ROW_ID ile anahtarlanır: aynı yöntemin farklı panel
            # seçimleriyle koşulmuş varyantları (repair_vnd@greedy_snake_v3 gibi)
            # AYRI sütun/satırdır, birbirini ezmez (2026-07-25).
            cells[m.get("row_id") or m["key"]] = {
                "gap": m.get("gap"), "time": m.get("time"),
                "gap_mean": gap_mean, "gap_std": gap_std, "n_seeds": n_seeds,
                "key": m["key"], "name": m.get("name"),
            }
        datasets[name] = {"n": d.get("n"), "bks": d.get("bks"), "cells": cells}
    return datasets


def write_result_tables(results_dir: Path = RESULTS,
                        out_dir: Path = TABLES) -> list[Path]:
    """Tablo 1/2 formatında sonuç tabloları: set × yöntem gap% (süre).

    Özet satırları: ortalama gap, ortalama süre, kazanılan-set sayısı
    (en düşük gap; beraberlikte kısa süre)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_result_matrix(results_dir)
    if not datasets:
        return []
    set_names = sorted(datasets, key=lambda s: (datasets[s]["n"] or 0, s))
    # Satır listesi ROW_ID'lerden kurulur ve METHOD_ORDER sırasına dizilir;
    # bir yöntemin varyantları taban satırının hemen ardından gelir.
    _all_ids = {rid for s in set_names for rid in datasets[s]["cells"]}

    def _sortkey(rid):
        base, _, choice = rid.partition("@")
        try:
            i = R.METHOD_ORDER.index(base)
        except ValueError:
            i = len(R.METHOD_ORDER)
        return (i, choice != "", choice)

    keys = sorted(_all_ids, key=_sortkey)
    # Gösterim adı: varyantlı satırlarda sonuç dosyasındaki "← <seçim>" adı,
    # yoksa PRETTY. Böylece tabloda hangi girdi/onarımla koşulduğu okunur.
    label_of = {}
    for rid in keys:
        base = rid.partition("@")[0]
        nm = None
        for s in set_names:
            c = datasets[s]["cells"].get(rid)
            if c and c.get("name"):
                nm = c["name"]
                break
        label_of[rid] = nm if (nm and "←" in nm) else R.PRETTY.get(base, base)

    def cell_gap(s, k):
        c = datasets[s]["cells"].get(k)
        return None if c is None else c["gap"]

    def cell_time(s, k):
        c = datasets[s]["cells"].get(k)
        return None if c is None else c["time"]

    # özet istatistikleri. Kazanma yarışı YALNIZ yarışan yöntemler arasında
    # yapılır: referans (LKH-3/Concorde) satırları hem aday hem rakip olarak
    # dışarıda tutulur (bkz. NON_COMPETING_STAGES) -- gap sütunları yerinde
    # kalır, "Kazanılan" sütunları "—" olur.
    competing = [k for k in keys
                 if R.STAGE.get(k.partition("@")[0]) not in NON_COMPETING_STAGES]
    summary = {}
    for k in keys:
        gaps = [g for g in (cell_gap(s, k) for s in set_names) if g is not None]
        times = [t for t in (cell_time(s, k) for s in set_names) if t is not None]
        wins = None if k not in competing else 0
        for s in (set_names if k in competing else ()):
            col = [(cell_gap(s, kk), cell_time(s, kk) or 0.0, kk)
                   for kk in competing if cell_gap(s, kk) is not None]
            if col and min(col)[2] == k:
                wins += 1
        summary[k] = {
            "mean_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "mean_time": round(sum(times) / len(times), 2) if times else None,
            "wins": wins, "coverage": len(gaps),
        }

    written = []

    def wins_str(k):
        # referans satirlari yarismaz -> bos hucre ("0 kazandi" demek DEGIL)
        w = summary[k]["wins"]
        return "—" if w is None else str(w)

    def fmt(s, k):
        c = datasets[s]["cells"].get(k)
        if c is None or c["gap"] is None:
            return "—"
        if c["gap_mean"] is not None and c["gap_std"] is not None:
            return f"{c['gap_mean']:.2f}±{c['gap_std']:.2f}"
        return f"{c['gap']:.2f}"

    md = out_dir / "sonuc_tablosu.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Sonuç Tablosu — gap% (BKS'ye göre) ± std (stokastik)\n\n")
        f.write("Set bilgisi: " + ", ".join(
            f"{s} (n={datasets[s]['n']}, BKS={datasets[s]['bks']})"
            for s in set_names) + "\n\n")
        f.write("| Yöntem | " + " | ".join(set_names) +
                " | Ort. gap | Ort. süre (s) | Kazanılan |\n")
        f.write("|" + "|".join("---" for _ in range(len(set_names) + 4)) + "|\n")
        for k in keys:
            sm = summary[k]
            f.write(f"| {label_of[k]} | " + " | ".join(fmt(s, k) for s in set_names)
                    + f" | {sm['mean_gap']} | {sm['mean_time']} | {wins_str(k)} |\n")
    written.append(md)

    csv_path = out_dir / "sonuc_tablosu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yontem", "anahtar"] + [f"{s}_gap" for s in set_names]
                   + [f"{s}_time" for s in set_names]
                   + ["ort_gap", "ort_time", "kazanilan", "kapsam"])
        for k in keys:
            sm = summary[k]
            w.writerow([label_of[k], k]
                       + [cell_gap(s, k) for s in set_names]
                       + [cell_time(s, k) for s in set_names]
                       + [sm["mean_gap"], sm["mean_time"], sm["wins"],
                          sm["coverage"]])
    written.append(csv_path)

    tex = out_dir / "sonuc_tablosu.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("% Otomatik üretildi: benchmark/method_table.py\n")
        f.write("\\begin{tabular}{l" + "r" * len(set_names) + "rrr}\n\\toprule\n")
        f.write("Yöntem & " + " & ".join(_tex_escape(s) for s in set_names)
                + " & Ort. & Süre & Kaz. \\\\\n\\midrule\n")
        for k in keys:
            sm = summary[k]
            f.write(_tex_escape(label_of[k]) + " & "
                    + " & ".join(fmt(s, k) for s in set_names)
                    + f" & {sm['mean_gap']} & {sm['mean_time']} & {wins_str(k)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    written.append(tex)
    return written


def main():
    paths = write_method_table()
    print("yöntem tablosu:", *[str(p) for p in paths], sep="\n  ")
    paths = write_result_tables()
    if paths:
        print("sonuç tablosu:", *[str(p) for p in paths], sep="\n  ")
    else:
        print("sonuç tablosu: results/*.json bulunamadı (önce benchmark koşun)")


if __name__ == "__main__":
    main()


# --- CERCEVE DENEYI (2026-09-07): kayitlar frame_methods.py'den (tek kaynak) ---
try:
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    import frame_methods as _FM
    COMPLEXITY.update(_FM.COMPLEXITY)
    ROLE.update(_FM.ROLE)
except Exception:   # tablo uretimi kayit yoksa da calissin
    pass

# --- 2026-09-07 (2. tur): onarim uclusu ve ILS geri geldi ---
COMPLEXITY.update({"repair_two_opt": "O(n·k) tur başına", "repair_or_opt": "O(n·k) tur başına",
                   "repair_relocate": "O(n·k) tur başına", "ils": "süre bütçeli (cfg ils_time)"})
ROLE.update({"repair_two_opt": "ablasyon (tek komşuluk)", "repair_or_opt": "ablasyon (tek komşuluk)",
             "repair_relocate": "ablasyon (tek komşuluk)", "ils": "ablasyon (ILS taban çizgisi)"})
