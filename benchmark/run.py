# -*- coding: utf-8 -*-
"""benchmark/run.py -- Makale benchmark koşucusu (tek komut).

Örnekler:
    # duman testi (3 küçük set, kısa bütçe, 3 tohum)
    python benchmark/run.py --sets ali535,kroA100,bcl380 --budget 0.3 --seeds 3 --quick

    # tam-küme P0 koşusu (138 set, tüm yöntemler) -- uzun
    python benchmark/run.py --all

    # belirli yöntemler
    python benchmark/run.py --sets kroA100 --methods strip,rotation_pca,glop_like

Çıktılar: results/{set}.json (runner şeması) + results/tables/ altında
yöntem/sonuç tabloları (--no-tables ile kapatılabilir).

GPU koşuları için kullanıcının Python'unu kullanın:
    /c/Python313/python.exe benchmark/run.py --sets ... (CUDA varsa GPU
    yöntemleri otomatik GPU'ya, yoksa zarif CPU yedeğine düşer)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import runner as R  # noqa: E402


def all_known_sets() -> list[str]:
    """Klasik 9 + kataloglanmış VLSI/TSPLIB95 setlerinin indirilmiş olanları."""
    names = list(R.ALL_DATASETS)
    for cat, cdir in ((R.VLSI, R.DATA_VLSI), (R.TSPLIB, R.DATA_TSPLIB)):
        for name in sorted(cat.CATALOG):
            if (cdir / f"{name}.tsp").exists():
                names.append(name)
    return names


def _delegate_to_dispatcher(args) -> int:
    """--jobs>1 verildiğinde koşumu hpc.dispatcher'a devret.

    Argümanları yeniden dizgeye çevirip dispatcher.main()'e veriyoruz;
    böylece iki ayrı argüman ayrıştırıcısını elle senkron tutmak yerine tek
    bir sözleşme kalıyor. Dağıtıcı her set için BU betiği --jobs vermeden
    (yani tek süreç kipinde) yeniden çağırır.
    """
    from hpc import dispatcher

    argv = ["--jobs", args.jobs, "--budget", str(args.budget)]
    if args.all:
        argv.append("--all")
    if args.sets:
        argv += ["--sets", args.sets]
    if args.methods:
        argv += ["--methods", args.methods]
    if args.seeds is not None:
        argv += ["--seeds", str(args.seeds)]
    if args.quick:
        argv.append("--quick")
    if args.force:
        argv.append("--force")
    if args.construction_max_s is not None:
        argv += ["--construction-max-s", str(args.construction_max_s)]
    if args.exact_time is not None:
        argv += ["--exact-time", str(args.exact_time)]
    if args.no_tables:
        argv.append("--no-tables")
    if args.reserve:
        argv += ["--reserve", str(args.reserve)]
    if args.dry_run:
        argv.append("--dry-run")
    return dispatcher.main(argv)


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy Snake benchmark koşucusu")
    ap.add_argument("--sets", type=str, default=None,
                    help="virgüllü set listesi (ör. ali535,kroA100,bcl380)")
    ap.add_argument("--all", action="store_true",
                    help="indirilmiş TÜM setler (klasik 9 + VLSI/Ülke/Sanat)")
    ap.add_argument("--methods", type=str, default=None,
                    help="virgüllü yöntem anahtarları; varsayılan: hepsi")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="yöntem süre üst sınırları için çarpan (time-budget)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="stokastik tohum sayısı (P1-6: >=10 önerilir)")
    ap.add_argument("--choices", type=str, default=None,
                    help="bu koşum için panel seçimlerini geçersiz kıl: "
                         "\"yöntem=seçim,...\" (panel çoğaltma özelliği "
                         "bunu kullanır; bkz. runner.set_choice_overrides)")
    ap.add_argument("--quick", action="store_true",
                    help="duman testi kipi (bütçeler kısalır)")
    ap.add_argument("--force", action="store_true",
                    help="devam modunu kapat; her şey yeniden koşulur")
    ap.add_argument("--construction-max-s", type=float, default=None,
                    help="inşa yöntemleri duvar-saati üst sınırı (s)")
    ap.add_argument("--exact-time", type=float, default=None,
                    help="exact/altın standart çözücülerin (LKH-3, Concorde) "
                         f"duvar-saati üst sınırı (s); "
                         f"varsayılan {R.EXACT_TIME_DEFAULT:.0f}s")
    ap.add_argument("--no-tables", action="store_true",
                    help="sonuç tablolarını üretme")
    # -- çok çekirdekli koşum -------------------------------------------------
    # Varsayılan 1: bu betik TEK süreçtir ve dağıtıcı tarafından çocuk olarak
    # başlatılır. 1'den büyük (ya da "auto") verildiğinde iş hpc.dispatcher'a
    # devredilir; o da her set için bu betiği yeniden çağırır. Tek giriş
    # noktası korunsun diye bayrak burada duruyor.
    ap.add_argument("--jobs", "-j", type=str, default="1",
                    help="eşzamanlı süreç sayısı; 'auto' = kullanılabilir "
                         "çekirdek sayısı (TRUBA/SLURM tahsisine saygılı). "
                         "1'den büyükse koşum hpc.dispatcher'a devredilir")
    ap.add_argument("--reserve", type=int, default=0,
                    help="--jobs auto'da boş bırakılacak çekirdek sayısı")
    ap.add_argument("--dry-run", action="store_true",
                    help="--jobs ile: koşma, yalnızca dağıtım planını yazdır")
    args = ap.parse_args()

    if args.jobs != "1" or args.dry_run:
        return _delegate_to_dispatcher(args)

    if args.all:
        targets = all_known_sets()
    elif args.sets:
        targets = [s.strip() for s in args.sets.split(",") if s.strip()]
    else:
        ap.error("--sets LISTE veya --all verin")

    methods = ([m.strip() for m in args.methods.split(",") if m.strip()]
               if args.methods else None)
    try:
        overrides = R.set_choice_overrides(R.parse_choice_overrides(args.choices))
    except ValueError as exc:
        ap.error(str(exc))
    if overrides:
        print(f"[benchmark] seçim geçersiz kılma: {overrides}")
    if methods:
        unknown = [m for m in methods if m not in R.METHOD_ORDER]
        if unknown:
            ap.error(f"bilinmeyen yöntem anahtarı: {unknown} "
                     f"(kayıtlı: {R.METHOD_ORDER})")

    print(f"[benchmark] {len(targets)} set: {', '.join(targets)}")
    print(f"[benchmark] yöntemler: {methods or 'TÜMÜ (' + str(len(R.METHOD_ORDER)) + ')'} "
          f"budget={args.budget} seeds={args.seeds} quick={args.quick}")

    failed = []
    t_start = time.perf_counter()
    for name in targets:
        try:
            R.run_dataset(name, methods=methods, time_budget=args.budget,
                          seeds_override=args.seeds, quick=args.quick,
                          force=args.force,
                          construction_max_s=args.construction_max_s,
                          exact_time=args.exact_time,
                          choice_overrides=overrides)
        except Exception as exc:
            import traceback
            failed.append(name)
            print(f"[benchmark] {name} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()

    if not args.no_tables:
        from benchmark.method_table import write_method_table, write_result_tables
        for p in write_method_table():
            print(f"[benchmark] tablo: {p}")
        for p in write_result_tables():
            print(f"[benchmark] tablo: {p}")

    dt = time.perf_counter() - t_start
    if failed:
        print(f"[benchmark] BAŞARISIZ setler: {', '.join(failed)} "
              f"({dt:.0f}s)")
        return 1
    print(f"[benchmark] tamam ({dt:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
