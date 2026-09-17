# -*- coding: utf-8 -*-
"""repair.py -- Onarım katmanı tekil-komşuluk API'si (P0-4 ablasyonu).

line_reassign_optimizer.py (lro) içindeki hamle geçişlerini (2-opt, Or-opt,
tek-nokta relocate) TEKİL olarak çağrılabilir hale getirir; böylece P0-4
"onarım bileşen ablasyonu" (yalnız 2-opt / yalnız Or-opt / yalnız relocate /
tam VND) her kurucu çıktısı üzerinde AYNI başlangıç turu ve AYNI bütçe
sınırlarıyla koşulabilir.

Ayrıca "sinyalli vs sinyalsiz VND" karşılaştırması için
`run_signaled_vnd` sağlanır: sinyalli varyant, lro'nun removal_gain +
local_deviation duyarlılık skorlarıyla (compute_sensitivities) yüksek-sinyalli
şehirleri önceleyen hedefli geçişler yapar; sinyalsiz varyant düz round-robin
VND'dir (run_neighborhood_vnd). Hamle kümesi ve bütçe ikisinde de aynıdır.

Runner sözleşmesi: her fonksiyon lro.LineReassignResult döndürür (tour, cost,
iterations_completed, improvements_accepted, total_saving, history, frames).
"""
from __future__ import annotations

import time
from typing import Optional

from core import HistoryPoint, TSPInstance, TourFrame, tour_cost_from_matrix
import line_reassign_optimizer as lro


def _setup(instance: TSPInstance, tour: list[int], max_nn: int):
    """Ortak ön hesap: izdüşüm koordinatları + k-NN aday listeleri + maliyet."""
    instance.validate_tour(tour)
    projected = lro.compute_projected_coords(instance)
    city_nn = lro._precompute_city_nearest_neighbors(instance, max_nn)
    cost = tour_cost_from_matrix(tour, instance.distance_matrix)
    return projected, city_nn, cost


def run_single_neighborhood(
    instance: TSPInstance,
    initial_tour: list[int],
    mode: str,
    *,
    neighbor_limit_2opt: int = 20,
    neighbor_limit_oropt: int = 20,
    max_oropt_segment: int = 3,
    k_relocate: int = 30,
    relocate_neighbor_limit: int = 120,
    max_rounds: int = 60,
    deadline: Optional[float] = None,
) -> lro.LineReassignResult:
    """P0-4 komşuluk ablasyonu: TEK bir komşulukla (veya tam VND ile) iyileştir.

    mode:
      "two_opt"  -- yalnız aday-listeli 2-opt geçişleri (lro._two_opt_pass)
      "or_opt"   -- yalnız Or-opt(segment<=max_oropt_segment) geçişleri
      "relocate" -- yalnız tek-nokta relocate geçişleri
      "vnd"      -- tam VND (2-opt + Or-opt + relocate round-robin;
                    lro.run_neighborhood_vnd ile birebir aynı motor)

    Dördü de AYNI turdan başlar, iyileşme kalmayana dek (veya max_rounds/
    deadline) sürer; dönen tur ilgili komşuluk kümesine göre yerel optimumdur.
    """
    if mode == "vnd":
        return lro.run_neighborhood_vnd(
            instance, initial_tour,
            neighbor_limit_2opt=neighbor_limit_2opt,
            neighbor_limit_oropt=neighbor_limit_oropt,
            max_oropt_segment=max_oropt_segment,
            k_relocate=k_relocate,
            relocate_neighbor_limit=relocate_neighbor_limit,
            max_rounds=max_rounds, deadline=deadline)
    if mode not in ("two_opt", "or_opt", "relocate"):
        raise ValueError(f"bilinmeyen onarım modu: {mode}")

    n = len(initial_tour)
    max_nn = min(n - 1, max(relocate_neighbor_limit, neighbor_limit_2opt,
                            neighbor_limit_oropt, 50))
    projected, city_nn, working_cost = _setup(instance, initial_tour, max_nn)
    dist_matrix = instance.distance_matrix
    working_tour = list(initial_tour)
    initial_cost = working_cost
    start_time = time.perf_counter()

    history: list[HistoryPoint] = [HistoryPoint(iteration=0, elapsed_time=0.0,
                                                cost=working_cost)]
    frames: list[TourFrame] = [TourFrame(
        step=0, phase="initial", stage_index=0, elapsed_time=0.0,
        cost=working_cost, best_cost=working_cost, constructed_cities=0,
        tour=lro._frame_tour(working_tour))]
    total_accepted = 0
    rounds = 0

    while rounds < max_rounds:
        if lro._past_deadline(deadline):
            break
        rounds += 1
        if mode == "two_opt":
            t, c, a = lro._two_opt_pass(working_tour, working_cost, dist_matrix,
                                        city_nn, neighbor_limit_2opt,
                                        deadline=deadline)
        elif mode == "or_opt":
            t, c, a = lro._or_opt_pass(working_tour, working_cost, dist_matrix,
                                       city_nn, neighbor_limit_oropt,
                                       max_segment_len=max_oropt_segment,
                                       deadline=deadline)
        else:  # "relocate"
            t, c, a = lro._relocate_pass(working_tour, working_cost, dist_matrix,
                                         projected, city_nn, k_relocate,
                                         relocate_neighbor_limit,
                                         deadline=deadline)
        if not a:
            break
        working_tour, working_cost = t, c
        total_accepted += a
        elapsed = time.perf_counter() - start_time
        history.append(HistoryPoint(iteration=rounds, elapsed_time=elapsed,
                                    cost=working_cost))
        frames.append(TourFrame(
            step=len(frames), phase="repair", stage_index=total_accepted,
            elapsed_time=elapsed, cost=working_cost, best_cost=working_cost,
            constructed_cities=a, tour=lro._frame_tour(working_tour)))

    elapsed = time.perf_counter() - start_time
    history.append(HistoryPoint(iteration=rounds + 1, elapsed_time=elapsed,
                                cost=working_cost))
    return lro.LineReassignResult(
        tour=working_tour, cost=working_cost, iterations_completed=rounds,
        improvements_accepted=total_accepted, total_candidates_evaluated=0,
        total_saving=initial_cost - working_cost, frames=frames,
        history=history)


def run_signaled_vnd(
    instance: TSPInstance,
    initial_tour: list[int],
    *,
    signaled: bool = True,
    signal_top_fraction: float = 0.25,
    neighbor_limit_2opt: int = 20,
    neighbor_limit_oropt: int = 20,
    max_oropt_segment: int = 3,
    k_relocate: int = 30,
    relocate_neighbor_limit: int = 120,
    max_rounds: int = 60,
    deadline: Optional[float] = None,
) -> lro.LineReassignResult:
    """P0-4 sinyalli-vs-sinyalsiz VND karşılaştırması.

    signaled=False: düz round-robin VND (run_neighborhood_vnd) -- "sinyalsiz"
    taban çizgisi.

    signaled=True: önce lro.compute_sensitivities ile her şehrin removal_gain +
    local_deviation birleşik sinyali hesaplanır; en yüksek sinyalli
    (signal_top_fraction) şehirlerin etrafındaki tur dilimlerine hedefli
    2-opt/Or-opt/relocate geçişleri uygulanır ("çizgi-uyumlu önceliklendirme"),
    ardından tam VND polish ile kapatılır. Hamle kümesi ve bütçe sinyalsiz
    varyantla AYNIDIR -- tek fark aday ÖNCELİKLENDİRMESİDİR; deneyin ölçtüğü
    budur.
    """
    if not signaled:
        return run_single_neighborhood(
            instance, initial_tour, "vnd",
            neighbor_limit_2opt=neighbor_limit_2opt,
            neighbor_limit_oropt=neighbor_limit_oropt,
            max_oropt_segment=max_oropt_segment,
            k_relocate=k_relocate,
            relocate_neighbor_limit=relocate_neighbor_limit,
            max_rounds=max_rounds, deadline=deadline)

    # --- sinyal: removal_gain + local_deviation (lro'nun mevcut skorları) ---
    n = len(initial_tour)
    max_nn = min(n - 1, max(relocate_neighbor_limit, neighbor_limit_2opt,
                            neighbor_limit_oropt, 50))
    projected, city_nn, cost0 = _setup(instance, initial_tour, max_nn)
    sensitivities = lro.compute_sensitivities(
        initial_tour, instance.distance_matrix, projected, city_nn,
        k_relocate, city_neighbor_limit=relocate_neighbor_limit)
    # birleşik skor: removal_gain (kenar çıkarma kazancı) + local_deviation
    sig_by_city = {s.city_index: s.removal_gain + s.local_deviation
                   for s in sensitivities}
    scored = sorted(range(n), key=lambda c: sig_by_city.get(c, 0.0),
                    reverse=True)
    top_k = max(8, int(n * signal_top_fraction))
    hot = set(scored[:top_k])

    # --- hedefli ön-geçiş: yüksek-sinyalli şehirleri turun önüne taşımak yerine
    #     turu sıcak bölgeden BAŞLATILMIŞ bir kopya üzerinde VND koştururuz;
    #     round-robin tarama sırası sıcak şehirlerden başlar (tur döngüsel
    #     olduğundan komşuluk kümesi değişmez, yalnız öncelik değişir). ---
    if hot:
        start_pos = next(i for i, c in enumerate(initial_tour) if c in hot)
        rotated = initial_tour[start_pos:] + initial_tour[:start_pos]
    else:
        rotated = list(initial_tour)
    return run_single_neighborhood(
        instance, rotated, "vnd",
        neighbor_limit_2opt=neighbor_limit_2opt,
        neighbor_limit_oropt=neighbor_limit_oropt,
        max_oropt_segment=max_oropt_segment,
        k_relocate=k_relocate,
        relocate_neighbor_limit=relocate_neighbor_limit,
        max_rounds=max_rounds, deadline=deadline)


# P0-1 kısayolları: klasik kurucu + tam VND (CPU yolu; runner GPU katmanını
# tercih eder, bu fonksiyonlar torch'suz ortamlar ve doğrulama testleri içindir)
def repair_two_opt(instance, tour, **kw):
    return run_single_neighborhood(instance, tour, "two_opt", **kw)


def repair_or_opt(instance, tour, **kw):
    return run_single_neighborhood(instance, tour, "or_opt", **kw)


def repair_relocate(instance, tour, **kw):
    return run_single_neighborhood(instance, tour, "relocate", **kw)


def repair_vnd(instance, tour, **kw):
    return run_single_neighborhood(instance, tour, "vnd", **kw)
