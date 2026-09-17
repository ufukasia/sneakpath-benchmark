# -*- coding: utf-8 -*-
"""Waterloo Üniversitesi VLSI TSP koleksiyonu kataloğu.

Kaynak: https://www.math.uwaterloo.ca/tsp/vlsi/  (Bonn Üniversitesi VLSI
verileri; Andre Rohe). Bu koleksiyon mikroçip/VLSI yerleşiminden türetilmiş
gerçek nokta kümeleridir -- snake-path + onarım hattının asıl hedef alanı.

CATALOG: name -> (n, best_known, status)
  status "opt" = optimalliği kanıtlanmış tur uzunluğu
  status "ub"  = bilinen en iyi tur (üst sınır; henüz kanıtlanmamış)
Değerler summary sayfasından (2026-07 itibarıyla). Gap bu değere göre
hesaplanır; "ub" örneklerinde negatif gap teorik olarak mümkündür (bilinen
en iyi turdan daha iyi bir tur bulunması = yeni dünya rekoru).

İndirme kalıbı: https://www.math.uwaterloo.ca/tsp/vlsi/{name}.tsp (EUC_2D)
"""
from __future__ import annotations

URL_PATTERN = "https://www.math.uwaterloo.ca/tsp/vlsi/{name}.tsp"

CATALOG: dict[str, tuple[int, int, str]] = {
    "xqf131":   (131,    564,     "opt"),
    "xqg237":   (237,    1019,    "opt"),
    "pma343":   (343,    1368,    "opt"),
    "pka379":   (379,    1332,    "opt"),
    "bcl380":   (380,    1621,    "opt"),
    "pbl395":   (395,    1281,    "opt"),
    "pbk411":   (411,    1343,    "opt"),
    "pbn423":   (423,    1365,    "opt"),
    "pbm436":   (436,    1443,    "opt"),
    "xql662":   (662,    2513,    "opt"),
    "rbx711":   (711,    3115,    "opt"),
    "rbu737":   (737,    3314,    "opt"),
    "dkg813":   (813,    3199,    "opt"),
    "lim963":   (963,    2789,    "opt"),
    "pbd984":   (984,    2797,    "opt"),
    "xit1083":  (1083,   3558,    "opt"),
    "dka1376":  (1376,   4666,    "opt"),
    "dca1389":  (1389,   5085,    "opt"),
    "dja1436":  (1436,   5257,    "opt"),
    "icw1483":  (1483,   4416,    "opt"),
    "fra1488":  (1488,   4264,    "opt"),
    "rbv1583":  (1583,   5387,    "opt"),
    "rby1599":  (1599,   5533,    "opt"),
    "fnb1615":  (1615,   4956,    "opt"),
    "djc1785":  (1785,   6115,    "opt"),
    "dcc1911":  (1911,   6396,    "opt"),
    "dkd1973":  (1973,   6421,    "opt"),
    "djb2036":  (2036,   6197,    "opt"),
    "dcb2086":  (2086,   6600,    "opt"),
    "bva2144":  (2144,   6304,    "opt"),
    "xqc2175":  (2175,   6830,    "opt"),
    "bck2217":  (2217,   6764,    "opt"),
    "xpr2308":  (2308,   7219,    "opt"),
    "ley2323":  (2323,   8352,    "opt"),
    "dea2382":  (2382,   8017,    "opt"),
    "rbw2481":  (2481,   7724,    "opt"),
    "pds2566":  (2566,   7643,    "opt"),
    "mlt2597":  (2597,   8071,    "opt"),
    "bch2762":  (2762,   8234,    "opt"),
    "irw2802":  (2802,   8423,    "opt"),
    "lsm2854":  (2854,   8014,    "opt"),
    "dbj2924":  (2924,   10128,   "opt"),
    "xva2993":  (2993,   8492,    "opt"),
    "pia3056":  (3056,   8258,    "opt"),
    "dke3097":  (3097,   10539,   "opt"),
    "lsn3119":  (3119,   9114,    "opt"),
    "lta3140":  (3140,   9517,    "opt"),
    "fdp3256":  (3256,   10008,   "opt"),
    "beg3293":  (3293,   9772,    "opt"),
    "dhb3386":  (3386,   11137,   "opt"),
    "fjs3649":  (3649,   9272,    "opt"),
    "fjr3672":  (3672,   9601,    "opt"),
    "dlb3694":  (3694,   10959,   "opt"),
    "ltb3729":  (3729,   11821,   "opt"),
    "xqe3891":  (3891,   11995,   "opt"),
    "xua3937":  (3937,   11239,   "opt"),
    "dkc3938":  (3938,   12503,   "opt"),
    "dkf3954":  (3954,   12538,   "opt"),
    "bgb4355":  (4355,   12723,   "opt"),
    "bgd4396":  (4396,   13009,   "opt"),
    "frv4410":  (4410,   10711,   "opt"),
    "bgf4475":  (4475,   13221,   "opt"),
    "xqd4966":  (4966,   15316,   "opt"),
    "fqm5087":  (5087,   13029,   "opt"),
    "fea5557":  (5557,   15445,   "opt"),
    "xsc6880":  (6880,   21535,   "opt"),
    "bnd7168":  (7168,   21834,   "opt"),
    "lap7454":  (7454,   19535,   "opt"),
    "ida8197":  (8197,   22338,   "opt"),
    "dga9698":  (9698,   27724,   "opt"),
    "xmc10150": (10150,  28387,   "opt"),
    "xvb13584": (13584,  37083,   "opt"),
    "xrb14233": (14233,  45462,   "ub"),
    "xia16928": (16928,  52850,   "ub"),
    "pjh17845": (17845,  48092,   "ub"),
    "frh19289": (19289,  55798,   "ub"),
    "fnc19402": (19402,  59287,   "ub"),
    "ido21215": (21215,  63517,   "ub"),
    "fma21553": (21553,  66527,   "ub"),
    "lsb22777": (22777,  60977,   "ub"),
    "xrh24104": (24104,  69294,   "ub"),
    "bbz25234": (25234,  69335,   "ub"),
    "irx28268": (28268,  72607,   "ub"),
    "fyg28534": (28534,  78562,   "ub"),
    "icx28698": (28698,  78087,   "ub"),
    "boa28924": (28924,  79622,   "ub"),
    "ird29514": (29514,  80353,   "ub"),
    "pbh30440": (30440,  88313,   "ub"),
    "xib32892": (32892,  96757,   "ub"),
    "fry33203": (33203,  97240,   "ub"),
    "bby34656": (34656,  99159,   "ub"),
    "pba38478": (38478,  108318,  "ub"),
    "ics39603": (39603,  106819,  "ub"),
    "rbz43748": (43748,  125183,  "ub"),
    "fht47608": (47608,  125104,  "ub"),
    "fna52057": (52057,  147789,  "ub"),
    "bna56769": (56769,  158078,  "ub"),
    "dan59296": (59296,  165371,  "ub"),
    "sra104815": (104814, 251342,  "ub"),   # 2026-09-09: DIMENSION 104814 (dosya adi 104815)
    "ara238025": (238025, 578761,  "ub"),
    "lra498378": (498378, 2168039, "ub"),
    "lrb744710": (744710, 1611232, "ub"),
}

BKS: dict[str, int] = {name: v[1] for name, v in CATALOG.items()}


def url_for(name: str) -> str:
    if name not in CATALOG:
        raise KeyError(f"VLSI kataloğunda yok: {name}")
    return URL_PATTERN.format(name=name)
