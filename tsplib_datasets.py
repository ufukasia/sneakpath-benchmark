# -*- coding: utf-8 -*-
"""Orijinal TSPLIB95 simetrik TSP koleksiyonu katalogu.

Kaynak: Heidelberg Universitesi (Gerhard Reinelt),
https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
Optimum degerler ayni sitedeki resmi STSP.html listesinden ALINMISTIR --
elle yazilmamistir (yanlis bir optimum butun gap sutununu sessizce bozar).

NEDEN BU KOLEKSIYON (2026-07-31, kullanici onceligi): TSPLIB'in icinde
GERCEK DEVRE verisi var -- delme (drilling) problemleri ve programlanabilir
mantik dizileri. Waterloo VLSI koleksiyonu cip YERLESIMI noktalaridir;
buradaki delme problemleri ise baski devre kartinda DELINECEK deliklerin
sirasidir. Ikisi ayni uygulama alaninin iki farkli asamasi: yerlesim ve
uretim. `CIRCUIT` alt kumesi (26 ornek, n=159..85900) bunlari isaretler.

KAPSAM:
  * 94 ornek KOORDINATLI (NODE_COORD_SECTION) -> bu katalogda.
  * 17 ornek EXPLICIT mesafe matrisidir (koordinat YOK) ve
    KAPSAM DISIDIR: bu calismanin butun yontemleri (serpantin bantlama,
    izgara k-NN, aci tespiti) noktalarin DUZLEMDEKI yerine dayanir,
    matristen bu bilgi geri uretilemez. Disarida birakilanlar:
    bayg29, bays29, brazil58, brg180, dantzig42, fri26, gr120, gr17, gr21, gr24, gr48, hk48, pa561, si1032, si175, si535, swiss42.

MESAFE TIPLERI: EUC_2D, GEO, CEIL_2D ve ATT. Son ikisi 2026-07-31'de
tsplib_engine'e eklendi (bkz. `metric_kind`); oncesinde CEIL_2D ve ATT
sessizce EUC_2D sanilirdi ve gap YANLIS cikardi. CEIL_2D ozellikle onemli --
o tipteki 4 ornegin 3'u DEVRE verisidir (pla7397/pla33810/pla85900).

CATALOG: name -> (n, best_known, status, edge_weight_type)
  Dort alanlidir (diger koleksiyonlar uc alanli): TSPLIB tek koleksiyonda
  dort farkli mesafe tipi barindiran tek koleksiyondur, tip satirda gorunmeli.
  status hepsinde "opt" -- resmi listede yalniz KANITLANMIS optimumlar var.

Indirme kalibi: {name}.tsp.gz (Heidelberg), yedek: GitHub aynasi.
NOT: klasik data/ dizinindeki 9 ornek (kroA100, rat783, ali535, gr666,
pcb3038, fl3795, fnl4461, rl5915, usa13509) bu katalogda DA vardir ama
dosyalari data/ altindadir; runner.dataset_path once orayi bakar.
"""
from __future__ import annotations

URL_PATTERN = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz"
#: Heidelberg zaman zaman erisilemez oluyor; ayni dosyalarin duz (.gz DEGIL)
#: hali bu aynada duruyor. server._download_vlsi once URL_PATTERN'i dener.
MIRROR_PATTERN = "https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp"

# name -> (n, optimum, status, edge_weight_type)
CATALOG: dict[str, tuple[int, int, str, str]] = {
    # ---------------- DEVRE / PCB ALT KUMESI (26 ornek) ----------------
    # Delme problemleri (Reinelt ve ark.) + programlanabilir mantik dizileri.
    # `CIRCUIT` kumesi asagida bu adlardan turetilir.
    "u159":        (159   , 42080     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "d198":        (198   , 15780     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "a280":        (280   , 2579      , "opt", "EUC_2D"),   # drilling problem (Ludwig)
    "fl417":       (417   , 11861     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "pcb442":      (442   , 50778     , "opt", "EUC_2D"),   # Drilling problem (Groetschel/Juenger/Reinelt)
    "d493":        (493   , 35002     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u574":        (574   , 36905     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "p654":        (654   , 34643     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "d657":        (657   , 48912     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u724":        (724   , 41910     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u1060":       (1060  , 224094    , "opt", "EUC_2D"),   # Drilling problem problem (Reinelt)
    "pcb1173":     (1173  , 56892     , "opt", "EUC_2D"),   # Drilling problem (Juenger/Reinelt)
    "d1291":       (1291  , 50801     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "fl1400":      (1400  , 20127     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u1432":       (1432  , 152970    , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "fl1577":      (1577  , 22249     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "d1655":       (1655  , 62128     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u1817":       (1817  , 57201     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "d2103":       (2103  , 80450     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u2152":       (2152  , 64253     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "u2319":       (2319  , 234256    , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "pcb3038":     (3038  , 137694    , "opt", "EUC_2D"),   # Drilling problem (Junger/Reinelt)
    "fl3795":      (3795  , 28772     , "opt", "EUC_2D"),   # Drilling problem (Reinelt)
    "pla7397":     (7397  , 23260728  , "opt", "CEIL_2D"),   # Programmed logic array (Johnson)
    "pla33810":    (33810 , 66048945  , "opt", "CEIL_2D"),   # Programmed logic array (Johnson)
    "pla85900":    (85900 , 142382641 , "opt", "CEIL_2D"),   # Programmed logic array (Johnson)
    # ---------------- DIGER ORNEKLER (68 ornek) ----------------
    "burma14":     (14    , 3323      , "opt", "GEO"),   # 14-Staedte in Burma (Zaw Win)
    "ulysses16":   (16    , 6859      , "opt", "GEO"),   # Odyssey of Ulysses (Groetschel/Padberg)
    "ulysses22":   (22    , 7013      , "opt", "GEO"),   # Odyssey of Ulysses (Groetschel/Padberg)
    "att48":       (48    , 10628     , "opt", "ATT"),   # 48 capitals of the US (Padberg/Rinaldi)
    "eil51":       (51    , 426       , "opt", "EUC_2D"),   # 51-city problem (Christofides/Eilon)
    "berlin52":    (52    , 7542      , "opt", "EUC_2D"),   # 52 locations in Berlin (Groetschel)
    "st70":        (70    , 675       , "opt", "EUC_2D"),   # 70-city problem (Smith/Thompson)
    "eil76":       (76    , 538       , "opt", "EUC_2D"),   # 76-city problem (Christofides/Eilon)
    "pr76":        (76    , 108159    , "opt", "EUC_2D"),   # 76-city problem (Padberg/Rinaldi)
    "gr96":        (96    , 55209     , "opt", "GEO"),   # Africa-Subproblem of 666-city TSP (Groetschel)
    "rat99":       (99    , 1211      , "opt", "EUC_2D"),   # Rattled grid (Pulleyblank)
    "kroA100":     (100   , 21282     , "opt", "EUC_2D"),   # 100-city problem A (Krolak/Felts/Nelson)
    "kroB100":     (100   , 22141     , "opt", "EUC_2D"),   # 100-city problem B (Krolak/Felts/Nelson)
    "kroC100":     (100   , 20749     , "opt", "EUC_2D"),   # 100-city problem C (Krolak/Felts/Nelson)
    "kroD100":     (100   , 21294     , "opt", "EUC_2D"),   # 100-city problem D (Krolak/Felts/Nelson)
    "kroE100":     (100   , 22068     , "opt", "EUC_2D"),   # 100-city problem E (Krolak/Felts/Nelson)
    "rd100":       (100   , 7910      , "opt", "EUC_2D"),   # 100-city random TSP (Reinelt)
    "eil101":      (101   , 629       , "opt", "EUC_2D"),   # 101-city problem (Christofides/Eilon)
    "lin105":      (105   , 14379     , "opt", "EUC_2D"),   # 105-city problem (Subproblem of lin318)
    "pr107":       (107   , 44303     , "opt", "EUC_2D"),   # 107-city problem (Padberg/Rinaldi)
    "pr124":       (124   , 59030     , "opt", "EUC_2D"),   # 124-city problem (Padberg/Rinaldi)
    "bier127":     (127   , 118282    , "opt", "EUC_2D"),   # 127 Biergaerten in Augsburg (Juenger/Reinelt)
    "ch130":       (130   , 6110      , "opt", "EUC_2D"),   # 130 city problem (Churritz)
    "pr136":       (136   , 96772     , "opt", "EUC_2D"),   # 136-city problem (Padberg/Rinaldi)
    "gr137":       (137   , 69853     , "opt", "GEO"),   # America-Subproblem of 666-city TSP (Groetschel)
    "pr144":       (144   , 58537     , "opt", "EUC_2D"),   # 144-city problem (Padberg/Rinaldi)
    "ch150":       (150   , 6528      , "opt", "EUC_2D"),   # 150 city Problem (churritz)
    "kroA150":     (150   , 26524     , "opt", "EUC_2D"),   # 150-city problem A (Krolak/Felts/Nelson)
    "kroB150":     (150   , 26130     , "opt", "EUC_2D"),   # 150-city problem B (Krolak/Felts/Nelson)
    "pr152":       (152   , 73682     , "opt", "EUC_2D"),   # 152-city problem (Padberg/Rinaldi)
    "rat195":      (195   , 2323      , "opt", "EUC_2D"),   # Rattled grid (Pulleyblank)
    "kroA200":     (200   , 29368     , "opt", "EUC_2D"),   # 200-city problem A (Krolak/Felts/Nelson)
    "kroB200":     (200   , 29437     , "opt", "EUC_2D"),   # 200-city problem B (Krolak/Felts/Nelson)
    "gr202":       (202   , 40160     , "opt", "GEO"),   # Europe-Subproblem of 666-city TSP (Groetschel)
    "ts225":       (225   , 126643    , "opt", "EUC_2D"),   # 225-city problem (Juenger,Raecke,Tschoecke)
    "tsp225":      (225   , 3916      , "opt", "EUC_2D"),   # A TSP problem (Reinelt)
    "pr226":       (226   , 80369     , "opt", "EUC_2D"),   # 226-city problem (Padberg/Rinaldi)
    "gr229":       (229   , 134602    , "opt", "GEO"),   # Asia/Australia-Subproblem of 666-city TSP (Groetschel)
    "gil262":      (262   , 2378      , "opt", "EUC_2D"),   # 262-city problem (Gillet/Johnson)
    "pr264":       (264   , 49135     , "opt", "EUC_2D"),   # 264-city problem (Padberg/Rinaldi)
    "pr299":       (299   , 48191     , "opt", "EUC_2D"),   # 299-city problem (Padberg/Rinaldi)
    "lin318":      (318   , 42029     , "opt", "EUC_2D"),   # 318-city problem (Lin/Kernighan)
    "linhp318":    (318   , 41345     , "opt", "EUC_2D"),   # Original 318-city problem (Lin/Kernighan)
    "rd400":       (400   , 15281     , "opt", "EUC_2D"),   # 400-city random TSP (Reinelt)
    "gr431":       (431   , 171414    , "opt", "GEO"),   # Europe/Asia/Australia-Subproblem of 666-city TSP (Groets
    "pr439":       (439   , 107217    , "opt", "EUC_2D"),   # 439-city problem (Padberg/Rinaldi)
    "att532":      (532   , 27686     , "opt", "ATT"),   # 532-city problem (Padberg/Rinaldi)
    "ali535":      (535   , 202339    , "opt", "GEO"),   # 535 Airports around the globe (Padberg/Rinaldi)
    "rat575":      (575   , 6773      , "opt", "EUC_2D"),   # Rattled grid (Pulleyblank)
    "gr666":       (666   , 294358    , "opt", "GEO"),   # 666 cities around the world (Groetschel)
    "rat783":      (783   , 8806      , "opt", "EUC_2D"),   # Rattled grid (Pulleyblank)
    "dsj1000":     (1000  , 18660188  , "opt", "CEIL_2D"),   # Clustered random problem (Johnson)
    "pr1002":      (1002  , 259045    , "opt", "EUC_2D"),   # 1002-city problem (Padberg/Rinaldi)
    "vm1084":      (1084  , 239297    , "opt", "EUC_2D"),   # 1084-city problem (Reinelt)
    "rl1304":      (1304  , 252948    , "opt", "EUC_2D"),   # 1304-city TSP (Reinelt)
    "rl1323":      (1323  , 270199    , "opt", "EUC_2D"),   # 1323-city TSP (Reinelt)
    "nrw1379":     (1379  , 56638     , "opt", "EUC_2D"),   # 1379 Orte in Nordrhein-Westfalen (Bachem/Wottawa)
    "vm1748":      (1748  , 336556    , "opt", "EUC_2D"),   # 1784-city problem (Reinelt)
    "rl1889":      (1889  , 316536    , "opt", "EUC_2D"),   # 1889-city TSP (Reinelt)
    "pr2392":      (2392  , 378032    , "opt", "EUC_2D"),   # 2392-city problem (Padberg/Rinaldi)
    "fnl4461":     (4461  , 182566    , "opt", "EUC_2D"),   # Die 5 neuen Laender Deutschlands (Ex-DDR) (Bachem/Wottaw
    "rl5915":      (5915  , 565530    , "opt", "EUC_2D"),   # 5915-city TSP (Reinelt)
    "rl5934":      (5934  , 556045    , "opt", "EUC_2D"),   # 5934-city TSP (Reinelt)
    "rl11849":     (11849 , 923288    , "opt", "EUC_2D"),   # 11849-city TSP (Reinelt)
    "usa13509":    (13509 , 19982859  , "opt", "EUC_2D"),   # Cities with population at least 500 in the continental U
    "brd14051":    (14051 , 469385    , "opt", "EUC_2D"),   # BR Deutschland in den Grenzen von 1989 (Bachem/Wottawa)
    "d15112":      (15112 , 1573084   , "opt", "EUC_2D"),   # Deutschland-Problem (A.Rohe)
    "d18512":      (18512 , 645238    , "opt", "EUC_2D"),   # Bundesrepublik Deutschland (mit Ex-DDR) (Bachem/Wottawa)
}

#: DEVRE alt kumesi -- kullanicinin oncelikli sinifi. Adlar CATALOG'daki
#: COMMENT alanindan turetildi (delme problemi / programlanabilir mantik
#: dizisi); "d15112" ve "d18512" ADLARINA RAGMEN buraya GIRMEZ, cunku onlar
#: Deutschland cografya problemleridir (COMMENT boyle diyor).
CIRCUIT: frozenset[str] = frozenset({
    "u159",
    "d198",
    "a280",
    "fl417",
    "pcb442",
    "d493",
    "u574",
    "p654",
    "d657",
    "u724",
    "u1060",
    "pcb1173",
    "d1291",
    "fl1400",
    "u1432",
    "fl1577",
    "d1655",
    "u1817",
    "d2103",
    "u2152",
    "u2319",
    "pcb3038",
    "fl3795",
    "pla7397",
    "pla33810",
    "pla85900",
})

BKS: dict[str, int] = {name: v[1] for name, v in CATALOG.items()}
EWT: dict[str, str] = {name: v[3] for name, v in CATALOG.items()}


def url_for(name: str) -> str:
    if name not in CATALOG:
        raise KeyError(f"TSPLIB katalogunda yok: {name}")
    return URL_PATTERN.format(name=name)


def mirror_for(name: str) -> str:
    if name not in CATALOG:
        raise KeyError(f"TSPLIB katalogunda yok: {name}")
    return MIRROR_PATTERN.format(name=name)


def circuit_names() -> list[str]:
    """DEVRE alt kumesi, n'e gore artan sirali."""
    return sorted(CIRCUIT, key=lambda k: CATALOG[k][0])
