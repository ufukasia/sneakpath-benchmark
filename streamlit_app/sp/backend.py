# -*- coding: utf-8 -*-
"""dashboard/server.py koprusu.

Bu sayfa HTTP KATMANINI ATLAR: koleksiyon listesi ve F0 hesabi zaten
server.py icinde duz Python fonksiyonlari olarak duruyor. Yeniden yazmak
iki kopya ve iki dogruluk kaynagi demekti; burada modul olarak import
edilip dogrudan cagriliyorlar -- cizilen turlar benchmark'ta kosan kodun
ta kendisinden gelir.

server.py'nin `if __name__ == "__main__"` korumasi var, yani import etmek
sunucuyu BASLATMAZ; yalnizca proje kokunu sys.path'e ekler.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
for _p in (str(ROOT), str(DASH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import server as _srv          # noqa: E402  (yol ayarindan SONRA gelmeli)

#: {koleksiyon: [{name, n, bks, downloaded, has_result}, ...]}
collections = _srv._collections

#: (a)/(b)/(c) panellerinin tamami -- tek cagri, sunucu hesabi.
compute_paradoks = _srv._compute_paradoks
