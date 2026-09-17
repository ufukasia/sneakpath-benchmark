# -*- coding: utf-8 -*-
"""TR/EN sozluk.

Sozlugun TEK KAYNAGI dashboard/index.html'dir; buraya `extract_meta.js` ile
JSON olarak kopyalanir. Paneldeki bir metin degisince tek komut yeter:

    node streamlit_app/extract_meta.js

Boylece bu sayfa ile yerel panel ayni metinleri kullanir; ikisini elle
senkron tutmak gerekmez.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_META_PATH = Path(__file__).resolve().parent / "meta.json"

LANGS = ("en", "tr")
DEFAULT_LANG = "en"          # panelle ayni varsayilan


@lru_cache(maxsize=1)
def _dict() -> dict:
    with _META_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["I18N"]


def T(key: str, lang: str = DEFAULT_LANG) -> str:
    """Sozluk erisimi. index.html'deki T() ile AYNI dusme sirasi:
    secili dil -> en -> anahtarin kendisi.

    TR sozlugu henuz eksiktir (223 anahtar); eksik anahtarda EN metin
    gorunur, bos string degil -- arayuz hicbir yerde bos kalmaz."""
    d = _dict()
    cur = d.get(lang) or {}
    if key in cur:
        return cur[key]
    if key in d["en"]:
        return d["en"][key]
    return key


def Tf(key: str, lang: str = DEFAULT_LANG, **reps) -> str:
    """T + {yer_tutucu} degistirme (index.html'deki Tf ile ayni sozlesme)."""
    s = T(key, lang)
    for k, v in reps.items():
        s = s.replace("{" + k + "}", str(v))
    return s
