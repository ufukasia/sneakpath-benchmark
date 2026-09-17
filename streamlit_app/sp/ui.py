# -*- coding: utf-8 -*-
"""Sayfalar arasi ortak arayuz parcalari."""
from __future__ import annotations

import streamlit as st

from .meta import DEFAULT_LANG, LANGS, T

_LANG_KEY = "sp_lang"


def lang() -> str:
    """Secili dil. Panelle ayni varsayilan (EN) ve ayni oturum sozlesmesi."""
    return st.session_state.get(_LANG_KEY, DEFAULT_LANG)


def sidebar_lang() -> str:
    """Kenar cubugunda dil secici; secim tum sayfalarda paylasilir."""
    cur = lang()
    with st.sidebar:
        pick = st.radio("Language / Dil", LANGS, horizontal=True,
                        index=LANGS.index(cur) if cur in LANGS else 0,
                        format_func=str.upper, key="_lang_radio")
    if pick != cur:
        st.session_state[_LANG_KEY] = pick
        st.rerun()
    return pick


def page(title_key: str, icon: str = "") -> str:
    """Sayfa basligi + dil secici; secili dili doner."""
    lg = sidebar_lang()
    st.title((icon + " " if icon else "") + T(title_key, lg))
    return lg


def setup(title: str = "Banded Parallel GE") -> None:
    st.set_page_config(page_title=title, layout="wide",
                       initial_sidebar_state="expanded")
