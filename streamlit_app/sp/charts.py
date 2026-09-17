# -*- coding: utf-8 -*-
"""Tur cizimi (Plotly).

Sayfa tek gorunume indirildiginde basarim profili / kutu grafigi / Pareth /
kazanma matrisi cizen fonksiyonlar kullanimsiz kaldi ve silindi; onlarin
karsiligi yerel paneldeki Istatistik sekmesidir (dashboard/index.html).
"""
from __future__ import annotations

import plotly.graph_objects as go

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(font_size=12),
    legend=dict(font_size=11),
)


def tour_plot(coords, tours, title: str = "", colors=None) -> go.Figure:
    """Tur cizimi (F0 panelleri).

    `tours` bir ya da daha cok indeks dizisi; her alt tur kendi rengiyle
    KAPALI cizilir (son dugum ilkine baglanir -- tur kapali bir dongudur,
    acik birakmak alt tur uzunlugunu goz yanilgisiyle kisa gosterirdi).
    Eksenler esit olcekli: bir bolumlemenin geometrisi ancak en/boy orani
    korunursa dogru okunur.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c[0] for c in coords], y=[c[1] for c in coords],
                             mode="markers", name="",
                             marker=dict(size=2.6, color="#1f2a44"),
                             hoverinfo="skip", showlegend=False))
    pal = colors or ["#2563eb", "#b45309", "#0f766e", "#be123c", "#7c3aed",
                     "#0891b2", "#a16207", "#475569"]
    for i, t in enumerate(tours or []):
        if not t:
            continue
        loop = list(t) + [t[0]]
        fig.add_trace(go.Scatter(x=[coords[j][0] for j in loop],
                                 y=[coords[j][1] for j in loop],
                                 mode="lines", name=f"{i + 1}",
                                 line=dict(color=pal[i % len(pal)], width=1.1),
                                 hoverinfo="skip",
                                 showlegend=len(tours) > 1))
    fig.update_layout(**_LAYOUT, height=420, title=title,
                      xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                      yaxis=dict(visible=False))
    return fig
