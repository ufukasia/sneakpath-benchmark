# -*- coding: utf-8 -*-
"""Bantli Paralel Greedy-Edge -- makalenin Sekil F0'i canli.

TEK SAYFA. Streamlit'in cok sayfali kipi (pages/ klasoru) bilerek
KULLANILMIYOR: tek bir gorunum icin kenar cubuguna gezinme listesi ve
dosya adindan turetilmis bir "app" girdisi koyuyor, ikisi de anlamsiz.

Uc panel: (a) kafes-hizali strip (theta=0), (b) seyrek strip taramasi
(theta_s), (c) secilen k ve stratejiyle uzamsal bolumleme. Turlarin TAMAMI
sunucu tarafinda, benchmark'ta kosan kodun ta kendisiyle hesaplanir
(server._compute_paradoks) -- burada istemci tarafi yaklasik kopya yoktur.

Kosum, karsilastirma tablolari, istatistik ve yonetim YEREL panelde kalir:
    python dashboard/server.py
Bu sayfa salt okunurdur; yikici bir islem sunmaz.

Calistirma:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st                                     # noqa: E402

from sp import backend, charts, meta, ui                   # noqa: E402

# Gorsellestirme butcesi: bunun ustundeki ornekler etkilesimli cizilemez
# (tur hesabi ve Plotly cizimi saniyelerce surer).
VIZ_MAX_N = 3000

ui.setup()
lg = ui.page("nav.viz", "🗺")
st.caption(meta.T("side.sub", lg)
           .replace("<b>", "**").replace("</b>", "**")
           .replace("<sub>", "").replace("</sub>", ""))

cols = backend.collections()
with st.sidebar:
    st.subheader(meta.T("grp.dataset", lg))
    cname = st.selectbox(meta.T("lbl.catFilter", lg), list(cols.keys()),
                         index=list(cols.keys()).index("vlsi") if "vlsi" in cols else 0)
    # Gorsellestirme butcesi: buyuk ornekler sayfayi saniyelerce kilitler.
    items = [it for it in cols[cname]
             if it.get("downloaded", True) and (it.get("n") or 0) <= VIZ_MAX_N]
    names = [it["name"] for it in items] or [it["name"] for it in cols[cname]]
    ds = st.selectbox(meta.T("lbl.dataset", lg), names,
                      index=names.index("xqf131") if "xqf131" in names else 0)

    st.subheader(meta.T("grp.parallel", lg))
    k = st.slider(meta.T("lbl.k", lg), 2, 16, 4)
    strategy = st.radio(meta.T("lbl.strategy", lg), ["kd", "band"], horizontal=True,
                        format_func=lambda s: meta.T("seg.kd" if s == "kd" else "seg.band", lg))
    opt = st.radio(meta.T("lbl.opt", lg), ["none", "vnd", "ils"], horizontal=True,
                   format_func=lambda s: meta.T({"none": "seg.none", "vnd": "seg.vnd",
                                                 "ils": "seg.ils"}[s], lg))


@st.cache_data(show_spinner=False, max_entries=32)
def _paradoks(ds: str, k: int, strategy: str, opt: str):
    return backend.compute_paradoks(ds, k=k, strategy=strategy, opt=opt)


with st.spinner(f"{ds} · k={k} · {strategy} · {opt}"):
    try:
        r = _paradoks(ds, k, strategy, opt)
    except Exception as exc:                       # noqa: BLE001
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

flat = r["coords"]
coords = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

# --- ust ozet -------------------------------------------------------------
c = st.columns(5)
c[0].metric(meta.T("top.dataset", lg), r["dataset"])
c[1].metric(meta.T("top.n", lg), r["n"])
c[2].metric("θ̂s", f"{r['theta_s']:.2f}°")
c[3].metric(meta.T("viz.build.makespan", lg).split("(")[0].strip(),
            f"{r['part']['makespan']:.0f}")
c[4].metric(meta.T("viz.build.balance", lg).split("=")[0].strip(),
            f"{r['part']['imbalance']:.3f}")

# --- uc panel -------------------------------------------------------------
# UC PANEL DE ayni iyilestirme katmanindan gecer. Sunucu VND/ILS secildiginde
# (a)/(b)'nin TEK turlarini da iyilestirip rep["tours_ab"] / rep["costs_ab"]
# icinde dondurur; bunlari yok sayip ham turu cizmek makalenin Adim 5'ini
# (cerceve yarisi: insada seyrek onde, derin aramada hizali onde) sayfada
# OKUNAMAZ kiliyordu -- yalniz (c) degisiyor gibi gorunuyordu.
rep = r.get("rep")
suffix = f" — {rep['mode'].upper()}" if rep else ""
tour_a = rep["tours_ab"][0] if rep else r["strip0"]["tour"]
cost_a = rep["costs_ab"][0] if rep else r["strip0"]["cost"]
tour_b = rep["tours_ab"][1] if rep else r["stripS"]["tour"]
cost_b = rep["costs_ab"][1] if rep else r["stripS"]["cost"]

p1, p2, p3 = st.columns(3)
with p1:
    st.plotly_chart(charts.tour_plot(coords, [tour_a],
                                     f"{meta.T('viz.ptA', lg)}{suffix} · {cost_a:.0f}",
                                     ["#800000"]), use_container_width=True)
with p2:
    st.plotly_chart(charts.tour_plot(coords, [tour_b],
                                     f"{meta.T('viz.ptBshort', lg)}{suffix} · {cost_b:.0f}",
                                     ["#003366"]), use_container_width=True)
with p3:
    part = rep or r["part"]
    st.plotly_chart(charts.tour_plot(coords, part["tours"],
                                     f"(c) k={r['part']['k']}{suffix} · "
                                     f"{meta.T('viz.build.makespan', lg).split('(')[0].strip()} "
                                     f"{part['makespan']:.0f}"),
                    use_container_width=True)

# Cerceve yarisi okumasi: iyilestirme sonrasi hizali (a) seyregi (b) gecti mi?
if rep:
    lead = "(a)" if cost_a < cost_b else "(b)"
    st.info(
        f"{meta.T('viz.ptA', lg)}: {r['strip0']['cost']:.0f} → {cost_a:.0f}  ·  "
        f"{meta.T('viz.ptBshort', lg)}: {r['stripS']['cost']:.0f} → {cost_b:.0f}  ·  "
        + (f"{rep['mode'].upper()} sonrası önde: {lead}" if lg == "tr"
           else f"ahead after {rep['mode'].upper()}: {lead}"))

# --- sayilar --------------------------------------------------------------
with st.expander(meta.T("viz.build.lens", lg).strip(" .:"), expanded=True):
    part = r["part"]

    def _lbl(key: str) -> str:
        """Sozluk metinleri panelde cumle icine gomulu geldigi icin bas/son
        noktalama tasiyor (". Parca basina nokta: "); markdown'a vermeden
        once temizlenir, yoksa ** yildizlari metne sizar."""
        return meta.T(key, lg).strip(" .:").strip()

    st.markdown(f"**{_lbl('viz.build.lens')}:** "
                + ", ".join(f"{v:.0f}" for v in part["lens"]))
    st.markdown(f"**{_lbl('viz.build.perPart')}:** "
                + ", ".join(str(v) for v in part["sizes"]))
    st.markdown(f"**{_lbl('viz.build.speed')}:** "
                f"{part['t_seq']:.3f} s / {part['t_par']:.3f} s")
    if rep:
        st.markdown(f"**{rep['mode'].upper()}** → "
                    + ", ".join(f"{v:.0f}" for v in rep["lens"]))

st.caption(meta.T("hint.parallel", lg).replace("<b>", "**").replace("</b>", "**"))
