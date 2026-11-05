#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?              ____  __      __
#?             / __ \/ /___  / /_
#?            / /_/ / / __ \/ __/
#?           / ____/ / /_/ / /_
#?          /_/   /_/\____/\__/
#?
#?
#? Name:        Plot.py
#? Purpose:     Datasheet-style HTML report (plotly) for the BSIM6 power MOSFET model.
#?
#? Author:      Mohamed Gueni ( mohamedgueni@outlook.com)
#?
#? Created:     04/1/2023
#? -------------------------------------------------------------------------------
#? -------------------------------------------------------------------------------
import  html
import  os

import  plotly.graph_objects        as  go
import  plotly.io                   as  pio
#?----------------------------------------------------------------------------------------------------------------------------------------
_PALETTE                    = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777"]
#?----------------------------------------------------------------------------------------------------------------------------------------
def _fig(title, xlab, ylab, ylog=False, xlog=False):
    """Empty plotly figure with the report's common layout (title, axis labels, optional log axes)."""
    fig                     = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16)),
        xaxis_title=xlab, yaxis_title=ylab,
        template="plotly_white", height=430,
        margin=dict(l=70, r=20, t=50, b=55),
        legend=dict(orientation="v", x=1.0, xanchor="right", y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        colorway=_PALETTE,
    )
    if ylog:
        fig.update_yaxes(type="log", exponentformat="power")
    if xlog:
        fig.update_xaxes(type="log")
    return fig
#?----------------------------------------------------------------------------------------------------------------------------------------
def _lines(fig, df, x, y, group, label_fmt, dash_by=None):
    """Add one line trace per group of `df` (x vs y) labelled by `label_fmt(key)`; `dash_by(key)` picks the dash style."""
    for i, (key, sub) in enumerate(df.groupby(group, sort=True)):
        name                = label_fmt(key)
        dash                = None
        if dash_by is not None:
            dash            = dash_by(key)
        fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="lines", name=name,
                                 line=dict(width=2, dash=dash)))
    return fig
#?----------------------------------------------------------------------------------------------------------------------------------------
def build_figures(d):
    """d: dict of pandas DataFrames produced by main.characterize()."""
    figs                    = []

    f                       = _fig("Output characteristics  I<sub>D</sub>(V<sub>DS</sub>)", "V<sub>DS</sub> [V]", "I<sub>D</sub> [A]")
    _lines(f, d["output"], "Vds", "Id", ["Tc", "Vgs"],
           lambda k: f"V<sub>GS</sub>={k[1]:g} V, {k[0]:g} °C",
           dash_by=lambda k: "dash" if k[0] > 30 else None)
    figs.append(f)

    f                       = _fig("Transfer characteristics  (V<sub>DS</sub> = 10 V) — note the ZTC crossing",
                                   "V<sub>GS</sub> [V]", "I<sub>D</sub> [A]")
    _lines(f, d["transfer"], "Vgs", "Id", "Tc", lambda k: f"{k:g} °C")
    figs.append(f)

    f                       = _fig("Subthreshold  (V<sub>DS</sub> = 10 V)", "V<sub>GS</sub> [V]", "I<sub>D</sub> [A]", ylog=True)
    sub                     = d["transfer"][d["transfer"]["Id"] > 0]
    _lines(f, sub, "Vgs", "Id", "Tc", lambda k: f"{k:g} °C")
    figs.append(f)

    f                       = _fig("On-resistance vs junction temperature  (I<sub>D</sub> = 20 A)",
                                   "T<sub>j</sub> [°C]", "R<sub>DS(on)</sub> [mΩ]")
    _lines(f, d["rdson"], "Tc", "Rdson_mohm", "Vgs", lambda k: f"V<sub>GS</sub>={k:g} V")
    figs.append(f)

    f                       = _fig("Threshold voltage vs temperature  (I<sub>D</sub> = 250 µA, V<sub>DS</sub> = V<sub>GS</sub>)",
                                   "T<sub>j</sub> [°C]", "V<sub>GS(th)</sub> [V]")
    f.add_trace(go.Scatter(x=d["vth"]["Tc"], y=d["vth"]["Vth"], mode="lines+markers", name="V<sub>th</sub>"))
    figs.append(f)

    f                       = _fig("Capacitances  (V<sub>GS</sub> = 0, f → 0)", "V<sub>DS</sub> [V]", "C [pF]", ylog=True)
    c                       = d["caps"]
    for col in ("Ciss", "Coss", "Crss"):
        f.add_trace(go.Scatter(x=c["Vds"], y=c[col] * 1e12, mode="lines", name=col, line=dict(width=2)))
    figs.append(f)

    f                       = _fig("Gate charge  (V<sub>DD</sub> = 50 V, I<sub>D</sub> = 20 A)", "Q<sub>G</sub> [nC]", "V [V]")
    g                       = d["gate_charge"]
    f.add_trace(go.Scatter(x=g["Qg"] * 1e9, y=g["Vgs"], mode="lines", name="V<sub>GS</sub>", line=dict(width=2)))
    f.add_trace(go.Scatter(x=g["Qg"] * 1e9, y=g["Vds"] / 5.0, mode="lines", name="V<sub>DS</sub> / 5",
                           line=dict(width=2, dash="dot")))
    figs.append(f)

    f                       = _fig("Body diode (third quadrant, V<sub>GS</sub> = 0)", "V<sub>SD</sub> [V]", "I<sub>S</sub> [A]", ylog=True)
    _lines(f, d["diode"], "Vsd", "Is", "Tc", lambda k: f"{k:g} °C")
    figs.append(f)

    f                       = _fig("Off-state leakage and avalanche  (V<sub>GS</sub> = 0)", "V<sub>DS</sub> [V]", "I<sub>DSS</sub> [A]", ylog=True)
    _lines(f, d["off"], "Vds", "Id", "Tc", lambda k: f"{k:g} °C")
    figs.append(f)
    return figs
#?----------------------------------------------------------------------------------------------------------------------------------------
def write_report(d, summary, path, title="BSIM6 Power MOSFET Model"):
    """Write the standalone HTML report (summary table + all figures) to `path` and return the path."""
    figs                    = build_figures(d)
    rows                    = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in summary.items())
    divs                    = []
    for i, fig in enumerate(figs):
        divs.append('<div class="card">' + pio.to_html(
            fig, full_html=False, include_plotlyjs=("cdn" if i == 0 else False),
            config={"displaylogo": False}) + "</div>")
    page                    = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#f6f7f9; color:#111827; }}
 header {{ padding: 24px 32px 8px; }}
 h1 {{ margin: 0 0 4px; font-size: 22px; }}
 p.sub {{ margin: 0; color:#6b7280; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap:16px; padding:16px 32px 32px; }}
 .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:6px; overflow:hidden; }}
 table {{ border-collapse: collapse; margin: 12px 32px; background:#fff; border:1px solid #e5e7eb; border-radius:10px; }}
 td {{ padding: 6px 14px; border-bottom:1px solid #f0f1f3; font-variant-numeric: tabular-nums; }}
 td:first-child {{ color:#374151; }}
 @media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; padding:16px; }} table {{ margin: 12px 16px; }} }}
</style></head>
<body>
<header><h1>{html.escape(title)}</h1>
<p class="sub">BSIM6.0 charge-based channel + physics-based drift, JFET, avalanche, body diode and capacitance models.</p></header>
<table>{rows}</table>
<div class="grid">
{''.join(divs)}
</div></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path
#?----------------------------------------------------------------------------------------------------------------------------------------
