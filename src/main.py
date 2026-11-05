#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?                              _
#?             ____ ___  ____ _(_)___
#?            / __ `__ \/ __ `/ / __ \
#?           / / / / / / /_/ / / / / /
#?          /_/ /_/ /_/\__,_/_/_/ /_/
#?
#?
#? Name:        main.py
#? Purpose:     Characterize the BSIM6-based power MOSFET and write a datasheet-style report
#?              (CSV data + interactive HTML plots) to ../data.
#?              Start it with `py main.py [--points N] [--out DIR]`.
#?
#? Author:      Mohamed Gueni ( mohamedgueni@outlook.com)
#?
#? Created:     21/05/2025
#? -------------------------------------------------------------------------------
#? -------------------------------------------------------------------------------
import  argparse
import  sys
import  time
from    pathlib                     import  Path

import  numpy                       as  np
import  pandas                      as  pd
from    rich                        import  box
from    rich.console                import  Console
from    rich.panel                  import  Panel
from    rich.table                  import  Table
from    rich.text                   import  Text
from    scipy.optimize              import  brentq

from    bsim6                       import  BSIM6Model
from    params                      import  BSIM6Params, PowerParams
from    power                       import  PowerMOSFET
from    Plot                        import  write_report
#?----------------------------------------------------------------------------------------------------------------------------------------
DATA_DIR                    = Path(__file__).resolve().parent.parent / "data"
K0                          = 273.15
#?----------------------------------------------------------------------------------------------------------------------------------------
def characterize(dev, points=60):
    d                       = {}

    rows                    = []
    for Tc in (25.0, 125.0):
        for vgs in (4.0, 4.5, 5.0, 6.0, 8.0, 10.0):
            for vds in np.linspace(0.0, 10.0, points):
                rows.append((Tc, vgs, vds, dev.id(vgs, vds, Tc + K0)))
    d["output"]             = pd.DataFrame(rows, columns=["Tc", "Vgs", "Vds", "Id"])

    rows                    = []
    for Tc in (-40.0, 25.0, 100.0, 175.0):
        for vgs in np.linspace(0.0, 8.0, 2 * points):
            rows.append((Tc, vgs, dev.id(vgs, 10.0, Tc + K0)))
    d["transfer"]           = pd.DataFrame(rows, columns=["Tc", "Vgs", "Id"])

    rows                    = []
    for vgs in (4.5, 6.0, 10.0):
        for Tc in np.linspace(-40.0, 175.0, 24):
            rows.append((vgs, Tc, 1e3 * dev.rdson(vgs, 20.0, Tc + K0)))
    d["rdson"]              = pd.DataFrame(rows, columns=["Vgs", "Tc", "Rdson_mohm"])

    d["vth"]                = pd.DataFrame([(Tc, dev.vth(T=Tc + K0)) for Tc in np.linspace(-40.0, 175.0, 24)],
                                           columns=["Tc", "Vth"])

    rows                    = []
    for vds in np.concatenate([np.linspace(0.0, 2.0, 21)[:-1], np.linspace(2.0, 100.0, 50)]):
        c                   = dev.capacitances(vds)
        rows.append((vds, c["Ciss"], c["Coss"], c["Crss"]))
    d["caps"]               = pd.DataFrame(rows, columns=["Vds", "Ciss", "Coss", "Crss"])

    qg, vg, vd, gq          = dev.gate_charge(Vdd=50.0, Iload=20.0, Vg_max=10.0)
    d["gate_charge"]        = pd.DataFrame({"Qg": qg, "Vgs": vg, "Vds": vd})

    rows                    = []
    for Tc in (25.0, 125.0, 175.0):
        for vsd in np.linspace(0.3, 1.3, points):
            rows.append((Tc, vsd, -dev.id(0.0, -vsd, Tc + K0)))
    d["diode"]              = pd.DataFrame(rows, columns=["Tc", "Vsd", "Is"])

    rows                    = []
    for Tc in (25.0, 100.0, 150.0):
        for vds in np.linspace(1.0, 1.08 * dev.breakdown_voltage(), points):
            rows.append((Tc, vds, dev.id(0.0, vds, Tc + K0)))
    d["off"]                = pd.DataFrame(rows, columns=["Tc", "Vds", "Id"])
    return d, gq
#?----------------------------------------------------------------------------------------------------------------------------------------
def summarize(dev, gq):
    c50                     = dev.capacitances(50.0)
    vsd                     = brentq(lambda v: -dev.id(0.0, -v, 298.15) - 20.0, 0.2, 3.0)
    return {
        "V(BR)DSS (25 °C)": f"{dev.breakdown_voltage():.1f} V",
        "VGS(th) @ 250 µA, 25 °C / 150 °C": f"{dev.vth():.2f} V / {dev.vth(T=423.15):.2f} V",
        "RDS(on) @ VGS=10 V, ID=20 A, 25 °C": f"{1e3 * dev.rdson():.2f} mΩ",
        "RDS(on) @ VGS=10 V, ID=20 A, 125 °C": f"{1e3 * dev.rdson(T=398.15):.2f} mΩ",
        "  of which drift region (25 °C)": f"{1e3 * dev.r_drift(298.15):.2f} mΩ",
        "IDSS @ 80 V, 25 °C / 150 °C": f"{dev.id(0, 80, 298.15):.2e} A / {dev.id(0, 80, 423.15):.2e} A",
        "Ciss / Coss / Crss @ VDS=50 V": f"{c50['Ciss']*1e12:.0f} / {c50['Coss']*1e12:.0f} / {c50['Crss']*1e12:.1f} pF",
        "Qg / Qgs / Qgd (50 V, 20 A, 10 V)": f"{gq['Qg']*1e9:.1f} / {gq['Qgs']*1e9:.1f} / {gq['Qgd']*1e9:.1f} nC",
        "Miller plateau": f"{gq['Vplateau']:.2f} V",
        "VSD @ IS=20 A (body diode)": f"{vsd:.3f} V",
    }
#?----------------------------------------------------------------------------------------------------------------------------------------
def summary_table(summary):
    """Datasheet summary as a rich table: parameter | test conditions | value."""
    table                   = Table(title="BSIM6 Power MOSFET — Datasheet Summary", title_style="bold magenta",
                                    box=box.ROUNDED, header_style="bold white on blue", border_style="blue",
                                    row_styles=["", "on grey11"], expand=False)
    table.add_column("Parameter", style="bold cyan", no_wrap=True)
    table.add_column("Conditions", style="yellow")
    table.add_column("Value", style="bold green", justify="right", no_wrap=True)
    for key, value in summary.items():
        sub                 = key.startswith(" ")                                      # indented key = sub-item of the row above
        key                 = key.strip()
        if " @ " in key:
            param, cond     = key.split(" @ ", 1)
        elif key.endswith(")") and " (" in key:
            param, cond     = key[:-1].rsplit(" (", 1)
        else:
            param, cond     = key, ""
        if sub:
            param           = Text("  └ " + param.removeprefix("of which "), style="cyan")
        table.add_row(param, cond, value)
    return table
#?----------------------------------------------------------------------------------------------------------------------------------------
def main():
    ap                      = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=60, help="points per sweep")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args                    = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")                     # Windows consoles
    except AttributeError:
        pass

    console                 = Console()
    dev                     = PowerMOSFET(BSIM6Model(BSIM6Params()), PowerParams())

    t0                      = time.perf_counter()
    with console.status("[bold cyan]Characterizing power MOSFET…"):
        data, gq            = characterize(dev, args.points)
        summary             = summarize(dev, gq)
        args.out.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.to_csv(args.out / f"{name}.csv", index=False)
        report              = write_report(data, summary, str(args.out / "power_mosfet_report.html"))

    console.print(summary_table(summary))
    console.print(Panel.fit(
        Text.assemble(("CSV data + report  ", "dim"), (str(args.out), "green"),
                      (f"  ({time.perf_counter() - t0:.1f} s)\n", "dim"),
                      ("Open               ", "dim"), (report, "bold blue underline")),
        border_style="green", title="[bold green]Done", title_align="left"))
#?----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#?----------------------------------------------------------------------------------------------------------------------------------------
