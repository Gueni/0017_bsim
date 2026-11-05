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
import  dataclasses
import  inspect
import  re
import  sys
import  time
from    datetime                    import  datetime
from    pathlib                     import  Path

import  numpy                       as  np
import  pandas                      as  pd
from    rich                        import  box
from    rich.console                import  Console
from    rich.panel                  import  Panel
from    rich.rule                   import  Rule
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
CLEAR_DATA                  = True                                                    # True: delete old csv/html/log files before each run
OUTPUT_KINDS                = {"csv": ".csv", "html": ".html", "log": ".log"}          # sub-folder of the output dir -> file type it holds
STAMP_FMT                   = "%Y-%m-%d_%H-%M-%S"                                     # date-time suffix of every generated file
LOG_WIDTH                   = 140                                                     # character width of the tables in bsim.log
_FIELD                      = re.compile(r"^\s+(\w+)\s*:[^=#]*(?:=[^#]*)?(?:#\s*(.*))?$")
_SECTION                    = re.compile(r"^\s+#\s*-+\s*(.*?)\s*-*\s*$")
#?----------------------------------------------------------------------------------------------------------------------------------------
def characterize(dev, points=60):
    """Run every datasheet sweep on `dev` (`points` per sweep); return ({name: DataFrame}, gate-charge info)."""
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
    """Key datasheet numbers as {label: formatted value}, used by the console table and the HTML report."""
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
def _styled_table(title, *columns):
    """Rich table in the report's common style; each column is (header, style, justify)."""
    table                   = Table(title=title, title_style="bold magenta", box=box.ROUNDED,
                                    header_style="bold white on blue", border_style="blue",
                                    row_styles=["", "on grey11"], title_justify="left")
    for header, style, justify in columns:
        table.add_column(header, style=style, justify=justify)
    return table
#?----------------------------------------------------------------------------------------------------------------------------------------
def card_table(card):
    """Every field of a model card (BSIM6Params / PowerParams) with its value and the description from params.py.

    Section headers (`# --- geometry ---`) and the trailing comment of each field are read from the
    dataclass source, so the table always matches the card definition.
    """
    cls                     = type(card)
    table                   = _styled_table(f"{cls.__name__} — model card",
                                            ("Parameter", "bold cyan", "left"),
                                            ("Value", "bold green", "right"),
                                            ("Default", "dim", "right"),
                                            ("Description", "yellow", "left"))
    defaults                = {f.name: f.default for f in dataclasses.fields(cls)}
    for line in inspect.getsource(cls).splitlines()[1:]:
        if m := _SECTION.match(line):
            table.add_row(Text(m[1], style="bold magenta"), "", "", "", end_section=False)
        elif (m := _FIELD.match(line)) and m[1] in defaults:
            value           = getattr(card, m[1])
            changed         = value != defaults[m[1]]
            table.add_row("  " + m[1], Text(f"{value:g}", style="bold red" if changed else ""),
                          f"{defaults[m[1]]:g}", Text(m[2] or ""))                    # Text: keep "[m]" units literal, not markup
    return table
#?----------------------------------------------------------------------------------------------------------------------------------------
def derived_table(dev):
    """Bias-independent quantities derived from the cards (effective geometry, Cox, BV, drift resistance ...)."""
    ch                      = dev.ch
    table                   = _styled_table("Derived device quantities (25 °C)",
                                            ("Quantity", "bold cyan", "left"),
                                            ("Value", "bold green", "right"),
                                            ("Unit", "yellow", "left"))
    for name, value, unit in (
            ("Leff  (effective channel length)", ch.Leff * 1e6, "µm"),
            ("Weff  (effective channel width)", ch.Weff, "m"),
            ("Cox   (oxide capacitance)", ch.Cox * 1e3, "fF/µm²"),
            ("γ     (body-effect coefficient)", ch.gamma_v, "√V"),
            ("Vbi   (body-drift built-in voltage)", dev.v_bi(298.15), "V"),
            ("V(BR)DSS (avalanche breakdown)", dev.breakdown_voltage(), "V"),
            ("V_JFET (JFET pinch-off)", dev.v_jfet(), "V"),
            ("R_drift (drift resistance)", dev.r_drift(298.15) * 1e3, "mΩ"),
            ("I_sat,drift (drift velocity saturation)", dev.i_drift_sat(298.15), "A")):
        table.add_row(name, f"{value:.4g}", unit)
    return table
#?----------------------------------------------------------------------------------------------------------------------------------------
def files_table(paths):
    """Output files with their size (rows x columns for CSVs)."""
    table                   = _styled_table("Output files",
                                            ("File", "bold cyan", "left"),
                                            ("Contents", "bold green", "right"),
                                            ("Path", "dim", "left"))
    for p in paths:
        if p.suffix == ".csv":
            df              = pd.read_csv(p)
            what            = f"{len(df)} rows × {len(df.columns)} cols"
        else:
            what            = f"{p.stat().st_size / 1024:.0f} kB"
        table.add_row(p.name, what, str(p.parent))
    return table
#?----------------------------------------------------------------------------------------------------------------------------------------
def write_log(path, args, dev, summary, outputs, elapsed, started):
    """Write the run log: run info, both model cards, derived quantities, datasheet summary and output files."""
    with open(path, "w", encoding="utf-8") as fh:
        log                 = Console(file=fh, width=LOG_WIDTH, color_system=None, force_terminal=False,
                                      legacy_windows=False, emoji=False)
        log.print(Rule(f"BSIM6 Power MOSFET run — {started:%Y-%m-%d %H:%M:%S}"))
        info                = _styled_table("Run", ("Setting", "bold cyan", "left"), ("Value", "bold green", "left"))
        for k, v in (("Python", sys.version.split()[0]), ("Points per sweep", str(args.points)),
                     ("Output directory", str(args.out)), ("Run time", f"{elapsed:.1f} s")):
            info.add_row(k, v)
        for renderable in (info, card_table(dev.ch.p), card_table(dev.p), derived_table(dev),
                           summary_table(summary), files_table(outputs)):
            log.print(renderable)
            log.print()
    return path
#?----------------------------------------------------------------------------------------------------------------------------------------
def clear_data(out_dir, console=None):
    """Delete previously generated files from out_dir/csv, out_dir/html and out_dir/log.

    Only files of the matching type are removed (*.csv in csv/, *.html in html/, *.log in log/);
    the folders themselves and any other files are kept. Returns the number of files deleted.
    """
    removed                 = 0
    for sub, ext in OUTPUT_KINDS.items():
        folder              = Path(out_dir) / sub
        if not folder.is_dir():
            continue
        for f in folder.glob(f"*{ext}"):
            if f.is_file():
                f.unlink()
                removed    += 1
    if console is not None:
        console.print(f"[bold yellow]Cleared[/] {removed} old file{'s' if removed != 1 else ''} "
                      f"from [green]{', '.join(OUTPUT_KINDS)}[/] in {out_dir}")
    return removed
#?----------------------------------------------------------------------------------------------------------------------------------------
def main():
    """Command-line entry point: characterize the default device, write CSVs + HTML report, print the summary."""
    ap                      = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=int, default=60, help="points per sweep")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--clear", action="store_true", default=CLEAR_DATA,
                    help="delete old csv/html/log files before this run (or set CLEAR_DATA = True)")
    args                    = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")                     # Windows consoles
    except AttributeError:
        pass

    console                 = Console()
    if args.clear:
        clear_data(args.out, console)
    dev                   = PowerMOSFET(BSIM6Model(BSIM6Params()), PowerParams())

    t0                      = time.perf_counter()
    started                 = datetime.now()
    stamp                   = started.strftime(STAMP_FMT)                              # same stamp on every file of this run
    with console.status("[bold cyan]Characterizing power MOSFET…"):
        data, gq            = characterize(dev, args.points)
        summary             = summarize(dev, gq)
        csv_dir             = args.out / "csv"
        html_dir            = args.out / "html"
        log_dir             = args.out / "log"
        for d in (csv_dir, html_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)
        outputs             = []
        for name, df in data.items():
            path            = csv_dir / f"{name}_{stamp}.csv"
            df.to_csv(path, index=False)
            outputs.append(path)
        report              = Path(write_report(data, summary, str(html_dir / f"BSIM6_report_{stamp}.html")))
        outputs.append(report)
        log                 = write_log(log_dir / f"bsim_{stamp}.log", args, dev, summary, outputs,
                                        time.perf_counter() - t0, started)

    def rel(p):
        """Path relative to the project root (parent of the output dir) when possible, for a compact panel."""
        try:
            return str(Path(p).resolve().relative_to(args.out.resolve().parent))
        except ValueError:
            return str(p)

    console.print(summary_table(summary))
    console.print(Panel.fit(
        Text.assemble(("Run     ", "dim"), (stamp, "bold magenta"), ("\n", ""),
                      ("CSV     ", "dim"), (rel(csv_dir), "green"),
                      (f"  ({len(data)} files, {time.perf_counter() - t0:.1f} s)\n", "dim"),
                      ("Report  ", "dim"), (rel(report), "bold blue underline"), ("\n", ""),
                      ("Log     ", "dim"), (rel(log), "bold yellow")),
        border_style="green", title="[bold green]Done", title_align="left"))
#?----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#?----------------------------------------------------------------------------------------------------------------------------------------
