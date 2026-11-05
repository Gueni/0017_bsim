# BSIM6 Power MOSFET Model

A Python implementation of the **BSIM6.0** charge-based MOSFET model, wrapped into a physics-based
**vertical power MOSFET** (VDMOS / trench). One command characterizes the device like a datasheet:
it writes CSV data, an interactive HTML report and a full run log.

## What is modelled

**BSIM6 channel** (`src/bsim6.py`, equation numbers follow the BSIM6.0 Technical Manual):

- Pinch-off potential with poly depletion, solved exactly instead of the closed-form fit
- Normalized source/drain inversion charges, so one current expression covers weak to strong inversion
- DIBL, body effect, slope factor, drain saturation voltage, smooth Vdseff
- Mobility degradation, velocity saturation, internal series resistance
- Output conductance (CLM, DIBL, SCBE), temperature scaling
- Charge-conserving intrinsic and overlap charges (Qg + Qd + Qs + Qb = 0)

**Power device** (`src/power.py`):

- Drift region: Arora mobility μ(N, T), velocity saturation (quasi-saturation)
- JFET shielding of the channel drain
- Avalanche breakdown, Miller multiplication and thermal-generation leakage (IDSS vs T)
- Body diode with series resistance and temperature scaling
- Nonlinear Ciss / Coss / Crss and gate charge, all derived from charges
- Steady-state electro-thermal self-heating

Not modelled: gate tunnelling, GIDL, noise, stress / well-proximity effects, binning, quantum-mechanical effects.

## Project layout

```
0017_bsim/
├── src/
│   ├── main.py      entry point: datasheet sweeps → CSV + HTML report + log
│   ├── params.py    model cards (BSIM6Params, PowerParams) and physical constants
│   ├── bsim6.py     BSIM6.0 channel model (DC + charges)
│   ├── power.py     vertical power MOSFET built around the BSIM6 channel
│   └── Plot.py      plotly HTML report
├── test/
│   └── test_bsim6.py  physics sanity tests (pytest or standalone)
├── data/            generated output (see below)
├── doc/             BSIM manuals (BSIM6.0 technical manual, BSIM4, BSIM3, BSIM-CMG, HSPICE)
└── assets/          reference CMOS SPICE model cards
```

## Requirements

Python 3.10 or newer, plus:

```
pip install numpy pandas scipy plotly rich
```

Tested with Python 3.14, numpy 2.5, pandas 3.0, scipy 1.18, plotly 6.8, rich 15.

## Usage

On Windows, use `py` if `python` is not on your PATH.

```
python src/main.py                   # default sweep resolution (60 points per sweep)
python src/main.py --points 120      # finer sweeps
python src/main.py --out D:/results  # write somewhere other than ./data
python src/main.py --clear           # delete old csv/html/log files first, keep only this run
```

To clear old files on every run, set `CLEAR_DATA = True` at the top of `src/main.py`. Clearing only
removes `*.csv` from `data/csv`, `*.html` from `data/html` and `*.log` from `data/log` (test logs included).
Other files and the folders themselves are left alone.

The terminal shows the datasheet summary as a table, then the paths of the generated files.

### Generated files

Every run stamps all of its files with the same start date-time (`YYYY-MM-DD_HH-MM-SS`), so earlier
results are never overwritten:

```
data/
├── csv/   output_<stamp>.csv, transfer_…, rdson_…, vth_…, caps_…, gate_charge_…, diode_…, off_…
├── html/  power_mosfet_report_<stamp>.html     interactive plots + summary table
└── log/   bsim_<stamp>.log                     run info, both model cards (value, default, description),
           test_bsim6_<stamp>.log               derived quantities, datasheet summary, file list
```

| CSV | Sweep |
|---|---|
| `output` | Id(Vds) at several Vgs, 25 °C and 125 °C |
| `transfer` | Id(Vgs) at Vds = 10 V, −40 … 175 °C |
| `rdson` | RDS(on) vs junction temperature at Id = 20 A |
| `vth` | VGS(th) vs temperature (Id = 250 µA, Vds = Vgs) |
| `caps` | Ciss, Coss, Crss vs Vds |
| `gate_charge` | Qg, Vgs, Vds along the gate-charge curve (50 V, 20 A) |
| `diode` | Body-diode Is(Vsd) |
| `off` | Off-state leakage and avalanche |

## Changing the device

All parameters live in `src/params.py`. Edit the defaults there, or override them in code:

```python
from bsim6  import BSIM6Model
from params import BSIM6Params, PowerParams
from power  import PowerMOSFET

dev = PowerMOSFET(BSIM6Model(BSIM6Params(TOX=40e-9, NDEP=2e23)),
                  PowerParams(ND_DRIFT=3e21, T_DRIFT=8e-6))

dev.id(Vgs=10.0, Vds=1.0, T=298.15)   # drain current [A]
dev.rdson(T=398.15)                   # on-resistance at 125 °C [ohm]
dev.capacitances(50.0)                # {'Ciss', 'Coss', 'Crss', ...} [F]
```

Values you change from the default are marked in the model-card tables of the run log.

## Tests

```
python test/test_bsim6.py    # rich results table + data/log/test_bsim6_<stamp>.log
python -m pytest test        # plain pytest
```

The 16 tests check physical behaviour, for example:

- source/drain symmetry and Gummel symmetry
- charge conservation
- monotonic and kink-free output and transfer curves
- Cgg → Cox in strong inversion
- a realistic subthreshold slope
- PMOS/NMOS mirroring
- Vth and RDS(on) temperature trends and the zero-temperature-coefficient crossing
- breakdown and leakage
- nonlinear capacitances
- body-diode conduction
- the Miller plateau in the gate-charge curve

## Author

Mohamed Gueni (mohamedgueni@outlook.com)
