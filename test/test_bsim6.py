"""Physics sanity tests for the BSIM6 core and the power MOSFET wrapper.

Run:  python -m pytest test   (or)   python test/test_bsim6.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bsim6 import BSIM6Model, BSIM6Params
from power_mosfet import PowerMOSFET

M = BSIM6Model()
DEV = PowerMOSFET()
T25, T150 = 298.15, 423.15


def test_zero_current_at_zero_vds():
    for vgs in (0.0, 2.0, 5.0, 10.0):
        assert M.ids(vgs, 0.0) == 0.0


def test_source_drain_symmetry():
    # BSIM6 is symmetric: swapping D and S flips the current exactly
    for vg in (1.0, 3.0, 8.0):
        a = M.evaluate(1.3, vg, 0.2, 0.0).Ids
        b = M.evaluate(0.2, vg, 1.3, 0.0).Ids
        assert math.isclose(a, -b, rel_tol=1e-12)


def test_gummel_symmetry_odd_current():
    # Id(Vx) with Vd = +Vx, Vs = -Vx must be odd and smooth through Vds = 0
    vx = np.linspace(-0.05, 0.05, 21)
    i = np.array([M.evaluate(x, 5.0, -x, 0.0).Ids for x in vx])
    assert np.allclose(i, -i[::-1], rtol=1e-10, atol=1e-15)
    d2 = np.diff(i, 2)
    assert np.max(np.abs(d2)) < 1e-2 * np.max(np.abs(i))


def test_charge_conservation():
    for vgs, vds, vbs in ((0.0, 0.0, 0.0), (-3.0, 1.0, 0.0), (3.0, 0.5, -1.0), (10.0, 5.0, 0.0)):
        r = M.evaluate(vds, vgs, 0.0, vbs)
        assert abs(r.Qg + r.Qd + r.Qs + r.Qb) < 1e-20


def test_monotonic_and_smooth_output():
    vds = np.linspace(0.0, 10.0, 401)
    for vgs in (3.0, 5.0, 10.0):
        i = np.array([M.ids(vgs, v) for v in vds])
        gds = np.diff(i)
        assert np.all(gds > 0.0)                       # positive output conductance
        # strong inversion: smooth through Vdsat, no kink (near threshold the
        # current legitimately saturates within a few kT/q, so skip Vgs=3)
        if vgs >= 5.0:
            assert np.max(np.abs(np.diff(gds))) < 2e-3 * i.max()


def test_monotonic_transfer():
    vgs = np.linspace(0.0, 15.0, 301)
    i = np.array([M.ids(v, 1.0) for v in vgs])
    assert np.all(np.diff(i) > 0.0)


def test_strong_inversion_cgg_equals_cox():
    c = M.capacitances(10.0, 0.0)
    assert math.isclose(c["Cgg"], M.Cox * M.Weff * M.Leff, rel_tol=0.05)


def test_subthreshold_slope_physical():
    # SS = n * Vt * ln10 with n = 1 + Cdep/Cox  (~2.8 for this thick-oxide body)
    i1, i2 = M.ids(1.6, 1.0), M.ids(1.7, 1.0)
    ss = 100.0 / math.log10(i2 / i1)
    assert 60.0 < ss < 250.0


def test_threshold_decreases_with_temperature():
    assert M.vth(T=T150) < M.vth(T=T25)
    assert DEV.vth(T=T150) < DEV.vth(T=T25)


def test_pmos_mirror():
    # BSIM6 uses eta = ETAMOB/3 for holes vs ETAMOB/2 for electrons (1.139)
    p = BSIM6Model(BSIM6Params(TYPE=-1, ETAMOB=1.5))
    assert math.isclose(p.ids(-5.0, -1.0), -M.ids(5.0, 1.0), rel_tol=1e-12)


def test_rdson_positive_tempco():
    assert 1.4 < DEV.rdson(T=398.15) / DEV.rdson(T=T25) < 2.2


def test_zero_temperature_coefficient_point_exists():
    lo = DEV.id(3.5, 10.0, T150) - DEV.id(3.5, 10.0, T25)
    hi = DEV.id(8.0, 10.0, T150) - DEV.id(8.0, 10.0, T25)
    assert lo > 0.0 > hi


def test_breakdown_and_leakage():
    bv = DEV.breakdown_voltage()
    assert 90.0 < bv < 120.0
    assert DEV.id(0.0, 0.9 * bv, T25) < 1e-5
    assert DEV.id(0.0, 1.02 * bv, T25) > 100 * DEV.id(0.0, 0.9 * bv, T25)
    assert DEV.id(0.0, 50.0, T150) > 100 * DEV.id(0.0, 50.0, T25)    # IDSS(T)


def test_capacitances_nonlinear():
    c0, c50 = DEV.capacitances(0.5), DEV.capacitances(50.0)
    assert c50["Crss"] < 0.2 * c0["Crss"]     # Crss collapses as drift depletes
    assert c50["Coss"] < c0["Coss"]
    assert c50["Ciss"] > 10 * c50["Crss"]


def test_body_diode_conducts_in_third_quadrant():
    assert -DEV.id(0.0, -0.9, T25) > 1.0
    # diode forward voltage drops with temperature at low current
    assert -DEV.id(0.0, -0.5, T150) > -DEV.id(0.0, -0.5, T25)


def test_gate_charge_has_miller_plateau():
    qg, vg, vd, info = DEV.gate_charge()
    assert np.all(np.diff(qg) > -1e-15)
    assert 3.0 < info["Vplateau"] < 5.5
    assert info["Qgd"] > 0 and info["Qg"] > info["Qgs"] + info["Qgd"]


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}  {e}")
    sys.exit(1 if fails else 0)
