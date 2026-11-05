#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?                                                                    ____     __
#?              ____  ____ _      _____  _____  ____ ___  ____  _____/ __/__  / /_
#?             / __ \/ __ \ | /| / / _ \/ ___/ / __ `__ \/ __ \/ ___/ /_/ _ \/ __/
#?            / /_/ / /_/ / |/ |/ /  __/ /    / / / / / / /_/ (__  ) __/  __/ /_
#?           / .___/\____/|__/|__/\___/_/____/_/ /_/ /_/\____/____/_/  \___/\__/
#?          /_/                        /_____/
#?
#?
#? Name:        power_mosfet.py
#? Purpose:     Physics-based vertical power MOSFET built around the BSIM6 channel.
#?
#? Author:      Mohamed Gueni ( mohamedgueni@outlook.com)
#?
#? Created:     04/1/2023
#? -------------------------------------------------------------------------------
#? -------------------------------------------------------------------------------
"""
A vertical (VDMOS / trench) power MOSFET is a short lateral MOS channel in
series with a lightly doped drift region that holds the blocking voltage.
BSIM6 models the channel; this module adds the parts BSIM does not cover:

  D o--[R_pkg_d]--[drift: R(T), quasi-saturation]--+-- JFET shielding --[ BSIM6 ]--[R_pkg_s]--o S
                                                   |                       |
                                         body diode / junction cap         G (Cgs_ov, Cgd(Vdg))

  * Drift resistance from doping and thickness, with Arora electron mobility
    mu_n(N, T) and velocity-saturated (quasi-saturation) conduction.
  * JFET shielding: the channel drain potential saturates at the JFET
    pinch-off voltage, so the channel does not see the full blocking voltage.
  * Avalanche: breakdown voltage from the critical field of the drift doping
    (non-punch-through / punch-through), Miller multiplication, and thermal
    generation leakage  I_gen = q ni W_dep A / (2 tau)  -> IDSS(T).
  * Body diode: Shockley diode with series resistance and SPICE T-scaling.
  * Capacitances from charges: MOS gate-drain over the drift (oxide in series
    with depletion -> strongly nonlinear Crss), body-drift junction (Coss),
    source overlap + BSIM6 intrinsic charges (Ciss).  Because everything is
    charge-based, gate charge Qg is an exact state function.
  * Electro-thermal: steady-state self-heating Tj = Ta + Rth * P.
"""
import  math
from    dataclasses                 import  dataclass, replace

from    scipy.optimize              import  brentq
from    scipy.special               import  wrightomega

from    bsim6                       import  BSIM6Model, BSIM6Params, Q, KB, EPS0
#?----------------------------------------------------------------------------------------------------------------------------------------
EPS_SI                      = 11.9 * EPS0
#?----------------------------------------------------------------------------------------------------------------------------------------
def mobility_arora(N_m3, T):
    """Electron mobility in Si [m^2/Vs] vs doping [m^-3] and T [K] (Arora 1982)."""
    N                       = N_m3 * 1e-6
    Tn                      = T / 300.0
    mu_min                  = 88.0 * Tn ** -0.57
    mu_0                    = 7.4e8 * T ** -2.33
    Nref                    = 1.26e17 * Tn ** 2.4
    alpha                   = 0.88 * Tn ** -0.146
    return 1e-4 * (mu_min + mu_0 / (1.0 + (N / Nref) ** alpha))
#?----------------------------------------------------------------------------------------------------------------------------------------
def critical_field(N_m3):
    """Avalanche critical field in Si [V/m] (Baliga: 4010 * N^(1/8) V/cm)."""
    return 4010.0 * (N_m3 * 1e-6) ** 0.125 * 100.0
#?----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class PowerParams:
    # --- drift region -----------------------------------------------------------
    ND_DRIFT   : float      = 4.0e21                                                   # drift doping [m^-3]  (4e15 cm^-3)
    T_DRIFT    : float      = 6.5e-6                                                   # drift thickness [m]
    A_ACTIVE   : float      = 2.0e-5                                                   # active area [m^2] (20 mm^2)
    K_SPREAD   : float      = 1.3                                                      # current spreading + JFET resistance factor
    VSAT_DRIFT : float      = 1.0e5                                                    # electron saturation velocity [m/s]
    W_JFET     : float      = 3.0e-6                                                   # JFET neck width between p-bodies [m]
    # --- package / metallization ------------------------------------------------
    RS_PKG     : float      = 0.4e-3                                                   # source (bond wires, metal) [ohm]
    RD_PKG     : float      = 0.3e-3                                                   # drain (substrate, lead frame) [ohm]
    TCR_PKG    : float      = 3.9e-3                                                   # metal temperature coefficient [1/K]
    RG         : float      = 1.2                                                      # internal gate resistance [ohm] (informative)
    # --- capacitance geometry ---------------------------------------------------
    A_GD       : float      = 1.0e-6                                                   # gate-over-drift (neck) area [m^2]
    A_J        : float      = 1.2e-5                                                   # body-drift junction area [m^2]
    CGS_OV     : float      = 2.2e-9                                                   # gate-source overlap / source-poly cap [F]
    FC         : float      = 0.5                                                      # forward-bias depletion-cap linearization
    # --- avalanche / leakage ----------------------------------------------------
    TC_BV      : float      = 1.0e-3                                                   # relative BV temperature coefficient [1/K]
    M_EXP      : float      = 5.0                                                      # Miller multiplication exponent
    TAU_G      : float      = 1.0e-6                                                   # generation lifetime [s]
    # --- body diode ---------------------------------------------------------------
    IS_D       : float      = 1.0e-11                                                  # saturation current at TNOM [A]
    N_D        : float      = 1.05                                                     # ideality factor
    RS_D       : float      = 2.5e-3                                                   # series resistance [ohm]
    XTI_D      : float      = 3.0
    EG_D       : float      = 1.12                                                     # [eV]
    # --- thermal ------------------------------------------------------------------
    RTH        : float      = 0.6                                                      # junction-to-ambient/heatsink [K/W]
#?----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    Id      : float                                                                    # total drain current [A]
    Ich     : float                                                                    # channel current [A]
    Idiode  : float                                                                    # body diode current (source->drain) [A]
    Vgs_int : float                                                                    # internal channel gate-source voltage [V]
    Vds_ch  : float                                                                    # voltage across the BSIM6 channel [V]
    Vdx     : float                                                                    # internal drain (top of drift) [V]
    Vdrift  : float                                                                    # voltage across the drift region [V]
    M       : float                                                                    # avalanche multiplication
    T       : float                                                                    # junction temperature [K]
#?----------------------------------------------------------------------------------------------------------------------------------------
class PowerMOSFET:
    def __init__(self, channel=None, params=None, **overrides):
        self.ch             = channel or BSIM6Model(BSIM6Params())
        self.p              = replace(params or PowerParams(), **overrides)
        self.TNOM           = self.ch.p.TNOM

    # ------------------------------------------------------------ physics bits
    def r_drift(self, T):
        p                   = self.p
        mu                  = mobility_arora(p.ND_DRIFT, T)
        return p.K_SPREAD * p.T_DRIFT / (Q * p.ND_DRIFT * mu * p.A_ACTIVE)

    def _v_drift(self, I, T):
        """Voltage across the drift region carrying current I (odd in I).

        Velocity-saturated conduction  I = V / (R0 (1 + |V|/Vc))
        -> V = I R0 / (1 - |I| R0 / Vc),   Vc = vsat * T_drift / mu.
        """
        p                   = self.p
        R0                  = self.r_drift(T)
        Vc                  = p.VSAT_DRIFT * p.T_DRIFT / mobility_arora(p.ND_DRIFT, T)
        x                   = abs(I) * R0 / Vc
        if x >= 1.0:
            return math.copysign(math.inf, I)
        return I * R0 / (1.0 - x)

    def i_drift_sat(self, T):
        p                   = self.p
        Vc                  = p.VSAT_DRIFT * p.T_DRIFT / mobility_arora(p.ND_DRIFT, T)
        return Vc / self.r_drift(T)

    def v_jfet(self):
        """JFET pinch-off voltage of the neck between adjacent p-bodies."""
        p                   = self.p
        Vbi                 = self.v_bi(self.TNOM)
        return max(Q * p.ND_DRIFT * (p.W_JFET / 2.0) ** 2 / (2.0 * EPS_SI) - Vbi, 0.5)

    def _shield(self, V):
        """Channel drain voltage saturates at the JFET pinch-off (odd, smooth)."""
        Vp                  = self.v_jfet()
        m                   = 4.0
        return V / (1.0 + abs(V / Vp) ** m) ** (1.0 / m)

    def v_bi(self, T):
        ni                  = self.ch.ni(T)
        return KB * T / Q * math.log(self.ch.p.NDEP * self.p.ND_DRIFT / ni ** 2)

    def breakdown_voltage(self, T=None):
        """Avalanche BV from the drift doping and thickness [V]."""
        T                   = self.TNOM if T is None else T
        p                   = self.p
        Ec                  = critical_field(p.ND_DRIFT)
        bv_npt              = EPS_SI * Ec ** 2 / (2.0 * Q * p.ND_DRIFT)
        w_npt               = 2.0 * bv_npt / Ec
        if w_npt > p.T_DRIFT:                                                          # punch-through design
            bv              = Ec * p.T_DRIFT - Q * p.ND_DRIFT * p.T_DRIFT ** 2 / (2.0 * EPS_SI)
        else:
            bv              = bv_npt
        return bv * (1.0 + p.TC_BV * (T - self.TNOM))

    def _multiplication(self, V, T):
        if V <= 0.0:
            return 1.0
        d                   = 1.0 - (V / self.breakdown_voltage(T)) ** self.p.M_EXP
        return 1.0 / max(d, 1e-4)                                                      # cap M at 1e4 beyond BV

    def _w_dep(self, V, T):
        p                   = self.p
        w                   = math.sqrt(2.0 * EPS_SI * max(self.v_bi(T) + V, 0.0) / (Q * p.ND_DRIFT))
        return min(w, p.T_DRIFT)

    def i_leak(self, Vds, T):
        """Thermal generation current of the reverse-biased body-drift junction."""
        if Vds <= 0.0:
            return 0.0
        p                   = self.p
        ni                  = self.ch.ni(T)
        return Q * ni * (self._w_dep(Vds, T) - self._w_dep(0.0, T)) * p.A_J / (2.0 * p.TAU_G)

    def i_diode(self, Vsd, T):
        """Body diode forward current (source -> drain) with series R [A]."""
        p                   = self.p
        r                   = T / self.TNOM
        nvt                 = p.N_D * KB * T / Q
        Is                  = p.IS_D * r ** (p.XTI_D / p.N_D) * math.exp((r - 1.0) * p.EG_D / nvt)
        Rs                  = p.RS_D * (1.0 + p.TCR_PKG * (T - self.TNOM))
        if Vsd < -5.0 * nvt:
            return -Is
        z                   = math.log(Is * Rs / nvt) + (Vsd + Is * Rs) / nvt
        return nvt / Rs * float(wrightomega(z).real) - Is

    def _r_pkg(self, T):
        f                   = 1.0 + self.p.TCR_PKG * (T - self.TNOM)
        return self.p.RS_PKG * f, self.p.RD_PKG * f

    # ------------------------------------------------------------- DC solver
    def _internal(self, I, Vgs, Vds, T):
        Rs, Rd              = self._r_pkg(T)
        Vdr                 = self._v_drift(I, T)
        Vdx                 = Vds - I * Rd - Vdr                                       # top of drift region wrt external S
        Vgi                 = Vgs - I * Rs
        Vch                 = self._shield(Vdx - I * Rs)
        return Vgi, Vch, Vdx, Vdr

    def solve(self, Vgs, Vds, T=298.15):
        """Isothermal DC operating point."""
        def g(I):
            Vgi, Vch, _, _  = self._internal(I, Vgs, Vds, T)
            if math.isinf(Vch):
                return -I
            return self.ch.ids(Vgi, Vch, 0.0, T) - I

        I0                  = self.ch.ids(Vgs, self._shield(Vds), 0.0, T)
        Isat                = self.i_drift_sat(T) * (1.0 - 1e-9)
        lim                 = min(abs(I0), Isat)
        if lim < 1e-30:
            Ich             = I0
        else:
            a, b            = (0.0, lim) if I0 > 0 else (-lim, 0.0)
            ga, gb          = g(a), g(b)
            if ga * gb > 0.0:
                # parasitic drops below floating-point resolution: no correction
                Ich         = b if abs(gb) < abs(ga) else a
            else:
                Ich         = brentq(g, a, b, xtol=1e-15, rtol=1e-12, maxiter=200)
        Vgi, Vch, Vdx, Vdr  = self._internal(Ich, Vgs, Vds, T)
        Idio                = self.i_diode(-Vds, T)
        M                   = self._multiplication(Vds, T)
        Id                  = (Ich + self.i_leak(Vds, T)) * M - Idio
        return OperatingPoint(Id=Id, Ich=Ich, Idiode=Idio, Vgs_int=Vgi, Vds_ch=Vch,
                              Vdx=Vdx, Vdrift=Vdr, M=M, T=T)

    def id(self, Vgs, Vds, T=298.15):
        return self.solve(Vgs, Vds, T).Id

    def solve_electrothermal(self, Vgs, Vds, Ta=298.15, Rth=None, tol=1e-4, maxiter=100):
        """Steady-state self-heating: Tj = Ta + Rth * Id * Vds.

        Raises RuntimeError on thermal runaway (no stable fixed point).
        """
        Rth                 = self.p.RTH if Rth is None else Rth
        Tj                  = Ta
        for _ in range(maxiter):
            op              = self.solve(Vgs, Vds, Tj)
            Tn              = Ta + Rth * op.Id * Vds
            if Tn > 800.0:
                raise RuntimeError("thermal runaway (Tj > 800 K)")
            if abs(Tn - Tj) < tol:
                return op
            Tj              = Tj + 0.7 * (Tn - Tj)
        raise RuntimeError("electro-thermal iteration did not converge")

    def rdson(self, Vgs=10.0, Id=20.0, T=298.15):
        """On-resistance at a given drain current [ohm]."""
        Vds                 = brentq(lambda v: self.id(Vgs, v, T) - Id, 1e-6, 20.0, xtol=1e-12)
        return Vds / Id

    def vth(self, Id=250e-6, T=298.15):
        """Datasheet threshold: Vgs at Id with Vds = Vgs."""
        return brentq(lambda v: self.id(v, v, T) - Id, 0.1, 15.0, xtol=1e-9)

    # -------------------------------------------------------------- charges
    def _q_mos_depl(self, V, Cox):
        """Charge per area of MOS (oxide + n-drift depletion) at drain-gate V."""
        p                   = self.p
        if V <= 0.0:
            return Cox * V                                                             # accumulation
        a                   = 1.0 / (2.0 * Q * EPS_SI * p.ND_DRIFT)
        b                   = 1.0 / Cox
        Qrt                 = Q * p.ND_DRIFT * p.T_DRIFT                               # depletion reaches the substrate
        Vrt                 = b * Qrt + a * Qrt ** 2
        if V <= Vrt:
            return 2.0 * V / (b + math.sqrt(b * b + 4.0 * a * V))
        return Qrt + (V - Vrt) / (b + p.T_DRIFT / EPS_SI)

    def _q_junction(self, Vj):
        """Depletion charge per area of the body-drift junction at reverse V."""
        p                   = self.p
        k                   = 2.0 * Q * EPS_SI * p.ND_DRIFT
        Qrt                 = Q * p.ND_DRIFT * p.T_DRIFT
        Vrt                 = Qrt ** 2 / k
        if Vj <= Vrt:
            return math.sqrt(k * Vj)
        return Qrt + (Vj - Vrt) * EPS_SI / p.T_DRIFT

    def _qj(self, Vds, T):
        Vbi                 = self.v_bi(T)
        Vj0                 = (1.0 - self.p.FC) * Vbi
        Vj                  = Vbi + Vds
        if Vj >= Vj0:
            q               = self._q_junction(Vj)
        else:                                                                          # forward bias: constant-C extension
            c0              = math.sqrt(2.0 * Q * EPS_SI * self.p.ND_DRIFT / Vj0) / 2.0
            q               = self._q_junction(Vj0) + c0 * (Vj - Vj0)
        return self.p.A_J * (q - self._q_junction(Vbi))

    def charges(self, Vgs, Vds, T=298.15, op=None):
        """Terminal charges (Qg, Qd) [C]; Qs = -(Qg + Qd)."""
        op                  = op or self.solve(Vgs, Vds, T)
        r                   = self.ch.evaluate(op.Vds_ch, op.Vgs_int, 0.0, 0.0, T)
        Qgd                 = self.p.A_GD * self._q_mos_depl(op.Vdx - Vgs, self.ch.Cox)
        Qg                  = r.Qg + self.p.CGS_OV * Vgs - Qgd
        Qd                  = r.Qd + Qgd + self._qj(Vds, T)
        return Qg, Qd

    def capacitances(self, Vds, Vgs=0.0, T=298.15, h=1e-3):
        """Datasheet capacitances Ciss, Coss, Crss [F]."""
        def q(vg, vd):
            return self.charges(vg, vd, T)
        qg_gp, qd_gp        = q(Vgs + h, Vds)
        qg_gm, qd_gm        = q(Vgs - h, Vds)
        qg_dp, qd_dp        = q(Vgs, Vds + h)
        qg_dm, qd_dm        = q(Vgs, Vds - h)
        Ciss                = (qg_gp - qg_gm) / (2 * h)
        Crss                = -(qg_dp - qg_dm) / (2 * h)
        Coss                = (qd_dp - qd_dm) / (2 * h)
        return {"Ciss": Ciss, "Coss": Coss, "Crss": Crss,
                "Cgs": Ciss - Crss, "Cgd": Crss, "Cds": Coss - Crss}

    def gate_charge(self, Vdd=50.0, Iload=20.0, Vg_max=10.0, T=298.15, n=80):
        """Quasi-static gate-charge curve (clamped inductive load).

        Returns lists (Qg [C], Vgs [V], Vds [V]) and a dict with Qgs, Qgd, Qg.
        """
        Vgp                 = brentq(lambda v: self.id(v, Vdd, T) - Iload, 0.5, 20.0, xtol=1e-9)
        Qg0, _              = self.charges(0.0, Vdd, T)
        path                = [(vg, Vdd) for vg in [Vgp * i / n for i in range(n + 1)]]
        Vds_on              = brentq(lambda v: self.id(Vg_max, v, T) - Iload, 1e-6, Vdd, xtol=1e-12)
        for i in range(1, n + 1):
            vd              = Vds_on * (Vdd / Vds_on) ** (1.0 - i / n)                 # log-spaced Vds collapse
            vg              = brentq(lambda v: self.id(v, vd, T) - Iload, 0.5, 25.0, xtol=1e-9)
            path.append((vg, vd))
        qg, vgs, vds        = [], [], []
        for vg, vd in path:
            qg.append(self.charges(vg, vd, T)[0] - Qg0)
            vgs.append(vg)
            vds.append(vd)
        Qgs                 = qg[n]
        # end of the Miller plateau: Vds has collapsed to within 5 % of its swing
        v_end               = Vds_on + 0.05 * (Vdd - Vds_on)
        vg_end              = brentq(lambda v: self.id(v, v_end, T) - Iload, 0.5, 25.0, xtol=1e-9)
        Qgd                 = self.charges(vg_end, v_end, T)[0] - Qg0 - Qgs
        return qg, vgs, vds, {"Qgs": Qgs, "Qgd": Qgd, "Qg": qg[-1], "Vplateau": Vgp}
#?----------------------------------------------------------------------------------------------------------------------------------------
