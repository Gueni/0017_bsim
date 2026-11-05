#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?              __         _           _____
#?             / /_  _____(_)___ ___  / ___/
#?            / __ \/ ___/ / __ `__ \/ __ \
#?           / /_/ (__  ) / / / / / / /_/ /
#?          /_.___/____/_/_/ /_/ /_/\____/
#?
#?
#? Name:        bsim6.py
#? Purpose:     BSIM6.0 charge-based bulk MOSFET model (DC + intrinsic charges).
#?              Based on the BSIM6.0 Technical Manual (UC Berkeley), chapters 1, 4 and 8.
#?
#? Author:      Mohamed Gueni ( mohamedgueni@outlook.com)
#?
#? Created:     04/1/2023
#? -------------------------------------------------------------------------------
#? -------------------------------------------------------------------------------
"""
BSIM6 is a *charge-based*, body-referenced and source/drain-symmetric model.
Instead of a threshold voltage + region switching (BSIM3/4), it computes the
normalized inversion charge at the source (qs) and drain (qd) from the
pinch-off potential psi_p, and gets a single current expression valid from
weak to strong inversion:

    Ids = 2 nq mu_eff (W/L) Cox nVt^2 (qs - qd)(qs + qd + 1) * Moc      (1.208)

Everything is continuous and infinitely differentiable, and the terminal
charges are conserving (Qg + Qd + Qs + Qb = 0).

Implemented (manual equation numbers in brackets):
  - terminal voltages Vdsx/Vbsx                            [1.52-1.61]
  - pinch-off potential with poly depletion                [1.62-1.86]
      (solved exactly with a bracketed root finder instead of
       the closed-form asymptotic fit used in Verilog-A)
  - normalized source/drain charges                        [1.96-1.127]
  - DIBL / body-effect threshold shifts, slope factor n    [1.128-1.138]
  - drain saturation voltage, Vdseff                       [1.139-1.149]
  - vertical-field mobility degradation                    [1.150-1.151]
  - internal bias-dependent series resistance (RDSMOD=0)   [1.152-1.157]
  - output conductance: CLM, DIBL, SCBE                    [1.162-1.176]
  - velocity saturation                                    [1.177-1.183]
  - temperature dependence                                 [4.7-4.19, 4.44-4.46]
  - intrinsic charges + overlap/fringe charges            [8.8-8.34]
Not implemented: gate tunnelling, GIDL, noise, stress/WPE, binning, QM effect,
DITS (DVTP*), junction diodes (handled by power_mosfet.py for power devices).
"""
import  math
from    dataclasses                 import  dataclass, replace

from    scipy.optimize              import  brentq
#?----------------------------------------------------------------------------------------------------------------------------------------
# Physical constants [SI]
Q                           = 1.602176634e-19                                          # C
KB                          = 1.380649e-23                                             # J/K
EPS0                        = 8.8541878128e-12                                         # F/m

_EXP_MAX                    = 700.0
#?----------------------------------------------------------------------------------------------------------------------------------------
def _smax(x, eps):
    """Smooth max(x, 0) with rounding width eps."""
    return 0.5 * (x + math.sqrt(x * x + 4.0 * eps * eps))
#?----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class BSIM6Params:
    """BSIM6 model card. Names follow the manual; units are SI.

    The defaults describe the channel of a ~100 V vertical power MOSFET
    (thick 50 nm gate oxide, 0.6 um diffused p-body channel, ~1.6 m total
    gate width). Swap them for an extracted model card of your device.
    """
    TYPE    : int           = 1                                                        # +1 NMOS, -1 PMOS
    # --- geometry -------------------------------------------------------------
    L       : float         = 0.6e-6                                                   # drawn channel length [m]
    W       : float         = 1.6                                                      # drawn (total) channel width [m]
    NF      : int           = 1                                                        # number of fingers
    LINT    : float         = 0.0                                                      # length offset [m]
    WINT    : float         = 0.0                                                      # width offset [m]
    XL      : float         = 0.0
    XW      : float         = 0.0
    # --- process ------------------------------------------------------------
    TOX     : float         = 50e-9                                                    # gate oxide thickness [m]
    EPSROX  : float         = 3.9
    EPSRSUB : float         = 11.9
    NDEP    : float         = 1.5e23                                                   # channel (p-body) doping [m^-3]
    NGATE   : float         = 1e26                                                     # poly gate doping [m^-3]; 0 = metal gate
    VFB     : float         = -0.95                                                    # flat-band voltage [V]
    XJ      : float         = 1.5e-6                                                   # junction depth [m]
    PHIN    : float         = 0.045
    NI0SUB  : float         = 1.0e16                                                   # ni at TNOM [m^-3]
    BG0SUB  : float         = 1.17                                                     # bandgap at 0 K [eV]
    TBGASUB : float         = 4.73e-4                                                  # [eV/K]
    TBGBSUB : float         = 636.0                                                    # [K]
    # --- subthreshold / short channel ----------------------------------------
    CIT     : float         = 0.0                                                      # [F/m^2]
    NFACTOR : float         = 0.0                                                      # [F/m^2] (manual eq. 1.131 adds it to CIT)
    CDSCD   : float         = 0.0                                                      # [F/m^2/V]
    CDSCB   : float         = 0.0                                                      # [F/m^2/V]
    K2      : float         = 0.0
    ETA0    : float         = 0.01                                                     # DIBL coefficient
    ETAB    : float         = 0.0
    # --- mobility -------------------------------------------------------------
    U0      : float         = 0.05                                                     # low-field mobility [m^2/Vs]
    ETAMOB  : float         = 1.0
    UA      : float         = 0.5                                                      # phonon/surface roughness [(cm/MV)^EU]
    EU      : float         = 1.5
    UD      : float         = 0.001                                                    # Coulomb scattering
    UCS     : float         = 2.0
    UC      : float         = 0.0                                                      # body-bias sensitivity of mobility [1/V]
    # --- velocity saturation -------------------------------------------------
    VSAT    : float         = 1.0e5                                                    # [m/s]
    PSAT    : float         = 1.0
    DELTA   : float         = 0.125                                                    # Vds -> Vdsat smoothing
    # --- output conductance ---------------------------------------------------
    PCLM    : float         = 0.05
    PCLMG   : float         = 0.0
    PDIBLC  : float         = 1e-3
    PDIBLCB : float         = 0.0
    PVAG    : float         = 1.0
    PSCBE1  : float         = 4.24e8                                                   # [V/m]
    PSCBE2  : float         = 1.0e-8                                                   # [m/V]
    # --- internal series resistance (RDSMOD = 0) ----------------------------
    RDSW    : float         = 0.0                                                      # [ohm*um^WR]
    RDSWMIN : float         = 0.0
    PRWG    : float         = 1.0
    PRWB    : float         = 0.0
    WR      : float         = 1.0
    # --- overlap / fringe capacitance ----------------------------------------
    CGSO    : float         = 0.0                                                      # [F/m]
    CGDO    : float         = 0.0                                                      # [F/m]
    CF      : float         = 0.0                                                      # [F/m]
    # --- temperature ----------------------------------------------------------
    TNOM    : float         = 298.15                                                   # [K]
    KT1     : float         = 0.0                                                      # VFB temperature coefficient [V]
    UTE     : float         = -1.5                                                     # U0 temperature exponent
    UA1     : float         = 0.0                                                      # [1/K]
    UC1     : float         = 0.0                                                      # [1/K]
    UD1     : float         = 0.0
    UCSTE   : float         = 0.0
    AT      : float         = 0.0                                                      # VSAT temperature exponent
    PRT     : float         = 0.0                                                      # RDSW temperature exponent
    TETA0   : float         = 0.0
    TFACTOR : float         = 0.0
#?----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class BSIM6Result:
    Ids    : float                                                                     # drain current (into drain) [A]
    Qg     : float                                                                     # terminal charges [C]
    Qd     : float
    Qs     : float
    Qb     : float
    qs     : float                                                                     # normalized source / drain inversion charge
    qd     : float
    psip   : float                                                                     # normalized pinch-off potential
    nq     : float                                                                     # charge slope factor
    Vdssat : float                                                                     # drain saturation voltage (wrt source) [V]
    Vdseff : float                                                                     # effective Vds [V]
    mueff  : float                                                                     # effective mobility [m^2/Vs]
    Moc    : float                                                                     # output-conductance multiplier
#?----------------------------------------------------------------------------------------------------------------------------------------
class BSIM6Model:
    """BSIM6.0 intrinsic MOSFET.

    Use ``evaluate(Vd, Vg, Vs, Vb, T)`` for the full result, or the
    convenience wrappers ``ids``, ``vth``, ``capacitances``.
    """

    def __init__(self, params=None, **overrides):
        self.p              = replace(params or BSIM6Params(), **overrides)
        self._tcache        = {}
        self._geometry()

    # ------------------------------------------------------------------ setup
    def _geometry(self):
        p                   = self.p
        self.Leff           = p.L + p.XL - 2.0 * p.LINT                                # (1.12)
        self.Weff           = p.W + p.XW - 2.0 * p.WINT                                # (1.13)
 
        if self.Leff <= 0 or self.Weff <= 0:raise ValueError("Leff and Weff must be positive")
 
        self.eps_sub        = p.EPSRSUB * EPS0
        self.eps_ox         = p.EPSROX * EPS0
        self.Cox            = 3.9 * EPS0 / p.TOX                                       # (1.5)
        self.eps_ratio      = p.EPSRSUB / 3.9                                          # (1.6)
        self.gamma_v        = math.sqrt(2.0 * Q * self.eps_sub * p.NDEP) / self.Cox    # [V^0.5]
        self.delta_pd       = p.NDEP / p.NGATE if p.NGATE > 0 else 0.0                 # (1.67)
        self.litl           = math.sqrt(p.EPSRSUB / p.EPSROX * p.TOX * p.XJ)           # (1.173)
        self._tcache.clear()

    def set_params(self, **kw):
        self.p              = replace(self.p, **kw)
        self._geometry()

    def _eg(self, T):
        p                   = self.p
        return p.BG0SUB - p.TBGASUB * T * T / (T + p.TBGBSUB)                          # (4.45)

    def ni(self, T):
        """Intrinsic carrier concentration [m^-3]               (4.46)."""
        p                   = self.p
        vt, vt0             = KB * T / Q, KB * p.TNOM / Q
        return p.NI0SUB * (T / p.TNOM) ** 1.5 * math.exp(
            self._eg(p.TNOM) / (2 * vt0) - self._eg(T) / (2 * vt))

    def _temp(self, T):
        """Temperature-scaled parameters (cached per temperature)."""
        key                 = round(float(T), 9)
        t                   = self._tcache.get(key)
        if t is not None:
            return t
        p, r                = self.p, T / self.p.TNOM
        t                   = dict(
                              Vt      = KB * T / Q,
                              ni      = self.ni(T),
                              VFB     = p.VFB - p.KT1 * (r - 1.0),                     # (4.7)
                              NFACTOR = p.NFACTOR + p.TFACTOR * (r - 1.0),             # (4.8)
                              ETA0    = p.ETA0 + p.TETA0 * (r - 1.0),                  # (4.9)
                              U0      = p.U0 * r ** p.UTE,                             # (4.10)
                              UA      = p.UA * (1.0 + p.UA1 * (T - p.TNOM)),           # (4.11)
                              UC      = p.UC * (1.0 + p.UC1 * (T - p.TNOM)),           # (4.12)
                              UD      = p.UD * r ** p.UD1,                             # (4.13)
                              UCS     = p.UCS * r ** p.UCSTE,                          # (4.14)
                              VSAT    = p.VSAT * r ** (-p.AT),                         # (4.16)
                              RDSW    = p.RDSW * r ** p.PRT,                           # (4.19)
                              RDSWMIN = p.RDSWMIN * r ** p.PRT,                        # (4.20)
        )
        t["phib_v"]         = t["Vt"] * math.log(p.NDEP / t["ni"])                     # Fermi potential [V]
        t["psist"]          = 0.4 + p.PHIN + t["phib_v"]                               # (1.128)
        if len(self._tcache) > 512:
            self._tcache.clear()
        self._tcache[key]   = t
        return t

    # ------------------------------------------------------- core sub-solvers
    @staticmethod
    def _psip(vgfb, gam, dpd):
        """Normalized pinch-off potential: exact root of (1.74).

        vgfb = psi + sign(psi)*gam*sqrt(h) + dpd*h,   h = e^-psi + psi - 1
        The right-hand side is strictly increasing in psi -> unique root.
        """
        if vgfb == 0.0:
            return 0.0

        def f(psi):
            h               = max(math.expm1(-psi) + psi, 0.0)
            s               = math.sqrt(h)
            if psi >= 0.0:
                return psi + gam * s + dpd * h - vgfb
            return psi - gam * s - vgfb

        if vgfb > 0.0:
            lo, hi          = 0.0, vgfb
        else:
            hi              = 0.0
            lo              = -2.0 * math.log1p(-vgfb / gam) - 2.0
            while f(lo) > 0.0:
                lo *= 2.0
        return brentq(f, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=200)

    @staticmethod
    def _lnq(psip, nq, gam, v):
        """ln of normalized inversion charge: root of (1.96)/(1.121).

        g(x) = 2 e^x + x + ln(2nq/gam) + ln(2 e^x nq/gam + 2 sqrt(psip)) - v
        with v = psip - 2*phib - vch.  g is strictly increasing in x.
        """
        k                   = 2.0 * nq / gam
        c1                  = math.log(k)
        sp2                 = 2.0 * math.sqrt(psip)

        def g(x):
            ex              = math.exp(min(x, _EXP_MAX))
            return 2.0 * ex + x + c1 + math.log(k * ex + sp2) - v

        xa                  = v - c1 - math.log(sp2)                                   # root <= xa (other terms >= 0)
        hi                  = xa if xa < 1.0 else min(xa, math.log(max(v, 1.0)) + 1.0)
        while g(hi) < 0.0:
            hi              = hi + 1.0 + abs(hi)
        lo                  = min(xa, hi) - 10.0
        while g(lo) > 0.0:
            lo              = lo - 10.0 - abs(lo)
        return brentq(g, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)

    def _dmob(self, t, Vbsx, qi, qb):
        """Mobility degradation factor (1.140-1.141, 1.150-1.151)."""
        p                   = self.p
        eta                 = 0.5 * p.ETAMOB if p.TYPE > 0 else p.ETAMOB / 3.0         # (1.139)
        Eeff                = 1e-8 * (qb + eta * qi) / (self.eps_ratio * p.TOX)        # [MV/cm]
        Eeff                = max(Eeff, 0.0)
        ua                  = t["UA"] + t["UC"] * Vbsx
        coul                = t["UD"] / (0.5 * (1.0 + qi / max(qb, 1e-12))) ** t["UCS"]
        return 1.0 + ua * Eeff ** p.EU + coul

    # --------------------------------------------------------------- evaluate
    def evaluate(self, Vd, Vg, Vs=0.0, Vb=0.0, T=298.15):
        """Full model evaluation at terminal voltages [V] and temperature [K]."""
        p, t                = self.p, self._temp(T)
        sgn                 = p.TYPE
        Vd, Vg, Vs, Vb      = sgn * Vd, sgn * Vg, sgn * Vs, sgn * Vb
        swap                = Vd < Vs                                                  # BSIM6 is symmetric: swap S/D
        if swap:
            Vd, Vs          = Vs, Vd

        Vt                  = t["Vt"]
        Vgb, Vsb            = Vg - Vb, Vs - Vb
        Vds                 = Vd - Vs
        Vdsx                = math.sqrt(Vds * Vds + 0.01) - 0.1                        # (1.60)
        Vbsx                = -(Vsb + 0.5 * (Vds - Vdsx))                              # (1.61)

        # slope factor and threshold shifts                         (1.131-1.138)
        n                   = 1.0 + (p.CIT + t["NFACTOR"] + p.CDSCD * Vdsx - p.CDSCB * Vbsx) / self.Cox
        n                   = max(n, 1.0)
        nVt                 = n * Vt
        dVth                = -p.K2 * Vbsx - (t["ETA0"] + p.ETAB * Vbsx) * Vdsx
        Vgfb                = Vgb - t["VFB"] - dVth

        # pinch-off potential and charges (all normalized to nVt)
        gam                 = self.gamma_v / math.sqrt(nVt)                            # (1.63)
        psip_raw            = self._psip(Vgfb / nVt, gam, self.delta_pd)
        psip                = _smax(psip_raw, 1e-3)
        nq                  = 1.0 + gam / (2.0 * math.sqrt(psip))                      # (1.116)
        vbase               = psip - 2.0 * t["phib_v"] / nVt
        lnqs                = self._lnq(psip, nq, gam, vbase - Vsb / nVt)
        qs                  = math.exp(lnqs)

        # source-side mobility, velocity-saturation parameter
        qis                 = 2.0 * nq * nVt * qs                                      # [V]
        qbs                 = nVt * gam * math.sqrt(_smax(psip - 2.0 * qs, 1e-3))      # [V]
        Dmobs               = self._dmob(t, Vbsx, qis, qbs)
        lc                  = t["U0"] * nVt / (Dmobs ** p.PSAT * t["VSAT"] * self.Leff)

        # drain saturation charge / voltage                         (1.143-1.147)
        ln_qdsat            = math.log(lc) + lnqs + math.log1p(qs) - math.log1p(lc * (1.0 + qs))
        qdsat               = math.exp(ln_qdsat)
        k                   = 2.0 * nq / gam
        vdsat               = vbase - (2.0 * qdsat + ln_qdsat + math.log(k)
                                       + math.log(k * qdsat + 2.0 * math.sqrt(psip)))
        Vdssat              = max(vdsat * nVt - Vsb, 1e-9)

        # effective Vds                                              (1.148)
        r                   = Vds / Vdssat
        if r > 0.0:
            lr              = min(math.log(r) / p.DELTA, _EXP_MAX)
            Vdseff          = Vds * math.exp(-p.DELTA * (lr + math.log1p(math.exp(-lr)))
                                             if lr > 0 else -p.DELTA * math.log1p(math.exp(lr)))
        else:
            Vdseff          = 0.0
        lnqd                = self._lnq(psip, nq, gam, vbase - (Vdseff + Vsb) / nVt)
        qd                  = math.exp(lnqd)

        # average charges, mobility                               (1.150-1.151)
        qia                 = nq * nVt * (qs + qd)
        qba                 = nVt * gam * math.sqrt(_smax(psip - (qs + qd), 1e-3))
        Dmob                = self._dmob(t, Vbsx, qia, qba)

        # internal series resistance                              (1.152-1.157)
        Dr                  = 1.0
        if t["RDSW"] > 0.0 or t["RDSWMIN"] > 0.0:
            T0              = 1.0 + p.PRWG * qia
            T1              = p.PRWB * (math.sqrt(max(t["psist"] - Vbsx, 1e-6)) - math.sqrt(t["psist"]))
            T2              = 1.0 / T0 + T1
            T3              = 0.5 * (T2 + math.sqrt(T2 * T2 + 0.01))
            Rds             = (t["RDSWMIN"] + t["RDSW"] * T3) / (self.Weff * 1e6) ** p.WR / p.NF
            Dr              = 1.0 + t["U0"] * self.Cox * self.Weff / self.Leff * qia * Rds

        # output conductance                                      (1.162-1.176)
        dV                  = max(Vds - Vdseff, 0.0)
        Esat                = 2.0 * t["VSAT"] / (t["U0"] / Dmob)
        EsatL               = Esat * self.Leff
        if p.PCLM > 0.0:
            if p.PCLMG >= 0.0:
                Cclm        = p.PCLM * (1.0 + p.PCLMG * qia / EsatL)
            else:
                Cclm        = p.PCLM / (1.0 - p.PCLMG * qia / EsatL)
            Vasat           = Vdssat + EsatL
            MCLM            = 1.0 + Cclm * math.log1p(dV / (Vasat * Cclm))
        else:
            MCLM            = 1.0
        if p.PDIBLC > 0.0:
            pvag            = (1.0 + p.PVAG * qia / EsatL) if p.PVAG >= 0 else 1.0 / (1.0 - p.PVAG * qia / EsatL)
            VADIBL          = ((qia + 2.0 * Vt) / p.PDIBLC
                               * (1.0 - Vdssat / (Vdssat + qia + 2.0 * Vt))
                               * pvag / (1.0 + p.PDIBLCB * Vbsx))
            MDIBL           = 1.0 + dV / VADIBL
        else:
            MDIBL           = 1.0
        if p.PSCBE2 > 0.0 and dV > 0.0:
            inv_vascbe      = p.PSCBE2 / self.Leff * math.exp(-min(p.PSCBE1 * self.litl / dV, _EXP_MAX))
            MSCBE           = 1.0 + dV * inv_vascbe
        else:
            MSCBE           = 1.0
        Moc                 = MCLM * MDIBL * MSCBE

        # velocity saturation and current                          (1.177-1.208)
        dq                  = qs - qd
        T1                  = 2.0 * lc * dq
        Dvsat               = 0.5 * (math.sqrt(1.0 + T1 * T1) + math.asinh(T1) / T1) if T1 > 1e-8 else 1.0
        mueff               = t["U0"] / (Dmob * Dvsat * Dr)
        Ids                 = (2.0 * nq * mueff * self.Weff / self.Leff * self.Cox * nVt * nVt
                               * dq * (qs + qd + 1.0) * Moc * p.NF)

        # intrinsic charges                                          (8.8-8.34)
        den                 = 1.0 + qs + qd
        brk                 = qs + qd + dq * dq / (3.0 * den)
        qB                  = Vgfb / nVt - psip_raw - (nq - 1.0) * brk
        fr                  = (dq / den) ** 2
        Qs_n                = nq / 3.0 * (2.0 * qs + qd + 0.5 * (1.0 + 0.8 * qs + 1.2 * qd) * fr)
        Qd_n                = nq / 3.0 * (qs + 2.0 * qd + 0.5 * (1.0 + 1.2 * qs + 0.8 * qd) * fr)
        WLC                 = p.NF * self.Weff * self.Leff * self.Cox * nVt
        QS, QD, QB          = -WLC * Qs_n, -WLC * Qd_n, -WLC * qB
        # overlap + fringe (bias-independent part)                  (8.35-8.40)
        Vgs_, Vgd_          = Vg - Vs, Vg - Vd
        QGSov               = p.NF * self.Weff * (p.CGSO + p.CF) * Vgs_
        QGDov               = p.NF * self.Weff * (p.CGDO + p.CF) * Vgd_
        QS -= QGSov
        QD -= QGDov
        QG                  = -(QS + QD + QB)

        if swap:
            Ids, QS, QD     = -Ids, QD, QS
            qs, qd          = qd, qs
        return BSIM6Result(Ids=sgn * Ids, Qg=sgn * QG, Qd=sgn * QD, Qs=sgn * QS,
                           Qb=sgn * QB, qs=qs, qd=qd, psip=psip_raw, nq=nq,
                           Vdssat=Vdssat, Vdseff=Vdseff, mueff=mueff, Moc=Moc)

    # ------------------------------------------------------------- helpers
    def ids(self, Vgs, Vds, Vbs=0.0, T=298.15):
        """Drain current with source as reference."""
        return self.evaluate(Vds, Vgs, 0.0, Vbs, T).Ids

    def vth(self, Vds=0.0, Vbs=0.0, T=298.15):
        """Physical threshold: gate voltage at which psi_p = 2*phi_F + Vsb.

        Obtained by inverting the pinch-off relation (1.74) - closed form.
        """
        p, t                = self.p, self._temp(T)
        Vt                  = t["Vt"]
        Vsb                 = -Vbs
        Vdsx                = math.sqrt(Vds * Vds + 0.01) - 0.1
        Vbsx                = -(Vsb + 0.5 * (Vds - Vdsx))
        n                   = max(1.0 + (p.CIT + t["NFACTOR"] + p.CDSCD * Vdsx - p.CDSCB * Vbsx) / self.Cox, 1.0)
        nVt                 = n * Vt
        dVth                = -p.K2 * Vbsx - (t["ETA0"] + p.ETAB * Vbsx) * Vdsx
        gam                 = self.gamma_v / math.sqrt(nVt)
        psi                 = (2.0 * t["phib_v"] + Vsb) / nVt
        h                   = math.expm1(-psi) + psi
        vgfb                = psi + gam * math.sqrt(h) + self.delta_pd * h
        return p.TYPE * (vgfb * nVt + t["VFB"] + dVth - Vsb)

    def capacitances(self, Vgs, Vds, Vbs=0.0, T=298.15, h=1e-4):
        """Small-signal capacitance matrix Cij = dQi/dVj (i != j: -dQi/dVj).

        Returns dict with Cgg, Cgs, Cgd, Cgb, Cdd, Cdg, Cds, Csg ... [F].
        """
        V0                  = {"d": Vds, "g": Vgs, "s": 0.0, "b": Vbs}
        out                 = {}
        for j in "gdsb":
            Vp, Vm          = dict(V0), dict(V0)
            Vp[j] += h
            Vm[j] -= h
            rp              = self.evaluate(Vp["d"], Vp["g"], Vp["s"], Vp["b"], T)
            rm              = self.evaluate(Vm["d"], Vm["g"], Vm["s"], Vm["b"], T)
            for i, attr in (("g", "Qg"), ("d", "Qd"), ("s", "Qs"), ("b", "Qb")):
                dQ          = (getattr(rp, attr) - getattr(rm, attr)) / (2.0 * h)
                out[f"C{i}{j}"] = dQ if i == j else -dQ
        return out
#?----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    m                       = BSIM6Model()
    print(f"Vth(25C) = {m.vth():.3f} V   Vth(150C) = {m.vth(T=423.15):.3f} V")
    for vgs in (2.0, 3.0, 4.0, 6.0, 10.0):
        print(f"Vgs={vgs:5.1f}  Id(Vds=0.1)={m.ids(vgs, 0.1):.4e}  Id(Vds=5)={m.ids(vgs, 5.0):.4e}")
#?----------------------------------------------------------------------------------------------------------------------------------------
