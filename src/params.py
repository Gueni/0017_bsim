#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?              ____  ____ __________ _____ ___  _____
#?             / __ \/ __ `/ ___/ __ `/ __ `__ \/ ___/
#?            / /_/ / /_/ / /  / /_/ / / / / / (__  )
#?           / .___/\__,_/_/   \__,_/_/ /_/ /_/____/
#?          /_/
#?
#?
#? Name:        params.py
#? Purpose:     Model cards for the BSIM6 channel and the vertical power MOSFET, plus physical constants.
#?              Swap the defaults here for an extracted model card of your device.
#?
#? Author:      Mohamed Gueni ( mohamedgueni@outlook.com)
#?
#? Created:     05/11/2026
#? -------------------------------------------------------------------------------
#? -------------------------------------------------------------------------------
from    dataclasses                 import  dataclass
#?----------------------------------------------------------------------------------------------------------------------------------------
# Physical constants [SI]
Q                           = 1.602176634e-19                                          # C
KB                          = 1.380649e-23                                             # J/K
EPS0                        = 8.8541878128e-12                                         # F/m
EPS_SI                      = 11.9 * EPS0
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
class PowerParams:
    """Power MOSFET card: drift region, package, capacitance geometry, avalanche, body diode, thermal. Units are SI."""
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
