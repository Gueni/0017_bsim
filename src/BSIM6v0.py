#!/usr/bin/env python
# coding=utf-8
# -------------------------------------------------------------------------------
#
#                 ______  ____  _______  _____
#                / __ \ \/ /  |/  / __ \/ ___/
#               / /_/ /\  / /|_/ / / / /\__ \
#              / ____/ / / /  / / /_/ /___/ /
#             /_/     /_/_/  /_/\____//____/
#
# Name:        BSIMv6_0.py
# Purpose:     Compute drain current using the BSIMv6.0 model
#
# Author:      Mohamed Gueni (mohamedgueni@outlook.com)
# Based on:    BSIMv6.0 Technical Manual 
# Created:     21/05/2025
# Licence:     Refer to the LICENSE file
# -------------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import webbrowser
import os

class BSIMv6_Model:
    """BSIMv6.0 MOSFET model implementation for circuit simulation.
    
    This class implements the BSIMv6.0 model for modern MOSFET transistors. 
    It includes advanced effects needed for nanometer-scale devices:
    - Quantum mechanical effects
    - Advanced short-channel effects
    - Stress effects
    - Advanced mobility degradation
    - Gate leakage current
    - Accurate temperature dependence
    - Self-heating effects
    
    Attributes:
        Various physical constants and model parameters initialized in __init__
    """
    
    def __init__(self):
        """Initialize BSIMv6.0 model with default parameters for 7nm FinFET."""
        # Physical constants (SI units)
        self.epsSi = 11.7 * 8.854e-12       # F/m, Silicon permittivity
        self.epsOx = 3.9 * 8.854e-12        # F/m, Silicon dioxide permittivity
        self.q = 1.602e-19                  # C, Electron charge
        self.k_B = 1.38e-23                 # J/K, Boltzmann constant
        self.hbar = 1.0545718e-34           # Reduced Planck constant
        self.m0 = 9.10938356e-31            # kg, Electron mass
        
        # Model selection flags
        self.mobMod = 11                    # Mobility model selector
        self.rdsMod = 1                     # Rds model selector
        self.igMod = 1                      # Gate current model selector
        
        # Threshold voltage related parameters
        self.VTH0 = 0.35                    # V, Zero-bias threshold voltage
        self.K1 = 0.6                       # √V, First body effect coefficient
        self.K2 = 0.02                      # -, Second body effect coefficient
        self.K3 = 1.5                       # -, Narrow width effect coefficient
        self.K3B = 0.0                      # -, Body effect on narrow width coefficient
        self.DVT0 = 2.5                     # -, Short-channel effect coefficient
        self.DVT1 = 0.6                     # -, Short-channel effect coefficient
        self.DVT2 = -0.03                   # 1/V, Short-channel effect coefficient
        self.DVT0W = 0.0                    # -, Narrow width effect coefficient
        self.DVT1W = 5.3e6                  # -, Narrow width effect coefficient
        self.DVT2W = -0.032                 # 1/V, Narrow width effect coefficient
        self.NLX = 1.5e-7                   # m, Lateral non-uniform doping parameter
        self.W0 = 2.5e-6                    # m, Narrow width parameter
        self.VOFF = -0.08                   # V, Offset voltage for subthreshold current
        self.NFACTOR = 1.0                  # -, Subthreshold swing coefficient
        self.CIT = 0.0                      # F/m², Interface trap capacitance
        self.CDSC = 0.0                     # F/m², Drain/source to channel coupling
        self.CDSCB = 0.0                    # F/m², Body bias dependence of CDSC
        self.CDSCD = 0.0                     # F/m², Drain bias dependence of CDSC
        
        # Mobility parameters
        self.U0 = 0.03                      # m²/V·s, Low-field mobility (7nm value)
        self.UA = 2.0e-9                    # m/V, First mobility degradation coefficient
        self.UB = 5.0e-19                   # (m/V)², Second mobility degradation coefficient
        self.UC = -0.05                     # -, Body-effect mobility coefficient
        self.UE = 1.0                       # -, Exponent for mobility degradation
        self.UCS = 1.0                      # -, Stress effect on mobility
        
        # Velocity saturation parameters
        self.VSAT = 1.0e5                   # m/s, Saturation velocity
        self.A0 = 1.0                       # -, Bulk charge effect coefficient
        self.A1 = 0.0                       # -, Saturation voltage parameter
        self.A2 = 1.0                       # -, Saturation voltage parameter
        self.B0 = 0.0                       # -, Width effect on Abulk
        self.B1 = 0.0                       # -, Width effect on Abulk
        self.AT = 3.0e4                     # m/s, Temperature coefficient for VSAT
        
        # Channel length modulation
        self.PCLM = 1.2                     # -, Channel length modulation coefficient
        self.PDIBLC1 = 0.45                 # -, DIBL coefficient
        self.PDIBLC2 = 0.45                 # -, DIBL coefficient
        self.PDIBLCB = -0.08                # -, Body bias effect on DIBL
        self.DROUT = 0.56                   # -, Output resistance coefficient
        self.PVAG = 1.0e6                   # 1/V, Gate voltage effect on output resistance
        
        # Subthreshold conduction
        self.DELTA = 0.01                   # -, Smoothing parameter
        self.ETA0 = 0.15                    # -, DIBL coefficient
        self.ETAB = -0.12                   # -, Body bias effect on DIBL
        self.DSUB = 1.2                      # -, Subthreshold DIBL coefficient
        
        # Quantum mechanical effects
        self.QMFACTOR = 1.0                  # -, Quantum mechanical correction factor
        self.EQM0 = 1.0e9                    # V/m, Quantum mechanical field parameter
        
        # Stress effects
        self.STUFF = 1.0                     # -, Stress effect parameter
        self.PSCE1 = 4.24e8                  # -, Stress effect coefficient 1
        self.PSCE2 = 1.0e-5                  # -, Stress effect coefficient 2
        
        # Temperature effects
        self.TNOM = 300.0                    # K, Nominal temperature
        self.KT1 = -0.15                     # V, Temperature coefficient for Vth
        self.KT1L = 1e-9                     # V·m, Temperature coefficient for Vth
        self.KT2 = 0.03                      # -, Temperature coefficient for Vth
        self.UTE = -1.8                      # -, Mobility temperature exponent
        self.UTL = 0.0                       # -, Length dependence of mobility temp effect
        self.PRT = 0.0                       # -, Temperature coefficient for Rds
        
        # Geometry parameters (7nm FinFET)
        self.LEFF = 700e-9                     # m, Effective channel length
        self.WEFF = 2000e-9                    # m, Effective channel width (fin width)
        self.HFIN = 30e-9                    # m, Fin height
        self.TFIN = 10e-9                    # m, Fin thickness
        self.TOX = 0.8e-9                    # m, Oxide thickness
        self.TOXM = 0.8e-9                   # m, Oxide thickness for modeling
        self.COX = self.epsOx / self.TOX     # F/m², Oxide capacitance
        self.COXM = self.epsOx / self.TOXM   # F/m², Oxide capacitance for modeling
        
        # Doping concentrations
        self.NCH = 1e24                      # m-3, Channel doping
        self.NGATE = 1e25                    # m-3, Gate doping
        self.NSD = 2e26                      # m-3, Source/drain doping
        
        # Parasitic resistance
        self.RDSW = 100.0                    # Ω·µm, Source/drain resistance
        self.PRW = 1.0                       # -, Width dependence for resistance
        self.PRWGT = 0.0                     # -, Gate bias effect on resistance
        self.PRWBT = 0.1                     # -, Body bias effect on resistance
        
        # Gate leakage current
        self.AIGBACC = 1.0e-8                # A, Gate current parameter
        self.BIGBACC = 4.0e7                 # V/m, Gate current parameter
        self.CIGBACC = 0.5                   # -, Gate current parameter
        self.NIGBACC = 1.0                   # -, Gate current parameter
        self.AIGBINV = 1.0e-8                # A, Gate current parameter
        self.BIGBINV = 4.0e7                 # V/m, Gate current parameter
        self.CIGBINV = 0.5                   # -, Gate current parameter
        self.NIGBINV = 3.0                   # -, Gate current parameter
        
        # Self-heating parameters
        self.RTH0 = 1.0e3                    # K/W, Thermal resistance
        self.CTH0 = 1.0e-9                   # J/K, Thermal capacitance
        self.POWER = 0.0                     # W, Power dissipation
        
        # Initialize state variables
        self.TDEV = self.TNOM                # K, Device temperature
        self.VGST = 0.0                      # V, Effective gate overdrive
        self.VDSAT = 0.0                     # V, Saturation voltage
        self.VDSEFF = 0.0                    # V, Effective drain voltage
        self.IDS = 0.0                       # A, Drain current
        
    def ni(self, T):
        """Intrinsic carrier concentration with temperature dependence."""
        return 1.45e16 * (T / self.TNOM) ** 1.5 * np.exp(21.5565981 - 6793.4246 / T)
    
    def vt(self, T):
        """Thermal voltage."""
        return self.k_B * T / self.q
    
    def Phi_s(self, T):
        """Surface potential."""
        return 2 * self.vt(T) * np.log(self.NCH / self.ni(T))
    
    def Xdep0(self, T):
        """Zero-bias depletion width."""
        return np.sqrt(2 * self.epsSi * self.Phi_s(T) / (self.q * self.NCH))
    
    def Vbi(self, T):
        """Built-in potential."""
        return self.vt(T) * np.log(self.NCH * self.NSD / self.ni(T)**2)
    
    def calculate_Vth(self, Vds, Vbs, T):
        """Calculate threshold voltage with advanced short-channel effects."""
        # Effective body bias
        Vbc = 0.9 * (self.Phi_s(T) - (self.K1**2) / (4 * self.K2**2))
        Vbseff = Vbc + 0.5 * (Vbs - Vbc - self.DELTA + 
                             np.sqrt((Vbs - Vbc - self.DELTA)**2 + 4 * self.DELTA * Vbc))
        
        # Depletion width and characteristic lengths
        Xdep = np.sqrt(2 * self.epsSi * (self.Phi_s(T) - Vbseff) / (self.q * self.NCH))
        lt = np.sqrt(self.epsSi * Xdep * self.TOX / (self.epsOx * (1 + self.DVT2 * Vbseff)))
        ltw = np.sqrt(self.epsSi * Xdep * self.TOX / (self.epsOx * (1 + self.DVT2W * Vbseff)))
        
        # Scale parameters for oxide thickness
        K1ox = self.K1 * (self.TOX / self.TOXM)
        K2ox = self.K2 * (self.TOX / self.TOXM)
        Vth0ox = self.VTH0 - K1ox * np.sqrt(self.Phi_s(T))
        
        # Calculate all terms of threshold voltage
        term1 = Vth0ox + K1ox * np.sqrt(self.Phi_s(T) - Vbseff) - K2ox * Vbseff
        term2 = K1ox * (np.sqrt(1 + self.NLX/self.LEFF) - 1) * np.sqrt(self.Phi_s(T))
        term3 = (self.K3 + self.K3B * Vbseff) * (self.TOX / (self.WEFF + self.W0)) * self.Phi_s(T)
        term4 = -self.DVT0W * (np.exp(-self.DVT1W * self.WEFF * self.LEFF/(2 * ltw)) + 
                              2 * np.exp(-self.DVT1W * self.WEFF * self.LEFF/ltw)) * (self.Vbi(T) - self.Phi_s(T))
        term5 = -self.DVT0 * (np.exp(-self.DVT1 * self.LEFF/(2 * lt)) + 
                              2 * np.exp(-self.DVT1 * self.LEFF/lt)) * (self.Vbi(T) - self.Phi_s(T))
        term6 = -(np.exp(-self.DSUB * self.LEFF/(2 * lt)) + 
                 2 * np.exp(-self.DSUB * self.LEFF/lt)) * (self.ETA0 + self.ETAB * Vbseff) * Vds
        
        # Temperature effect
        delta_T = (T / self.TNOM) - 1
        term7 = (self.KT1 + self.KT1L/self.LEFF + self.KT2 * Vbseff) * delta_T
        
        # Quantum mechanical effect
        Eeff = (self.q * self.NCH * Xdep) / (2 * self.epsSi)
        term8 = -self.QMFACTOR * (Eeff / self.EQM0) * self.vt(T)
        
        Vth = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
        return Vth
    
    def calculate_mobility(self, Vgs, Vds, Vbs, T):
        """Calculate effective mobility with advanced degradation models."""
        Vth = self.calculate_Vth(Vds, Vbs, T)
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        
        # Temperature effect on mobility
        U0_T = self.U0 * (T/self.TNOM)**(self.UTE + self.UTL/self.LEFF)
        
        # Vertical field mobility degradation
        Eeff = (Vgsteff + 2 * Vth) / (6 * self.TOX)
        
        if self.mobMod == 11:  # Advanced mobility model for FinFETs
            denom = (1 + (self.UA * Eeff + self.UB * Eeff**2) * 
                    (1 + self.UC * Vbs) * self.UCS)
            mob_eff = U0_T / denom
        else:  # Standard mobility model
            denom = 1 + (self.UA * Eeff + self.UB * Eeff**2) * (1 + self.UC * Vbs)
            mob_eff = U0_T / denom
            
        return mob_eff
    
    def calculate_Vgsteff(self, Vgs, Vds, Vbs, T):
        """Calculate effective gate overdrive with advanced smoothing."""
        Vth = self.calculate_Vth(Vds, Vbs, T)
        Vgst = Vgs - Vth
        
        # Calculate subthreshold slope factor
        Cd = self.epsSi / self.Xdep0(T)
        lt = np.sqrt(self.epsSi * self.Xdep0(T) * self.TOX / self.epsOx)
        
        term1 = np.exp(-self.DVT1 * self.LEFF / (2 * lt))
        term2 = np.exp(-self.DVT1 * self.LEFF / lt)
        
        n = 1 + self.NFACTOR * (Cd / self.COX) + \
            ((self.CDSC + self.CDSCD * Vds + self.CDSCB * Vbs) * 
             (term1 + 2 * term2)) / self.COX + \
            self.CIT / self.COX
        
        # Smooth transition between subthreshold and strong inversion
        Vgsteff = n * self.vt(T) * np.log(1 + np.exp(Vgst / (n * self.vt(T))))
        
        # Additional smoothing for better convergence
        Vgsteff = Vgsteff / (1 + (Vgsteff / (10 * n * self.vt(T)))**20)**(1/20)
        
        return Vgsteff
    
    def calculate_Abulk(self, Vbs, T):
        """Calculate bulk charge effect coefficient."""
        # Scale K1 for oxide thickness
        K1ox = self.K1 * (self.TOX / self.TOXM)
        
        term1 = 1 + (K1ox / (2 * np.sqrt(self.Phi_s(T) - Vbs))) * \
                (self.A0 * self.LEFF / (self.LEFF + 2 * np.sqrt(self.Xdep0(T) * self.TOX))) * \
                (1 - self.A1 * (self.LEFF / (self.LEFF + 2 * np.sqrt(self.Xdep0(T) * self.TOX))))
        
        term2 = (self.B0 / (self.WEFF + self.B1)) / (1 + self.VOFF * Vbs)
        
        Abulk = term1 + term2
        return Abulk
    
    def calculate_Vdsat(self, Vgs, Vds, Vbs, T):
        """Calculate saturation voltage with advanced effects."""
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Abulk = self.calculate_Abulk(Vbs, T)
        mob_eff = self.calculate_mobility(Vgs, Vds, Vbs, T)
        
        # Saturation velocity temperature dependence
        VSAT_T = self.VSAT - self.AT * (T / self.TNOM - 1)
        
        # Saturation electric field
        Esat = 2 * VSAT_T / mob_eff
        
        # Calculate Vdsat
        if self.rdsMod == 0:
            Vdsat = (Esat * self.LEFF * (Vgsteff + 2 * self.vt(T))) / \
                    (Abulk * Esat * self.LEFF + Vgsteff + 2 * self.vt(T))
        else:
            # More accurate model with Rds effect
            Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
            if Rds == 0:
                Vdsat = (Esat * self.LEFF * (Vgsteff + 2 * self.vt(T))) / \
                        (Abulk * Esat * self.LEFF + Vgsteff + 2 * self.vt(T))
            else:
                # Solve quadratic equation for Vdsat
                a = Abulk**2 * self.WEFF * VSAT_T * self.COX * Rds
                b = -(Vgsteff + 2 * self.vt(T) + Abulk * Esat * self.LEFF + 
                      3 * Abulk * (Vgsteff + 2 * self.vt(T)) * self.WEFF * VSAT_T * self.COX * Rds)
                c = (Vgsteff + 2 * self.vt(T)) * Esat * self.LEFF + \
                    2 * (Vgsteff + 2 * self.vt(T))**2 * self.WEFF * VSAT_T * self.COX * Rds
                
                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    discriminant = 0
                
                Vdsat = (-b - np.sqrt(discriminant)) / (2 * a)
        
        return Vdsat
    
    def calculate_Rds(self, Vds, Vgs, Vbs, T):
        """Calculate bias-dependent source/drain resistance."""
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Rds = self.RDSW * (1 + self.PRWGT * Vgsteff + self.PRWBT * 
                          (np.sqrt(self.Phi_s(T) - Vbs) - np.sqrt(self.Phi_s(T)))) / (self.WEFF * 1e6)**self.PRW
        
        # Temperature effect
        Rds = Rds * (1 + self.PRT * (T / self.TNOM - 1))
        
        return Rds
    
    def calculate_Vdseff(self, Vds, Vgs, Vbs, T):
        """Calculate effective Vds with advanced smoothing."""
        Vdsat = self.calculate_Vdsat(Vgs, Vds, Vbs, T)
        
        # Smooth transition around Vdsat
        Vdseff = Vdsat - 0.5 * (Vdsat - Vds - self.DELTA + 
                                np.sqrt((Vdsat - Vds - self.DELTA)**2 + 4 * self.DELTA * Vdsat))
        
        # Additional smoothing for better convergence
        Vdseff = Vdseff / (1 + (Vdseff / (10 * Vdsat))**20)**(1/20)
        
        return Vdseff
    
    def calculate_Ids_linear(self, Vds, Vgs, Vbs, T):
        """Calculate linear region current."""
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Vdseff = self.calculate_Vdseff(Vds, Vgs, Vbs, T)
        mob_eff = self.calculate_mobility(Vgs, Vds, Vbs, T)
        Abulk = self.calculate_Abulk(Vbs, T)
        Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
        
        # Saturation velocity temperature dependence
        VSAT_T = self.VSAT - self.AT * (T / self.TNOM - 1)
        
        # Saturation electric field
        Esat = 2 * VSAT_T / mob_eff
        
        # Calculate current without Rds
        Ids0 = mob_eff * self.COX * (self.WEFF / self.LEFF) * \
               Vgsteff * Vdseff * (1 - Vdseff / (2 * Abulk * (Vgsteff + 2 * self.vt(T)))) / \
               (1 + Vdseff / (Esat * self.LEFF))
        
        # Add Rds effect
        if Rds == 0:
            Ids = Ids0
        else:
            Ids = Ids0 / (1 + Rds * Ids0 / Vdseff)
            
        return Ids
    
    def calculate_Ids_saturation(self, Vds, Vgs, Vbs, T):
        """Calculate saturation region current with advanced effects."""
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Vdsat = self.calculate_Vdsat(Vgs, Vds, Vbs, T)
        Vdseff = self.calculate_Vdseff(Vds, Vgs, Vbs, T)
        Abulk = self.calculate_Abulk(Vbs, T)
        mob_eff = self.calculate_mobility(Vgs, Vds, Vbs, T)
        Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
        
        # Saturation velocity temperature dependence
        VSAT_T = self.VSAT - self.AT * (T / self.TNOM - 1)
        
        # Characteristic length for CLM
        lit = np.sqrt(self.epsSi * self.Xdep0(T) * self.TOX / self.epsOx)
        
        # Calculate Idsat
        Idsat = self.WEFF * VSAT_T * self.COX * (Vgsteff - Abulk * Vdsat)
        
        # Add Rds effect to Idsat
        if Rds > 0:
            Idsat = Idsat / (1 + Rds * Idsat / Vdsat)
        
        # Channel length modulation
        VACLM = ((Vgsteff - Abulk * Vdsat) + Abulk * Esat * self.LEFF) / \
                (self.PCLM * Abulk * Esat * lit) * (Vdseff - Vdsat)
        
        # DIBL effect on output resistance
        theta_rout = self.PDIBLC1 * (np.exp(-self.DROUT * self.LEFF / (2 * lit)) + 
                                    2 * np.exp(-self.DROUT * self.LEFF / lit)) + self.PDIBLC2
        
        VADIBLC = (Vgsteff + 2 * self.vt(T)) / (theta_rout * (1 + self.PDIBLCB * Vbs)) * \
                 (1 - (Abulk * Vdsat) / (Abulk * Vdsat + Vgsteff + 2 * self.vt(T)))
        
        # Substrate current induced body effect
        inv_VASCBE = (self.PSCE2 / self.LEFF) * np.exp(-self.PSCE1 * lit / (Vdseff - Vdsat))
        VASCBE = 1 / inv_VASCBE if inv_VASCBE != 0 else 1e20
        
        # Combine all effects
        VA = (1 + (self.PVAG * Vgsteff) / (Esat * self.LEFF)) * \
             ((1 / VACLM) + (1 / VADIBLC))**-1
        
        Ids = Idsat * (1 + (Vdseff - Vdsat) / VA) * \
              (1 + (Vdseff - Vdsat) / VASCBE)
        
        return Ids
    
    def calculate_Ig(self, Vgs, Vds, Vbs, T):
        """Calculate gate leakage current."""
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Vth = self.calculate_Vth(Vds, Vbs, T)
        Tox_ratio = self.TOX / self.TOXM
        
        # Accumulation region gate current
        if Vgs < Vth:
            Ig_acc = self.AIGBACC * Tox_ratio * \
                    (Vgs - Vth) * np.exp(-self.BIGBACC * self.TOX * 
                    (1 + self.CIGBACC * (Vgs - Vth))) * \
                    (Vgs - Vth)**self.NIGBACC
            
            Ig_inv = 0.0
        else:
            # Inversion region gate current
            Ig_inv = self.AIGBINV * Tox_ratio * \
                    (Vgs - Vth) * np.exp(-self.BIGBINV * self.TOX * 
                    (1 + self.CIGBINV * (Vgs - Vth))) * \
                    (Vgs - Vth)**self.NIGBINV
            
            Ig_acc = 0.0
        
        Ig = Ig_acc + Ig_inv
        return Ig
    
    def calculate_self_heating(self, Ids, Vds, T):
        """Calculate self-heating effect on device temperature."""
        power = Ids * Vds
        delta_T = power * self.RTH0
        Tdev = T + delta_T
        
        # Simple first-order thermal response
        self.TDEV = Tdev - (Tdev - self.TDEV) * np.exp(-1 / (self.RTH0 * self.CTH0))
        
        return self.TDEV
    
    def compute(self, Vgs, Vds, Vbs=0.0, T=300.0):
        """Main interface to calculate drain current."""
        # Calculate effective voltages
        Vgsteff = self.calculate_Vgsteff(Vgs, Vds, Vbs, T)
        Vdsat = self.calculate_Vdsat(Vgs, Vds, Vbs, T)
        Vdseff = self.calculate_Vdseff(Vds, Vgs, Vbs, T)
        
        # Calculate current based on operating region
        if Vgsteff <= 0:  # Subthreshold region
            # Use same calculation as BSIM3 but with updated parameters
            Vth = self.calculate_Vth(Vds, Vbs, T)
            Vgst = Vgs - Vth
            n = self.NFACTOR + (self.CIT + self.CDSC + self.CDSCD * Vds + 
                               self.CDSCB * Vbs) / self.COX
            
            Isub = self.U0 * (self.WEFF / self.LEFF) * np.sqrt(self.q * self.epsSi * self.NCH * 
                   (self.vt(T)**2) / (2 * self.Phi_s(T))) * \
                   (1 - np.exp(-Vds / self.vt(T))) * \
                   np.exp((Vgst - self.VOFF) / (n * self.vt(T)))
            
            Ids = Isub
        else:
            if Vdseff < Vdsat:  # Linear region
                Ids = self.calculate_Ids_linear(Vds, Vgs, Vbs, T)
            else:  # Saturation region
                Ids = self.calculate_Ids_saturation(Vds, Vgs, Vbs, T)
        
        # Calculate gate current if model is enabled
        if self.igMod == 1:
            Ig = self.calculate_Ig(Vgs, Vds, Vbs, T)
        else:
            Ig = 0.0
        
        # Calculate self-heating if enabled
        if self.RTH0 > 0:
            Tdev = self.calculate_self_heating(Ids, Vds, T)
        else:
            Tdev = T
        
        # Store state variables
        self.VGST = Vgsteff
        self.VDSAT = Vdsat
        self.VDSEFF = Vdseff
        self.IDS = Ids
        
        return Ids

# Main execution and plotting
if __name__ == "__main__":
    model = BSIMv6_Model()
    
    # Test conditions
    vds_range = np.linspace(0, 1.0, 100)  # Up to 1V for 7nm device
    vgs_range = np.linspace(0, 1.0, 100)  # Up to 1V for 7nm device
    temp_range = np.linspace(250, 400, 100)
    vbs_range = np.linspace(-0.5, 0, 5)  # Reduced range for 7nm
    
    # Create HTML file for plots
    html_file = r"D:\WORKSPACE\BSIM-Python-Implementation\data\bsimv6_plots.html"
    html_content = """
    <html>
    <head>
        <title>BSIMv6.0 Model Characterization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .plot-container {
                width: 100%;
                margin-bottom: 50px;
                border-bottom: 1px solid #ccc;
                padding-bottom: 20px;
            }
            h2 {
                color: #2c3e50;
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; font-family: Arial, sans-serif;">BSIMv6.0 Model Characterization (7nm FinFET)</h1>
    """
    
    # Plot 1: Id-Vds characteristics
    fig1 = go.Figure()
    for vgs in np.linspace(0.2, 1.0, 5):
        ids = [model.compute(vgs, vds, 0, 300) for vds in vds_range]
        fig1.add_trace(go.Scatter(
            x=vds_range, 
            y=np.array(ids)*1e6,  # Convert to µA
            name=f'Vgs={vgs:.1f}V',
            mode='lines'
        ))
    
    fig1.update_layout(
        title="Drain Current vs Drain Voltage",
        xaxis_title="Drain Voltage (Vds) [V]",
        yaxis_title="Drain Current (Ids) [µA]",
        legend_title="Gate Voltage"
    )
    
    html_content += """
    <div class="plot-container">
        <h2>Drain Current vs Drain Voltage</h2>
        {div1}
    </div>
    """.format(div1=fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # Plot 2: Id-Vgs characteristics (log scale)
    fig2 = go.Figure()
    for vds in [0.05, 0.5, 1.0]:
        ids = [model.compute(vgs, vds, 0, 300) for vgs in vgs_range]
        fig2.add_trace(go.Scatter(
            x=vgs_range, 
            y=np.maximum(1e-20, np.array(ids)*1e6),  # Convert to µA
            name=f'Vds={vds:.2f}V',
            mode='lines'
        ))
    
    fig2.update_layout(
        title="Drain Current vs Gate Voltage (log scale)",
        xaxis_title="Gate Voltage (Vgs) [V]",
        yaxis_title="Drain Current (Ids) [µA]",
        yaxis_type="log",
        legend_title="Drain Voltage"
    )
    
    html_content += """
    <div class="plot-container">
        <h2>Drain Current vs Gate Voltage (log scale)</h2>
        {div2}
    </div>
    """.format(div2=fig2.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # Plot 3: Temperature dependence
    fig3 = go.Figure()
    for vgs in np.linspace(0.4, 1.0, 4):
        ids = [model.compute(vgs, 0.5, 0, T) for T in temp_range]
        fig3.add_trace(go.Scatter(
            x=temp_range, 
            y=np.array(ids)*1e6,  # Convert to µA
            name=f'Vgs={vgs:.1f}V',
            mode='lines'
        ))
    
    fig3.update_layout(
        title="Temperature Dependence of Drain Current",
        xaxis_title="Temperature [K]",
        yaxis_title="Drain Current (Ids) [µA]",
        legend_title="Gate Voltage"
    )
    
    html_content += """
    <div class="plot-container">
        <h2>Temperature Dependence of Drain Current</h2>
        {div3}
    </div>
    """.format(div3=fig3.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # Plot 4: Body bias effect
    fig4 = go.Figure()
    for vbs in vbs_range:
        ids = [model.compute(vgs, 0.5, vbs, 300) for vgs in vgs_range]
        fig4.add_trace(go.Scatter(
            x=vgs_range, 
            y=np.array(ids)*1e6,  # Convert to µA
            name=f'Vbs={vbs:.1f}V',
            mode='lines'
        ))
    
    fig4.update_layout(
        title="Body Bias Effect on Drain Current",
        xaxis_title="Gate Voltage (Vgs) [V]",
        yaxis_title="Drain Current (Ids) [µA]",
        legend_title="Body Voltage"
    )
    
    html_content += """
    <div class="plot-container">
        <h2>Body Bias Effect on Drain Current</h2>
        {div4}
    </div>
    """.format(div4=fig4.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # Plot 5: Vth vs Vds
    vth_vds = [model.calculate_Vth(vds, 0, 300) for vds in vds_range]
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=vds_range,
        y=vth_vds,
        mode='lines'
    ))
    
    fig5.update_layout(
        title="Threshold Voltage vs Drain Voltage",
        xaxis_title="Drain Voltage (Vds) [V]",
        yaxis_title="Threshold Voltage (Vth) [V]"
    )
    
    html_content += """
    <div class="plot-container">
        <h2>Threshold Voltage vs Drain Voltage</h2>
        {div5}
    </div>
    """.format(div5=fig5.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # Close HTML tags
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Open in default browser
    webbrowser.open('file://' + os.path.realpath(html_file))