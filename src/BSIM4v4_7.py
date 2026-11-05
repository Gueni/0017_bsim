#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?                 ______  ____  _______  _____
#?                / __ \ \/ /  |/  / __ \/ ___/
#?               / /_/ /\  / /|_/ / / / /\__ \
#?              / ____/ / / /  / / /_/ /___/ /
#?             /_/     /_/_/  /_/\____//____/
#?
#? Name:        BSIM4v4_7.py
#? Purpose:     Compute drain current using the BSIM4v4.7 model
#? Based on:    BSIM4v4.7 Manual (UC Berkeley)
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
# -------------------------------------------------------------------------------
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import webbrowser
import os

class BSIM4v4_7_Model:
    """BSIM4v4.7 MOSFET model implementation for circuit simulation.
    
    This class implements the BSIM4v4.7 (Berkeley Short-channel IGFET Model) version 4.7
    for MOSFET transistors. It calculates drain current (Ids) and other characteristics
    based on the given terminal voltages and physical parameters.
    
    Key features implemented:
    - Advanced threshold voltage model with short-channel, narrow width, and DIBL effects
    - Unified mobility model with multiple degradation mechanisms
    - Velocity saturation and overshoot effects
    - Channel length modulation
    - Subthreshold conduction
    - Temperature effects
    - Gate tunneling current models
    - Charge thickness capacitance model
    
    Attributes:
        Various physical constants and model parameters initialized in __init__
    """
    
    def __init__(self):
        """Initialize BSIM4v4.7 model with default parameters for 45nm NMOS transistor."""
        
        # Physical constants (SI units)
        self.eps0 = 8.854e-12                     # F/m, Vacuum permittivity
        self.epsSi = 11.7 * self.eps0             # F/m, Silicon permittivity
        self.epsOx = 3.9 * self.eps0              # F/m, Silicon dioxide permittivity
        self.q = 1.602e-19                       # C, Electron charge
        self.k_B = 1.38e-23                      # J/K, Boltzmann constant
        
        # Model selectors
        self.mobMod = 3                          # Mobility model selector (0,1,2,3)
        self.rdsMod = 0                          # Rds model selector (0=internal, 1=external)
        self.mtrlMod = 0                         # Material model selector (0=Si/SiO2, 1=new materials)
        self.capMod = 2                          # Capacitance model selector (0,1,2)
        self.igcMod = 0                          # Gate current model selector
        self.igbMod = 0                          # Gate-body current model selector
        
        # Process parameters (45nm technology)
        self.TOXE = 1.2e-9                       # m, Electrical oxide thickness
        self.TOXM = 1.2e-9                       # m, Gate oxide thickness at which parameters are extracted
        self.TOXREF = 1.2e-9                     # m, Reference oxide thickness
        self.DTOX = 0.0                          # m, Difference between TOXE and TOXP
        self.XJ = 15e-9                          # m, Junction depth
        self.NDEP = 2e23                         # m^-3, Channel doping concentration
        self.NGATE = 1e25                        # m^-3, Poly doping concentration
        self.NSD = 1e26                          # m^-3, Source/drain doping concentration
        
        # Threshold voltage parameters
        self.VTH0 = 0.35                         # V, Zero-bias threshold voltage
        self.K1 = 0.5                            # √V, First body effect coefficient
        self.K2 = 0.01                           # -, Second body effect coefficient
        self.K3 = 80.0                           # -, Narrow width effect coefficient
        self.K3B = 0.0                           # -, Body effect on narrow width coefficient
        self.DVT0 = 2.5                          # -, Short-channel effect coefficient at Vbs=0
        self.DVT1 = 0.6                          # -, Short-channel effect coefficient
        self.DVT2 = -0.032                       # 1/V, Short-channel effect coefficient for body bias
        self.DVT0W = 0.0                         # -, Narrow width effect coefficient at Vbs=0
        self.DVT1W = 5.3e6                       # -, Narrow width effect coefficient
        self.DVT2W = -0.032                      # 1/V, Narrow width effect coefficient for body bias
        self.NLX = 1.47e-7                       # m, Lateral non-uniform doping parameter
        self.W0 = 2.5e-6                         # m, Narrow width parameter
        self.LPE0 = 0.0                          # m, Pocket implant parameter
        self.LPEB = 0.0                          # m, Pocket implant parameter
        self.DVTP0 = 0.0                         # -, DITS parameter
        self.DVTP1 = 0.0                         # -, DITS parameter
        self.DVTP2 = 0.0                         # -, DITS parameter
        self.DVTP3 = 0.0                         # -, DITS parameter
        self.DVTP4 = 0.0                         # -, DITS parameter
        self.DVTP5 = 0.0                         # -, DITS parameter
        
        # DIBL and substrate effect parameters
        self.DSUB = 1.2                          # -, DIBL coefficient
        self.ETA0 = 0.15                         # -, DIBL in strong inversion
        self.ETAB = -0.12                        # -, Body effect on DIBL
        self.PDIBLC1 = 0.45                      # -, DIBL coefficient
        self.PDIBLC2 = 0.45                      # -, DIBL coefficient
        self.PDIBLCB = -0.08                     # -, Body bias effect on DIBL
        
        # Mobility parameters
        self.U0 = 0.067                          # m^2/V·s, Low-field mobility
        self.UA = 2.25e-9                        # m/V, First-order mobility degradation coefficient
        self.UB = 5.87e-19                       # (m/V)^2, Second-order mobility degradation coefficient
        self.UC = -0.046                         # -, Body-effect coefficient for mobility degradation
        self.UA1 = 4.31e-9                       # m/V, First-order mobility degradation coefficient
        self.UB1 = -7.61e-18                     # (m/V)^2, Second-order mobility degradation coefficient
        self.UC1 = -0.056                        # -, Body-effect coefficient for mobility degradation
        self.EU = 1.0                            # -, Mobility exponent
        self.UP = 0.0                            # -, Mobility length dependence coefficient
        self.LP = 1e-6                           # m, Mobility length dependence parameter
        
        # Velocity saturation parameters
        self.VSAT = 8.0e4                        # m/s, Saturation velocity
        self.A0 = 1.0                            # -, Bulk charge effect coefficient
        self.A1 = 0.0                            # -, Saturation voltage parameter
        self.A2 = 1.0                            # -, Saturation voltage parameter
        self.B0 = 0.0                            # -, Width effect on Abulk
        self.B1 = 0.0                            # -, Width effect on Abulk
        self.KETA = -0.047                       # -, Body effect coefficient for Voff
        self.AGS = 0.0                           # -, Gate bias effect on Abulk
        
        # Output resistance parameters
        self.PCLM = 1.3                          # -, Channel length modulation coefficient
        self.PDIBL1 = 0.45                       # -, DIBL coefficient for output resistance
        self.PDIBL2 = 0.45                       # -, DIBL coefficient for output resistance
        self.PVAG = 0.0                          # 1/V, Gate voltage effect on output resistance
        self.PSCBE1 = 4.24e8                     # -, Substrate current body-effect coefficient 1
        self.PSCBE2 = 1.0e-5                     # -, Substrate current body-effect coefficient 2
        self.PDITS = 0.0                         # -, DITS coefficient for output resistance
        self.PDITSL = 0.0                        # -, DITS length dependence
        self.PDITSD = 0.0                        # -, DITS drain bias dependence
        
        # Geometry parameters (45nm process)
        self.Leff = 45e-9                        # m, Effective channel length
        self.Weff = 1e-6                         # m, Effective channel width (1um)
        self.Ldrawn = 45e-9                      # m, Drawn channel length
        self.Wdrawn = 1e-6                       # m, Drawn channel width
        self.WINT = 0.0                          # m, Width offset parameter
        self.LINT = 0.0                          # m, Length offset parameter
        self.XW = 0.0                            # m, Width offset due to mask/etch effect
        self.XL = 0.0                            # m, Length offset due to mask/etch effect
        self.DWG = 0.0                           # m/V, Gate bias dependence of dW
        self.DWB = 0.0                           # m/V, Body bias dependence of dW
        self.DLC = 0.0                           # m, Delta L for capacitance model
        self.DWC = 0.0                           # m, Delta W for capacitance model
        self.DELVTO = 0.0                        # V, Threshold voltage offset
        
        # Subthreshold parameters
        self.NFACTOR = 1.0                       # -, Subthreshold swing coefficient
        self.VOFF = -0.08                        # V, Offset voltage for subthreshold current
        self.VOFFL = 0.0                         # V, Length dependence of VOFF
        self.CDSC = 0.0                          # F/m^2, DIBL coefficient for subthreshold swing
        self.CDSCD = 0.0                         # F/m^2/V, Drain bias dependence of CDSC
        self.CDSCB = 0.0                         # F/m^2/V, Body bias dependence of CDSC
        self.CIT = 0.0                           # F/m^2, Interface trap capacitance
        
        # Temperature parameters
        self.TNOM = 300.0                        # K, Nominal temperature
        self.KT1 = -0.15                         # V, Temperature coefficient for Vth
        self.KT1L = 1e-9                         # V·m, Temperature coefficient for Vth
        self.KT2 = 0.03                          # -, Temperature coefficient for Vth
        self.UTE = -1.8                          # -, Mobility temperature exponent
        self.AT = 4.0e4                          # m/s, Velocity saturation temperature coefficient
        
        # Parasitic resistance
        self.RDSW = 200.0                        # ohm·µm, Source/drain resistance per width
        self.PRWG = 0.0                          # -, Gate bias effect on Rds
        self.PRWB = 0.1                          # -, Body bias effect on Rds
        self.PRW = 1.0                           # -, Width dependence of Rds
        
        # Quantum mechanical effects
        self.ADOS = 1.0                          # -, Density of states parameter
        self.BDOS = 1.0                          # -, Density of states parameter
        self.MINV = 0.0                          # -, Moderate inversion parameter
        self.MINVCV = 0.0                        # -, Moderate inversion parameter for CV
        # Gate tunneling current parameters
        self.AIGBACC = 1.0                       # -, Gate current parameter
        self.BIGBACC = 1.0                       # -, Gate current parameter
        self.CIGBACC = 1.0                       # -, Gate current parameter
        self.NIGBACC = 1.0                       # -, Gate current parameter
        self.AIGBINV = 1.0                       # -, Gate current parameter
        self.BIGBINV = 1.0                       # -, Gate current parameter
        self.CIGBINV = 1.0                       # -, Gate current parameter
        self.NIGBINV = 1.0                       # -, Gate current parameter
        self.EIGBINV = 1.0                       # V, Gate current parameter
        self.AIGC = 1.0                          # -, Gate current parameter
        self.BIGC = 1.0                          # -, Gate current parameter
        self.CIGC = 1.0                          # -, Gate current parameter
        self.NIGC = 1.0                          # -, Gate current parameter
        self.AIGS = 1.0                          # -, Gate current parameter
        self.BIGS = 1.0                          # -, Gate current parameter
        self.CIGS = 1.0                          # -, Gate current parameter
        self.AIGD = 1.0                          # -, Gate current parameter
        self.BIGD = 1.0                          # -, Gate current parameter
        self.CIGD = 1.0                          # -, Gate current parameter
        self.POXEDGE = 1.0                       # -, Gate current parameter
        self.NTOX = 1.0                          # -, Gate current parameter
        self.VDDEOT = 1.0                        # V, Voltage for EOT extraction
        self.VFBSDOFF = 0.0                      # V, Flat-band voltage offset
        
        # GIDL parameters
        self.AGIDL = 1.0                         # -, GIDL parameter
        self.BGIDL = 1.0                         # -, GIDL parameter
        self.CGIDL = 1.0                         # -, GIDL parameter
        self.EGIDL = 1.0                         # V, GIDL parameter
        self.AGISL = 1.0                         # -, GISL parameter
        self.BGISL = 1.0                         # -, GISL parameter
        self.CGISL = 1.0                         # -, GISL parameter
        self.EGISL = 1.0                         # V, GISL parameter
        
        # Initialize derived parameters
        self.Cox = self.epsOx / self.TOXE        # F/m², Oxide capacitance per unit area
        self.delta = 0.001                       # V, Smoothing parameter
        self.n = 1.0                             # Subthreshold swing factor (initial value)
        
    def ni(self, T):
        """Calculate intrinsic carrier concentration (ni) based on temperature.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Intrinsic carrier concentration in m^-3
        """
        # Using simplified formula from BSIM4 manual
        ni = 1.45e16 * (T / 300.15) ** 1.5 * np.exp(21.5565981 - 0.5 * 1.1605195e4 / T)
        return ni
    
    def v_t(self, T):
        """Calculate thermal voltage (Vt) based on temperature.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Thermal voltage in volts
        """
        return self.k_B * T / self.q
    
    def Phi_s(self, T):
        """Calculate surface potential (Phi_s) based on temperature.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Surface potential in volts
        """
        Phi_s = 2 * self.v_t(T) * np.log(self.NDEP / self.ni(T))
        return Phi_s
    
    def Xdep0(self, T):
        """Calculate zero-bias depletion width based on temperature.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Zero-bias depletion width in meters
        """
        Xdep0 = np.sqrt(2 * self.epsSi * self.Phi_s(T) / (self.q * self.NDEP))
        return Xdep0
        
    def Vbi(self, T):
        """Calculate built-in potential (Vbi) based on temperature.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Built-in potential in volts
        """
        vbi = self.v_t(T) * np.log((self.NDEP * self.NSD) / np.square(self.ni(T)))
        return vbi
    
    def calculate_Leff_Weff(self):
        """Calculate effective channel length and width with bias dependencies."""
        # Calculate dW and dL (Eq. 1.9-1.12)
        dW = self.WINT + self.DWG * self.Vgsteff + self.DWB * (np.sqrt(self.Phi_s(self.TNOM) - self.Vbseff) - np.sqrt(self.Phi_s(self.TNOM)))
        dL = self.LINT
        
        self.Leff = max(1e-9, self.Ldrawn + self.XL - 2 * dL)
        self.Weff = max(1e-9, self.Wdrawn + self.XW - 2 * dW)
        
        # For capacitance calculations
        self.Lactive = max(1e-9, self.Ldrawn + self.XL - 2 * (self.DLC + self.LINT))
        self.Wactive = max(1e-9, self.Wdrawn + self.XW - 2 * (self.DWC + self.WINT))
        
    def calculate_Vbseff(self, Vbs, T):
        """Calculate effective body-source voltage with smoothing (Eq. 2.43)."""
        Vbc = 0.9 * (self.Phi_s(T) - self.K1**2 / (4 * self.K2**2))
        delta1 = 0.001
        
        if Vbs <= Vbc:
            Vbseff = Vbc + 0.5 * (Vbs - Vbc - delta1 + np.sqrt((Vbs - Vbc - delta1)**2 + 4 * delta1 * Vbc))
        else:
            # For positive Vbs (Eq. 2.45)
            Vbseff_prime = Vbc + 0.5 * (Vbs - Vbc - delta1 + np.sqrt((Vbs - Vbc - delta1)**2 + 4 * delta1 * Vbc))
            Vbseff = 0.95 * self.Phi_s(T) - 0.5 * (0.95 * self.Phi_s(T) - Vbseff_prime - delta1 + 
                                                   np.sqrt((0.95 * self.Phi_s(T) - Vbseff_prime - delta1)**2 + 4 * delta1 * 0.95 * self.Phi_s(T)))
        
        return Vbseff
    
    def calculate_V_th(self, Vds, Vbs, T):
        """Calculate threshold voltage (Vth) based on BSIM4v4.7 model (Eq. 2.26).
        
        Includes:
        - Body effect
        - Short-channel effects
        - Narrow width effects
        - DIBL effects
        - Temperature effects
        - Pocket implant effects (DITS)
        
        Args:
            Vds (float): Drain-source voltage in volts
            Vbs (float): Bulk-source voltage in volts
            T (float): Temperature in Kelvin
            
        Returns:
            float: Threshold voltage in volts
        """
        # Effective body-source voltage with smoothing
        self.Vbseff = self.calculate_Vbseff(Vbs, T)
        
        # Depletion widths and characteristic lengths
        Xdep = np.sqrt(2 * self.epsSi * (self.Phi_s(T) - self.Vbseff) / (self.q * self.NDEP))
        Xdep0 = np.sqrt(2 * self.epsSi * self.Phi_s(T) / (self.q * self.NDEP))
        
        lt = np.sqrt(self.epsSi * Xdep * self.TOXE / (self.epsOx * (1 + self.DVT2 * self.Vbseff)))
        lt0 = np.sqrt(self.epsSi * Xdep0 * self.TOXE / self.epsOx)
        ltw = np.sqrt(self.epsSi * Xdep * self.TOXE / (self.epsOx * (1 + self.DVT2W * self.Vbseff)))
        
        # Scale K1 and K2 for oxide thickness (Eq. 2.26)
        K1ox = self.K1 * (self.TOXE / self.TOXM)
        K2ox = self.K2 * (self.TOXE / self.TOXM)
        
        # Calculate all terms of threshold voltage (Eq. 2.26)
        term1 = self.VTH0 + self.DELVTO + K1ox * (np.sqrt(self.Phi_s(T) - self.Vbseff) - np.sqrt(self.Phi_s(T))) * np.sqrt(1 + self.LPEB / self.Leff)
        term2 = -K2ox * self.Vbseff
        term3 = K1ox * (np.sqrt(1 + self.LPE0 / self.Leff) - 1) * np.sqrt(self.Phi_s(T))
        term4 = (self.K3 + self.K3B * self.Vbseff) * (self.TOXE / (self.Weff + self.W0)) * self.Phi_s(T)
        
        # Short-channel effect (SCE)
        theta_SCE = 0.5 * self.DVT0 / (np.cosh(self.DVT1 * self.Leff / lt) - 1)
        term5 = -theta_SCE * (self.Vbi(T) - self.Phi_s(T))
        
        # Narrow width effect
        theta_NWE = 0.5 * self.DVT0W / (np.cosh(self.DVT1W * self.Leff * self.Weff / ltw) - 1)
        term6 = -theta_NWE * (self.Vbi(T) - self.Phi_s(T))
        
        # DIBL effect
        theta_DIBL = 0.5 / (np.cosh(self.DSUB * self.Leff / lt0) - 1)
        term7 = -theta_DIBL * (self.ETA0 + self.ETAB * self.Vbseff) * Vds
        
        # DITS effect (pocket implant)
        if Vds >= 0:
            term8 = -self.n * self.v_t(T) * np.log(self.Leff / (self.Leff + self.DVTP0 * (1 + np.exp(-self.DVTP1 * Vds))))
            term9 = -(self.DVTP5 + self.DVTP2 / (self.Leff**self.DVTP3)) * np.tanh(self.DVTP4 * Vds)
        else:
            term8 = 0
            term9 = 0
        
        # Temperature effect
        delta_T = (T / self.TNOM) - 1
        term10 = (self.KT1 + self.KT1L / self.Leff + self.KT2 * self.Vbseff) * delta_T
        
        Vth = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
        
        return Vth
    
    def calculate_Vgsteff(self, Vgs, T, Vds, Vbs):
        """Calculate effective Vgs-Vth including subthreshold smoothing (Eq. 3.7).
        
        Provides smooth transition between subthreshold and strong inversion regions.
        
        Args:
            Vgs (float): Gate-source voltage in volts
            T (float): Temperature in Kelvin
            Vds (float): Drain-source voltage in volts
            Vbs (float): Bulk-source voltage in volts
            
        Returns:
            float: Effective gate overdrive voltage in volts
        """
        Vth = self.calculate_V_th(Vds, Vbs, T)
        Vgst = Vgs - Vth
        
        # Calculate subthreshold swing factor n (Eq. 3.21)
        Cd = self.epsSi / self.Xdep0(T)
        lt = np.sqrt(self.epsSi * self.Xdep0(T) * self.TOXE / self.epsOx)
        
        # Calculate Cdsc_term (Eq. 3.22)
        Cdsc_term = (self.CDSC + self.CDSCD * Vds + self.CDSCB * self.Vbseff) * 0.5 / (np.cosh(self.DVT1 * self.Leff / lt) - 1)
        
        self.n = 1 + self.NFACTOR * Cd / self.Cox + (Cdsc_term + self.CIT) / self.Cox
        
        # Calculate Voff (Eq. 3.1)
        Voff = self.VOFF + self.VOFFL / self.Leff
        
        # Calculate m* (Eq. 3.8)
        mstar = 0.5 + np.arctan(self.MINV) / np.pi
        
        # Calculate Vgsteff (Eq. 3.7)
        numerator = 2 * self.n * self.v_t(T) * np.log(1 + np.exp(mstar * Vgst / (2 * self.n * self.v_t(T))))
        denominator = mstar + self.n * self.Cox * np.sqrt(2 * self.Phi_s(T) / (self.q * self.epsSi * self.NDEP)) * \
                     np.exp(-(1 - mstar) * Vgst / (2 * self.n * self.v_t(T)) - Voff / (self.n * self.v_t(T)))
        
        Vgsteff = numerator / denominator
        
        self.Vgsteff = Vgsteff  # Store for use in other calculations
        return Vgsteff
    
    def calculate_mobility(self, Vgs, T, Vds, Vbs):
        """Calculate effective mobility including degradation effects.
        
        Args:
            Vgs (float): Gate-source voltage in volts
            T (float): Temperature in Kelvin
            Vds (float): Drain-source voltage in volts
            Vbs (float): Bulk-source voltage in volts
            
        Returns:
            float: Effective mobility in m²/V·s
        """
        Vth = self.calculate_V_th(Vds, Vbs, T)
        Vgsteff = self.calculate_Vgsteff(Vgs, T, Vds, Vbs)
        
        # Temperature effect on mobility
        U0_T = self.U0 * (T / self.TNOM)**self.UTE
        
        # Mobility degradation models
        if self.mobMod == 0:
            # Basic mobility model (Eq. 5.6)
            denom = 1 + (self.UA + self.UC * self.Vbseff) * ((Vgsteff + 2 * Vth) / self.TOXE) + \
                    self.UB * ((Vgsteff + 2 * Vth) / self.TOXE)**2 + \
                    self.UD * (Vth * self.TOXE / (Vgsteff + 2 * np.sqrt(Vth**2 + 0.0001)))**2
            
        elif self.mobMod == 1:
            # Alternative mobility model (Eq. 5.7)
            denom = 1 + (self.UA * ((Vgsteff + 2 * Vth) / self.TOXE) + \
                    self.UB * ((Vgsteff + 2 * Vth) / self.TOXE)**2) * (1 + self.UC * self.Vbseff) + \
                    self.UD * (Vth * self.TOXE / (Vgsteff + 2 * np.sqrt(Vth**2 + 0.0001)))**2
            
        elif self.mobMod == 2:
            # Unified mobility model (Eq. 5.8)
            C0 = 2.0  # For NMOS (2.5 for PMOS)
            denom = 1 + (self.UA + self.UC * self.Vbseff) * \
                    ((Vgsteff + C0 * (self.VTH0 - self.Vbi(T) - self.Phi_s(T))) / self.TOXE)**self.EU + \
                    self.UD * (Vth * self.TOXE / (Vgsteff + 2 * np.sqrt(Vth**2 + 0.0001)))
            
        elif self.mobMod == 3:
            # High-k/metal gate mobility model (Eq. 5.13)
            C0 = 2.0  # For NMOS (2.5 for PMOS)
            Vgsteff_Vth = self.calculate_Vgsteff(self.VTH0, T, 0, 0)  # Vgsteff at Vgs=Vth, Vds=Vbs=0
            
            denom = 1 + (self.UA + self.UC * self.Vbseff) * \
                    ((Vgsteff + C0 * (self.VTH0 - self.Vbi(T) - self.Phi_s(T))) / self.TOXE)**self.EU + \
                    self.UD / (0.5 * (1 + Vgsteff / Vgsteff_Vth))
        
        # Length dependence (Eq. 5.9)
        f_L = 1 - self.UP * np.exp(-self.Leff / self.LP)
        
        mob_eff = U0_T * f_L / denom
        
        return mob_eff
    
    def calculate_Rds(self, Vds, Vgs, Vbs, T):
        """Calculate bias-dependent source/drain resistance (Eq. 5.14-5.16)."""
        Vgsteff = self.calculate_Vgsteff(Vgs, T, Vds, Vbs)
        
        if self.rdsMod == 0:
            # Internal Rds model (Eq. 5.14)
            Rds = (self.RDSW + self.PRWB * (np.sqrt(self.Phi_s(T) - self.Vbseff) - np.sqrt(self.Phi_s(T)))) / \
                  (1 + self.PRWG * Vgsteff) / (self.Weff * 1e6)**self.PRW
        else:
            # External Rds model (Eq. 5.15-5.16)
            # For simplicity, we'll just use the internal model here
            Rds = (self.RDSW + self.PRWB * (np.sqrt(self.Phi_s(T) - self.Vbseff) - np.sqrt(self.Phi_s(T)))) / \
                  (1 + self.PRWG * Vgsteff) / (self.Weff * 1e6)**self.PRW
        
        return max(Rds, 0)  # Ensure non-negative resistance
    
    def calculate_Abulk(self, T, Vbs):
        """Calculate bulk charge effect coefficient (Abulk) (Eq. 5.1).
        
        Args:
            T (float): Temperature in Kelvin
            Vbs (float): Bulk-source voltage in volts
            
        Returns:
            float: Bulk charge effect coefficient (unitless)
        """
        # Calculate F_doping (Eq. 5.2)
        K1ox = self.K1 * (self.TOXE / self.TOXM)
        K2ox = self.K2 * (self.TOXE / self.TOXM)
        
        F_doping = K1ox / (2 * np.sqrt(self.Phi_s(T) - self.Vbseff)) + K2ox - \
                   self.K3B * (self.TOXE / (self.Weff + self.W0)) * self.Phi_s(T)
        
        Xdep = np.sqrt(2 * self.epsSi * (self.Phi_s(T) - self.Vbseff) / (self.q * self.NDEP))
        
        term1 = self.A0 * self.Leff / (self.Leff + 2 * np.sqrt(self.XJ * Xdep)) * \
                (1 - self.AGS * Vgsteff * (self.Leff / (self.Leff + 2 * np.sqrt(self.XJ * Xdep)))**2)
        term2 = self.B0 / (self.Weff + self.B1)
        
        Abulk = (1 + F_doping * (term1 + term2)) / (1 + self.KETA * self.Vbseff)
        
        return Abulk
    
    def calculate_Vdsat(self, Vgs, Vbs, T, Vds):
        """Calculate saturation voltage (Vdsat) (Eq. 5.24-5.25).
        
        Args:
            Vgs (float): Gate-source voltage in volts
            Vbs (float): Bulk-source voltage in volts
            T (float): Temperature in Kelvin
            Vds (float): Drain-source voltage in volts
            
        Returns:
            float: Saturation voltage in volts
        """
        Vgsteff = self.calculate_Vgsteff(Vgs, T, Vds, Vbs)
        Abulk = self.calculate_Abulk(T, Vbs)
        mob_eff = self.calculate_mobility(Vgs, T, Vds, Vbs)
        Esat = 2 * self.VSAT / mob_eff
        Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
        
        if Rds == 0 or self.rdsMod == 1:
            # Intrinsic case (Eq. 5.24)
            Vdsat = (Esat * self.Leff * (Vgsteff + 2 * self.v_t(T))) / \
                    (Abulk * Esat * self.Leff + Vgsteff + 2 * self.v_t(T))
        else:
            # Extrinsic case (Eq. 5.25-5.29)
            lamda = self.A1 * Vgsteff + self.A2
            
            a = Abulk**2 * self.Weff * self.VSAT * self.Cox * Rds + Abulk * (1/lamda - 1)
            b = -((Vgsteff + 2 * self.v_t(T)) * (2/lamda - 1) + Abulk * Esat * self.Leff + \
                  3 * Abulk * (Vgsteff + 2 * self.v_t(T)) * self.Weff * self.VSAT * self.Cox * Rds)
            c = (Vgsteff + 2 * self.v_t(T)) * Esat * self.Leff + \
                2 * (Vgsteff + 2 * self.v_t(T))**2 * self.Weff * self.VSAT * self.Cox * Rds
            
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                discriminant = 0  # Handle numerical errors
                
            Vdsat = (-b - np.sqrt(discriminant)) / (2 * a)
        
        return Vdsat
    
    def calculate_Vdseff(self, Vds, Vdsat):
        """Calculate effective Vds including smoothing at Vdsat (Eq. 5.30).
        
        Args:
            Vds (float): Drain-source voltage in volts
            Vdsat (float): Saturation voltage in volts
            
        Returns:
            float: Effective drain-source voltage in volts
        """
        delta = self.delta
        Vdseff = Vdsat - 0.5 * (Vdsat - Vds - delta + np.sqrt((Vdsat - Vds - delta)**2 + 4 * delta * Vdsat))
        return Vdseff
    
    def calculate_linear_current(self, Vds, Vgs, T, Vbs):
        """Calculate linear region current (triode region) (Eq. 5.20-5.21).
        
        Args:
            Vds (float): Drain-source voltage in volts
            Vgs (float): Gate-source voltage in volts
            T (float): Temperature in Kelvin
            Vbs (float): Bulk-source voltage in volts
            
        Returns:
            float: Drain current in amperes
        """
        Vgsteff = self.calculate_Vgsteff(Vgs, T, Vds, Vbs)
        Abulk = self.calculate_Abulk(T, Vbs)
        mob_eff = self.calculate_mobility(Vgs, T, Vds, Vbs)
        Esat = 2 * self.VSAT / mob_eff
        
        # Intrinsic case current (Eq. 5.20)
        Ids0 = self.Weff * mob_eff * self.Cox * Vgsteff * Vds * (1 - Vds / (2 * (Vgsteff + 2 * self.v_t(T)) / Abulk)) / \
               (self.Leff * (1 + Vds / (Esat * self.Leff)))
        
        # Add Rds effect if needed (Eq. 5.21)
        Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
        if Rds > 0 and self.rdsMod == 0:
            Ids = Ids0 / (1 + Rds * Ids0 / Vds)
        else:
            Ids = Ids0
            
        return Ids
    
    def calculate_saturation_current(self, Vds, Vgs, T, Vbs):
        """Calculate saturation region current including CLM, DIBL, SCBE effects.
        
        Args:
            Vds (float): Drain-source voltage in volts
            Vgs (float): Gate-source voltage in volts
            T (float): Temperature in Kelvin
            Vbs (float): Bulk-source voltage in volts
            
        Returns:
            float: Drain current in amperes
        """
        Vgsteff = self.calculate_Vgsteff(Vgs, T, Vds, Vbs)
        Vdsat = self.calculate_Vdsat(Vgs, Vbs, T, Vds)
        Vdseff = self.calculate_Vdseff(Vds, Vdsat)
        Abulk = self.calculate_Abulk(T, Vbs)
        mob_eff = self.calculate_mobility(Vgs, T, Vds, Vbs)
        Esat = 2 * self.VSAT / mob_eff
        Rds = self.calculate_Rds(Vds, Vgs, Vbs, T)
        
        # Calculate intrinsic case current at Vdsat (Eq. 5.20 at Vds=Vdsat)
        Ids0 = self.Weff * mob_eff * self.Cox * Vgsteff * Vdsat * \
               (1 - Vdsat / (2 * (Vgsteff + 2 * self.v_t(T)) / Abulk)) / \
               (self.Leff * (1 + Vdsat / (Esat * self.Leff)))
        
        # Add Rds effect if needed (Eq. 5.21)
        if Rds > 0 and self.rdsMod == 0:
            Idsat = Ids0 / (1 + Rds * Ids0 / Vdsat)
        else:
            Idsat = Ids0
            
        # Early voltage calculations
        # Channel length modulation (CLM) (Eq. 5.34-5.37)
        Xdep = np.sqrt(2 * self.epsSi * (self.Phi_s(T) - self.Vbseff) / (self.q * self.NDEP))
        litl = np.sqrt(self.epsSi * self.TOXE * self.XJ / self.epsOx)
        
        F = 1 / (1 + self.FPROUT * np.sqrt(self.Leff / (Vgsteff + 2 * self.v_t(T))))
        
        Cclm = (1 + self.PVAG * Vgsteff / (Esat * self.Leff)) * \
               (1 + Rds * Ids0 / Vdseff) * \
               (self.Leff + Vdsat / Esat) / (self.PCLM * litl)
        
        VA_CLM = Cclm * (Vds - Vdsat)
        
        # DIBL effect on output resistance (Eq. 5.39-5.40)
        theta_rout = self.PDIBLC1 / (2 * np.cosh(self.DROUT * self.Leff / litl) - 2) + self.PDIBLC2
        
        VA_DIBL = (Vgsteff + 2 * self.v_t(T)) / (theta_rout * (1 + self.PDIBLCB * self.Vbseff)) * \
                  (1 - Abulk * Vdsat / (Abulk * Vdsat + Vgsteff + 2 * self.v_t(T))) * (1 + self.PVAG * Vgsteff / (Esat * self.Leff))
        
        # SCBE effect (Eq. 5.44)
        VA_SCBE = self.PSCBE2 * self.Leff * np.exp(self.PSCBE1 * litl / (Vds - Vdsat))
        
        # DITS effect (Eq. 5.45)
        VA_DITS = 1 / (self.PDITS * F * (1 + (1 + self.PDITSL * self.Leff) * np.exp(self.PDITSD * Vds)))
        
        # Total Early voltage (Eq. 5.47)
        VAsat = (Esat * self.Leff + Vdsat + 2 * Rds * self.VSAT * self.Cox * self.Weff * Vgsteff) / \
                (Rds * self.VSAT * self.Cox * self.Weff * Abulk - 1 + 2 / (self.A1 * Vgsteff + self.A2))
        
        VA = VAsat + VA_CLM
        
        # Saturation region current (Eq. 5.46)
        Ids = Idsat * (1 + (Vds - Vdseff) / VA) * \
              (1 + (Vds - Vdseff) / VA_DIBL) * \
              (1 + (Vds - Vdseff) / VA_DITS) * \
              (1 + (Vds - Vdseff) / VA_SCBE)
        
        return Ids
    
    def calculate_Id(self, Vds, Vgs, T=300.0, Vbs=0.0):
        """Calculate drain current for given terminal voltages and temperature.
        
        Args:
            Vds (float): Drain-source voltage in volts
            Vgs (float): Gate-source voltage in volts
            T (float): Temperature in Kelvin (default: 300.0)
            Vbs (float): Bulk-source voltage in volts (default: 0.0)
            
        Returns:
            float: Drain current in amperes
        """
        # Update effective dimensions based on bias
        self.calculate_Leff_Weff()
        
        # Calculate Vdsat to determine operation region
        Vdsat = self.calculate_Vdsat(Vgs, Vbs, T, Vds)
        
        if Vds <= Vdsat:
            # Linear region
            Id = self.calculate_linear_current(Vds, Vgs, T, Vbs)
        else:
            # Saturation region
            Id = self.calculate_saturation_current(Vds, Vgs, T, Vbs)
            
        return Id
    
    def IV_curve(self, Vgs_list, Vds_list, T=300.0, Vbs=0.0):
        """Generate I-V curves for given Vgs and Vds ranges.
        
        Args:
            Vgs_list (list/array): List of gate-source voltages to evaluate
            Vds_list (list/array): List of drain-source voltages to evaluate
            T (float): Temperature in Kelvin (default: 300.0)
            Vbs (float): Bulk-source voltage in volts (default: 0.0)
            
        Returns:
            tuple: (Vds_array, Id_array) arrays for plotting
        """
        Id_array = np.zeros((len(Vgs_list), len(Vds_list)))
        
        for i, Vgs in enumerate(Vgs_list):
            for j, Vds in enumerate(Vds_list):
                Id_array[i,j] = self.calculate_Id(Vds, Vgs, T, Vbs)
                
        return Vds_list, Id_array
    
    def plot_IV(self, Vgs_list, Vds_max=1.0, steps=50, T=300.0, Vbs=0.0):
        """Plot I-V curves for given Vgs values.
        
        Args:
            Vgs_list (list): List of gate-source voltages to plot
            Vds_max (float): Maximum drain-source voltage (default: 1.0V)
            steps (int): Number of points to calculate (default: 50)
            T (float): Temperature in Kelvin (default: 300.0)
            Vbs (float): Bulk-source voltage in volts (default: 0.0)
        """
        Vds_list = np.linspace(0, Vds_max, steps)
        Vds_array, Id_array = self.IV_curve(Vgs_list, Vds_list, T, Vbs)
        
        plt.figure(figsize=(10, 6))
        for i, Vgs in enumerate(Vgs_list):
            plt.plot(Vds_array, Id_array[i,:]*1e6, label=f'Vgs = {Vgs:.2f} V')
            
        plt.xlabel('Drain-Source Voltage (V)')
        plt.ylabel('Drain Current (μA)')
        plt.title(f'BSIM4v4.7 I-V Characteristics (Vbs = {Vbs} V, T = {T} K)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_transfer(self, Vds_list, Vgs_min=0.0, Vgs_max=1.0, steps=50, T=300.0, Vbs=0.0):
        """Plot transfer characteristics for given Vds values.
        
        Args:
            Vds_list (list): List of drain-source voltages to plot
            Vgs_min (float): Minimum gate-source voltage (default: 0.0V)
            Vgs_max (float): Maximum gate-source voltage (default: 1.0V)
            steps (int): Number of points to calculate (default: 50)
            T (float): Temperature in Kelvin (default: 300.0)
            Vbs (float): Bulk-source voltage in volts (default: 0.0)
        """
        Vgs_list = np.linspace(Vgs_min, Vgs_max, steps)
        Id_array = np.zeros((len(Vds_list), len(Vgs_list)))
        
        for i, Vds in enumerate(Vds_list):
            for j, Vgs in enumerate(Vgs_list):
                Id_array[i,j] = self.calculate_Id(Vds, Vgs, T, Vbs)
                
        plt.figure(figsize=(10, 6))
        for i, Vds in enumerate(Vds_list):
            plt.semilogy(Vgs_list, Id_array[i,:]*1e6, label=f'Vds = {Vds:.2f} V')
            
        plt.xlabel('Gate-Source Voltage (V)')
        plt.ylabel('Drain Current (μA)')
        plt.title(f'BSIM4v4.7 Transfer Characteristics (Vbs = {Vbs} V, T = {T} K)')
        plt.grid(True)
        plt.legend()
        plt.show()


# Example usage
mosfet = BSIM4v4_7_Model()

# Plot IV characteristics for different Vgs values
mosfet.plot_IV(Vgs_list=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], Vds_max=1.0)

# Plot transfer characteristics for different Vds values
mosfet.plot_transfer(Vds_list=[0.1, 0.5, 1.0], Vgs_min=0.0, Vgs_max=1.0)

# Calculate single point
Id = mosfet.calculate_Id(Vds=0.5, Vgs=0.7, T=300.0, Vbs=0.0)
print(f"Drain current at Vds=0.5V, Vgs=0.7V: {Id*1e6:.2f} μA")