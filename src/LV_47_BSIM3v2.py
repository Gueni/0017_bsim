#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?Add commentMore actions
#?                 ______  ____  _______  _____
#?                / __ \ \/ /  |/  / __ \/ ___/
#?               / /_/ /\  / /|_/ / / / /\__ \
#?              / ____/ / / /  / / /_/ /___/ /
#?             /_/     /_/_/  /_/\____//____/
#?
#? Name:        LV_47_BSIM3v2.py
#? Purpose:     Compute drain current using the BSIM model
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#?
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import numpy as np
from Equations import Equations
#? -------------------------------------------------------------------------------
class BSIM3v2_Model:
    def __init__(self):
        self.eq = Equations()
        
        # Fundamental constants
        self.q = 1.6e-19       # Elementary charge [C]
        self.k = 1.38e-23      # Boltzmann constant [J/K]
        self.eps0 = 8.854e-12  # Vacuum permittivity [F/m]
        
        # Material properties
        self.eps_ox = 3.9 * self.eps0
        self.eps_si = 11.7 * self.eps0
        
        # Device dimensions
        self.L = 1e-6          # Channel length [m]
        self.W = 1e-6          # Channel width [m]
        self.DL = 0.2e-6       # Channel length reduction [m]
        self.DW = -0.1e-6      # Channel width reduction [m]
        self.TOX = 7.0e-9      # Gate oxide thickness [m]
        self.COX = self.eps_ox / self.TOX  # Oxide capacitance [F/m²]
        self.XJ = 1.5e-7       # Junction depth [m]
        
        # Doping concentrations
        self.NPEAK = 1.5e23    # Peak channel doping [cm⁻³]
        
        # Threshold voltage parameters
        self.VTH0 = -0.8       # Zero-bias threshold voltage [V]
        self.PHI = 0.7         # Surface potential [V]
        self.K1 = 0.5          # First-order body effect [V^½]
        self.K2 = 0.03         # Second-order body effect
        self.K3 = 0            # Narrow width effect coefficient
        self.DELVTO = 0        # Threshold voltage shift [V]
        
        # Mobility parameters
        self.U0 = 7e-03 * 1e4  # Low-field mobility [cm²/V·s]
        self.UA = 1e-09        # First-order mobility degradation [m/V]
        self.UB = 0            # Second-order mobility degradation [m²/V²]
        self.UC = -3e-02       # Body-effect mobility degradation [1/V]
        
        # Velocity saturation
        self.VSAT = 9e6        # Saturation velocity [m/s]
        self.PCLM = 77         # Channel length modulation coefficient
        
        # Short-channel effects
        self.DVT0 = 48         # Short-channel effect coefficient
        self.DVT1 = 0.6        # Short-channel effect coefficient
        self.DVT2 = -5e-4      # Short-channel effect coefficient [1/V]
        
        # Temperature parameters
        self.T = 300           # Operating temperature [K]
        self.TNOM = 300.15     # Nominal temperature [K] (27°C)
        self.KT1 = -0.3        # Temperature coefficient of VTH0
        self.KT2 = -0.03       # Second order temperature coefficient
        
        # Process/geometry corrections
        self.W0 = 2.5e-6       # Narrow width effect coefficient [m]
        self.NLX = 0           # Lateral non-uniform doping parameter
        self.A0 = 0.87         # Bulk charge effect coefficient
        self.VOFF = -0.07      # Offset voltage [V]
        self.NFACTOR = 1.5     # Subthreshold swing coefficient
        self.ETA0 = 0          # DIBL coefficient
        
        # Model flags
        self.BULKMOD = 1       # Bulk charge model
        self.SATMOD = 2        # Velocity saturation model
        self.SUBTHMOD = 2      # Subthreshold model
        
        # Other parameters
        self.RDS0 = 180        # Source-drain resistance [Ω·μm]
        self.PDIBL1 = 0        # DIBL parameter
        self.PDIBL2 = 2e-11    # DIBL parameter
        self.DROUT = 0         # Drain-induced barrier lowering coefficient
        self.PSCBE1 = 0        # Substrate current body-effect
        self.PSCBE2 = 1e-28    # Substrate current body-effect
        self.LITL = 4.5e-08    # Length scaling parameter
        self.EM = 4e7          # Mobility exponent
        self.LDD = 0           # LDD region length
        self.ueff = self.U0 * 1e-4  # Convert mobility to m²/V·s
        
        # Additional parameters
        self.CIT = -3e-05      # Interface trap capacitance [F/m²]
        self.CDSC = 6e-02      # Drain/source to channel coupling capacitance [F/m²]
        self.VGLOW = -0.12     # Low voltage limit for mobility
        self.VGHIGH = 0.12     # High voltage limit for mobility
        self.AT = 33000        # Temperature exponent
        self.UA1 = 4e-09       # Temperature-dependent mobility parameter
        self.UB1 = 7e-18       # Temperature-dependent mobility parameter
        self.UC1 = 0           # Temperature-dependent mobility parameter
        self.PVAG = 0          # Gate voltage parameter
        self.ETA = 0           # DIBL coefficient
        self.KETA = 0
    
    def vth(self, vsb, T=350):
        """Calculate threshold voltage with body and drain bias effects"""
        Vth0_temp = self.VTH0 + (self.KT1 + self.KT2 * vsb) * (T/self.TNOM - 1)
        sqrt_phi = np.sqrt(self.eq.phi(T))
        sqrt_phi_vsb = np.sqrt(self.eq.phi(T) + vsb)
        
        Vth = Vth0_temp
        Vth += self.K1 * (sqrt_phi_vsb - sqrt_phi)
        Vth -= self.K2 * vsb
        
        # Narrow width effect
        narrow_width_term = (self.K3 + self.K3 * vsb) * \
                          (self.TOX / ((self.W - 2 * self.DW) + self.W0)) * self.eq.phi(T)
        Vth += narrow_width_term
        
        # Short channel effect
        Xdep = np.sqrt(2 * self.eps_si * (self.eq.phi(T) + vsb) / (self.q * self.NPEAK * 1e6))
        lt = np.sqrt(3 * self.TOX * Xdep) * (1 + self.DVT2 * vsb)
        
        theta_th = self.DVT0 * (np.exp(-self.DVT1 * (self.L - 2 * self.DL) / (2 * lt)) + \
                  2 * np.exp(-self.DVT1 * (self.L - 2 * self.DL) / lt))
        
        Eg = 1.16 - (7.02e-4) * T**2 / (T + 1108.0)
        ni = 1.45e10 * (T / 300.15)**1.5 * np.exp(21.5565981 - Eg / (2 * self.eq.phi_t(T)))
        Vbi = self.k * T / self.q * np.log(1e22 * self.NPEAK / ni**2)
        delta_Vth = theta_th * (Vbi - self.eq.phi(T))
        
        return Vth - delta_Vth
    
    def compute(self, Vgs, Vds, vsb=0.0, T=350):
        """Calculate drain current for given biases"""
        try:
            Vth = self.vth(vsb, T)
            Vgst = Vgs - Vth
            
            if Vgst <= 0:
                return 0.0  # Cutoff region
                
            # Bulk charge effect
            phi = self.eq.phi(T)
            Xdep = np.sqrt(2 * self.eps_si * (phi + vsb) / (self.q * self.NPEAK * 1e6))
            T1 = 2 * np.sqrt(self.XJ * Xdep)
            T1s = np.sqrt(phi + vsb) if vsb >= 0 else phi * np.sqrt(phi) / (phi - vsb/2)
            
            if self.BULKMOD == 1:
                Abulk = (1 + (self.K1 * self.A0 * (self.L - 2 * self.DL)) / 
                        (((self.L - 2 * self.DL) + T1) * T1s * 2)) / (1 + self.KETA * vsb)
            else:
                Abulk = (self.K1 * self.A0 * (self.L - 2 * self.DL) / 
                        (((self.L - 2 * self.DL) + T1) * np.sqrt(phi * 2))) / (1 + self.KETA * vsb)
            
            # Saturation voltage calculation
            Esat = 2 * self.VSAT / self.ueff
            
            if self.RDS0 == 0:
                Vdsat = (Esat * (self.L - 2 * self.DL) * Vgst) / (Abulk * Esat * (self.L - 2 * self.DL) + Vgst)
            else:
                Rds = self.RDS0 * 1e-6 / (self.W - 2 * self.DW)  # Convert to Ω·m
                Pfactor = min(1, 1)  # Simplified
                
                Tmpa = Abulk * (Abulk * (self.W - 2 * self.DW) * self.VSAT * self.COX * Rds - 1 + 1/Pfactor)
                Tmph = Vgst * (2/Pfactor - 1) + Abulk * Esat * (self.L - 2 * self.DL) + \
                      3 * Abulk * Vgst * (self.W - 2 * self.DW) * self.VSAT * self.COX * Rds
                Tmpc = Vgst * Esat * (self.L - 2 * self.DL) + \
                      Vgst**2 * 2 * (self.W - 2 * self.DW) * self.VSAT * self.COX * Rds
                
                Vdsat = (Tmph - np.sqrt(Tmph**2 - 4 * Tmpa * Tmpc)) / (2 * Tmpa)
            
            # Drain current calculation
            beta = self.ueff * self.COX * (self.W - 2 * self.DW) / (self.L - 2 * self.DL)
            
            if Vds <= Vdsat:
                # Linear region
                return beta * (Vgst - Abulk * Vds/2) * Vds
            
            # Saturation region
            Ids0 = beta * (Vgst - Abulk * Vdsat/2) * Vdsat
            
            # Channel length modulation
            litl = np.sqrt(3 * self.TOX * Xdep)
            lambda_clm = (Abulk * Esat * (self.L - 2 * self.DL) + Vgst) / (2 * litl * self.EM)
            
            Vasat = (Esat * (self.L - 2 * self.DL) + Vdsat) / (2 - 1)
            Fvag = 1 + self.PVAG * Vgst / (Esat * (self.L - 2 * self.DL))
            Va = Vasat + Fvag * ((1 + self.ETA * self.LDD / litl) /
                (self.PCLM * Abulk)) * ((Abulk * Esat * (self.L - 2 * self.DL) + 
                Vgst - lambda_clm * (Vds - Vdsat)) * (Vds - Vdsat) / (Esat * litl))
            
            return Ids0 * (1 + (Vds - Vdsat) / Va)
            
        except Exception as e:
            print(f"Error in calculation: {str(e)}")
            return float('nan')

#? -------------------------------------------------------------------------------

if __name__ == "__main__":
    model = BSIM3v2_Model()
    
    # Define sweep ranges
    Vgs_values = [15, 18, 20]
    Vds_values = [100, 600, 900]
    vsb = 0.35
    T = 400
    
    # Print table header
    print("="*80)
    print(f"{'Vgs (V)':<10} | {'Vds (V)':<10} | {'Vsb (V)':<10} | {'Temperature (K)':<15} | {'Drain Current (A)':<20}")
    print("-"*80)
    
    # Sweep through values
    for Vgs in Vgs_values:
        for Vds in Vds_values:
            Id = model.compute(Vgs=Vgs, Vds=Vds, vsb=vsb, T=T)
            print(f"{Vgs:<10} | {Vds:<10} | {vsb:<10} | {T:<15} | {Id:.6e}")
    
    print("="*80)
    #? -------------------------------------------------------------------------------
