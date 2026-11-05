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
#? Name:        LV_13_BSIM.py
#? Purpose:     Compute drain current using the Level 13 BSIM model
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual - Level 13 (BSIM) Model
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
from Equations import Equations
import numpy as np
#? -------------------------------------------------------------------------------
class BSIMLevel13Model:
    def __init__(self):
        self.eq = Equations()
        
        # Fundamental physical constants
        self.q      = 1.6e-19           # Elementary charge [C]
        self.k      = 1.38e-23          # Boltzmann constant [J/K]
        
        # Device geometry parameters
        self.Leff   = 1e-6              # Effective channel length [m]
        self.Weff   = 1e-6              # Effective channel width [m]
        self.COX    = 3.453e-4          # Oxide capacitance per unit area [F/m²]
        
        # Threshold voltage parameters
        self.VFB    = 3.48523E-02       # Flat-band voltage [V]
        self.PHI    = 7.87811E-01       # Surface potential [V]
        self.K1     = 9.01356E-01       # First-order body effect coefficient [V^½]
        self.K2     = 4.83095E-02       # Second-order body effect coefficient
        
        # Mobility parameters
        self.zu0    = 0.02              # Zero-field mobility coefficient
        self.zx2u0  = 2.05529E-03       # Second-order mobility degradation coefficient
        self.uo     = 5.81155E-02       # Low-field mobility [cm²/V·s]
        self.zu1    = 0.00000E+02       # First-order mobility parameter
        self.zx2u1  = 0.00000E+02       # Second-order mobility parameter
        self.zx3u1  = -1.64733E-02      # Third-order mobility parameter
        
        # Subthreshold conduction parameters
        self.zn0    = 2.00000E+02       # Subthreshold slope coefficient
        self.znb    = 0.00000E+02       # Body effect on subthreshold slope
        self.znd    = 0.00000E+02       # Drain effect on subthreshold slope
        
        # Velocity saturation parameters
        self.zeta   = 2.11768E-03       # Drain-induced barrier lowering coefficient
        self.zx2e   = -7.95688E-04      # Second-order DIBL coefficient
        self.zx3e   = 2.14262E-03       # Third-order DIBL coefficient
        
        # Temperature parameters
        self.TNOM   = 300               # Nominal temperature [K]
        
        # Supply voltage parameters
        self.VDDM   = 5.00000E+00       # Critical voltage for high drain field mobility [V]

    def vth(self, vsb, vds , T):
        """
        Calculate the threshold voltage with body and drain effects
        vsb: Source-bulk voltage [V]
        vds: Drain-source voltage [V]
        """
        gamma   = self.K1 - self.K2 * np.sqrt(self.eq.phi(T) + vsb)
        xeta    = self.zeta - self.zx2e * vsb + self.zx3e * (vds - self.VDDM)
        vth     = self.VFB + gamma * np.sqrt(self.eq.phi(T) + vsb) - xeta * vds
        return vth

    def subthreshold_current(self, vgs, vth, vds, T):
        """
        Calculate subthreshold current when Vgs < Vth
        vgs: Gate-source voltage [V]
        vth: Threshold voltage [V]
        vds: Drain-source voltage [V]
        T: Temperature [K]
        """
        if self.zn0 >= 200:
            return 0.0
        vt      = self.k * T / self.q  
        beta0   = self.uo * self.COX * (self.Weff / self.Leff)
        xn      = self.zn0 - self.znb * vsb + self.znd * vds
        lexp    = beta0 * np.square(vt) * np.exp(1.8) * np.exp((vgs - vth) / (xn * vt)) * (1 - np.exp(-vds / vt))
        Ilim    = 4.5 * beta0 * np.square(vt)
        isub    = (Ilim * lexp) / (Ilim + lexp)
        return isub

    def compute(self, Vgs, Vds, vsb=0.0, T=300.0):

            Id = 0.0  # Initialize drain current
            # Calculate threshold voltage
            vth         = self.vth(vsb, Vds,T)
            
            # Body effect coefficient
            g           = 1 - 1 / (1.744 + 0.8364 * (self.eq.phi(T) + vsb))
            body        = 1 + (g * self.K1) / (2 * np.sqrt(self.eq.phi(T) + vsb))
            
            # Mobility modulation coefficient
            xu1         = self.zu1 - self.zx2u1 * vsb + self.zx3u1 * (Vds - self.VDDM)
            
            # Calculate saturation voltage
            vc          = (xu1 * (Vgs - vth)) / body
            sqrt_arg    = max(0, 1 + 2 * vc)  # Ensure non-negative
            arg         = 0.5 * (1 + vc + np.sqrt(sqrt_arg))
            vdsat       = (Vgs - vth) / (body * np.sqrt(arg)) if arg > 0 else 0
            
            # Determine operation region
            region      = "cutoff" if Vgs <= vth else "ON"
            
            # Calculate effective mobility
            xu0         = self.zu0 - self.zx2u0 * vsb
            ueff        = self.uo / (1 + xu0 * (Vgs - vth))
            beta        = ueff * self.COX * self.Weff / self.Leff
            
            # Calculate drain current based on region
            if region == "ON":
                if Vds < vdsat:  # Linear/triode region
                    Id = (beta / (1 + xu1 * Vds)) * ((Vgs - vth) * Vds - (body / 2) * np.square(Vds))
                else:  # Saturation region
                    Id = (beta / (2 * body * arg)) * np.square(Vgs - vth)
                
                # Add subthreshold current if applicable
                if self.zn0 < 200:
                    Id += self.subthreshold_current(Vgs, vth, Vds, T)        
            return np.abs(Id)


if __name__ == "__main__":
    model = BSIMLevel13Model()
    
    # Define sweep ranges
    Vgs_values = [15, 18, 20]
    Vds_values = [100,600, 900]
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