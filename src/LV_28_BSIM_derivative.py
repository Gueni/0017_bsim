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
#? Name:        LV_28_BSIM2_mod.py
#? Purpose:     Compute drain current using the BSIM model
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual - Level 28 (BSIM2) Model
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
from Equations import Equations
import numpy as np

class BSIM_Model:
    def __init__(self):
        self.eq = Equations()
        # FUNDAMENTAL PHYSICAL CONSTANTS
        self.q      = 1.6e-19          # Elementary charge [C]
        self.k      = 1.38e-23         # Boltzmann constant [J/K]
        self.eps0   = 8.854e-12        # Vacuum permittivity [F/m]
        self.COX    = 3.453e-4         # Oxide capacitance per unit area [F/m²]
        # THRESHOLD VOLTAGE PARAMETERS
        self.K1     = 0.5              # First-order body effect [V^½]
        self.K2     = 0.05             # Second-order body effect
        self.zvfb   = -0.368           # Flat-band voltage [V] (alternative)
        # MOBILITY PARAMETERS
        self.muz    = 600e-4           # Temperature-adjusted mobility
        self.zu0    = 0.02             # Zero-field mobility coefficient
        self.U1     = 0.01             # Drain-field mobility reduction [μm/V]
        self.ETA0   = 0.0              # Linear Vds threshold coefficient
        self.X3E    = 0.0              # Vds correction to linear vds threshold coefficient
        self.X2E    = 0.0              # Vsb correction to linear vds threshold coefficient
        # SECOND-ORDER MOBILITY CORRECTIONS
        self.X2M    = 0.0              # Vsb correction to mobility [cm²/V²·s]
        self.X2U0   = 0.0              # Vsb reduction to mobility degradation [1/V²]
        self.X2U1   = 0.0              # Vsb reduction to drain field mobility reduction
        self.X3MS   = 5.0              # Vds correction for high drain field mobility [cm²/V²·s]
        self.X33M   = 0.0              # Gate field reduction of X3MS [cm²/V²·s]
        self.X3U1   = 0.0              # Vds reduction to drain field mobility reduction factor
        # VOLTAGE PARAMETERS
        self.vddm   = 5.0              # Critical voltage for high drain field mobility [V]
        # WEAK INVERSION PARAMETERS
        self.NO     = 100              # Weak inversion coefficient
        self.N0     = 1.0              # Weak inversion parameter
        self.NB0    = 0.0              # Vsb correction to weak inversion
        self.ND0    = 0.0              # Vds correction to weak inversion
        self.WFAC   = 1.0              # Weak inversion factor
        self.A0     = 0.1              # Weak inversion transition parameter
        self.WFACU  = 1.0              # Weak inversion factor coefficient
        # GEOMETRY PARAMETERS
        self.weff   = 1e-6             # Effective channel width [m]
        self.leff   = 1e-6             # Effective channel length [m]

    def compute(self, Vgs, Vds, vsb=0.0, T=350):
        # Calculate thermal voltage
        vtherm = self.k * T / self.q
        
        # Threshold Voltage Calculation (Page 10-11)
        xbs = np.sqrt(self.eq.phi(T) - vsb)
        xeta = self.ETA0 + self.X2E * vsb + self.X3E * Vds
        vth = self.zvfb + self.eq.phi(T) + self.K1 * xbs - self.K2 * xbs**2 - xeta * Vds

        # Effective Mobility Calculation (Page 11)
        vgst = Vgs - vth
        cx3ms = self.X3MS / (self.muz + self.X33M * vgst)
        meff = (self.muz + self.X2M * vsb) * (1 + cx3ms * (self.vddm + Vds - np.sqrt(self.vddm**2 + Vds**2)))
        zu0 = self.zu0 + self.X2U0 * vsb
        ueff = meff / (1 + zu0 * vgst)
        beta = ueff * self.COX * self.weff / self.leff
        
        # Saturation Voltage Calculation (Page 12)
        g = 1 - 1 / (1.744 + 0.8364 * xbs**2)
        body = (1 + g * self.K1) / (2 * xbs)
        xu1 = self.U1 + vsb * self.X2U1  # Changed from self.zu1 to self.U1
        rx = np.sqrt(body**2 + 2 * body * xu1 * vgst + self.X3U1 * 4 * vgst**2)
        vdsat = (2 * vgst) / (body + rx)
        
        # Determine operating region and calculate current (Page 12-13)
        if vgst <= 0:  # Cutoff region
            Id = 0.0
        elif Vds < vdsat:  # Linear region
            Id = beta * (vgst - body/2 * Vds) * Vds / (1 + (xu1 + self.X3U1 * Vds) * Vds)
        else:  # Saturation region
            Id = beta * (vgst - body/2 * vdsat) * vdsat / (1 + (xu1 + self.X3U1 * vdsat) * vdsat)
        
        # Weak inversion current (Page 13-15)
        if self.NO < 200:  # Only calculate if weak inversion is enabled
            xn = self.N0 + self.NB0 * vsb + self.ND0 * Vds
            xweak = (Vgs - vth) / (xn * vtherm)
            
            if xweak < -self.WFAC + self.A0:
                Iweak = np.exp(xweak)
            elif xweak < 0:
                Iweak = np.exp(xweak - self._calculate_wf(xweak))
            elif xweak < self.A0:
                Iweak = np.exp(xweak - self._calculate_wf(xweak)) - xweak**2
            else:
                Iweak = 0
            
            Id += Iweak * (1 - np.exp(-Vds/vtherm))
        
        return Id
    
    def _calculate_wf(self, xweak):
        """Helper function to calculate weak inversion factor (Page 14)"""
        term = (xweak + self.WFAC - self.A0)**2
        denom = (1 + xweak + self.WFAC - self.A0) * (1 + self.WFACU * (xweak + self.WFAC - self.A0))
        return term / denom
#? -------------------------------------------------------------------------------

if __name__ == "__main__":
    model = BSIM_Model()
    
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
