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
#? Name:        LV_39_BSIM2.py
#? Purpose:     Compute drain current using the BSIM2 (Level 39) model
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual - Level 39 (BSIM2) Model
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
from Equations  import Equations
import numpy as np
#? -------------------------------------------------------------------------------
#? NOTE: ° This is a simplified version of the BSIM3v3 model.
#?       ° It does not include all the parameters and equations of the original model.
#?       ° The device characteristics are modeled by process-oriented model parameters,
#?         which are mapped into model parameters at a specific bias voltage.
#? -------------------------------------------------------------------------------
class BSIM2Model:
    def __init__(self):
        self.eq         = Equations()
        
        #? =============================================================================================
        #? Fundamental physical constants
        #? =============================================================================================
        self.q          = 1.6e-19       # Elementary charge [C]
        self.k          = 1.38e-23      # Boltzmann constant [J/K]
        self.eps0       = 8.854e-12     # Vacuum permittivity [F/m]
        self.eps_si     = 11.7          # Silicon relative permittivity
        self.eps_ox     = 3.9           # Oxide relative permittivity
        self.Vtm        = 0.0259        # Thermal voltage at 300K [V]
        
        #? =============================================================================================
        #? Device dimensions (Geometry)
        #? =============================================================================================
        # Drawn dimensions
        self.L          = 1e-6          # Channel length [m]
        self.W          = 1e-6          # Channel width [m]
        
        # Process variations
        self.DL         = 0.0           # Channel length reduction [m]
        self.DW         = 0.0           # Channel width reduction [m]
        self.XL         = -50e-9        # Length correction [m]
        self.XW         = 300e-9        # Width correction [m]
        self.LD         = 50e-9         # Lateral diffusion length [m]
        self.WD         = 500e-9        # Width diffusion [m]
        
        # Effective dimensions
        self.Leff       = self.L - 2*self.LD - self.DL + self.XL  # Effective channel length [m]
        self.Weff       = self.W - 2*self.WD - self.DW + self.XW  # Effective channel width [m]
        
        #? =============================================================================================
        #? Oxide parameters
        #? =============================================================================================
        self.TOX        = 150e-10       # Gate oxide thickness [m]
        self.TOXM       = 20e-9         # Alternative oxide thickness [m]
        self.COX        = 3.453e-4      # Oxide capacitance per unit area [F/m²]
        self.tox        = self.TOX      # Alias for oxide thickness
        
        #? =============================================================================================
        #? Junction parameters
        #? =============================================================================================
        self.XJ         = 0.15e-6       # Junction depth [m]
        
        #? =============================================================================================
        #? Doping concentrations
        #? =============================================================================================
        self.NSUB       = 1e15          # Substrate doping [cm⁻³]
        self.NPEAK      = 1.7e17        # Peak channel doping [cm⁻³]
        self.NCH        = self.NPEAK    # Channel doping alias
        self.NGATE      = 1e18          # Polysilicon gate doping [cm⁻³]
        self.NSS        = 1.0e4         # Surface state density [cm⁻²]
        
        #? =============================================================================================
        #? Threshold voltage parameters
        #? =============================================================================================
        self.VTH0       = 0.7           # Zero-bias threshold voltage [V]
        self.VFB        = -0.576        # Flat-band voltage [V]
        self.VFB0       = -0.8          # Alternative flat-band voltage [V]
        self.GAMMA      = 0.5276        # Body effect coefficient [V^½]
        self.K1         = 0.5           # First-order body effect [V^½]
        self.K2         = 0.05          # Second-order body effect
        self.DELVTO     = 0             # Threshold voltage shift [V]
        self.VGHIGH     = 0.127         # High gate voltage [V]
        self.VGLOW      = -0.0782       # Low gate voltage [V]
        self.ETA =0
        #? =============================================================================================
        #? Mobility parameters
        #? =============================================================================================
        self.U0         = 670           # Low-field mobility [cm²/V·s]
        self.MUZ        = 500           # Zero-field mobility [cm²/V·s]
        self.MUS        = 300           # High-field mobility [cm²/V·s]
        self.MU_eff     = 0.01          # Effective mobility
        
        # Mobility reduction coefficients
        self.UA         = 2.25e-9       # First-order mobility degradation [m/V]
        self.UB         = 5.87e-19      # Second-order mobility degradation [m²/V²]
        self.UC         = -4.65e-11     # Body-effect mobility degradation [1/V]
        self.U1         = 0.01          # Drain-field mobility reduction [μm/V]
        self.U1S        = 5.87e-19      # Second-order mobility degradation [m²/V²]
        self.U1B        = 0.0           # Body-effect on U1 [μm/V²]
        self.U1D        = 0.0           # Drain-effect on U1 [μm/V²]
        
        #? =============================================================================================
        #? Velocity saturation & CLM
        #? =============================================================================================
        self.VSAT       = 8e4           # Saturation velocity [m/s]
        self.PCLM       = 1.3           # Channel length modulation coefficient
        
        #? =============================================================================================
        #? Short-channel effects
        #? =============================================================================================
        self.DVT0       = 2.2           # Short-channel effect coefficient
        self.DVT1       = 0.53          # Short-channel effect coefficient
        self.DVT2       = -0.032        # Short-channel effect coefficient [1/V]
        self.ETA0       = 0.1           # DIBL coefficient
        self.ETAB       = -0.07         # Body effect on DIBL
        
        #? =============================================================================================
        #? Temperature parameters
        #? =============================================================================================
        self.TCV        = 0.0           # Flat-band voltage temperature coefficient
        self.KT1        = 0.0           # Temperature coefficient of VTH0
        self.KT2        = 0.022         # Second order temperature coefficient
        
        #? =============================================================================================
        #? Process/Geometry corrections
        #? =============================================================================================
        self.LMLT       = 0.85          # Channel length multiplier
        self.WMLT       = 0.85          # Channel width multiplier
        
        #? =============================================================================================
        #? Impact ionization
        #? =============================================================================================
        self.AI0        = 1.84          # Impact ionization coefficient
        self.BI0        = 20            # Impact ionization exponent [1/V]
        
        #? =============================================================================================
        #? BSIM2 (LEVEL 39) specific parameters
        #? =============================================================================================
        # Threshold voltage parameters
        self.VOF0       = 0.477         # Subthreshold offset voltage [V]
        self.VOFB       = -0.034        # Body-bias subthreshold offset [V/V]
        self.VOFD       = -0.069        # Drain-bias subthreshold offset [V/V]
        
        # Subthreshold swing parameters
        self.n0         = 0.837         # Subthreshold swing coefficient
        self.nb         = 0.666         # Body-effect on subthreshold swing
        self.nd         = 0.0           # Drain-effect on subthreshold swing
        
        # Impact ionization parameters
        self.ai0        = 1.84          # Impact ionization coefficient A
        self.bi0        = 20            # Impact ionization coefficient B [1/V]
        self.aib        = 1           # Body-effect on impact ionization A
        self.bib        = 1           # Body-effect on impact ionization B [1/V]
        
        # Device type
        self.ch_type    = 1             # +1 for n-channel, -1 for p-channel
        self.TPG        = 1.0           # Type of gate material

    def vth(self, vsb, Vds, T):
        
        Vbi     =   self.VFB + self.eq.phi(T)
        Vth     =   Vbi + self.K1*np.sqrt(self.eq.phi(T)-vsb) -self.K2*(self.eq.phi(T)-vsb) - self.ETA * Vds
        return Vth
 
#? -------------------------------------------------------------------------------
    def strong_inversion_current(self, Vgs, Vds, vsb, T):
        Vth     = self.vth(vsb, Vds, T)
        Vgst    = Vgs - Vth
        # Body effect coefficient
        g       = 1 - 1 / (1.744 + 0.8364 * (self.eq.phi(T) - vsb))
        a       = 1 + g * self.K1 / (2 * np.sqrt(self.eq.phi(T) - vsb))
        # Saturation voltage
        Vc = self.U1S * Vgst / (a * (1 + self.UA * Vgst + self.UB * Vgst**2))
        K = (1 + Vc + np.sqrt(1 + 2 * Vc)) / 2
        Vdsat = Vgst / (a * K)
        
        # Beta factor
        beta0 = (self.Weff / self.Leff) * self.MU_eff * self.COX
        
        # Linear region current
        if Vds < Vdsat:
            U1 = self.U1S * (1 - (Vdsat - Vds) * self.U1D * (Vds - Vdsat)**2 / Vdsat**2)
            Ids = beta0 * (Vgst - a/2 * Vds) * Vds / (1 + self.UA * Vgst + self.UB * Vgst**2 + U1 * Vds)
        else:  # Saturation region
            Ids = beta0 * np.square(Vgst) / (2 * a * K * (1 + self.UA * Vgst + self.UB * Vgst**2))
            # Impact ionization
            AI = self.ai0  + self.aib  * vsb
            BI = self.bi0  + self.bib  * vsb
            if BI != 0 and AI != 0:
                f = AI * np.exp(-BI / (Vds - Vdsat))
                Ids *= (1 + f)
                
        return Ids
    
    def weak_inversion_current(self, Vgs, Vds, vsb, T):
        """Calculate weak inversion current"""
        Vth = self.vth(vsb, Vds, T)
        Vgst = Vgs - Vth

        N0   = self.n0 
        NB   = self.nb 
        ND   = self.nd
        VOF0 = self.VOF0 
        VOFB = self.VOFB 
        VOFD = self.VOFD 
        
        # Subthreshold swing
        N = N0 + NB / np.sqrt(self.eq.phi(T)  - vsb) + ND * Vds
        VOFF = VOF0 + VOFB * vsb + VOFD * Vds
        
        # Weak inversion current
        beta0 = (self.Weff / self.Leff) * self.MU_eff * self.COX
        term = np.exp((Vgst - VOFF) / (N * self.Vtm))
        Ids = beta0 * self.Vtm**2 * term * (1 - np.exp(-Vds / self.Vtm))
        
        # Impact ionization
        AI = self.ai0  + self.aib  * vsb
        BI = self.bi0  + self.bib  * vsb
        if BI != 0 and AI != 0:
            Vdsat = 0.1  # Approximate for weak inversion
            f = AI * np.exp(-BI / (Vds - Vdsat))
            Ids *= (1 + f)
            
        return Ids
    
    def transition_region_current(self, Vgs, Vds, vsb, T):
        """Calculate current in transition region"""
        # Use cubic interpolation between weak and strong inversion
        Vth = self.vth(vsb, Vds, T)
        VGHIGH = self.VGHIGH 
        VGLOW = self.VGLOW 
        
        # Calculate boundary currents
        I_weak = self.weak_inversion_current(Vth + VGLOW, Vds, vsb, T)
        I_strong = self.strong_inversion_current(Vth + VGHIGH, Vds, vsb, T)
        
        # Cubic coefficients
        C = np.zeros(4)
        C[0] = I_weak
        C[1] = 0  # Slope at weak inversion boundary
        C[2] = 3*(I_strong - I_weak) / (VGHIGH - VGLOW)**2
        C[3] = -2*(I_strong - I_weak) / (VGHIGH - VGLOW)**3
        
        # Evaluate cubic polynomial
        Vgst = Vgs - Vth
        x = Vgst - VGLOW
        Ids = C[0] + C[1]*x + C[2]*x**2 + C[3]*x**3
        
        return Ids
    
    def compute(self, Vgs, Vds, vsb=1, T=350):
        Vth = self.vth(vsb, Vds, T)
        Vgst = Vgs - Vth
        VGHIGH = self.VGHIGH 
        VGLOW = self.VGLOW 
        # Determine operation region
        if Vgst > VGHIGH:  # Strong inversion
            Ids = self.strong_inversion_current(Vgs, Vds, vsb, T)
        elif Vgst < VGLOW:  # Weak inversion
            Ids = self.weak_inversion_current(Vgs, Vds, vsb, T)
        else:  # Transition region
            Ids = self.transition_region_current(Vgs, Vds, vsb, T)
        return Ids
#? -------------------------------------------------------------------------------

# Example usage
if __name__ == "__main__":
    # Example model parameters (from the provided listing)
    # Create BSIM2 model instance
    model = BSIM2Model()
    
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