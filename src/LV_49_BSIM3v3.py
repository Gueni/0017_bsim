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
#? Name:        LV_49_BSIM3v3.py
#? Purpose:     Compute drain current using the BSIM3v3 model (Level 49)
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual - Level 49 (BSIM3) Model(3.0, 3.1, or 3.11)
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import numpy as np
from vars import MOSFETParameters
#? -------------------------------------------------------------------------------
#? NOTE: ° This is a simplified version of the BSIM3v3 model.
#?       ° It does not include all the parameters and equations of the original model.
#?       ° The device characteristics are modeled by process-oriented model parameters,
#?         which are mapped into model parameters at a specific bias voltage.
#? -------------------------------------------------------------------------------
class BSIM3v3_Model:
    def __init__(self):
        self.mosfet = MOSFETParameters()
        self.DLC    = self.v("LINT")
        self.DWC    = self.v("WINT")
        self.GAMMA1 = np.sqrt(2 * self.v("q") * self.v("eps_si") * self.v("NCH") * 1e6) / self.v("COX")
        self.GAMMA2 = np.sqrt(2 * self.v("q") * self.v("eps_si") * self.v("NSUB") * 1e6) / self.v("COX")
        phi_s       = self._calculate_phi_s()
        self.VBX    = phi_s - (self.v("q") * self.v("NCH") * 1e6 * self.v("XT")**2) / (2 * self.v("eps_si"))
        ni          = self._calculate_ni(self.v("TNOM"))
        self.VBI    = (self.v("k") * self.v("TNOM") / self.v("q")) * np.log((self.v("NCH") * 1e6 * 1e20) / (ni**2))
    
    def v(self, param_name):
        """Helper method to access MOSFET parameters"""
        return self.mosfet.var(param_name)
    
    def _calculate_phi_s(self):
        """Calculate surface potential."""
        ni          = self._calculate_ni(self.v("TNOM"))
        return 2 * (self.v("k") * self.v("TNOM") / self.v("q")) * np.log(self.v("NCH") * 1e6 / ni)
    
    def _calculate_ni(self, T):
        """Calculate intrinsic carrier concentration at temperature T."""
        # Bandgap energy
        Eg          = self.v("EG0") - self.v("EGSLOPE") * (np.square(T)) / (T + self.v("EGTEMP"))  
        # Intrinsic carrier concentration
        term1       = (T / 300.15)**1.5
        term2       = np.exp(21.5565981 - (self.v("q") * Eg) / (2 * self.v("k") * T))
        return 1.45e10 * term1 * term2
    # # =============================================================================
    # # Intrinsic Carrier Concentration (ni) Calculation in BSIM3v3 Model
    # # =============================================================================
    # # Equation:
    # #   ni = 1.45e10 * (T/300.15)^1.5 * exp(21.5565981 - q*Eg(T)/(2*kB*T))
    # #
    # # Where:
    # #   - T: Temperature in Kelvin
    # #   - Eg(T): Temperature-dependent bandgap energy
    # #   - q: Elementary charge (1.602e-19 C)
    # #   - kB: Boltzmann constant (8.617e-5 eV/K or 1.380e-23 J/K)
    # # =============================================================================

    # def _calculate_ni(self,T):
    #     """
    #     Calculate intrinsic carrier concentration (ni) for silicon at temperature T.
        
    #     Parameters:
    #         T (float): Temperature in Kelvin
        
    #     Returns:
    #         float: Intrinsic carrier concentration in cm^-3
    #     """
        
    #     # -------------------------------------------------------------------------
    #     # Constants
    #     # -------------------------------------------------------------------------
    #     NI_300K = 1.45e10       # Intrinsic carrier conc. at 300.15K (cm^-3)
    #     T_NOM = 300.15          # Reference temperature (K)
    #     EXPONENT = 1.5           # Temperature exponent for density of states
    #     PRE_EXP = 21.5565981     # Pre-exponential constant
        
    #     # Fundamental physical constants
    #     q = 1.602e-19           # Elementary charge (C)
    #     kB = 8.617e-5           # Boltzmann constant (eV/K)
        
    #     # -------------------------------------------------------------------------
    #     # Temperature-dependent bandgap calculation (Eg(T))
    #     # -------------------------------------------------------------------------
    #     # Simplified Eg(T) model for silicon:
    #     # Eg(T) = EG0 - (EGSLOPE*T^2)/(T + EGTEMP)
    #     EG0 = 1.17              # Bandgap at 0K (eV)
    #     EGSLOPE = 4.73e-4       # Empirical parameter (eV/K)
    #     EGTEMP = 636.0          # Empirical parameter (K)
        
    #     Eg = EG0 - (EGSLOPE * T**2) / (T + EGTEMP)
        
    #     # -------------------------------------------------------------------------
    #     # Calculate ni
    #     # -------------------------------------------------------------------------
    #     # Temperature scaling term
    #     temp_ratio = (T / T_NOM) ** EXPONENT
        
    #     # Exponential term
    #     exp_arg = PRE_EXP - (q * Eg) / (2 * kB * T)
        
    #     # Final calculation
    #     ni = NI_300K * temp_ratio * np.exp(exp_arg)
        
    #     return ni

    # # =============================================================================
    # # Key Notes:
    # # 1. The 1.45e10 cm^-3 is the well-known ni value for silicon at 300K
    # # 2. The (T/300.15)^1.5 term accounts for temperature-dependent density of states
    # # 3. The exponential term handles the Boltzmann statistics of carrier excitation
    # # 4. The bandgap narrowing effect at higher T is included via Eg(T)
    # # 5. This matches the ni calculation used in BSIM3/4 models for MOSFET simulation
    # # =============================================================================
    
    def _calculate_vth(self, Vbs, Vds,T):
        """Calculate threshold voltage accounting for body effect, DIBL, etc."""
        # Temperature adjustment
        delta_T     = T / self.v("TNOM") - 1
        print(delta_T)
        vth0        = self.v("VTH0") - self.v("KT1") * delta_T - self.v("KT1L") / self.v("Leff") * delta_T
        # Calculate from GAMMA1 and GAMMA2
        term1       = (self.GAMMA2 - self.GAMMA1) * (np.sqrt(self.v("PHI") - Vbs) - np.sqrt(self.v("PHI")))
        term2       = 2 * np.sqrt(self.v("PHI")) * (np.sqrt(self.v("PHI") - self.v("VBM")) - np.sqrt(self.v("PHI"))) + self.v("VBM")
        K2          = term1 / term2
        gamma_eff   = self.GAMMA2 + 2 * K2 * np.sqrt(self.v("PHI") - Vbs)
        # Narrow width effect
        k3_eff      = self.v("K3") + self.v("K3B") * Vbs
        w0_eff      = self.v("W0") + self.v("Weff")
        # Short channel effects
        DVT0        = self.v("DVT0") * (1 + self.v("DVT1") * np.exp(-self.v("DVT2") * Vbs))
        DVT0W       = self.v("DVT0W") * (1 + self.v("DVT1W") * np.exp(-self.v("DVT2W") * Vbs))
        # Calculate threshold voltage
        vth         = (vth0 + gamma_eff * np.sqrt(self.v("PHI") - Vbs) + k3_eff * self.v("PHI") * self.v("TOX") / (w0_eff * self.v("Weff")) - (DVT0 + DVT0W) * Vds)
        return vth
    
    def _calculate_mobility(self, Vgs, Vbs, Vds,T):
        """Calculate effective mobility accounting for degradation effects."""
        # Temperature adjustment
        delta_T     = T / self.v("TNOM") - 1
        # What is delta_T?
        # It measures how much hotter (delta_T > 0) or colder (delta_T < 0) the transistor is compared to its reference temperature (TNOM).
        # What happens when delta_T > 0 (transistor is hotter)?
        # Mobility drops (electrons scatter more due to heat).
        # Threshold voltage decreases (easier to turn on the transistor).
        # What happens when delta_T < 0 (transistor is colder)?
        # Mobility improves (less scattering).
        # Threshold voltage increases (harder to turn on).
        # Why use delta_T instead of just temperature?
        # It simplifies the math—most temperature effects are linear near TNOM, so delta_T makes calculations cleaner and faster.
        u0_temp     = self.v("U0") * (T / self.v("TNOM"))**self.v("UTE")
        # Calculate Vgsteff
        vth         = self._calculate_vth(Vbs, Vds,T)
        Vgsteff     = max(Vgs - vth, 0)
        # Mobility degradation terms
        UA_eff      = self.v("UA") + self.v("UA1") * delta_T
        UB_eff      = self.v("UB") + self.v("UB1") * delta_T
        UC_eff      = self.v("UC") + self.v("UC1") * delta_T
        denom       = (1 + UA_eff * Vgsteff + UB_eff * Vgsteff**2 + UC_eff * Vbs)
        # Effective mobility (convert from cm^2/V·s to m^2/V·s)
        ueff        = u0_temp / denom * 1e-4
        return ueff
    
    def compute(self, Vgs, Vds, Vbs=0.0, T=300.0):
        """
        Compute the drain current (Id) using the BSIM3v3 model.
        """
        # Calculate threshold voltage
        vth         = self._calculate_vth(Vbs, Vds,T)
        Vgsteff     = max(Vgs - vth, 0)
        # Cutoff region
        if Vgsteff <= 0:
            return 0.0
        # Calculate effective mobility
        ueff        = self._calculate_mobility(Vgs, Vbs, Vds,T)
        # Calculate beta (conduction parameter)
        beta        = ueff * self.v("COX") * self.v("Weff") / self.v("Leff")
        # Calculate Abulk (bulk charge effect)
        Abulk       = (self.v("A0") + self.v("AGS") * Vgs) * (1 + self.v("KETA") * Vbs)
        # Calculate saturation voltage
        """Calculate saturation voltage."""
        # Saturation field
        Esat        = 2 * self.v("VSAT") / (self._calculate_mobility(0, 0, 0,300) * 1e4)  # Convert mobility back to cm^2/V·s for Esat
        # Saturation voltage
        if self.v("A1") != 0 or self.v("A2") != 1:
            vdsat   = Vgsteff / (Abulk + Esat * self.v("Leff"))
        else:
            vdsat   = Vgsteff / (Abulk * (1 + Esat * self.v("Leff") / Vgsteff))
        denom_r     = (1 + self.v("RDSW") * (1 + self.v("PRWG") * Vgsteff + self.v("PRWB") * (np.sqrt(self.v("PHI") - Vbs)- np.sqrt(self.v("PHI")))) * self.v("Weff"))
        Rds         = 1 / denom_r if denom_r != 0 else 0
        # Linear region
        if Vds < vdsat:
            Id_lin = beta * (Vgsteff - 0.5 * Abulk * Vds) * Vds / (1 + self.v("A2") * Vds / Vgsteff)
            Id     = Id_lin / (1 + Rds * beta * (Vgsteff - 0.5 * Abulk * Vds) * Vds)
        # Saturation region
        else:
            Id_sat = 0.5 * beta * vdsat**2 / (1 + self.v("A2") * vdsat / Vgsteff)
            # Channel length modulation
            l      = min(self.v("PCLM") * self.v("TOX") / (self.v("Leff") * (vdsat + Vds)), 0.5)
            # DIBL effect
            theta  = self.v("ETA0") + self.v("ETAB") * Vbs
            Id     = Id_sat * (1 + l * (Vds - vdsat)) * (1 + theta * (Vds - vdsat))
        return Id
#? -------------------------------------------------------------------------------