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
#? Name:        LV_1_Shichman_Hodges.py
#? Purpose:     Compute drain current using the Shichman-Hodges model
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual - Level 1 (Shichman-Hodges) Model
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import numpy as np
from vars import MOSFETParameters
mosfet = MOSFETParameters()
def v(param_name):
    return mosfet.var(param_name)
#? -------------------------------------------------------------------------------
class ShichmanHodgesModel:
    def vto(self, T):
        # Calculate energy gap using empirical temperature-dependent formula
        eg      = v("EG0") - v("EGSLOPE") * (np.square(T)) / (T + v("EGTEMP"))  
        # Calculate intrinsic carrier concentration (ni) using temperature-dependent formula
        ni      = v("NI0") * (T/v("TNOM"))**v("NITEXP") * np.exp(v("q")*eg/(2*v("k")) * (1/v("TNOM") - 1/T)) 
        # Calculate thermal voltage 
        vt      = v("k") * T / v("q")  
        # Calculate body effect coefficient (GAMMA)
        GAMMA   = v("GAMMA")
        # Calculate work function difference (phi_ms) based on gate type
        if v("TPG") == 0:  # Aluminum gate
            # For Al gate: phi_ms = -Eg/2 - (ch_type)*PHI/2 - PHIMS_OFFSET
            phi_ms = -eg/2 - v("ch_type") * v("PHI")/2 - v("PHIMS_OFFSET")  # PHIMS_OFFSET=0.05
        else:  # Polysilicon gate (TPG=±1)
            if v("NGATE") is None:
                # If no gate doping specified, use simplified formula
                phi_ms = v("ch_type") * (-v("TPG") * eg/2 - v("PHI")/2)
            else:
                # Ensure NGATE has a minimum value
                if v("NGATE") <= 0:
                    mosfet.NGATE = 1e18  # Update the parameter directly
                # For doped poly gate: depends on gate doping concentration
                phi_ms = v("ch_type") * (-v("TPG") * vt * np.log(v("NGATE")*1e6/ni) - v("PHI")/2)
        # Calculate flat-band voltage (vfb)
        # Includes work function difference, interface charge effect (NSS), and DELVTO adjustment
        vfb         = phi_ms - (v("q") * v("NSS") / v("COX")) + v("DELVTO")
        # Calculate zero-bias threshold voltage (VTO)
        VTO         = vfb + v("ch_type") * (GAMMA * np.sqrt(v("PHI")) + v("PHI"))
        
        return VTO

    def vth(self, T, vsb):
        # Get zero-bias threshold voltage
        vto = self.vto(T)            
        # Calculate built-in voltage (vbi)
        vbi = vto - v("GAMMA") * np.sqrt(v("PHI")) 
        # Calculate threshold voltage with body bias effect
        if vsb < 0:
            # For negative VSB (uncommon case), use linear approximation
            Vth = vbi + v("GAMMA") * (np.sqrt(v("PHI")) + 1/2 * (vsb/np.sqrt(v("PHI"))))
        elif vsb >= 0:
            # Standard case for VSB ≥ 0: full body effect formula
            Vth = vbi + v("GAMMA") * np.sqrt(v("PHI") + vsb) 
        
        return Vth
#? -------------------------------------------------------------------------------
    def compute(self, Vgs, Vds, vsb=0.0, T=350):
        Vth         = self.vth(T=T, vsb=vsb)  #! Threshold voltage
        Vsat        = Vgs - Vth
        W_over_L    = v("Weff") / v("Leff")

        if   Vsat <= 0:      region = "cutoff"       #! Vgs <= Vth
        elif Vsat >= Vds:    region = "linear"       #! vds <= vgs-Vth
        elif Vsat <  Vds:    region = "saturation"   #! Vds >= Vgs - Vth

        match region:
            case "cutoff":   
                Id = 0.0
            case "linear":   
                Id = v("KP") * W_over_L * (1 + v("LAMBDA") * Vds) * (Vsat - (Vds/2)) * Vds
            case "saturation":   
                Id = 1/2 * v("KP") * W_over_L * (1 + v("LAMBDA") * Vds) * np.square(Vsat)
        return Id
#? -------------------------------------------------------------------------------