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
#? Name:        Equations.py
#? Purpose:     Define shared sub-equations used across models.
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#? Based on:    HSPICE Manual 
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import numpy as np
#? -------------------------------------------------------------------------------
#? This code defines a class `Equations` that contains methods to compute various
#? semiconductor parameters such as intrinsic carrier concentration, surface 
#? potential, threshold voltage, and flatband voltage.The class uses physical
#? constants and parameters loaded from a logger, and it provides methods to 
#? compute these parameters based on temperature and other conditions. The code 
#? also includes a main block to test the functionality of the class by printing
#? some computed values for a given temperature.
#? -------------------------------------------------------------------------------

class Equations:
    def __init__(self):
        # ======================================================
        # FUNDAMENTAL PHYSICAL CONSTANTS
        # ======================================================
        self.q                 = 1.6e-19      # Elementary charge [C]
        self.k                 = 1.38e-23     # Boltzmann constant [J/K]
        self.ε0                = 8.854e-12    # Vacuum permittivity [F/m]

        # ======================================================
        # MATERIAL PROPERTIES
        # ======================================================
        # Permittivities
        self.eps_ox            = 3.9 * self.ε0   # SiO2 permittivity [F/m]
        self.eps_si            = 11.7 * self.ε0  # Silicon permittivity [F/m]
        self.eps_sic           = 9.7e-11         # SiC permittivity [F/m]

        # Carrier properties
        self.ni                = 1.5e8          # Intrinsic carrier concentration [1/cm³]
        self.NSS               = 1.0            # Surface state density [1/cm²]

        # ======================================================
        # DEVICE DIMENSIONS (GEOMETRY)
        # ======================================================
        # Process variations
        self.dpw               = 1e-6           # P-well separation [m]
        self.XJPW              = 0.1e-6         # P-well junction depth [m]
        self.H_by_eff          = 1e-7           # Effective channel height [m]

        # Oxide parameters
        self.TOX               = 150e-10        # Gate oxide thickness [m]
        self.tox               = self.TOX        # Alias for oxide thickness

        # ======================================================
        # DOPING CONCENTRATIONS
        # ======================================================
        self.NSUB              = 1e15           # Substrate doping [cm⁻³]
        self.Nsurf             = 1e17           # Surface doping [cm⁻³]
        self.PPW               = 1e17           # P-well doping [cm⁻³]
        self.NJFET             = 1e22           # JFET region doping [m⁻³]

        # ======================================================
        # THRESHOLD VOLTAGE PARAMETERS
        # ======================================================
        self.VFB               = -0.576         # Flat-band voltage [V]
        self.Vsurf             = 2.0            # Surface potential transition [V]

        # ======================================================
        # MOBILITY PARAMETERS
        # ======================================================
        self.mu                = 0.01           # General mobility [m²/V·s]

        # ======================================================
        # ADDITIONAL PARAMETERS FOR COMPLETENESS
        # ======================================================
        self.mjsurf            = 0.5            # Surface junction grading coefficient
        self.mj                = 0.5            # Bulk junction grading coefficient

    def phi_t(self, T):
        """
        Calculate the thermal voltage at temperature T.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Thermal voltage (k*T/q)
        """
        return (self.k * T) / self.q

    def phi_(self, T):
        """
        Calculate the potential difference based on thermal voltage and doping concentrations.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Potential difference value
        """
        return self.phi_t(T) * np.log(self.NJFET * self.PPW / (np.square(self.ni)))

    def alpha(self):
        """
        Calculate the alpha parameter for JFET modeling.
        
        Returns:
            float: Alpha parameter value
        """
        return np.sqrt((2 * self.eps_sic * self.PPW) / (self.q * self.NJFET * (self.NJFET + self.PPW)))

    def VTO_func(self, T):
        """
        Calculate the threshold voltage function for JFET.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Threshold voltage function value
        """
        return float(self.phi(T) - np.square(self.dpw / (2 * self.alpha())))

    def rho(self):
        """
        Calculate the resistivity of the JFET channel.
        
        Returns:
            float: Resistivity value
        """
        return 1 / (self.q * self.NJFET * self.mu)

    def beta_func(self, T):
        """
        Calculate the beta function for JFET modeling.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Beta function value
        """
        return ((2 * self.H_by_eff) / (self.XJPW * self.rho() * (-self.VTO_func(T)))) * ((self.dpw / 2) - self.alpha() * np.sqrt(self.phi(T)))

    def COX(self):
        """
        Calculate the oxide capacitance per unit area.
        
        Returns:
            float: Oxide capacitance value
        """
        return self.eps_ox / self.tox

    def Wdep1_num(self, VDS_val, VGS_val):
        """
        Calculate the first part of depletion width.
        
        Args:
            VDS_val (float): Drain-source voltage
            VGS_val (float): Gate-source voltage
            
        Returns:
            float: First part of depletion width
        """
        return np.sqrt((2 * self.eps_sic) / (self.q * self.Nsurf) *
                       min(VDS_val - VGS_val - self.VFB, self.Vsurf) ** self.mjsurf)

    def Wdep2_num(self, VDS_val, VGS_val, T_val):
        """
        Calculate the second part of depletion width.
        
        Args:
            VDS_val (float): Drain-source voltage
            VGS_val (float): Gate-source voltage
            T_val (float): Temperature in Kelvin
            
        Returns:
            float: Second part of depletion width
        """
        return np.sqrt((2 * self.eps_sic) / (self.q * self.Nsurf) *
                       min(VDS_val - VGS_val - self.VFB - self.Vsurf, self.phi(T_val) - self.VTO_func(T_val)) ** self.mj)

    def Cdep_num(self, VDS_val, VGS_val, T_val):
        """
        Calculate the depletion capacitance.
        
        Args:
            VDS_val (float): Drain-source voltage
            VGS_val (float): Gate-source voltage
            T_val (float): Temperature in Kelvin
            
        Returns:
            float: Depletion capacitance value
        """
        return self.eps_sic / (self.Wdep1_num(VDS_val, VGS_val) + self.Wdep2_num(VDS_val, VGS_val, T_val))

    def phi(self, T):
        """
        Calculate the potential difference between intrinsic Fermi level and Fermi level of doped substrate.
        
        Args:
            T (float): Temperature in Kelvin
            
        Returns:
            float: Potential difference in volts
        """
        phi = 2 * self.phi_t(T) * np.log(self.NSUB / self.ni)       # Surface potential (PHI)
        return phi