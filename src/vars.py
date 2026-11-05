import numpy as np

class MOSFETParameters:
    def __init__(self):
        # Fundamental Physical Constants
        self.q                = 1.60218e-19    # Elementary charge [C]
        self.k                = 1.38065e-23    # Boltzmann constant [J/K]
        self.eps0             = 8.85418782e-12 # Vacuum permittivity [F/m]
        self.EG0              = 1.16           # (eV) Silicon bandgap at 0K
        self.EGSLOPE          = 7.02e-4        # Temperature coefficient for bandgap
        self.EGTEMP           = 1108           # (K) Temperature parameter in bandgap equation
        self.NI0              = 1.45e10        # Intrinsic carrier concentration at 300K
        self.NITEXP           = 1.5            # Exponent for temperature dependence of ni
        self.PHIMS_OFFSET     = 0.05           # (V) Work function difference offset for Al gate

        # Material Properties
        self.eps_ox           = 3.9 * self.eps0  # SiO2 permittivity [F/m]
        self.eps_si           = 11.7 * self.eps0 # Silicon permittivity [F/m]
        self.ni               = 1.45e16          # Intrinsic carrier concentration [m^-3]

        # Basic Model Parameters
        self.TOX              = 1.00000E-08     # Gate oxide thickness [m]
        self.VTH0             = 0.6094574       # Threshold voltage at Vbs=0 [V]
        self.VFB              = -0.576          # Flat-band voltage [V]
        self.NSUB             = 6.0e16          # Substrate doping concentration [cm^-3]
        self.NCH              = 1.024685E+17    # Peak doping concentration near interface [cm^-3]
        self.NPEAK            = 1.024685E+17    # Channel doping alias
        self.K1               = 0.5341038       # First-order body effect coefficient [V^1/2]
        self.K2               = 1.703463E-03    # Second-order body effect coefficient
        self.K3               = -17.24589       # Narrow width effect coefficient
        self.K3B              = 4.139039        # Body width coefficient of narrow width effect [1/V]
        self.W0               = 1e-6            # Narrow width effect coefficient [m]

        # Short channel effect coefficients
        self.DVT0W            = 0.0             # Narrow width coefficient 0 for Vth at small L [1/m]
        self.DVT1W            = 5.3e6           # Narrow width coefficient 1 for Vth at small L [1/m]
        self.DVT2W            = -0.032          # Narrow width coefficient 2 for Vth at small L [1/V]
        self.DVT0             = 0.1767506       # Short channel effect coefficient 0 for Vth
        self.DVT1             = 0.5109418       # Short channel effect coefficient 1 for Vth
        self.DVT2             = -0.05           # Short channel effect coefficient 2 for Vth [1/V]

        # DIBL (Drain Induced Barrier Lowering) coefficients
        self.ETA0             = 0.0145072       # Subthreshold region DIBL coefficient
        self.ETAB             = -3.870303E-03   # Subthreshold region DIBL coefficient [1/V]
        self.VBM              = -3.0            # Maximum substrate bias for Vth calculation [V]

        # Mobility parameters
        self.U0               = 307.2991        # Low field mobility (nMOS) [cm^2/V·s]
        self.UA               = -1.748481E-09   # First-order mobility degradation [m/V]
        self.UB               = 3.178541E-18    # Second-order mobility degradation [m^2/V^2]
        self.UC               = 1.3623e-10      # Body bias sensitivity of mobility [1/V]
        self.U1               = 0.01            # Drain-field mobility reduction [μm/V]

        # Bulk charge effect coefficients
        self.A0               = 0.4976366       # Bulk charge effect coefficient for channel length
        self.AGS              = 1.2             # Gate bias coefficient of Abulk [1/V]
        self.B0               = 0.0             # Bulk charge effect coefficient for channel width [m]
        self.B1               = 0.0             # Bulk charge effect width offset [m]
        self.KETA             = -2.195445E-02   # Body-bias coefficient of bulk charge effect [1/V]

        # Subthreshold parameters
        self.VOFF             = -9.623903E-02   # Offset voltage in subthreshold region [V]
        self.NFACTOR          = 0.8408191       # Subthreshold region swing

        # Saturation velocity parameters
        self.VSAT             = 97662.05        # Saturation velocity [m/s]
        self.A1               = 0.0332883       # First nonsaturation factor [1/V]
        self.A2               = 0.9             # Second nonsaturation factor

        # Parasitic resistance parameters
        self.RDSW             = 298.873         # Parasitic source/drain resistance per unit width [ohm·μm]
        self.PRWG             = -0.001          # Gate bias effect coefficient of RDSW [1/V]
        self.PRWB             = -2.24e-4        # Body effect coefficient of RDSW [1/V^1/2]
        self.WR               = 1.0             # Width offset from Weff for Rds calculation

        # Interface state and coupling capacitance
        self.CIT              = 3.994609E-04    # Interface state capacitance [F/m^2]
        self.CDSC             = 1.130797E-04    # Drain/source and channel coupling capacitance [F/m^2]

        # Channel length modulation
        self.PCLM             = 1.813153        # Coefficient of channel length modulation

        # DIBL effect coefficients
        self.PDIBLC1          = 2.003703E-02    # DIBL effect coefficient 1
        self.PDIBLC2          = 0.00129051      # DIBL effect coefficient 2
        self.DROUT            = 0.56            # Length dependence coefficient of DIBL correction

        # Substrate current induced body effect
        self.PSCBE1           = 4.24e8          # Substrate current effect exponent 1 [V/m]
        self.PSCBE2           = 1.0e-5          # Substrate current effect coefficient 2 [V/m]
        self.PHI              = None            # Surface potential [V]

        # Early voltage parameters
        self.PVAG             = 0.0             # Gate dependence of Early voltage
        self.DELTA            = 0.01            # Effective Vds parameter [V]

        # AC and Capacitance Parameters
        self.COX              = 3.453e-4        # Oxide capacitance per unit area [F/m²]

        # Junction capacitances
        self.CJ               = 5.79e-4         # Zero-bias bulk junction capacitance [F/m^2]
        self.CJSW             = 3.96e-10        # Sidewall junction capacitance [F/m]
        self.PB               = 1.0             # Bulk junction contact potential [V]
        self.PBSW             = 1.0             # Sidewall junction contact potential [V]
        self.MJ               = 0.5             # Bulk junction grading coefficient
        self.MJSW             = 0.33            # Sidewall junction grading coefficient

        # Length and Width Parameters
        self.L                = 1e-6            # Channel length alias
        self.W                = 1e-6            # Channel width alias
        self.Leff             = 1e-6            # Alternative effective length [m]
        self.Weff             = 1e-6            # Alternative effective width [m]

        self.WINT             = -2.02E-07       # Width offset fitting parameter [m]
        self.WLN              = 1.0             # Power of length dependence of width offset
        self.WW               = 0.0             # Coefficient of width dependence for width offset [m^WWN]
        self.WWN              = 1.0             # Power of width dependence of width offset
        self.WWL              = 0.0             # Coefficient of L/W cross term for width offset [m^(WLN+WWN)]
        self.WL               = 0.0             # Coefficient for width dependence
        self.DWG              = 0.0             # Coefficient of Weff's gate dependence [m/V]
        self.DWB              = 0.0             # Coefficient of Weff's body bias dependence [m/V^1/2]

        self.LINT             = 3.75860E-08     # Length offset fitting parameter [m]
        self.LL               = 0.0             # Coefficient of length dependence for length offset [m^LLN]
        self.LLN              = 1.0             # Power of length dependence of length offset
        self.LW               = 0.0             # Coefficient of width dependence for length offset [m^LWN]
        self.LWN              = 1.0             # Power of width dependence of length offset
        self.LWL              = 0.0             # Coefficient of L/W cross term for length offset [m^(LLN+LWN)]
        self.LD               = 50e-9           # Lateral diffusion length [m]

        # Temperature Parameters
        self.T                = 300.0           # Operating temperature [K]
        self.TNOM             = 300.0           # Nominal temperature [K] (alias TREF)
        self.KT1              = 0.0             # Temperature coefficient for Vth [V]
        self.KT1L             = 0.0             # Temperature coefficient for channel length dependence [m·V]
        self.KT2              = 0.022           # Body bias coefficient of Vth temperature effect
        self.UTE              = -1.5            # Mobility temperature exponent
        self.UA1              = 4.31e-9         # Temperature coefficient for UA [m/V]
        self.UB1              = -7.61e-18       # Temperature coefficient for UB [(m/V)^2]
        self.UC1              = -5.69e-11       # Temperature coefficient for UC [m/V^2]
        self.AT               = 3.3e4           # Temperature coefficient for saturation velocity [m/s]
        self.PRT              = 0.0             # Temperature coefficient for RDSW [ohm·μm]
        self.XTI              = 3.0             # Junction current temperature exponent

        # Process Parameters
        self.XT               = 1.55e-7         # Doping depth [m]
        self.NGATE            = 1e18 * 1e6      # polysilicon gate doping [cm⁻³ → m⁻³]
        self.NSS              = 1.0e4           # Surface state density [cm⁻² → m⁻²]
        self.GAMMA            = 0.5276          # Body effect coefficient [V^½]
        self.LAMBDA           = 0.0             # Channel-length modulation parameter [1/V]
        self.ch_type          = 1               # +1 for n-channel -1 for p-channel
        self.TPG              = 1.0             # type of gate material
        self.DELVTO           = 0.0             # Zero-bias threshold voltage shift (V)
        self.KP               = 2.0718e-5       # Intrinsic transconductance parameter

    def calculate_param(self, param_name):
        """
        Calculate MOSFET parameters that can be derived from other parameters.
        For parameters that must be measured/characterized, returns None.
        """
        try:
            match param_name:
                # Fundamental constants - shouldn't need calculation
                case "q" | "k" | "eps0":
                    return None
                    
                # Material properties
                case "eps_ox":
                    return self.eps0 * 3.9  # SiO2 relative permittivity is 3.9
                case "eps_si":
                    return self.eps0 * 11.7  # Si relative permittivity is 11.7
                    
                # Basic model parameters
                case "COX":
                    # Oxide capacitance per unit area
                    return self.eps_ox / self.TOX
                case "GAMMA":
                    # Body effect coefficient
                    return np.sqrt(2 * self.q * self.eps_si * self.NSUB * 1e6) / self.COX
                    
                case "KP":
                    # Transconductance parameter
                    return self.U0 * 1e-4 * self.COX
                case "VTH0":
                    # Threshold voltage at Vbs=0
                    phi = self.calculate_param("PHI")
                    return self.VFB + phi + self.GAMMA * np.sqrt(phi)
                case "PHI":
                    # Bulk Fermi potential
                    return 2 * self.k * self.T / self.q * np.log(self.NSUB * 1e6 / self.ni)
                    
                # Temperature-dependent calculations
                case "UA" if hasattr(self, "UA1") and hasattr(self, "T") and hasattr(self, "TNOM"):
                    return self.UA + self.UA1 * (self.T - self.TNOM)
                case "UB" if hasattr(self, "UB1") and hasattr(self, "T") and hasattr(self, "TNOM"):
                    return self.UB + self.UB1 * (self.T - self.TNOM)
                case "UC" if hasattr(self, "UC1") and hasattr(self, "T") and hasattr(self, "TNOM"):
                    return self.UC + self.UC1 * (self.T - self.TNOM)
                case "VSAT" if hasattr(self, "AT") and hasattr(self, "T") and hasattr(self, "TNOM"):
                    return self.VSAT - self.AT * (self.T/self.TNOM - 1)
                    
                # Length/width dependent calculations
                case "Leff":
                    # Effective channel length
                    return self.L - 2 * self.LD - self.LINT
                case "Weff":
                    # Effective channel width
                    return self.W - 2 * self.WINT
                    
                # If parameter can't be calculated
                case _:
                    return None
        except AttributeError:
            return None

    def var(self, param_name):
        """
        Get a parameter value, calculating it if necessary.
        Raises AttributeError if the parameter doesn't exist.
        """
        if not hasattr(self, param_name):
            raise AttributeError(f"Parameter '{param_name}' not found in MOSFET parameters")
        
        # Get the current value
        value = getattr(self, param_name)
        
        # If value is None, try to calculate it
        if value is None:
            calculated_value = self.calculate_param(param_name)
            if calculated_value is not None:
                return calculated_value
        
        return value