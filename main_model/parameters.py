"""Centralized parameter definitions for the surface-ocean carbon box model.

All units are SI unless otherwise stated.
Carbon concentrations are in mol C m^-3.
Fluxes are mol C m^-2 s^-1.
"""

from dataclasses import dataclass

from main_model.modules.carbonate_solver import REFERENCE_SEAWATER_DENSITY_KG_PER_M3


@dataclass(frozen=True)
class Params:
    # ============================================================
    # Physical / forcing parameters
    # ============================================================

    S: float = 30.0
    # Salinity [psu]

    U10: float = 7.0
    # 10-m wind speed [m s^-1] (controls gas transfer velocity)

    pCO2_air: float = 420.0
    # Atmospheric partial pressure of CO2 [µatm]

    # ============================================================
    # Seasonal MLD forcing
    # ============================================================
    
    mld: float = 50.0
    # Fixed mixed-layer depth [m]

    mld_seasonality: bool = True
    # Toggle seasonal mixed-layer-depth forcing

    mld_winter: float = 90.0
    # Winter mixed-layer depth [m]

    mld_summer: float = 20.0
    # Summer mixed-layer depth [m]

    mld_peak_day: float = 20.0
    # Day of year of maximum MLD [days]

    deep_entrainment_dic: float = 2.25
    # Deep-water DIC concentration used for entrainment [mol C m^-3]

    deep_entrainment_ta: float = 2.35
    # Deep-water TA concentration used for entrainment [mol m^-3]

    deep_entrainment_ldoc: float = 0.001
    # Deep-water LDOC concentration used for entrainment [mol C m^-3]

    deep_entrainment_sdoc: float = 0.005
    # Deep-water SDOC concentration used for entrainment [mol C m^-3]

    deep_entrainment_rdoc: float = 0.015
    # Deep-water RDOC concentration used for entrainment [mol C m^-3]

    
    # ============================================================
    # Seasonal temperature forcing
    # ============================================================

    seasonality: bool = True
    # Toggle all seasonal forcing (temperature + light)

    T_min: float = 2.0
    # Minimum surface temperature [°C]

    T_max: float = 16.0
    # Maximum surface temperature [°C]

    seasonal_cycle_days: float = 365.0
    # Length of the seasonal cycle [days]

    temperature_peak_day: float = 190.0
    # Day of year of maximum SST [days]

    # ============================================================
    # Seasonal light forcing for photosynthesis
    # ============================================================

    light_phase_days: float = 0.0
    # Phase shift for light forcing [days]

    light_peak_day: float = 185.0
    # Day of year of maximum light [days]

    light_sharpness: float = 2.0
    # Shape exponent for the seasonal light curve (>1 sharpens summer peak)

    light_winter: float = 50.0
    # Winter minimum photosynthetically active radiation [µmol photons m^-2 s^-1]

    light_summer: float = 850.0
    # Summer maximum photosynthetically active radiation [µmol photons m^-2 s^-1]


    # ============================================================
    # Carbonate system configuration
    # ============================================================

    S0_ta: float = 35.0
    # Reference salinity for alkalinity scaling [psu]

    ta0_mol_per_m3: float = 2300e-6 * REFERENCE_SEAWATER_DENSITY_KG_PER_M3
    # Reference total alkalinity at S0_ta [mol m^-3]
    # (2300 µmol kg^-1 converted using reference seawater density)


    # ============================================================
    # Biology and DOC parameters (reduced DIC-DOC formulation)
    # ============================================================

    mu_bio: float = 4.5e-8
    # Biological growth-rate scaling µ [s^-1] in Fprod = µ * PP(L)

    # Fitted production-irradiance parameters (section 6.1)
    pp_Pmax: float = 4.880930
    pp_K_L: float  = 134.427668
    pp_n: float    = 1.076659
    pp_K_I: float  = 1666.298377

    # DOC partitioning fractions (must sum to 1)
    alpha_l: float = 0.45
    alpha_s: float = 0.55
    alpha_r: float = 0.0

    # DOC remineralization rates [s^-1]
    lambda_l: float = 6.0e-6
    lambda_s: float = 9.0e-7
    lambda_r: float = 8.0e-10

    # DOC aging rates [s^-1]
    gamma_l: float = 4.0e-8
    gamma_s: float = 8.0e-9

    # ============================================================
    # Integration settings
    # ============================================================

    years: float = 30.0
    # Total simulation length [years]

    dt_output: float = 24 * 3600
    # Output interval [s]

    plot_last_year_only: bool = True
    # If True, diagnostics focus on final year

    # ============================================================
    # Initial conditions
    # ============================================================

    pCO2_sw_init: float = 400.0
    # Initial surface seawater pCO2 [µatm]

    LDOC0: float = 0.003
    # Initial labile DOC [mol C m^-3]

    SDOC0: float = 0.025
    # Initial semi-labile DOC [mol C m^-3]

    RDOC0: float = 0.015
    # Initial refractory DOC [mol C m^-3]


# Default instance
PARAMS = Params()
