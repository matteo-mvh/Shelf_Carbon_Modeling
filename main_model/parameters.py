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

    U10: float = 6.0
    # 10-m wind speed [m s^-1] (controls gas transfer velocity)

    h: float = 50.0
    # Mixed-layer depth [m]

    pCO2_air: float = 420.0
    # Atmospheric partial pressure of CO2 [µatm]


    # ============================================================
    # Seasonal temperature forcing
    # ============================================================

    seasonality: bool = True
    # Toggle seasonal temperature forcing

    T_min: float = 2.0
    # Minimum surface temperature [°C]

    T_max: float = 20.0
    # Maximum surface temperature [°C]


    # ============================================================
    # Seasonal light forcing for photosynthesis
    # ============================================================

    light_seasonality: bool = True
    # Toggle seasonal light forcing

    light_phase_days: float = 0.0
    # Phase shift for light forcing [days]
    # Light is computed as a squared sinusoid and normalized to [0, 1].


    # ============================================================
    # Carbonate system configuration
    # ============================================================

    S0_ta: float = 35.0
    # Reference salinity for alkalinity scaling [psu]

    ta0_mol_per_m3: float = 2300e-6 * REFERENCE_SEAWATER_DENSITY_KG_PER_M3
    # Reference total alkalinity at S0_ta [mol m^-3]
    # (2300 µmol kg^-1 converted using reference seawater density)


    # ============================================================
    # Biology parameters (toy ecosystem)
    # ============================================================

    biology_on: bool = True
    # Toggle biological processes on/off

    Pmax: float = 1.0e-8
    # Maximum glucose production rate [mol glucose m^-3 s^-1]

    Km_C: float = 1.0e-2
    # Half-saturation constant for DIC limitation [mol C m^-3]

    tau_remin_days: float = 60.0
    # Remineralization timescale [days]


    # ============================================================
    # Integration settings
    # ============================================================

    years: float = 5.0
    # Total simulation length [years]

    dt_output: float = 24 * 3600
    # Output interval [s]

    plot_last_year_only: bool = False
    # If True, diagnostics focus on final year


    # ============================================================
    # Initial conditions
    # ============================================================

    pCO2_sw_init: float = 300.0
    # Initial surface seawater pCO2 [µatm]

    G0: float = 0.0
    # Initial glucose concentration [mol glucose m^-3]


# Default instance
PARAMS = Params()
