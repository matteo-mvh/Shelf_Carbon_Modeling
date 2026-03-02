"""Centralized parameter definitions for the surface-ocean carbon box model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # ---- Physical / forcing ----
    S: float = 30.0
    U10: float = 6.0
    h: float = 50.0  # mixed-layer depth [m]
    pCO2_air: float = 420.0  # [uatm]

    # Seasonal temperature forcing
    seasonality: bool = True
    T_min: float = 2.0
    T_max: float = 20.0

    # ---- Biology toggle and parameters (toy) ----
    biology_on: bool = True
    Pmax: float = 5.0e-7  # mol glucose m^-3 s^-1
    Km_C: float = 1.0e-2  # mol/m^3 (proxy-limitation on DIC)
    Tref: float = 15.0  # degC
    Q10: float = 2.0
    tau_remin_days: float = 60.0  # days (0 -> off)

    # ---- Integration settings ----
    years: float = 5.0
    dt_output: float = 24 * 3600  # seconds (daily output)

    # ---- Initial conditions (defined via initial seawater pCO2) ----
    pCO2_sw_init: float = 300.0  # [uatm]
    G0: float = 0.0  # initial glucose [mol glucose m^-3]


PARAMS = Params()
