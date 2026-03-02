"""Centralized parameter definitions for the surface-ocean carbon box model."""

from dataclasses import dataclass

from main_model.modules.carbonate_solver import REFERENCE_SEAWATER_DENSITY_KG_PER_M3


@dataclass(frozen=True)
class Params:
    # ---- Physical / forcing ----
    S: float = 30.0
    U10: float = 6.0
    h: float = 50.0
    pCO2_air: float = 420.0

    # Seasonal temperature forcing
    seasonality: bool = True
    T_min: float = 2.0
    T_max: float = 20.0

    # ---- Carbonate configuration ----
    speciation_on: bool = True
    tau_spec_seconds: float = 2.0 * 3600.0
    S0_ta: float = 35.0
    ta0_mol_per_m3: float = 2300e-6 * REFERENCE_SEAWATER_DENSITY_KG_PER_M3

    # ---- Biology toggle and parameters (toy) ----
    biology_on: bool = True
    Pmax: float = 5.0e-7
    Km_C: float = 1.0e-2
    Tref: float = 15.0
    Q10: float = 2.0
    tau_remin_days: float = 60.0

    # ---- Integration settings ----
    years: float = 5.0
    dt_output: float = 24 * 3600
    plot_last_year_only: bool = False

    # ---- Initial conditions ----
    pCO2_sw_init: float = 300.0
    G0: float = 0.0


PARAMS = Params()
