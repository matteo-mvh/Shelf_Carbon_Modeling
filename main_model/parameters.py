"""Centralized parameter definitions for the surface-ocean carbon box model.

All model modules should import parameters from this file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParameters:
    """Container for model-wide parameters.

    Values below are placeholders so module integration can proceed with
    a stable interface before final calibration.
    """

    # Time control
    dt_days: float = 1.0
    simulation_days: int = 365

    # Surface box properties
    mixed_layer_depth_m: float = 30.0
    temperature_c: float = 15.0
    salinity_psu: float = 35.0

    # Atmospheric forcing
    atmospheric_pco2_uatm: float = 420.0
    wind_speed_m_per_s: float = 7.0

    # Initial dissolved inorganic carbon system state
    dic_umol_per_kg: float = 2100.0
    alkalinity_umol_per_kg: float = 2300.0

    # Biology placeholders
    primary_production_umol_c_per_kg_per_day: float = 2.0
    respiration_umol_c_per_kg_per_day: float = 1.5


PARAMS = ModelParameters()
