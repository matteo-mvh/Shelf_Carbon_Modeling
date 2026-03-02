"""Centralized parameter definitions for the modular surface-ocean carbon model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Configuration for the model integration."""

    # Time control
    dt_output: float = 24.0 * 3600.0
    years: int = 5

    # Surface box and forcing
    h: float = 50.0
    T_min: float = 2.0
    T_max: float = 20.0
    seasonality: bool = True
    S: float = 30.0
    U10: float = 6.0
    pCO2_air: float = 420.0
    pCO2_sw_init: float = 300.0

    # Biology
    biology_on: bool = True
    Pmax: float = 5.0e-7
    Km_C: float = 1.0e-2
    Tref: float = 15.0
    Q10: float = 2.0
    tau_remin_days: float = 60.0
    G0: float = 0.0


# Backwards-compatible alias used by earlier scaffold code.
ModelParameters = Params
PARAMS = Params()
