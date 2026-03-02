"""Carbonate chemistry helper functions."""

import numpy as np


def solubility_co2(T: float, S: float) -> float:
    """Weiss (1974) CO2 solubility K0 (mol m^-3 µatm^-1)."""
    Tk = float(T) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    lnK0 = A1 + A2 * (100 / Tk) + A3 * np.log(Tk / 100) + S * (B1 + B2 * (Tk / 100) + B3 * (Tk / 100) ** 2)
    return float(np.exp(lnK0))


def pco2_from_dic_proxy(DIC: float, T: float, S: float):
    """Simple proxy pCO2 = DIC / K0, returns (pCO2_sw, K0)."""
    K0 = solubility_co2(T, S)
    return float(DIC / K0), float(K0)


# Backwards-compatible scaffold name.
def solve_carbonate_system(state: dict) -> dict:
    pco2, _ = pco2_from_dic_proxy(state.get("dic_umol_per_kg", 0.0), 15.0, 35.0)
    return {"pco2_ocean_uatm": pco2, "ph_total_scale": 8.0}
