"""Carbonate chemistry solver module scaffold."""

import numpy as np

from parameters import PARAMS

REFERENCE_SEAWATER_DENSITY_KG_PER_M3 = 1025.0


def solubility_co2_weiss74(T, S):
    """Weiss (1974) CO2 solubility K0 (mol m^-3 uatm^-1)."""
    Tk = np.asarray(T, dtype=float) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    lnK0 = (
        A1
        + A2 * (100.0 / Tk)
        + A3 * np.log(Tk / 100.0)
        + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
    )
    return np.exp(lnK0)


def pco2_from_dic_proxy(DIC, T, S):
    """
    Proxy diagnostic: pCO2_sw = DIC / K0.
    (Later you'll replace with full carbonate speciation using DIC + TA.)
    """
    K0 = float(solubility_co2_weiss74(T, S))
    return float(DIC) / K0, K0


def solve_carbonate_system(state: dict) -> dict:
    """Return simple carbonate-system diagnostics from DIC and Weiss solubility."""
    dic_umol_per_kg = float(state.get("dic_umol_per_kg", PARAMS.dic_umol_per_kg))
    temperature_c = float(state.get("temperature_c", PARAMS.temperature_c))
    salinity_psu = float(state.get("salinity_psu", PARAMS.salinity_psu))

    dic_mol_per_m3 = (
        dic_umol_per_kg * 1e-6 * REFERENCE_SEAWATER_DENSITY_KG_PER_M3
    )
    pco2_ocean_uatm, k0_mol_per_m3_uatm = pco2_from_dic_proxy(
        dic_mol_per_m3, temperature_c, salinity_psu
    )

    return {
        "pco2_ocean_uatm": pco2_ocean_uatm,
        "ph_total_scale": 8.0,
        "k0_mol_per_m3_uatm": k0_mol_per_m3_uatm,
    }
