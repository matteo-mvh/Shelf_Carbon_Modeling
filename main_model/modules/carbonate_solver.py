"""Carbonate chemistry and air-sea CO2 diagnostic helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

REFERENCE_SEAWATER_DENSITY_KG_PER_M3 = 1025.0


def solubility_co2_weiss74(T, S, rho=REFERENCE_SEAWATER_DENSITY_KG_PER_M3):
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
    k0_mol_kg_atm = np.exp(lnK0)
    return k0_mol_kg_atm * rho * 1e-6


def K1_K2(T, S):
    """Mehrbach (1973)-style refit for carbonic acid constants."""
    Tk = T + 273.15
    pK1 = 3633.86 / Tk - 61.2172 + 9.6777 * np.log(Tk) - 0.011555 * S + 0.0001152 * S**2
    pK2 = 471.78 / Tk + 25.9290 - 3.16967 * np.log(Tk) - 0.01781 * S + 0.0001122 * S**2
    return 10 ** (-pK1), 10 ** (-pK2)


def Kw_const(T):
    """Pure water dissociation constant (approx)."""
    Tk = T + 273.15
    return np.exp(148.96502 - 13847.26 / Tk - 23.6521 * np.log(Tk))


def Kb_dickson(T, S):
    """Dickson (1990) boric acid dissociation constant, Kb (mol kg^-1)."""
    Tk = T + 273.15
    sqrtS = np.sqrt(S)
    lnKb = (
        (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S**1.5 - 0.0996 * S**2) / Tk
        + (148.0248 + 137.1942 * sqrtS + 1.62142 * S)
        + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * np.log(Tk)
        + 0.053105 * sqrtS * Tk
    )
    return np.exp(lnKb)


def total_boron(S, rho=REFERENCE_SEAWATER_DENSITY_KG_PER_M3):
    """Uppström (1974): total boron converted to mol m^-3."""
    tb_mol_kg = 0.0004157 * (S / 35.0)
    return tb_mol_kg * rho


def ta_from_salinity(S, ta0, S0=35.0):
    """Scale alkalinity with salinity via dilution/concentration."""
    return ta0 * (S / S0)


def bracket_root(fun, x_min=0.0, x_max=14.0, n=2000):
    """Find robust pH bracket for root finding."""
    xs = np.linspace(x_min, x_max, n)
    fs = np.array([fun(x) for x in xs], dtype=float)
    mask = np.isfinite(fs)
    xs, fs = xs[mask], fs[mask]
    if len(xs) < 2:
        raise ValueError("Residual is non-finite across the bracket search range.")

    signs = np.sign(fs)
    for i in range(len(xs) - 1):
        if signs[i] == 0:
            return xs[i] - 1e-6, xs[i] + 1e-6
        if signs[i] * signs[i + 1] < 0:
            return xs[i], xs[i + 1]

    j = int(np.argmin(np.abs(fs)))
    return max(x_min, xs[j] - 0.5), min(x_max, xs[j] + 0.5)


def _solve_ph_from_residual(residual, pH_guess=None, x_min=0.0, x_max=14.0):
    """Solve pH robustly from a residual function.

    Preferred path uses Brent with a sign-change bracket. If no sign change can
    be found (which can occur transiently for extreme/unphysical states during
    ODE stepping), fall back to the pH value that minimizes |residual| on a
    dense scan. This prevents hard model crashes while keeping continuity.
    """
    if pH_guess is not None:
        bracket = _find_ph_bracket_from_guess(residual, pH_guess, x_min=x_min, x_max=x_max)
        if bracket is None:
            a, b = bracket_root(residual, x_min=x_min, x_max=x_max)
        else:
            a, b = bracket
    else:
        a, b = bracket_root(residual, x_min=x_min, x_max=x_max)

    fa, fb = residual(a), residual(b)
    if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) * np.sign(fb) <= 0:
        return brentq(residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)

    xs = np.linspace(x_min, x_max, 4000)
    fs = np.array([residual(x) for x in xs], dtype=float)
    mask = np.isfinite(fs)
    if not np.any(mask):
        raise ValueError("Residual is non-finite across the pH search range.")
    xs_valid = xs[mask]
    fs_valid = fs[mask]
    return float(xs_valid[int(np.argmin(np.abs(fs_valid)))])


def _find_ph_bracket_from_guess(residual, pH_guess, x_min=3.0, x_max=11.0):
    """Try to bracket the pH root near a prior estimate before full scanning."""
    half_widths = (0.15, 0.35, 0.75, 1.5)
    for hw in half_widths:
        a = max(x_min, float(pH_guess) - hw)
        b = min(x_max, float(pH_guess) + hw)
        fa, fb = residual(a), residual(b)
        if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) * np.sign(fb) <= 0:
            return a, b
    return None


def speciate_from_dic_ta(dic, ta, T, S, pH_guess=None):
    """Given DIC and TA (mol m^-3), solve pH and carbonate species.

    Uses the simplified TA closure:
    TA = [HCO3-] + 2[CO3--] + [OH-] - [H+].
    """
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)

    def residual(pH):
        H = 10 ** (-pH)
        co2 = dic / (1 + K1 / H + K1 * K2 / H**2)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2
        oh = Kw / H
        ta_calc = hco3 + 2 * co3 + oh - H
        return float(ta) - ta_calc

    pH = _solve_ph_from_residual(residual, pH_guess=pH_guess)
    H = 10 ** (-pH)
    co2 = dic / (1 + K1 / H + K1 * K2 / H**2)
    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2
    return co2, hco3, co3, pH


def initialize_dic_from_pco2(pco2, ta, T, S):
    """Given pCO2 (uatm) and TA, return DIC at equilibrium."""
    K0 = solubility_co2_weiss74(T, S)
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)
    co2 = K0 * pco2

    def residual(pH):
        H = 10 ** (-pH)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2
        oh = Kw / H
        ta_calc = hco3 + 2 * co3 + oh - H
        return float(ta) - ta_calc

    pH = _solve_ph_from_residual(residual)
    H = 10 ** (-pH)
    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2
    return co2 + hco3 + co3


def pco2_from_dic_proxy(DIC, T, S):
    """Compatibility proxy diagnostic (used only if speciation is disabled)."""
    K0 = float(solubility_co2_weiss74(T, S))
    return float(DIC) / K0, K0
