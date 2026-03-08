"""NPZD-based calibration of PP(L) parameters used by the DOC biology module.

This module reproduces the provided NPZ light-sweep workflow and fits
"saturation + Hill" parameters:

    PP(L) = A1 * L / (L + K1) + A2 * L^n2 / (L^n2 + K2^n2)

Use ``apply_fitted_light_parameters_to_file`` only when you want to update the
PP(L) defaults in ``main_model/parameters.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class LightFitResult:
    """Fitted saturation + Hill coefficients for PP(L)."""

    pp_A1: float
    pp_K1: float
    pp_A2: float
    pp_K2: float
    pp_n2: float


DEFAULT_NPZ_PARAMS = {
    # Phytoplankton growth
    "mu_max": 0.9,
    "k_L": 100.0,
    "k_N": 1.5,
    # Grazing
    "g_max": 0.85,
    "k_P": 1.8,
    # Losses / recycling
    "m_P": 0.04,
    "m_Z": 0.06,
    "beta": 0.65,
    "remin_P": 1.0,
    "remin_Z": 1.0,
    # Constant light forcing
    "L_const": 180.0,
}


def npz_model(t, y, p):
    """Three-compartment NPZ model with constant light forcing."""
    del t

    nutrient, phyto, zoo = y

    nutrient = max(float(nutrient), 0.0)
    phyto = max(float(phyto), 0.0)
    zoo = max(float(zoo), 0.0)

    light = float(p["L_const"])

    f_light = light / (light + float(p["k_L"])) if light > 0 else 0.0
    f_nutrient = nutrient / (nutrient + float(p["k_N"])) if nutrient > 0 else 0.0

    growth = float(p["mu_max"]) * f_light * f_nutrient * phyto
    grazing = float(p["g_max"]) * zoo * phyto / (phyto + float(p["k_P"])) if phyto > 0 else 0.0
    mort_phyto = float(p["m_P"]) * phyto
    mort_zoo = float(p["m_Z"]) * zoo**2

    d_nutrient_dt = (
        -growth
        + float(p["remin_P"]) * mort_phyto
        + float(p["remin_Z"]) * mort_zoo
        + (1.0 - float(p["beta"])) * grazing
    )
    d_phyto_dt = growth - grazing - mort_phyto
    d_zoo_dt = float(p["beta"]) * grazing - mort_zoo

    return [d_nutrient_dt, d_phyto_dt, d_zoo_dt]


def sat_plus_hill(light, A1, K1, A2, K2, n2):
    """Saturation + Hill form for PP(L) used by the carbon model."""
    light = np.asarray(light, dtype=float)
    term1 = A1 * light / (light + K1)
    term2 = A2 * (light**n2) / (light**n2 + K2**n2)
    return term1 + term2


def fit_light_parameters_for_params(
    light_levels=None,
    y0=None,
    t_end=5000.0,
    avg_window=2000.0,
):
    """Run NPZ light sweep and fit PP(L) coefficients for Params.pp_* fields."""
    if light_levels is None:
        light_levels = np.linspace(0.0, 1500.0, 100)
    else:
        light_levels = np.asarray(light_levels, dtype=float)

    if y0 is None:
        y_current = np.array([12.0, 0.1, 0.1], dtype=float)
    else:
        y_current = np.asarray(y0, dtype=float)

    t_eval = np.linspace(0.0, float(t_end), 5000)
    phyto_end_values = []

    for light in light_levels:
        params_run = DEFAULT_NPZ_PARAMS.copy()
        params_run["L_const"] = float(light)

        solution = solve_ivp(
            fun=lambda t, y: npz_model(t, y, params_run),
            t_span=(0.0, float(t_end)),
            y0=y_current,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not solution.success:
            raise RuntimeError(f"NPZ integration failed at L={light}: {solution.message}")

        mask_last = solution.t >= (float(t_end) - float(avg_window))
        phyto_avg = float(np.mean(solution.y[1, mask_last]))
        phyto_end_values.append(phyto_avg)

        # Sequential sweep continuation.
        y_current = solution.y[:, -1]

    # Convert phytoplankton from nitrogen to carbon units.
    carbon_to_nitrogen_ratio = 106.0 / 16.0
    phyto_end_values = np.asarray(phyto_end_values, dtype=float) * carbon_to_nitrogen_ratio

    p0 = [1.5, 80.0, 6.0, 450.0, 5.0]
    lower_bounds = [0.0, 1.0, 0.0, 1.0, 1.0]
    upper_bounds = [20.0, 1000.0, 20.0, 2000.0, 10.0]

    popt, _ = curve_fit(
        sat_plus_hill,
        light_levels,
        phyto_end_values,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000,
    )

    return LightFitResult(
        pp_A1=float(popt[0]),
        pp_K1=float(popt[1]),
        pp_A2=float(popt[2]),
        pp_K2=float(popt[3]),
        pp_n2=float(popt[4]),
    )


def format_params_block(result: LightFitResult) -> str:
    """Return ready-to-paste lines for the Params PP(L) section."""
    return (
        f"pp_A1: float = {result.pp_A1:.6f}\n"
        f"pp_K1: float = {result.pp_K1:.6f}\n"
        f"pp_A2: float = {result.pp_A2:.6f}\n"
        f"pp_K2: float = {result.pp_K2:.6f}\n"
        f"pp_n2: float = {result.pp_n2:.6f}"
    )


def apply_fitted_light_parameters_to_file(parameters_path: str | Path = "main_model/parameters.py") -> LightFitResult:
    """Fit PP(L) coefficients and overwrite pp_* defaults in parameters.py."""
    result = fit_light_parameters_for_params()
    path = Path(parameters_path)
    text = path.read_text(encoding="utf-8")

    replacements = {
        "pp_A1": result.pp_A1,
        "pp_K1": result.pp_K1,
        "pp_A2": result.pp_A2,
        "pp_K2": result.pp_K2,
        "pp_n2": result.pp_n2,
    }

    for key, value in replacements.items():
        pattern = rf"^{key}: float = [^\n]+$"
        replacement = f"{key}: float = {value:.6f}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"Could not update parameter line for {key} in {path}")

    path.write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    fitted = apply_fitted_light_parameters_to_file()
    print("Updated main_model/parameters.py with fitted PP(L) parameters:")
    print(format_params_block(fitted))
