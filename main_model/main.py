"""Main integration script for the modular surface-ocean carbon box model."""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from main_model.parameters import Params
from main_model.modules.biology import tendencies as bio_tendencies
from main_model.modules.carbonate_solver import (
    initialize_dic_from_pco2,
    solubility_co2_weiss74,
    speciate_from_dic_ta,
    ta_from_salinity,
)
from main_model.modules.gas_exchange import co2_flux_and_tendency, k_wanninkhof92
from main_model.modules.plotting import save_diagnostics_plot


def seasonal_temperature(t, T_min, T_max, seasonality=True):
    """Seasonal temperature in degC. t in seconds."""
    t = np.atleast_1d(t).astype(float)
    period = 365.0 * 24.0 * 3600.0
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return np.full_like(t, T_mean)
    amplitude = 0.5 * (T_max - T_min)
    return T_mean + amplitude * np.sin(2.0 * np.pi * t / period)


def rhs(t, y, p: Params):
    """RHS for y=[CO2*, HCO3-, CO3--, G]."""
    co2, hco3, co3, G = [float(v) for v in y]
    T = float(seasonal_temperature(t, p.T_min, p.T_max, p.seasonality)[0])

    K0 = float(solubility_co2_weiss74(T, p.S))
    co2_eq = K0 * p.pCO2_air
    F, dco2_flux = co2_flux_and_tendency(co2, co2_eq, p.U10, T, p.h)

    if p.speciation_on:
        dic = co2 + hco3 + co3
        ta_t = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)
        co2_tgt, hco3_tgt, co3_tgt, _ = speciate_from_dic_ta(dic, ta_t, T, p.S)
        dco2_rel = (co2_tgt - co2) / p.tau_spec_seconds
        dhco3_dt = (hco3_tgt - hco3) / p.tau_spec_seconds
        dco3_dt = (co3_tgt - co3) / p.tau_spec_seconds
    else:
        dco2_rel = 0.0
        dhco3_dt = 0.0
        dco3_dt = 0.0

    if p.biology_on:
        dDIC_bio, dG_dt, _, _ = bio_tendencies(
            DIC=co2 + hco3 + co3,
            G=G,
            T=T,
            Pmax=p.Pmax,
            Km_C=p.Km_C,
            Tref=p.Tref,
            Q10=p.Q10,
            tau_remin_days=p.tau_remin_days,
        )
        if p.speciation_on:
            dco2_bio = dDIC_bio * (co2 / max(co2 + hco3 + co3, 1e-16))
            dco2_flux += dco2_bio
        else:
            dco2_flux += dDIC_bio
    else:
        dG_dt = 0.0

    _ = F
    return [dco2_flux + dco2_rel, dhco3_dt, dco3_dt, dG_dt]


def initialize_state(p: Params):
    """Initialize carbonate species and glucose."""
    T0 = float(seasonal_temperature(0.0, p.T_min, p.T_max, p.seasonality)[0])
    ta = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)

    if p.speciation_on:
        dic0 = initialize_dic_from_pco2(p.pCO2_sw_init, ta, T0, p.S)
        co2_0, hco3_0, co3_0, _ = speciate_from_dic_ta(dic0, ta, T0, p.S)
    else:
        K0_0 = float(solubility_co2_weiss74(T0, p.S))
        co2_0 = K0_0 * p.pCO2_sw_init
        hco3_0 = 0.0
        co3_0 = 0.0

    return [float(co2_0), float(hco3_0), float(co3_0), float(p.G0)]


def run(p: Params):
    """Runs the model and returns outputs as a dict of arrays."""
    y0 = initialize_state(p)

    t_end = p.years * 365.0 * 24.0 * 3600.0
    t_eval = np.arange(0.0, t_end + p.dt_output, p.dt_output)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=p.dt_output,
    )

    T = seasonal_temperature(sol.t, p.T_min, p.T_max, p.seasonality)
    co2, hco3, co3, G = sol.y
    dic = co2 + hco3 + co3
    K0 = solubility_co2_weiss74(T, p.S)
    pco2_sw = co2 / K0

    k_series = np.array([k_wanninkhof92(p.U10, Ti) for Ti in T])
    co2_eq = K0 * p.pCO2_air
    F = k_series * (co2 - co2_eq)

    pH = np.full_like(co2, np.nan)
    if p.speciation_on:
        ta = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)
        for i, Ti in enumerate(T):
            _, _, _, pH[i] = speciate_from_dic_ta(dic[i], ta, float(Ti), p.S)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_co2 = 100.0 * co2 / dic
        frac_hco3 = 100.0 * hco3 / dic
        frac_co3 = 100.0 * co3 / dic

    return {
        "speciation_on": p.speciation_on,
        "success": sol.success,
        "message": sol.message,
        "t_s": sol.t,
        "t_days": sol.t / (24.0 * 3600.0),
        "T_C": T,
        "CO2": co2,
        "HCO3": hco3,
        "CO3": co3,
        "DIC": dic,
        "G": G,
        "DOC": 6.0 * G,
        "pH": pH,
        "pCO2_sw": pco2_sw,
        "F": F,
        "frac_CO2": frac_co2,
        "frac_HCO3": frac_hco3,
        "frac_CO3": frac_co3,
    }


def main():
    out_on = run(Params(speciation_on=True, biology_on=False))
    out_off = run(Params(speciation_on=False, biology_on=False))

    print("Speciation ON success:", out_on["success"])
    print("Speciation OFF success:", out_off["success"])
    print("Final pCO2_sw ON:", out_on["pCO2_sw"][-1])
    print("Final pCO2_sw OFF:", out_off["pCO2_sw"][-1])

    figure_path = save_diagnostics_plot(out_on, out_off)
    print("Saved plot:", figure_path)
    return out_on, out_off


if __name__ == "__main__":
    main()
