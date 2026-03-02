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
from main_model.modules.biology import (
    glucose_production_rate,
    remin_rate_from_tau_days,
    tendencies as bio_tendencies,
)
from main_model.modules.carbonate_solver import (
    initialize_dic_from_pco2,
    solubility_co2_weiss74,
    speciate_from_dic_ta,
    ta_from_salinity,
)
from main_model.modules.gas_exchange import co2_flux_and_tendency, k_wanninkhof92
from main_model.modules.plotting import save_diagnostics_plot
from main_model.modules.plotting import save_biology_comparison_plot


def seasonal_temperature(t, T_min, T_max, seasonality=True):
    """Seasonal temperature in degC. t in seconds."""
    t = np.atleast_1d(t).astype(float)
    period = 365.0 * 24.0 * 3600.0
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return np.full_like(t, T_mean)
    amplitude = 0.5 * (T_max - T_min)
    return T_mean + amplitude * np.sin(2.0 * np.pi * t / period)


def rhs(t, y, p: Params, pH_guess=None):
    """RHS for y=[DIC, G] with diagnostic carbonate speciation."""
    dic, G = [float(v) for v in y]
    T = float(seasonal_temperature(t, p.T_min, p.T_max, p.seasonality)[0])
    ta_t = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)

    co2, _, _, pH = speciate_from_dic_ta(dic, ta_t, T, p.S, pH_guess=pH_guess)

    K0 = float(solubility_co2_weiss74(T, p.S))
    co2_eq = K0 * p.pCO2_air
    _, dDIC_flux = co2_flux_and_tendency(co2, co2_eq, p.U10, T, p.h)

    if p.biology_on:
        dDIC_bio, dG_dt, _, _ = bio_tendencies(
            DIC=dic,
            G=G,
            T=T,
            Pmax=p.Pmax,
            Km_C=p.Km_C,
            Tref=p.Tref,
            Q10=p.Q10,
            tau_remin_days=p.tau_remin_days,
        )
    else:
        dDIC_bio = 0.0
        dG_dt = 0.0

    return [dDIC_flux + dDIC_bio, dG_dt], pH


def initialize_state(p: Params):
    """Initialize DIC and glucose."""
    T0 = float(seasonal_temperature(0.0, p.T_min, p.T_max, p.seasonality)[0])
    ta = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)

    dic0 = initialize_dic_from_pco2(p.pCO2_sw_init, ta, T0, p.S)
    return [float(dic0), float(p.G0)]


def run(p: Params):
    """Runs the model and returns outputs as a dict of arrays."""
    y0 = initialize_state(p)

    t_end = p.years * 365.0 * 24.0 * 3600.0
    t_eval = np.arange(0.0, t_end + p.dt_output, p.dt_output)

    ph_cache = {"value": 8.1}

    def rhs_cached(t, y):
        dydt, pH = rhs(t, y, p, pH_guess=ph_cache["value"])
        ph_cache["value"] = pH
        return dydt

    sol = solve_ivp(
        fun=rhs_cached,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=p.dt_output,
    )

    T = seasonal_temperature(sol.t, p.T_min, p.T_max, p.seasonality)
    dic, G = sol.y
    K0 = solubility_co2_weiss74(T, p.S)
    co2 = np.full_like(dic, np.nan)
    hco3 = np.full_like(dic, np.nan)
    co3 = np.full_like(dic, np.nan)
    pH = np.full_like(dic, np.nan)
    ta = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)
    pH_guess = ph_cache["value"]
    for i, Ti in enumerate(T):
        co2[i], hco3[i], co3[i], pH[i] = speciate_from_dic_ta(
            dic[i], ta, float(Ti), p.S, pH_guess=pH_guess
        )
        pH_guess = pH[i]

    pco2_sw = co2 / K0
    k_series = np.array([k_wanninkhof92(p.U10, Ti) for Ti in T])
    co2_eq = K0 * p.pCO2_air
    F = k_series * (co2 - co2_eq)

    if p.biology_on:
        P_glucose = np.array(
            [
                glucose_production_rate(dic_i, Ti, p.Pmax, p.Km_C, Tref=p.Tref, Q10=p.Q10)
                for dic_i, Ti in zip(dic, T)
            ]
        )
        R_glucose = remin_rate_from_tau_days(p.tau_remin_days) * G
    else:
        P_glucose = np.zeros_like(dic)
        R_glucose = np.zeros_like(dic)

    glucose_c_flux = 6.0 * P_glucose * p.h
    remin_c_flux = 6.0 * R_glucose * p.h

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_co2 = 100.0 * co2 / dic
        frac_hco3 = 100.0 * hco3 / dic
        frac_co3 = 100.0 * co3 / dic

    return {
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
        "glucose_prod_flux": glucose_c_flux,
        "remin_flux": remin_c_flux,
        "pH": pH,
        "pCO2_sw": pco2_sw,
        "F": F,
        "frac_CO2": frac_co2,
        "frac_HCO3": frac_hco3,
        "frac_CO3": frac_co3,
    }


def main():
    params_on = Params(biology_on=True)
    params_off = Params(biology_on=False)

    out_on = run(params_on)
    out_off = run(params_off)

    print("Run success (ON):", out_on["success"])
    print("Run success (OFF):", out_off["success"])
    print("Final pCO2_sw (ON):", out_on["pCO2_sw"][-1])
    print("Final pCO2_sw (OFF):", out_off["pCO2_sw"][-1])

    t = out_on["t_s"]
    sec_per_year = 365.0 * 24.0 * 3600.0
    mask_last_year = t >= (t[-1] - sec_per_year)

    uptake_on_last = np.trapezoid((-out_on["F"])[mask_last_year], t[mask_last_year])
    uptake_off_last = np.trapezoid((-out_off["F"])[mask_last_year], t[mask_last_year])
    delta_uptake_last = uptake_on_last - uptake_off_last

    dic_on = out_on["DIC"]
    doc_on = out_on["DOC"]
    d_doc_dt = np.gradient(doc_on, t)
    drawdown_amount_last_year = np.trapezoid(d_doc_dt[mask_last_year], t[mask_last_year])

    dic_mean_last = np.mean(dic_on[mask_last_year])
    doc_mean_last = np.mean(doc_on[mask_last_year])
    total_c_mean_last = np.mean((dic_on + doc_on)[mask_last_year])

    print("=== Last-year (final 365 days) ===")
    print(f"Air->sea uptake (biology OFF): {uptake_off_last:.6e} mol C m^-2 yr^-1")
    print(f"Air->sea uptake (biology ON) : {uptake_on_last:.6e} mol C m^-2 yr^-1")
    print(f"Biology effect (ON - OFF)   : {delta_uptake_last:.6e} mol C m^-2 yr^-1")
    print(f"Uptake enhancement factor (ON / OFF): {uptake_on_last / uptake_off_last:.3f}")
    print("")
    print(f"Net bio DIC->DOC conversion (ON): {drawdown_amount_last_year:.6e} mol C m^-3 yr^-1")
    print(f"Mean DIC (ON)   : {dic_mean_last:.6e} mol C m^-3")
    print(f"Mean DOC (ON)   : {doc_mean_last:.6e} mol C m^-3")
    print(f"Mean Total (ON) : {total_c_mean_last:.6e} mol C m^-3")

    figure_path = save_biology_comparison_plot(
        out_on,
        out_off,
        plot_last_year_only=params_on.plot_last_year_only,
    )
    print("Saved plot:", figure_path)

    single_run_plot_path = save_diagnostics_plot(
        out_on,
        output_path="results/main_model_diagnostics_on.png",
        plot_last_year_only=params_on.plot_last_year_only,
    )
    print("Saved single-run plot (ON):", single_run_plot_path)
    return out_on, out_off


if __name__ == "__main__":
    main()
