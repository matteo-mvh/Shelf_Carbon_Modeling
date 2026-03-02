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
from main_model.modules.gas_exchange import co2_flux_and_tendency
from main_model.modules.plotting import save_diagnostics_plot
from main_model.modules.plotting import save_biology_comparison_plot


DEFAULT_NON_SEASONAL_MLD_METERS = 50.0


def seasonal_temperature(
    t,
    T_min,
    T_max,
    seasonality=True,
    peak_day=230.0,
    cycle_days=365.0,
):
    """Seasonal temperature in degC. t in seconds."""
    t = np.atleast_1d(t).astype(float)
    period = cycle_days * 24.0 * 3600.0
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return np.full_like(t, T_mean)
    peak_seconds = peak_day * 24.0 * 3600.0

    temp_norm = 0.5 * (1.0 + np.cos(2.0 * np.pi * (t - peak_seconds) / period))
    return T_min + (T_max - T_min) * temp_norm

def seasonal_light(
    t,
    seasonality=True,
    phase_days=0.0,
    peak_day=172.0,
    sharpness=2.0,
    winter_light=120.0,
    summer_light=1000.0,
    cycle_days=365.0,
):
    """
    Seasonal light forcing [µmol photons m^-2 s^-1].
    """
    t = np.atleast_1d(t).astype(float)
    period = cycle_days * 24.0 * 3600.0

    mean_light = 0.5 * (winter_light + summer_light)
    if not seasonality:
        return np.full_like(t, mean_light)

    phase_seconds = phase_days * 24.0 * 3600.0
    peak_seconds = peak_day * 24.0 * 3600.0

    base = 0.5 * (1.0 + np.cos(2.0 * np.pi * (t - peak_seconds - phase_seconds) / period))
    light_norm = np.clip(base, 0.0, 1.0) ** sharpness

    return winter_light + (summer_light - winter_light) * light_norm


def seasonal_mld(
    t,
    seasonality=True,
    winter_depth=80.0,
    summer_depth=20.0,
    peak_day=15.0,
    cycle_days=365.0,
):
    """Seasonal mixed-layer depth [m]."""
    t = np.atleast_1d(t).astype(float)
    if not seasonality:
        return np.full_like(t, DEFAULT_NON_SEASONAL_MLD_METERS)

    period = cycle_days * 24.0 * 3600.0
    peak_seconds = peak_day * 24.0 * 3600.0
    mld_norm = 0.5 * (1.0 + np.cos(2.0 * np.pi * (t - peak_seconds) / period))
    return summer_depth + (winter_depth - summer_depth) * mld_norm


def seasonal_mld_tendency(
    t,
    seasonality=True,
    winter_depth=80.0,
    summer_depth=20.0,
    peak_day=15.0,
    cycle_days=365.0,
):
    """Time tendency of seasonal mixed-layer depth [m s^-1]."""
    t = np.atleast_1d(t).astype(float)
    if not seasonality:
        return np.zeros_like(t)

    period = cycle_days * 24.0 * 3600.0
    peak_seconds = peak_day * 24.0 * 3600.0
    omega = 2.0 * np.pi / period
    amplitude = 0.5 * (winter_depth - summer_depth)
    return -amplitude * omega * np.sin(omega * (t - peak_seconds))


def rhs(t, y, p: Params, pH_guess=None):
    """RHS for y=[DIC, G, TA] with diagnostic carbonate speciation."""
    dic, G, ta_t = [float(v) for v in y]
    T = float(
        seasonal_temperature(
            t,
            p.T_min,
            p.T_max,
            p.seasonality,
            p.temperature_peak_day,
            p.seasonal_cycle_days,
        )[0]
    )
    light = float(seasonal_light(
        t,
        p.light_seasonality,
        p.light_phase_days,
        p.light_peak_day,
        p.light_sharpness,
        p.light_winter,
        p.light_summer,
        p.seasonal_cycle_days,
    )[0])
    h_mld = float(
        seasonal_mld(
            t,
            p.mld_seasonality,
            p.mld_winter,
            p.mld_summer,
            p.mld_peak_day,
            p.seasonal_cycle_days,
        )[0]
    )
    dhdt = float(
        seasonal_mld_tendency(
            t,
            p.mld_seasonality,
            p.mld_winter,
            p.mld_summer,
            p.mld_peak_day,
            p.seasonal_cycle_days,
        )[0]
    )
    we = max(dhdt, 0.0)

    co2, _, _, pH = speciate_from_dic_ta(dic, ta_t, T, p.S, pH_guess=pH_guess)

    K0 = float(solubility_co2_weiss74(T, p.S))
    co2_eq = K0 * p.pCO2_air
    _, dDIC_flux = co2_flux_and_tendency(co2, co2_eq, p.U10, T, h_mld)

    dDIC_mld = -(dhdt / h_mld) * dic
    dDIC_ent = (we / h_mld) * (p.DIC_deep - dic)

    dTA_mld = -(dhdt / h_mld) * ta_t
    dTA_ent = (we / h_mld) * (p.TA_deep - ta_t)

    dG_mld = -(dhdt / h_mld) * G
    dG_ent = (we / h_mld) * (p.G_deep - G)

    if p.biology_on:
        dDIC_bio, dG_dt, _, _ = bio_tendencies(
            DIC=dic,
            G=G,
            light=light,
            Pmax=p.Pmax,
            Km_C=p.Km_C,
            tau_remin_days=p.tau_remin_days,
            light_half_saturation=p.light_half_saturation,
        )
    else:
        dDIC_bio = 0.0
        dG_dt = 0.0

    return [
        dDIC_flux + dDIC_bio + dDIC_mld + dDIC_ent,
        dG_dt + dG_mld + dG_ent,
        dTA_mld + dTA_ent,
    ], pH


def initialize_state(p: Params):
    """Initialize DIC, glucose, and TA."""
    T0 = float(
        seasonal_temperature(
            0.0,
            p.T_min,
            p.T_max,
            p.seasonality,
            p.temperature_peak_day,
            p.seasonal_cycle_days,
        )[0]
    )
    ta = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)

    dic0 = initialize_dic_from_pco2(p.pCO2_sw_init, ta, T0, p.S)
    return [float(dic0), float(p.G0), float(ta)]


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

    T = seasonal_temperature(
        sol.t,
        p.T_min,
        p.T_max,
        p.seasonality,
        p.temperature_peak_day,
        p.seasonal_cycle_days,
    )
    light = seasonal_light(
        sol.t,
        p.light_seasonality,
        p.light_phase_days,
        p.light_peak_day,
        p.light_sharpness,
        p.light_winter,
        p.light_summer,
        p.seasonal_cycle_days,
    )
    mld = seasonal_mld(
        sol.t,
        p.mld_seasonality,
        p.mld_winter,
        p.mld_summer,
        p.mld_peak_day,
        p.seasonal_cycle_days,
    )
    dic, G, ta = sol.y
    K0 = solubility_co2_weiss74(T, p.S)
    co2 = np.full_like(dic, np.nan)
    hco3 = np.full_like(dic, np.nan)
    co3 = np.full_like(dic, np.nan)
    pH = np.full_like(dic, np.nan)
    pH_guess = ph_cache["value"]
    for i, Ti in enumerate(T):
        co2[i], hco3[i], co3[i], pH[i] = speciate_from_dic_ta(
            dic[i], ta[i], float(Ti), p.S, pH_guess=pH_guess
        )
        pH_guess = pH[i]

    pco2_sw = co2 / K0
    co2_eq = K0 * p.pCO2_air
    F = np.array(
        [
            co2_flux_and_tendency(co2_i, co2_eq_i, p.U10, Ti, h_i)[0]
            for co2_i, co2_eq_i, Ti, h_i in zip(co2, co2_eq, T, mld)
        ]
    )

    if p.biology_on:
        P_glucose = np.array(
            [
                glucose_production_rate(
                    dic_i,
                    li,
                    p.Pmax,
                    p.Km_C,
                    p.light_half_saturation,
                )
                for dic_i, li in zip(dic, light)
            ]
        )
        R_glucose = remin_rate_from_tau_days(p.tau_remin_days) * G
    else:
        P_glucose = np.zeros_like(dic)
        R_glucose = np.zeros_like(dic)

    glucose_c_flux = 6.0 * P_glucose * mld
    remin_c_flux = 6.0 * R_glucose * mld

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
        "Light": light,
        "MLD": mld,
        "CO2": co2,
        "HCO3": hco3,
        "CO3": co3,
        "DIC": dic,
        "G": G,
        "TA": ta,
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

    t = out_on["t_s"]
    sec_per_year = 365.0 * 24.0 * 3600.0
    mask_last_year = t >= (t[-1] - sec_per_year)

    uptake_on_last = np.trapezoid((-out_on["F"])[mask_last_year], t[mask_last_year])
    uptake_off_last = np.trapezoid((-out_off["F"])[mask_last_year], t[mask_last_year])

    pco2_on_last = out_on["pCO2_sw"][mask_last_year]
    pco2_off_last = out_off["pCO2_sw"][mask_last_year]
    delta_uptake_last = uptake_on_last - uptake_off_last

    dic_on = out_on["DIC"]
    doc_on = out_on["DOC"]
    d_doc_dt = np.gradient(doc_on, t)
    drawdown_amount_last_year = np.trapezoid(d_doc_dt[mask_last_year], t[mask_last_year])

    dic_mean_last = np.mean(dic_on[mask_last_year])
    doc_mean_last = np.mean(doc_on[mask_last_year])
    total_c_mean_last = np.mean((dic_on + doc_on)[mask_last_year])

    print("=== Last-year (final 365 days) ===")
    print(f"pCO2_sw mean (biology OFF): {np.mean(pco2_off_last):.3f} uatm")
    print(f"pCO2_sw mean (biology ON) : {np.mean(pco2_on_last):.3f} uatm")
    print(f"Atmospheric pCO2          : {params_on.pCO2_air:.3f} uatm")
    print("")
    flux_abs_on = np.abs(out_on["F"])
    flux_abs_off = np.abs(out_off["F"])
    print(
        "|F| diagnostics (ON) [mol C m^-2 s^-1]: "
        f"min={np.min(flux_abs_on):.3e}, mean={np.mean(flux_abs_on):.3e}, max={np.max(flux_abs_on):.3e}"
    )
    print(
        "|F| diagnostics (OFF) [mol C m^-2 s^-1]: "
        f"min={np.min(flux_abs_off):.3e}, mean={np.mean(flux_abs_off):.3e}, max={np.max(flux_abs_off):.3e}"
    )
    print("")
    print(f"Air->sea uptake (biology OFF): {uptake_off_last:.6e} mol C m^-2 (integrated over last year)")
    print(f"Air->sea uptake (biology ON) : {uptake_on_last:.6e} mol C m^-2 (integrated over last year)")
    print(f"Biology effect (ON - OFF)   : {delta_uptake_last:.6e} mol C m^-2 (integrated over last year)")
    print(f"Uptake enhancement factor (ON / OFF): {uptake_on_last / uptake_off_last:.3f}")
    print("")
    print(f"ΔDOC over last year (ON): {drawdown_amount_last_year:.6e} mol C m^-3")
    print(f"Mean DIC (ON)   : {dic_mean_last:.6e} mol C m^-3")
    print(f"Mean DOC (ON)   : {doc_mean_last:.6e} mol C m^-3")
    print(f"Mean Total (ON) : {total_c_mean_last:.6e} mol C m^-3")
    print("")
    print("DIC_deep:", params_on.DIC_deep, "mol m^-3")
    print("TA_deep :", params_on.TA_deep,  "mol m^-3")
    print("G_deep  :", params_on.G_deep,   "mol glucose m^-3")

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
