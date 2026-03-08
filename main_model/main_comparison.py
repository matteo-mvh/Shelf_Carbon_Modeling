"""Main integration script for the modular surface-ocean carbon box model.

State vector:
    y = [DIC, LDOC, SDOC, RDOC]

Key assumptions in this version:
- Total alkalinity (TA) is fixed from salinity and is therefore constant
  because salinity is held constant.
- Deep-water entrainment is applied when seasonal MLD deepens.
- Carbonate speciation is diagnosed from (DIC, TA, T, S).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
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
from main_model.modules.gas_exchange import co2_flux_and_tendency
from main_model.modules.plotting import (
    save_diagnostics_plot,
    save_entrainment_fitting_plot,
    save_outputs_overview_plot,
)


def open_plot(path: str) -> bool:
    """Open a saved plot file in the default viewer/browser."""
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return False

    if sys.platform.startswith("darwin"):
        completed = subprocess.run(["open", abs_path], check=False)
        return completed.returncode == 0

    if os.name == "nt":
        try:
            os.startfile(abs_path)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False

    for launcher in ("xdg-open", "gio", "gnome-open", "kde-open"):
        launcher_path = shutil.which(launcher)
        if launcher_path:
            completed = subprocess.run([launcher_path, abs_path], check=False)
            if completed.returncode == 0:
                return True

    return webbrowser.open(f"file://{abs_path}", new=2)


def seasonal_temperature(t, T_min, T_max, seasonality=True, peak_day=230.0, cycle_days=365.0):
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




def seasonal_mld(t, seasonality=False, winter_depth=80.0, summer_depth=20.0, peak_day=15.0, cycle_days=365.0, fixed_depth=50.0):
    t = np.atleast_1d(t).astype(float)
    if not seasonality:
        return np.full_like(t, fixed_depth)

    period = cycle_days * 24.0 * 3600.0
    peak_seconds = peak_day * 24.0 * 3600.0
    mld_norm = 0.5 * (1.0 + np.cos(2.0 * np.pi * (t - peak_seconds) / period))
    return summer_depth + (winter_depth - summer_depth) * mld_norm



def seasonal_mld_tendency(
    t,
    seasonality=False,
    winter_depth=80.0,
    summer_depth=20.0,
    peak_day=15.0,
    cycle_days=365.0,
):
    t = np.atleast_1d(t).astype(float)
    if not seasonality:
        return np.zeros_like(t)

    period = cycle_days * 24.0 * 3600.0
    peak_seconds = peak_day * 24.0 * 3600.0
    omega = 2.0 * np.pi / period
    amplitude = 0.5 * (winter_depth - summer_depth)
    return -amplitude * omega * np.sin(omega * (t - peak_seconds))

def rhs(t, y, p: Params, ta_const: float, pH_guess=None):
    """RHS for y=[DIC, LDOC, SDOC, RDOC] with fixed TA and MLD-driven entrainment."""
    dic, ldoc, sdoc, rdoc = [float(v) for v in y]

    dic = max(dic, 1e-12)
    ldoc = max(ldoc, 0.0)
    sdoc = max(sdoc, 0.0)
    rdoc = max(rdoc, 0.0)

    T = float(
        seasonal_temperature(
            t, p.T_min, p.T_max, p.seasonality, p.temperature_peak_day, p.seasonal_cycle_days
        )[0]
    )
    light = float(
        seasonal_light(
            t,
            p.seasonality,
            p.light_phase_days,
            p.light_peak_day,
            p.light_sharpness,
            p.light_winter,
            p.light_summer,
            p.seasonal_cycle_days,
        )[0]
    )
    h_mld = float(
        seasonal_mld(
            t, p.mld_seasonality, p.mld_winter, p.mld_summer, p.mld_peak_day, p.seasonal_cycle_days, p.mld
        )[0]
    )
    dhdt = float(
        seasonal_mld_tendency(
            t, p.mld_seasonality, p.mld_winter, p.mld_summer, p.mld_peak_day, p.seasonal_cycle_days
        )[0]
    )
    co2, _, _, pH = speciate_from_dic_ta(dic, ta_const, T, p.S, pH_guess=pH_guess)
    K0 = float(solubility_co2_weiss74(T, p.S))
    co2_eq = K0 * p.pCO2_air
    _, dDIC_flux = co2_flux_and_tendency(co2, co2_eq, p.U10, T, h_mld)

    dldoc_bio, dsdoc_bio, drdoc_bio, fprod, fremin = bio_tendencies(
        light=light,
        ldoc=ldoc,
        sdoc=sdoc,
        rdoc=rdoc,
        mu=p.mu_bio,
        alpha_l=p.alpha_l,
        alpha_s=p.alpha_s,
        alpha_r=p.alpha_r,
        lambda_l=p.lambda_l,
        lambda_s=p.lambda_s,
        lambda_r=p.lambda_r,
        gamma_l=p.gamma_l,
        gamma_s=p.gamma_s,
        Pmax=p.pp_Pmax,
        K_L=p.pp_K_L,
        n=p.pp_n,
    )
    dDIC_bio = -fprod + fremin

    deepening_rate = max(dhdt, 0.0) if p.mld_seasonality else 0.0
    if h_mld <= 0.0:
        raise ValueError("Mixed-layer depth must remain strictly positive.")

    entrainment_factor = deepening_rate / h_mld
    dDIC_entrain = entrainment_factor * (p.deep_entrainment_dic - dic)
    dLDOC_entrain = entrainment_factor * (p.deep_entrainment_ldoc - ldoc)
    dSDOC_entrain = entrainment_factor * (p.deep_entrainment_sdoc - sdoc)
    dRDOC_entrain = entrainment_factor * (p.deep_entrainment_rdoc - rdoc)

    return [
        dDIC_flux + dDIC_bio + dDIC_entrain,
        dldoc_bio + dLDOC_entrain,
        dsdoc_bio + dSDOC_entrain,
        drdoc_bio + dRDOC_entrain,
    ], pH


def initialize_state(p: Params, ta_const: float):
    T0 = float(
        seasonal_temperature(
            0.0, p.T_min, p.T_max, p.seasonality, p.temperature_peak_day, p.seasonal_cycle_days
        )[0]
    )
    dic0 = initialize_dic_from_pco2(p.pCO2_sw_init, ta_const, T0, p.S)
    return [float(dic0), float(p.LDOC0), float(p.SDOC0), float(p.RDOC0)]


def run(p: Params):
    ta_const = ta_from_salinity(p.S, p.ta0_mol_per_m3, p.S0_ta)

    y0 = initialize_state(p, ta_const)
    t_end = p.years * 365.0 * 24.0 * 3600.0
    n_out = int(round(t_end / p.dt_output)) + 1
    t_eval = np.linspace(0.0, t_end, n_out)

    ph_cache = {"value": 8.1}

    def rhs_cached(t, y):
        dydt, pH = rhs(t, y, p, ta_const=ta_const, pH_guess=ph_cache["value"])
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
        sol.t, p.T_min, p.T_max, p.seasonality, p.temperature_peak_day, p.seasonal_cycle_days
    )
    light = seasonal_light(
        sol.t,
        p.seasonality,
        p.light_phase_days,
        p.light_peak_day,
        p.light_sharpness,
        p.light_winter,
        p.light_summer,
        p.seasonal_cycle_days,
    )
    mld = seasonal_mld(
        sol.t, p.mld_seasonality, p.mld_winter, p.mld_summer, p.mld_peak_day, p.seasonal_cycle_days, p.mld
    )
    dhdt = seasonal_mld_tendency(
        sol.t, p.mld_seasonality, p.mld_winter, p.mld_summer, p.mld_peak_day, p.seasonal_cycle_days
    )

    dic, ldoc, sdoc, rdoc = sol.y

    K0 = solubility_co2_weiss74(T, p.S)
    co2 = np.full_like(dic, np.nan)
    hco3 = np.full_like(dic, np.nan)
    co3 = np.full_like(dic, np.nan)
    pH = np.full_like(dic, np.nan)

    pH_guess = ph_cache["value"]
    for i, Ti in enumerate(T):
        co2[i], hco3[i], co3[i], pH[i] = speciate_from_dic_ta(
            max(dic[i], 1e-12), ta_const, float(Ti), p.S, pH_guess=pH_guess
        )
        pH_guess = pH[i]

    pco2_sw = co2 / K0
    co2_eq = K0 * p.pCO2_air
    F_ex = np.array(
        [
            co2_flux_and_tendency(co2_i, co2_eq_i, p.U10, Ti, h_i)[0]
            for co2_i, co2_eq_i, Ti, h_i in zip(co2, co2_eq, T, mld)
        ]
    )

    pp_light = np.zeros_like(light)
    fremin = np.zeros_like(light)

    for i, (li, ld_i, sd_i, rd_i) in enumerate(zip(light, ldoc, sdoc, rdoc)):
        _, _, _, pp_light[i], fremin[i] = bio_tendencies(
            light=li,
            ldoc=max(ld_i, 0.0),
            sdoc=max(sd_i, 0.0),
            rdoc=max(rd_i, 0.0),
            mu=p.mu_bio,
            alpha_l=p.alpha_l,
            alpha_s=p.alpha_s,
            alpha_r=p.alpha_r,
            lambda_l=p.lambda_l,
            lambda_s=p.lambda_s,
            lambda_r=p.lambda_r,
            gamma_l=p.gamma_l,
            gamma_s=p.gamma_s,
            Pmax=p.pp_Pmax,
            K_L=p.pp_K_L,
            n=p.pp_n,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        entrainment_rate = np.where(mld > 0.0, np.maximum(dhdt, 0.0) / mld, 0.0)

    dDIC_entrain = entrainment_rate * (p.deep_entrainment_dic - dic)
    dLDOC_entrain = entrainment_rate * (p.deep_entrainment_ldoc - ldoc)
    dSDOC_entrain = entrainment_rate * (p.deep_entrainment_sdoc - sdoc)
    dRDOC_entrain = entrainment_rate * (p.deep_entrainment_rdoc - rdoc)

    sinking_rate = np.where(dhdt < 0.0, -dhdt, 0.0)
    F_sink_DIC = sinking_rate * dic

    doc = ldoc + sdoc + rdoc
    ta = np.full_like(dic, ta_const)

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
        "dMLD_dt": dhdt,
        "CO2": co2,
        "HCO3": hco3,
        "CO3": co3,
        "DIC": dic,
        "LDOC": ldoc,
        "SDOC": sdoc,
        "RDOC": rdoc,
        "TA": ta,
        "DOC": doc,
        "fprod": pp_light,
        "fremin": fremin,
        "pH": pH,
        "pCO2_sw": pco2_sw,
        "pCO2_air": np.full_like(pco2_sw, p.pCO2_air, dtype=float),
        "delta_pCO2": pco2_sw - p.pCO2_air,
        "F_ex": F_ex,
        "dDIC_entrain": dDIC_entrain,
        "dLDOC_entrain": dLDOC_entrain,
        "dSDOC_entrain": dSDOC_entrain,
        "dRDOC_entrain": dRDOC_entrain,
        "F_sink_DIC": F_sink_DIC,
        "frac_CO2": frac_co2,
        "frac_HCO3": frac_hco3,
        "frac_CO3": frac_co3,
    }


def main_comparison():
    params = Params()

    out = run(params)

    print("Run success:", out["success"])

    if out["success"]:
        single_run_plot_path = save_diagnostics_plot(
            out,
            output_path="results/main_model_diagnostics.png",
            plot_last_year_only=params.plot_last_year_only,
        )
        print("Saved single-run plot:", single_run_plot_path)

        overview_plot_path = save_outputs_overview_plot(
            out,
            output_path="results/main_model_outputs_overview.png",
            plot_last_year_only=True,
        )
        print("Saved outputs overview plot:", overview_plot_path)

        entrainment_plot_path = save_entrainment_fitting_plot(
            out,
            output_path="results/entrainment_fitting_plot.png",
            plot_last_year_only=True,
        )
        print("Saved entrainment fitting plot:", entrainment_plot_path)

        if open_plot(overview_plot_path):
            print("Opened outputs overview plot:", overview_plot_path)
        elif open_plot(single_run_plot_path):
            print("Opened diagnostics plot:", single_run_plot_path)
    else:
        print("Skipping diagnostics plot because integration failed.")
        print("Message:", out["message"])

    return out


if __name__ == "__main__":
    main_comparison()
