#!/usr/bin/env python3
"""Main integration script for the modular surface-ocean carbon box model."""

import numpy as np
from scipy.integrate import solve_ivp

from params import Params
from state import State
from modules.carbonate_solver import pco2_from_dic_proxy
from modules.gas_exchange import dic_tendency_from_air_sea
from modules.biology import tendencies as bio_tendencies

def seasonal_temperature(t, T_min, T_max, seasonality=True):
    """Seasonal temperature in degC. t in seconds."""
    t = float(t)
    P = 365.0 * 24.0 * 3600.0
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return T_mean
    A = 0.5 * (T_max - T_min)
    return T_mean + A * np.sin(2.0 * np.pi * t / P)

def rhs(t, y, p: Params):
    """
    Unified RHS for y = [DIC, G].
    If biology is OFF, G is still present but will remain ~constant (G' = 0).
    """
    DIC, G = float(y[0]), float(y[1])

    # Forcing
    T = seasonal_temperature(t, p.T_min, p.T_max, p.seasonality)

    # Carbonate proxy diagnostics (placeholder)
    pCO2_sw, K0 = pco2_from_dic_proxy(DIC, T, p.S)

    # Air-sea exchange
    dDIC_phys, F = dic_tendency_from_air_sea(
        DIC=DIC, T=T, S=p.S, U10=p.U10, pCO2_air=p.pCO2_air, h=p.h, K0=K0
    )

    # Biology (optional)
    if p.biology_on:
        dDIC_bio, dG_dt, Pprod, Rremin = bio_tendencies(
            DIC=DIC, G=G, T=T,
            Pmax=p.Pmax, Km_C=p.Km_C, Tref=p.Tref, Q10=p.Q10,
            tau_remin_days=p.tau_remin_days
        )
    else:
        dDIC_bio, dG_dt = 0.0, 0.0

    dDIC_dt = dDIC_phys + dDIC_bio
    return [dDIC_dt, dG_dt]

def initialize_state(p: Params) -> State:
    """
    Initialize DIC from initial seawater pCO2 (proxy using K0 at t=0).
    """
    T0 = seasonal_temperature(0.0, p.T_min, p.T_max, p.seasonality)
    # Use proxy carbonate relationship: DIC0 = K0(T0,S) * pCO2_sw_init
    pCO2_sw0 = p.pCO2_sw_init
    pCO2_tmp, K0 = pco2_from_dic_proxy(DIC=1.0, T=T0, S=p.S)  # just to get K0; DIC ignored
    # Better: call K0 directly via pco2_from_dic_proxy pattern:
    # pco2_from_dic_proxy returns (DIC/K0, K0) so K0 is correct regardless of DIC value
    DIC0 = K0 * pCO2_sw0

    st = State(DIC=float(DIC0), G=float(p.G0))
    # Fill diagnostics placeholders
    st.pCO2_sw = float(pCO2_sw0)
    st.DOC = 6.0 * st.G
    return st

def run(p: Params):
    """
    Runs the model and returns outputs as a dict of arrays (no plotting).
    """
    st0 = initialize_state(p)

    t_end = p.years * 365.0 * 24.0 * 3600.0
    t_eval = np.arange(0.0, t_end + p.dt_output, p.dt_output)

    y0 = [st0.DIC, st0.G]

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    # Diagnostics time series (placeholders)
    T = np.array([seasonal_temperature(ti, p.T_min, p.T_max, p.seasonality) for ti in sol.t])
    pCO2_sw = np.empty_like(sol.t, dtype=float)
    K0_arr = np.empty_like(sol.t, dtype=float)

    for i in range(len(sol.t)):
        pCO2_sw[i], K0_arr[i] = pco2_from_dic_proxy(sol.y[0, i], T[i], p.S)

    DOC = 6.0 * sol.y[1, :]
    TotalC = sol.y[0, :] + DOC

    return {
        "t_s": sol.t,
        "t_days": sol.t / (24.0 * 3600.0),
        "T_C": T,
        "DIC": sol.y[0, :],
        "G": sol.y[1, :],
        "DOC": DOC,
        "TotalC": TotalC,
        "pCO2_air": p.pCO2_air * np.ones_like(sol.t),
        "pCO2_sw": pCO2_sw,
        "K0": K0_arr,
        "success": sol.success,
        "message": sol.message,
    }

def main():
    p = Params(biology_on=True)   # toggle here
    out = run(p)
    # No plotting/printing by design.
    # Put saving/export here later (np.savez, netcdf, csv, etc.)
    return out

if __name__ == "__main__":
    main()
