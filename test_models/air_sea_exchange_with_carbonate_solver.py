"""
Air–Sea CO2 Exchange Model with Toggleable Carbonate Speciation.

Two modes:
1) Speciation OFF:
   - State variable: CO2* only (mol m^-3), treated as simple DIC.
   - pCO2_sw = CO2*/K0

2) Speciation ON:
   - State variables: CO2*, HCO3-, CO3-- (mol m^-3)
   - Total DIC = CO2* + HCO3- + CO3--
   - TA held constant
   - Species relax toward equilibrium speciation on a fast timescale tau_spec
   - Air-sea gas exchange acts on CO2* only
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RHO = 1025.0  # kg m^-3


# ============================================================
# 1) Physical parameterizations
# ============================================================
def solubility_co2(T, S, rho=RHO):
    """
    Weiss (1974) CO2 solubility.
    Returns K0 in mol m^-3 µatm^-1, so that CO2*(mol m^-3) = K0 * pCO2(µatm)
    """
    Tk = np.asarray(T) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578

    lnK0 = A1 + A2 * (100 / Tk) + A3 * np.log(Tk / 100) + S * (
        B1 + B2 * (Tk / 100) + B3 * (Tk / 100) ** 2
    )

    K0_mol_kg_atm = np.exp(lnK0)
    return K0_mol_kg_atm * rho * 1e-6


def k_wanninkhof(U10, T):
    """Wanninkhof (1992) gas transfer velocity (m s^-1)."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660) ** (-0.5)
    return k_cm_hr / 100 / 3600


def seasonal_t_seconds(t, T_min=2, T_max=20, seasonality=True):
    """Seasonal temperature cycle, returns array (°C)."""
    t = np.atleast_1d(t).astype(float)
    P = 365 * 24 * 3600
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return np.full_like(t, T_mean)
    A = 0.5 * (T_max - T_min)
    return T_mean + A * np.sin(2 * np.pi * t / P)


# ============================================================
# 2) Carbonate + borate equilibrium constants
# ============================================================
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


def total_boron(S, rho=RHO):
    """Uppström (1974): total boron converted to mol m^-3."""
    tb_mol_kg = 0.0004157 * (S / 35.0)
    return tb_mol_kg * rho


def ta_from_salinity(S, ta0, S0=35.0):
    """Scale alkalinity with salinity via dilution/concentration."""
    return ta0 * (S / S0)


# ============================================================
# 3) Robust bracketing for brentq
# ============================================================
def bracket_root(fun, x_min=3.0, x_max=11.0, n=1200):
    """Find a robust sign-change bracket or a small bracket near minimum residual."""
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
    a = max(x_min, xs[j] - 0.3)
    b = min(x_max, xs[j] + 0.3)
    return a, b


# ============================================================
# 4) Speciation solver: (DIC, TA) -> (CO2*, HCO3-, CO3--, pH)
# ============================================================
def speciate_from_dic_ta(dic, ta, T, S):
    """Given DIC and TA (both mol m^-3), solve pH and return carbonate species."""
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)
    Kb = Kb_dickson(T, S)
    TB = total_boron(S)

    def residual(pH):
        H = 10 ** (-pH)
        co2 = dic / (1 + K1 / H + K1 * K2 / H**2)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2
        oh = Kw / H
        boh4 = TB * Kb / (Kb + H)
        ta_calc = hco3 + 2 * co3 + boh4 + oh - H
        return ta_calc - ta

    a, b = bracket_root(residual, 3.0, 11.0, 1200)
    fa, fb = residual(a), residual(b)
    if np.sign(fa) * np.sign(fb) > 0:
        raise ValueError(
            f"Could not bracket pH root. residual({a:.3f})={fa:.3e}, residual({b:.3f})={fb:.3e} "
            f"(DIC={dic:.3e}, TA={ta:.3e}, T={T:.2f}, S={S})"
        )

    pH = brentq(residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)
    H = 10 ** (-pH)

    co2 = dic / (1 + K1 / H + K1 * K2 / H**2)
    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2
    return co2, hco3, co3, pH


# ============================================================
# 5) Initialization from pCO2 + TA
# ============================================================
def initialize_dic_from_pco2(pco2, ta, T, S):
    """Given pCO2 (µatm) and TA, find DIC consistent with equilibrium speciation."""
    K0 = solubility_co2(T, S)
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)
    Kb = Kb_dickson(T, S)
    TB = total_boron(S)

    co2 = K0 * pco2

    def residual(pH):
        H = 10 ** (-pH)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2
        oh = Kw / H
        boh4 = TB * Kb / (Kb + H)
        ta_calc = hco3 + 2 * co3 + boh4 + oh - H
        return ta_calc - ta

    a, b = bracket_root(residual, 3.0, 11.0, 1200)
    fa, fb = residual(a), residual(b)
    if np.sign(fa) * np.sign(fb) > 0:
        raise ValueError(
            f"Could not bracket initial pH root. residual({a:.3f})={fa:.3e}, residual({b:.3f})={fb:.3e} "
            f"(pCO2={pco2}, TA={ta:.3e}, T={T:.2f}, S={S})"
        )

    pH = brentq(residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)
    H = 10 ** (-pH)
    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2
    return co2 + hco3 + co3


# ============================================================
# 6) RHS: speciation OFF vs ON
# ============================================================
def rhs_off(t, y, S, U10, pco2_air, h, temp_params):
    """RHS for speciation-off case, y=[CO2*]."""
    co2 = y[0]
    T = float(seasonal_t_seconds(t, **temp_params)[0])
    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    co2_eq = K0 * pco2_air
    F = k * (co2 - co2_eq)
    return [-F / h]


def rhs_on(t, y, S, U10, pco2_air, h, ta0, S0, temp_params, tau_spec):
    """RHS for speciation-on case, y=[CO2*, HCO3-, CO3--]."""
    co2, hco3, co3 = y
    T = float(seasonal_t_seconds(t, **temp_params)[0])

    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    co2_eq = K0 * pco2_air
    F = k * (co2 - co2_eq)
    dco2_flux = -F / h

    dic = co2 + hco3 + co3
    ta_t = ta_from_salinity(S, ta0, S0)
    co2_tgt, hco3_tgt, co3_tgt, _ = speciate_from_dic_ta(dic, ta_t, T, S)

    dco2_rel = (co2_tgt - co2) / tau_spec
    dhco3_rel = (hco3_tgt - hco3) / tau_spec
    dco3_rel = (co3_tgt - co3) / tau_spec
    return [dco2_flux + dco2_rel, dhco3_rel, dco3_rel]


# ============================================================
# 7) Model runner
# ============================================================
def run_model(speciation_on=True):
    S = 30
    U10 = 6
    h = 50
    pco2_air = 420

    ta0 = 2300e-6 * RHO
    S0 = 35.0
    ta = ta_from_salinity(S, ta0, S0)

    temp_params = {"T_min": 2, "T_max": 20, "seasonality": True}
    tau_spec = 2 * 3600.0

    years = 5
    t_end = years * 365 * 24 * 3600
    t_eval = np.arange(0, t_end + 1, 24 * 3600)
    t_span = (t_eval[0], t_eval[-1])

    T0 = float(seasonal_t_seconds(0.0, **temp_params)[0])
    dic0 = initialize_dic_from_pco2(300, ta, T0, S)

    if not speciation_on:
        K0_0 = solubility_co2(T0, S)
        co2_0 = float(K0_0 * 300.0)
        sol = solve_ivp(
            rhs_off,
            t_span,
            [co2_0],
            t_eval=t_eval,
            args=(S, U10, pco2_air, h, temp_params),
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=24 * 3600,
        )
        t = sol.t
        co2 = sol.y[0]
        hco3 = np.zeros_like(co2)
        co3 = np.zeros_like(co2)
        pH = np.full_like(co2, np.nan)
    else:
        co2_0, hco3_0, co3_0, _ = speciate_from_dic_ta(dic0, ta, T0, S)
        sol = solve_ivp(
            rhs_on,
            t_span,
            [co2_0, hco3_0, co3_0],
            t_eval=t_eval,
            args=(S, U10, pco2_air, h, ta0, S0, temp_params, tau_spec),
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=24 * 3600,
        )
        t = sol.t
        co2, hco3, co3 = sol.y

        pH = np.zeros_like(co2)
        T_series = seasonal_t_seconds(t, **temp_params)
        for i, Ti in enumerate(T_series):
            ta_t = ta_from_salinity(S, ta0, S0)
            _, _, _, pH[i] = speciate_from_dic_ta(co2[i] + hco3[i] + co3[i], ta_t, float(Ti), S)

    T_series = seasonal_t_seconds(t, **temp_params)
    K0_series = solubility_co2(T_series, S)
    pco2_sw = co2 / K0_series

    k_series = np.array([k_wanninkhof(U10, Ti) for Ti in T_series])
    co2_eq_series = K0_series * pco2_air
    F = k_series * (co2 - co2_eq_series)

    dic = co2 + hco3 + co3

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_co2 = 100.0 * co2 / dic
        frac_hco3 = 100.0 * hco3 / dic
        frac_co3 = 100.0 * co3 / dic

    return {
        "speciation_on": speciation_on,
        "t": t,
        "time_days": t / (24 * 3600),
        "T": T_series,
        "CO2": co2,
        "HCO3": hco3,
        "CO3": co3,
        "DIC": dic,
        "pH": pH,
        "pCO2_sw": pco2_sw,
        "F": F,
        "frac_CO2": frac_co2,
        "frac_HCO3": frac_hco3,
        "frac_CO3": frac_co3,
    }


def slice_for_plot(out, mask):
    """Slice all time-dependent arrays for a plotting window."""
    return {
        k: (v[mask] if isinstance(v, np.ndarray) and v.shape == out["t"].shape else v)
        for k, v in out.items()
    }


def main():
    plot_last_year = False

    out_on = run_model(speciation_on=True)
    out_off = run_model(speciation_on=False)

    t = out_on["t"]
    sec_per_year = 365 * 24 * 3600
    mask_last_year = t >= (t[-1] - sec_per_year)

    uptake_on_last = np.trapezoid((-out_on["F"])[mask_last_year], t[mask_last_year])
    uptake_off_last = np.trapezoid((-out_off["F"])[mask_last_year], t[mask_last_year])
    delta_uptake_last = uptake_on_last - uptake_off_last
    factor_uptake = (uptake_on_last / uptake_off_last) if uptake_off_last != 0 else np.nan

    dic_on_mean_last = np.mean(out_on["DIC"][mask_last_year])
    dic_off_mean_last = np.mean(out_off["DIC"][mask_last_year])
    delta_dic_mean_last = dic_on_mean_last - dic_off_mean_last
    factor_dic_mean = (dic_on_mean_last / dic_off_mean_last) if dic_off_mean_last != 0 else np.nan

    print("=== Last-year (final 365 days) ===")
    print(f"Air→sea uptake (speciation OFF): {uptake_off_last:.6e} mol C m^-2 yr^-1")
    print(f"Air→sea uptake (speciation ON) : {uptake_on_last:.6e} mol C m^-2 yr^-1")
    print(f"Speciation effect (ON - OFF)   : {delta_uptake_last:.6e} mol C m^-2 yr^-1")
    print(f"Uptake factor (ON / OFF)       : {factor_uptake:.3f}")
    print("")
    print(f"Mean DIC (speciation OFF)      : {dic_off_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC (speciation ON)       : {dic_on_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC diff (ON - OFF)       : {delta_dic_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC factor (ON / OFF)     : {factor_dic_mean:.3f}")

    if plot_last_year:
        onp = slice_for_plot(out_on, mask_last_year)
        offp = slice_for_plot(out_off, mask_last_year)
    else:
        onp, offp = out_on, out_off

    td = onp["time_days"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(td, offp["DIC"], label="DIC (speciation OFF; CO2* only)")
    axes[0].plot(td, onp["DIC"], label="DIC (speciation ON; CO2*+HCO3-+CO3--)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("DIC (mol C m$^{-3}$)")
    axes[0].set_title("DIC comparison (speciation ON vs OFF)")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].plot(td, onp["pH"], label="pH (speciation ON)")
    axes[1].set_ylabel("pH")
    axes[1].set_title("pH (speciation ON)")
    axes[1].grid(True)
    axes[1].legend()

    eps = 1e-6
    axes[2].plot(td, np.maximum(onp["frac_CO2"], eps), label="CO2* / DIC (%)")
    axes[2].plot(td, np.maximum(onp["frac_HCO3"], eps), label="HCO3- / DIC (%)")
    axes[2].plot(td, np.maximum(onp["frac_CO3"], eps), label="CO3-- / DIC (%)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-1, 100)
    axes[2].set_yticks([0.1, 1, 10, 100])
    axes[2].set_yticklabels(["0.1%", "1%", "10%", "100%"])
    axes[2].set_ylabel("Percent of DIC")
    axes[2].set_title("Carbonate speciation fractions (log scale)")
    axes[2].grid(True, which="both")
    axes[2].legend()

    axes[3].plot(td, offp["F"], label="Air–sea CO2 flux (OFF)")
    axes[3].plot(td, onp["F"], label="Air–sea CO2 flux (ON)")
    axes[3].axhline(0, linestyle="--")
    axes[3].set_ylabel("F (mol C m$^{-2}$ s$^{-1}$)")
    axes[3].set_xlabel("Time (days)")
    axes[3].set_title("Air–sea CO2 flux")
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
