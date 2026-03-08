"""
Air–Sea CO2 Exchange Test Model with DIC-Only Carbonate Chemistry
-----------------------------------------------------------------

Two modes:
1) Carbonate solver OFF:
   - State variable: DIC_simple only (mol m^-3)
   - Treated as directly exchangeable CO2*-like carbon
   - pCO2_sw = DIC_simple / K0

2) Carbonate solver ON:
   - State variable: DIC only (mol m^-3)
   - Carbonate species are diagnosed from equilibrium:
         CO2*, HCO3-, CO3-- = f(DIC, TA, T, S)
   - Air-sea gas exchange acts only on diagnosed CO2*
   - TA is held constant (or salinity-scaled if desired)

This version is consistent with the project documentation:
the model evolves DIC, while carbonate speciation is solved
diagnostically at each timestep.
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
    Returns K0 in mol m^-3 µatm^-1, so that:
        CO2* (mol m^-3) = K0 * pCO2 (µatm)
    """
    Tk = np.asarray(T) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578

    lnK0 = (
        A1
        + A2 * (100.0 / Tk)
        + A3 * np.log(Tk / 100.0)
        + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
    )

    K0_mol_kg_atm = np.exp(lnK0)
    return K0_mol_kg_atm * rho * 1e-6  # mol m^-3 µatm^-1


def k_wanninkhof(U10, T):
    """Wanninkhof (1992) gas transfer velocity in m s^-1."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660.0) ** (-0.5)
    return k_cm_hr / 100.0 / 3600.0


def seasonal_t_seconds(t, T_min=2.0, T_max=20.0, seasonality=True):
    """Seasonal temperature cycle, returns temperature in °C."""
    t = np.atleast_1d(t).astype(float)
    P = 365.0 * 24.0 * 3600.0
    T_mean = 0.5 * (T_min + T_max)

    if not seasonality:
        return np.full_like(t, T_mean)

    A = 0.5 * (T_max - T_min)
    return T_mean + A * np.sin(2.0 * np.pi * t / P)


# ============================================================
# 2) Carbonate + borate equilibrium constants
# ============================================================
def K1_K2(T, S):
    """Mehrbach-style carbonic acid dissociation constants."""
    Tk = T + 273.15
    pK1 = 3633.86 / Tk - 61.2172 + 9.6777 * np.log(Tk) - 0.011555 * S + 0.0001152 * S**2
    pK2 = 471.78 / Tk + 25.9290 - 3.16967 * np.log(Tk) - 0.01781 * S + 0.0001122 * S**2
    return 10.0 ** (-pK1), 10.0 ** (-pK2)


def Kw_const(T):
    """Approximate water dissociation constant."""
    Tk = T + 273.15
    return np.exp(148.96502 - 13847.26 / Tk - 23.6521 * np.log(Tk))


def Kb_dickson(T, S):
    """Dickson (1990) boric acid dissociation constant."""
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
    """Uppström (1974): total boron in mol m^-3."""
    tb_mol_kg = 0.0004157 * (S / 35.0)
    return tb_mol_kg * rho


def ta_from_salinity(S, ta0, S0=35.0):
    """Scale alkalinity with salinity."""
    return ta0 * (S / S0)


# ============================================================
# 3) Robust root bracketing
# ============================================================
def bracket_root(fun, x_min=3.0, x_max=11.0, n=1200):
    """Find a sign-change bracket for Brent root finding."""
    xs = np.linspace(x_min, x_max, n)
    fs = np.array([fun(x) for x in xs], dtype=float)

    mask = np.isfinite(fs)
    xs, fs = xs[mask], fs[mask]

    if len(xs) < 2:
        raise ValueError("Residual is non-finite across search range.")

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
# 4) Carbonate speciation solver: (DIC, TA) -> species + pH
# ============================================================
def speciate_from_dic_ta(dic, ta, T, S):
    """
    Given DIC and TA (mol m^-3), solve for pH and return:
        CO2*, HCO3-, CO3--, pH
    """
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)
    Kb = Kb_dickson(T, S)
    TB = total_boron(S)

    def residual(pH):
        H = 10.0 ** (-pH)

        co2 = dic / (1.0 + K1 / H + K1 * K2 / H**2)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2

        oh = Kw / H
        boh4 = TB * Kb / (Kb + H)

        ta_calc = hco3 + 2.0 * co3 + boh4 + oh - H
        return ta_calc - ta

    a, b = bracket_root(residual, 3.0, 11.0, 1200)
    fa, fb = residual(a), residual(b)

    if np.sign(fa) * np.sign(fb) > 0:
        raise ValueError(
            f"Could not bracket pH root. residual({a:.3f})={fa:.3e}, "
            f"residual({b:.3f})={fb:.3e}, DIC={dic:.3e}, TA={ta:.3e}, T={T:.2f}, S={S}"
        )

    pH = brentq(residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)
    H = 10.0 ** (-pH)

    co2 = dic / (1.0 + K1 / H + K1 * K2 / H**2)
    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2

    return co2, hco3, co3, pH


# ============================================================
# 5) Initialization from pCO2 + TA
# ============================================================
def initialize_dic_from_pco2(pco2, ta, T, S):
    """
    Given pCO2 (µatm) and TA, find equilibrium DIC (mol m^-3).
    """
    K0 = solubility_co2(T, S)
    K1, K2 = K1_K2(T, S)
    Kw = Kw_const(T)
    Kb = Kb_dickson(T, S)
    TB = total_boron(S)

    co2 = K0 * pco2

    def residual(pH):
        H = 10.0 ** (-pH)
        hco3 = co2 * K1 / H
        co3 = co2 * K1 * K2 / H**2
        oh = Kw / H
        boh4 = TB * Kb / (Kb + H)
        ta_calc = hco3 + 2.0 * co3 + boh4 + oh - H
        return ta_calc - ta

    a, b = bracket_root(residual, 3.0, 11.0, 1200)
    fa, fb = residual(a), residual(b)

    if np.sign(fa) * np.sign(fb) > 0:
        raise ValueError(
            f"Could not bracket initial pH root. residual({a:.3f})={fa:.3e}, "
            f"residual({b:.3f})={fb:.3e}, pCO2={pco2}, TA={ta:.3e}, T={T:.2f}, S={S}"
        )

    pH = brentq(residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=200)
    H = 10.0 ** (-pH)

    hco3 = co2 * K1 / H
    co3 = co2 * K1 * K2 / H**2

    return co2 + hco3 + co3


# ============================================================
# 6) RHS: carbonate OFF vs ON
# ============================================================
def rhs_off(t, y, S, U10, pco2_air, h, temp_params):
    """
    Simple case:
    y = [DIC_simple]

    DIC_simple is treated like directly exchangeable CO2*-like carbon.
    """
    dic_simple = y[0]

    T = float(seasonal_t_seconds(t, **temp_params)[0])
    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)

    co2_eq = K0 * pco2_air
    F_ex = k * (dic_simple - co2_eq)  # mol m^-2 s^-1

    ddic_dt = -F_ex / h
    return [ddic_dt]


def rhs_on(t, y, S, U10, pco2_air, h, ta0, S0, temp_params):
    """
    Carbonate chemistry case:
    y = [DIC]

    Carbonate species are diagnosed from equilibrium, and only CO2*
    participates in air-sea exchange.
    """
    dic = y[0]

    T = float(seasonal_t_seconds(t, **temp_params)[0])
    ta_t = ta_from_salinity(S, ta0, S0)

    co2, _, _, _ = speciate_from_dic_ta(dic, ta_t, T, S)

    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    co2_eq = K0 * pco2_air

    F_ex = k * (co2 - co2_eq)  # mol m^-2 s^-1
    ddic_dt = -F_ex / h

    return [ddic_dt]


# ============================================================
# 7) Diagnostics
# ============================================================
def diagnose_from_dic(t, dic, S, U10, pco2_air, ta0, S0, temp_params, carbonate_on):
    """
    Diagnose time series of carbonate variables and fluxes from DIC.
    """
    T_series = seasonal_t_seconds(t, **temp_params)
    K0_series = solubility_co2(T_series, S)
    k_series = np.array([k_wanninkhof(U10, Ti) for Ti in T_series])

    n = len(t)
    co2 = np.zeros(n)
    hco3 = np.zeros(n)
    co3 = np.zeros(n)
    pH = np.full(n, np.nan)

    if carbonate_on:
        ta_t = ta_from_salinity(S, ta0, S0)
        for i, Ti in enumerate(T_series):
            co2[i], hco3[i], co3[i], pH[i] = speciate_from_dic_ta(dic[i], ta_t, float(Ti), S)
    else:
        co2[:] = dic
        hco3[:] = 0.0
        co3[:] = 0.0

    pco2_sw = co2 / K0_series
    co2_eq = K0_series * pco2_air
    F_ex = k_series * (co2 - co2_eq)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_co2 = 100.0 * co2 / dic
        frac_hco3 = 100.0 * hco3 / dic
        frac_co3 = 100.0 * co3 / dic

    return {
        "T": T_series,
        "CO2": co2,
        "HCO3": hco3,
        "CO3": co3,
        "pH": pH,
        "pCO2_sw": pco2_sw,
        "F_ex": F_ex,
        "frac_CO2": frac_co2,
        "frac_HCO3": frac_hco3,
        "frac_CO3": frac_co3,
    }


# ============================================================
# 8) Model runner
# ============================================================
def run_model(carbonate_on=True):
    S = 30.0
    U10 = 6.0
    h = 50.0
    pco2_air = 420.0

    ta0 = 2300e-6 * RHO
    S0 = 35.0
    ta = ta_from_salinity(S, ta0, S0)

    temp_params = {"T_min": 2.0, "T_max": 20.0, "seasonality": True}

    years = 5
    t_end = years * 365 * 24 * 3600
    t_eval = np.arange(0.0, t_end + 1.0, 24.0 * 3600.0)
    t_span = (t_eval[0], t_eval[-1])

    T0 = float(seasonal_t_seconds(0.0, **temp_params)[0])

    if carbonate_on:
        dic0 = initialize_dic_from_pco2(300.0, ta, T0, S)
        sol = solve_ivp(
            rhs_on,
            t_span,
            [dic0],
            t_eval=t_eval,
            args=(S, U10, pco2_air, h, ta0, S0, temp_params),
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=24 * 3600,
        )
    else:
        # In the simple case, initialize directly from CO2 equilibrium with 300 µatm
        dic0 = solubility_co2(T0, S) * 300.0
        sol = solve_ivp(
            rhs_off,
            t_span,
            [dic0],
            t_eval=t_eval,
            args=(S, U10, pco2_air, h, temp_params),
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=24 * 3600,
        )

    t = sol.t
    dic = sol.y[0]

    diag = diagnose_from_dic(
        t=t,
        dic=dic,
        S=S,
        U10=U10,
        pco2_air=pco2_air,
        ta0=ta0,
        S0=S0,
        temp_params=temp_params,
        carbonate_on=carbonate_on,
    )

    return {
        "carbonate_on": carbonate_on,
        "t": t,
        "time_days": t / (24.0 * 3600.0),
        "DIC": dic,
        **diag,
    }


def slice_for_plot(out, mask):
    """Slice all time-dependent arrays for plotting."""
    return {
        k: (v[mask] if isinstance(v, np.ndarray) and v.shape == out["t"].shape else v)
        for k, v in out.items()
    }


# ============================================================
# 9) Main script
# ============================================================
def main():
    plot_last_year = False

    out_on = run_model(carbonate_on=True)
    out_off = run_model(carbonate_on=False)

    t = out_on["t"]
    sec_per_year = 365 * 24 * 3600
    mask_last_year = t >= (t[-1] - sec_per_year)

    uptake_on_last = np.trapezoid((-out_on["F_ex"])[mask_last_year], t[mask_last_year])
    uptake_off_last = np.trapezoid((-out_off["F_ex"])[mask_last_year], t[mask_last_year])
    delta_uptake_last = uptake_on_last - uptake_off_last
    factor_uptake = uptake_on_last / uptake_off_last if uptake_off_last != 0 else np.nan

    dic_on_mean_last = np.mean(out_on["DIC"][mask_last_year])
    dic_off_mean_last = np.mean(out_off["DIC"][mask_last_year])
    delta_dic_mean_last = dic_on_mean_last - dic_off_mean_last
    factor_dic_mean = dic_on_mean_last / dic_off_mean_last if dic_off_mean_last != 0 else np.nan

    print("=== Last-year (final 365 days) ===")
    print(f"Air→sea uptake (carbonate OFF): {uptake_off_last:.6e} mol C m^-2 yr^-1")
    print(f"Air→sea uptake (carbonate ON) : {uptake_on_last:.6e} mol C m^-2 yr^-1")
    print(f"Carbonate effect (ON - OFF)   : {delta_uptake_last:.6e} mol C m^-2 yr^-1")
    print(f"Uptake factor (ON / OFF)      : {factor_uptake:.3f}")
    print("")
    print(f"Mean DIC (carbonate OFF)      : {dic_off_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC (carbonate ON)       : {dic_on_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC diff (ON - OFF)      : {delta_dic_mean_last:.6e} mol C m^-3")
    print(f"Mean DIC factor (ON / OFF)    : {factor_dic_mean:.3f}")

    if plot_last_year:
        onp = slice_for_plot(out_on, mask_last_year)
        offp = slice_for_plot(out_off, mask_last_year)
    else:
        onp, offp = out_on, out_off

    td = onp["time_days"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(td, offp["DIC"], label="DIC (carbonate OFF; directly exchangeable)")
    axes[0].plot(td, onp["DIC"], label="DIC (carbonate ON; speciation diagnosed)")
    axes[0].set_ylabel("DIC (mol C m$^{-3}$)")
    axes[0].set_title("DIC comparison")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(td, onp["pH"], label="pH (carbonate ON)")
    axes[1].set_ylabel("pH")
    axes[1].set_title("Diagnosed pH")
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
    axes[2].set_title("Carbonate fractions (carbonate ON)")
    axes[2].grid(True, which="both")
    axes[2].legend()

    axes[3].plot(td, offp["F_ex"], label="Air–sea CO2 flux (OFF)")
    axes[3].plot(td, onp["F_ex"], label="Air–sea CO2 flux (ON)")
    axes[3].axhline(0.0, linestyle="--")
    axes[3].set_ylabel("F_ex (mol C m$^{-2}$ s$^{-1}$)")
    axes[3].set_xlabel("Time (days)")
    axes[3].set_title("Air–sea CO2 flux")
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
