"""
Air–Sea CO2 Exchange with Simple Biology (EST)
-----------------------------------------------

This test model extends a 1-box mixed-layer air–sea CO2 exchange setup with a
very simple biological conversion between dissolved inorganic carbon (DIC) and
an idealized organic carbon pool represented as glucose.

Model features:
- Weiss (1974) CO2 solubility
- Wanninkhof (1992) gas transfer velocity
- Optional seasonal temperature forcing
- Biology toggle:
  * OFF: physical air–sea exchange only
  * ON : DIC <-> glucose dynamics with Q10-scaled production and remineralization

The script prints last-year diagnostics and plots either the full simulation or
just the final year.
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1) Physical parameterizations
# ============================================================
def solubility_co2(T, S):
    """Weiss (1974) CO2 solubility K0 (mol m^-3 µatm^-1). Scalar or array T."""
    Tk = np.asarray(T) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    lnK0 = (
        A1
        + A2 * (100 / Tk)
        + A3 * np.log(Tk / 100)
        + S * (B1 + B2 * (Tk / 100) + B3 * (Tk / 100) ** 2)
    )
    return np.exp(lnK0)


def k_wanninkhof(U10, T):
    """Wanninkhof (1992) gas transfer velocity k (m/s)."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660) ** (-0.5)
    return k_cm_hr / 100 / 3600  # cm/hr -> m/s


def seasonal_t_seconds(t, T_min=2, T_max=20, seasonality=True):
    """Seasonal temperature cycle, returns array (°C)."""
    t = np.atleast_1d(t).astype(float)
    P = 365 * 24 * 3600
    T_mean = 0.5 * (T_min + T_max)
    if not seasonality:
        return np.full_like(t, T_mean, dtype=float)
    A = 0.5 * (T_max - T_min)
    return T_mean + A * np.sin(2 * np.pi * t / P)


# ============================================================
# 2) Simple biology: CO2 <-> glucose
# ============================================================
def glucose_production_rate(DIC, T, Pmax, Km_C, Tref=20.0, Q10=2.0):
    """P in mol glucose m^-3 s^-1, temp-dependent (Q10) and CO2-limited."""
    fT = Q10 ** ((T - Tref) / 10.0)
    dic_pos = max(float(DIC), 0.0)
    lim = dic_pos / (Km_C + dic_pos)
    return Pmax * fT * lim


# ============================================================
# 3) RHS: OFF (1 state) and ON (2 states)
# ============================================================
def rhs_off(t, y, S, U10, pCO2_air, h, temp_params):
    """Physical-only model RHS for y = [DIC]."""
    dic = y[0]
    T = seasonal_t_seconds(t, **temp_params)[0]
    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    dic_eq = K0 * pCO2_air
    F = k * (dic - dic_eq)  # mol C m^-2 s^-1 (positive ocean->atm)
    d_dic_dt = -F / h
    return [d_dic_dt]


def rhs_on(t, y, S, U10, pCO2_air, h, temp_params, uptake_params, remin_rate):
    """Physics + simple biology RHS for y = [DIC, G]."""
    dic, G = y
    T = seasonal_t_seconds(t, **temp_params)[0]

    # air-sea
    k = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    dic_eq = K0 * pCO2_air
    F = k * (dic - dic_eq)
    d_dic_phys = -F / h

    # biology
    P = glucose_production_rate(dic, T, **uptake_params)
    R = remin_rate * G
    dGdt = P - R
    d_dic_bio = -6.0 * P + 6.0 * R

    return [d_dic_phys + d_dic_bio, dGdt]


# ============================================================
# 4) Model runner (toggle ON/OFF)
# ============================================================
def run_model(biology_on=True):
    # Physical
    S = 30
    U10 = 6
    h = 50
    pCO2_air = 420
    pCO2_w_init = 300

    # Temperature forcing
    temp_params = {"T_min": 2, "T_max": 20, "seasonality": True}

    # Biology params (toy)
    uptake_params = {"Pmax": 5.0e-7, "Km_C": 1.0e-2, "Tref": 15.0, "Q10": 2.0}
    tau_remin_days = 60.0
    remin_rate = 0.0 if tau_remin_days <= 0 else 1.0 / (tau_remin_days * 24 * 3600)

    # Time
    years = 5
    t_end = years * 365 * 24 * 3600
    t_eval = np.arange(0, t_end + 1, 24 * 3600)
    t_span = (t_eval[0], t_eval[-1])

    # Init DIC from initial pCO2
    T0 = seasonal_t_seconds(0.0, **temp_params)[0]
    K0_0 = solubility_co2(T0, S)
    dic0 = float(K0_0 * pCO2_w_init)

    if not biology_on:
        sol = solve_ivp(
            rhs_off,
            t_span,
            [dic0],
            t_eval=t_eval,
            args=(S, U10, pCO2_air, h, temp_params),
            rtol=1e-6,
            atol=1e-9,
        )
        t = sol.t
        dic = sol.y[0]
        G = np.zeros_like(dic)
    else:
        sol = solve_ivp(
            rhs_on,
            t_span,
            [dic0, 0.0],
            t_eval=t_eval,
            args=(S, U10, pCO2_air, h, temp_params, uptake_params, remin_rate),
            rtol=1e-6,
            atol=1e-9,
        )
        t = sol.t
        dic = sol.y[0]
        G = sol.y[1]

    # Diagnostics
    T = seasonal_t_seconds(t, **temp_params)
    K0 = solubility_co2(T, S)
    pco2_sw = dic / K0

    k_arr = np.array([k_wanninkhof(U10, Ti) for Ti in T])
    dic_eq = K0 * pCO2_air
    F = k_arr * (dic - dic_eq)  # mol C m^-2 s^-1 (positive ocean->atm)

    doc = 6.0 * G
    total_c = dic + doc

    if biology_on:
        P_arr = np.array(
            [glucose_production_rate(dic[i], T[i], **uptake_params) for i in range(len(dic))]
        )
        R_arr = remin_rate * G
        net_drawdown_rate = 6.0 * (P_arr - R_arr)
    else:
        net_drawdown_rate = np.zeros_like(dic)

    return {
        "biology_on": biology_on,
        "t": t,
        "time_days": t / (24 * 3600),
        "T": T,
        "DIC": dic,
        "DOC": doc,
        "TotalC": total_c,
        "pCO2_sw": pco2_sw,
        "F": F,
        "net_drawdown_rate": net_drawdown_rate,
    }


# ============================================================
# 5) Metrics + plotting controls
# ============================================================
def slice_for_plot(out, mask):
    """Slice all time-dependent arrays for a plotting window."""
    return {
        k: (v[mask] if isinstance(v, np.ndarray) and v.shape == out["t"].shape else v)
        for k, v in out.items()
    }


def main():
    plot_last_year = False

    out_on = run_model(True)
    out_off = run_model(False)

    t = out_on["t"]
    sec_per_year = 365 * 24 * 3600
    mask_last_year = t >= (t[-1] - sec_per_year)

    # Uptake (air->sea) = -F (because F>0 is ocean->air)
    uptake_on_last = np.trapz((-out_on["F"])[mask_last_year], t[mask_last_year])
    uptake_off_last = np.trapz((-out_off["F"])[mask_last_year], t[mask_last_year])
    delta_uptake_last = uptake_on_last - uptake_off_last

    # Bio conversion (DIC->DOC) over last year
    drawdown_amount_last_year = np.trapz(
        out_on["net_drawdown_rate"][mask_last_year], t[mask_last_year]
    )

    # Last-year mean pools (ON)
    dic_mean_last = np.mean(out_on["DIC"][mask_last_year])
    doc_mean_last = np.mean(out_on["DOC"][mask_last_year])
    total_c_mean_last = np.mean(out_on["TotalC"][mask_last_year])

    print("=== Last-year (final 365 days) ===")
    print(f"Air→sea uptake (biology OFF): {uptake_off_last:.6e} mol C m^-2 yr^-1")
    print(f"Air→sea uptake (biology ON) : {uptake_on_last:.6e} mol C m^-2 yr^-1")
    print(f"Biology effect (ON - OFF)   : {delta_uptake_last:.6e} mol C m^-2 yr^-1")
    print(f"Uptake enhancement factor (ON / OFF): {uptake_on_last / uptake_off_last:.3f}")
    print("")
    print(f"Net bio DIC→DOC conversion (ON): {drawdown_amount_last_year:.6e} mol C m^-3 yr^-1")
    print(f"Mean DIC (ON)   : {dic_mean_last:.6e} mol C m^-3")
    print(f"Mean DOC (ON)   : {doc_mean_last:.6e} mol C m^-3")
    print(f"Mean Total (ON) : {total_c_mean_last:.6e} mol C m^-3")

    if plot_last_year:
        onp = slice_for_plot(out_on, mask_last_year)
        offp = slice_for_plot(out_off, mask_last_year)
    else:
        onp, offp = out_on, out_off

    td = onp["time_days"]

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(td, offp["pCO2_sw"], label="pCO₂_sw (biology OFF)")
    axes[0].plot(td, onp["pCO2_sw"], label="pCO₂_sw (biology ON)")
    axes[0].axhline(420, linestyle="--", label="pCO₂_air")
    axes[0].set_ylabel("pCO₂ (µatm)")
    axes[0].set_title("Surface pCO₂ (diagnosed from DIC proxy)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(td, offp["DIC"], label="DIC (biology OFF)")
    axes[1].plot(td, onp["DIC"], label="DIC (biology ON)")
    axes[1].set_ylabel("DIC (mol C m⁻³)")
    axes[1].set_title("Dissolved Inorganic Carbon (DIC)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(td, onp["DOC"], label="DOC (biology ON)")
    axes[2].plot(td, onp["TotalC"], label="Total Carbon = DIC + DOC (biology ON)")
    axes[2].plot(td, offp["TotalC"], linestyle="--", label="Total Carbon (OFF = DIC)")
    axes[2].set_ylabel("Carbon (mol C m⁻³)")
    axes[2].set_title("DOC and Total Carbon")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(td, offp["F"], label="Air–sea CO₂ flux (OFF)")
    axes[3].plot(td, onp["F"], label="Air–sea CO₂ flux (ON)")
    axes[3].axhline(0, linestyle="--")
    axes[3].set_ylabel("F (mol C m⁻² s⁻¹)")
    axes[3].set_title("Air–sea CO₂ flux (positive = ocean → atmosphere)")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(td, onp["T"], label="Temperature")
    axes[4].set_ylabel("T (°C)")
    axes[4].set_xlabel("Time (days)")
    axes[4].set_title("Temperature forcing")
    axes[4].grid(True)
    axes[4].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
