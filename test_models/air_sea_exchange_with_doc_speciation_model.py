"""
Air–Sea CO2 Exchange with DOC Speciation (UNITS FIXED)
-----------------------------------------------------

This is a single-box "CO2* + DOC" model (NOT full carbonate chemistry).

Key fixes vs your original:
1) Weiss (1974) solubility K0 is computed in mol kg^-1 atm^-1, then converted to
   K0_m3_uatm in mol m^-3 µatm^-1 using rho/1e6.
2) State variable renamed from DIC -> CO2 (meaning CO2*; dissolved CO2 + H2CO3).
3) Primary production is capped so it cannot consume more CO2 than available
   over the timestep (prevents runaway when parameters are aggressive).
4) A mild hard cap prevents negative pools from numerical overshoot.

DOC speciation:
- LDOC: low light half-sat kL, fast remin
- SDOC: mid kL, mid remin
- RDOC: high kL, slow remin
Aging: LDOC -> SDOC -> RDOC
Internal biology is carbon-conservative by construction.
Total carbon changes only by air–sea exchange.
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

SEC_PER_DAY = 24 * 3600
SEC_PER_YEAR = 365 * SEC_PER_DAY


# ----------------------------
# Physics
# ----------------------------
def solubility_co2_weiss_m3_uatm(T, S, rho=1025.0):
    """
    Weiss (1974) CO2 solubility K0.

    Returns:
      K0 in mol m^-3 µatm^-1

    Notes:
      Weiss formulation yields K0 in mol kg^-1 atm^-1.
      Convert to mol m^-3 atm^-1 via rho (kg m^-3),
      then to mol m^-3 µatm^-1 via /1e6.
    """
    Tk = np.asarray(T, dtype=float) + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    lnK0 = (
        A1
        + A2 * (100.0 / Tk)
        + A3 * np.log(Tk / 100.0)
        + S * (B1 + B2 * (Tk / 100.0) + B3 * (Tk / 100.0) ** 2)
    )
    K0_mol_kg_atm = np.exp(lnK0)
    return K0_mol_kg_atm * rho / 1e6  # mol m^-3 µatm^-1


def k_wanninkhof_1992(U10, T):
    """Wanninkhof (1992) gas transfer velocity k (m s^-1)."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660.0) ** (-0.5)
    return k_cm_hr / 100.0 / 3600.0  # m/s


# ----------------------------
# Forcing
# ----------------------------
def seasonal_sine(t, vmin, vmax, phase_shift_days=0.0):
    """Seasonal sinusoid in [vmin, vmax]."""
    t = np.atleast_1d(t).astype(float)
    mean = 0.5 * (vmin + vmax)
    amp = 0.5 * (vmax - vmin)
    phase = 2.0 * np.pi * phase_shift_days * SEC_PER_DAY / SEC_PER_YEAR
    return mean + amp * np.sin(2.0 * np.pi * t / SEC_PER_YEAR + phase)


def positive_light(t, i_min=5.0, i_max=220.0):
    """Seasonal irradiance (W m^-2), clipped non-negative."""
    I = seasonal_sine(t, i_min, i_max, phase_shift_days=-80.0)
    return np.maximum(I, 0.0)


def temperature(t, t_min=2.0, t_max=20.0):
    """Seasonal temperature forcing (°C)."""
    return seasonal_sine(t, t_min, t_max, phase_shift_days=-100.0)


# ----------------------------
# Biology helpers
# ----------------------------
def q10_factor(T, Tref=15.0, q10=2.0):
    """Q10 temperature scaling factor."""
    return q10 ** ((np.asarray(T) - Tref) / 10.0)


def light_limitation(I, k_L):
    """Michaelis-Menten-like light limitation term I/(kL + I)."""
    I = np.asarray(I, dtype=float)
    return I / (k_L + I)


# ----------------------------
# Model RHS
# ----------------------------
def rhs(
    t,
    y,
    S,
    U10,
    pco2_air_uatm,
    h,
    pp_max,
    kL,
    frac_pp,
    remin_rates_ref,
    age_l_to_s,
    age_s_to_r,
    km_co2,
    tref_bio,
    q10_pp,
    q10_remin,
    rho,
    dt_pp_cap,  # seconds: production cap horizon (uses solver step estimate)
):
    """
    RHS for y = [CO2, LDOC, SDOC, RDOC].

    CO2 here means CO2* (mol C m^-3), not total DIC.
    """
    co2, ldoc, sdoc, rdoc = (float(y[0]), float(y[1]), float(y[2]), float(y[3]))

    # Hard floor (prevents negative pools from small numerical overshoot)
    co2 = max(co2, 0.0)
    ldoc = max(ldoc, 0.0)
    sdoc = max(sdoc, 0.0)
    rdoc = max(rdoc, 0.0)

    T = float(temperature(t)[0])
    I = float(positive_light(t)[0])

    # --- Air-sea exchange (acts on CO2* only) ---
    k_gas = k_wanninkhof_1992(U10, T)
    K0 = float(solubility_co2_weiss_m3_uatm(T, S, rho=rho))
    co2_eq = K0 * pco2_air_uatm
    F = k_gas * (co2 - co2_eq)  # mol m^-2 s^-1 ; positive: ocean -> air
    d_co2_phys = -F / h

    # --- Potential PP (carbon-limited + light-limited + temperature) ---
    c_lim = co2 / (km_co2 + co2) if co2 > 0 else 0.0
    f_light = np.array([float(light_limitation(I, kL_i)) for kL_i in kL], dtype=float)
    fT_pp = float(q10_factor(T, Tref=tref_bio, q10=q10_pp))
    pp_total_potential = pp_max * c_lim * fT_pp  # mol m^-3 s^-1

    # Route PP to DOC classes (and apply class-specific light sensitivity)
    prod_potential = pp_total_potential * frac_pp * f_light  # mol m^-3 s^-1, 3-vector

    # --- Cap production so it can't consume more CO2 than available in a short horizon ---
    # This avoids pathological blow-ups when pp_max is high relative to the CO2* inventory.
    # We cap based on an effective dt (seconds). Use dt_pp_cap (e.g. 1 day) as a safe horizon.
    prod_sum = float(np.sum(prod_potential))
    if prod_sum > 0.0 and co2 > 0.0:
        # maximum allowable total production rate so that prod_sum * dt <= co2
        max_prod_sum = co2 / max(dt_pp_cap, 1.0)
        if prod_sum > max_prod_sum:
            prod_potential *= (max_prod_sum / prod_sum)

    prod = prod_potential  # final production rates

    # --- Remineralization DOC -> CO2* (temperature forced) ---
    fT_remin = float(q10_factor(T, Tref=tref_bio, q10=q10_remin))
    remin_rates = remin_rates_ref * fT_remin
    remin = np.array(
        [remin_rates[0] * ldoc, remin_rates[1] * sdoc, remin_rates[2] * rdoc],
        dtype=float,
    )

    # --- Aging connections ---
    age_ls = age_l_to_s * ldoc
    age_sr = age_s_to_r * sdoc

    d_ldoc = prod[0] - remin[0] - age_ls
    d_sdoc = prod[1] - remin[1] + age_ls - age_sr
    d_rdoc = prod[2] - remin[2] + age_sr

    # Internal conservation: CO2 decreases by PP sum, increases by remin sum
    d_co2_bio = -float(np.sum(prod)) + float(np.sum(remin))

    return [d_co2_phys + d_co2_bio, d_ldoc, d_sdoc, d_rdoc]


# ----------------------------
# Run + diagnostics
# ----------------------------
def run_model():
    # Physical setup
    S = 30.0
    U10 = 6.0
    h = 50.0            # m mixed layer depth
    rho = 1025.0        # kg m^-3
    pco2_air = 420.0    # µatm
    pco2_sw_init = 400.0  # µatm

    # Biology and DOC speciation setup
    # Tune pp_max to be plausible for CO2* inventory (~0.005–0.02 mol/m^3).
    # 1e-7 mol/m^3/s ~ 0.0086 mol/m^3/day
    pp_max = 5.0e-9  # mol C m^-3 s^-1 (reduced vs your original)
    km_co2 = 1.0e-3  # mol m^-3; set near CO2* scale (not DIC scale)

    # LDOC low kL, SDOC mid kL, RDOC high kL
    kL = np.array([20.0, 80.0, 160.0], dtype=float)

    # Fraction of PP routed to DOC classes
    frac_pp = np.array([0.50, 0.35, 0.15], dtype=float)
    frac_pp = frac_pp / np.sum(frac_pp)

    # LDOC fast remin, SDOC mid, RDOC slow
    tau_days = np.array([20.0, 90.0, 400.0], dtype=float)
    remin_rates_ref = 1.0 / (tau_days * SEC_PER_DAY)  # s^-1

    tref_bio = 15.0
    q10_pp = 2.0
    q10_remin = 1.8

    # Aging rates
    age_l_to_s = 1.0 / (120.0 * SEC_PER_DAY)
    age_s_to_r = 1.0 / (240.0 * SEC_PER_DAY)

    # Production cap horizon (seconds)
    dt_pp_cap = 1.0 * SEC_PER_DAY

    # Time
    years = 10
    t_eval = np.arange(0.0, years * SEC_PER_YEAR + 1.0, SEC_PER_DAY)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    # Initial state: CO2* from K0 * pCO2 (both consistent units)
    T0 = float(temperature(0.0)[0])
    K0_0 = float(solubility_co2_weiss_m3_uatm(T0, S, rho=rho))
    co2_0 = K0_0 * pco2_sw_init  # mol m^-3
    y0 = [co2_0, 0.0, 0.0, 0.0]

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        t_eval=t_eval,
        args=(
            S,
            U10,
            pco2_air,
            h,
            pp_max,
            kL,
            frac_pp,
            remin_rates_ref,
            age_l_to_s,
            age_s_to_r,
            km_co2,
            tref_bio,
            q10_pp,
            q10_remin,
            rho,
            dt_pp_cap,
        ),
        rtol=1e-7,
        atol=1e-12,
    )

    co2, ldoc, sdoc, rdoc = sol.y
    doc_total = ldoc + sdoc + rdoc
    total_c = co2 + doc_total

    T = temperature(sol.t)
    I = positive_light(sol.t)

    K0 = solubility_co2_weiss_m3_uatm(T, S, rho=rho)
    pco2_sw = co2 / K0

    k_gas = np.array([k_wanninkhof_1992(U10, float(Ti)) for Ti in T])
    co2_eq = K0 * pco2_air
    F = k_gas * (co2 - co2_eq)  # mol m^-2 s^-1

    # Carbon conservation diagnostics:
    # d(TotalC)/dt should equal -F/h (biology conservative)
    dtotal_num = np.gradient(total_c, sol.t)
    dtotal_expected = -F / h
    conservation_rmse = np.sqrt(np.mean((dtotal_num - dtotal_expected) ** 2))

    return {
        "time_days": sol.t / SEC_PER_DAY,
        "CO2": co2,
        "LDOC": ldoc,
        "SDOC": sdoc,
        "RDOC": rdoc,
        "DOC_total": doc_total,
        "TotalC": total_c,
        "Temp": T,
        "Light": I,
        "pCO2_sw": pco2_sw,
        "F": F,
        "conservation_rmse": conservation_rmse,
        "K0": K0,
    }


def main():
    out = run_model()

    print("=== DOC speciation diagnostics (final year mean) ===")
    last = out["time_days"] >= (out["time_days"][-1] - 365.0)
    print(f"Mean CO2*     : {np.mean(out['CO2'][last]):.6e} mol C m^-3")
    print(f"Mean DOC total: {np.mean(out['DOC_total'][last]):.6e} mol C m^-3")
    print(f"Mean LDOC     : {np.mean(out['LDOC'][last]):.6e} mol C m^-3")
    print(f"Mean SDOC     : {np.mean(out['SDOC'][last]):.6e} mol C m^-3")
    print(f"Mean RDOC     : {np.mean(out['RDOC'][last]):.6e} mol C m^-3")
    print(
        "Conservation RMSE [d(TotalC)/dt - (-F/h)]: "
        f"{out['conservation_rmse']:.6e} mol C m^-3 s^-1"
    )
    print(f"Mean pCO2_sw  : {np.mean(out['pCO2_sw'][last]):.2f} µatm")

    td = out["time_days"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Plot 1: Total Carbon, CO2*, DOC
    axes[0].plot(td, out["TotalC"], label="Total Carbon = CO2* + DOC")
    axes[0].plot(td, out["CO2"], label="CO2*")
    axes[0].plot(td, out["DOC_total"], label="DOC (total)")
    axes[0].set_ylabel("Carbon (mol C m$^{-3}$)")
    axes[0].set_title("Carbon pools (CO2* + DOC)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: DOC total + LDOC/SDOC/RDOC
    axes[1].plot(td, out["DOC_total"], linewidth=2.0, label="DOC total")
    axes[1].plot(td, out["LDOC"], label="LDOC")
    axes[1].plot(td, out["SDOC"], label="SDOC")
    axes[1].plot(td, out["RDOC"], label="RDOC")
    axes[1].set_ylabel("DOC (mol C m$^{-3}$)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("DOC speciation")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # Additional plot: LAST YEAR ONLY
    # Top: Total Carbon + CO2* + Total DOC
    # Mid: LDOC / SDOC / RDOC
    # Bottom: nondimensional forcing
    # ============================================================
    
    last_mask = td >= (td[-1] - 365.0)
    td_last = td[last_mask] - td[last_mask][0]
    
    co2_last = out["CO2"][last_mask]
    totalc_last = out["TotalC"][last_mask]
    doc_total_last = out["DOC_total"][last_mask]
    
    ldoc_last = out["LDOC"][last_mask]
    sdoc_last = out["SDOC"][last_mask]
    rdoc_last = out["RDOC"][last_mask]
    
    temp_last = out["Temp"][last_mask]
    light_last = out["Light"][last_mask]
    
    # Non-dimensionalize forcing (0–1 over last year)
    temp_nd = (temp_last - temp_last.min()) / (temp_last.max() - temp_last.min())
    light_nd = (light_last - light_last.min()) / (light_last.max() - light_last.min())
    
    fig2, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 1.0]}
    )
    
    # ----------------------------
    # Top panel: Carbon pools
    # ----------------------------
    ax_top.plot(td_last, totalc_last, label="Total Carbon", linewidth=2.5)
    ax_top.plot(td_last, co2_last, label="CO2*")
    ax_top.plot(td_last, doc_total_last, label="DOC (total)")
    ax_top.set_ylabel("Carbon (mol C m$^{-3}$)")
    ax_top.set_xlim(0, 365)
    ax_top.grid(True)
    ax_top.legend(loc="upper left")
    ax_top.set_title("Last Year: Carbon Pools, DOC Speciation & Forcing")
    
    # ----------------------------
    # Middle panel: DOC speciation
    # ----------------------------
    ax_mid.plot(td_last, ldoc_last, label="LDOC")
    ax_mid.plot(td_last, sdoc_last, label="SDOC")
    ax_mid.plot(td_last, rdoc_last, label="RDOC")
    ax_mid.set_ylabel("DOC (mol C m$^{-3}$)")
    ax_mid.grid(True)
    ax_mid.legend(loc="upper left")
    
    # ----------------------------
    # Bottom panel: Forcing
    # ----------------------------
    ax_bot.plot(td_last, temp_nd, linestyle="--", label="Temp (0–1)")
    ax_bot.plot(td_last, light_nd, linestyle=":", label="Light (0–1)")
    ax_bot.set_ylabel("Forcing (0–1)")
    ax_bot.set_ylim(0, 1)
    ax_bot.grid(True)
    ax_bot.legend(loc="upper left")
    
    # Season labels
    ax_bot.set_xlabel("Season")
    ax_bot.set_xticks([0, 91, 182, 273, 365])
    ax_bot.set_xticklabels(["Winter", "Spring", "Summer", "Autumn", "Winter"])
    
    plt.tight_layout()


if __name__ == "__main__":
    main()
