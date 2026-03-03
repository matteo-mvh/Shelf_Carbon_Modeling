"""
Air–Sea CO2 Exchange with DOC Speciation
----------------------------------------

Single-file test model that extends a 1-box air–sea CO2 system with explicit
DOC classes:
- LDOC: low light half-saturation (kL), high remineralization
- SDOC: mid kL, mid remineralization
- RDOC: high kL, low remineralization

Primary production (PP) transfers carbon from DIC to DOC classes as fixed
fractions. Bacterial remineralization returns each DOC class to DIC. Aging
pathways interconnect DOC classes (LDOC -> SDOC -> RDOC).

All biological transformations are internally carbon-conservative by
construction (DIC + LDOC + SDOC + RDOC changes only by air–sea exchange).
"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


SEC_PER_DAY = 24 * 3600
SEC_PER_YEAR = 365 * SEC_PER_DAY


def solubility_co2(T, S):
    """Weiss (1974) CO2 solubility K0 (mol m^-3 µatm^-1)."""
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
    """Wanninkhof (1992) gas transfer velocity k (m s^-1)."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660) ** (-0.5)
    return k_cm_hr / 100 / 3600


def seasonal_sine(t, vmin, vmax, phase_shift_days=0.0):
    """Seasonal sinusoid in [vmin, vmax]."""
    t = np.atleast_1d(t).astype(float)
    mean = 0.5 * (vmin + vmax)
    amp = 0.5 * (vmax - vmin)
    phase = 2.0 * np.pi * phase_shift_days * SEC_PER_DAY / SEC_PER_YEAR
    return mean + amp * np.sin(2 * np.pi * t / SEC_PER_YEAR + phase)


def positive_light(t, i_min=5.0, i_max=220.0):
    """Seasonal irradiance (W m^-2), clipped non-negative for safety."""
    I = seasonal_sine(t, i_min, i_max, phase_shift_days=-80.0)
    return np.maximum(I, 0.0)


def temperature(t, t_min=2.0, t_max=20.0):
    """Seasonal temperature forcing (°C)."""
    return seasonal_sine(t, t_min, t_max, phase_shift_days=0.0)


def q10_factor(T, Tref=15.0, q10=2.0):
    """Q10 temperature scaling factor."""
    return q10 ** ((np.asarray(T) - Tref) / 10.0)


def light_limitation(I, k_L):
    """Michaelis-Menten-like light limitation term I/(kL + I)."""
    I = np.asarray(I)
    return I / (k_L + I)


def rhs(
    t,
    y,
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
    km_dic,
    tref_bio,
    q10_pp,
    q10_remin,
):
    """RHS for y = [DIC, LDOC, SDOC, RDOC]."""
    dic, ldoc, sdoc, rdoc = y

    T = temperature(t)[0]
    I = positive_light(t)[0]

    # Air-sea exchange (affects DIC only)
    k_gas = k_wanninkhof(U10, T)
    K0 = solubility_co2(T, S)
    dic_eq = K0 * pco2_air
    F = k_gas * (dic - dic_eq)  # positive: ocean -> atmosphere
    d_dic_phys = -F / h

    # Carbon-limited primary production potential
    dic_pos = max(float(dic), 0.0)
    c_lim = dic_pos / (km_dic + dic_pos)

    # Class-specific PP production with different kL and temperature forcing
    f_light = np.array([light_limitation(I, kL_i) for kL_i in kL], dtype=float)
    fT_pp = q10_factor(T, Tref=tref_bio, q10=q10_pp)
    pp_total = pp_max * c_lim * float(fT_pp)
    prod = pp_total * frac_pp * f_light  # [LDOC, SDOC, RDOC]

    # Remineralization DOC -> DIC (temperature-forced)
    fT_remin = float(q10_factor(T, Tref=tref_bio, q10=q10_remin))
    remin_rates = remin_rates_ref * fT_remin
    remin = np.array(
        [remin_rates[0] * ldoc, remin_rates[1] * sdoc, remin_rates[2] * rdoc],
        dtype=float,
    )

    # Aging connections
    age_ls = age_l_to_s * ldoc
    age_sr = age_s_to_r * sdoc

    d_ldoc = prod[0] - remin[0] - age_ls
    d_sdoc = prod[1] - remin[1] + age_ls - age_sr
    d_rdoc = prod[2] - remin[2] + age_sr

    # Internal conservation: DIC decreases by PP sum, increases by remin sum
    d_dic_bio = -np.sum(prod) + np.sum(remin)

    return [d_dic_phys + d_dic_bio, d_ldoc, d_sdoc, d_rdoc]


def run_model():
    # Physical setup
    S = 30.0
    U10 = 6.0
    h = 50.0
    pco2_air = 420.0
    pco2_sw_init = 300.0

    # Biology and DOC speciation setup
    pp_max = 6.5e-7  # mol C m^-3 s^-1
    km_dic = 1.0e-2

    # Requested ordering:
    # LDOC low kL, SDOC mid kL, RDOC high kL
    kL = np.array([20.0, 80.0, 160.0])

    # Fraction of PP routed to DOC classes
    frac_pp = np.array([0.50, 0.35, 0.15])
    frac_pp = frac_pp / np.sum(frac_pp)

    # Requested ordering:
    # LDOC high remin, SDOC mid remin, RDOC low remin
    tau_days = np.array([20.0, 90.0, 400.0])
    remin_rates_ref = 1.0 / (tau_days * SEC_PER_DAY)

    # Shared temperature scaling for biology
    tref_bio = 15.0
    q10_pp = 2.0
    q10_remin = 1.8

    # Aging rates
    age_l_to_s = 1.0 / (120.0 * SEC_PER_DAY)
    age_s_to_r = 1.0 / (240.0 * SEC_PER_DAY)

    # Time
    years = 8
    t_eval = np.arange(0.0, years * SEC_PER_YEAR + 1.0, SEC_PER_DAY)
    t_span = (t_eval[0], t_eval[-1])

    # Initial state
    T0 = temperature(0.0)[0]
    K0_0 = solubility_co2(T0, S)
    dic0 = float(K0_0 * pco2_sw_init)
    y0 = [dic0, 0.0, 0.0, 0.0]

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
            km_dic,
            tref_bio,
            q10_pp,
            q10_remin,
        ),
        rtol=1e-7,
        atol=1e-10,
    )

    dic, ldoc, sdoc, rdoc = sol.y
    doc_total = ldoc + sdoc + rdoc
    total_c = dic + doc_total

    T = temperature(sol.t)
    I = positive_light(sol.t)
    K0 = solubility_co2(T, S)
    pco2_sw = dic / K0

    k_gas = np.array([k_wanninkhof(U10, Ti) for Ti in T])
    dic_eq = K0 * pco2_air
    F = k_gas * (dic - dic_eq)

    # Carbon conservation diagnostics (internal biology)
    # Numerically, d(TotalC)/dt should equal d(DIC)/dt from air-sea only = -F/h.
    dtotal_num = np.gradient(total_c, sol.t)
    dtotal_expected = -F / h
    conservation_rmse = np.sqrt(np.mean((dtotal_num - dtotal_expected) ** 2))

    return {
        "time_days": sol.t / SEC_PER_DAY,
        "DIC": dic,
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
    }


def main():
    out = run_model()

    print("=== DOC speciation diagnostics (final year mean) ===")
    last = out["time_days"] >= (out["time_days"][-1] - 365)
    print(f"Mean DIC      : {np.mean(out['DIC'][last]):.6e} mol C m^-3")
    print(f"Mean DOC total: {np.mean(out['DOC_total'][last]):.6e} mol C m^-3")
    print(f"Mean LDOC     : {np.mean(out['LDOC'][last]):.6e} mol C m^-3")
    print(f"Mean SDOC     : {np.mean(out['SDOC'][last]):.6e} mol C m^-3")
    print(f"Mean RDOC     : {np.mean(out['RDOC'][last]):.6e} mol C m^-3")
    print(
        "Conservation RMSE [d(TotalC)/dt - (-F/h)]: "
        f"{out['conservation_rmse']:.6e} mol C m^-3 s^-1"
    )

    td = out["time_days"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Requested plot 1: Total Carbon, DIC, DOC
    axes[0].plot(td, out["TotalC"], label="Total Carbon = DIC + DOC")
    axes[0].plot(td, out["DIC"], label="DIC")
    axes[0].plot(td, out["DOC_total"], label="DOC (total)")
    axes[0].set_ylabel("Carbon (mol C m$^{-3}$)")
    axes[0].set_title("Carbon pools")
    axes[0].grid(True)
    axes[0].legend()

    # Requested plot 2: DOC total + LDOC/SDOC/RDOC
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


if __name__ == "__main__":
    main()
