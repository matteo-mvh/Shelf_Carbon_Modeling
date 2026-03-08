"""NPZDO-based calibration of PP(L) parameters used by the DOC biology module.

This module keeps the same NPZDO equations/forcing used in the reference script,
stripped down to only the computations needed for light-parameter fitting.

Running this module updates ``main_model/parameters.py`` with peaked PP(L) parameters:

    PP(L) = Pmax * L^n / (K_L^n + L^n) * exp(-L / K_I)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LightFitResult:
    """Fitted peaked-production coefficients for PP(L)."""

    pp_Pmax: float
    pp_K_L: float
    pp_n: float
    pp_K_I: float


def _build_npzdo_reference_outputs(years: float = 10.0):
    """Run the reference NPZDO model and return (L_last, P_depthavg_C)."""
    # Biology parameters
    P_growth_max = 2.0
    Z_consum_max = 1.0

    k_L = 20.0
    k_N = 0.3
    k_Z = 1.5
    k_O = 100.0

    m_P = 0.07
    m_Z = 0.1

    e_N = 0.3
    e_D = 0.3
    r = 0.5

    # Stoichiometry
    y_P = 9.0
    y_N = 10.0

    # Air-sea and bottom O2 exchange/sink
    O2_atm = 260.0
    k_O_surf = 1.0
    O2_n = 0.0
    k_O_bot = 1.0

    # Domain/grid
    depth = 200.0
    nz = 50
    z = np.linspace(0.0, depth, nz)
    dz = z[1] - z[0]

    # Sinking/advection [N, P, Z, D, O]
    W = [0.0, 5.0, 0.0, 7.0, 0.0]

    # Initial conditions
    N0 = np.linspace(0.0, 100.0, nz)
    P0 = np.linspace(0.5, 0.0, nz)
    Z0 = 0.1 * np.ones(nz)
    D0 = np.linspace(0.0, 100.0, nz)
    O0 = np.linspace(250.0, 0.0, nz)

    y0 = np.concatenate([N0, P0, Z0, D0, O0])

    iN = slice(0, nz)
    iP = slice(nz, 2 * nz)
    iZ = slice(2 * nz, 3 * nz)
    iD = slice(3 * nz, 4 * nz)
    iO = slice(4 * nz, 5 * nz)

    t_max = 365.0 * float(years)
    t_span = (0.0, t_max)
    t_eval = np.linspace(0.0, t_max, 1000)

    def day_of_year(t):
        return float(t % 365.0)

    def get_limits(Lz, N, P, O, delta_O=10.0, oxyg_switch=True):
        light_lim = Lz / (Lz + k_L + 1e-12)
        nut_lim = N / (N + k_N + 1e-12)
        graze_lim = P / (P + k_Z + 1e-12)
        if oxyg_switch:
            oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
        else:
            oxyg_lim = O / (O + k_O + 1e-12)
        return light_lim, nut_lim, graze_lim, oxyg_lim

    def getLIGHTandKAPPAS(t, P=None, D=None, Lightswitch=False, Seasonality=True, bio_attenuation=True):
        k_water = 0.1
        k_bio = 0.20

        if bio_attenuation and (P is not None) and (D is not None):
            P_pos = np.maximum(P, 0.0)
            D_pos = np.maximum(D, 0.0)
            bio_integral = np.cumsum(P_pos + D_pos) * dz
        else:
            bio_integral = 0.0

        if Seasonality:
            doy = day_of_year(t)

            zMix = 0.05 * depth
            zMixWinter = 0.8 * depth
            tMaxSpring = 90.0
            zetaMaxSteep = 2.0
            z_mix = 0.5 * (1 - np.sin(2 * np.pi * (doy - tMaxSpring) / 365.0)) ** zetaMaxSteep
            z_mix = z_mix * (zMixWinter - zMix) + zMix

            kappa_top_summer = 5.0
            kappa_top_winter = 15.0
            kappa_bottom_summer = 0.5
            kappa_bottom_winter = 15.0

            season_shape = 0.5 * (1 - np.sin(2 * np.pi * (doy - tMaxSpring) / 365.0))
            kappa_top = kappa_top_summer + (kappa_top_winter - kappa_top_summer) * (season_shape**zetaMaxSteep)
            kappa_bottom = kappa_bottom_summer + (kappa_bottom_winter - kappa_bottom_summer) * (
                season_shape**zetaMaxSteep
            )
            zeta_mix = 10.0
            kappa_center = 0.5 * (1 - np.tanh((z - z_mix) / zeta_mix)) * (kappa_top - kappa_bottom) + kappa_bottom

            L0_min = 50.0
            L0_max = 1000.0
            if Lightswitch:
                spring_center = 90.0
                autumn_center = 260.0
                spring_width = 20.0
                autumn_width = 30.0
                spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
                autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
                seasonal_shape = spring_switch * autumn_switch
                L0 = L0_min + (L0_max - L0_min) * seasonal_shape
            else:
                phase_shift = 80.0
                seasonal_shape = (1 + np.sin(2 * np.pi * (doy - phase_shift) / 365.0)) ** 2
                L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4.0
        else:
            kappa_surface = 10.0
            kappa_bottom = 1.0
            z_transition = 50.0
            zeta_mix = 10.0
            kappa_center = 0.5 * (1 - np.tanh((z - z_transition) / zeta_mix)) * (kappa_surface - kappa_bottom)
            kappa_center = kappa_center + kappa_bottom
            L0 = 1400.0

        kappa_interface = np.zeros(nz + 1)
        kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
        kappa_interface[0] = kappa_center[0]
        kappa_interface[nz] = kappa_center[-1]

        if bio_attenuation and (P is not None) and (D is not None):
            Lz = L0 * np.exp(-k_water * z - k_bio * bio_integral)
        else:
            Lz = L0 * np.exp(-k_water * z)

        return kappa_interface, Lz, L0

    def surface_flux(tracer_name, C_surface):
        if tracer_name == "O":
            return (O2_atm - C_surface) * k_O_surf
        return 0.0

    def bottom_flux(tracer_name, C_bottom):
        del C_bottom
        if tracer_name == "O":
            O_use = 0.0
            return (O_use + O2_n) * k_O_bot
        return 0.0

    def vertical_transport(C, kappa_interface, w=0.0, tracer_name=""):
        J = np.zeros(nz + 1)

        if w >= 0:
            Ja = w * C[:-1]
        else:
            Ja = w * C[1:]

        Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
        J[1:nz] = Ja + Jd

        J[0] = surface_flux(tracer_name, C[0])
        J[nz] = bottom_flux(tracer_name, C[-1])

        return -(J[1:] - J[:-1]) / dz

    def rhs(t, y):
        N = y[iN]
        P = y[iP]
        Z = y[iZ]
        D = y[iD]
        O = y[iO]

        kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)
        light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N, P, O)

        P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
        Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

        eps = 0.0
        r_factor = r * (eps + (1 - eps) * oxyg_lim)

        N_uptake = P_growth * P
        P_mort = m_P * P
        Z_grazing = Z_consum * Z
        Z_mort = m_Z * Z**2
        remin = r_factor * D

        dN_reac = -N_uptake + e_N * Z_grazing + remin
        dP_reac = N_uptake - Z_grazing - P_mort
        dZ_reac = (1 - e_N - e_D) * Z_grazing - Z_mort
        dD_reac = P_mort + e_D * Z_grazing + Z_mort - remin
        dO_reac = -y_N * e_N * Z_grazing - y_N * remin + y_P * N_uptake

        dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N") + dN_reac
        dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
        dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
        dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
        dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

        return np.concatenate([dN, dP, dZ, dD, dO])

    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-7,
        atol=1e-10,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(f"NPZDO integration failed: {sol.message}")

    CN = 106.0 / 16.0

    # Keep this identical to the reference script behavior.
    t_mask = sol.t >= (t_max - 4 * 365.0)
    t_last = sol.t[t_mask]

    P_last = sol.y[iP, t_mask]
    z_mask_eu = z <= 50.0
    P_depthavg_N = np.mean(P_last[z_mask_eu, :], axis=0)
    P_depthavg_C = P_depthavg_N * CN

    L_last = np.array([getLIGHTandKAPPAS(ti)[2] for ti in t_last])

    return L_last, P_depthavg_C


def peaked_pp(light, Pmax, K_L, n, K_I):
    """Peaked PP(L): Hill rise + exponential high-light inhibition."""
    light = np.asarray(light, dtype=float)
    light = np.maximum(light, 0.0)
    n_eff = np.maximum(float(n), 1e-12)
    K_L_eff = np.maximum(float(K_L), 1e-12)
    K_I_eff = np.maximum(float(K_I), 1e-12)

    hill_part = light**n_eff / (K_L_eff**n_eff + light**n_eff)
    inhib_part = np.exp(-light / K_I_eff)

    return float(Pmax) * hill_part * inhib_part


def fit_light_parameters_for_params(n_bins: int = 20):
    """Run NPZDO model and fit PP(L) coefficients for Params.pp_* fields."""
    L_data, P_data = _build_npzdo_reference_outputs()

    bins = np.linspace(L_data.min(), L_data.max(), int(n_bins) + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    P_binned = np.full(int(n_bins), np.nan)

    for i in range(int(n_bins)):
        mask = (L_data >= bins[i]) & (L_data < bins[i + 1])
        if np.any(mask):
            P_binned[i] = np.mean(P_data[mask])
            P_binned[i] = np.percentile(P_data[mask], 50)

    valid = np.isfinite(P_binned)

    p0 = [np.nanmax(P_binned), 200.0, 2.0, 1000.0]
    bounds = ([0.0, 1e-6, 0.5, 1e-6], [1e4, 1e4, 10.0, 1e5])

    popt, _ = curve_fit(
        peaked_pp,
        bin_centers[valid],
        P_binned[valid],
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )

    Pmax_fit, K_L_fit, n_fit, K_I_fit = popt

    # ============================================================
    # Plot data + bins + fitted peaked PP curve
    # ============================================================
    L_fit = np.linspace(0.0, np.max(L_data), 1000)
    P_fit = peaked_pp(L_fit, Pmax_fit, K_L_fit, n_fit, K_I_fit)

    plt.figure(figsize=(10, 6))

    plt.plot(
        L_data,
        P_data,
        "o",
        ms=3,
        alpha=0.25,
        color="tab:blue",
        label="NPZDO output points",
    )

    plt.plot(
        bin_centers[valid],
        P_binned[valid],
        "x",
        ms=7,
        color="black",
        label="Binned means",
    )

    plt.plot(
        L_fit,
        P_fit,
        lw=2.5,
        color="black",
        label=f"Peaked fit (n={n_fit:.2f}, K_I={K_I_fit:.1f})",
    )

    plt.xlabel("Surface light $L_0$ [$\\mu$mol photons m$^{-2}$ s$^{-1}$]")
    plt.ylabel("Depth-averaged phytoplankton $\\overline{P}$ [mmol C m$^{-3}$]")
    plt.title("NPZDO-derived PP(L) calibration")
    plt.grid(True, alpha=0.3)
    plt.xlim([min(L_fit),max(L_fit)])
    plt.legend()
    plt.tight_layout()
    plt.show()

    return LightFitResult(
        pp_Pmax=float(Pmax_fit),
        pp_K_L=float(K_L_fit),
        pp_n=float(n_fit),
        pp_K_I=float(K_I_fit),
    )

def format_params_block(result: LightFitResult) -> str:
    """Return ready-to-paste lines for the Params PP(L) section."""
    return (
        f"pp_Pmax: float = {result.pp_Pmax:.6f}\n"
        f"pp_K_L: float = {result.pp_K_L:.6f}\n"
        f"pp_n: float = {result.pp_n:.6f}\n"
        f"pp_K_I: float = {result.pp_K_I:.6f}"
    )


def apply_fitted_light_parameters_to_file(parameters_path: str | Path | None = None) -> LightFitResult:
    """Fit PP(L) coefficients and overwrite pp_* defaults in parameters.py."""
    result = fit_light_parameters_for_params()

    if parameters_path is None:
        # Light_Parameter.py is in main_model/modules/
        # parameters.py is in main_model/
        path = Path(__file__).resolve().parents[1] / "parameters.py"
    else:
        path = Path(parameters_path).resolve()

    text = path.read_text(encoding="utf-8")

    replacements = {
        "pp_Pmax": result.pp_Pmax,
        "pp_K_L": result.pp_K_L,
        "pp_n": result.pp_n,
        "pp_K_I": result.pp_K_I,
    }

    for key, value in replacements.items():
        pattern = rf"^\s*{key}: float = [^\n]+$"
        replacement = f"    {key}: float = {value:.6f}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"Could not update parameter line for {key} in {path}")

    path.write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    fitted = apply_fitted_light_parameters_to_file()
    print("Updated main_model/parameters.py with fitted PP(L) parameters:")
    print(format_params_block(fitted))
