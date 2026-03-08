"""Plotting utilities for model diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_START_DAY = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def _normalize_to_seasonal_range(values):
    """Normalize an array to [0, 1] using its own min/max."""
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _apply_time_axis_format(axes, td, plot_last_year_only):
    """Format the shared x-axis as month labels for one-year windows, else keep day units."""
    if plot_last_year_only:
        x0 = float(np.nanmin(td))
        month_tick_positions = x0 + _MONTH_START_DAY
        for ax in axes:
            ax.set_xticks(month_tick_positions)
        axes[-1].set_xlabel("Month")
        axes[-1].set_xticklabels(_MONTH_LABELS)
    else:
        axes[-1].set_xlabel("Time (days)")


def save_diagnostics_plot(
    out: dict,
    output_path: str = "results/main_model_diagnostics.png",
    plot_last_year_only: bool = True,
):
    """Save a multi-panel figure for key carbon-system diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = out["t_s"]
    sec_per_year = 365 * 24 * 3600
    mask = t >= (t[-1] - sec_per_year) if plot_last_year_only else np.ones_like(t, dtype=bool)

    outp = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == t.shape else v) for k, v in out.items()}
    td = outp["t_days"]
    time_window = "last year" if plot_last_year_only else "full simulation"

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

    total_carbon = outp["DIC"] + outp["DOC"]
    axes[0].plot(td, outp["DOC"], label="DOC")
    axes[0].plot(td, outp["DIC"], label="DIC")
    axes[0].plot(td, total_carbon, label="Total carbon (DOC + DIC)")
    axes[0].set_ylabel("Carbon (mol C m$^{-3}$)")
    axes[0].set_yscale("log")
    axes[0].set_title(f"DOC, DIC, and total carbon ({time_window})")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(td, outp["frac_CO2"], label="CO2* / DIC (%)")
    axes[1].plot(td, outp["frac_HCO3"], label="HCO3- / DIC (%)")
    axes[1].plot(td, outp["frac_CO3"], label="CO3-- / DIC (%)")
    axes[1].set_ylabel("Percent of DIC")
    axes[1].set_title("Carbonate speciation fractions")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-1, 100)
    axes[1].set_yticks([0.1, 1, 10, 100])
    axes[1].set_yticklabels(["0.1%", "1%", "10%", "100%"])
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(td, outp["pH"], label="pH")
    axes[2].set_ylabel("pH")
    axes[2].set_title("pH")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(td, outp["F"], label="Air–sea CO2 flux")
    axes[3].axhline(0, linestyle="--")
    axes[3].set_ylabel("F (mol C m$^{-2}$ s$^{-1}$)")
    axes[3].set_title("Air–sea CO2 flux")
    axes[3].grid(True)
    axes[3].legend()

    temp_norm = _normalize_to_seasonal_range(outp["T_C"])
    light_norm = _normalize_to_seasonal_range(outp["Light"])
    mld_norm = _normalize_to_seasonal_range(outp["MLD"])

    ax4 = axes[4]
    ax4.plot(td, temp_norm, label="Temperature (normalized)", color="tab:red")
    ax4.plot(td, light_norm, label="Light (normalized)", color="tab:blue")
    ax4.plot(td, mld_norm, label="MLD (normalized)", color="tab:green")
    ax4.set_ylabel("Normalized forcing (0-1)")
    ax4.set_ylim(0.0, 1.0)
    ax4.set_title("Forcing parameters (normalized to seasonal min/max)")
    ax4.grid(True)

    ax4.legend(loc="upper right")
    _apply_time_axis_format(axes, td, plot_last_year_only)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_biology_comparison_plot(
    out_on: dict,
    out_off: dict,
    output_path: str = "results/biology_toggle_comparison.png",
    plot_last_year_only: bool = True,
):
    """Save side-by-side ON/OFF biology diagnostics for DIC-DOC pools.

    Handles runs with different output lengths (e.g., if one solve terminates early).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t_on = out_on["t_s"]
    t_off = out_off["t_s"]
    sec_per_year = 365 * 24 * 3600
    mask_on = t_on >= (t_on[-1] - sec_per_year) if plot_last_year_only else np.ones_like(t_on, dtype=bool)
    mask_off = t_off >= (t_off[-1] - sec_per_year) if plot_last_year_only else np.ones_like(t_off, dtype=bool)

    onp = {k: (v[mask_on] if isinstance(v, np.ndarray) and v.shape == t_on.shape else v) for k, v in out_on.items()}
    offp = {k: (v[mask_off] if isinstance(v, np.ndarray) and v.shape == t_off.shape else v) for k, v in out_off.items()}

    td_on = onp["t_days"]
    td_off = offp["t_days"]
    time_window = "last year" if plot_last_year_only else "full simulation"

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

    axes[0].plot(td_off, offp["DIC"], label="DIC (biology OFF)")
    axes[0].plot(td_on, onp["DIC"], label="DIC (biology ON)")
    axes[0].plot(td_on, onp["DOC"], label="DOC total (biology ON)")
    axes[0].plot(td_on, onp["DIC"] + onp["DOC"], label="Total C = DIC + DOC (ON)")
    axes[0].set_ylabel("Carbon (mol C m$^{-3}$)")
    axes[0].set_title(f"Carbon pools with biology toggle ({time_window})")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(td_off, offp["pCO2_sw"], label="pCO2_sw (OFF)")
    axes[1].plot(td_on, onp["pCO2_sw"], label="pCO2_sw (ON)")
    axes[1].axhline(420.0, linestyle="--", label="pCO2_air")
    axes[1].set_ylabel("pCO2 (uatm)")
    axes[1].set_title("Surface pCO2")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(td_off, offp["F"], label="Air-sea flux (OFF)")
    axes[2].plot(td_on, onp["F"], label="Air-sea flux (ON)")
    axes[2].plot(td_on, onp["fremin"], label="DOC remin flux (ON)")
    axes[2].plot(td_on, -onp["fprod"], label="Biological production sink (ON)")
    axes[2].axhline(0, linestyle="--")
    axes[2].set_ylabel("F (mol C m$^{-2}$ s$^{-1}$)")
    axes[2].set_title("Air-sea and biology carbon fluxes")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(td_on, onp["pH"], label="pH (ON)")
    axes[3].plot(td_off, offp["pH"], label="pH (OFF)", linestyle="--")
    axes[3].set_ylabel("pH")
    axes[3].set_title("pH")
    axes[3].grid(True)
    axes[3].legend()

    temp_norm = _normalize_to_seasonal_range(onp["T_C"])
    light_norm = _normalize_to_seasonal_range(onp["Light"])
    mld_norm = _normalize_to_seasonal_range(onp["MLD"])

    ax4 = axes[4]
    ax4.plot(td_on, temp_norm, label="Temperature (normalized)", color="tab:red")
    ax4.plot(td_on, light_norm, label="Light (normalized)", color="tab:blue")
    ax4.plot(td_on, mld_norm, label="MLD (normalized)", color="tab:green")
    ax4.set_ylabel("Normalized forcing (0-1)")
    ax4.set_ylim(0.0, 1.0)
    ax4.set_title("Forcing parameters (normalized to seasonal min/max)")
    ax4.grid(True)

    ax4.legend(loc="upper right")
    _apply_time_axis_format(axes, td_on, plot_last_year_only)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
