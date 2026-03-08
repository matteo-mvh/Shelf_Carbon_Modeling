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


def _slice_output_window(out: dict, plot_last_year_only: bool):
    """Return a time-windowed copy of model output arrays."""
    t = out["t_s"]
    sec_per_year = 365 * 24 * 3600
    mask = t >= (t[-1] - sec_per_year) if plot_last_year_only else np.ones_like(t, dtype=bool)
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == t.shape else v) for k, v in out.items()}


def save_diagnostics_plot(
    out: dict,
    output_path: str = "results/main_model_diagnostics.png",
    plot_last_year_only: bool = True,
):
    """Save a multi-panel figure for key carbon-system diagnostics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    outp = _slice_output_window(out, plot_last_year_only)
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

    axes[3].plot(td, outp["F_ex"], label="Air–sea CO2 flux")
    axes[3].axhline(0, linestyle="--")
    axes[3].set_ylabel("F_ex (mol C m$^{-2}$ s$^{-1}$)")
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


def save_outputs_overview_plot(
    out: dict,
    output_path: str = "results/main_model_outputs_overview.png",
    plot_last_year_only: bool = True,
):
    """Save a wide non-diagnostics figure focused on key yearly model outputs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    outp = _slice_output_window(out, plot_last_year_only)
    td = outp["t_days"]

    total_carbon = outp["DIC"] + outp["DOC"]
    bio_uptake = outp["fprod"] - outp["fremin"]

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(4, 2, width_ratios=[3.8, 2.0], hspace=0.35, wspace=0.25)

    left_axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    for ax in left_axes[1:]:
        ax.sharex(left_axes[0])

    left_axes[0].plot(td, outp["DOC"], label="DOC", lw=2)
    left_axes[0].plot(td, outp["DIC"], label="DIC", lw=2)
    left_axes[0].plot(td, total_carbon, label="Total Carbon", lw=2.2)
    left_axes[0].set_yscale("log")
    left_axes[0].set_ylabel("mol C m$^{-3}$")
    left_axes[0].set_title("Carbon pools (last year)")
    left_axes[0].grid(True, alpha=0.35)
    left_axes[0].legend(loc="best")

    left_axes[1].plot(td, outp["F_ex"], label="F_ex (Air-Sea Exchange)", lw=2)
    left_axes[1].plot(td, bio_uptake, label="Biouptake = F_prod - F_remin", lw=2)
    left_axes[1].axhline(0.0, color="black", ls="--", lw=1)
    left_axes[1].set_ylabel("mol C m$^{-2}$ s$^{-1}$")
    left_axes[1].set_title("Carbon fluxes")
    left_axes[1].grid(True, alpha=0.35)
    left_axes[1].legend(loc="best")

    left_axes[2].plot(td, outp["pH"], color="tab:purple", lw=2, label="pH")
    left_axes[2].set_ylabel("pH")
    left_axes[2].set_title("pH")
    left_axes[2].grid(True, alpha=0.35)
    left_axes[2].legend(loc="best")

    temp_norm = _normalize_to_seasonal_range(outp["T_C"])
    light_norm = _normalize_to_seasonal_range(outp["Light"])
    mld_norm = _normalize_to_seasonal_range(outp["MLD"])
    left_axes[3].plot(td, temp_norm, label="Temperature", color="tab:red", lw=2)
    left_axes[3].plot(td, light_norm, label="Light", color="tab:blue", lw=2)
    left_axes[3].plot(td, mld_norm, label="MLD", color="tab:green", lw=2)
    left_axes[3].set_ylabel("Normalized")
    left_axes[3].set_ylim(0.0, 1.0)
    left_axes[3].set_title("Forcing parameters")
    left_axes[3].grid(True, alpha=0.35)
    left_axes[3].legend(loc="best")

    _apply_time_axis_format(left_axes, td, plot_last_year_only)

    ax_dic_pie = fig.add_subplot(grid[0, 1])
    dic_species = np.array([
        np.nanmean(outp["CO2"]),
        np.nanmean(outp["HCO3"]),
        np.nanmean(outp["CO3"]),
    ])
    dic_species = np.clip(dic_species, 0.0, None)
    dic_labels = ["CO2*", "HCO3-", "CO3--"]
    ax_dic_pie.pie(dic_species, labels=dic_labels, autopct="%1.1f%%", startangle=90)
    ax_dic_pie.set_title("Mean DIC species share\n(last year)")

    ax_doc_pie = fig.add_subplot(grid[1, 1])
    doc_species = np.array([
        np.nanmean(outp["LDOC"]),
        np.nanmean(outp["SDOC"]),
        np.nanmean(outp["RDOC"]),
    ])
    doc_species = np.clip(doc_species, 0.0, None)
    doc_labels = ["LDOC", "SDOC", "RDOC"]
    ax_doc_pie.pie(doc_species, labels=doc_labels, autopct="%1.1f%%", startangle=90)
    ax_doc_pie.set_title("Mean DOC species share\n(last year)")

    ax_stats_main = fig.add_subplot(grid[2, 1])
    ax_stats_main.axis("off")
    main_metrics = (
        f"Total Carbon mean: {np.nanmean(total_carbon):.4f} mol C m$^{{-3}}$\n"
        f"Total DIC mean: {np.nanmean(outp['DIC']):.4f} mol C m$^{{-3}}$\n"
        f"Total DOC mean: {np.nanmean(outp['DOC']):.4f} mol C m$^{{-3}}$\n"
        f"Mean pH: {np.nanmean(outp['pH']):.3f}"
    )
    ax_stats_main.text(
        0.01,
        0.95,
        main_metrics,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
    )
    ax_stats_main.set_title("Stored carbon metrics")

    ax_stats_extra = fig.add_subplot(grid[3, 1])
    ax_stats_extra.axis("off")
    extra_metrics = (
        f"Mean Air-Sea F_ex: {np.nanmean(outp['F_ex']):.3e} mol C m$^{{-2}}$ s$^{{-1}}$\n"
        f"Mean Biouptake (F_prod - F_remin): {np.nanmean(bio_uptake):.3e} mol C m$^{{-2}}$ s$^{{-1}}$\n"
        f"Mean ΔpCO2: {np.nanmean(outp['delta_pCO2']):.2f} µatm\n"
        f"pH range: {np.nanmin(outp['pH']):.3f} - {np.nanmax(outp['pH']):.3f}"
    )
    ax_stats_extra.text(
        0.01,
        0.95,
        extra_metrics,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
    )
    ax_stats_extra.set_title("Additional key outputs")

    fig.suptitle("Shelf carbon model outputs overview", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
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

    axes[1].plot(td_off, offp["delta_pCO2"], label="ΔpCO2 (OFF)")
    axes[1].plot(td_on, onp["delta_pCO2"], label="ΔpCO2 (ON)")
    axes[1].axhline(0.0, linestyle="--", label="Equilibrium (ΔpCO2 = 0)")
    axes[1].set_ylabel("ΔpCO2 (uatm)")
    axes[1].set_title("Air-sea pCO2 difference")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(td_off, offp["F_ex"], label="Air-sea flux (OFF)")
    axes[2].plot(td_on, onp["F_ex"], label="Air-sea flux (ON)")
    axes[2].plot(td_on, onp["fremin"] - onp["fprod"], label="Biouptake = fremin - fprod (ON)")
    axes[2].axhline(0, linestyle="--")
    axes[2].set_ylabel("F_ex (mol C m$^{-2}$ s$^{-1}$)")
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
