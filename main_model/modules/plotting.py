"""Plotting utilities for model diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_diagnostics_plot(
    out_on: dict,
    out_off: dict,
    output_path: str = "results/main_model_diagnostics.png",
    plot_last_year_only: bool = True,
):
    """Save a multi-panel figure comparing speciation ON vs OFF runs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = out_on["t_s"]
    sec_per_year = 365 * 24 * 3600
    mask = t >= (t[-1] - sec_per_year) if plot_last_year_only else np.ones_like(t, dtype=bool)

    onp = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == t.shape else v) for k, v in out_on.items()}
    offp = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape == t.shape else v) for k, v in out_off.items()}
    td = onp["t_days"]
    time_window = "last year" if plot_last_year_only else "full simulation"

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(td, offp["DIC"], label="DIC (speciation OFF; CO2* only)")
    axes[0].plot(td, onp["DIC"], label="DIC (speciation ON; CO2*+HCO3-+CO3--)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("DIC (mol C m$^{-3}$)")
    axes[0].set_title(f"DIC comparison (speciation ON vs OFF, {time_window})")
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

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
