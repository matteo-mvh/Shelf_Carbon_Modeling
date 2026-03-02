"""Plotting utilities for model diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    axes[0].set_title(f"DOC, DIC, and total carbon ({time_window})")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(td, outp["frac_CO2"], label="CO2* / DIC (%)")
    axes[1].plot(td, outp["frac_HCO3"], label="HCO3- / DIC (%)")
    axes[1].plot(td, outp["frac_CO3"], label="CO3-- / DIC (%)")
    axes[1].set_ylabel("Percent of DIC")
    axes[1].set_title("Carbonate speciation fractions")
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

    axes[4].plot(td, outp["T_C"], label="Temperature")
    axes[4].set_ylabel("Temperature (°C)")
    axes[4].set_xlabel("Time (days)")
    axes[4].set_title("Temperature")
    axes[4].grid(True)
    axes[4].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
