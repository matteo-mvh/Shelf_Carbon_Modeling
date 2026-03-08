"""Single-run entry point for the modular surface-ocean carbon box model.

This script runs the model with biology enabled and saves diagnostics for that run.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from main_model.main_comparison import open_plot, run
from main_model.modules.plotting import save_diagnostics_plot, save_outputs_overview_plot
from main_model.parameters import Params


def main():
    params_on = Params(biology_on=True)
    out_on = run(params_on)

    print("Run success (ON):", out_on["success"])

    if out_on["success"]:
        single_run_plot_path = save_diagnostics_plot(
            out_on,
            output_path="results/main_model_diagnostics_on.png",
            plot_last_year_only=params_on.plot_last_year_only,
        )
        print("Saved single-run plot (ON):", single_run_plot_path)

        overview_plot_path = save_outputs_overview_plot(
            out_on,
            output_path="results/main_model_outputs_overview_on.png",
            plot_last_year_only=True,
        )
        print("Saved outputs overview plot (ON):", overview_plot_path)

        if open_plot(overview_plot_path):
            print("Opened outputs overview plot (ON):", overview_plot_path)
        elif open_plot(single_run_plot_path):
            print("Opened single-run plot (ON):", single_run_plot_path)
        else:
            print("Could not automatically open plots (ON):", overview_plot_path)
    else:
        print("Skipping ON diagnostics plot because ON integration failed.")
        print("ON message:", out_on["message"])

    return out_on


if __name__ == "__main__":
    main()
