"""Regression tests for plotting utilities."""

from pathlib import Path

import numpy as np

from main_model.modules.plotting import save_diagnostics_plot



def _build_mock_output(n: int = 20) -> dict:
    """Create minimal model output dictionary required by save_diagnostics_plot."""
    t_s = np.linspace(0.0, 2.0 * 365.0 * 24.0 * 3600.0, n)
    t_days = t_s / (24.0 * 3600.0)

    dic = np.linspace(1.9, 2.1, n)
    doc = np.linspace(0.1, 0.2, n)

    return {
        "t_s": t_s,
        "t_days": t_days,
        "DOC": doc,
        "DIC": dic,
        "frac_CO2": np.linspace(1.0, 2.0, n),
        "frac_HCO3": np.linspace(85.0, 90.0, n),
        "frac_CO3": np.linspace(8.0, 10.0, n),
        "pH": np.linspace(8.0, 8.2, n),
        "F_ex": np.linspace(-1e-6, 1e-6, n),
        "T_C": np.linspace(2.0, 8.0, n),
        "Light": np.linspace(0.2, 1.0, n),
        "MLD": np.linspace(10.0, 40.0, n),
    }



def test_save_diagnostics_plot_handles_single_run_time_vector(tmp_path: Path):
    """Ensure single-run diagnostics plotting does not reference compare-run variables."""
    output_path = tmp_path / "diagnostics.png"
    out = _build_mock_output()

    returned_path = save_diagnostics_plot(out, output_path=str(output_path), plot_last_year_only=True)

    assert Path(returned_path).exists()
