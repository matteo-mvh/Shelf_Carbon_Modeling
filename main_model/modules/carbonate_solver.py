"""Carbonate chemistry solver module scaffold."""

from parameters import PARAMS


def solve_carbonate_system(state: dict) -> dict:
    """Return placeholder carbonate-system diagnostics."""
    _ = state
    _ = PARAMS
    return {
        "pco2_ocean_uatm": 0.0,
        "ph_total_scale": 8.0,
    }
