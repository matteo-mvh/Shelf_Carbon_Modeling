"""Simple biology tendencies."""


def tendencies(DIC: float, G: float, T: float, Pmax: float, Km_C: float, Tref: float, Q10: float, tau_remin_days: float):
    """Return (dDIC_bio, dG_dt, production, remineralization)."""
    fT = Q10 ** ((T - Tref) / 10.0)
    dic_pos = max(float(DIC), 0.0)
    lim = dic_pos / (Km_C + dic_pos)
    Pprod = Pmax * fT * lim

    remin_rate = 0.0 if tau_remin_days <= 0 else 1.0 / (tau_remin_days * 24.0 * 3600.0)
    Rremin = remin_rate * G

    dG_dt = Pprod - Rremin
    dDIC_bio = -6.0 * Pprod + 6.0 * Rremin
    return float(dDIC_bio), float(dG_dt), float(Pprod), float(Rremin)


# Backwards-compatible scaffold name.
def compute_biological_carbon_tendency(state: dict) -> float:
    _ = state
    return 0.0
