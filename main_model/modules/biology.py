"""Simple biology parameterizations for DIC and glucose dynamics."""


def glucose_production_rate(DIC, T, Pmax, Km_C, Tref=20.0, Q10=2.0):
    """P in mol glucose m^-3 s^-1 (toy), Q10 temp scaling + DIC limitation."""
    fT = float(Q10) ** ((float(T) - float(Tref)) / 10.0)
    dic_pos = max(float(DIC), 0.0)
    lim = dic_pos / (float(Km_C) + dic_pos)
    return float(Pmax) * fT * lim


def remin_rate_from_tau_days(tau_days):
    """Convert remineralization time scale in days to first-order rate (s^-1)."""
    tau_days = float(tau_days)
    if tau_days <= 0.0:
        return 0.0
    return 1.0 / (tau_days * 24.0 * 3600.0)


def tendencies(DIC, G, T, Pmax, Km_C, Tref, Q10, tau_remin_days):
    """Toy biology: G' = P-R and DIC' = -6P+6R."""
    P = glucose_production_rate(DIC, T, Pmax=Pmax, Km_C=Km_C, Tref=Tref, Q10=Q10)
    r_remin = remin_rate_from_tau_days(tau_remin_days)
    R = r_remin * float(G)
    return -6.0 * P + 6.0 * R, P - R, P, R
