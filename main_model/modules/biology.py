"""Simple biology parameterizations for DIC and glucose dynamics."""


def glucose_production_rate(DIC, light, Pmax, Km_C):
    """P in mol glucose m^-3 s^-1 (toy), light forcing + DIC limitation."""
    light_norm = min(max(float(light), 0.0), 1.0)
    dic_pos = max(float(DIC), 0.0)
    lim = dic_pos / (float(Km_C) + dic_pos)
    return float(Pmax) * light_norm * lim


def remin_rate_from_tau_days(tau_days):
    """Convert remineralization time scale in days to first-order rate (s^-1)."""
    tau_days = float(tau_days)
    if tau_days <= 0.0:
        return 0.0
    return 1.0 / (tau_days * 24.0 * 3600.0)


def tendencies(DIC, G, light, Pmax, Km_C, tau_remin_days):
    """Toy biology: G' = P-R and DIC' = -6P+6R."""
    P = glucose_production_rate(DIC, light, Pmax=Pmax, Km_C=Km_C)
    r_remin = remin_rate_from_tau_days(tau_remin_days)
    R = r_remin * float(G)
    return -6.0 * P + 6.0 * R, P - R, P, R
