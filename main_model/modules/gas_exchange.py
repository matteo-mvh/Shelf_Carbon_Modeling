"""Gas exchange helpers for air-sea CO2 transfer."""


def k_wanninkhof92(U10: float, T: float) -> float:
    """Wanninkhof (1992) gas transfer velocity k (m/s)."""
    T = float(T)
    sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * (float(U10) ** 2) * (sc / 660.0) ** (-0.5)
    return k_cm_hr / 100.0 / 3600.0


def co2_flux_and_tendency(co2: float, co2_eq: float, U10: float, T: float, h: float):
    """Return air-sea CO2 flux F and resulting co2 tendency dco2/dt."""
    k = k_wanninkhof92(U10, T)
    F = k * (float(co2) - float(co2_eq))
    dco2_dt = -F / float(h)
    return F, dco2_dt


def dic_tendency_from_air_sea(DIC, T, S, U10, pCO2_air, h, K0):
    """Backward-compatible API: computes tendency from DIC treated as CO2*."""
    _ = S
    co2_eq = float(K0) * float(pCO2_air)
    F, dco2_dt = co2_flux_and_tendency(DIC, co2_eq, U10, T, h)
    return dco2_dt, F
