"""Air-sea gas exchange parameterizations."""


def k_wanninkhof(U10: float, T: float) -> float:
    """Wanninkhof (1992) gas transfer velocity k (m/s)."""
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * U10**2 * (Sc / 660.0) ** (-0.5)
    return k_cm_hr / 100.0 / 3600.0


def dic_tendency_from_air_sea(DIC: float, T: float, S: float, U10: float, pCO2_air: float, h: float, K0: float):
    """Return DIC tendency and flux with F>0 ocean->air."""
    _ = S
    k = k_wanninkhof(U10, T)
    dic_eq = K0 * pCO2_air
    F = k * (DIC - dic_eq)
    dDIC_phys = -F / h
    return float(dDIC_phys), float(F)


# Backwards-compatible scaffold name.
def compute_air_sea_co2_flux(state: dict) -> float:
    _ = state
    return 0.0
