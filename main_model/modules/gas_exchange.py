"""Gas exchange helpers for air-sea CO2 transfer."""

from parameters import PARAMS


def k_wanninkhof92(U10: float, T: float) -> float:
    """Wanninkhof (1992) gas transfer velocity k (m/s)."""
    T = float(T)
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3
    k_cm_hr = 0.31 * (float(U10) ** 2) * (Sc / 660.0) ** (-0.5)
    return k_cm_hr / 100.0 / 3600.0  # cm/hr -> m/s


def dic_tendency_from_air_sea(
    DIC: float,
    T: float,
    S: float,
    U10: float,
    pCO2_air: float,
    h: float,
    K0: float,
) -> tuple[float, float]:
    """
    Air-sea flux: F = k*(DIC - K0*pCO2_air)  (positive ocean->atm)
    dDIC/dt = -F/h
    """
    _ = S
    k = k_wanninkhof92(U10, T)
    dic_eq = K0 * float(pCO2_air)
    F = k * (float(DIC) - dic_eq)   # mol C m^-2 s^-1 (positive ocean->atm)
    dDIC_dt = -F / float(h)         # mol C m^-3 s^-1
    return dDIC_dt, F


def compute_air_sea_co2_flux(state: dict) -> float:
    """Return DIC tendency in umol kg-1 day-1 from air-sea gas exchange."""
    dic_umol_per_kg = float(state.get("dic_umol_per_kg", PARAMS.dic_umol_per_kg))
    k0 = float(state.get("k0", 0.0))
    d_dic_dt, _ = dic_tendency_from_air_sea(
        DIC=dic_umol_per_kg,
        T=float(state.get("temperature_c", PARAMS.temperature_c)),
        S=float(state.get("salinity_psu", PARAMS.salinity_psu)),
        U10=float(state.get("wind_speed_m_per_s", PARAMS.wind_speed_m_per_s)),
        pCO2_air=float(state.get("atmospheric_pco2_uatm", PARAMS.atmospheric_pco2_uatm)),
        h=float(state.get("mixed_layer_depth_m", PARAMS.mixed_layer_depth_m)),
        K0=k0,
    )
    return d_dic_dt * 86400.0
