"""
Air–Sea CO₂ Exchange Model
--------------------------

This script simulates the evolution of ocean surface CO₂ over multiple years
under the influence of:
- Gas transfer (Wanninkhof 1992)
- CO₂ solubility (Weiss 1974)
- Optional seasonal temperature forcing
- Air–sea CO₂ flux feedback on ocean CO₂

It solves a 1-box mixed-layer ODE for dissolved CO₂ concentration C(t)
and produces:
1. Ocean pCO₂ (diagnosed from C)
2. Air–sea CO₂ flux
3. Temperature anomaly

---------------------------------------------------------------------------
Parameter List (with units)
---------------------------------------------------------------------------
S              : Salinity (PSU)
U10            : Wind speed at 10 m (m/s)
h              : Mixed-layer depth (m)
pCO2_air       : Atmospheric pCO₂ (µatm)
pCO2_w_init    : Initial ocean pCO₂ (µatm)
C              : Dissolved CO₂ concentration (mol m⁻³)  [state variable]
T_min, T_max   : Seasonal temperature range (°C)
t_end          : Simulation length (s)
t_eval         : Output sampling interval (s)
---------------------------------------------------------------------------
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Physical parameterizations
# ============================================================
def solubility_co2(T, S):
    """
    Weiss (1974) CO₂ solubility in seawater.
    T : temperature (°C)
    S : salinity (PSU)
    Returns: K0 (mol m⁻³ µatm⁻¹)
    """
    Tk = T + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578

    lnK0 = (
        A1
        + A2 * (100 / Tk)
        + A3 * np.log(Tk / 100)
        + S * (B1 + B2 * (Tk / 100) + B3 * (Tk / 100) ** 2)
    )
    return np.exp(lnK0)


def k_wanninkhof(U10, T):
    """
    Wanninkhof (1992) gas transfer velocity.
    U10 : wind speed (m/s)
    T   : temperature (°C)
    Returns: k (m/s)
    """
    Sc = 2073.1 - 125.62 * T + 3.6276 * T**2 - 0.043219 * T**3  # Schmidt number
    k_cm_hr = 0.31 * U10**2 * (Sc / 660) ** (-0.5)
    return k_cm_hr / 100 / 3600  # convert cm/hr → m/s


# ============================================================
# 2. Seasonal temperature forcing
# ============================================================
def seasonal_t_seconds(t, T_min=2, T_max=20, seasonality=True):
    """
    Seasonal temperature cycle.
    t : time (s) (scalar or array)
    Returns: T(t) in °C
    """
    t = np.atleast_1d(t)

    P = 365 * 24 * 3600  # seconds per year
    T_mean = 0.5 * (T_min + T_max)

    if not seasonality:
        return np.full_like(t, T_mean, dtype=float)

    A = 0.5 * (T_max - T_min)
    return T_mean + A * np.sin(2 * np.pi * t / P)


# ============================================================
# 3. ODE: time evolution of dissolved CO₂ concentration
# ============================================================
def rhs(t, C, S, U10, pCO2_air, h):
    """
    RHS of dC/dt.
    t : time (s)
    C : dissolved CO₂ concentration (mol m⁻³)
    Returns: dC/dt (mol m⁻³ s⁻¹)
    """
    T = seasonal_t_seconds(t)[0]      # °C (scalar)
    k = k_wanninkhof(U10, T)          # m/s
    K0 = solubility_co2(T, S)         # mol m⁻³ µatm⁻¹

    C_eq = K0 * pCO2_air              # mol m⁻³ (equilibrium conc.)
    F = k * (C - C_eq)                # mol m⁻² s⁻¹ (ocean → atmosphere)

    dCdt = -F / h                     # mol m⁻³ s⁻¹ (tendency in mixed layer)
    return dCdt


# ============================================================
# 4. Model parameters (with units)
# ============================================================
S = 30                # PSU
U10 = 6               # m/s
h = 50                # m
pCO2_air = 420        # µatm
pCO2_w_init = 300     # µatm

years = 5
t_end = years * 365 * 24 * 3600     # s
t_span = (0, t_end)
t_eval = np.arange(0, t_end, 24 * 3600)  # s (daily)

# Initial concentration from initial pCO₂ at t=0
T0 = seasonal_t_seconds(0.0)[0]     # °C
K0_0 = solubility_co2(T0, S)        # mol m⁻³ µatm⁻¹
C_init = K0_0 * pCO2_w_init         # mol m⁻³


# ============================================================
# 5. Solve ODE
# ============================================================
sol = solve_ivp(rhs, t_span, [C_init], args=(S, U10, pCO2_air, h), t_eval=t_eval)

C_t = sol.y[0]                   # mol m⁻³
T_t = seasonal_t_seconds(sol.t)  # °C

# Time in days for plotting
time_days = t_eval / (24 * 3600)


# ============================================================
# 6. Plotting: Dissolved CO₂ and Temperature
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- 1. Dissolved CO₂ concentration ---
axes[0].plot(time_days, C_t, color="tab:blue", label="Dissolved CO₂")
axes[0].set_ylabel("C (mol m⁻³)")
axes[0].set_title("Dissolved CO₂ Concentration")
axes[0].grid(True)
axes[0].legend()

# --- 2. Temperature ---
axes[1].plot(time_days, T_t, color="tab:orange", label="Temperature")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_xlabel("Time (days)")
axes[1].set_title("Temperature")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
