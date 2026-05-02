from __future__ import annotations

# ============================================================
# FOUR-BOX DIC MODEL WITH COPERNICUS + WIND FORCING
# ============================================================
#
# Boxes:
#   Box 1 = shelf surface (0-5 m)
#   Box 2 = shelf bottom  (5-100 m)
#   Box 3 = ocean surface (0-5 m)
#   Box 4 = ocean deep    (5-100 m)
#
# Main idea:
#   Surface pCO2 is affected by air-sea exchange, biology, vertical/horizontal
#   transport, and deep open-boundary flushing/export.
#
# Main changes in this version:
#   1) TA0_UMOLKG is NOT calibrated anymore.
#   2) BIO_NCP_MULTIPLIER is calibrated again because the surface pCO2 was
#      probably being pulled too low by too strong prescribed DIC uptake.
#   3) Deep export cannot be tuned to exactly zero anymore.
#   4) Deep boundary DIC defaults/ranges are lowered so deep export can actually
#      remove carbon when DIC_box > DIC_boundary.
#   5) Added separate shelf/ocean monthly ΔpCO2 metrics to diagnose which
#      region causes the bad fit.
#   6) Added SURFACE_DEEP_CONNECTIVITY_MULTIPLIER so surface water can more
#      strongly communicate with bottom/deep boxes.
#   7) Replaced np.trapz with a safe helper because some numpy versions no
#      longer expose np.trapz.
#
# Requirements:
#   pip install numpy pandas xarray matplotlib copernicusmarine cartopy requests gsw
# ============================================================

from pathlib import Path
from typing import Dict, Tuple
import json
import hashlib
import importlib.util
import pprint

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine
import requests
import gsw

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# 1) USER SETTINGS
# ============================================================

def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


BASE_DIR = get_base_dir()

DATA_DIR = BASE_DIR / "copernicus_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WIND_DIR = BASE_DIR / "open_meteo_wind"
WIND_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = BASE_DIR / "four_box_dic_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CARBON_TS_DIR = BASE_DIR / "carbon_timeseries"
RECAD_NPZ_PATH = CARBON_TS_DIR / "ReCAD_pco2_transect_timeseries.npz"

USE_PARAMETER_FILE = True
SAVE_PARAMETERS_AFTER_CALIBRATION = True
PARAMETER_FILE = BASE_DIR / "four_box_parameters.py"


# ------------------------------------------------------------
# SIMULATION / DOWNLOAD PERIOD
# ------------------------------------------------------------

START_DATE = "2000-01-01"
END_DATE = "2002-12-31"

START_DATETIME = f"{START_DATE}T00:00:00"
END_DATETIME = f"{END_DATE}T00:00:00"


# ------------------------------------------------------------
# COPERNICUS SPATIAL REGION
# ------------------------------------------------------------

LON_MIN = -76.2
LON_MAX = -71.0

LAT_MIN = 36.4
LAT_MAX = 37.6

LON_SPLIT = -74.5


# ------------------------------------------------------------
# WIND POINT
# ------------------------------------------------------------

WIND_LAT = (LAT_MAX + LAT_MIN) / 2
WIND_LON = LON_SPLIT


# ------------------------------------------------------------
# Copernicus dataset / variables
# ------------------------------------------------------------

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

THETAO_VAR = "thetao"
SO_VAR = "so"

DOWNLOAD_MIN_DEPTH = 0.49402499198913574
DOWNLOAD_MAX_DEPTH = 100.0

SURFACE_MIN_DEPTH = 0.0
SURFACE_MAX_DEPTH = 5.0

BOTTOM_MIN_DEPTH = 5.0
BOTTOM_MAX_DEPTH = 100.0


# ------------------------------------------------------------
# Cache control
# ------------------------------------------------------------

REDOWNLOAD_RAW_COPERNICUS = False
REBUILD_FORCING_CACHE = False
REDOWNLOAD_WIND = False


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

SHELF_AREA_M2 = 1.5e10
OCEAN_AREA_M2 = 3.0e10

SURFACE_THICKNESS_M = 5.0
DEEP_THICKNESS_M = 95.0


# ------------------------------------------------------------
# Initial DIC
# ------------------------------------------------------------

INITIAL_DIC_FALLBACK = np.array([
    2050.0,
    2140.0,
    2050.0,
    2200.0,
], dtype=float)

USE_RECAD_INITIAL_SURFACE_DIC = True

SHELF_BOTTOM_DIC_OFFSET = 30.0
OCEAN_DEEP_DIC_OFFSET = 70.0


# ------------------------------------------------------------
# Air-sea CO2 settings
# ------------------------------------------------------------

PCO2_AIR_UATM = 420.0
AIRSEA_K_MULTIPLIER = 0.60


# ------------------------------------------------------------
# Dynamic deep export / open-boundary flushing settings
# ------------------------------------------------------------

DEEP_EXPORT_ON = True

# Keep this non-zero. If calibration sets it to zero, export is disabled.
DEEP_EXPORT_BASE_D1 = 0.0020
DEEP_EXPORT_MAX_D1 = 0.0300

# Lower defaults than before. Export removes carbon only when DIC_box > boundary_DIC.
SHELF_BOTTOM_BOUNDARY_DIC = 2000.0
OCEAN_DEEP_BOUNDARY_DIC = 2100.0

SHELF_BOTTOM_EXPORT_FRACTION = 0.35

DEEP_EXPORT_DIC_EXCESS_REF = 100.0
DEEP_EXPORT_DIC_EXCESS_ALPHA = 1.50

DEEP_EXPORT_DENSITY_REF = 0.25
DEEP_EXPORT_DENSITY_ALPHA = 0.75

DEEP_EXPORT_SEASONAL_AMPLITUDE = 0.35
DEEP_EXPORT_SEASONAL_PEAK_DAY = 70.0

# Old name kept only so old parameter files do not crash.
BOX4_EXPORT_DIC_MMOL_M3_D = 0.0


# ------------------------------------------------------------
# Density / DIC-density settings
# ------------------------------------------------------------

MOLAR_MASS_C_KG_PER_MOL = 12.0107e-3
DIC_DENSITY_CORRECTION_ON = True


# ------------------------------------------------------------
# Density-driven exchange tuning
# ------------------------------------------------------------

EXCHANGE_BASE_RATE_D1 = {
    (0, 1): 0.00008,
    (0, 2): 0.00012,
    (1, 3): 0.00008,
    (2, 3): 0.00004,
}

RHO_THRESHOLD = 0.00
RHO_FRICTION = 0.80

STABLE_VERTICAL_FACTOR = 0.03
UNSTABLE_VERTICAL_MULTIPLIER = 2.0

VERTICAL_PAIR_DEPTH_DIFF_M = 10.0
MAX_EXCHANGE_FRACTION_D1 = 0.0002


# ------------------------------------------------------------
# Extra connectivity tuning
# ------------------------------------------------------------
# This parameter helps surface water reach deeper boxes.
# It multiplies the exchange of:
#   Box 1 <-> Box 2
#   Box 3 <-> Box 4
#   Box 2 <-> Box 4
# ------------------------------------------------------------

SURFACE_DEEP_CONNECTIVITY_MULTIPLIER = 1.0


# ------------------------------------------------------------
# Background mixing tuning
# ------------------------------------------------------------

BACKGROUND_MIXING_BASE_D1 = 0.00002

WIND_MIXING_REF_M_S = 8.0
WIND_MIXING_ALPHA = 1.0

DIC_DIFF_REF_MMOL_M3 = 100.0
DIC_DIFF_FACTOR_MIN = 0.0
DIC_DIFF_FACTOR_MAX = 3.0

MAX_TOTAL_EXCHANGE_FRACTION_D1 = 0.0005


# ------------------------------------------------------------
# Biological DIC drawdown
# ------------------------------------------------------------

BIO_PUMP_ON = True

# Start lower than before. This is now calibrated, because too strong direct
# surface DIC uptake can make surface pCO2 unrealistically low.
BIO_NCP_MULTIPLIER = 0.5

SHELF_NCP_MAX_MMOL_M3_D = 0.24
OCEAN_NCP_MAX_MMOL_M3_D = 0.16

SPRING_BLOOM_DAY = 135.0
SPRING_BLOOM_WIDTH = 45.0

SUMMER_BROAD_DAY = 205.0
SUMMER_BROAD_WIDTH = 85.0

FALL_BLOOM_DAY = 275.0
FALL_BLOOM_WIDTH = 35.0

SUMMER_WEIGHT = 0.30
FALL_WEIGHT = 0.25

BIO_REMIN_FRACTION_TO_LOWER_BOX = 0.45


# ------------------------------------------------------------
# Carbonate settings
# ------------------------------------------------------------
# NOTE:
# TA0_UMOLKG is deliberately NOT calibrated in this version.
# Change it manually in four_box_parameters.py if needed.
# ------------------------------------------------------------

TA0_UMOLKG = 2350.0
S0 = 35.0

PH_MIN = 6.5
PH_MAX = 9.2


# ============================================================
# AUTOMATIC ITERATIVE CALIBRATION SETTINGS
# ============================================================

CALIBRATE_TO_RECAD = True

TUNING_MAX_ITERATIONS = 100
TUNING_TARGET_R_VALUE = 0.80

TUNING_START_PERCENT = 0.40
TUNING_STEP_DECAY = 0.90
TUNING_MIN_PERCENT = 0.01

# RMSE/bias are kept important because your previous run had okay bias but bad RMSE.
TUNING_WEIGHT_R = 0.40
TUNING_WEIGHT_RMSE = 0.80
TUNING_WEIGHT_BIAS = 0.40

TUNING_STOP_AFTER_NO_IMPROVEMENT_CYCLES = 2
TUNING_MIN_R_IMPROVEMENT_PER_CYCLE = 0.000
TUNING_R_EQUAL_TOLERANCE = 1e-4

# Print options:
#   "r_only" = only print r/RMSE/bias after each step
#   "chosen" = print parameter + metrics
#   "silent" = no per-step output
# Recommended while debugging: "chosen"
TUNING_PRINT_MODE = "chosen"

SAVE_FIGURES = False
PLOT_ONLY_SOCAT_COMPARISON = True
SHOW_CALIBRATION_PROGRESS = True


# ============================================================
# PARAMETER FILE HELPERS
# ============================================================

PARAMETER_NAMES_FOR_FILE = [
    "INITIAL_DIC_FALLBACK",
    "USE_RECAD_INITIAL_SURFACE_DIC",
    "SHELF_BOTTOM_DIC_OFFSET",
    "OCEAN_DEEP_DIC_OFFSET",

    "PCO2_AIR_UATM",
    "AIRSEA_K_MULTIPLIER",

    "BOX4_EXPORT_DIC_MMOL_M3_D",

    "DEEP_EXPORT_ON",
    "DEEP_EXPORT_BASE_D1",
    "DEEP_EXPORT_MAX_D1",
    "SHELF_BOTTOM_BOUNDARY_DIC",
    "OCEAN_DEEP_BOUNDARY_DIC",
    "SHELF_BOTTOM_EXPORT_FRACTION",
    "DEEP_EXPORT_DIC_EXCESS_REF",
    "DEEP_EXPORT_DIC_EXCESS_ALPHA",
    "DEEP_EXPORT_DENSITY_REF",
    "DEEP_EXPORT_DENSITY_ALPHA",
    "DEEP_EXPORT_SEASONAL_AMPLITUDE",
    "DEEP_EXPORT_SEASONAL_PEAK_DAY",

    "DIC_DENSITY_CORRECTION_ON",

    "EXCHANGE_BASE_RATE_D1",
    "RHO_THRESHOLD",
    "RHO_FRICTION",
    "STABLE_VERTICAL_FACTOR",
    "UNSTABLE_VERTICAL_MULTIPLIER",
    "MAX_EXCHANGE_FRACTION_D1",

    "SURFACE_DEEP_CONNECTIVITY_MULTIPLIER",

    "BACKGROUND_MIXING_BASE_D1",
    "WIND_MIXING_REF_M_S",
    "WIND_MIXING_ALPHA",
    "DIC_DIFF_REF_MMOL_M3",
    "DIC_DIFF_FACTOR_MIN",
    "DIC_DIFF_FACTOR_MAX",
    "MAX_TOTAL_EXCHANGE_FRACTION_D1",

    "BIO_PUMP_ON",
    "BIO_NCP_MULTIPLIER",
    "SHELF_NCP_MAX_MMOL_M3_D",
    "OCEAN_NCP_MAX_MMOL_M3_D",
    "SPRING_BLOOM_DAY",
    "SPRING_BLOOM_WIDTH",
    "SUMMER_BROAD_DAY",
    "SUMMER_BROAD_WIDTH",
    "FALL_BLOOM_DAY",
    "FALL_BLOOM_WIDTH",
    "SUMMER_WEIGHT",
    "FALL_WEIGHT",
    "BIO_REMIN_FRACTION_TO_LOWER_BOX",

    "TA0_UMOLKG",
    "S0",
    "PH_MIN",
    "PH_MAX",

    "CALIBRATE_TO_RECAD",
    "TUNING_MAX_ITERATIONS",
    "TUNING_TARGET_R_VALUE",
    "TUNING_START_PERCENT",
    "TUNING_STEP_DECAY",
    "TUNING_MIN_PERCENT",
    "TUNING_WEIGHT_R",
    "TUNING_WEIGHT_RMSE",
    "TUNING_WEIGHT_BIAS",
    "TUNING_STOP_AFTER_NO_IMPROVEMENT_CYCLES",
    "TUNING_MIN_R_IMPROVEMENT_PER_CYCLE",
    "TUNING_R_EQUAL_TOLERANCE",
    "TUNING_PRINT_MODE",

    "SAVE_FIGURES",
    "PLOT_ONLY_SOCAT_COMPARISON",
    "SHOW_CALIBRATION_PROGRESS",
]


def _python_parameter_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, dict):
        return {
            _parameter_file_key(k): _python_parameter_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_python_parameter_value(v) for v in value]

    return value


def _tuple_key_from_parameter_file(key):
    if isinstance(key, tuple):
        return key

    if isinstance(key, str):
        stripped = key.strip()

        if stripped.startswith("(") and stripped.endswith(")"):
            parts = stripped.strip("()").split(",")

            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))

    return key


def _parameter_file_key(key):
    if isinstance(key, tuple):
        return str(key)

    return key


def _normalise_loaded_parameter(name: str, value):
    if name == "INITIAL_DIC_FALLBACK":
        return np.asarray(value, dtype=float)

    if name == "EXCHANGE_BASE_RATE_D1":
        return {
            _tuple_key_from_parameter_file(k): float(v)
            for k, v in dict(value).items()
        }

    return value


def _load_parameters_dict_from_py(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("four_box_parameters", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load parameter file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PARAMETERS"):
        raise AttributeError(
            f"Parameter file must define a PARAMETERS dictionary: {path}"
        )

    params = getattr(module, "PARAMETERS")

    if not isinstance(params, dict):
        raise TypeError("PARAMETERS must be a dictionary.")

    return params


def load_parameter_file_if_needed() -> None:
    if not USE_PARAMETER_FILE:
        return

    if not PARAMETER_FILE.exists():
        save_parameter_file()
        print(f"\nCreated default parameter file:\n{PARAMETER_FILE}")
        return

    data = _load_parameters_dict_from_py(PARAMETER_FILE)

    if "BOX4_EXPORT_DIC_MMOL_M3_D" in data and "DEEP_EXPORT_BASE_D1" not in data:
        old_sink = float(data["BOX4_EXPORT_DIC_MMOL_M3_D"])
        data["DEEP_EXPORT_BASE_D1"] = min(max(old_sink, 0.0002), 0.03)

    updated = []

    for name in PARAMETER_NAMES_FOR_FILE:
        if name in data:
            globals()[name] = _normalise_loaded_parameter(name, data[name])
            updated.append(name)

    print(f"\nLoaded {len(updated)} parameters from:\n{PARAMETER_FILE}")


def current_parameters_dict() -> dict:
    out = {}

    for name in PARAMETER_NAMES_FOR_FILE:
        if name in globals():
            out[name] = _python_parameter_value(globals()[name])

    return out


def save_parameter_file() -> None:
    data = current_parameters_dict()

    header = (
        "# ============================================================\n"
        "# FOUR-BOX DIC MODEL PARAMETERS\n"
        "# This file is automatically read by the main model script.\n"
        "# After calibration, calibrated values are written back here.\n"
        "# ============================================================\n\n"
    )

    with open(PARAMETER_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("PARAMETERS = ")
        f.write(pprint.pformat(data, sort_dicts=False, width=120))
        f.write("\n")

    print(f"\nSaved parameter file:\n{PARAMETER_FILE}")


# ============================================================
# 2) HASHED CACHE HELPERS
# ============================================================

def stable_hash(obj: dict, n: int = 12) -> str:
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def forcing_cache_key() -> str:
    obj = {
        "dataset_id": DATASET_ID,
        "thetao_var": THETAO_VAR,
        "so_var": SO_VAR,
        "start_datetime": START_DATETIME,
        "end_datetime": END_DATETIME,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_split": LON_SPLIT,
        "download_min_depth": DOWNLOAD_MIN_DEPTH,
        "download_max_depth": DOWNLOAD_MAX_DEPTH,
        "surface_min_depth": SURFACE_MIN_DEPTH,
        "surface_max_depth": SURFACE_MAX_DEPTH,
        "bottom_min_depth": BOTTOM_MIN_DEPTH,
        "bottom_max_depth": BOTTOM_MAX_DEPTH,
    }
    return stable_hash(obj)


def raw_copernicus_key(variable: str) -> str:
    obj = {
        "dataset_id": DATASET_ID,
        "variable": variable,
        "start_datetime": START_DATETIME,
        "end_datetime": END_DATETIME,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "download_min_depth": DOWNLOAD_MIN_DEPTH,
        "download_max_depth": DOWNLOAD_MAX_DEPTH,
    }
    return stable_hash(obj)


def wind_cache_key() -> str:
    obj = {
        "source": "open_meteo_archive",
        "lat": WIND_LAT,
        "lon": WIND_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "variable": "wind_speed_10m",
        "unit": "ms",
        "timezone": "UTC",
    }
    return stable_hash(obj)


# ============================================================
# 3) BOX DEFINITIONS
# ============================================================

BOX_NAMES = [
    "Box 1: Shelf surface",
    "Box 2: Shelf bottom",
    "Box 3: Ocean surface",
    "Box 4: Ocean deep",
]

BOX_SHORT = ["B1", "B2", "B3", "B4"]

BOX_LAYOUT = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (1, 1),
}

PAIR_LIST = [(0, 1), (0, 2), (1, 3), (2, 3)]

PAIR_LABELS = {
    (0, 1): "Box 1 ↔ Box 2",
    (0, 2): "Box 1 ↔ Box 3",
    (1, 3): "Box 2 ↔ Box 4",
    (2, 3): "Box 3 ↔ Box 4",
}

BOX_AREAS = np.array([
    SHELF_AREA_M2,
    SHELF_AREA_M2,
    OCEAN_AREA_M2,
    OCEAN_AREA_M2,
], dtype=float)

BOX_THICKNESSES = np.array([
    SURFACE_THICKNESS_M,
    DEEP_THICKNESS_M,
    SURFACE_THICKNESS_M,
    DEEP_THICKNESS_M,
], dtype=float)

BOX_MID_DEPTHS = np.array([2.5, 52.5, 2.5, 52.5], dtype=float)
BOX_VOLUMES = BOX_AREAS * BOX_THICKNESSES

BOX_LONS = np.array([
    0.5 * (LON_MIN + LON_SPLIT),
    0.5 * (LON_MIN + LON_SPLIT),
    0.5 * (LON_SPLIT + LON_MAX),
    0.5 * (LON_SPLIT + LON_MAX),
], dtype=float)

BOX_LATS = np.array([
    0.5 * (LAT_MIN + LAT_MAX),
    0.5 * (LAT_MIN + LAT_MAX),
    0.5 * (LAT_MIN + LAT_MAX),
    0.5 * (LAT_MIN + LAT_MAX),
], dtype=float)


# ============================================================
# 4) BASIC PHYSICS / CARBON CHEMISTRY
# ============================================================

def integrate_daily(values: np.ndarray) -> float:
    """
    Safe replacement for np.trapz.

    This assumes one value per day and integrates over days.
    """
    values = np.asarray(values, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, dx=1.0))

    if len(values) <= 1:
        return 0.0

    return float(np.sum(0.5 * (values[1:] + values[:-1])))


def pressure_dbar_from_depth(depth_m: np.ndarray, lat: np.ndarray) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return gsw.p_from_z(-depth_m, lat)


def density_teos10_with_dic(temp_c: np.ndarray, sal_psu: np.ndarray, dic_mmol_m3: np.ndarray) -> np.ndarray:
    temp_c = np.asarray(temp_c, dtype=float)
    sal_psu = np.asarray(sal_psu, dtype=float)
    dic_mmol_m3 = np.asarray(dic_mmol_m3, dtype=float)

    p_dbar = pressure_dbar_from_depth(BOX_MID_DEPTHS, BOX_LATS)
    SA = gsw.SA_from_SP(sal_psu, p_dbar, BOX_LONS, BOX_LATS)
    CT = gsw.CT_from_pt(SA, temp_c)
    rho_base = gsw.rho(SA, CT, p_dbar)

    if DIC_DENSITY_CORRECTION_ON:
        rho_dic = dic_mmol_m3 * MOLAR_MASS_C_KG_PER_MOL / 1000.0
        return rho_base + rho_dic

    return rho_base


def pressure_pa(rho: np.ndarray, mid_depth_m: np.ndarray) -> np.ndarray:
    return np.asarray(rho) * 9.81 * np.asarray(mid_depth_m)


def schmidt_number_co2(temp_c: np.ndarray) -> np.ndarray:
    T = np.asarray(temp_c, dtype=float)
    Sc = 2116.8 - 136.25 * T + 4.7353 * T**2 - 0.092307 * T**3 + 0.0007555 * T**4
    return np.maximum(Sc, 1.0)


def gas_transfer_velocity_m_per_day(wind_m_s: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    wind_m_s = np.maximum(np.asarray(wind_m_s, dtype=float), 0.0)
    Sc = schmidt_number_co2(temp_c)
    k_cm_h = 0.251 * wind_m_s**2 * (Sc / 660.0) ** (-0.5)
    k_m_d = k_cm_h * 0.01 * 24.0
    return AIRSEA_K_MULTIPLIER * k_m_d


def carbonate_constants(temp_c: np.ndarray, sal: np.ndarray) -> Dict[str, np.ndarray]:
    T = np.asarray(temp_c, dtype=float) + 273.15
    S = np.maximum(np.asarray(sal, dtype=float), 0.1)

    lnK0 = (
        -58.0931
        + 90.5069 * (100.0 / T)
        + 22.2940 * np.log(T / 100.0)
        + S * (0.027766 - 0.025888 * (T / 100.0) + 0.0050578 * (T / 100.0) ** 2)
    )
    K0 = np.exp(lnK0)

    pK1 = 3633.86 / T - 61.2172 + 9.67770 * np.log(T) - 0.011555 * S + 0.0001152 * S * S
    pK2 = 471.78 / T + 25.9290 - 3.16967 * np.log(T) - 0.01781 * S + 0.0001122 * S * S
    K1 = 10.0 ** (-pK1)
    K2 = 10.0 ** (-pK2)

    sqrtS = np.sqrt(S)
    lnKb = (
        (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S * sqrtS - 0.0996 * S * S) / T
        + 148.0248
        + 137.1942 * sqrtS
        + 1.62142 * S
        + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * np.log(T)
        + 0.053105 * sqrtS * T
    )
    Kb = np.exp(lnKb)
    BT = 0.0004157 * S / 35.0

    lnKw = (
        148.96502
        - 13847.26 / T
        - 23.6521 * np.log(T)
        + (118.67 / T - 5.977 + 1.0495 * np.log(T)) * np.sqrt(S)
        - 0.01615 * S
    )
    Kw = np.exp(lnKw)

    return {"K0": K0, "K1": K1, "K2": K2, "Kb": Kb, "Kw": Kw, "BT": BT}


def carbonate_speciation(
    dic_mmol_m3: np.ndarray,
    temp_c: np.ndarray,
    sal: np.ndarray,
    rho_kg_m3: np.ndarray,
) -> Dict[str, np.ndarray]:
    dic_mmol_m3 = np.maximum(np.asarray(dic_mmol_m3, dtype=float), 1e-9)
    sal = np.asarray(sal, dtype=float)
    rho_kg_m3 = np.asarray(rho_kg_m3, dtype=float)

    dic_umolkg = dic_mmol_m3 * 1000.0 / np.maximum(rho_kg_m3, 1e-12)
    TA_umolkg = TA0_UMOLKG * sal / max(S0, 1e-12)

    CT = dic_umolkg * 1e-6
    TA = TA_umolkg * 1e-6

    const = carbonate_constants(temp_c, sal)
    K0 = const["K0"]
    K1 = const["K1"]
    K2 = const["K2"]
    Kb = const["Kb"]
    Kw = const["Kw"]
    BT = const["BT"]

    def alk_from_H(H):
        denom = H * H + K1 * H + K1 * K2
        HCO3 = CT * (K1 * H / denom)
        CO3 = CT * (K1 * K2 / denom)
        BOH4 = BT * Kb / (Kb + H)
        OH = Kw / H
        return HCO3 + 2.0 * CO3 + BOH4 + OH - H

    lo = np.full_like(CT, PH_MIN, dtype=float)
    hi = np.full_like(CT, PH_MAX, dtype=float)

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        Hmid = 10.0 ** (-mid)
        fmid = alk_from_H(Hmid) - TA
        hi = np.where(fmid > 0.0, mid, hi)
        lo = np.where(fmid <= 0.0, mid, lo)

    pH = 0.5 * (lo + hi)
    H = 10.0 ** (-pH)
    denom = H * H + K1 * H + K1 * K2

    alpha0 = H * H / denom
    alpha1 = K1 * H / denom
    alpha2 = K1 * K2 / denom

    CO2_molkg = CT * alpha0
    HCO3_molkg = CT * alpha1
    CO3_molkg = CT * alpha2

    pco2_uatm = CO2_molkg / np.maximum(K0, 1e-30) * 1e6
    pco2_air_atm = PCO2_AIR_UATM * 1e-6
    CO2_eq_air_molkg = K0 * pco2_air_atm

    CO2_mmol_m3 = CO2_molkg * rho_kg_m3 * 1000.0
    CO2_eq_air_mmol_m3 = CO2_eq_air_molkg * rho_kg_m3 * 1000.0

    return {
        "DIC_umolkg": dic_umolkg,
        "TA_umolkg": TA_umolkg,
        "pH": pH,
        "pCO2_uatm": np.clip(pco2_uatm, 1.0, 10000.0),
        "CO2_mmol_m3": CO2_mmol_m3,
        "CO2_eq_air_mmol_m3": CO2_eq_air_mmol_m3,
        "HCO3_umolkg": HCO3_molkg * 1e6,
        "CO3_umolkg": CO3_molkg * 1e6,
    }


def pco2_from_dic_box(box_index: int, dic_value: float, temp_value: float, sal_value: float) -> float:
    dic = np.full(4, 2050.0, dtype=float)
    temp = np.full(4, temp_value, dtype=float)
    sal = np.full(4, sal_value, dtype=float)
    dic[box_index] = dic_value

    rho = density_teos10_with_dic(temp_c=temp, sal_psu=sal, dic_mmol_m3=dic)
    carb = carbonate_speciation(dic_mmol_m3=dic, temp_c=temp, sal=sal, rho_kg_m3=rho)
    return float(carb["pCO2_uatm"][box_index])


def solve_dic_for_target_pco2(
    box_index: int,
    target_pco2: float,
    temp_value: float,
    sal_value: float,
    dic_min: float = 1500.0,
    dic_max: float = 2800.0,
) -> float:
    lo = dic_min
    hi = dic_max

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p_mid = pco2_from_dic_box(box_index, mid, temp_value, sal_value)
        if p_mid > target_pco2:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# ============================================================
# 5) WIND DOWNLOAD / PROCESSING
# ============================================================

def download_open_meteo_wind_if_needed() -> Path:
    key = wind_cache_key()
    out_file = WIND_DIR / f"open_meteo_wind_10m_{START_DATE}_to_{END_DATE}_{key}.csv"

    if out_file.exists() and not REDOWNLOAD_WIND:
        print(f"\nUsing existing Open-Meteo wind file:\n{out_file}")
        return out_file

    if out_file.exists() and REDOWNLOAD_WIND:
        print(f"\nRemoving old wind file:\n{out_file}")
        out_file.unlink()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": WIND_LAT,
        "longitude": WIND_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    print("\nDownloading Open-Meteo wind data...")
    print(f"Location: lat={WIND_LAT}, lon={WIND_LON}")
    print(f"Period:   {START_DATE} to {END_DATE}")

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        print("\nOpen-Meteo response:")
        print(data)
        raise KeyError("Open-Meteo response does not contain 'hourly'.")

    hourly = data["hourly"]
    time_index = pd.to_datetime(hourly["time"], utc=True)
    if getattr(time_index, "tz", None) is not None:
        time_index = time_index.tz_convert(None)

    df = pd.DataFrame({
        "time": time_index,
        "latitude": WIND_LAT,
        "longitude": WIND_LON,
        "wind_speed_10m_m_s": hourly["wind_speed_10m"],
    })

    df = df[["time", "latitude", "longitude", "wind_speed_10m_m_s"]]
    df.to_csv(out_file, index=False)

    print(f"Saved wind file:\n{out_file}")
    print(df.head())
    print(df.tail())
    return out_file


def load_wind_for_model_time(model_time: np.ndarray) -> np.ndarray:
    wind_path = download_open_meteo_wind_if_needed()
    df = pd.read_csv(wind_path, parse_dates=["time"])

    wind_time = df["time"].values.astype("datetime64[ns]")
    wind_values = df["wind_speed_10m_m_s"].values.astype(float)

    model_time = np.asarray(model_time, dtype="datetime64[ns]")
    x_wind = wind_time.astype("datetime64[s]").astype(float)
    x_model = model_time.astype("datetime64[s]").astype(float)

    wind_interp = np.interp(x_model, x_wind, wind_values, left=wind_values[0], right=wind_values[-1])

    print("\nWind forcing diagnostics:")
    print("  model time range:", model_time[0], "to", model_time[-1])
    print("  wind file range: ", wind_time[0], "to", wind_time[-1])
    print("  wind min/max:", np.nanmin(wind_interp), np.nanmax(wind_interp))
    print("  wind mean:", np.nanmean(wind_interp))
    return wind_interp


# ============================================================
# 6) COPERNICUS DOWNLOAD / OPEN / PROCESS
# ============================================================

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}

    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"

    for possible_depth in ["depth", "deptht", "lev", "level"]:
        if possible_depth in ds.coords and possible_depth != "depth":
            rename_map[possible_depth] = "depth"
            break

    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def download_if_needed(variable: str, out_path: Path) -> None:
    if out_path.exists() and not REDOWNLOAD_RAW_COPERNICUS:
        print(f"\nUsing existing raw file:\n{out_path}")
        return

    if out_path.exists() and REDOWNLOAD_RAW_COPERNICUS:
        print(f"\nRemoving old raw file:\n{out_path}")
        out_path.unlink()

    print(f"\nDownloading {variable} to:\n{out_path}")

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=[variable],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=START_DATETIME,
        end_datetime=END_DATETIME,
        minimum_depth=DOWNLOAD_MIN_DEPTH,
        maximum_depth=DOWNLOAD_MAX_DEPTH,
        output_filename=out_path.name,
        output_directory=str(out_path.parent),
    )


def subset_lat_range(da: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    lat_vals = da["lat"].values
    if lat_vals[0] <= lat_vals[-1]:
        return da.sel(lat=slice(lat_min, lat_max))
    return da.sel(lat=slice(lat_max, lat_min))


def subset_lon_range(da: xr.DataArray, lon_min: float, lon_max: float) -> xr.DataArray:
    lon_vals = da["lon"].values
    if lon_vals[0] <= lon_vals[-1]:
        return da.sel(lon=slice(lon_min, lon_max))
    return da.sel(lon=slice(lon_max, lon_min))


def get_spatial_subset(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    ds = standardize_coords(ds)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise KeyError("Could not find lon/lat coordinates after standardization.")

    da = ds[var_name]
    da = subset_lat_range(da, LAT_MIN, LAT_MAX)
    da = subset_lon_range(da, LON_MIN, LON_MAX)
    return da


def depth_subset_with_fallback(da: xr.DataArray, depth_min: float, depth_max: float, preferred_name: str) -> xr.DataArray:
    if "depth" not in da.dims:
        print(f"\nWARNING: no depth dimension for {preferred_name}; using data as-is.")
        return da

    depth = da["depth"]
    selected = da.where((depth >= depth_min) & (depth <= depth_max), drop=True)

    if selected.sizes.get("depth", 0) > 0:
        print(f"{preferred_name}: using depth range {depth_min}-{depth_max} m with depths {selected['depth'].values}")
        return selected

    selected = da.where(depth >= depth_min, drop=True)
    if selected.sizes.get("depth", 0) > 0:
        print(
            f"\nWARNING: {preferred_name}: no depths inside {depth_min}-{depth_max} m. "
            f"Using available depths >= {depth_min} m instead: {selected['depth'].values}"
        )
        return selected

    print(f"\nWARNING: {preferred_name}: no valid depths found. Using all available depths.")
    return da


def safe_mean_box(sub_da: xr.DataArray, box_name: str) -> xr.DataArray:
    if sub_da.size == 0:
        raise ValueError(f"{box_name}: selected data is empty.")

    dims_to_mean = [d for d in ["depth", "lat", "lon"] if d in sub_da.dims]
    out = sub_da.mean(dim=dims_to_mean, skipna=True)

    if np.all(~np.isfinite(out.values)):
        raise ValueError(f"{box_name}: mean result is all NaN.")

    return out


def build_box_timeseries(da: xr.DataArray, var_label: str) -> xr.DataArray:
    if "depth" not in da.dims:
        raise ValueError(f"{var_label}: expected a depth dimension.")

    print(f"\nBuilding 4-box forcing for {var_label}")
    print("Available depths:", da["depth"].values)
    print("Latitude values:", da["lat"].values)
    print("Longitude range:", float(da["lon"].min()), "to", float(da["lon"].max()))

    shelf_da = subset_lon_range(da, LON_MIN, LON_SPLIT - 1e-6)
    ocean_da = subset_lon_range(da, LON_SPLIT, LON_MAX)

    if shelf_da.sizes.get("lon", 0) == 0:
        raise ValueError("Shelf region has zero longitude points.")
    if ocean_da.sizes.get("lon", 0) == 0:
        raise ValueError("Ocean region has zero longitude points.")

    shelf_surface = depth_subset_with_fallback(shelf_da, SURFACE_MIN_DEPTH, SURFACE_MAX_DEPTH, f"{var_label} Box 1 shelf surface")
    shelf_bottom = depth_subset_with_fallback(shelf_da, BOTTOM_MIN_DEPTH, BOTTOM_MAX_DEPTH, f"{var_label} Box 2 shelf bottom")
    ocean_surface = depth_subset_with_fallback(ocean_da, SURFACE_MIN_DEPTH, SURFACE_MAX_DEPTH, f"{var_label} Box 3 ocean surface")
    ocean_deep = depth_subset_with_fallback(ocean_da, BOTTOM_MIN_DEPTH, BOTTOM_MAX_DEPTH, f"{var_label} Box 4 ocean deep")

    box1 = safe_mean_box(shelf_surface, f"{var_label} Box 1")
    box2 = safe_mean_box(shelf_bottom, f"{var_label} Box 2")
    box3 = safe_mean_box(ocean_surface, f"{var_label} Box 3")
    box4 = safe_mean_box(ocean_deep, f"{var_label} Box 4")

    out = xr.concat([box1, box2, box3, box4], dim="box")
    out = out.transpose("time", "box")
    out = out.assign_coords(box=np.arange(4))
    return out


def validate_forcing(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    print(f"\n{name} forcing diagnostics:")
    print("  shape:", arr.shape)
    print("  total NaNs:", np.count_nonzero(~np.isfinite(arr)))
    print("  min:", np.nanmin(arr))
    print("  max:", np.nanmax(arr))

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} forcing still contains NaNs.")

    return arr


def plot_spatial_extent(lon_vals: np.ndarray, lat_vals: np.ndarray) -> None:
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    shelf_mask = lon2d < LON_SPLIT
    ocean_mask = lon2d >= LON_SPLIT

    fig = plt.figure(figsize=(11, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([LON_MIN - 0.3, LON_MAX + 0.3, LAT_MIN - 0.2, LAT_MAX + 0.2], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#bcd7ff", zorder=0)
    ax.coastlines(resolution="10m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.7)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    ax.scatter(lon2d[shelf_mask], lat2d[shelf_mask], s=25, color="tab:green", alpha=0.7, transform=ccrs.PlateCarree(), label="Shelf grid cells")
    ax.scatter(lon2d[ocean_mask], lat2d[ocean_mask], s=25, color="tab:orange", alpha=0.7, transform=ccrs.PlateCarree(), label="Ocean grid cells")
    ax.scatter([WIND_LON], [WIND_LAT], s=90, color="tab:red", marker="*", edgecolor="black", transform=ccrs.PlateCarree(), label="Open-Meteo wind point", zorder=5)
    ax.plot([LON_SPLIT, LON_SPLIT], [LAT_MIN, LAT_MAX], "k--", linewidth=2, transform=ccrs.PlateCarree(), label=f"Split at lon = {LON_SPLIT}")

    rect_lon = [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]
    rect_lat = [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]
    ax.plot(rect_lon, rect_lat, color="red", linewidth=2, transform=ccrs.PlateCarree(), label="Selected averaging box")

    ax.set_title("Spatial extent, Copernicus grid cells, and wind point")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(OUT_DIR / "spatial_extent_used_for_averaging.png", dpi=200, bbox_inches="tight")

    plt.show()


def load_or_build_forcing() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_force = forcing_cache_key()
    key_t = raw_copernicus_key(THETAO_VAR)
    key_s = raw_copernicus_key(SO_VAR)

    thetao_path = DATA_DIR / f"thetao_{key_t}.nc"
    so_path = DATA_DIR / f"so_{key_s}.nc"
    forcing_cache_path = DATA_DIR / f"four_box_forcing_{key_force}.npz"

    print("\nCache keys:")
    print("  thetao raw:", key_t)
    print("  so raw:    ", key_s)
    print("  forcing:   ", key_force)

    if forcing_cache_path.exists() and not REBUILD_FORCING_CACHE:
        print(f"\nUsing processed forcing cache:\n{forcing_cache_path}")
        with np.load(forcing_cache_path, allow_pickle=True) as data:
            time = data["time"].astype("datetime64[ns]")
            temp = data["temp"]
            sal = data["sal"]
            lon_vals = data["lon_vals"]
            lat_vals = data["lat_vals"]
        plot_spatial_extent(lon_vals, lat_vals)
        return time, temp, sal

    if forcing_cache_path.exists() and REBUILD_FORCING_CACHE:
        print(f"\nRemoving old forcing cache:\n{forcing_cache_path}")
        forcing_cache_path.unlink()

    download_if_needed(THETAO_VAR, thetao_path)
    download_if_needed(SO_VAR, so_path)

    ds_t = xr.open_dataset(thetao_path)
    ds_s = xr.open_dataset(so_path)

    print("\nOpened thetao dataset:")
    print(ds_t)
    print("\nOpened so dataset:")
    print(ds_s)

    da_t = get_spatial_subset(ds_t, THETAO_VAR)
    da_s = get_spatial_subset(ds_s, SO_VAR)

    lon_vals = da_t["lon"].values
    lat_vals = da_t["lat"].values

    print("\nSelected grid information:")
    print("Number of lon values:", len(lon_vals))
    print("Number of lat values:", len(lat_vals))
    print("Lon range:", float(np.min(lon_vals)), "to", float(np.max(lon_vals)))
    print("Lat range:", float(np.min(lat_vals)), "to", float(np.max(lat_vals)))

    plot_spatial_extent(lon_vals, lat_vals)

    t_box = build_box_timeseries(da_t, "thetao")
    s_box = build_box_timeseries(da_s, "so")
    t_box, s_box = xr.align(t_box, s_box, join="inner")

    time = t_box["time"].values.astype("datetime64[ns]")
    temp = validate_forcing(t_box.values.astype(float), "Temperature")
    sal = validate_forcing(s_box.values.astype(float), "Salinity")

    metadata = {
        "dataset_id": DATASET_ID,
        "start_datetime": START_DATETIME,
        "end_datetime": END_DATETIME,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_split": LON_SPLIT,
        "download_min_depth": DOWNLOAD_MIN_DEPTH,
        "download_max_depth": DOWNLOAD_MAX_DEPTH,
        "forcing_cache_key": key_force,
    }

    np.savez_compressed(
        forcing_cache_path,
        time=time,
        temp=temp,
        sal=sal,
        lon_vals=lon_vals,
        lat_vals=lat_vals,
        metadata=np.array(json.dumps(metadata)),
    )

    ds_t.close()
    ds_s.close()

    print(f"\nSaved processed forcing cache:\n{forcing_cache_path}")
    return time, temp, sal


# ============================================================
# 7) RECAD / SOCAT COMPARISON
# ============================================================

def load_recad_spatial_means() -> Dict[str, np.ndarray] | None:
    if not RECAD_NPZ_PATH.exists():
        print(f"\nReCAD file not found, skipping comparison:\n{RECAD_NPZ_PATH}")
        return None

    print(f"\nOpening ReCAD comparison file:\n{RECAD_NPZ_PATH}")

    with np.load(RECAD_NPZ_PATH, allow_pickle=True) as data:
        keys = list(data.keys())
        if "pco2" not in keys:
            raise KeyError(f"ReCAD file has no 'pco2'. Keys: {keys}")
        if "lon" not in keys:
            raise KeyError(f"ReCAD file has no 'lon'. Keys: {keys}")
        if "time_datetime64" not in keys:
            raise KeyError(f"ReCAD file has no 'time_datetime64'. Keys: {keys}")

        pco2 = np.asarray(data["pco2"], dtype=float)
        lon = np.asarray(data["lon"], dtype=float)
        time = np.asarray(data["time_datetime64"], dtype="datetime64[ns]")

    shelf_mask = lon < LON_SPLIT
    ocean_mask = lon >= LON_SPLIT

    if not np.any(shelf_mask):
        raise ValueError("No ReCAD shelf points found.")
    if not np.any(ocean_mask):
        raise ValueError("No ReCAD ocean points found.")

    shelf_mean = np.nanmean(pco2[:, shelf_mask], axis=1)
    ocean_mean = np.nanmean(pco2[:, ocean_mask], axis=1)

    t0 = np.datetime64(START_DATETIME, "ns")
    t1 = np.datetime64(END_DATETIME, "ns")
    time_mask = (time >= t0) & (time <= t1)

    shelf_sel = shelf_mean[time_mask]
    ocean_sel = ocean_mean[time_mask]
    time_sel = time[time_mask]

    print("\nReCAD comparison summary:")
    print("  total time points:", len(time))
    print("  selected time points:", np.count_nonzero(time_mask))
    print("  shelf points:", np.count_nonzero(shelf_mask))
    print("  ocean points:", np.count_nonzero(ocean_mask))

    return {
        "time": time_sel,
        "shelf_mean": shelf_sel,
        "ocean_mean": ocean_sel,
        "shelf_delta": PCO2_AIR_UATM - shelf_sel,
        "ocean_delta": PCO2_AIR_UATM - ocean_sel,
    }


def monthly_series(time: np.ndarray, values: np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(time))
    return s.resample("MS").mean()


def get_recad_monthly_delta(recad: Dict[str, np.ndarray] | None) -> Tuple[pd.Series | None, pd.Series | None]:
    if recad is None or len(recad["time"]) == 0:
        return None, None
    shelf = monthly_series(recad["time"], recad["shelf_delta"])
    ocean = monthly_series(recad["time"], recad["ocean_delta"])
    return shelf, ocean


# ============================================================
# 8) FOUR-BOX DIC MODEL
# ============================================================

def air_sea_dic_tendency_from_carb(carb: Dict[str, np.ndarray], box_index: int, k_m_d: float, thickness_m: float) -> float:
    co2_eq = carb["CO2_eq_air_mmol_m3"][box_index]
    co2_water = carb["CO2_mmol_m3"][box_index]
    return (k_m_d / max(thickness_m, 1e-12)) * (co2_eq - co2_water)


def is_vertical_pair(i: int, j: int) -> bool:
    return abs(BOX_MID_DEPTHS[i] - BOX_MID_DEPTHS[j]) >= VERTICAL_PAIR_DEPTH_DIFF_M


def gravitational_stability_factor(i: int, j: int, rho: np.ndarray) -> float:
    if not is_vertical_pair(i, j):
        return 1.0

    depth_i = BOX_MID_DEPTHS[i]
    depth_j = BOX_MID_DEPTHS[j]
    rho_i = rho[i]
    rho_j = rho[j]
    drho = rho_i - rho_j

    if abs(drho) < RHO_THRESHOLD:
        return 1.0

    if rho_i > rho_j:
        denser_depth = depth_i
        lighter_depth = depth_j
    else:
        denser_depth = depth_j
        lighter_depth = depth_i

    if denser_depth > lighter_depth:
        return STABLE_VERTICAL_FACTOR
    return UNSTABLE_VERTICAL_MULTIPLIER


def exchange_connectivity_multiplier(pair: tuple[int, int]) -> float:
    pair = tuple(sorted(pair))

    if pair in [(0, 1), (2, 3), (1, 3)]:
        return SURFACE_DEEP_CONNECTIVITY_MULTIPLIER

    return 1.0


def background_mixing_rate(i: int, j: int, dic: np.ndarray, wind_m_s: float) -> float:
    dic_diff = abs(dic[i] - dic[j])
    wind_factor = 1.0 + WIND_MIXING_ALPHA * (max(wind_m_s, 0.0) / WIND_MIXING_REF_M_S) ** 2
    dic_factor = dic_diff / max(DIC_DIFF_REF_MMOL_M3, 1e-12)
    dic_factor = np.clip(dic_factor, DIC_DIFF_FACTOR_MIN, DIC_DIFF_FACTOR_MAX)

    pair = tuple(sorted((i, j)))
    conn = exchange_connectivity_multiplier(pair)

    return BACKGROUND_MIXING_BASE_D1 * wind_factor * dic_factor * conn


def combined_exchange(
    i: int,
    j: int,
    dic: np.ndarray,
    rho: np.ndarray,
    volumes: np.ndarray,
    wind_m_s: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    pair = tuple(sorted((i, j)))
    conn = exchange_connectivity_multiplier(pair)
    base_rate = EXCHANGE_BASE_RATE_D1[pair] * conn

    drho_signed = rho[i] - rho[j]
    drho = abs(drho_signed)

    if drho <= RHO_THRESHOLD:
        q_density = 0.0
    else:
        rho_drive = (drho - RHO_THRESHOLD) / (drho + RHO_FRICTION + 1e-12)
        stability_factor = gravitational_stability_factor(i, j, rho)
        q_density = base_rate * rho_drive * stability_factor
        q_density = min(q_density, MAX_EXCHANGE_FRACTION_D1 * conn)

    q_background = background_mixing_rate(i=i, j=j, dic=dic, wind_m_s=wind_m_s)
    q_total = min(q_density + q_background, MAX_TOTAL_EXCHANGE_FRACTION_D1 * conn)

    Q_total = q_total * min(volumes[i], volumes[j])

    tendencies = np.zeros(4, dtype=float)
    tendencies[i] += Q_total / volumes[i] * (dic[j] - dic[i])
    tendencies[j] += Q_total / volumes[j] * (dic[i] - dic[j])

    dic_sign = np.sign(dic[i] - dic[j])
    if dic_sign == 0:
        dic_sign = np.sign(drho_signed)
    signed_Q_total = dic_sign * Q_total

    return tendencies, q_total, q_density, q_background, signed_Q_total


def seasonal_ncp_shape(day_of_year: float) -> float:
    spring = np.exp(-0.5 * ((day_of_year - SPRING_BLOOM_DAY) / SPRING_BLOOM_WIDTH) ** 2)
    summer = np.exp(-0.5 * ((day_of_year - SUMMER_BROAD_DAY) / SUMMER_BROAD_WIDTH) ** 2)
    fall = np.exp(-0.5 * ((day_of_year - FALL_BLOOM_DAY) / FALL_BLOOM_WIDTH) ** 2)
    return spring + SUMMER_WEIGHT * summer + FALL_WEIGHT * fall


def biological_dic_tendency(current_time: np.datetime64) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    if not BIO_PUMP_ON:
        return out

    ts = pd.Timestamp(current_time)
    day = float(ts.dayofyear)
    shape = seasonal_ncp_shape(day)

    shelf_uptake = BIO_NCP_MULTIPLIER * SHELF_NCP_MAX_MMOL_M3_D * shape
    ocean_uptake = BIO_NCP_MULTIPLIER * OCEAN_NCP_MAX_MMOL_M3_D * shape

    out[0] -= shelf_uptake
    out[2] -= ocean_uptake

    out[1] += BIO_REMIN_FRACTION_TO_LOWER_BOX * shelf_uptake * BOX_VOLUMES[0] / BOX_VOLUMES[1]
    out[3] += BIO_REMIN_FRACTION_TO_LOWER_BOX * ocean_uptake * BOX_VOLUMES[2] / BOX_VOLUMES[3]
    return out


def deep_export_seasonal_factor(current_time: np.datetime64) -> float:
    if DEEP_EXPORT_SEASONAL_AMPLITUDE <= 0.0:
        return 1.0
    day = float(pd.Timestamp(current_time).dayofyear)
    phase = 2.0 * np.pi * (day - DEEP_EXPORT_SEASONAL_PEAK_DAY) / 365.25
    factor = 1.0 + DEEP_EXPORT_SEASONAL_AMPLITUDE * np.cos(phase)
    return max(0.0, float(factor))


def deep_export_rate_for_box(
    box_index: int,
    current_time: np.datetime64,
    dic: np.ndarray,
    rho: np.ndarray,
) -> float:
    if not DEEP_EXPORT_ON:
        return 0.0

    if box_index == 1:
        boundary_dic = SHELF_BOTTOM_BOUNDARY_DIC
        base = DEEP_EXPORT_BASE_D1 * SHELF_BOTTOM_EXPORT_FRACTION
    elif box_index == 3:
        boundary_dic = OCEAN_DEEP_BOUNDARY_DIC
        base = DEEP_EXPORT_BASE_D1
    else:
        return 0.0

    dic_excess = max(float(dic[box_index] - boundary_dic), 0.0)
    dic_factor = 1.0 + DEEP_EXPORT_DIC_EXCESS_ALPHA * dic_excess / max(DEEP_EXPORT_DIC_EXCESS_REF, 1e-12)

    bottom_density_contrast = abs(float(rho[1] - rho[3]))
    density_factor = 1.0 + DEEP_EXPORT_DENSITY_ALPHA * bottom_density_contrast / max(DEEP_EXPORT_DENSITY_REF, 1e-12)

    seasonal_factor = deep_export_seasonal_factor(current_time)

    q = base * dic_factor * density_factor * seasonal_factor
    return float(np.clip(q, 0.0, DEEP_EXPORT_MAX_D1))


def deep_export_tendency(
    current_time: np.datetime64,
    dic: np.ndarray,
    rho: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tendency = np.zeros(4, dtype=float)
    rates = np.zeros(4, dtype=float)
    carbon_export_mmol_d = np.zeros(4, dtype=float)

    for box_index, boundary_dic in [(1, SHELF_BOTTOM_BOUNDARY_DIC), (3, OCEAN_DEEP_BOUNDARY_DIC)]:
        q = deep_export_rate_for_box(box_index, current_time, dic, rho)
        rates[box_index] = q

        tendency[box_index] = q * (boundary_dic - dic[box_index])
        carbon_export_mmol_d[box_index] = q * BOX_VOLUMES[box_index] * (dic[box_index] - boundary_dic)

    return tendency, rates, carbon_export_mmol_d


def build_initial_dic(temp0: np.ndarray, sal0: np.ndarray, recad: Dict[str, np.ndarray] | None, verbose: bool = True) -> np.ndarray:
    initial = INITIAL_DIC_FALLBACK.copy()

    if not USE_RECAD_INITIAL_SURFACE_DIC:
        return initial

    if recad is None or len(recad["time"]) == 0:
        if verbose:
            print("\nNo ReCAD/SOCAT initial pCO2 available. Using fallback initial DIC.")
        return initial

    shelf_pco2 = float(pd.Series(recad["shelf_mean"]).dropna().iloc[0])
    ocean_pco2 = float(pd.Series(recad["ocean_mean"]).dropna().iloc[0])

    shelf_dic = solve_dic_for_target_pco2(0, shelf_pco2, float(temp0[0]), float(sal0[0]))
    ocean_dic = solve_dic_for_target_pco2(2, ocean_pco2, float(temp0[2]), float(sal0[2]))

    initial[0] = shelf_dic
    initial[1] = shelf_dic + SHELF_BOTTOM_DIC_OFFSET
    initial[2] = ocean_dic
    initial[3] = ocean_dic + OCEAN_DEEP_DIC_OFFSET

    if verbose:
        print("\nInitial DIC estimated from first ReCAD/SOCAT pCO2:")
        print(f"  Shelf pCO2 target: {shelf_pco2:.2f} µatm -> DIC {initial[0]:.2f} mmol C m-3")
        print(f"  Ocean pCO2 target: {ocean_pco2:.2f} µatm -> DIC {initial[2]:.2f} mmol C m-3")
        print(f"  Initial DIC vector: {initial}")

    return initial


def run_dic_model(
    time: np.ndarray,
    temp_forcing: np.ndarray,
    sal_forcing: np.ndarray,
    wind_m_s: np.ndarray,
    initial_dic: np.ndarray,
) -> Dict[str, np.ndarray]:
    n_time = len(time)

    dic = np.full((n_time, 4), np.nan, dtype=float)
    pH = np.full((n_time, 4), np.nan, dtype=float)
    pco2 = np.full((n_time, 4), np.nan, dtype=float)
    rho = np.full((n_time, 4), np.nan, dtype=float)
    rho_no_dic = np.full((n_time, 4), np.nan, dtype=float)
    rho_dic_contribution = np.full((n_time, 4), np.nan, dtype=float)
    pressure = np.full((n_time, 4), np.nan, dtype=float)

    piston_velocity = np.full((n_time, 4), np.nan, dtype=float)
    airsea_tendency = np.full((n_time, 4), np.nan, dtype=float)
    bio_tendency = np.full((n_time, 4), np.nan, dtype=float)
    deep_export_tend = np.full((n_time, 4), np.nan, dtype=float)
    deep_export_rates = np.full((n_time, 4), np.nan, dtype=float)
    deep_carbon_export_mmol_d = np.full((n_time, 4), np.nan, dtype=float)

    combined_exchange_fluxes = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    total_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    density_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    background_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}

    dic[0, :] = np.asarray(initial_dic, dtype=float)

    for k in range(n_time - 1):
        dt_days = float((time[k + 1] - time[k]) / np.timedelta64(1, "D"))

        temp_now = temp_forcing[k, :]
        sal_now = sal_forcing[k, :]
        wind_now = wind_m_s[k]

        rho_base_now = density_teos10_with_dic(temp_now, sal_now, np.zeros(4))
        rho_now = density_teos10_with_dic(temp_now, sal_now, dic[k, :])
        rho_dic_now = rho_now - rho_base_now
        pressure_now = pressure_pa(rho_now, BOX_MID_DEPTHS)

        k_now = gas_transfer_velocity_m_per_day(np.full(4, wind_now), temp_now)
        carb_now = carbonate_speciation(dic[k, :], temp_now, sal_now, rho_now)

        rho[k, :] = rho_now
        rho_no_dic[k, :] = rho_base_now
        rho_dic_contribution[k, :] = rho_dic_now
        pressure[k, :] = pressure_now
        piston_velocity[k, :] = k_now
        pH[k, :] = carb_now["pH"]
        pco2[k, :] = carb_now["pCO2_uatm"]

        dCdt = np.zeros(4, dtype=float)

        for pair in PAIR_LIST:
            i, j = pair
            tendencies_pair, q_total, q_density, q_background, signed_Q_total = combined_exchange(
                i=i,
                j=j,
                dic=dic[k, :],
                rho=rho_now,
                volumes=BOX_VOLUMES,
                wind_m_s=wind_now,
            )
            dCdt += tendencies_pair
            total_exchange_rates[pair][k] = q_total
            density_exchange_rates[pair][k] = q_density
            background_exchange_rates[pair][k] = q_background
            combined_exchange_fluxes[pair][k] = signed_Q_total

        airsea_tendency[k, 0] = air_sea_dic_tendency_from_carb(carb_now, 0, k_now[0], SURFACE_THICKNESS_M)
        airsea_tendency[k, 2] = air_sea_dic_tendency_from_carb(carb_now, 2, k_now[2], SURFACE_THICKNESS_M)
        airsea_tendency[k, 1] = 0.0
        airsea_tendency[k, 3] = 0.0

        dCdt[0] += airsea_tendency[k, 0]
        dCdt[2] += airsea_tendency[k, 2]

        bio_now = biological_dic_tendency(time[k])
        bio_tendency[k, :] = bio_now
        dCdt += bio_now

        export_now, export_q_now, carbon_export_now = deep_export_tendency(time[k], dic[k, :], rho_now)
        deep_export_tend[k, :] = export_now
        deep_export_rates[k, :] = export_q_now
        deep_carbon_export_mmol_d[k, :] = carbon_export_now
        dCdt += export_now

        dic_next = dic[k, :] + dt_days * dCdt

        if not np.all(np.isfinite(dic_next)):
            print("\nERROR: non-finite DIC produced.")
            print("time index:", k)
            print("time:", time[k])
            print("dic current:", dic[k, :])
            print("dCdt:", dCdt)
            print("temp:", temp_now)
            print("sal:", sal_now)
            print("rho:", rho_now)
            print("rho from DIC:", rho_dic_now)
            print("wind:", wind_now)
            print("k_m_d:", k_now)
            print("pco2:", pco2[k, :])
            print("deep export tendency:", export_now)
            raise FloatingPointError("DIC became non-finite.")

        dic[k + 1, :] = np.clip(dic_next, 1e-9, None)

    temp_last = temp_forcing[-1, :]
    sal_last = sal_forcing[-1, :]
    wind_last = wind_m_s[-1]

    rho_base_last = density_teos10_with_dic(temp_last, sal_last, np.zeros(4))
    rho_last = density_teos10_with_dic(temp_last, sal_last, dic[-1, :])
    rho[-1, :] = rho_last
    rho_no_dic[-1, :] = rho_base_last
    rho_dic_contribution[-1, :] = rho_last - rho_base_last
    pressure[-1, :] = pressure_pa(rho_last, BOX_MID_DEPTHS)

    k_last = gas_transfer_velocity_m_per_day(np.full(4, wind_last), temp_last)
    piston_velocity[-1, :] = k_last

    carb_last = carbonate_speciation(dic[-1, :], temp_last, sal_last, rho_last)
    pH[-1, :] = carb_last["pH"]
    pco2[-1, :] = carb_last["pCO2_uatm"]

    airsea_tendency[-1, :] = 0.0
    bio_tendency[-1, :] = biological_dic_tendency(time[-1])
    deep_export_tend[-1, :], deep_export_rates[-1, :], deep_carbon_export_mmol_d[-1, :] = deep_export_tendency(time[-1], dic[-1, :], rho_last)

    for pair in PAIR_LIST:
        combined_exchange_fluxes[pair][-1] = 0.0
        total_exchange_rates[pair][-1] = 0.0
        density_exchange_rates[pair][-1] = 0.0
        background_exchange_rates[pair][-1] = 0.0

    return {
        "time": time,
        "temp": temp_forcing,
        "sal": sal_forcing,
        "wind_m_s": wind_m_s,
        "piston_velocity_m_d": piston_velocity,
        "dic": dic,
        "pH": pH,
        "pco2": pco2,
        "rho": rho,
        "rho_no_dic": rho_no_dic,
        "rho_dic_contribution": rho_dic_contribution,
        "pressure": pressure,
        "combined_exchange_fluxes": combined_exchange_fluxes,
        "total_exchange_rates": total_exchange_rates,
        "density_exchange_rates": density_exchange_rates,
        "background_exchange_rates": background_exchange_rates,
        "airsea_tendency": airsea_tendency,
        "bio_tendency": bio_tendency,
        "deep_export_tendency": deep_export_tend,
        "deep_export_rates": deep_export_rates,
        "deep_carbon_export_mmol_d": deep_carbon_export_mmol_d,
    }


# ============================================================
# 9) METRICS + ITERATIVE CALIBRATION
# ============================================================

def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 0.0 or np.nanstd(y) <= 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def monthly_delta_arrays(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    if recad is None or len(recad["time"]) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    model_time = result["time"]
    pco2 = result["pco2"]

    model_shelf_delta = monthly_series(model_time, PCO2_AIR_UATM - pco2[:, 0])
    model_ocean_delta = monthly_series(model_time, PCO2_AIR_UATM - pco2[:, 2])

    obs_shelf_delta, obs_ocean_delta = get_recad_monthly_delta(recad)
    if obs_shelf_delta is None or obs_ocean_delta is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    model_parts = []
    obs_parts = []

    for model, obs in [(model_shelf_delta, obs_shelf_delta), (model_ocean_delta, obs_ocean_delta)]:
        df = pd.concat([model.rename("model"), obs.rename("obs")], axis=1, join="inner").dropna()
        if len(df) > 0:
            model_parts.append(df["model"].values.astype(float))
            obs_parts.append(df["obs"].values.astype(float))

    if len(model_parts) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(model_parts), np.concatenate(obs_parts)


def monthly_delta_metrics(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> dict:
    model_values, obs_values = monthly_delta_arrays(result, recad)
    if len(model_values) == 0:
        return {"r_value": np.nan, "rmse": np.nan, "bias_model_minus_obs": np.nan, "n_points": 0}

    error = model_values - obs_values
    return {
        "r_value": _safe_corrcoef(model_values, obs_values),
        "rmse": float(np.sqrt(np.nanmean(error ** 2))),
        "bias_model_minus_obs": float(np.nanmean(error)),
        "n_points": int(len(error)),
    }


def monthly_delta_metrics_by_region(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> dict:
    """
    Separate shelf/ocean diagnostics. This is important because the combined
    metric can hide that one region fits okay while the other fails.
    """
    if recad is None or len(recad["time"]) == 0:
        return {}

    model_time = result["time"]
    pco2 = result["pco2"]

    model_shelf = monthly_series(model_time, PCO2_AIR_UATM - pco2[:, 0])
    model_ocean = monthly_series(model_time, PCO2_AIR_UATM - pco2[:, 2])

    obs_shelf, obs_ocean = get_recad_monthly_delta(recad)

    out = {}

    for name, model, obs in [
        ("shelf", model_shelf, obs_shelf),
        ("ocean", model_ocean, obs_ocean),
    ]:
        if obs is None:
            out[name] = {"r_value": np.nan, "rmse": np.nan, "bias": np.nan, "n": 0}
            continue

        df = pd.concat([model.rename("model"), obs.rename("obs")], axis=1, join="inner").dropna()

        if len(df) < 3:
            out[name] = {"r_value": np.nan, "rmse": np.nan, "bias": np.nan, "n": len(df)}
            continue

        error = df["model"].values - df["obs"].values

        out[name] = {
            "r_value": _safe_corrcoef(df["model"].values, df["obs"].values),
            "rmse": float(np.sqrt(np.nanmean(error ** 2))),
            "bias": float(np.nanmean(error)),
            "n": int(len(df)),
            "model_min": float(np.nanmin(df["model"].values)),
            "model_max": float(np.nanmax(df["model"].values)),
            "obs_min": float(np.nanmin(df["obs"].values)),
            "obs_max": float(np.nanmax(df["obs"].values)),
        }

    return out


def monthly_delta_rmse(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> float:
    return float(monthly_delta_metrics(result, recad)["rmse"])


def monthly_delta_bias(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> float:
    return float(monthly_delta_metrics(result, recad)["bias_model_minus_obs"])


def _candidate_values(current: float, percent: float, min_value: float, max_value: float, zero_step: float) -> list[float]:
    current = float(current)

    # Important: if a value loaded from the parameter file is outside the new
    # tuning range, clamp the current value into the valid range first. This
    # prevents old DEEP_EXPORT_BASE_D1 = 0.0 from surviving forever.
    current = min(max(current, min_value), max_value)

    step = max(abs(current) * percent, abs(zero_step) * percent)
    values = [current, current - step, current + step]
    clipped = []

    for value in values:
        value = min(max(float(value), min_value), max_value)
        if not any(np.isclose(value, old, rtol=1e-12, atol=1e-15) for old in clipped):
            clipped.append(value)

    return clipped


def _candidate_score(metrics: dict, reference_metrics: dict) -> float:
    r = metrics.get("r_value", np.nan)
    rmse = metrics.get("rmse", np.nan)
    bias = metrics.get("bias_model_minus_obs", np.nan)

    r_score = -1.0 if not np.isfinite(r) else float(r)

    ref_rmse = reference_metrics.get("rmse", np.nan)
    if not np.isfinite(ref_rmse) or ref_rmse <= 0.0:
        ref_rmse = 100.0
    rmse_score = -float(rmse) / ref_rmse if np.isfinite(rmse) else -1e6

    ref_bias = abs(reference_metrics.get("bias_model_minus_obs", np.nan))
    if not np.isfinite(ref_bias) or ref_bias <= 1.0:
        ref_bias = 25.0
    bias_score = -abs(float(bias)) / ref_bias if np.isfinite(bias) else -1e6

    return TUNING_WEIGHT_R * r_score + TUNING_WEIGHT_RMSE * rmse_score + TUNING_WEIGHT_BIAS * bias_score


def _is_better_candidate(candidate: dict, current: dict) -> bool:
    cand_r = candidate.get("r_value", np.nan)
    curr_r = current.get("r_value", np.nan)
    cand_rmse = candidate.get("rmse", np.inf)
    curr_rmse = current.get("rmse", np.inf)
    cand_bias = abs(candidate.get("bias_model_minus_obs", np.inf))
    curr_bias = abs(current.get("bias_model_minus_obs", np.inf))

    if not np.isfinite(cand_r):
        return False
    if not np.isfinite(curr_r):
        return True

    if cand_r > curr_r + TUNING_R_EQUAL_TOLERANCE:
        return True

    if abs(cand_r - curr_r) <= TUNING_R_EQUAL_TOLERANCE:
        if np.isfinite(cand_rmse) and np.isfinite(curr_rmse) and cand_rmse < curr_rmse:
            return True
        if np.isfinite(cand_rmse) and np.isfinite(curr_rmse) and abs(cand_rmse - curr_rmse) <= 1e-6:
            if np.isfinite(cand_bias) and np.isfinite(curr_bias) and cand_bias < curr_bias:
                return True

    return False


def _plot_r_value_development(history_df: pd.DataFrame, out_dir: Path) -> None:
    if history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df["iteration"], history_df["best_r_value"], linewidth=1.5, label="Best r-value so far")
    ax.plot(history_df["iteration"], history_df["candidate_r_value"], linewidth=0.8, alpha=0.5, label="Chosen candidate r-value")
    ax.axhline(TUNING_TARGET_R_VALUE, color="k", linestyle="--", linewidth=1.0, label=f"target r = {TUNING_TARGET_R_VALUE}")
    ax.set_xlabel("Calibration iteration")
    ax.set_ylabel("Pearson r-value")
    ax.set_title("Development of monthly ΔpCO2 correlation during tuning")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / "calibration_r_value_development.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if SHOW_CALIBRATION_PROGRESS:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved r-value development plot to:\n{out_path}")


def _tuning_parameters() -> list[dict]:
    """
    Main place to change the parameters that are tuned.

    IMPORTANT:
    TA0_UMOLKG is deliberately not included here anymore.
    BIO_NCP_MULTIPLIER is included because direct biological DIC uptake strongly
    controls whether surface pCO2 becomes too low.
    """
    return [
        # Biology first, because previous runs suggested surface pCO2 was too low.
        {"name": "BIO_NCP_MULTIPLIER", "min": 0.0, "max": 3.0, "zero_step": 0.20},

        # Deep export cannot go to 0 anymore.
        {"name": "DEEP_EXPORT_BASE_D1", "min": 0.0002, "max": 0.10, "zero_step": 0.002},
        {"name": "DEEP_EXPORT_MAX_D1", "min": 0.001, "max": 0.10, "zero_step": 0.01},

        # Lower boundary ranges allow export to actually remove DIC.
        {"name": "OCEAN_DEEP_BOUNDARY_DIC", "min": 1800.0, "max": 2400.0, "zero_step": 100.0},
        {"name": "SHELF_BOTTOM_BOUNDARY_DIC", "min": 1800.0, "max": 2300.0, "zero_step": 100.0},

        {"name": "SHELF_BOTTOM_EXPORT_FRACTION", "min": 0.0, "max": 1.0, "zero_step": 0.10},
        {"name": "DEEP_EXPORT_DIC_EXCESS_ALPHA", "min": 0.0, "max": 8.0, "zero_step": 1.0},
        {"name": "DEEP_EXPORT_DENSITY_ALPHA", "min": 0.0, "max": 8.0, "zero_step": 1.0},
        {"name": "DEEP_EXPORT_SEASONAL_AMPLITUDE", "min": 0.0, "max": 1.5, "zero_step": 0.20},

        # Surface-bottom/deep connectivity.
        {"name": "SURFACE_DEEP_CONNECTIVITY_MULTIPLIER", "min": 0.2, "max": 20.0, "zero_step": 1.0},
        {"name": "BACKGROUND_MIXING_BASE_D1", "min": 0.0, "max": 0.005, "zero_step": 2.0e-5},
        {"name": "AIRSEA_K_MULTIPLIER", "min": 0.02, "max": 10.0, "zero_step": 0.50},
    ]


def _clamp_tuned_parameters_to_bounds(param_specs: list[dict]) -> None:
    """Clamp loaded values to the allowed calibration range before tuning."""
    for spec in param_specs:
        name = spec["name"]
        if name not in globals():
            continue
        old_value = float(globals()[name])
        new_value = min(max(old_value, float(spec["min"])), float(spec["max"]))
        if not np.isclose(old_value, new_value, rtol=1e-12, atol=1e-15):
            print(f"  Clamped {name}: {old_value} -> {new_value}")
            globals()[name] = new_value


def calibrate_to_recad(
    time: np.ndarray,
    temp_forcing: np.ndarray,
    sal_forcing: np.ndarray,
    wind_m_s: np.ndarray,
    recad: Dict[str, np.ndarray] | None,
) -> np.ndarray:
    if recad is None or len(recad["time"]) == 0:
        print("\nCalibration skipped because ReCAD/SOCAT data is unavailable.")
        return build_initial_dic(temp_forcing[0, :], sal_forcing[0, :], recad)

    print("\n============================================================")
    print("Iterative calibration against monthly ReCAD/SOCAT ΔpCO2")
    print("============================================================")
    print("TA0_UMOLKG is NOT calibrated in this version.")
    print("Calibration includes BIO_NCP_MULTIPLIER, deep export, deep boundary DIC, and connectivity.")
    print(f"Maximum iterations: {TUNING_MAX_ITERATIONS}")
    print(f"Stop if r-value exceeds: {TUNING_TARGET_R_VALUE}")
    print(f"Initial percentage step: ±{100*TUNING_START_PERCENT:.1f}%")
    print(f"Minimum percentage step: ±{100*TUNING_MIN_PERCENT:.1f}%")
    print(f"Print mode: {TUNING_PRINT_MODE}")

    param_specs = _tuning_parameters()

    print("\nClamping loaded tuned parameters to allowed ranges if needed:")
    _clamp_tuned_parameters_to_bounds(param_specs)

    original_values = {spec["name"]: float(globals()[spec["name"]]) for spec in param_specs}

    print("\nTuned parameters:")
    for spec in param_specs:
        print(f"  {spec['name']} = {globals()[spec['name']]}")

    def evaluate_current() -> tuple[dict, np.ndarray, Dict[str, np.ndarray]]:
        initial_dic = build_initial_dic(temp_forcing[0, :], sal_forcing[0, :], recad, verbose=False)
        result = run_dic_model(time, temp_forcing, sal_forcing, wind_m_s, initial_dic)
        return monthly_delta_metrics(result, recad), initial_dic, result

    best_metrics, best_initial_dic, _ = evaluate_current()
    best_r = best_metrics["r_value"] if np.isfinite(best_metrics["r_value"]) else -np.inf
    best_rmse = best_metrics["rmse"]
    best_bias = best_metrics["bias_model_minus_obs"]

    rows = []
    history = []

    best_r_at_cycle_start = best_r
    no_improvement_cycles = 0

    print(f"\ninitial r = {best_r:.4f}, RMSE = {best_rmse:.2f}, bias = {best_bias:.2f}")

    for iteration in range(1, TUNING_MAX_ITERATIONS + 1):
        spec = param_specs[(iteration - 1) % len(param_specs)]
        cycle_index = (iteration - 1) // len(param_specs)
        percent = max(TUNING_MIN_PERCENT, TUNING_START_PERCENT * (TUNING_STEP_DECAY ** cycle_index))

        name = spec["name"]
        current_value = min(max(float(globals()[name]), float(spec["min"])), float(spec["max"]))
        globals()[name] = current_value

        candidates = _candidate_values(current_value, percent, float(spec["min"]), float(spec["max"]), float(spec["zero_step"]))

        reference_metrics = best_metrics.copy()
        candidate_rows = []

        for value in candidates:
            globals()[name] = float(value)
            metrics, initial_dic, result = evaluate_current()
            score = _candidate_score(metrics, reference_metrics)

            row = {
                "iteration": iteration,
                "cycle": cycle_index + 1,
                "step_percent": percent,
                "parameter": name,
                "candidate_value": float(value),
                "score": score,
                **metrics,
                "initial_dic_1": initial_dic[0],
                "initial_dic_2": initial_dic[1],
                "initial_dic_3": initial_dic[2],
                "initial_dic_4": initial_dic[3],
                "final_dic_1": result["dic"][-1, 0],
                "final_dic_2": result["dic"][-1, 1],
                "final_dic_3": result["dic"][-1, 2],
                "final_dic_4": result["dic"][-1, 3],
                "total_positive_deep_export_mmol": integrate_daily(
                    np.nansum(np.maximum(result["deep_carbon_export_mmol_d"], 0.0), axis=1)
                ),
            }

            for ps in param_specs:
                row[ps["name"]] = float(globals()[ps["name"]])

            candidate_rows.append(row)
            rows.append(row)

        globals()[name] = current_value

        current_candidate = None
        for row in candidate_rows:
            if np.isclose(row["candidate_value"], current_value, rtol=1e-12, atol=1e-15):
                current_candidate = row
                break

        if current_candidate is None:
            current_candidate = {
                "score": _candidate_score(best_metrics, reference_metrics),
                **best_metrics,
            }

        selected = max(candidate_rows, key=lambda row: row["score"])

        score_gain = selected["score"] - current_candidate["score"]
        rmse_gain = current_candidate["rmse"] - selected["rmse"]
        bias_gain = abs(current_candidate["bias_model_minus_obs"]) - abs(selected["bias_model_minus_obs"])
        r_gain = selected["r_value"] - current_candidate["r_value"]

        ACCEPT_SCORE_GAIN = 1e-5
        ACCEPT_R_LOSS_LIMIT = 0.002
        ACCEPT_RMSE_GAIN_MIN = 0.25
        ACCEPT_BIAS_GAIN_MIN = 0.25

        accept_by_r = _is_better_candidate(selected, best_metrics)

        accept_by_score = (
            np.isfinite(score_gain)
            and score_gain > ACCEPT_SCORE_GAIN
            and np.isfinite(r_gain)
            and r_gain >= -ACCEPT_R_LOSS_LIMIT
        )

        accept_by_rmse_bias = (
            np.isfinite(r_gain)
            and r_gain >= -ACCEPT_R_LOSS_LIMIT
            and (
                (np.isfinite(rmse_gain) and rmse_gain > ACCEPT_RMSE_GAIN_MIN)
                or (np.isfinite(bias_gain) and bias_gain > ACCEPT_BIAS_GAIN_MIN)
            )
        )

        if accept_by_r or accept_by_score or accept_by_rmse_bias:
            globals()[name] = float(selected["candidate_value"])
            accepted = True
            best_metrics, best_initial_dic, _ = evaluate_current()
            best_r = best_metrics["r_value"] if np.isfinite(best_metrics["r_value"]) else -np.inf
            best_rmse = best_metrics["rmse"]
            best_bias = best_metrics["bias_model_minus_obs"]
        else:
            globals()[name] = current_value
            accepted = False

        history_row = {
            "iteration": iteration,
            "cycle": cycle_index + 1,
            "step_percent": percent,
            "parameter": name,
            "selected_value": float(globals()[name]),
            "accepted": accepted,
            "candidate_r_value": selected["r_value"],
            "candidate_rmse": selected["rmse"],
            "candidate_bias": selected["bias_model_minus_obs"],
            "best_r_value": best_r,
            "best_rmse": best_rmse,
            "best_bias_model_minus_obs": best_bias,
        }

        for ps in param_specs:
            history_row[ps["name"]] = float(globals()[ps["name"]])

        history.append(history_row)

        if TUNING_PRINT_MODE == "r_only":
            mark = "accepted" if accepted else "kept"
            print(f"iteration {iteration:03d}: r = {best_r:.4f}, RMSE = {best_rmse:.2f}, bias = {best_bias:.2f} ({mark})")
        elif TUNING_PRINT_MODE == "chosen":
            mark = "accepted" if accepted else "kept"
            print(
                f"[{iteration:03d}/{TUNING_MAX_ITERATIONS}] {name} -> {globals()[name]:.8g} "
                f"({mark}) | step=±{100*percent:5.2f}% | "
                f"r={best_r:7.4f} | RMSE={best_rmse:8.3f} | bias={best_bias:8.3f}"
            )

        if np.isfinite(best_r) and best_r >= TUNING_TARGET_R_VALUE:
            print(f"\nStopping because r-value reached {best_r:.4f} >= {TUNING_TARGET_R_VALUE:.4f}.")
            break

        completed_full_cycle = (iteration % len(param_specs) == 0)
        if completed_full_cycle:
            cycle_improvement = best_r - best_r_at_cycle_start
            if not np.isfinite(cycle_improvement) or cycle_improvement <= TUNING_MIN_R_IMPROVEMENT_PER_CYCLE:
                no_improvement_cycles += 1
            else:
                no_improvement_cycles = 0

            if TUNING_PRINT_MODE != "silent":
                print(
                    f"cycle {cycle_index + 1} finished: Δr = {cycle_improvement:.5f}, "
                    f"no-improvement cycles = {no_improvement_cycles}/{TUNING_STOP_AFTER_NO_IMPROVEMENT_CYCLES}"
                )

            best_r_at_cycle_start = best_r

            if no_improvement_cycles >= TUNING_STOP_AFTER_NO_IMPROVEMENT_CYCLES:
                print(
                    f"\nStopping because {no_improvement_cycles} complete cycles had no real improvement "
                    f"(Δr <= {TUNING_MIN_R_IMPROVEMENT_PER_CYCLE})."
                )
                break

    df = pd.DataFrame(rows)
    hist_df = pd.DataFrame(history)

    results_path = OUT_DIR / "iterative_calibration_candidate_results.csv"
    history_path = OUT_DIR / "iterative_calibration_history.csv"
    df.to_csv(results_path, index=False)
    hist_df.to_csv(history_path, index=False)

    _plot_r_value_development(hist_df, OUT_DIR)

    print(f"\nSaved candidate results to:\n{results_path}")
    print(f"Saved tuning history to:\n{history_path}")

    print("\nSelected final parameter set:")
    print(f"  TA0_UMOLKG = {TA0_UMOLKG}  (not calibrated)")
    for spec in param_specs:
        print(f"  {spec['name']} = {globals()[spec['name']]}")
    print(f"  BIO_PUMP_ON = {BIO_PUMP_ON}")
    print(f"  final r-value = {best_r:.4f}")
    print(f"  final RMSE = {best_rmse:.3f} µatm")
    print(f"  final bias = {best_bias:.3f} µatm")

    if not np.isfinite(best_rmse):
        print("\nWARNING: calibration metrics were not finite. Restoring original parameter values.")
        for key, value in original_values.items():
            globals()[key] = value
        best_initial_dic = build_initial_dic(temp_forcing[0, :], sal_forcing[0, :], recad, verbose=True)

    if SAVE_PARAMETERS_AFTER_CALIBRATION:
        save_parameter_file()

    return best_initial_dic


# ============================================================
# 10) PLOTTING HELPERS
# ============================================================

def monthly_mean(time: np.ndarray, values: np.ndarray) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    s = monthly_series(time, values)
    return s.index, s.values


def disable_y_offset(ax):
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def make_four_box_figure(title: str, y_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    for i in range(4):
        r, c = BOX_LAYOUT[i]
        ax = axes[r, c]
        ax.set_title(BOX_NAMES[i])
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        disable_y_offset(ax)

    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    return fig, axes


def plot_four_box_time_series(time: np.ndarray, values: np.ndarray, title: str, y_label: str, color: str, out_name: str | None = None):
    fig, axes = make_four_box_figure(title=title, y_label=y_label)

    for i in range(4):
        r, c = BOX_LAYOUT[i]
        axes[r, c].plot(time, values[:, i], color=color, linewidth=1.3)
        disable_y_offset(axes[r, c])

        if "DIC" in title and "density" not in title.lower():
            v = values[:, i]
            pad = max(1.0, 0.08 * (np.nanmax(v) - np.nanmin(v)))
            axes[r, c].set_ylim(np.nanmin(v) - pad, np.nanmax(v) + pad)

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_wind_and_piston_velocity(time: np.ndarray, wind_m_s: np.ndarray, piston_velocity: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time, wind_m_s, color="tab:purple", linewidth=1.2)
    axes[0].set_title("Open-Meteo 10 m wind speed used for both surface boxes")
    axes[0].set_ylabel("Wind speed (m s$^{-1}$)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, piston_velocity[:, 0], label="Box 1 shelf surface", linewidth=1.2)
    axes[1].plot(time, piston_velocity[:, 2], label="Box 3 ocean surface", linewidth=1.2)
    axes[1].set_title("Wind-dependent CO$_2$ piston velocity")
    axes[1].set_ylabel("k (m d$^{-1}$)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_combined_exchange_fluxes(time: np.ndarray, combined_exchange_fluxes: Dict[Tuple[int, int], np.ndarray], out_name: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Combined density-driven + background exchange", fontsize=14)

    positions = {(0, 1): (0, 0), (1, 3): (1, 0), (0, 2): (0, 1), (2, 3): (1, 1)}

    for pair in PAIR_LIST:
        r, c = positions[pair]
        ax = axes[r, c]
        i, j = pair
        signed_Q_m3_d = combined_exchange_fluxes[pair]
        reference_volume = min(BOX_VOLUMES[i], BOX_VOLUMES[j])
        signed_q_percent_d = 100.0 * signed_Q_m3_d / reference_volume

        ax.plot(time, signed_q_percent_d, linewidth=1.2)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(PAIR_LABELS[pair])
        ax.set_ylabel("Signed combined exchange\n(% of smaller box d$^{-1}$)")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.92, f"+ = {BOX_SHORT[i]} -> {BOX_SHORT[j]}", transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        disable_y_offset(ax)

    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_exchange_rate_components(
    time: np.ndarray,
    total_rates: Dict[Tuple[int, int], np.ndarray],
    density_rates: Dict[Tuple[int, int], np.ndarray],
    background_rates: Dict[Tuple[int, int], np.ndarray],
    out_name: str | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Exchange-rate components", fontsize=14)
    positions = {(0, 1): (0, 0), (1, 3): (1, 0), (0, 2): (0, 1), (2, 3): (1, 1)}

    for pair in PAIR_LIST:
        r, c = positions[pair]
        ax = axes[r, c]
        ax.plot(time, total_rates[pair], label="total", linewidth=1.3)
        ax.plot(time, density_rates[pair], label="density", linewidth=1.0)
        ax.plot(time, background_rates[pair], label="background", linewidth=1.0)
        ax.set_title(PAIR_LABELS[pair])
        ax.set_ylabel("Exchange rate (d$^{-1}$)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        disable_y_offset(ax)

    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_surface_pco2(time: np.ndarray, pco2: np.ndarray, recad: Dict[str, np.ndarray] | None = None, out_name: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Surface box pCO$_2$ compared with ReCAD/SOCAT", fontsize=14)

    axes[0].plot(time, pco2[:, 0], linewidth=1.4, label="Model Box 1 shelf surface")
    if recad is not None and len(recad["time"]) > 0:
        axes[0].plot(recad["time"], recad["shelf_mean"], marker="o", linewidth=1.2, label="ReCAD/SOCAT shelf mean")
    axes[0].set_title("Shelf surface")
    axes[0].set_ylabel("pCO$_2$ (µatm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, pco2[:, 2], linewidth=1.4, label="Model Box 3 ocean surface")
    if recad is not None and len(recad["time"]) > 0:
        axes[1].plot(recad["time"], recad["ocean_mean"], marker="o", linewidth=1.2, label="ReCAD/SOCAT ocean mean")
    axes[1].set_title("Ocean surface")
    axes[1].set_ylabel("pCO$_2$ (µatm)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def add_delta_labels(ax):
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax.text(0.01, 0.92, "Ingassing / ocean uptake\n$\\Delta pCO_2 > 0$", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.text(0.01, 0.08, "Outgassing\n$\\Delta pCO_2 < 0$", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))


def plot_surface_delta_pco2(time: np.ndarray, pco2: np.ndarray, recad: Dict[str, np.ndarray] | None = None, out_name: str | None = None):
    delta_box1 = PCO2_AIR_UATM - pco2[:, 0]
    delta_box3 = PCO2_AIR_UATM - pco2[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(r"Surface box $\Delta pCO_2 = 420 - pCO_2$", fontsize=14)

    axes[0].plot(time, delta_box1, linewidth=1.4, label="Model Box 1 shelf surface")
    if recad is not None and len(recad["time"]) > 0:
        axes[0].plot(recad["time"], recad["shelf_delta"], marker="o", linewidth=1.2, label="ReCAD/SOCAT shelf mean")
    add_delta_labels(axes[0])
    axes[0].set_title("Shelf surface")
    axes[0].set_ylabel(r"$\Delta pCO_2$ (µatm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, delta_box3, linewidth=1.4, label="Model Box 3 ocean surface")
    if recad is not None and len(recad["time"]) > 0:
        axes[1].plot(recad["time"], recad["ocean_delta"], marker="o", linewidth=1.2, label="ReCAD/SOCAT ocean mean")
    add_delta_labels(axes[1])
    axes[1].set_title("Ocean surface")
    axes[1].set_ylabel(r"$\Delta pCO_2$ (µatm)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_monthly_delta_pco2(time: np.ndarray, pco2: np.ndarray, recad: Dict[str, np.ndarray] | None = None, out_name: str | None = None):
    delta_shelf = PCO2_AIR_UATM - pco2[:, 0]
    delta_ocean = PCO2_AIR_UATM - pco2[:, 2]

    model_shelf = monthly_series(time, delta_shelf)
    model_ocean = monthly_series(time, delta_ocean)
    obs_shelf, obs_ocean = get_recad_monthly_delta(recad)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(r"Monthly mean $\Delta pCO_2 = 420 - pCO_2$", fontsize=14)

    axes[0].plot(model_shelf.index, model_shelf.values, marker="o", linewidth=1.4, label="Model monthly mean")
    if obs_shelf is not None:
        axes[0].plot(obs_shelf.index, obs_shelf.values, marker="o", linewidth=1.2, label="ReCAD/SOCAT shelf mean")
    add_delta_labels(axes[0])
    axes[0].set_title("Shelf surface")
    axes[0].set_ylabel(r"$\Delta pCO_2$ (µatm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(model_ocean.index, model_ocean.values, marker="o", linewidth=1.4, label="Model monthly mean")
    if obs_ocean is not None:
        axes[1].plot(obs_ocean.index, obs_ocean.values, marker="o", linewidth=1.2, label="ReCAD/SOCAT ocean mean")
    add_delta_labels(axes[1])
    axes[1].set_title("Ocean surface")
    axes[1].set_ylabel(r"$\Delta pCO_2$ (µatm)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_deep_export_diagnostics(time: np.ndarray, deep_export_rates: np.ndarray, deep_export_tendency: np.ndarray, deep_carbon_export_mmol_d: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Dynamic deep carbon export / open-boundary flushing", fontsize=14)

    axes[0].plot(time, deep_export_rates[:, 1], label="Box 2 shelf bottom", linewidth=1.3)
    axes[0].plot(time, deep_export_rates[:, 3], label="Box 4 ocean deep", linewidth=1.3)
    axes[0].set_ylabel("q export (d$^{-1}$)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, deep_export_tendency[:, 1], label="Box 2 shelf bottom", linewidth=1.3)
    axes[1].plot(time, deep_export_tendency[:, 3], label="Box 4 ocean deep", linewidth=1.3)
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("DIC tendency\n(mmol C m$^{-3}$ d$^{-1}$)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time, deep_carbon_export_mmol_d[:, 1] / 1e12, label="Box 2 shelf bottom", linewidth=1.3)
    axes[2].plot(time, deep_carbon_export_mmol_d[:, 3] / 1e12, label="Box 4 ocean deep", linewidth=1.3)
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Carbon export\n(10$^{12}$ mmol C d$^{-1}$)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_airsea_tendency(time: np.ndarray, airsea_tendency: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Wind-dependent air-sea DIC tendency", fontsize=14)

    axes[0].plot(time, airsea_tendency[:, 0], linewidth=1.3)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_title("Box 1: Shelf surface")
    axes[0].set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, airsea_tendency[:, 2], linewidth=1.3)
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_title("Box 3: Ocean surface")
    axes[1].set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_bio_tendency(time: np.ndarray, bio_tendency: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Prescribed biological DIC tendency", fontsize=14)

    axes[0].plot(time, bio_tendency[:, 0], label="Box 1 shelf surface", linewidth=1.3)
    axes[0].plot(time, bio_tendency[:, 1], label="Box 2 shelf bottom", linewidth=1.3)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_title("Shelf boxes")
    axes[0].set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, bio_tendency[:, 2], label="Box 3 ocean surface", linewidth=1.3)
    axes[1].plot(time, bio_tendency[:, 3], label="Box 4 ocean deep", linewidth=1.3)
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_title("Ocean boxes")
    axes[1].set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_density_diagnostics(time: np.ndarray, rho: np.ndarray, rho_no_dic: np.ndarray, rho_dic_contribution: np.ndarray):
    plot_four_box_time_series(time, rho, "Density including TEOS-10 + DIC mass correction", "Density (kg m$^{-3}$)", "tab:brown", "density_total_four_boxes.png")
    plot_four_box_time_series(time, rho_dic_contribution, "DIC contribution to density", "Density contribution (kg m$^{-3}$)", "tab:purple", "density_dic_contribution_four_boxes.png")


# ============================================================
# 11) MAIN
# ============================================================

def main():
    load_parameter_file_if_needed()

    print("\n============================================================")
    print("Loading / building Copernicus forcing")
    print("============================================================")
    time, temp_forcing, sal_forcing = load_or_build_forcing()

    print("\n============================================================")
    print("Loading wind forcing")
    print("============================================================")
    wind_m_s = load_wind_for_model_time(time)

    print("\n============================================================")
    print("Loading ReCAD/SOCAT comparison")
    print("============================================================")
    recad = load_recad_spatial_means()

    if CALIBRATE_TO_RECAD:
        initial_dic = calibrate_to_recad(time, temp_forcing, sal_forcing, wind_m_s, recad)
    else:
        initial_dic = build_initial_dic(temp_forcing[0, :], sal_forcing[0, :], recad, verbose=True)

    print("\n============================================================")
    print("Running final four-box DIC model")
    print("============================================================")

    result = run_dic_model(time, temp_forcing, sal_forcing, wind_m_s, initial_dic)

    metrics = monthly_delta_metrics(result, recad)
    rmse = metrics["rmse"]
    bias = metrics["bias_model_minus_obs"]
    r_value = metrics["r_value"]

    region_metrics = monthly_delta_metrics_by_region(result, recad)

    model_values, obs_values = monthly_delta_arrays(result, recad)
    model_range_text = "not available"
    if len(model_values) > 0:
        model_range_text = f"[{np.nanmin(model_values):.2f}, {np.nanmax(model_values):.2f}] µatm"

    daily_positive_export = np.nansum(
        np.maximum(result["deep_carbon_export_mmol_d"], 0.0),
        axis=1,
    )
    total_deep_export_mmol = integrate_daily(daily_positive_export)

    print("\nForcing / simulation summary:")
    print(f"Dataset: {DATASET_ID}")
    print(f"Time steps: {len(result['time'])}")
    print(f"Time range: {result['time'][0]} -> {result['time'][-1]}")
    print(f"Volumes [m3]: {BOX_VOLUMES}")
    print(f"Initial DIC [mmol C m-3]: {initial_dic}")
    print(f"Final DIC [mmol C m-3]:   {result['dic'][-1, :]}")
    print(f"Final density [kg m-3]:   {result['rho'][-1, :]}")
    print(f"Final DIC density part:   {result['rho_dic_contribution'][-1, :]}")
    print(f"Final pH:                 {result['pH'][-1, :]}")
    print(f"Final pCO2 [µatm]:        {result['pco2'][-1, :]}")
    print(f"Wind mean [m s-1]:        {np.nanmean(result['wind_m_s']):.3f}")
    print(
        f"Piston velocity mean shelf/ocean [m d-1]: "
        f"{np.nanmean(result['piston_velocity_m_d'][:, 0]):.4f}, "
        f"{np.nanmean(result['piston_velocity_m_d'][:, 2]):.4f}"
    )
    print(f"TA0_UMOLKG:                    {TA0_UMOLKG}  (not calibrated)")
    print(f"BIO_PUMP_ON:                   {BIO_PUMP_ON}")
    print(f"BIO_NCP_MULTIPLIER:            {BIO_NCP_MULTIPLIER}")
    print(f"AIRSEA_K_MULTIPLIER:           {AIRSEA_K_MULTIPLIER}")
    print(f"BACKGROUND_MIXING_BASE_D1:     {BACKGROUND_MIXING_BASE_D1}")
    print(f"SURFACE_DEEP_CONNECTIVITY_MULTIPLIER: {SURFACE_DEEP_CONNECTIVITY_MULTIPLIER}")
    print(f"DEEP_EXPORT_ON:                {DEEP_EXPORT_ON}")
    print(f"DEEP_EXPORT_BASE_D1:           {DEEP_EXPORT_BASE_D1}")
    print(f"DEEP_EXPORT_MAX_D1:            {DEEP_EXPORT_MAX_D1}")
    print(f"SHELF_BOTTOM_BOUNDARY_DIC:     {SHELF_BOTTOM_BOUNDARY_DIC}")
    print(f"OCEAN_DEEP_BOUNDARY_DIC:       {OCEAN_DEEP_BOUNDARY_DIC}")
    print(f"Total positive deep carbon export: {total_deep_export_mmol/1e15:.3f} x10^15 mmol C over simulation")
    print(f"Monthly ΔpCO2 r-value:         {r_value:.4f}")
    print(f"Monthly ΔpCO2 RMSE:            {rmse:.3f} µatm")
    print(f"Monthly ΔpCO2 bias model-obs:  {bias:.3f} µatm")
    print(f"Monthly ΔpCO2 model range:     {model_range_text}")

    print("\nMonthly ΔpCO2 metrics by region:")
    if len(region_metrics) == 0:
        print("  not available")
    else:
        for region, m in region_metrics.items():
            if not np.isfinite(m.get("rmse", np.nan)):
                print(f"  {region}: not enough data")
                continue
            print(
                f"  {region}: "
                f"r = {m['r_value']:.4f}, "
                f"RMSE = {m['rmse']:.2f}, "
                f"bias = {m['bias']:.2f}, "
                f"model range = [{m['model_min']:.2f}, {m['model_max']:.2f}], "
                f"obs range = [{m['obs_min']:.2f}, {m['obs_max']:.2f}], "
                f"n = {m['n']}"
            )

    if not PLOT_ONLY_SOCAT_COMPARISON:
        plot_four_box_time_series(result["time"], result["temp"], "Temperature forcing", "Temperature (°C)", "tab:red", "temperature_forcing_four_boxes.png")
        plot_four_box_time_series(result["time"], result["sal"], "Salinity forcing", "Salinity", "tab:orange", "salinity_forcing_four_boxes.png")
        plot_wind_and_piston_velocity(result["time"], result["wind_m_s"], result["piston_velocity_m_d"], "wind_and_piston_velocity.png")
        plot_four_box_time_series(result["time"], result["dic"], "DIC", "DIC (mmol C m$^{-3}$)", "tab:blue", "dic_four_boxes.png")
        plot_density_diagnostics(result["time"], result["rho"], result["rho_no_dic"], result["rho_dic_contribution"])
        plot_four_box_time_series(result["time"], result["pH"], "pH", "pH", "tab:green", "pH_four_boxes.png")
        plot_combined_exchange_fluxes(result["time"], result["combined_exchange_fluxes"], "combined_exchange_fluxes.png")
        plot_exchange_rate_components(result["time"], result["total_exchange_rates"], result["density_exchange_rates"], result["background_exchange_rates"], "exchange_rate_components.png")
        plot_airsea_tendency(result["time"], result["airsea_tendency"], "airsea_dic_tendency.png")
        plot_bio_tendency(result["time"], result["bio_tendency"], "bio_dic_tendency.png")
        plot_deep_export_diagnostics(result["time"], result["deep_export_rates"], result["deep_export_tendency"], result["deep_carbon_export_mmol_d"], "deep_export_diagnostics.png")

    plot_surface_pco2(result["time"], result["pco2"], recad, "surface_pco2_compared_with_recad.png")
    plot_surface_delta_pco2(result["time"], result["pco2"], recad, "surface_delta_pco2_compared_with_recad.png")
    plot_monthly_delta_pco2(result["time"], result["pco2"], recad, "monthly_delta_pco2_compared_with_recad.png")

    print("\nDone.")
    print(f"Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
