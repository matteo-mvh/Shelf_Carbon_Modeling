from __future__ import annotations

# ============================================================
# FOUR-BOX DIC-DOC MODEL WITH COPERNICUS + WIND FORCING
# ============================================================
#
# Boxes:
#   Box 1 = shelf surface (0-5 m)
#   Box 2 = shelf bottom  (5-100 m)
#   Box 3 = ocean surface (0-5 m)
#   Box 4 = ocean deep    (5-100 m)
#
# State variables per box:
#   DIC, LDOC, SDOC, RDOC  [mmol C m-3]
#
# DOC concept:
#   DIC --PP(L)--> LDOC
#   LDOC --aging--> SDOC --aging--> RDOC
#   LDOC, SDOC, RDOC --remineralization--> DIC
#
# Tunable process multipliers:
#   TOTAL_PROD_MULTIPLIER   multiplies total production DIC -> LDOC
#   TOTAL_REMIN_MULTIPLIER  multiplies total DOC remineralization to DIC
#   AIRSEA_K_MULTIPLIER     multiplies wind-dependent air-sea exchange
#   BOX4_EXPORT_DIC_MMOL_M3_D removes DIC from Box 4
#
# Requirements:
#   pip install numpy pandas xarray matplotlib copernicusmarine cartopy requests gsw
# ============================================================

from pathlib import Path
from typing import Dict, Tuple
import json
import hashlib

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
        return Path(__file__).resolve().parents[2]
    return Path.cwd()


BASE_DIR = get_base_dir()
DATA_DIR = BASE_DIR / "copernicus_cache"
WIND_DIR = BASE_DIR / "open_meteo_wind"
OUT_DIR = BASE_DIR / "four_box_dic_doc_output"
CARBON_TS_DIR = BASE_DIR / "carbon_timeseries"
RECAD_NPZ_PATH = CARBON_TS_DIR / "ReCAD_pco2_transect_timeseries.npz"

for folder in [DATA_DIR, WIND_DIR, OUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# SIMULATION / DOWNLOAD PERIOD
# ------------------------------------------------------------

START_DATE = "2000-01-01"
END_DATE   = "2002-12-31"
START_DATETIME = f"{START_DATE}T00:00:00"
END_DATETIME   = f"{END_DATE}T00:00:00"


# ------------------------------------------------------------
# SPATIAL REGION
# ------------------------------------------------------------

LON_MIN = -76.2
LON_MAX = -71.0
LAT_MIN = 36.4
LAT_MAX = 37.6
LON_SPLIT = -74.5

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
# Initial state [mmol C m-3]
# ------------------------------------------------------------
# Surface DIC can optionally be initialized from ReCAD/SOCAT pCO2.
# DOC pools are simple first-guess values and can be changed here.

USE_RECAD_INITIAL_SURFACE_DIC = True

INITIAL_DIC_FALLBACK = np.array([2050.0, 2140.0, 2050.0, 2200.0], dtype=float)
SHELF_BOTTOM_DIC_OFFSET = 70.0
OCEAN_DEEP_DIC_OFFSET = 120.0

INITIAL_LDOC = np.array([5.0, 2.0, 4.0, 1.0], dtype=float)
INITIAL_SDOC = np.array([18.0, 10.0, 16.0, 8.0], dtype=float)
INITIAL_RDOC = np.array([58.0, 50.0, 55.0, 48.0], dtype=float)

STATE_NAMES = ["DIC", "LDOC", "SDOC", "RDOC"]
DIC, LDOC, SDOC, RDOC = 0, 1, 2, 3
NSTATE = 4
NBOX = 4


# ------------------------------------------------------------
# Air-sea CO2 settings
# ------------------------------------------------------------

PCO2_AIR_UATM = 420.0
AIRSEA_K_MULTIPLIER = 0.60


# ------------------------------------------------------------
# Box 4 DIC export sink
# ------------------------------------------------------------

BOX4_EXPORT_DIC_MMOL_M3_D = 0.0


# ------------------------------------------------------------
# Density / DIC-density settings
# ------------------------------------------------------------

MOLAR_MASS_C_KG_PER_MOL = 12.0107e-3
DIC_DENSITY_CORRECTION_ON = True


# ------------------------------------------------------------
# Density-driven + background exchange tuning
# ------------------------------------------------------------

EXCHANGE_BASE_RATE_D1 = {
    (0, 1): 0.00008,   # B1 <-> B2 shelf vertical
    (0, 2): 0.00012,   # B1 <-> B3 surface horizontal
    (1, 3): 0.00008,   # B2 <-> B4 bottom horizontal
    (2, 3): 0.00004,   # B3 <-> B4 ocean vertical
}

RHO_THRESHOLD = 0.00
RHO_FRICTION = 0.80
STABLE_VERTICAL_FACTOR = 0.03
UNSTABLE_VERTICAL_MULTIPLIER = 2.0
VERTICAL_PAIR_DEPTH_DIFF_M = 10.0
MAX_EXCHANGE_FRACTION_D1 = 0.0002

BACKGROUND_MIXING_BASE_D1 = 0.00002
WIND_MIXING_REF_M_S = 8.0
WIND_MIXING_ALPHA = 1.0
DIC_DIFF_REF_MMOL_M3 = 100.0
DIC_DIFF_FACTOR_MIN = 0.0
DIC_DIFF_FACTOR_MAX = 3.0
MAX_TOTAL_EXCHANGE_FRACTION_D1 = 0.0005


# ------------------------------------------------------------
# DIC-DOC process parameters
# ------------------------------------------------------------
# Production follows the same seasonal/light shape as in your PP script:
#   1.5 * P * I/(I+100) * day_len * bloom_factor
# Here P is replaced by DOC_PROD_SCALE_MMOL_M3_D and a tunable multiplier.

TOTAL_PROD_MULTIPLIER = 1.0
TOTAL_REMIN_MULTIPLIER = 1.0

DOC_PROD_SCALE_MMOL_M3_D = 3.0
PROD_LIGHT_HALF_SAT_W_M2 = 100.0
LIGHT_SURFACE_W_M2 = 1000.0
LIGHT_EXTINCTION_M_1 = 0.10

# Baseline rates kept from your previous DIC-DOC setup.
GAMMA_LDOC_D1 = 0.40
GAMMA_SDOC_D1 = 0.05
GAMMA_RDOC_D1 = 8.64e-06
LAMBDA_LDOC_TO_SDOC_D1 = 0.05
LAMBDA_SDOC_TO_RDOC_D1 = 0.02

# If True, remineralization is computed as one total remin pool and
# fractioned back over LDOC/SDOC/RDOC based on their baseline contributions.
FRACTION_TOTAL_REMIN_BY_POOL = True


# ------------------------------------------------------------
# Carbonate settings
# ------------------------------------------------------------

TA0_UMOLKG = 2350.0
S0 = 35.0
PH_MIN = 6.5
PH_MAX = 9.2


# ============================================================
# AUTOMATIC CALIBRATION PARAMETER RANGES
# ============================================================
# Edit these lists to change the search grid.
# Total runs = len(TA) * len(PROD) * len(REMIN) * len(AIRSEA) * len(BOX4)

CALIBRATE_TO_RECAD = True
PRINT_TOP_N_CALIBRATION_RESULTS = 15

CALIBRATION_TA0_GRID = [2300.0, 2325.0, 2350.0, 2375.0, 2400.0]
CALIBRATION_TOTAL_PROD_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
CALIBRATION_TOTAL_REMIN_GRID = [0.50, 0.75, 1.00, 1.25, 1.50]
CALIBRATION_AIRSEA_MULT_GRID = [0.35, 0.55, 0.75, 1.00]
CALIBRATION_BOX4_EXPORT_GRID = [0.0, 0.0005, 0.0010, 0.0020]

SAVE_FIGURES = False


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
BOX_LAYOUT = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
PAIR_LIST = [(0, 1), (0, 2), (1, 3), (2, 3)]
PAIR_LABELS = {
    (0, 1): "Box 1 ↔ Box 2",
    (0, 2): "Box 1 ↔ Box 3",
    (1, 3): "Box 2 ↔ Box 4",
    (2, 3): "Box 3 ↔ Box 4",
}

BOX_AREAS = np.array([SHELF_AREA_M2, SHELF_AREA_M2, OCEAN_AREA_M2, OCEAN_AREA_M2], dtype=float)
BOX_THICKNESSES = np.array([SURFACE_THICKNESS_M, DEEP_THICKNESS_M, SURFACE_THICKNESS_M, DEEP_THICKNESS_M], dtype=float)
BOX_MID_DEPTHS = np.array([2.5, 52.5, 2.5, 52.5], dtype=float)
BOX_VOLUMES = BOX_AREAS * BOX_THICKNESSES
BOX_LONS = np.array([
    0.5 * (LON_MIN + LON_SPLIT),
    0.5 * (LON_MIN + LON_SPLIT),
    0.5 * (LON_SPLIT + LON_MAX),
    0.5 * (LON_SPLIT + LON_MAX),
], dtype=float)
BOX_LATS = np.full(4, 0.5 * (LAT_MIN + LAT_MAX), dtype=float)


# ============================================================
# 4) PHYSICS / CARBON CHEMISTRY
# ============================================================

def pressure_dbar_from_depth(depth_m: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return gsw.p_from_z(-np.asarray(depth_m, dtype=float), np.asarray(lat, dtype=float))


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
    Sc = 2116.8 - 136.25*T + 4.7353*T**2 - 0.092307*T**3 + 0.0007555*T**4
    return np.maximum(Sc, 1.0)


def gas_transfer_velocity_m_per_day(wind_m_s: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    wind_m_s = np.maximum(np.asarray(wind_m_s, dtype=float), 0.0)
    Sc = schmidt_number_co2(temp_c)
    k_cm_h = 0.251 * wind_m_s**2 * (Sc / 660.0) ** (-0.5)
    return AIRSEA_K_MULTIPLIER * k_cm_h * 0.01 * 24.0


def carbonate_constants(temp_c: np.ndarray, sal: np.ndarray) -> Dict[str, np.ndarray]:
    T = np.asarray(temp_c, dtype=float) + 273.15
    S = np.maximum(np.asarray(sal, dtype=float), 0.1)

    lnK0 = (-58.0931 + 90.5069 * (100.0 / T) + 22.2940 * np.log(T / 100.0)
            + S * (0.027766 - 0.025888 * (T / 100.0) + 0.0050578 * (T / 100.0) ** 2))
    K0 = np.exp(lnK0)

    pK1 = 3633.86 / T - 61.2172 + 9.67770 * np.log(T) - 0.011555 * S + 0.0001152 * S * S
    pK2 = 471.78 / T + 25.9290 - 3.16967 * np.log(T) - 0.01781 * S + 0.0001122 * S * S
    K1 = 10.0 ** (-pK1)
    K2 = 10.0 ** (-pK2)

    sqrtS = np.sqrt(S)
    lnKb = ((-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S * sqrtS - 0.0996 * S * S) / T
            + 148.0248 + 137.1942 * sqrtS + 1.62142 * S
            + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * np.log(T)
            + 0.053105 * sqrtS * T)
    Kb = np.exp(lnKb)
    BT = 0.0004157 * S / 35.0

    lnKw = (148.96502 - 13847.26 / T - 23.6521 * np.log(T)
            + (118.67 / T - 5.977 + 1.0495 * np.log(T)) * np.sqrt(S) - 0.01615 * S)
    Kw = np.exp(lnKw)

    return {"K0": K0, "K1": K1, "K2": K2, "Kb": Kb, "Kw": Kw, "BT": BT}


def carbonate_speciation(dic_mmol_m3: np.ndarray, temp_c: np.ndarray, sal: np.ndarray, rho_kg_m3: np.ndarray) -> Dict[str, np.ndarray]:
    dic_mmol_m3 = np.maximum(np.asarray(dic_mmol_m3, dtype=float), 1e-9)
    sal = np.asarray(sal, dtype=float)
    rho_kg_m3 = np.asarray(rho_kg_m3, dtype=float)

    dic_umolkg = dic_mmol_m3 * 1000.0 / np.maximum(rho_kg_m3, 1e-12)
    TA_umolkg = TA0_UMOLKG * sal / max(S0, 1e-12)

    CT = dic_umolkg * 1e-6
    TA = TA_umolkg * 1e-6

    const = carbonate_constants(temp_c, sal)
    K0, K1, K2, Kb, Kw, BT = (const[k] for k in ["K0", "K1", "K2", "Kb", "Kw", "BT"])

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
        fmid = alk_from_H(10.0 ** (-mid)) - TA
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
    rho = density_teos10_with_dic(temp, sal, dic)
    return float(carbonate_speciation(dic, temp, sal, rho)["pCO2_uatm"][box_index])


def solve_dic_for_target_pco2(box_index: int, target_pco2: float, temp_value: float, sal_value: float,
                              dic_min: float = 1500.0, dic_max: float = 2400.0) -> float:
    lo, hi = dic_min, dic_max
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
    df.to_csv(out_file, index=False)
    print(f"Saved wind file:\n{out_file}")
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
    return ds.rename(rename_map) if rename_map else ds


def download_if_needed(variable: str, out_path: Path) -> None:
    if out_path.exists() and not REDOWNLOAD_RAW_COPERNICUS:
        print(f"\nUsing existing raw file:\n{out_path}")
        return
    if out_path.exists() and REDOWNLOAD_RAW_COPERNICUS:
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
    vals = da["lat"].values
    return da.sel(lat=slice(lat_min, lat_max)) if vals[0] <= vals[-1] else da.sel(lat=slice(lat_max, lat_min))


def subset_lon_range(da: xr.DataArray, lon_min: float, lon_max: float) -> xr.DataArray:
    vals = da["lon"].values
    return da.sel(lon=slice(lon_min, lon_max)) if vals[0] <= vals[-1] else da.sel(lon=slice(lon_max, lon_min))


def get_spatial_subset(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    ds = standardize_coords(ds)
    da = ds[var_name]
    da = subset_lat_range(da, LAT_MIN, LAT_MAX)
    da = subset_lon_range(da, LON_MIN, LON_MAX)
    return da


def depth_subset_with_fallback(da: xr.DataArray, depth_min: float, depth_max: float, name: str) -> xr.DataArray:
    if "depth" not in da.dims:
        print(f"\nWARNING: no depth dimension for {name}; using data as-is.")
        return da

    depth = da["depth"]
    selected = da.where((depth >= depth_min) & (depth <= depth_max), drop=True)
    if selected.sizes.get("depth", 0) > 0:
        print(f"{name}: using depths {selected['depth'].values}")
        return selected

    selected = da.where(depth >= depth_min, drop=True)
    if selected.sizes.get("depth", 0) > 0:
        print(f"WARNING: {name}: using all depths >= {depth_min}: {selected['depth'].values}")
        return selected

    print(f"WARNING: {name}: no valid depths found. Using all available depths.")
    return da


def safe_mean_box(sub_da: xr.DataArray, box_name: str) -> xr.DataArray:
    if sub_da.size == 0:
        raise ValueError(f"{box_name}: selected data is empty.")
    dims = [d for d in ["depth", "lat", "lon"] if d in sub_da.dims]
    out = sub_da.mean(dim=dims, skipna=True)
    if np.all(~np.isfinite(out.values)):
        raise ValueError(f"{box_name}: mean result is all NaN.")
    return out


def build_box_timeseries(da: xr.DataArray, var_label: str) -> xr.DataArray:
    print(f"\nBuilding 4-box forcing for {var_label}")
    print("Available depths:", da["depth"].values)
    print("Latitude values:", da["lat"].values)
    print("Longitude range:", float(da["lon"].min()), "to", float(da["lon"].max()))

    shelf_da = subset_lon_range(da, LON_MIN, LON_SPLIT - 1e-6)
    ocean_da = subset_lon_range(da, LON_SPLIT, LON_MAX)

    box1 = safe_mean_box(depth_subset_with_fallback(shelf_da, SURFACE_MIN_DEPTH, SURFACE_MAX_DEPTH, f"{var_label} Box 1"), f"{var_label} Box 1")
    box2 = safe_mean_box(depth_subset_with_fallback(shelf_da, BOTTOM_MIN_DEPTH, BOTTOM_MAX_DEPTH, f"{var_label} Box 2"), f"{var_label} Box 2")
    box3 = safe_mean_box(depth_subset_with_fallback(ocean_da, SURFACE_MIN_DEPTH, SURFACE_MAX_DEPTH, f"{var_label} Box 3"), f"{var_label} Box 3")
    box4 = safe_mean_box(depth_subset_with_fallback(ocean_da, BOTTOM_MIN_DEPTH, BOTTOM_MAX_DEPTH, f"{var_label} Box 4"), f"{var_label} Box 4")

    out = xr.concat([box1, box2, box3, box4], dim="box")
    out = out.transpose("time", "box")
    return out.assign_coords(box=np.arange(4))


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
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    ax.scatter(lon2d[shelf_mask], lat2d[shelf_mask], s=25, color="tab:green", alpha=0.7, transform=ccrs.PlateCarree(), label="Shelf grid cells")
    ax.scatter(lon2d[ocean_mask], lat2d[ocean_mask], s=25, color="tab:orange", alpha=0.7, transform=ccrs.PlateCarree(), label="Ocean grid cells")
    ax.scatter([WIND_LON], [WIND_LAT], s=90, color="tab:red", marker="*", edgecolor="black", transform=ccrs.PlateCarree(), label="Open-Meteo wind point", zorder=5)
    ax.plot([LON_SPLIT, LON_SPLIT], [LAT_MIN, LAT_MAX], "k--", linewidth=2, transform=ccrs.PlateCarree(), label=f"Split at lon = {LON_SPLIT}")
    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN], [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN], color="red", linewidth=2, transform=ccrs.PlateCarree(), label="Selected averaging box")
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
        forcing_cache_path.unlink()

    download_if_needed(THETAO_VAR, thetao_path)
    download_if_needed(SO_VAR, so_path)

    ds_t = xr.open_dataset(thetao_path)
    ds_s = xr.open_dataset(so_path)

    da_t = get_spatial_subset(ds_t, THETAO_VAR)
    da_s = get_spatial_subset(ds_s, SO_VAR)

    lon_vals = da_t["lon"].values
    lat_vals = da_t["lat"].values
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

    np.savez_compressed(forcing_cache_path, time=time, temp=temp, sal=sal,
                        lon_vals=lon_vals, lat_vals=lat_vals, metadata=np.array(json.dumps(metadata)))
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
        for key in ["pco2", "lon", "time_datetime64"]:
            if key not in keys:
                raise KeyError(f"ReCAD file has no '{key}'. Keys: {keys}")
        pco2 = np.asarray(data["pco2"], dtype=float)
        lon = np.asarray(data["lon"], dtype=float)
        time = np.asarray(data["time_datetime64"], dtype="datetime64[ns]")

    shelf_mask = lon < LON_SPLIT
    ocean_mask = lon >= LON_SPLIT
    shelf_mean = np.nanmean(pco2[:, shelf_mask], axis=1)
    ocean_mean = np.nanmean(pco2[:, ocean_mask], axis=1)

    t0 = np.datetime64(START_DATETIME, "ns")
    t1 = np.datetime64(END_DATETIME, "ns")
    time_mask = (time >= t0) & (time <= t1)

    return {
        "time": time[time_mask],
        "shelf_mean": shelf_mean[time_mask],
        "ocean_mean": ocean_mean[time_mask],
        "shelf_delta": PCO2_AIR_UATM - shelf_mean[time_mask],
        "ocean_delta": PCO2_AIR_UATM - ocean_mean[time_mask],
    }


def monthly_series(time: np.ndarray, values: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(time)).resample("MS").mean()


def get_recad_monthly_delta(recad: Dict[str, np.ndarray] | None) -> Tuple[pd.Series | None, pd.Series | None]:
    if recad is None or len(recad["time"]) == 0:
        return None, None
    return monthly_series(recad["time"], recad["shelf_delta"]), monthly_series(recad["time"], recad["ocean_delta"])


# ============================================================
# 8) MODEL PROCESS FUNCTIONS
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
    rho_i, rho_j = rho[i], rho[j]
    if abs(rho_i - rho_j) < RHO_THRESHOLD:
        return 1.0
    denser_depth = BOX_MID_DEPTHS[i] if rho_i > rho_j else BOX_MID_DEPTHS[j]
    lighter_depth = BOX_MID_DEPTHS[j] if rho_i > rho_j else BOX_MID_DEPTHS[i]
    return STABLE_VERTICAL_FACTOR if denser_depth > lighter_depth else UNSTABLE_VERTICAL_MULTIPLIER


def background_mixing_rate(i: int, j: int, dic: np.ndarray, wind_m_s: float) -> float:
    dic_diff = abs(dic[i] - dic[j])
    wind_factor = 1.0 + WIND_MIXING_ALPHA * (max(wind_m_s, 0.0) / WIND_MIXING_REF_M_S) ** 2
    dic_factor = np.clip(dic_diff / max(DIC_DIFF_REF_MMOL_M3, 1e-12), DIC_DIFF_FACTOR_MIN, DIC_DIFF_FACTOR_MAX)
    return BACKGROUND_MIXING_BASE_D1 * wind_factor * dic_factor


def combined_exchange_rate(i: int, j: int, dic: np.ndarray, rho: np.ndarray, wind_m_s: float) -> Tuple[float, float, float]:
    pair = tuple(sorted((i, j)))
    base_rate = EXCHANGE_BASE_RATE_D1[pair]

    drho_signed = rho[i] - rho[j]
    drho = abs(drho_signed)
    if drho <= RHO_THRESHOLD:
        q_density = 0.0
    else:
        rho_drive = (drho - RHO_THRESHOLD) / (drho + RHO_FRICTION + 1e-12)
        q_density = base_rate * rho_drive * gravitational_stability_factor(i, j, rho)
        q_density = min(q_density, MAX_EXCHANGE_FRACTION_D1)

    q_background = background_mixing_rate(i, j, dic, wind_m_s)
    q_total = min(q_density + q_background, MAX_TOTAL_EXCHANGE_FRACTION_D1)
    return q_total, q_density, q_background


def apply_conservative_exchange(y: np.ndarray, i: int, j: int, q_total: float) -> np.ndarray:
    tendencies = np.zeros_like(y)
    Q_total = q_total * min(BOX_VOLUMES[i], BOX_VOLUMES[j])
    for s in range(NSTATE):
        tendencies[s, i] += Q_total / BOX_VOLUMES[i] * (y[s, j] - y[s, i])
        tendencies[s, j] += Q_total / BOX_VOLUMES[j] * (y[s, i] - y[s, j])
    return tendencies


def day_length_relative(day_of_year: float) -> float:
    day_length = 12.0 + 6.0 * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.0)
    return np.clip((day_length - 6.0) / 12.0, 0.0, 1.0)


def production_bloom_factor(day_of_year: float) -> float:
    return min(
        0.03
        + 0.165 * (1.0 + np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.0))
        + np.exp(-0.5 * ((day_of_year - 80.0) / 20.0) ** 2)
        + 0.3 * np.exp(-0.5 * ((day_of_year - 250.0) / 15.0) ** 2),
        1.0,
    )


def mean_light_in_layer(depth_top: float, depth_bottom: float) -> float:
    k = LIGHT_EXTINCTION_M_1
    z0 = max(0.0, depth_top)
    z1 = max(z0 + 1e-9, depth_bottom)
    return LIGHT_SURFACE_W_M2 * (np.exp(-k * z0) - np.exp(-k * z1)) / (k * (z1 - z0))


def box_mean_light() -> np.ndarray:
    return np.array([
        mean_light_in_layer(0.0, SURFACE_THICKNESS_M),
        mean_light_in_layer(SURFACE_THICKNESS_M, SURFACE_THICKNESS_M + DEEP_THICKNESS_M),
        mean_light_in_layer(0.0, SURFACE_THICKNESS_M),
        mean_light_in_layer(SURFACE_THICKNESS_M, SURFACE_THICKNESS_M + DEEP_THICKNESS_M),
    ], dtype=float)


BOX_LIGHT_W_M2 = box_mean_light()


def doc_process_tendency(current_time: np.datetime64, y: np.ndarray) -> np.ndarray:
    """
    DIC-DOC process model:
      DIC --production--> LDOC
      LDOC --lambda_S--> SDOC
      SDOC --lambda_R--> RDOC
      LDOC/SDOC/RDOC --remin--> DIC

    Production is light/seasonally limited following the uploaded PP shape.
    Remineralization is one tunable total term, then fractioned over LDOC/SDOC/RDOC
    according to baseline pool-specific contributions.
    """
    out = np.zeros_like(y)

    ts = pd.Timestamp(current_time)
    day = float(ts.dayofyear)
    day_len = day_length_relative(day)
    bloom = production_bloom_factor(day)

    light_lim = BOX_LIGHT_W_M2 / (BOX_LIGHT_W_M2 + PROD_LIGHT_HALF_SAT_W_M2 + 1e-12)
    prod = TOTAL_PROD_MULTIPLIER * 1.5 * DOC_PROD_SCALE_MMOL_M3_D * light_lim * day_len * bloom

    # No production if DIC is already very depleted.
    prod = np.minimum(prod, 0.80 * y[DIC, :])

    out[DIC, :] -= prod
    out[LDOC, :] += prod

    # Aging terms
    aging_L_to_S = LAMBDA_LDOC_TO_SDOC_D1 * y[LDOC, :]
    aging_S_to_R = LAMBDA_SDOC_TO_RDOC_D1 * y[SDOC, :]

    out[LDOC, :] -= aging_L_to_S
    out[SDOC, :] += aging_L_to_S

    out[SDOC, :] -= aging_S_to_R
    out[RDOC, :] += aging_S_to_R

    # Remineralization terms
    base_L = GAMMA_LDOC_D1 * y[LDOC, :]
    base_S = GAMMA_SDOC_D1 * y[SDOC, :]
    base_R = GAMMA_RDOC_D1 * y[RDOC, :]
    base_total = base_L + base_S + base_R

    if FRACTION_TOTAL_REMIN_BY_POOL:
        total_remin = TOTAL_REMIN_MULTIPLIER * base_total
        fL = np.divide(base_L, base_total, out=np.zeros_like(base_L), where=base_total > 0)
        fS = np.divide(base_S, base_total, out=np.zeros_like(base_S), where=base_total > 0)
        fR = np.divide(base_R, base_total, out=np.zeros_like(base_R), where=base_total > 0)
        remin_L = fL * total_remin
        remin_S = fS * total_remin
        remin_R = fR * total_remin
    else:
        remin_L = TOTAL_REMIN_MULTIPLIER * base_L
        remin_S = TOTAL_REMIN_MULTIPLIER * base_S
        remin_R = TOTAL_REMIN_MULTIPLIER * base_R

    out[LDOC, :] -= remin_L
    out[SDOC, :] -= remin_S
    out[RDOC, :] -= remin_R
    out[DIC, :] += remin_L + remin_S + remin_R

    return out


def build_initial_dic(temp0: np.ndarray, sal0: np.ndarray, recad: Dict[str, np.ndarray] | None) -> np.ndarray:
    initial = INITIAL_DIC_FALLBACK.copy()

    if not USE_RECAD_INITIAL_SURFACE_DIC or recad is None or len(recad["time"]) == 0:
        return initial

    shelf_pco2 = float(pd.Series(recad["shelf_mean"]).dropna().iloc[0])
    ocean_pco2 = float(pd.Series(recad["ocean_mean"]).dropna().iloc[0])

    shelf_dic = solve_dic_for_target_pco2(0, shelf_pco2, float(temp0[0]), float(sal0[0]))
    ocean_dic = solve_dic_for_target_pco2(2, ocean_pco2, float(temp0[2]), float(sal0[2]))

    initial[0] = shelf_dic
    initial[1] = shelf_dic + SHELF_BOTTOM_DIC_OFFSET
    initial[2] = ocean_dic
    initial[3] = ocean_dic + OCEAN_DEEP_DIC_OFFSET

    print("\nInitial DIC estimated from first ReCAD/SOCAT pCO2:")
    print(f"  Shelf pCO2 target: {shelf_pco2:.2f} µatm -> DIC {initial[0]:.2f} mmol C m-3")
    print(f"  Ocean pCO2 target: {ocean_pco2:.2f} µatm -> DIC {initial[2]:.2f} mmol C m-3")
    print(f"  Initial DIC vector: {initial}")
    return initial


def build_initial_state(temp0: np.ndarray, sal0: np.ndarray, recad: Dict[str, np.ndarray] | None) -> np.ndarray:
    y0 = np.zeros((NSTATE, NBOX), dtype=float)
    y0[DIC, :] = build_initial_dic(temp0, sal0, recad)
    y0[LDOC, :] = INITIAL_LDOC
    y0[SDOC, :] = INITIAL_SDOC
    y0[RDOC, :] = INITIAL_RDOC
    return y0


def run_dic_doc_model(time: np.ndarray, temp_forcing: np.ndarray, sal_forcing: np.ndarray,
                      wind_m_s: np.ndarray, initial_state: np.ndarray) -> Dict[str, np.ndarray]:
    n_time = len(time)

    state = np.full((n_time, NSTATE, NBOX), np.nan, dtype=float)
    pH = np.full((n_time, NBOX), np.nan, dtype=float)
    pco2 = np.full((n_time, NBOX), np.nan, dtype=float)
    rho = np.full((n_time, NBOX), np.nan, dtype=float)
    rho_no_dic = np.full((n_time, NBOX), np.nan, dtype=float)
    rho_dic_contribution = np.full((n_time, NBOX), np.nan, dtype=float)
    piston_velocity = np.full((n_time, NBOX), np.nan, dtype=float)
    airsea_tendency = np.full((n_time, NBOX), np.nan, dtype=float)
    doc_tendency = np.full((n_time, NSTATE, NBOX), np.nan, dtype=float)

    combined_exchange_fluxes = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    total_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    density_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    background_exchange_rates = {pair: np.full(n_time, np.nan, dtype=float) for pair in PAIR_LIST}
    box4_export_tendency = np.full(n_time, np.nan, dtype=float)

    state[0, :, :] = np.asarray(initial_state, dtype=float)

    for k in range(n_time - 1):
        dt_days = float((time[k + 1] - time[k]) / np.timedelta64(1, "D"))
        temp_now = temp_forcing[k, :]
        sal_now = sal_forcing[k, :]
        wind_now = wind_m_s[k]
        y = state[k, :, :].copy()
        dic_now = y[DIC, :]

        rho_base_now = density_teos10_with_dic(temp_now, sal_now, np.zeros(NBOX))
        rho_now = density_teos10_with_dic(temp_now, sal_now, dic_now)
        rho_dic_now = rho_now - rho_base_now
        k_now = gas_transfer_velocity_m_per_day(np.full(NBOX, wind_now), temp_now)
        carb_now = carbonate_speciation(dic_now, temp_now, sal_now, rho_now)

        rho[k, :] = rho_now
        rho_no_dic[k, :] = rho_base_now
        rho_dic_contribution[k, :] = rho_dic_now
        piston_velocity[k, :] = k_now
        pH[k, :] = carb_now["pH"]
        pco2[k, :] = carb_now["pCO2_uatm"]

        dydt = np.zeros((NSTATE, NBOX), dtype=float)

        for pair in PAIR_LIST:
            i, j = pair
            q_total, q_density, q_background = combined_exchange_rate(i, j, dic_now, rho_now, wind_now)
            dydt += apply_conservative_exchange(y, i, j, q_total)

            total_exchange_rates[pair][k] = q_total
            density_exchange_rates[pair][k] = q_density
            background_exchange_rates[pair][k] = q_background

            Q_total = q_total * min(BOX_VOLUMES[i], BOX_VOLUMES[j])
            dic_sign = np.sign(dic_now[i] - dic_now[j])
            if dic_sign == 0:
                dic_sign = np.sign(rho_now[i] - rho_now[j])
            combined_exchange_fluxes[pair][k] = dic_sign * Q_total

        # Air-sea exchange acts on DIC only in the two surface boxes.
        airsea_tendency[k, 0] = air_sea_dic_tendency_from_carb(carb_now, 0, k_now[0], SURFACE_THICKNESS_M)
        airsea_tendency[k, 2] = air_sea_dic_tendency_from_carb(carb_now, 2, k_now[2], SURFACE_THICKNESS_M)
        airsea_tendency[k, 1] = 0.0
        airsea_tendency[k, 3] = 0.0
        dydt[DIC, 0] += airsea_tendency[k, 0]
        dydt[DIC, 2] += airsea_tendency[k, 2]

        # DIC-DOC processes.
        proc = doc_process_tendency(time[k], y)
        doc_tendency[k, :, :] = proc
        dydt += proc

        # Optional Box 4 DIC export.
        box4_export_tendency[k] = BOX4_EXPORT_DIC_MMOL_M3_D
        dydt[DIC, 3] -= BOX4_EXPORT_DIC_MMOL_M3_D

        y_next = y + dt_days * dydt
        if not np.all(np.isfinite(y_next)):
            print("\nERROR: non-finite state produced.")
            print("time index:", k)
            print("time:", time[k])
            print("state current:", y)
            print("dydt:", dydt)
            print("temp:", temp_now)
            print("sal:", sal_now)
            print("rho:", rho_now)
            print("wind:", wind_now)
            print("pco2:", pco2[k, :])
            raise FloatingPointError("State became non-finite.")

        state[k + 1, :, :] = np.clip(y_next, 1e-12, None)

    # Last diagnostics
    temp_last = temp_forcing[-1, :]
    sal_last = sal_forcing[-1, :]
    wind_last = wind_m_s[-1]
    dic_last = state[-1, DIC, :]
    rho_base_last = density_teos10_with_dic(temp_last, sal_last, np.zeros(NBOX))
    rho_last = density_teos10_with_dic(temp_last, sal_last, dic_last)
    k_last = gas_transfer_velocity_m_per_day(np.full(NBOX, wind_last), temp_last)
    carb_last = carbonate_speciation(dic_last, temp_last, sal_last, rho_last)

    rho[-1, :] = rho_last
    rho_no_dic[-1, :] = rho_base_last
    rho_dic_contribution[-1, :] = rho_last - rho_base_last
    piston_velocity[-1, :] = k_last
    pH[-1, :] = carb_last["pH"]
    pco2[-1, :] = carb_last["pCO2_uatm"]
    airsea_tendency[-1, :] = 0.0
    doc_tendency[-1, :, :] = doc_process_tendency(time[-1], state[-1, :, :])
    box4_export_tendency[-1] = BOX4_EXPORT_DIC_MMOL_M3_D

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
        "state": state,
        "dic": state[:, DIC, :],
        "ldoc": state[:, LDOC, :],
        "sdoc": state[:, SDOC, :],
        "rdoc": state[:, RDOC, :],
        "pH": pH,
        "pco2": pco2,
        "rho": rho,
        "rho_no_dic": rho_no_dic,
        "rho_dic_contribution": rho_dic_contribution,
        "combined_exchange_fluxes": combined_exchange_fluxes,
        "total_exchange_rates": total_exchange_rates,
        "density_exchange_rates": density_exchange_rates,
        "background_exchange_rates": background_exchange_rates,
        "airsea_tendency": airsea_tendency,
        "doc_tendency": doc_tendency,
        "box4_export_tendency": box4_export_tendency,
    }


# ============================================================
# 9) CALIBRATION
# ============================================================

def monthly_delta_rmse(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> float:
    if recad is None or len(recad["time"]) == 0:
        return np.nan

    model_shelf = monthly_series(result["time"], PCO2_AIR_UATM - result["pco2"][:, 0])
    model_ocean = monthly_series(result["time"], PCO2_AIR_UATM - result["pco2"][:, 2])
    obs_shelf, obs_ocean = get_recad_monthly_delta(recad)

    errors = []
    for model, obs in [(model_shelf, obs_shelf), (model_ocean, obs_ocean)]:
        if obs is None:
            continue
        df = pd.concat([model.rename("model"), obs.rename("obs")], axis=1, join="inner").dropna()
        if len(df) > 0:
            errors.append((df["model"].values - df["obs"].values) ** 2)

    if len(errors) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(np.concatenate(errors))))


def monthly_delta_bias(result: Dict[str, np.ndarray], recad: Dict[str, np.ndarray] | None) -> float:
    if recad is None or len(recad["time"]) == 0:
        return np.nan

    model_shelf = monthly_series(result["time"], PCO2_AIR_UATM - result["pco2"][:, 0])
    model_ocean = monthly_series(result["time"], PCO2_AIR_UATM - result["pco2"][:, 2])
    obs_shelf, obs_ocean = get_recad_monthly_delta(recad)

    errors = []
    for model, obs in [(model_shelf, obs_shelf), (model_ocean, obs_ocean)]:
        if obs is None:
            continue
        df = pd.concat([model.rename("model"), obs.rename("obs")], axis=1, join="inner").dropna()
        if len(df) > 0:
            errors.append(df["model"].values - df["obs"].values)

    if len(errors) == 0:
        return np.nan
    return float(np.nanmean(np.concatenate(errors)))


def calibrate_to_recad(time: np.ndarray, temp_forcing: np.ndarray, sal_forcing: np.ndarray,
                       wind_m_s: np.ndarray, recad: Dict[str, np.ndarray] | None) -> np.ndarray:
    global TA0_UMOLKG, TOTAL_PROD_MULTIPLIER, TOTAL_REMIN_MULTIPLIER
    global AIRSEA_K_MULTIPLIER, BOX4_EXPORT_DIC_MMOL_M3_D

    if recad is None or len(recad["time"]) == 0:
        print("\nCalibration skipped because ReCAD/SOCAT data is unavailable.")
        return build_initial_state(temp_forcing[0, :], sal_forcing[0, :], recad)

    print("\n============================================================")
    print("Calibrating against monthly ReCAD/SOCAT ΔpCO2")
    print("============================================================")
    print("\nCalibration grid:")
    print(f"  TA0_UMOLKG: {CALIBRATION_TA0_GRID}")
    print(f"  TOTAL_PROD_MULTIPLIER: {CALIBRATION_TOTAL_PROD_GRID}")
    print(f"  TOTAL_REMIN_MULTIPLIER: {CALIBRATION_TOTAL_REMIN_GRID}")
    print(f"  AIRSEA_K_MULTIPLIER: {CALIBRATION_AIRSEA_MULT_GRID}")
    print(f"  BOX4_EXPORT_DIC_MMOL_M3_D: {CALIBRATION_BOX4_EXPORT_GRID}")

    n_total = (len(CALIBRATION_TA0_GRID) * len(CALIBRATION_TOTAL_PROD_GRID) *
               len(CALIBRATION_TOTAL_REMIN_GRID) * len(CALIBRATION_AIRSEA_MULT_GRID) *
               len(CALIBRATION_BOX4_EXPORT_GRID))
    print(f"\nTotal calibration runs: {n_total}")

    original = (TA0_UMOLKG, TOTAL_PROD_MULTIPLIER, TOTAL_REMIN_MULTIPLIER,
                AIRSEA_K_MULTIPLIER, BOX4_EXPORT_DIC_MMOL_M3_D)

    rows = []
    run_id = 0

    for ta in CALIBRATION_TA0_GRID:
        for prod_mult in CALIBRATION_TOTAL_PROD_GRID:
            for remin_mult in CALIBRATION_TOTAL_REMIN_GRID:
                for airsea_mult in CALIBRATION_AIRSEA_MULT_GRID:
                    for box4_export in CALIBRATION_BOX4_EXPORT_GRID:
                        run_id += 1
                        TA0_UMOLKG = float(ta)
                        TOTAL_PROD_MULTIPLIER = float(prod_mult)
                        TOTAL_REMIN_MULTIPLIER = float(remin_mult)
                        AIRSEA_K_MULTIPLIER = float(airsea_mult)
                        BOX4_EXPORT_DIC_MMOL_M3_D = float(box4_export)

                        initial_state = build_initial_state(temp_forcing[0, :], sal_forcing[0, :], recad)
                        result = run_dic_doc_model(time, temp_forcing, sal_forcing, wind_m_s, initial_state)
                        rmse = monthly_delta_rmse(result, recad)
                        bias = monthly_delta_bias(result, recad)

                        rows.append({
                            "run_id": run_id,
                            "rmse": rmse,
                            "bias_model_minus_obs": bias,
                            "TA0_UMOLKG": TA0_UMOLKG,
                            "TOTAL_PROD_MULTIPLIER": TOTAL_PROD_MULTIPLIER,
                            "TOTAL_REMIN_MULTIPLIER": TOTAL_REMIN_MULTIPLIER,
                            "AIRSEA_K_MULTIPLIER": AIRSEA_K_MULTIPLIER,
                            "BOX4_EXPORT_DIC_MMOL_M3_D": BOX4_EXPORT_DIC_MMOL_M3_D,
                            "initial_dic_1": initial_state[DIC, 0],
                            "initial_dic_2": initial_state[DIC, 1],
                            "initial_dic_3": initial_state[DIC, 2],
                            "initial_dic_4": initial_state[DIC, 3],
                            "final_dic_1": result["dic"][-1, 0],
                            "final_dic_2": result["dic"][-1, 1],
                            "final_dic_3": result["dic"][-1, 2],
                            "final_dic_4": result["dic"][-1, 3],
                        })

                        print(
                            f"[{run_id:4d}/{n_total:4d}] "
                            f"RMSE={rmse:8.3f} | bias={bias:8.3f} | "
                            f"TA0={TA0_UMOLKG:7.1f} | prod={TOTAL_PROD_MULTIPLIER:5.2f} | "
                            f"remin={TOTAL_REMIN_MULTIPLIER:5.2f} | airsea={AIRSEA_K_MULTIPLIER:5.2f} | "
                            f"B4export={BOX4_EXPORT_DIC_MMOL_M3_D:8.5f}"
                        )

    df = pd.DataFrame(rows).sort_values("rmse")
    calibration_path = OUT_DIR / "calibration_results.csv"
    df.to_csv(calibration_path, index=False)
    print(f"\nSaved calibration results to:\n{calibration_path}")
    print("\nBest calibration results:")
    print(df.head(PRINT_TOP_N_CALIBRATION_RESULTS).to_string(index=False))

    best = df.iloc[0]
    if not np.isfinite(best["rmse"]):
        print("\nWARNING: best RMSE was not finite. Restoring original parameters.")
        (TA0_UMOLKG, TOTAL_PROD_MULTIPLIER, TOTAL_REMIN_MULTIPLIER,
         AIRSEA_K_MULTIPLIER, BOX4_EXPORT_DIC_MMOL_M3_D) = original
        return build_initial_state(temp_forcing[0, :], sal_forcing[0, :], recad)

    TA0_UMOLKG = float(best["TA0_UMOLKG"])
    TOTAL_PROD_MULTIPLIER = float(best["TOTAL_PROD_MULTIPLIER"])
    TOTAL_REMIN_MULTIPLIER = float(best["TOTAL_REMIN_MULTIPLIER"])
    AIRSEA_K_MULTIPLIER = float(best["AIRSEA_K_MULTIPLIER"])
    BOX4_EXPORT_DIC_MMOL_M3_D = float(best["BOX4_EXPORT_DIC_MMOL_M3_D"])

    best_initial_state = build_initial_state(temp_forcing[0, :], sal_forcing[0, :], recad)

    print("\nSelected best parameter set:")
    print(f"  TA0_UMOLKG = {TA0_UMOLKG}")
    print(f"  TOTAL_PROD_MULTIPLIER = {TOTAL_PROD_MULTIPLIER}")
    print(f"  TOTAL_REMIN_MULTIPLIER = {TOTAL_REMIN_MULTIPLIER}")
    print(f"  AIRSEA_K_MULTIPLIER = {AIRSEA_K_MULTIPLIER}")
    print(f"  BOX4_EXPORT_DIC_MMOL_M3_D = {BOX4_EXPORT_DIC_MMOL_M3_D}")
    print(f"  best RMSE = {float(best['rmse']):.3f} µatm")
    print(f"  best bias = {float(best['bias_model_minus_obs']):.3f} µatm")

    return best_initial_state


# ============================================================
# 10) PLOTTING HELPERS
# ============================================================

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
    fig, axes = make_four_box_figure(title, y_label)
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
        signed_q_percent_d = 100.0 * combined_exchange_fluxes[pair] / min(BOX_VOLUMES[i], BOX_VOLUMES[j])
        ax.plot(time, signed_q_percent_d, linewidth=1.2)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(PAIR_LABELS[pair])
        ax.set_ylabel("Signed combined exchange\n(% of smaller box d$^{-1}$)")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.92, f"+ = {BOX_SHORT[i]} -> {BOX_SHORT[j]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        disable_y_offset(ax)
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_exchange_rate_components(time: np.ndarray, total_rates: Dict[Tuple[int, int], np.ndarray],
                                  density_rates: Dict[Tuple[int, int], np.ndarray],
                                  background_rates: Dict[Tuple[int, int], np.ndarray], out_name: str | None = None):
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


def add_delta_labels(ax):
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax.text(0.01, 0.92, "Ingassing / ocean uptake\n$\\Delta pCO_2 > 0$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.text(0.01, 0.08, "Outgassing\n$\\Delta pCO_2 < 0$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))


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


def plot_monthly_delta_pco2(time: np.ndarray, pco2: np.ndarray, recad: Dict[str, np.ndarray] | None = None, out_name: str | None = None):
    model_shelf = monthly_series(time, PCO2_AIR_UATM - pco2[:, 0])
    model_ocean = monthly_series(time, PCO2_AIR_UATM - pco2[:, 2])
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


def plot_airsea_tendency(time: np.ndarray, airsea_tendency: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Wind-dependent air-sea DIC tendency", fontsize=14)
    for ax, b, title in [(axes[0], 0, "Box 1: Shelf surface"), (axes[1], 2, "Box 3: Ocean surface")]:
        ax.plot(time, airsea_tendency[:, b], linewidth=1.3)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time")
    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_doc_process_tendency(time: np.ndarray, doc_tendency: np.ndarray, out_name: str | None = None):
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("DIC-DOC process tendencies", fontsize=14)
    for s, name in enumerate(STATE_NAMES):
        ax = axes[s]
        for b in range(NBOX):
            ax.plot(time, doc_tendency[:, s, b], linewidth=1.2, label=BOX_SHORT[b])
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(name)
        ax.set_ylabel("mmol C m$^{-3}$ d$^{-1}$")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    if SAVE_FIGURES and out_name is not None:
        fig.savefig(OUT_DIR / out_name, dpi=200, bbox_inches="tight")
    plt.show()


def plot_density_diagnostics(time: np.ndarray, rho: np.ndarray, rho_dic_contribution: np.ndarray):
    plot_four_box_time_series(time, rho, "Density including TEOS-10 + DIC mass correction", "Density (kg m$^{-3}$)", "tab:brown", "density_total_four_boxes.png")
    plot_four_box_time_series(time, rho_dic_contribution, "DIC contribution to density", "Density contribution (kg m$^{-3}$)", "tab:purple", "density_dic_contribution_four_boxes.png")


# ============================================================
# 11) MAIN
# ============================================================

def main():
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
        initial_state = calibrate_to_recad(time, temp_forcing, sal_forcing, wind_m_s, recad)
    else:
        initial_state = build_initial_state(temp_forcing[0, :], sal_forcing[0, :], recad)

    print("\n============================================================")
    print("Running final four-box DIC-DOC model")
    print("============================================================")
    result = run_dic_doc_model(time, temp_forcing, sal_forcing, wind_m_s, initial_state)
    rmse = monthly_delta_rmse(result, recad)
    bias = monthly_delta_bias(result, recad)

    print("\nForcing / simulation summary:")
    print(f"Dataset: {DATASET_ID}")
    print(f"Time steps: {len(result['time'])}")
    print(f"Time range: {result['time'][0]} -> {result['time'][-1]}")
    print(f"Volumes [m3]: {BOX_VOLUMES}")
    print(f"Initial DIC [mmol C m-3]: {initial_state[DIC, :]}")
    print(f"Initial LDOC/SDOC/RDOC:")
    print(f"  LDOC: {initial_state[LDOC, :]}")
    print(f"  SDOC: {initial_state[SDOC, :]}")
    print(f"  RDOC: {initial_state[RDOC, :]}")
    print(f"Final DIC [mmol C m-3]:   {result['dic'][-1, :]}")
    print(f"Final LDOC [mmol C m-3]:  {result['ldoc'][-1, :]}")
    print(f"Final SDOC [mmol C m-3]:  {result['sdoc'][-1, :]}")
    print(f"Final RDOC [mmol C m-3]:  {result['rdoc'][-1, :]}")
    print(f"Final density [kg m-3]:   {result['rho'][-1, :]}")
    print(f"Final pH:                 {result['pH'][-1, :]}")
    print(f"Final pCO2 [µatm]:        {result['pco2'][-1, :]}")
    print(f"TA0_UMOLKG:                    {TA0_UMOLKG}")
    print(f"TOTAL_PROD_MULTIPLIER:         {TOTAL_PROD_MULTIPLIER}")
    print(f"TOTAL_REMIN_MULTIPLIER:        {TOTAL_REMIN_MULTIPLIER}")
    print(f"AIRSEA_K_MULTIPLIER:           {AIRSEA_K_MULTIPLIER}")
    print(f"BOX4_EXPORT_DIC_MMOL_M3_D:     {BOX4_EXPORT_DIC_MMOL_M3_D}")
    print(f"Monthly ΔpCO2 RMSE:            {rmse:.3f} µatm")
    print(f"Monthly ΔpCO2 bias model-obs:  {bias:.3f} µatm")

    plot_four_box_time_series(result["time"], result["temp"], "Temperature forcing", "Temperature (°C)", "tab:red", "temperature_forcing_four_boxes.png")
    plot_four_box_time_series(result["time"], result["sal"], "Salinity forcing", "Salinity", "tab:orange", "salinity_forcing_four_boxes.png")
    plot_wind_and_piston_velocity(result["time"], result["wind_m_s"], result["piston_velocity_m_d"], "wind_and_piston_velocity.png")
    plot_four_box_time_series(result["time"], result["dic"], "DIC", "DIC (mmol C m$^{-3}$)", "tab:blue", "dic_four_boxes.png")
    plot_four_box_time_series(result["time"], result["ldoc"], "LDOC", "LDOC (mmol C m$^{-3}$)", "tab:green", "ldoc_four_boxes.png")
    plot_four_box_time_series(result["time"], result["sdoc"], "SDOC", "SDOC (mmol C m$^{-3}$)", "tab:olive", "sdoc_four_boxes.png")
    plot_four_box_time_series(result["time"], result["rdoc"], "RDOC", "RDOC (mmol C m$^{-3}$)", "tab:cyan", "rdoc_four_boxes.png")
    plot_density_diagnostics(result["time"], result["rho"], result["rho_dic_contribution"])
    plot_four_box_time_series(result["time"], result["pH"], "pH", "pH", "tab:green", "pH_four_boxes.png")
    plot_combined_exchange_fluxes(result["time"], result["combined_exchange_fluxes"], "combined_exchange_fluxes.png")
    plot_exchange_rate_components(result["time"], result["total_exchange_rates"], result["density_exchange_rates"], result["background_exchange_rates"], "exchange_rate_components.png")
    plot_airsea_tendency(result["time"], result["airsea_tendency"], "airsea_dic_tendency.png")
    plot_doc_process_tendency(result["time"], result["doc_tendency"], "doc_process_tendency.png")
    plot_surface_pco2(result["time"], result["pco2"], recad, "surface_pco2_compared_with_recad.png")
    plot_monthly_delta_pco2(result["time"], result["pco2"], recad, "monthly_delta_pco2_compared_with_recad.png")

    print("\nDone.")
    print(f"Output folder: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
