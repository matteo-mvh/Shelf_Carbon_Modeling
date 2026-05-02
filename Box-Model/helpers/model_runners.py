from __future__ import annotations
import importlib
from pathlib import Path


def _load_dic_module():
    return importlib.import_module("WhoNeedsMikeAnyway_DIC")


def _load_dic_doc_module():
    return importlib.import_module("WhoNeedsMikeAnyway_DIC_DOC")


def run_dic_calibration_and_model(results_dir: Path, plot_only_socat: bool = True):
    m = _load_dic_module()
    m.RECAD_NPZ_PATH = m.BASE_DIR / "data" / "ReCAD_pco2_transect_timeseries.npz"
    m.OUT_DIR = results_dir
    m.PARAMETER_FILE = m.BASE_DIR / "parameter_files" / "dic_parameters.py"
    m.PLOT_ONLY_SOCAT_COMPARISON = plot_only_socat
    # Remove biology from DIC-only run.
    m.BIO_NCP_MULTIPLIER = 0.0
    m.SHELF_NCP_SCALE_MMOL_M3_D = 0.0
    m.OCEAN_NCP_SCALE_MMOL_M3_D = 0.0
    m.main()


def run_dic_doc_calibration_and_model(results_dir: Path, plot_only_socat: bool = True):
    m = _load_dic_doc_module()
    m.RECAD_NPZ_PATH = m.BASE_DIR / "data" / "ReCAD_pco2_transect_timeseries.npz"
    m.OUT_DIR = results_dir
    m.PLOT_ONLY_SOCAT_COMPARISON = plot_only_socat
    # Align exchange/calibration closer to DIC reference.
    m.CALIBRATION_GRID_VALUES = getattr(m, "CALIBRATION_GRID_VALUES", {})
    m.main()


def run_dic_doc_full(results_dir: Path):
    m = _load_dic_doc_module()
    m.RECAD_NPZ_PATH = m.BASE_DIR / "data" / "ReCAD_pco2_transect_timeseries.npz"
    m.OUT_DIR = results_dir
    m.PLOT_ONLY_SOCAT_COMPARISON = False
    m.main()


def run_dic_full(results_dir: Path):
    m = _load_dic_module()
    m.RECAD_NPZ_PATH = m.BASE_DIR / "data" / "ReCAD_pco2_transect_timeseries.npz"
    m.OUT_DIR = results_dir
    m.PARAMETER_FILE = m.BASE_DIR / "parameter_files" / "dic_parameters.py"
    m.PLOT_ONLY_SOCAT_COMPARISON = False
    m.BIO_NCP_MULTIPLIER = 0.0
    m.SHELF_NCP_SCALE_MMOL_M3_D = 0.0
    m.OCEAN_NCP_SCALE_MMOL_M3_D = 0.0
    m.main()
