from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import importlib.util

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PARAM_DIR = BASE_DIR / "parameter_files"


def create_timestamped_results(prefix: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def load_parameter_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_recad_path() -> Path:
    return DATA_DIR / "ReCAD_pco2_transect_timeseries.npz"
