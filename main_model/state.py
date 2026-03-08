from dataclasses import dataclass


@dataclass
class State:
    # Prognostic states
    DIC: float
    LDOC: float
    SDOC: float
    RDOC: float
    TA: float

    # Diagnostics
    pCO2_sw: float = float("nan")
    DOC: float = float("nan")
