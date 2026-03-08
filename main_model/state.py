from dataclasses import dataclass


@dataclass
class State:
    # Prognostic states
    DIC: float
    LDOC: float
    SDOC: float
    RDOC: float

    # Diagnostics / prescribed fields
    TA_const: float
    pCO2_sw: float = float("nan")
    DOC: float = float("nan")
