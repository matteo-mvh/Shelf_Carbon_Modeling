"""State container for prognostic variables and diagnostics."""

from dataclasses import dataclass


@dataclass
class State:
    DIC: float
    G: float
    pCO2_sw: float = 0.0
    DOC: float = 0.0
