from dataclasses import dataclass


@dataclass
class State:
    # Prognostic states
    DIC: float  # mol C m^-3 (proxy)
    G: float    # mol glucose m^-3
    TA: float   # mol m^-3

    # Common diagnostics (placeholders)
    pCO2_sw: float = float("nan")  # uatm
    DOC: float = float("nan")      # mol C m^-3
