# Shelf Carbon Modeling

A modular Python repository for prototyping a **surface-ocean shelf carbon box model** before transfer to operational-style workflows (e.g., DHI MIKE / Ecolab).

## Project documentation

- Overleaf project notes: https://www.overleaf.com/read/qvfpkvxcncrd#6ebd3d

## What this repository is for

This repo is a sandbox for testing and refining carbon-process formulations in a controlled environment. It is designed to:

- test alternative equation structures quickly,
- isolate process contributions (physics, chemistry, biology),
- compare sensitivity to forcing and parameters,
- generate diagnostics/figures that support model interpretation,
- prepare clean formulations that can be translated to MIKE/Ecolab components.

In short: this is a **development and validation workspace** for simplified shelf-carbon dynamics.

---

## Model scope and conceptual setup

### State variables in the coupled model

The main coupled model integrates the following prognostic state:

- `DIC` (dissolved inorganic carbon),
- `TA` (total alkalinity),
- `LDOC` (labile dissolved organic carbon),
- `SDOC` (semi-labile dissolved organic carbon),
- `RDOC` (refractory dissolved organic carbon).

Carbonate species (`CO2*`, `HCO3-`, `CO3--`) and `pH` are diagnosed from `(DIC, TA, T, S)` each step.

### Process families represented

1. **Air–sea exchange**
   - CO2 flux driven by seawater vs atmospheric equilibrium CO2 difference.
   - Flux converted to mixed-layer concentration tendency using dynamic MLD.

2. **Carbonate chemistry**
   - Temperature/salinity-dependent solubility and equilibrium constants.
   - DIC partitioned into carbonate species and pH is solved iteratively.

3. **Biology and DOC cycling**
   - Light-limited production removes DIC and partitions carbon into DOC pools.
   - Remineralization and DOC aging pathways return carbon toward inorganic form.

4. **Seasonal mixed-layer dynamics**
   - Seasonal MLD can be enabled/disabled.
   - During **deepening** (`dMLD/dt > 0`), entrainment is represented as explicit deep-source plus dilution terms.
   - During **shoaling** (`dMLD/dt < 0`), a sinking/export diagnostic (`F_sink_DIC`) is reported.

---

## Repository structure

```text
Shelf_Carbon_Modeling/
├── README.md
├── main_model/
│   ├── main.py
│   ├── main_comparison.py
│   ├── parameters.py
│   ├── state.py
│   └── modules/
│       ├── gas_exchange.py
│       ├── carbonate_solver.py
│       ├── biology.py
│       ├── plotting.py
│       └── Light_Parameter.py
└── test_models/
    ├── air_sea_co2_exchange_model.py
    ├── air_sea_exchange_with_carbonate_solver.py
    ├── air_sea_exchange_with_easy_biology_est.py
    ├── air_sea_exchange_with_doc_speciation_model.py
    └── test_plotting.py
```

### `main_model/` (integrated model)

- `main.py`
  - Primary single-run entry point.
  - Runs simulation and writes diagnostic plots to `results/`.
- `main_comparison.py`
  - Core integrator and process coupling logic.
  - Implements seasonal forcing functions, RHS tendencies, and output diagnostics dictionary.
- `parameters.py`
  - Centralized dataclass parameter store for forcing, chemistry, biology, entrainment, numerics, and initial conditions.
- `modules/gas_exchange.py`
  - Gas transfer parameterization and air–sea CO2 tendency calculation.
- `modules/carbonate_solver.py`
  - CO2 solubility, salinity-scaled TA utilities, DIC/TA speciation solver, and initialization helpers.
- `modules/biology.py`
  - DOC production, remineralization, and pool-transfer tendencies.
- `modules/plotting.py`
  - Diagnostics plots, output overview plots, biology-comparison plotting, and entrainment fitting visualization.
- `modules/Light_Parameter.py`
  - Utility to calibrate peaked production-light parameters (`Pmax`, `K_L`, `n`, `K_I`) from NPZDO-style reference behavior and write fitted values into `parameters.py`.

### `test_models/` (process-isolated prototypes)

These scripts intentionally isolate single process combinations for debugging and interpretation:

- `air_sea_co2_exchange_model.py`: physical gas-exchange baseline,
- `air_sea_exchange_with_carbonate_solver.py`: gas exchange + carbonate chemistry,
- `air_sea_exchange_with_easy_biology_est.py`: gas exchange + reduced biology,
- `air_sea_exchange_with_doc_speciation_model.py`: DOC-speciation-oriented variant,
- `test_plotting.py`: plotting utility checks/examples.

---

## Forcing and configuration overview

Configured in `main_model/parameters.py`:

- **Atmospheric and physical setup**: salinity, wind speed, atmospheric pCO2.
- **Temperature forcing**: min/max temperature and peak day.
- **Light forcing**: winter/summer light, peak day, phase shift, and shape sharpness.
- **MLD forcing**: fixed or seasonal mixed-layer depth, seasonal extrema and phase.
- **Deep entrainment endmembers**: deep DIC/TA/LDOC/SDOC/RDOC concentrations.
- **Biological and DOC kinetics**: growth scaling, production-light coefficients, partition fractions, remineralization and aging rates.
- **Run controls**: simulation years, output interval, and plot-window selection.

---

## How to run

From repository root:

```bash
python -m main_model.main
```

This will:

1. run the coupled simulation,
2. print integration success/failure,
3. save figures to `results/`:
   - `main_model_diagnostics.png`,
   - `main_model_outputs_overview.png`,
   - `entrainment_fitting_plot.png`.

If possible in your OS/session, the script will also attempt to open a generated plot automatically.

---

## Key outputs returned by the solver

`run(...)` in `main_comparison.py` returns a dictionary containing, among others:

- time vectors (`t_s`, `t_days`),
- forcings (`T_C`, `Light`, `MLD`, `dMLD_dt`),
- carbonate state (`CO2`, `HCO3`, `CO3`, `pH`, `pCO2_sw`, `delta_pCO2`),
- carbon pools (`DIC`, `TA`, `LDOC`, `SDOC`, `RDOC`, `DOC`),
- fluxes/tendencies (`F_ex`, `fprod`, `fremin`, entrainment source and dilution terms),
- diagnostics (`frac_CO2`, `frac_HCO3`, `frac_CO3`, `F_sink_DIC`).

These outputs are designed to support both mechanistic interpretation and plotting/report workflows.

---

## Typical workflow for development

1. Start with one of the `test_models` scripts to isolate behavior.
2. Modify equations/parameters and validate sensitivities.
3. Port stable changes to the coupled `main_model`.
4. Run integrated simulations and inspect plots/diagnostics.
5. Record assumptions and implications for MIKE/Ecolab translation.

---

## Notes on intent and boundaries

- This is a **research/prototyping** codebase, not an operational forecasting system.
- Simplicity and process transparency are prioritized over ecological complexity.
- The architecture is kept modular so each process family can be independently improved.

## Team and course metadata

- **Period:** Spring 2026
- **ECTS:** 5
- **Evaluation:** Pass/Not Pass (presentation in modeller meeting)
- **Supervisors:** Ken H. Andersen, Andre Visser, Anders Erichsen
- **Students:** Teo Vecchini, Matteo von Houwald

## Core references

- MIKE user documentation.
- Tsunogai, S., Watanabe, S., & Sato, T. (1999). *Is there a “continental shelf pump” for the absorption of atmospheric CO2?* Tellus B, 51(3), 701–712.
- Yool, A., & Fasham, M. J. R. (2001). *An examination of the “continental shelf pump” in an open ocean general circulation model.* Global Biogeochemical Cycles, 15(4), 831–844.
