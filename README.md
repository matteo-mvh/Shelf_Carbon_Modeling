# Shelf Carbon Modeling

A lightweight repository for developing and testing simplified dissolved inorganic carbon (DIC) transport equations before implementation in **DHI MIKE / Ecolab** workflows.

## Link to PDF Project Documentation

https://www.overleaf.com/read/qvfpkvxcncrd#6ebd3d

## Purpose

This project supports a Spring 2026 special course focused on hydrodynamic modeling of carbon transport on continental shelves. The repository is intended as a practical sandbox to:

- prototype and compare equation formulations,
- perform sensitivity analyses of key physical and biogeochemical parameters,
- prepare transferable model components for integration into MIKE setups,
- document assumptions and recommendations for a stand-alone DIC-oriented Ecolab template.

In short: this is an equation-and-sensitivity development space that feeds into production-style DHI models.

## Course context

The course investigates how atmospheric CO₂ uptake and DIC redistribution are controlled by:

- air–sea exchange,
- advection and diffusion,
- chemical speciation/equilibria,
- seasonal stratification and circulation,
- cross-shelf exchange processes that can temporarily isolate carbon from atmospheric contact.

Model development emphasizes simplified two-dimensional hydrodynamic frameworks to explore first-order controls on DIC transport.

## Learning goals reflected in this repository

The repository work should contribute to:

1. Reviewing and synthesizing literature on vertical and cross-shelf carbon transport.
2. Understanding shelf-region physics and impacts on DIC dynamics.
3. Hands-on MIKE workflow practice.
4. Implementing carbon modules by adapting MIKE Ecolab templates or simple biogeochemical model concepts (e.g., MOPS-like simplifications).
5. Recommending how DIC can be integrated as a permanent stand-alone Ecolab template product.
6. Running cross-shelf DIC transport experiments to estimate temporary oceanic carbon capture and associated timescales.
7. Supporting bi-weekly modeller meetings and presentation outputs.

## Current repository structure

- `README.md` — project overview, model structure, and workflow guidance.
- `test_models/` — process-isolation prototypes used to test one or a few mechanisms at a time.
  - `air_sea_co2_exchange_model.py` — **physical baseline**: isolates air–sea gas exchange with seasonal temperature forcing.
  - `air_sea_exchange_with_carbonate_solver.py` — **carbonate chemistry isolation**: compares simple CO₂-only behavior against explicit carbonate speciation under fixed alkalinity.
  - `air_sea_exchange_with_easy_biology_est.py` — **biology isolation**: adds a simple DIC↔DOC/glucose loop to quantify first-order biological drawdown effects.
- `main_model/` — integrated model where physical + chemical + biological components are solved together in one modular workflow.
  - `main.py` — top-level simulation driver and diagnostics for ON/OFF biology comparison.
  - `parameters.py` — centralized parameter set for forcing, chemistry, biology, and numerics.
  - `state.py` — state container definitions.
  - `modules/`
    - `gas_exchange.py` — gas transfer and air–sea flux tendency functions.
    - `carbonate_solver.py` — carbonate system and TA/DIC speciation utilities.
    - `biology.py` — biological production/remineralization tendencies.
    - `plotting.py` — diagnostics plotting utilities for integrated runs.

## How to interpret `test_models` vs `main_model`

### `test_models`: isolated-factor experiments

The `test_models` folder is intentionally split into targeted experiments where factors are isolated and behavior is easy to diagnose:

- **Physical-only test** to understand air–sea equilibration behavior.
- **Physical + carbonate-speciation test** to isolate chemistry effects on pCO₂ and species partitioning.
- **Physical + simple biology test** to isolate biological drawdown and remineralization impacts.

These scripts are useful for fast sanity checks, conceptual sensitivity scans, and equation debugging before combining everything.

### `main_model`: full coupled experiment

The `main_model` combines the same process families in a single, modular coupled run:

- air–sea gas exchange,
- carbonate equilibrium/speciation,
- biological uptake/remineralization,
- integrated diagnostics and comparison workflows.

Compared to `test_models`, the `main_model` is designed to represent the full system together, with additional forcing and diagnostics in one place.

The current coupled formulation uses **fixed total alkalinity (TA)** diagnosed from salinity and removes **deep-water entrainment terms**, so carbonate chemistry is solved from `(DIC, TA, T, S)` at each step.

## Forcing parameters (current status)

As of now, the primary forcing parameters are:

1. **Temperature** — seasonal forcing that modifies solubility, speciation, and gas transfer sensitivity.
2. **Light** — seasonal forcing used in biological production limitations.
3. **MLD (Mixed Layer Depth)** — controls flux-to-concentration conversion and can be fixed or seasonal with its own toggle.

> Note: one shared `seasonality` switch controls both temperature and light forcing. MLD has a separate seasonality toggle, and its tendency contributes concentration-dilution terms via `-(dh/dt)/h`. Entrainment/source-water exchange is not included in the reduced formulation.

## Typical workflow

1. Define a minimal governing equation set for transport and exchange.
2. Implement or modify a compact test model with clear parameter exposure.
3. Run parameter sweeps (mixing, gas transfer, biology, chemistry, forcing timing, etc.).
4. Diagnose sensitivities and identify robust/fragile assumptions.
5. Translate stable formulations into MIKE/Ecolab-compatible forms.
6. Report findings in modeller meetings and track decisions.

## Scope and boundaries

- This repository is for **rapid model prototyping and sensitivity control**, not a full operational shelf-carbon forecast system.
- Simplicity and transparency are prioritized over high ecological complexity.
- Any MIKE-specific implementation should remain traceable to tested equations developed here.

## Team and course metadata

- **Period:** Spring 2026
- **ECTS:** 5
- **Evaluation:** Pass/Not Pass (presentation in modeller meeting)
- **Supervisors:** Ken H. Andersen, Andre Visser, Anders Erichsen
- **Students:** Teo Vecchini, Matteo von Houwald

## Core references

- MIKE user documentation.
- Tsunogai, S., Watanabe, S., & Sato, T. (1999). *Is there a “continental shelf pump” for the absorption of atmospheric CO₂?* Tellus B, 51(3), 701–712.
- Yool, A., & Fasham, M. J. R. (2001). *An examination of the “continental shelf pump” in an open ocean general circulation model.* Global Biogeochemical Cycles, 15(4), 831–844.
