# MaintPlanAI

Implementation of **Failure-uncertainty-aware joint production and maintenance under shared resources.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

`prepare` generates a controlled scheduling benchmark with six machines, 48 jobs, three families, 14 shifts and independent design/evaluation random tapes. The included lifetime records are generated inputs; job, resource and simulation coefficients are defined in `config.json`.

```bash
python -m maintplanai prepare --config config.json --output runs/full
```

## Training and inference

```bash
python -m maintplanai fit --config config.json --output runs/full
python -m maintplanai run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m maintplanai evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `MaintPlanAI`, `no-risk`, `point`, `two-stage`, `periodic`, `reactive`.

The run command evaluates these controls together.

## Outputs

All main outputs are written below `runs/full/`.

`replications.csv` reports realized cost, downtime, tardiness, job moves, production, kit use, electricity, GHG, scrap and overtime. `action_cards.csv` compares maintain-now, next-shift and continue on common scenarios. `metrics.csv` aggregates risk/service/resource endpoints; `prognostic_metrics.csv` contains horizon-specific ECE and Brier score.

```bash
python -m maintplanai plot --config config.json --output runs/full
```
