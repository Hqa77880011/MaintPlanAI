# MaintPlanAI

Implementation of **Failure-uncertainty-aware joint production and maintenance under shared resources.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

`prepare` creates the configured controlled scheduling benchmark: job processing times, qualifications, releases, due dates and families; lifetime-fitting records; and independent design/evaluation random-number tapes. `instance.json` exposes the job schema. `lifetimes.npz` stores per-machine lifetimes and censoring indicators and can be replaced with equivalent prognostic training records. Simulation coefficients are explicit JSON fields.

## Workflow

```bash
python -m maintplanai prepare
python -m maintplanai fit
python -m maintplanai run
python -m maintplanai evaluate
python -m maintplanai plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`machines,jobs,families,shifts,shift_hours` define scale. `technicians,kits,kit_lead_shifts` define shared resources. `candidate_plans,scenarios,risk_weight` control feasible-plan search and mean/CVaR valuation. Processing, setup, maintenance, repair, tardiness, stability and unfinished-work fields define the cost boundary. Power and inventory coefficients separately define electricity, GHG and material accounting. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`replications.csv` reports realized cost, downtime, tardiness, job moves, production, kit use, electricity, GHG, scrap and overtime. `action_cards.csv` compares maintain-now, next-shift and continue on common scenarios. `metrics.csv` aggregates risk/service/resource endpoints; `prognostic_metrics.csv` contains horizon-specific ECE and Brier score.

## Implementation and controls

A censored Weibull likelihood fits failure distributions. Constructive schedule candidates jointly vary maintenance placements and production order, then are ranked by sample-average loss plus CVaR. This finite search is the optimizer implementation. Nominal tasks fit inside shifts; failure-induced completion delays and known busy state carry forward. Only the first shift is executed. Spare replenishment persists between epochs, and stability compares against the previous look-ahead assignment. Idle electricity is included.

Controls: `MaintPlanAI`, `no-risk`, `point`, `two-stage`, `periodic`, `reactive`. The run command evaluates the listed controls together.

```bash
python -m maintplanai.compare --inputs "runs/*/replications.csv" --reference MaintPlanAI --metrics cost downtime greenhouse_gas --unit replication
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

