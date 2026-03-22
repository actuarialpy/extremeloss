# Design and ecosystem fit

## Why this package exists

`lossmodels` already covers general loss modeling.
`risksim` already covers simulation.

`extremeloss` is intended to cover the part that is hardest to estimate well with naive simulation:

- very small tail probabilities
- extreme quantiles
- extreme-region TVaR
- threshold-based tail extrapolation

## Design choices

### 1. Array-first, model-compatible

Low-level functions work directly on NumPy arrays.
Where convenient, estimators also accept model-like objects with `sample(size)`.
This keeps the package usable on its own while still fitting the broader ecosystem.

### 2. Thin public API

Important functions are re-exported at the package root:

```python
from extremeloss import estimate_tail_probability, fit_pot, return_period
```

That mirrors the clean root-level API style used in `risksim` while preserving domain subpackages similar to `lossmodels`.

### 3. Results as first-class objects

Instead of returning only floats, estimation and EVT functions return structured results:

- `TailEstimateResult`
- `GPDFit`
- `ThresholdScan`

This makes the package easier to extend later with richer diagnostics, confidence intervals, and reporting.

## Relationship to `lossmodels`

A typical long-term integration story is:

1. define a severity or aggregate model in `lossmodels`
2. sample from it directly or through `risksim`
3. pass the simulated losses into `extremeloss`

Example shape:

```python
# illustrative only
sev = SomeSeverityModel(...)
losses = sev.sample(size=100_000)
fit = fit_pot(losses, threshold=100_000.0)
```

## Relationship to `risksim`

`risksim` is already built around model-like objects that expose `.sample(size)`.
`extremeloss` follows the same idea, so rare-event and EVT analysis can sit naturally downstream of simulated portfolios.

## Current MVP boundaries

The current scaffold intentionally does not yet include:

- advanced conditional Monte Carlo
- adaptive importance sampling
- block-maxima / GEV workflows
- multivariate extremes
- bootstrap confidence intervals
- production reporting/export layers

Those can be added later without changing the core package shape.
