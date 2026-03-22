# extremeloss documentation

`extremeloss` is a Python library for **extreme-loss estimation**, **peaks-over-threshold modeling**, and **tail-risk diagnostics**.

It is designed to sit alongside:

- `lossmodels` for loss-distribution and aggregate-loss modeling
- `risksim` for simulation of losses, contracts, and portfolios

## Scope

The current MVP is organized around three layers:

1. **Estimation**
   - empirical exceedance probabilities
   - empirical VaR and TVaR
   - importance-sampling estimators for tail probability, VaR, and TVaR
2. **Extreme value modeling**
   - exceedance extraction
   - generalized Pareto fitting
   - peaks-over-threshold workflows
   - Hill and Pickands tail-index estimators
   - threshold diagnostics
3. **Analytics**
   - summary tables for extreme losses
   - return periods and return levels
   - lightweight plotting helpers

## Documentation map

- [Getting started](guides/getting-started.md)
- [Package overview](guides/package-overview.md)
- [Design and ecosystem fit](guides/design.md)
- [Examples overview](examples/README.md)
- [API reference](api/reference.md)

## First example

```python
import numpy as np
from extremeloss import estimate_tail_probability, fit_pot

rng = np.random.default_rng(123)
losses = rng.lognormal(mean=2.0, sigma=0.9, size=20_000)

result = estimate_tail_probability(losses, threshold=100.0)
print(result.summary())

fit = fit_pot(losses, threshold=50.0)
print(fit.summary())
print(fit.var(0.995))
```
