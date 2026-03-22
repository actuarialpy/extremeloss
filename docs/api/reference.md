# API reference

## Top-level exports

```python
from extremeloss import (
    GPDFit,
    TailEstimateResult,
    ThresholdScan,
    effective_sample_size,
    empirical_tvar,
    empirical_var,
    estimate_tail_probability,
    estimate_tail_probability_is,
    estimate_tvar,
    estimate_tvar_is,
    estimate_var,
    estimate_var_is,
    exceedance_probability,
    extract_exceedances,
    extreme_loss_summary,
    fit_gpd,
    fit_pot,
    gpd_tail_probability,
    gpd_tvar,
    gpd_var,
    hill_curve,
    hill_estimator,
    mean_excess,
    pickands_estimator,
    return_level,
    return_period,
    threshold_diagnostic_table,
)
```

## Results module

### `TailEstimateResult`

Dataclass used for empirical and importance-sampling estimators.

#### Fields

- `estimate: float`
- `method: str`
- `stderr: float | None = None`
- `ci: tuple[float, float] | None = None`
- `n: int | None = None`
- `effective_n: float | None = None`
- `threshold: float | None = None`
- `quantile: float | None = None`
- `diagnostics: dict[str, Any] = {}`

#### Methods

- `.summary() -> dict[str, Any]`

### `GPDFit`

Dataclass representing a fitted generalized Pareto model above a threshold.

#### Methods

- `.tail_probability(x: float) -> float`
- `.var(p: float) -> float`
- `.tvar(p: float) -> float`
- `.return_level(period: float) -> float`
- `.summary() -> dict[str, Any]`

### `ThresholdScan`

Container for threshold-diagnostic arrays.

#### Methods

- `.to_dict() -> dict[str, np.ndarray]`

## Estimation module

### Empirical estimators

#### `estimate_tail_probability(data, threshold, *, size=None, alpha=0.05)`
Estimate `P(X > threshold)` from observed or simulated losses.

#### `estimate_var(data, q, *, size=None, alpha=0.05)`
Estimate empirical VaR at quantile `q`.

#### `estimate_tvar(data, q, *, size=None, alpha=0.05)`
Estimate empirical TVaR at quantile `q`.

#### `estimate_var_tvar(data, q, *, size=None, alpha=0.05)`
Return a dictionary containing VaR, TVaR, and tail-probability summaries.

### Importance-sampling estimators

#### `estimate_tail_probability_is(losses, weights, threshold, *, alpha=0.05)`
Weighted estimator for exceedance probability.

#### `estimate_var_is(losses, weights, q)`
Weighted quantile estimator.

#### `estimate_tvar_is(losses, weights, q)`
Weighted TVaR estimator.

#### `effective_sample_size(weights)`
Return the standard importance-sampling ESS based on normalized weights.

## EVT module

### Peaks over threshold

#### `extract_exceedances(data, threshold)`
Return positive exceedances above the threshold.

#### `fit_pot(data, threshold, method="mle")`
Fit a GPD to exceedances extracted from the data.

#### `fit_gpd(excesses, threshold=0.0, method="mle")`
Fit a GPD directly to excess values.

### Tail function helpers

#### `gpd_tail_probability(x, threshold, xi, beta, exceedance_fraction)`
Evaluate the fitted GPD-based exceedance probability at `x`.

#### `gpd_var(p, threshold, xi, beta, exceedance_fraction)`
Return the EVT extrapolated VaR at probability level `p`.

#### `gpd_tvar(p, threshold, xi, beta, exceedance_fraction)`
Return the EVT extrapolated TVaR at probability level `p`.

### Tail-index and threshold diagnostics

#### `hill_estimator(data, k)`
Classic Hill estimator for heavy-tail index.

#### `pickands_estimator(data, k)`
Pickands estimator based on upper-order statistics.

#### `hill_curve(data, k_grid=None)`
Return a grid of Hill estimates.

#### `mean_excess(data, thresholds)`
Return mean residual life values across a threshold grid.

#### `threshold_diagnostic_table(data, thresholds)`
Fit POT models across a threshold grid and summarize stability.

## Analytics module

### `return_period(probability)`
Convert exceedance probability to expected return period.

### `return_level(period, fit)`
Return the fitted extreme level associated with a return period.

### `extreme_loss_summary(losses, *, thresholds=None, quantiles=(...))`
Return a lightweight dictionary of summary statistics and tail diagnostics.
