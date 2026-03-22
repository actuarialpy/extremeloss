# Package overview

## `extremeloss.estimation`

Core empirical and weighted estimators.

### Main functions

- `estimate_tail_probability(data, threshold, size=None, alpha=0.05)`
- `estimate_var(data, q, size=None, alpha=0.05)`
- `estimate_tvar(data, q, size=None, alpha=0.05)`
- `estimate_var_tvar(data, q, size=None, alpha=0.05)`
- `estimate_tail_probability_is(losses, weights, threshold, alpha=0.05)`
- `estimate_var_is(losses, weights, q)`
- `estimate_tvar_is(losses, weights, q)`
- `exceedance_probability(losses, threshold)`
- `exceedance_curve(losses, thresholds)`

### Result object

Most estimation functions return `TailEstimateResult` with fields such as:

- `estimate`
- `method`
- `stderr`
- `ci`
- `n`
- `effective_n`
- `threshold`
- `quantile`
- `diagnostics`

## `extremeloss.evt`

Extreme value theory utilities focused on peaks over threshold.

### Main functions

- `extract_exceedances(data, threshold)`
- `fit_gpd(excesses, threshold=0.0, method="mle")`
- `fit_pot(data, threshold, method="mle")`
- `gpd_tail_probability(x, threshold, xi, beta, exceedance_fraction)`
- `gpd_var(p, threshold, xi, beta, exceedance_fraction)`
- `gpd_tvar(p, threshold, xi, beta, exceedance_fraction)`
- `hill_estimator(data, k)`
- `pickands_estimator(data, k)`
- `hill_curve(data, k_grid=None)`
- `mean_excess(data, thresholds)`
- `threshold_diagnostic_table(data, thresholds)`

### EVT result objects

- `GPDFit`
- `ThresholdScan`

`GPDFit` supports:

- `.summary()`
- `.tail_probability(x)`
- `.var(p)`
- `.tvar(p)`
- `.return_level(period)`

## `extremeloss.analytics`

Applied summaries and diagnostics.

### Main functions

- `return_period(probability)`
- `return_level(period, fit)`
- `exceedance_frequency(losses, threshold)`
- `var_tvar_diagnostic_table(losses, quantiles=(...))`
- `extreme_loss_summary(losses, thresholds=None, quantiles=(...))`

## `extremeloss.plotting`

Lightweight plotting wrappers:

- `plot_exceedance_curve`
- `plot_mean_excess`
- `plot_hill_curve`

These return a Matplotlib axis and are meant to be simple helpers, not a full charting framework.
