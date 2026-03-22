# Examples

The `examples/` directory contains small, runnable scripts that demonstrate the current MVP API.

## Included examples

### `empirical_tail_analysis.py`
Basic empirical tail analysis on simulated lognormal losses.

Shows:
- empirical tail probability
- empirical VaR and TVaR
- exceedance curves
- extreme-loss summary output

### `pot_gpd_workflow.py`
Peaks-over-threshold workflow using a GPD fit.

Shows:
- threshold selection grid
- exceedance extraction
- POT fitting
- EVT VaR / TVaR / return level calculations

### `importance_sampling_demo.py`
Simple importance-sampling illustration using a proposal distribution with heavier tails.

Shows:
- normalized weights
- effective sample size
- weighted tail probability
- weighted VaR and TVaR

### `diagnostic_plots.py`
Generates a few quick diagnostic plots and saves them to `examples/output/`.

Shows:
- exceedance curve
- mean excess plot
- Hill curve

## Running the examples

From the repository root:

```bash
python examples/empirical_tail_analysis.py
python examples/pot_gpd_workflow.py
python examples/importance_sampling_demo.py
python examples/diagnostic_plots.py
```
