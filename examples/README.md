# Examples directory

These scripts are intentionally small and dependency-light.

## Files

### Core workflows

- `empirical_tail_analysis.py`
- `pot_gpd_workflow.py`
- `importance_sampling_demo.py`
- `diagnostic_plots.py`

### Extended workflows

- `conditional_mc_demo.py`
- `block_maxima_gev_workflow.py`
- `bootstrap_uncertainty_demo.py`
- `integration_helpers_demo.py`

## Usage

From the repository root:

```bash
python examples/empirical_tail_analysis.py
python examples/pot_gpd_workflow.py
python examples/importance_sampling_demo.py
python examples/conditional_mc_demo.py
python examples/block_maxima_gev_workflow.py
python examples/bootstrap_uncertainty_demo.py
python examples/integration_helpers_demo.py
python examples/diagnostic_plots.py
```

The scripts prepend `src/` to `sys.path`, so they can be run directly from a fresh clone of the repo.
