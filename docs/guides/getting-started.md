# Getting started

## Installation

Editable install:

```bash
pip install -e .
```

Development install:

```bash
pip install -e .[dev]
```

Run the test suite:

```bash
pytest -q
```

## Package layout

```text
src/extremeloss/
├── estimation/
├── evt/
├── analytics/
├── utils/
├── results.py
├── protocols.py
└── plotting.py
```

## Main concepts

### Array-first API

Most functions accept a one-dimensional array of losses:

```python
import numpy as np
from extremeloss import estimate_var

losses = np.array([5.0, 7.0, 9.0, 15.0, 40.0])
result = estimate_var(losses, q=0.95)
print(result.estimate)
```

### Model-like inputs

Some functions also accept objects implementing `sample(size)`.
That matches the integration style used in `risksim`.

```python
import numpy as np
from extremeloss import estimate_tail_probability

class LognormalModel:
    def __init__(self, mean: float, sigma: float, seed: int = 123):
        self.mean = mean
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int) -> np.ndarray:
        return self.rng.lognormal(self.mean, self.sigma, size=size)

model = LognormalModel(mean=2.0, sigma=0.7)
result = estimate_tail_probability(model, threshold=50.0, size=100_000)
print(result.summary())
```

## Typical workflow

1. Obtain losses from data or simulation.
2. Estimate empirical tail quantities.
3. Fit an EVT model above a threshold.
4. Use the fitted tail model for extreme-region VaR, TVaR, and return-level calculations.
5. Review threshold and stability diagnostics.

## Recommended starting points

- `estimate_tail_probability`
- `estimate_var`
- `estimate_tvar`
- `fit_pot`
- `threshold_diagnostic_table`
- `extreme_loss_summary`
