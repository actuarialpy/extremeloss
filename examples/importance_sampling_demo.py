"""Simple importance-sampling demo for extreme tail quantities."""

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm

from extremeloss import effective_sample_size, estimate_tail_probability_is, estimate_tvar_is, estimate_var_is


def lognormal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    scale = np.exp(mean)
    return lognorm.pdf(x, s=sigma, scale=scale)


rng = np.random.default_rng(7)

# Target distribution: Lognormal(mean=2.0, sigma=0.6)
target_mean = 2.0
target_sigma = 0.6

# Proposal distribution: heavier tail to oversample extreme losses.
proposal_mean = 2.4
proposal_sigma = 0.9

losses = rng.lognormal(mean=proposal_mean, sigma=proposal_sigma, size=40_000)
raw_weights = lognormal_pdf(losses, target_mean, target_sigma) / lognormal_pdf(
    losses, proposal_mean, proposal_sigma
)

threshold = 80.0
q = 0.995

tail_result = estimate_tail_probability_is(losses, raw_weights, threshold=threshold)
var_result = estimate_var_is(losses, raw_weights, q=q)
tvar_result = estimate_tvar_is(losses, raw_weights, q=q)

print(f"Effective sample size: {effective_sample_size(raw_weights):.2f}")
print("Importance-sampling tail estimate")
print(tail_result.summary())
print()
print(f"IS VaR({q:.3f}): {var_result.estimate:.4f}")
print(f"IS TVaR({q:.3f}): {tvar_result.estimate:.4f}")
