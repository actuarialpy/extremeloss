"""Importance-sampling demo for extreme tail quantities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import lognorm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import (  # noqa: E402
    effective_sample_size,
    estimate_exceedance_curve_is,
    estimate_mean_is,
    estimate_tail_probability_is,
    estimate_tvar_is,
    estimate_var_is,
    importance_sampling_diagnostics,
    stabilize_weights,
)


def lognormal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    scale = np.exp(mean)
    return lognorm.pdf(x, s=sigma, scale=scale)


def main() -> None:
    rng = np.random.default_rng(7)

    target_mean = 2.0
    target_sigma = 0.6
    proposal_mean = 2.4
    proposal_sigma = 0.9

    losses = rng.lognormal(mean=proposal_mean, sigma=proposal_sigma, size=40_000)
    raw_weights = lognormal_pdf(losses, target_mean, target_sigma) / lognormal_pdf(
        losses, proposal_mean, proposal_sigma
    )
    weights = stabilize_weights(raw_weights, clip_quantile=0.995)

    threshold = 80.0
    q = 0.995

    mean_result = estimate_mean_is(losses, weights)
    tail_result = estimate_tail_probability_is(losses, weights, threshold=threshold)
    var_result = estimate_var_is(losses, weights, q=q)
    tvar_result = estimate_tvar_is(losses, weights, q=q)
    curve = estimate_exceedance_curve_is(losses, weights, thresholds=[40.0, 60.0, 80.0, 100.0])

    print(f"Effective sample size: {effective_sample_size(weights):.2f}")
    print("Weight diagnostics")
    print(importance_sampling_diagnostics(weights))
    print()
    print("Importance-sampling mean estimate")
    print(mean_result.summary())
    print()
    print("Importance-sampling tail estimate")
    print(tail_result.summary())
    print()
    print(f"IS VaR({q:.3f}): {var_result.estimate:.4f}")
    print(f"IS TVaR({q:.3f}): {tvar_result.estimate:.4f}")
    print()
    print("Weighted exceedance curve")
    for u, p in zip(curve["thresholds"], curve["probabilities"], strict=True):
        print(f"  u={u:7.2f}  P(X > u)={p:.6f}")


if __name__ == "__main__":
    main()
