"""Conditional Monte Carlo demo using precomputed conditional summaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import estimate_tail_probability_cmc, estimate_tvar_cmc  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(2026)

    conditional_probs = np.clip(rng.beta(a=0.8, b=30.0, size=5_000), 0.0, 1.0)
    conditional_tail_means = rng.lognormal(mean=4.3, sigma=0.25, size=5_000)

    tail_result = estimate_tail_probability_cmc(conditional_probs, threshold=100.0)
    tvar_result = estimate_tvar_cmc(conditional_tail_means, q=0.995, threshold=100.0)

    print("Conditional Monte Carlo tail probability")
    print(tail_result.summary())
    print()
    print("Conditional Monte Carlo TVaR")
    print(tvar_result.summary())


if __name__ == "__main__":
    main()
