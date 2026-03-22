"""Basic empirical tail analysis for simulated losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import (  # noqa: E402
    empirical_tvar,
    empirical_var,
    estimate_tail_probability,
    exceedance_probability,
    extreme_loss_summary,
)
from extremeloss.estimation import exceedance_curve  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(12345)
    losses = rng.lognormal(mean=2.1, sigma=0.85, size=25_000)

    threshold = 75.0
    q = 0.99

    tail_result = estimate_tail_probability(losses, threshold=threshold)
    var_99 = empirical_var(losses, q)
    tvar_99 = empirical_tvar(losses, q)
    curve = exceedance_curve(losses, thresholds=np.array([25.0, 50.0, 75.0, 100.0, 150.0]))
    summary = extreme_loss_summary(losses, thresholds=np.array([50.0, 75.0, 100.0]))

    print("Empirical tail probability summary")
    print(tail_result.summary())
    print()
    print(f"Exceedance probability above {threshold:.1f}: {exceedance_probability(losses, threshold):.6f}")
    print(f"Empirical VaR({q:.3f}): {var_99:.4f}")
    print(f"Empirical TVaR({q:.3f}): {tvar_99:.4f}")
    print()
    print("Exceedance curve")
    for u, p in zip(curve["thresholds"], curve["probabilities"], strict=True):
        print(f"  u={u:7.2f}  P(X > u)={p:.6f}")
    print()
    print("Extreme loss summary")
    print(summary)


if __name__ == "__main__":
    main()
