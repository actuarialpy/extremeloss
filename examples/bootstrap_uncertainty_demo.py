"""Bootstrap uncertainty estimation for tail statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import bootstrap_tail_probability, bootstrap_tvar, bootstrap_var  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(314)
    losses = rng.lognormal(mean=2.0, sigma=0.9, size=8_000)

    tail_boot = bootstrap_tail_probability(losses, threshold=80.0, n_resamples=250, random_state=1)
    var_boot = bootstrap_var(losses, q=0.99, n_resamples=250, random_state=2)
    tvar_boot = bootstrap_tvar(losses, q=0.99, n_resamples=250, random_state=3)

    print("Bootstrap tail probability")
    print(tail_boot.summary())
    print()
    print("Bootstrap VaR")
    print(var_boot.summary())
    print()
    print("Bootstrap TVaR")
    print(tvar_boot.summary())


if __name__ == "__main__":
    main()
