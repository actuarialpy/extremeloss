"""Block-maxima / GEV workflow for extreme losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import block_return_level, fit_block_maxima, make_blocks  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(99)
    losses = rng.pareto(a=3.2, size=36_500) * 20.0

    block_size = 365
    maxima = make_blocks(losses, block_size=block_size)
    fit = fit_block_maxima(losses, block_size=block_size)

    print(f"Number of block maxima: {maxima.size}")
    print("GEV fit")
    print(fit.summary())
    for period in [10.0, 20.0, 50.0]:
        print(f"Return level for period {period:5.1f}: {block_return_level(period, fit):.4f}")


if __name__ == "__main__":
    main()
