"""Demo of duck-typed integration helpers for lossmodels- and risksim-style objects."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extremeloss import (  # noqa: E402
    component_tail_metrics,
    fit_pot_from_lossmodel,
    layer_tail_metrics,
    losses_from_risksim,
    sample_lossmodel,
    tail_summary_from_risksim,
)


class ToySeverity:
    def __init__(self, seed: int = 123):
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int) -> np.ndarray:
        return self.rng.lognormal(mean=2.0, sigma=0.8, size=size)


@dataclass
class ToySimulationResult:
    losses: np.ndarray
    gross_losses: np.ndarray
    retained_losses: np.ndarray
    ceded_losses: np.ndarray
    component_losses: np.ndarray
    component_names: list[str]
    layer_losses: np.ndarray
    layer_names: list[str]


def main() -> None:
    model = ToySeverity(seed=77)
    sampled = sample_lossmodel(model, size=20_000)
    fit = fit_pot_from_lossmodel(model, size=20_000, threshold=50.0)

    print("Lossmodel-style sampling")
    print(f"sample mean: {sampled.mean():.4f}")
    print("POT fit from model")
    print(fit.summary())
    print()

    rng = np.random.default_rng(2026)
    component_losses = rng.gamma(shape=2.0, scale=15.0, size=(10_000, 3))
    layer_losses = rng.gamma(shape=1.5, scale=18.0, size=(10_000, 2))
    gross_losses = component_losses.sum(axis=1)
    retained_losses = 0.85 * gross_losses
    ceded_losses = 0.15 * gross_losses

    result = ToySimulationResult(
        losses=gross_losses,
        gross_losses=gross_losses,
        retained_losses=retained_losses,
        ceded_losses=ceded_losses,
        component_losses=component_losses,
        component_names=["attritional", "large", "cat"],
        layer_losses=layer_losses,
        layer_names=["working", "excess"],
    )

    retained = losses_from_risksim(result, view="retained")
    print("Risksim-style retained view summary")
    print(f"retained mean: {retained.mean():.4f}")
    print(tail_summary_from_risksim(result, view="ceded", thresholds=[5.0, 10.0, 15.0]))
    print()
    print("Component tail metrics")
    print(component_tail_metrics(result, q=0.99, threshold=60.0))
    print()
    print("Layer tail metrics")
    print(layer_tail_metrics(result, q=0.99, threshold=50.0))


if __name__ == "__main__":
    main()
