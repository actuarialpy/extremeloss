"""Integration test: extremeloss tail helpers against a real risksim SimulationResult.

Mirrors test_spliced_integration.py (the lossmodels bridge): skips cleanly if the
optional sibling is not installed. This pins the duck-typed risksim contract against an
actual simulation result, not just a hand-rolled stand-in, so a future rename of a
risksim loss view (e.g. ceded_losses) is caught here rather than silently passing.
"""
from __future__ import annotations

import numpy as np
import pytest

risksim = pytest.importorskip("risksim")
pytest.importorskip("lossmodels")

from lossmodels.aggregate import CollectiveRiskModel  # noqa: E402
from lossmodels.frequency import Poisson  # noqa: E402
from lossmodels.severity import Lognormal  # noqa: E402
from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem  # noqa: E402

from extremeloss import (  # noqa: E402
    component_tail_metrics,
    layer_tail_metrics,
    losses_from_risksim,
    tail_summary_from_risksim,
)

SIZE = 10_000


@pytest.fixture
def result():
    np.random.seed(2026)
    portfolio = Portfolio(
        [
            PortfolioItem("medical", CollectiveRiskModel(Poisson(lam=3.0), Lognormal(mu=3.0, sigma=0.6))),
            PortfolioItem("rx", CollectiveRiskModel(Poisson(lam=2.0), Lognormal(mu=2.3, sigma=0.5))),
        ],
        name="tower_demo",
    )
    program = ContractProgram(
        [
            AggregateLayer(attachment=50.0, limit=50.0, name="layer_1"),
            AggregateLayer(attachment=100.0, limit=100.0, name="layer_2"),
        ],
        name="two_layer_tower",
    )
    return portfolio.simulate(size=SIZE, contract=program)


@pytest.mark.parametrize("view", ["losses", "gross", "gross_losses", "retained", "retained_losses", "ceded", "ceded_losses"])
def test_losses_from_risksim_every_view(result, view):
    losses = losses_from_risksim(result, view=view)
    assert losses.ndim == 1 and losses.shape[0] == SIZE
    assert np.isfinite(losses).all()


def test_unknown_view_raises(result):
    with pytest.raises(ValueError):
        losses_from_risksim(result, view="not_a_view")


def test_tail_summary_from_risksim(result):
    summary = tail_summary_from_risksim(result, view="ceded", quantiles=(0.95, 0.99))
    assert summary["n"] == SIZE
    assert "var_tvar" in summary


def test_component_tail_metrics_track_component_names(result):
    metrics = component_tail_metrics(result, q=0.99, threshold=60.0)
    assert set(metrics) == set(result.component_names)
    for stats in metrics.values():
        assert stats["tvar"] >= stats["var"]          # TVaR is the mean beyond VaR
        assert "exceedance_probability" in stats       # threshold was supplied


def test_layer_tail_metrics_track_layer_names(result):
    metrics = layer_tail_metrics(result, q=0.99)
    assert set(metrics) == set(result.layer_names)
    for stats in metrics.values():
        assert stats["tvar"] >= stats["var"]
