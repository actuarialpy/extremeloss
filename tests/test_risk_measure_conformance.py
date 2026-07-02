"""Ecosystem conformance test for empirical VaR / TVaR.

This file is intentionally IDENTICAL (except the import block) across
risksim, lossmodels, and extremeloss. The three packages deliberately keep
independent implementations to preserve their zero-dependency design; this
test pins them to one shared convention:

    VaR_q  = inf{x : F(x) >= q}                      (lower empirical quantile)
    TVaR_q = (1/(1-q)) * integral_q^1 VaR_u du       (average-quantile / ES)

If this test fails in one repo, the ecosystem has diverged: fix the
implementation, never the expected values below.
"""
import numpy as np
import pytest

from extremeloss.estimation.metrics import empirical_tvar as tvar
from extremeloss.estimation.metrics import empirical_var as var


def test_var_is_inverted_cdf_order_statistic():
    x = np.arange(1.0, 101.0)
    grid = np.array([0.5, 0.9, 0.95, 0.977, 0.99, 0.995])
    expected = np.quantile(x, grid, method="inverted_cdf")
    got = np.array([var(x, float(q)) for q in grid])
    assert np.allclose(got, expected)


def test_pinned_values_uniform_1_to_100():
    x = np.arange(1.0, 101.0)
    assert var(x, 0.95) == pytest.approx(95.0)
    assert tvar(x, 0.95) == pytest.approx(98.0)
    assert var(x, 0.99) == pytest.approx(99.0)
    assert tvar(x, 0.99) == pytest.approx(100.0)


def test_pinned_values_with_atoms():
    # Empirical dist has a large atom at 1: VaR_.5 = 1 and
    # TVaR_.5 = (1/(0.5)) * [1*0.3 + 10*0.2] = 4.6 exactly.
    x = [1.0, 1.0, 1.0, 1.0, 10.0]
    assert var(x, 0.5) == pytest.approx(1.0)
    assert tvar(x, 0.5) == pytest.approx(4.6)


def test_tvar_dominates_var_everywhere():
    rng = np.random.default_rng(42)
    x = rng.lognormal(10.0, 1.4, size=5001)
    for q in np.linspace(0.01, 0.995, 199):
        assert tvar(x, float(q)) >= var(x, float(q)) - 1e-9


def test_tvar_equals_mean_of_top_block_at_integer_rank():
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 5000.0, size=1000)
    top50 = np.sort(x)[-50:]
    assert tvar(x, 0.95) == pytest.approx(float(np.mean(top50)))


def test_flexible_inputs():
    pd = pytest.importorskip("pandas")
    x = list(range(1, 101))
    s = pd.Series(x, dtype=float)
    assert var(x, 0.95) == var(np.array(x, dtype=float), 0.95) == var(s, 0.95)
    assert isinstance(var(x, 0.95), float)
